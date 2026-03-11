from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from evaluator import TwoPairComparisonSet, ThreePairComparisonSet
from evaluator.blend_dataset import make_blender_dataloader
from evaluator.vitscore import ViTEvaluator
from trainer.base_trainer import BaseTrainer
from trainer.step import StepOutput
from utils.logger import onehot_to_levels, get_logger

logger = get_logger(__file__)

class VAETrainer(BaseTrainer):

    vit_single_instruction: bool
    vit_blended_instruction: bool

    trainable_clip: bool

    def __init__(
        self,
        model,
        clip_model,
        beta_scheduler,
        lr_scheduler,
        optimizer,
        device,
        config,
        num_codes: int,
        vit_score_single: bool = True,
        vit_score_blend: bool = True,
        vit_batch_size: int = 32,
        n_vit_blend_samples: int = 500,
        vit_eval_freq: int = 1,
        trainable_clip: bool = False,
        clip_optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.clip_model = clip_model

        self.vq_beta_scheduler = beta_scheduler
        self.vq_lr_scheduler = lr_scheduler

        self.trainable_clip = trainable_clip
        self.clip_optimizer = clip_optimizer

        assert (not self.trainable_clip) or (self.clip_optimizer is not None), "If trainable_clip is True, clip_optimizer must be provided."

        self.optimizer = optimizer
        self.device = device
        self.config = config

        # VQ-VAE
        self.num_codes = num_codes

        self.vit_score_single = vit_score_single
        self.vit_score_blend = vit_score_blend
        self.n_vit_blend_samples = n_vit_blend_samples
        self.vit_eval_freq = vit_eval_freq

        if self.use_vit_evaluator:
            self.vit_evaluator = ViTEvaluator(batch_size=vit_batch_size, device=device)
            self.vit_evaluator.preload()


    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        self.vq_beta_scheduler.step(epoch=epoch)
        self.vq_lr_scheduler.step()

    # =====================================================
    # Train (epoch-level)
    # =====================================================
    def train(self, data_loader, epoch):
        self.model.train()

        epoch_recon = 0.0
        epoch_vq = 0.0
        epoch_total = 0.0
        n_batches = 0.0

        beta = self.vq_beta_scheduler.get()
        last_indices = None

        for batch in tqdm(data_loader, desc=f"epoch {epoch}/{self.prefix}/train"):

            levels = batch['level'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            if self.trainable_clip is False:
                with torch.no_grad():
                    self.clip_model.eval()
                    c_emb = self.clip_model.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                self.clip_model.train()
                c_emb = self.clip_model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # ---- forward ----
            logits, indices, vq_loss = self.model(levels, c_emb)
            recon_loss = F.binary_cross_entropy_with_logits(logits, levels)
            loss = recon_loss + beta * vq_loss

            if self.trainable_clip:
                self.clip_optimizer.zero_grad(set_to_none=True)

            # ---- backward ----
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if self.trainable_clip:
                self.clip_optimizer.step()

            # ---- stats ----
            epoch_recon += recon_loss.item()
            epoch_vq += vq_loss.item()
            epoch_total += loss.item()
            n_batches += 1

            last_indices = indices

        return StepOutput(
            loss=epoch_total / n_batches,
            metrics={
                "recon_loss": epoch_recon / n_batches,
                "vq_loss": epoch_vq / n_batches,
                "total_loss": epoch_total / n_batches,
                "vq_beta": beta,
                "lr": self.vq_lr_scheduler.get_last_lr()[0],
                "beta": self.vq_beta_scheduler.coef,
            },
            extra={
                "indices": last_indices,

            },
        )

    # =====================================================
    # Eval (epoch-level)
    # =====================================================
    @torch.no_grad()
    def eval(self, data_loader, epoch):
        self.model.eval()

        epoch_recon = 0.0
        epoch_total = 0.0
        n_batches = 0

        all_levels, all_logits, all_texts, all_indices = list(), list(), list(), list()
        gt_levels, pred_levels = list(), list()

        all_embeds = list()
        all_games = list()

        vit_dataframes = dict()

        rendering_outputs = dict()

        """
        SingleInstructionDataset Evaluation Loop
        """
        for batch in tqdm(data_loader, desc=f"epoch {epoch}/{self.prefix}/eval-single"):
            games = batch['games']
            levels = batch['level'].to(self.device)
            texts = batch['raw_text']

            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)


                self.clip_model.eval()
                c_emb = self.clip_model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                all_embeds.append(c_emb)

            logits = self.model.sample(c_emb)
            recon_loss = F.binary_cross_entropy_with_logits(logits, levels)

            epoch_recon += recon_loss.item()
            n_batches += 1

            all_levels.extend(levels.detach().cpu())
            all_logits.extend(logits.detach().cpu())
            all_texts.extend(texts)
            all_games.extend(games)

            gt_levels.extend(onehot_to_levels(torch.argmax(levels, dim=1).cpu()))
            pred_levels.extend(onehot_to_levels(torch.argmax(logits, dim=1).cpu()))

        gt_levels = torch.from_numpy(np.stack(gt_levels))
        pred_levels = torch.from_numpy(np.stack(pred_levels))

        rendering_outputs.update({
            "single_levels": all_levels,
            "single_logits": all_logits,
            "single_texts": all_texts,
            "single_gt_levels": gt_levels,
            "single_pred_levels": pred_levels
        })

        if self.vit_score_single and (epoch % self.vit_eval_freq == 0):
            logger.info("Evaluating Single Instruction ViT Score...")
            evaluation_levels = [TwoPairComparisonSet(gt, pred, metadata=(game, text, 1.0))
                                 for gt, pred, game, text in zip(gt_levels, pred_levels, all_games, all_texts)]
            two_pair, _ = self.vit_evaluator.run(comparisons=evaluation_levels)

            vit_dataframes["single_instruction_dataframe"] = two_pair.to_dataframe()

        """
        BlendedInstructionDataset Evaluation Loop
        """
        if self.vit_score_blend:

            all_embeds = torch.cat(all_embeds)
            blended_levels = list()
            a_levels = list()
            b_levels = list()
            texts = list()
            metadatas = list()

            blend_data_loader = make_blender_dataloader(data_loader=data_loader,
                                                        text_embeddings=all_embeds.detach().cpu(),
                                                        max_data_length=self.config.n_vit_blend_samples)

            for batch in tqdm(blend_data_loader, desc=f"epoch {epoch}/{self.prefix}/eval-blend"):

                c_emb = batch['interpolated_embeddings'].to(self.device)
                a_level = batch['level_a']
                b_level = batch['level_b']
                blend_ratios = batch['blend_ratios']
                a_game, b_game = batch['game_a'], batch['game_b']
                text_a, text_b = batch['text_a'], batch['text_b']

                texts.append((text_a, text_b))

                logits = self.model.sample(c_emb)

                blended_levels.extend(onehot_to_levels(torch.argmax(logits, dim=1).cpu()))
                a_levels.extend(onehot_to_levels(torch.argmax(a_level, dim=1).cpu()))
                b_levels.extend(onehot_to_levels(torch.argmax(b_level, dim=1).cpu()))

                for (ra, rb), ga, ta, gb, tb in zip(blend_ratios, a_game, text_a, b_game, text_b):
                    metadatas.append(((ga, ta, ra), (gb, tb, rb)))

            blended_levels = torch.from_numpy(np.stack(blended_levels))
            a_levels = torch.from_numpy(np.stack(a_levels))
            b_levels = torch.from_numpy(np.stack(b_levels))

            rendering_outputs.update({
                "blend_gt_levels_a": a_levels,
                "blend_gt_levels_b": b_levels,
                "blend_pred_levels": blended_levels,
                "blend_texts": texts,
                "blend_metadatas": metadatas,
            })

            if (epoch % self.vit_eval_freq == 0):
                logger.info("Evaluating Blended Instruction ViT Score...")

                evaluation_levels = [ThreePairComparisonSet(a, b, blended, metadata=metadata) for a, b, blended, metadata
                                     in zip(a_levels, b_levels, blended_levels, metadatas)]
                _, three_pair = self.vit_evaluator.run(comparisons=evaluation_levels)
                vit_dataframes["blended_instruction_dataframe"] = three_pair.to_dataframe()


        """
        BlendedInstructionDataset Evaluation Loop
        """

        return StepOutput(
            loss=epoch_total / n_batches,
            metrics={
                "recon_loss": epoch_recon / n_batches,
            },
            extra={
                **rendering_outputs,
                **vit_dataframes,
            },
        )

    @property
    def use_vit_evaluator(self) -> bool:
        return self.vit_score_single or self.vit_score_blend
