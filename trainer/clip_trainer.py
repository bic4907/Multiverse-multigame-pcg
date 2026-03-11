from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from models.clip.loss import BaselineLoss, SpecificInstructLoss, GeneralInstructLoss, DifficultyLoss, topk_accuracy
from .base_trainer import BaseTrainer
import torch

from .step import StepOutput

LOSS_CLASS = {
    "base": BaselineLoss,
    "spec": SpecificInstructLoss,
    "gen": GeneralInstructLoss,
    "diff": DifficultyLoss,
}

class CLIPTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        device,
        active_losses,
        loss_weights,
        specific_text_embedding_path: str = None,
        general_text_embedding_path: str = None,
        spec_threshold: float = 0.7,
        gen_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        
        self.loss_modules = []
        self.loss_weights = loss_weights

        for name in active_losses:
            if name == "spec":
                self.loss_modules.append(
                    SpecificInstructLoss(
                        embedding_path=specific_text_embedding_path,
                        threshold=spec_threshold,
                        device=device,
                    )
                )
            elif name == "gen":
                self.loss_modules.append(
                    GeneralInstructLoss(
                        embedding_path=general_text_embedding_path,
                        threshold=gen_threshold,
                        device=device,
                    )
                )
            else:
                self.loss_modules.append(
                    LOSS_CLASS[name](device=device)
                )

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        self.lr_scheduler.step()

    # =====================================================
    # Train (epoch-level)
    # =====================================================
    def train(self, data_loader: DataLoader, epoch: int):
        self.model.train()

        for batch in tqdm(data_loader, desc=f"epoch {epoch}/{self.prefix}/train"):
            
            # ---- forward ----
            level_emb, text_emb = self.model(batch)

            logit_scale = self.model.temperature.clamp(-4.6, 2.3)
            logit_scale = logit_scale.exp()
            logits = logit_scale * (level_emb @ text_emb.T)
            
            # ==== loss ====
            loss_i2t_total = 0.0
            loss_t2i_total = 0.0

            for loss_module in self.loss_modules:
                # Generate mask
                if hasattr(loss_module, "get_mask"):
                    loss_module.get_mask(batch)

                # Compute loss
                logit_i2t, logit_t2i = loss_module.loss(logits)

                weight = self.loss_weights[loss_module.name]

                loss_i2t_total += weight * logit_i2t
                loss_t2i_total += weight * logit_t2i
            
            loss = (loss_i2t_total + loss_t2i_total) / 2

            # ---- backward ----
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return StepOutput(
            loss=loss,
            metrics={
                "loss_i2t": loss_i2t_total.item(),
                "loss_t2i": loss_t2i_total.item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
            },
            extra={

            }
        )

    # =====================================================
    # Eval (epoch-level)
    # =====================================================
    @torch.no_grad()
    def eval(self, data_loader: DataLoader, epoch: int):
        self.model.eval()

        for batch in tqdm(data_loader, desc=f"epoch {epoch}/{self.prefix}/eval"):
            level_emb, text_emb = self.model(batch)

            logit_scale = self.model.temperature.clamp(-4.6, 2.3)
            logit_scale = logit_scale.exp()
            logits = logit_scale * (level_emb @ text_emb.T)

           # ==== loss ====
            loss_i2t_total = 0.0
            loss_t2i_total = 0.0

            for loss_module in self.loss_modules:
                # Generate mask
                if hasattr(loss_module, "get_mask"):
                    loss_module.get_mask(batch)

                # Compute loss
                logit_i2t, logit_t2i = loss_module.loss(logits)

                weight = self.loss_weights[loss_module.name]

                loss_i2t_total += weight * logit_i2t
                loss_t2i_total += weight * logit_t2i
            
            loss = (loss_i2t_total + loss_t2i_total) / 2

            acc_i2t = topk_accuracy(logits)
            acc_t2i = topk_accuracy(logits.T)

        return StepOutput(
            loss=loss,
            metrics={
                "loss_i2t": loss_i2t_total.item(),
                "loss_t2i": loss_t2i_total.item(),
                "i2t_top1": acc_i2t["top1"],
                "t2i_top1": acc_t2i["top1"],
                "i2t_top5": acc_i2t["top5"],
                "t2i_top5": acc_t2i["top5"],
            }
        )