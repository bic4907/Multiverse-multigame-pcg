import os

import hydra
import torch
import wandb
import pandas as pd
from os.path import join, abspath

from conf.config import CVAETrainConfig
from conf.initializer.vae import init_config
from data_loader import make_dataloaders
from evaluator.vit_aggregate import process_single_instruction, process_blended_instruction
from models.clip.clip_model import CLIPFrozenModel
from models.vae.ema_vqvae import EMA_VQVAE
from trainer.vae_trainer import VAETrainer
from utils.logger import get_logger, select_render_samples, visualize_and_log
from utils.path import get_log_dir
from utils.schedular.late_linear import LateLinearDecaySchedular
from utils.wandb import upload_to_wandb
from instruct.utils import get_csv_path

logger = get_logger(__file__)

def train(config: CVAETrainConfig):
    # ---- dataloaders ----
    train_loader, val_loader = make_dataloaders(
        get_csv_path(config.instruction_csv),
        batch_size=config.batch_size,
        game_list=config.game_list,
        level_size=config.level_size,
        active_losses=config.active_losses,
        sample_ratio=config.sample_ratio,
        val_ratio=config.val_ratio,
        clip_model=config.clip_model,
        seed=config.dataset_split_seed,
        device=config.device,
    )

    clip_model = CLIPFrozenModel(
        embedding_dim=config.embedding_dim,
        text_encoder_model=config.clip_model
    ).to(config.device)

    vae_model = EMA_VQVAE(
        n_channel=train_loader.dataset.num_classes,
        num_codes=config.num_codes,
        code_dim=config.code_dim,
        beta_vq=config.vq_beta,
        cond_dim=config.embedding_dim
    ).to(config.device)

    logger.info("Number of classes: {}".format(train_loader.dataset.num_classes))

    vae_optimizer = torch.optim.AdamW(vae_model.parameters(), lr=config.lr)

    vq_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer,
        T_max=config.n_epochs,
        eta_min=config.lr * 0.05,
    )
    vq_beta_scheduler = LateLinearDecaySchedular(
        coef_init=config.vq_beta_coef,
        coef_min=config.vq_beta_coef_min,
        start_epoch=config.vq_beta_start_decay,
        total_epochs=config.n_epochs,
    )

    vae_trainer = VAETrainer(model=vae_model,
                             clip_model=clip_model,
                             lr_scheduler=vq_lr_scheduler,
                             beta_scheduler=vq_beta_scheduler,
                             optimizer=vae_optimizer,
                             device=config.device,
                             config=config,
                             num_codes=config.num_codes,
                             vit_score_single=config.vit_score_single,
                             vit_score_blend=config.vit_score_blend,
                             vit_batch_size=config.vit_batch_size,
                             n_vit_blend_samples=config.n_vit_blend_samples,
                             vit_eval_freq=config.vit_eval_freq,
                             prefix="vae")

    # DataFrame 수집을 위한 리스트
    all_single_dfs = []
    all_blended_dfs = []

    # =========================
    # Epoch loop
    # =========================
    for epoch in range(config.n_epochs):

        vae_train_metrics = vae_trainer.train(
            data_loader=train_loader,
            epoch=epoch,
        )

        vae_eval_metrics = vae_trainer.eval(
            data_loader=val_loader,
            epoch=epoch,
        )

        vae_trainer.on_epoch_end(epoch)

        vae_log_dir = get_log_dir(root_dir=config.exp_path, epoch=1, sub_dir="vae")

        vit_scores = dict()

        if 'single_instruction_dataframe' in vae_eval_metrics.extra:
            df = vae_eval_metrics.extra['single_instruction_dataframe']
            df.to_csv(join(vae_log_dir, f"single_instruction.csv"), index=False)

            # epoch 정보를 첫 번째 컬럼에 추가
            df_with_epoch = df.copy()
            df_with_epoch.insert(0, 'epoch', epoch + 1)
            all_single_dfs.append(df_with_epoch)

            single_vit = process_single_instruction(df)
            vit_scores.update(single_vit)

        if 'blended_instruction_dataframe' in vae_eval_metrics.extra:
            df = vae_eval_metrics.extra['blended_instruction_dataframe']
            df.to_csv(join(vae_log_dir, f"blended_instruction.csv"), index=False)

            # epoch 정보를 첫 번째 컬럼에 추가
            df_with_epoch = df.copy()
            df_with_epoch.insert(0, 'epoch', epoch + 1)
            all_blended_dfs.append(df_with_epoch)

            blend_vit = process_blended_instruction(df)
            vit_scores.update(blend_vit)

        logger.info(f"[Epoch {epoch + 1}] "
                    f"VAE Train Loss: {vae_train_metrics.metrics['total_loss']:.4f} | "
                    f"Eval Recon Loss: {vae_eval_metrics.metrics['recon_loss']:.4f}")

        if (epoch + 1) % config.render_interval == 0:
            if ('single_gt_levels' in vae_eval_metrics.extra and
                    'single_pred_levels' in vae_eval_metrics.extra):

                render_gt, render_pred, render_texts = select_render_samples(
                    vae_eval_metrics.extra['single_pred_levels'],
                    vae_eval_metrics.extra['single_texts'],
                    config.n_render_samples,
                    vae_eval_metrics.extra['single_gt_levels'],
                )

                visualize_and_log(
                    render_pred,
                    render_texts,
                    epoch,
                    config,
                    "result",
                    render_gt,
                    mode="single",
                )

            if ('blend_gt_levels_a' in vae_eval_metrics.extra and
                'blend_gt_levels_b' in vae_eval_metrics.extra and
                    'blend_pred_levels' in vae_eval_metrics.extra):

                render_gt_levels_a, render_gt_levels_b, render_pred_levels, render_texts, metadatas = (
                    select_render_samples(
                    vae_eval_metrics.extra['blend_pred_levels'],
                    vae_eval_metrics.extra['blend_texts'],
                    config.n_render_samples,
                    vae_eval_metrics.extra['blend_gt_levels_a'],
                    vae_eval_metrics.extra['blend_gt_levels_b'],
                    metadatas=vae_eval_metrics.extra['blend_metadatas'],
                ))

                visualize_and_log(
                    render_pred_levels,
                    render_texts,
                    epoch,
                    config,
                    "result",
                    render_gt_levels_a,
                    render_gt_levels_b,
                    metadatas=metadatas,
                    mode="blend",
                )

        if wandb.run:
            wandb.log({
                **{f"vae/train/{k}": v for k, v in vae_train_metrics.metrics.items()},
                **{f"vae/eval/{k}": v for k, v in vae_eval_metrics.metrics.items()},
                **{f"{k}": v for k, v in vit_scores.items()},
                "epoch": epoch + 1,
            })

    # =========================
    # 최종 DataFrame 저장 및 업로드
    # =========================

    # Single instruction DataFrame 병합
    if all_single_dfs:
        combined_single_df = pd.concat(all_single_dfs, ignore_index=True)
        single_output_path = join(config.exp_path, "single_instruction.csv")
        combined_single_df.to_csv(single_output_path, index=False)
        logger.info(f"Saved combined single instruction dataframe to {single_output_path}")

        upload_to_wandb(
            artifact_name=f"single_instruction",
            save_path=single_output_path,
        )

    # Blended instruction DataFrame 병합
    if all_blended_dfs:
        combined_blended_df = pd.concat(all_blended_dfs, ignore_index=True)
        blended_output_path = join(config.exp_path, "blended_instruction.csv")
        combined_blended_df.to_csv(blended_output_path, index=False)
        logger.info(f"Saved combined blended instruction dataframe to {blended_output_path}")

        upload_to_wandb(
            artifact_name=f"blended_instruction",
            save_path=blended_output_path,
        )


# ============================================================
# Entry
# ============================================================

@hydra.main(version_base=None, config_path="conf", config_name="train_cvae")
def main(config: CVAETrainConfig):
    init_config(config)

    logger.info(f"Experiment Directory: {config.exp_path}")

    if config.use_wandb:
        wandb.init(
            id=config.wandb_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.exp_name,
            config=dict(config),
        )
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")

    train(config)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
