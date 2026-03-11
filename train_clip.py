import hydra
import torch
import wandb

from conf.config import CLIPTrainConfig
from conf.initializer.clip import init_config
from data_loader import make_dataloaders
from models.clip.clip_model import CLIPModel
from trainer.clip_trainer import CLIPTrainer
from multiprocessing import Process, Queue
from analysis.tsne_visualize import log_tsne_async, tsne_worker
from utils.logger import get_logger, tsne_logger
from instruct.utils import get_csv_path

logger = get_logger(__file__)

def train(config: CLIPTrainConfig, tsne_queue_in=None, tsne_queue_out=None):
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

    clip_model = CLIPModel(
        num_classes=train_loader.dataset.num_classes,
        embedding_dim=config.embedding_dim,
        drop_rate=config.drop_rate,
        init_temperature=config.init_temperature,
        text_encoder_model=config.clip_model
    ).to(config.device)
    
    clip_optimizer = torch.optim.AdamW(clip_model.parameters(), lr=config.lr)
    clip_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        clip_optimizer,
        T_max=config.n_epochs,
        eta_min=config.lr * 0.05,
    )

    clip_trainer = CLIPTrainer(model=clip_model,
                            optimizer=clip_optimizer,
                            lr_scheduler=clip_lr_scheduler,
                            device=config.device,
                            active_losses=config.active_losses,
                            loss_weights=config.loss_weights,
                            specific_text_embedding_path=config.specific_text_embedding_path,
                            general_text_embedding_path=config.general_text_embedding_path,
                            spec_threshold=config.spec_threshold,
                            gen_threshold=config.gen_threshold,
                            prefix="clip")

    # =========================
    # Epoch loop
    # =========================
    for epoch in range(config.n_epochs):

        clip_train_metrics = clip_trainer.train(
            data_loader=train_loader,
            epoch=epoch,
        )

        clip_eval_metrics = clip_trainer.eval(
            data_loader=val_loader,
            epoch=epoch,
        )

        if config.draw_tsne:
            if (epoch + 1) % config.tsne_interval == 0:
                log_tsne_async(
                    model=clip_model,
                    dataloader=val_loader,
                    device=config.device,
                    step=epoch + 1,
                    tsne_queue=tsne_queue_in,
                    prefix="result"
                )
            if not tsne_queue_out.empty():
                tsne_logger(tsne_queue_out, wait=False)

        clip_trainer.on_epoch_end(epoch)

        logger.info(f"[Epoch {epoch + 1}] "
                    f"CLIP Train Loss: {clip_train_metrics.metrics['loss_i2t'] + clip_train_metrics.metrics['loss_t2i']:.4f} | "
                    f"Eval Loss: {clip_eval_metrics.metrics['loss_i2t'] + clip_eval_metrics.metrics['loss_t2i']:.4f}")

        if wandb.run:
            wandb.log({
                **{f"clip/train/{k}": v for k, v in clip_train_metrics.metrics.items()},
                **{f"clip/eval/{k}": v for k, v in clip_eval_metrics.metrics.items()},
                "epoch": epoch + 1,
            })

    if config.draw_tsne:
        tsne_queue_in.put(None)
        tsne_logger(queue_out=tsne_queue_out, wait=True)


# ============================================================
# Entry
# ============================================================

@hydra.main(version_base=None, config_path="conf", config_name="train_clip")
def main(config: CLIPTrainConfig):
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
        wandb.define_metric("epoch")

    if config.draw_tsne:
        tsne_queue_in  = Queue(maxsize=8)
        tsne_queue_out = Queue(maxsize=8)
        tsne_process = Process(
        target=tsne_worker,
            args=(tsne_queue_in, tsne_queue_out),
            daemon=True
        )
        tsne_process.start()
        
        train(config, tsne_queue_in, tsne_queue_out)

    else:
        train(config)
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()