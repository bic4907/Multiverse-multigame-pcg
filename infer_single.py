import hydra
import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPTokenizer
import json

from conf.config import RolloutConfig
from conf.initializer.rollout import init_config
from data_loader import make_indexed_level_dataset
from models.clip.clip_model import CLIPModel
from models.vae.ema_vqvae import EMA_VQVAE
from utils.logger import get_logger, onehot_to_levels
from utils.path import get_log_dir
from instruct.utils import get_csv_path
from utils.checkpoint import CheckpointManager, load_checkpoint_with_fallback
from utils.renderer import render_level

logger = get_logger(__file__)


@torch.no_grad()
def inference(config: RolloutConfig):
    # ---- dataloaders ----
    dataset = make_indexed_level_dataset(
        get_csv_path(config.instruction_csv),
        game_list=config.game_list,
        level_size=config.level_size,
        device=config.device,
    )

    clip_model = CLIPModel(
        num_classes=dataset.num_classes,
        embedding_dim=config.embedding_dim,
        drop_rate=config.drop_rate,
        init_temperature=config.init_temperature,
        text_encoder_model=config.clip_model
    ).to(config.device)
    clip_model.eval()

    vae_model = EMA_VQVAE(
        n_channel=dataset.num_classes,
        num_codes=config.num_codes,
        code_dim=config.code_dim,
        beta_vq=config.vq_beta,
        cond_dim=config.embedding_dim,
    ).to(config.device)
    vae_model.eval()

    # Initialize tokenizer for text processing
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model)

    logger.info("Number of classes: {}".format(dataset.num_classes))

    # instantiate checkpoint manager (always create for loading, even if not saving)
    checkpoint_manager = CheckpointManager(
        log_dir_root=config.exp_path,
        config=config,
        save_interval=config.save_interval if config.save_ckpt else 0,
        save_keep=config.save_keep if config.save_ckpt else 0,
        logger=logger,
        get_log_dir_fn=get_log_dir,
    )

    # register models to the checkpoint manager
    checkpoint_manager.register(clip=clip_model, vae=vae_model)

    # Load checkpoint with fallback logic (checkpoint_path > checkpoint_epoch > latest)
    load_checkpoint_with_fallback(
        checkpoint_manager=checkpoint_manager,
        config=config,
        device=config.device,
        model_names=['clip', 'vae'],
        logger=logger
    )

    # Number of samples to generate per game
    num_samples = getattr(config, 'num_samples', 10)

    # Iterate over each game
    for game in config.game_list:
        logger.info(f"Generating {num_samples} levels for game: {game}")

        # Create game-specific directory
        game_dir = join(config.exp_path, config.output_dir, game)
        os.makedirs(game_dir, exist_ok=True)

        # Store all generated levels and metadata for this game
        all_metadata = []
        all_rendered_images = []

        # Generate multiple samples
        for sample_idx in range(num_samples):
            logger.info(f"  Generating sample {sample_idx + 1}/{num_samples}")

            # Get level data from dataset
            level_data = dataset.get_by_game(game, sample_idx)
            text = level_data['text']

            logger.info(f"  Instruction: {text}")

            # Tokenize text
            text_inputs = tokenizer(
                [text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs["input_ids"].to(config.device)
            attention_mask = text_inputs["attention_mask"].to(config.device)

            # Get embedding from CLIP
            c_emb = clip_model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Generate level with VAE
            with torch.no_grad():
                logits = vae_model.sample(c_emb)
                pred_level = torch.argmax(logits, dim=1)  # (1, H, W)

            # Convert to discrete level
            pred_level_discrete = onehot_to_levels(pred_level)[0].cpu().numpy()  # (H, W)

            # Render the level
            rendered_img = render_level(pred_level_discrete, tile_size=16, return_numpy=True)
            all_rendered_images.append(rendered_img)

            # Save individual level
            level_filename = join(game_dir, f"level_{sample_idx:03d}.npy")
            img_filename = join(game_dir, f"level_{sample_idx:03d}.png")

            np.save(level_filename, pred_level_discrete)
            Image.fromarray(rendered_img).save(img_filename)

            logger.info(f"  Saved to {level_filename} and {img_filename}")

            # Store metadata for this sample
            all_metadata.append({
                "sample_id": sample_idx,
                "instruction": text,
                "level_file": f"level_{sample_idx:03d}.npy",
                "image_file": f"level_{sample_idx:03d}.png"
            })

        # Save all metadata as JSON
        metadata_filename = join(game_dir, "metadata.json")
        metadata = {
            "game": game,
            "num_samples": num_samples,
            "samples": all_metadata
        }
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_filename}")

        # Create visualization grid
        logger.info(f"Creating visualization grid for {game}")

        # Calculate grid dimensions (prefer wider grid)
        n_cols = min(5, num_samples)
        n_rows = (num_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3.5*n_rows))

        # Handle single row/column case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot all samples
        for idx in range(num_samples):
            row = idx // n_cols
            col = idx % n_cols

            axes[row, col].imshow(all_rendered_images[idx])
            axes[row, col].set_title(f"Sample {idx}", fontsize=9, pad=3)
            axes[row, col].axis('off')

            # Add instruction text below
            instruction_text = all_metadata[idx]['instruction']
            if len(instruction_text) > 60:
                instruction_text = instruction_text[:57] + '...'
            axes[row, col].text(0.5, -0.02, instruction_text,
                               ha='center', va='top', transform=axes[row, col].transAxes,
                               fontsize=6, wrap=True, style='italic',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                                       alpha=0.3, edgecolor='none'))

        # Hide unused subplots
        for idx in range(num_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.3)
        fig_filename = join(game_dir, "overview.png")
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        logger.info(f"Saved visualization to {fig_filename}")

    logger.info("Single game inference complete!")


@hydra.main(version_base="1.3", config_path="conf", config_name="infer_single")
def main(config: RolloutConfig):
    init_config(config)

    logger.info(f"Experiment Directory: {config.exp_path}")

    inference(config)


if __name__ == "__main__":
    main()

