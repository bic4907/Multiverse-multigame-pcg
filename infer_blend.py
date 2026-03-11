import hydra
import torch
from itertools import combinations
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
def train(config: RolloutConfig):
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

    # make combination with collections module
    blend_ratios = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]

    comb = combinations(config.game_list, 2)
    for c in comb:
        # make directory for each combination
        comb_dir = join(config.exp_path, config.output_dir, f"{'_'.join(c)}")
        os.makedirs(comb_dir, exist_ok=True)
        game_a, game_b = c[0], c[1]

        logger.info(f"Blending {game_a} to {game_b}, saving to {comb_dir}")

        level_a = dataset.get_by_game(game_a, 0)
        level_b = dataset.get_by_game(game_b, 0)

        text_a, text_b = level_a['text'], level_b['text']

        # get embeddings
        logger.info(f"Text A: {text_a}")
        logger.info(f"Text B: {text_b}")

        # Tokenize text_a
        text_inputs_a = tokenizer(
            [text_a],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_a = text_inputs_a["input_ids"].to(config.device)
        attention_mask_a = text_inputs_a["attention_mask"].to(config.device)

        # Tokenize text_b
        text_inputs_b = tokenizer(
            [text_b],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_b = text_inputs_b["input_ids"].to(config.device)
        attention_mask_b = text_inputs_b["attention_mask"].to(config.device)

        # Get embeddings from CLIP
        c_emb_a = clip_model.text_encoder(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a
        )
        c_emb_b = clip_model.text_encoder(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b
        )

        # Render original levels for visualization
        level_a_array = level_a['level'].argmax(dim=0).cpu().numpy()  # (H, W)
        level_b_array = level_b['level'].argmax(dim=0).cpu().numpy()  # (H, W)

        level_a_discrete = onehot_to_levels(torch.from_numpy(level_a_array).unsqueeze(0))[0].cpu().numpy()
        level_b_discrete = onehot_to_levels(torch.from_numpy(level_b_array).unsqueeze(0))[0].cpu().numpy()

        rendered_level_a = render_level(level_a_discrete, tile_size=16, return_numpy=True)
        rendered_level_b = render_level(level_b_discrete, tile_size=16, return_numpy=True)

        # Save original levels as individual images
        Image.fromarray(rendered_level_a).save(join(comb_dir, f"original_{game_a}.png"))
        Image.fromarray(rendered_level_b).save(join(comb_dir, f"original_{game_b}.png"))

        # Store generated levels and their rendered versions
        generated_levels = []
        rendered_images = []

        # interpolate text_a and b with different ratios
        for ratio_a, ratio_b in blend_ratios:
            logger.info(f"Generating blend with ratio {ratio_a:.2f}:{ratio_b:.2f}")

            # Interpolate embeddings
            c_emb_blend = ratio_a * c_emb_a + ratio_b * c_emb_b

            # inference with vae
            with torch.no_grad():
                logits = vae_model.sample(c_emb_blend)
                pred_level = torch.argmax(logits, dim=1)  # (1, H, W)

            # Convert to discrete level
            pred_level_discrete = onehot_to_levels(pred_level)[0].cpu().numpy()  # (H, W)
            generated_levels.append(pred_level_discrete)

            # Render the level
            rendered_img = render_level(pred_level_discrete, tile_size=16, return_numpy=True)
            rendered_images.append(rendered_img)

            # save levels with the ratio postfix
            ratio_str = f"{ratio_a:.2f}_{ratio_b:.2f}".replace('.', 'p')
            level_filename = join(comb_dir, f"level_{ratio_str}.npy")
            img_filename = join(comb_dir, f"level_{ratio_str}.png")

            np.save(level_filename, pred_level_discrete)

            # Save rendered image
            Image.fromarray(rendered_img).save(img_filename)
            logger.info(f"Saved level to {level_filename} and {img_filename}")

        # Save instruction metadata as JSON
        metadata = {
            "game_a": game_a,
            "game_b": game_b,
            "instruction_a": text_a,
            "instruction_b": text_b,
            "blend_ratios": [{"ratio_a": ra, "ratio_b": rb} for ra, rb in blend_ratios],
        }
        metadata_filename = join(comb_dir, "metadata.json")
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_filename}")

        # and make a matplot figure to visualize the blending results
        # Include original levels A and B on the left
        total_plots = 2 + len(blend_ratios)  # 2 original + blended
        fig, axes = plt.subplots(1, total_plots, figsize=(3*total_plots, 4.5))

        # Plot original level A
        axes[0].imshow(rendered_level_a)
        axes[0].set_title(f"{game_a}\n(Original A)", fontsize=9, pad=3)
        axes[0].axis('off')
        # Show full text with wrapping
        axes[0].text(0.5, -0.02, text_a,
                     ha='center', va='top', transform=axes[0].transAxes,
                     fontsize=6, wrap=True, style='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3, edgecolor='none'))

        # Plot original level B
        axes[1].imshow(rendered_level_b)
        axes[1].set_title(f"{game_b}\n(Original B)", fontsize=9, pad=3)
        axes[1].axis('off')
        # Show full text with wrapping
        axes[1].text(0.5, -0.02, text_b,
                     ha='center', va='top', transform=axes[1].transAxes,
                     fontsize=6, wrap=True, style='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3, edgecolor='none'))

        # Plot blended levels
        for idx, ((ratio_a, ratio_b), img) in enumerate(zip(blend_ratios, rendered_images)):
            ax_idx = idx + 2  # Offset by 2 for original levels
            axes[ax_idx].imshow(img)
            axes[ax_idx].set_title(f"{game_a} {ratio_a:.0%}\n{game_b} {ratio_b:.0%}",
                                   fontsize=9, pad=3)
            axes[ax_idx].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        fig_filename = join(comb_dir, "blending_overview.png")
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        logger.info(f"Saved visualization to {fig_filename}")

    logger.info("Blending complete!")


@hydra.main(version_base="1.3", config_path="conf", config_name="infer_blend")
def main(config: RolloutConfig):
    init_config(config)

    logger.info(f"Experiment Directory: {config.exp_path}")

    train(config)


if __name__ == "__main__":
    main()