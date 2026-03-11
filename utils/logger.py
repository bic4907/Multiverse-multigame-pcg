import logging
from os.path import basename

import numpy as np
import torch
import os
import wandb
import matplotlib.pyplot as plt
from queue import Empty
from PIL import Image, ImageDraw
from utils.renderer import render_level


def get_logger(file_name: str):
    name = basename(file_name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger



def onehot_to_levels(level: torch.Tensor) -> torch.Tensor:
    """0/1 or one-hot → tile index (start from 1)"""
    return level + 1


def select_render_samples(
    pred_levels: torch.Tensor,
    texts: list[str],
    n_render_samples: int,
    *gt_levels: torch.Tensor,
    metadatas: list = None,
):
    """
    Unified function to select render samples for both single and blended modes.

    Args:
        pred_levels: Predicted levels tensor
        texts: List of text descriptions
        n_render_samples: Number of samples to select
        *gt_levels: Variable number of ground truth level tensors (1 for single, 2+ for blended)
        metadatas: Optional metadata list for blended mode

    Returns:
        Tuple of selected samples (gt_levels..., pred_levels, texts, metadatas?)
    """
    k = min(pred_levels.shape[0], n_render_samples)

    result = []
    for gt in gt_levels:
        result.append(gt[:k].detach().cpu())

    result.append(pred_levels[:k].detach().cpu())
    result.append(texts[:k])

    if metadatas is not None:
        result.append(metadatas[:k])

    return tuple(result)


def concat_images_with_labels(
    images: list[Image.Image],
    labels: list[str],
):
    """
    Concatenate multiple images horizontally with text labels.

    Args:
        images: List of PIL Images to concatenate
        labels: List of labels for each image

    Returns:
        Concatenated PIL Image
    """
    if not images or len(images) != len(labels):
        raise ValueError("images and labels must have the same non-zero length")

    w, h = images[0].size
    n_images = len(images)

    canvas = Image.new("RGB", (w * n_images, h + 30), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Paste images and add labels
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (w * i, 30))
        # Center text in each column
        text_x = w * i + w // 2 - len(label) * 3
        draw.text((text_x, 5), label, fill=(0, 0, 0))

    return canvas


def visualize_and_log(
    pred_levels: torch.Tensor,
    texts: list[str],
    step: int,
    config,
    log_prefix: str,
    *gt_levels: torch.Tensor,
    metadatas: list = None,
    mode: str = "single",
    tile_size: int = 32,
):
    """
    Unified function to visualize and log levels to wandb.

    Args:
        pred_levels: Predicted levels tensor (K, H, W)
        texts: List of text descriptions
        step: Current training step
        config: Config object with exp_path
        log_prefix: Prefix for wandb logging
        *gt_levels: Variable number of ground truth tensors
        metadatas: Optional metadata for captions
        mode: "single" or "blend" for logging path
        tile_size: Size of tiles for rendering
    """
    step_dir = os.path.join(config.exp_path, "train", f"{step}")
    os.makedirs(step_dir, exist_ok=True)

    wandb_images = []
    n_samples = pred_levels.shape[0]

    # Determine labels based on number of ground truths
    if len(gt_levels) == 1:
        labels = ["Ground Truth", "Predicted"]
    elif len(gt_levels) == 2:
        labels = ["Ground Truth A", "Ground Truth B", "Predicted"]
    else:
        labels = [f"Ground Truth {i+1}" for i in range(len(gt_levels))] + ["Predicted"]

    for i in range(n_samples):
        # Render all ground truth levels
        rendered_images = []
        for gt in gt_levels:
            gt_np = gt[i].numpy().astype(np.int32)
            gt_img = render_level(gt_np, tile_size=tile_size, return_numpy=False)
            rendered_images.append(gt_img)

        # Render predicted level
        pred_np = pred_levels[i].numpy().astype(np.int32)
        pred_img = render_level(pred_np, tile_size=tile_size, return_numpy=False)
        rendered_images.append(pred_img)

        # Concatenate all images
        merged = concat_images_with_labels(rendered_images, labels)

        # Save image
        img_filename = f"{mode}_image_{i}.png"
        img_path = os.path.join(step_dir, img_filename)
        merged.save(img_path)

        # Create caption
        if metadatas is not None and i < len(metadatas):
            caption = f"[step {step}] {str(metadatas[i])}"
        else:
            caption = f"[step {step}] {texts[i]}"

        wandb_images.append(
            wandb.Image(merged, caption=caption)
        )

    if wandb.run:
        wandb.log(
            {f"{log_prefix}/{mode}": wandb_images, "step": step}
        )

def tsne_logger(queue_out, wait=False):
    if wait:
        while True:
            item = queue_out.get()
            if item is None:
                break
            fig, step, prefix = item

            name = "tsne" if prefix is None else f"{prefix}/tsne"
            wandb.log({name: wandb.Image(fig), "epoch": step})
            plt.close(fig)
    else:
        while True:
            try:
                item = queue_out.get_nowait()
                if item is None:
                    break
                fig, step, prefix = item
            except Empty:
                break

            name = "tsne" if prefix is None else f"{prefix}/tsne"
            wandb.log({name: wandb.Image(fig), "epoch": step})
            plt.close(fig)
