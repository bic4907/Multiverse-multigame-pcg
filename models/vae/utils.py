import torch
import wandb
from transformers import CLIPModel, CLIPProcessor


def load_frozen_clip(clip_model: str, device: str):
    model = CLIPModel.from_pretrained(clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False
    return model, processor


@torch.no_grad()
def clip_text_embed(texts, model, processor, device):
    inputs = processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    emb = model.get_text_features(**inputs)  # (B, 512)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def vq_stats_from_indices(indices: torch.Tensor, num_codes: int):
    # indices: (B, H, W) long
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=num_codes).float()

    probs = counts / (counts.sum() + 1e-9)
    entropy = -(probs * (probs + 1e-9).log()).sum()
    perplexity = torch.exp(entropy)

    dead_ratio = (counts == 0).float().mean()
    top1_ratio = (counts.max() / (counts.sum() + 1e-9))

    return {
        "vq/perplexity": perplexity.item(),
        "vq/entropy": entropy.item(),
        "vq/dead_code_ratio": dead_ratio.item(),
        "vq/top1_code_ratio": top1_ratio.item(),
        "vq/used_codes": (counts > 0).sum().item(),
        "vq/code_usage_hist": wandb.Histogram(counts.cpu().numpy())
    }, counts


import torch
from typing import Dict


def vq_stats_from_counts(
    counts: torch.Tensor,
    *,
    prefix: str = "vq",
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Args:
        counts: (num_codes,) long or float tensor
                누적된 code usage count (epoch 단위 권장)
        prefix: wandb/log key prefix
    Returns:
        dict of scalar statistics
    """
    counts = counts.float()
    total = counts.sum()

    if total < eps:
        # 안전장치: 아무 code도 안 쓰인 경우
        return {
            f"{prefix}/usage_entropy": 0.0,
            f"{prefix}/perplexity": 1.0,
            f"{prefix}/active_codes": 0.0,
            f"{prefix}/dead_code_ratio": 1.0,
            f"{prefix}/max_usage_ratio": 0.0,
            f"{prefix}/min_usage_ratio": 0.0,
        }

    probs = counts / total

    # Shannon entropy
    entropy = -(probs * (probs + eps).log()).sum()

    # Perplexity = exp(entropy)
    perplexity = torch.exp(entropy)

    # Active / dead codes
    active_codes = (counts > 0).sum()
    dead_code_ratio = 1.0 - active_codes.float() / counts.numel()

    stats = {
        f"{prefix}/usage_entropy": entropy.item(),
        f"{prefix}/perplexity": perplexity.item(),
        f"{prefix}/active_codes": active_codes.item(),
        f"{prefix}/dead_code_ratio": dead_code_ratio.item(),
        f"{prefix}/max_usage_ratio": probs.max().item(),
        f"{prefix}/min_usage_ratio": probs[probs > 0].min().item()
        if (counts > 0).any()
        else 0.0,
    }

    return stats
