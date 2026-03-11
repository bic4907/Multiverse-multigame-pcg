from __future__ import annotations

from tqdm import tqdm
from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

from evaluator.base import BaseEvaluator
from evaluator.comparison_set import TwoPairComparisonBatch, ThreePairComparisonBatch
from utils.logger import get_logger
from utils.renderer import render_level

from evaluator import (
    TwoPairComparisonSet,
    TwoPairComparisonScore,
    ThreePairComparisonSet,
    ThreePairComparisonScore,
)

logger = get_logger(__file__)


class ViTEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def preload(self):
        # dummy forward to preload model weights to GPU with batch size
        if self.device == "cpu":
            logger.warning(f"Using CPU for ViT model. This may be significantly slower than GPU.")
        
        logger.info(f"Preloading ViT model to {self.device} with batch_size={self.batch_size}...")

        dummy_imgs = [Image.new("RGB", (224, 224), color="white") for _ in range(self.batch_size)]
        inputs = self.processor(images=dummy_imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        _ = self.model(**inputs)

        logger.info(f"ViT model preloaded to {self.device} (tested with batch_size={self.batch_size}).")


    @torch.no_grad()
    def _embed_level(self, level: np.ndarray) -> torch.Tensor:
        """
        level: (H, W) ndarray
        return: (D,) torch tensor on CPU
        """
        img = render_level(level)  # RGB np.ndarray (H, W, 3)
        img = Image.fromarray(img).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS (1, D)

        if self.normalize:
            emb = F.normalize(emb, dim=-1)

        return emb.squeeze(0).detach().cpu()  # (D,)

    @torch.no_grad()
    def _embed_levels_batch(self, levels: List[np.ndarray]) -> torch.Tensor:
        """
        levels: List[(H, W)]
        return: (N, D) torch tensor on CPU
        """
        imgs = [Image.fromarray(render_level(level)).convert("RGB") for level in levels]

        num_levels = len(imgs)
        num_batches = (num_levels + self.batch_size - 1) // self.batch_size

        all_embs = []
        for i in tqdm(range(0, len(imgs), self.batch_size), desc="Embedding levels with ViT"):
            batch_imgs = imgs[i : i + self.batch_size]

            inputs = self.processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # (B, D)

            if self.normalize:
                emb = F.normalize(emb, dim=-1)

            all_embs.append(emb.detach().cpu())

        if len(all_embs) == 0:
            # (0, D) empty tensor
            hidden = self.model.config.hidden_size
            return torch.empty((0, hidden), dtype=torch.float32)

        result = torch.cat(all_embs, dim=0)  # (N, D)
        return result

    @torch.no_grad()
    def evaluate_two_pair(
        self, comparison: List[TwoPairComparisonSet]
    ) -> List[TwoPairComparisonScore]:
        scores: List[TwoPairComparisonScore] = []
        for pair in comparison:
            emb_a = self._embed_level(pair.a)
            emb_b = self._embed_level(pair.b)
            score = torch.sum(emb_a * emb_b).item()
            scores.append(TwoPairComparisonScore(score))
        return scores

    @torch.no_grad()
    def evaluate_three_pair(
        self, comparison: List[ThreePairComparisonSet]
    ) -> List[ThreePairComparisonScore]:
        scores: List[ThreePairComparisonScore] = []
        for triplet in comparison:
            emb_a = self._embed_level(triplet.a)
            emb_b = self._embed_level(triplet.b)
            emb_c = self._embed_level(triplet.c)
            metadata = triplet.metadata

            score_ab = torch.sum(emb_a * emb_b).item()
            score_ac = torch.sum(emb_a * emb_c).item()
            scores.append(ThreePairComparisonScore(score_ab, score_ac, metadata=metadata))
        return scores

    def _check_comparison_types(self, comparisons):
        types = set(type(c) for c in comparisons)
        if len(types) > 1:
            type_names = ", ".join([t.__name__ for t in types])
            logger.warning(
                f"Mixed comparison types detected: {type_names}. "
                "Returned results may have heterogeneous shapes."
            )
        return types

    def run(
        self,
        comparisons: List[Union[TwoPairComparisonSet, ThreePairComparisonSet]],
    ) -> Tuple[TwoPairComparisonBatch, ThreePairComparisonBatch]:
        """
        Returns:
          - TwoPairComparisonBatch: scores for all TwoPairComparisonSet in input order (within the two-only list)
          - ThreePairComparisonBatch: scores for all ThreePairComparisonSet in input order (within the three-only list)

        NOTE: If you want a single list aligned 1:1 with `comparisons`, you'll need a different return structure.
        """
        self._check_comparison_types(comparisons)

        # ---- 1) flatten all levels + build an index map (FIX: don't misuse Score objects)
        levels: List[np.ndarray] = []
        index_map: List[Tuple] = []  # ("two", i, j) or ("three", i, j, k)

        for comp in comparisons:
            if isinstance(comp, TwoPairComparisonSet):
                i = len(levels)
                levels.append(comp.a)
                j = len(levels)
                levels.append(comp.b)
                index_map.append(("two", i, j, comp.metadata))

            elif isinstance(comp, ThreePairComparisonSet):
                i = len(levels)
                levels.append(comp.a)
                j = len(levels)
                levels.append(comp.b)
                k = len(levels)
                levels.append(comp.c)
                index_map.append(("three", i, j, k, comp.metadata))

            else:
                raise ValueError(f"Unsupported comparison set type: {type(comp)}")

        # ---- 2) batch embedding
        embs = self._embed_levels_batch(levels)  # (N, D) on CPU

        # ---- 3) compute scores (FIX: correct indexing & build proper score objects)
        two_scores: List[TwoPairComparisonScore] = []
        three_scores: List[ThreePairComparisonScore] = []

        for info in index_map:
            tag = info[0]

            if tag == "two":
                _, i, j, metadata = info
                score = torch.sum(embs[i] * embs[j]).item()
                two_scores.append(TwoPairComparisonScore(score, metadata=metadata))

            elif tag == "three":
                _, i, j, k, metadata = info
                score_ac = torch.sum(embs[i] * embs[k]).item()
                score_bc = torch.sum(embs[j] * embs[k]).item()
                three_scores.append(
                    ThreePairComparisonScore(score_ac, score_bc, metadata=metadata)
                )

            else:
                raise RuntimeError(f"Unknown index_map tag: {tag}")

        # ---- 4) pack batches (keep your existing batch API usage consistent)
        two_pair_batch = TwoPairComparisonBatch(two_scores)
        three_pair_batch = ThreePairComparisonBatch(three_scores)

        return two_pair_batch, three_pair_batch


if __name__ == "__main__":
    from PIL import ImageDraw, ImageFont

    # make numpy array level
    level_a = np.random.randint(1, 3, size=(16, 16), dtype=np.uint8)
    level_b = np.random.randint(1, 3, size=(16, 16), dtype=np.uint8)
    level_c = np.random.randint(1, 3, size=(16, 16), dtype=np.uint8)

    image_a = render_level(level_a)
    image_b = render_level(level_b)

    vit = ViTEvaluator()
    two_pair, three_pair = vit.run(
        comparisons=[
            TwoPairComparisonSet(level_a, level_b),
            TwoPairComparisonSet(level_a, level_c),
            ThreePairComparisonSet(level_a, level_b, level_c),
        ]
    )

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    img_a = Image.fromarray(image_a).convert("RGB")
    img_b = Image.fromarray(image_b).convert("RGB")

    draw_a = ImageDraw.Draw(img_a)
    draw_b = ImageDraw.Draw(img_b)
    draw_a.text((10, 10), "Level A", fill="black", font=font)
    draw_b.text((10, 10), "Level B", fill="black", font=font)

    combined_img = Image.new("RGB", (img_a.width + img_b.width, img_a.height), color="white")
    combined_img.paste(img_a, (0, 0))
    combined_img.paste(img_b, (img_a.width, 0))

    draw_combined = ImageDraw.Draw(combined_img)

    # NOTE: this assumes TwoPairComparisonBatch.scores is array-like (np/tensor/list)
    # If it's a python list, this still works with np.mean.
    mean_score = float(np.mean(two_pair.scores)) if len(two_pair.scores) > 0 else float("nan")
    score_text = f"ViT Score (mean of 2-pairs): {mean_score:.4f}"

    draw_combined.text((10, 10), score_text, fill="black", font=font)
    # combined_img.show()

    print("TwoPair batch:", two_pair, "scores:", two_pair.scores)
    print("ThreePair batch:", three_pair, "scores_ab:", three_pair.scores)
