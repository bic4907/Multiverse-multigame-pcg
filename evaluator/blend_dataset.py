from dataclasses import dataclass
from typing import Union, List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_loader import LevelData
from utils.logger import get_logger


logger = get_logger(__file__)

@dataclass
class PairedLevelData:
    level_data_a: LevelData
    embedding_a: np.ndarray
    level_data_b: LevelData
    embedding_b: np.ndarray

@dataclass
class BlendRatio:
    ratio_a: float
    ratio_b: float

@dataclass
class BlendedLevelData:
    paired_data: PairedLevelData
    blend_ratio: BlendRatio
    interpolated_embedding: np.ndarray

@dataclass
class TextBlendedLevelData:
    level_data_a: LevelData
    level_data_b: LevelData
    blend_type: str
    level_id_a: int
    level_id_b: int

class BlendedLevelDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        text_embeddings: Union[torch.Tensor, np.ndarray],
        blend_ratios: List[BlendRatio],
        max_data_length: int,
    ):
        self.base_dataset = base_dataset
        self.blend_ratios = blend_ratios
        self.max_data_length = max_data_length

        if isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.detach().cpu().numpy()
        self.text_embeddings = text_embeddings

        assert len(self.text_embeddings) == len(self.base_dataset), \
            "text_embeddings and dataset length mismatch"

        self.n = len(self.base_dataset)

    def __len__(self):
        return self.max_data_length

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        i = rng.randint(0, self.n)

        # Ensure j is from a different game than i
        max_attempts = 100
        j = None
        for attempt in range(max_attempts):
            j_candidate = rng.randint(0, self.n - 1)
            if j_candidate >= i:
                j_candidate += 1

            # Check if games are different
            game_i = self.base_dataset.samples[i].game
            game_j = self.base_dataset.samples[j_candidate].game

            if game_i != game_j:
                j = j_candidate
                break

        if j is None:
            # If we couldn't find a different game after max_attempts, use any j and log a warning
            j = rng.randint(0, self.n - 1)
            if j >= i:
                j += 1
            logger.warning(f"Could not find different game pair after {max_attempts} attempts for idx={idx}. Using same game pair.")

        ratio = self.blend_ratios[
            rng.randint(0, len(self.blend_ratios))
        ]

        emb_a = self.text_embeddings[i]
        emb_b = self.text_embeddings[j]

        interp_emb = (
                ratio.ratio_a * emb_a +
                ratio.ratio_b * emb_b
        )

        paired = PairedLevelData(
            level_data_a=self.base_dataset[i],
            embedding_a=emb_a,
            level_data_b=self.base_dataset[j],
            embedding_b=emb_b,
        )

        return BlendedLevelData(
            paired_data=paired,
            blend_ratio=ratio,
            interpolated_embedding=interp_emb,
        )

    @staticmethod
    def collate(batch: List[BlendedLevelData]):
        levels_a = [item.paired_data.level_data_a['level'] for item in batch]
        texts_a = [item.paired_data.level_data_a['text'] for item in batch]
        game_a = [item.paired_data.level_data_a['game'] for item in batch]

        levels_b = [item.paired_data.level_data_b['level'] for item in batch]
        texts_b = [item.paired_data.level_data_b['text'] for item in batch]
        game_b = [item.paired_data.level_data_b['game'] for item in batch]

        blend_ratios = [
            (item.blend_ratio.ratio_a, item.blend_ratio.ratio_b)
            for item in batch
        ]

        interp_embeddings = torch.from_numpy(
            np.stack(
                [item.interpolated_embedding for item in batch],
                axis=0
            )
        ).float()

        return {
            "game_a": game_a,
            "level_a": torch.stack(levels_a),
            "text_a": texts_a,
            "game_b": game_b,
            "level_b": torch.stack(levels_b),
            "text_b": texts_b,
            "blend_ratios": blend_ratios,
            "interpolated_embeddings": interp_embeddings,
        }

def make_blender_dataloader(
    data_loader: DataLoader,
    text_embeddings: Union[torch.Tensor, np.ndarray],
    max_data_length: int = 1000,
    blend_ratios: List[BlendRatio] = [BlendRatio(0.0, 1.0), BlendRatio(0.25, 0.75), BlendRatio(0.5, 0.5), BlendRatio(0.75, 0.25), BlendRatio(1.0, 0.0)],
    batch_size: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    """
    Wrap an existing dataloader into a blended-level dataloader
    without materializing all pair combinations.
    """

    base_dataset = data_loader.dataset

    blended_dataset = BlendedLevelDataset(
        base_dataset=base_dataset,
        text_embeddings=text_embeddings,
        blend_ratios=blend_ratios,
        max_data_length=max_data_length,
    )

    return DataLoader(
        blended_dataset,
        batch_size=batch_size or data_loader.batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=blended_dataset.collate,
    )

class TextBlendedLevelDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        blend_type: List[str],
        max_data_length: int,
    ):
        self.base_dataset = base_dataset
        self.max_data_length = max_data_length
        self.blend_type = blend_type

        self.n = len(self.base_dataset)

    def __len__(self):
        return self.max_data_length

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        i = rng.randint(0, self.n)

        # Ensure j is from a different game than i
        max_attempts = 100
        j = None
        for attempt in range(max_attempts):
            j_candidate = rng.randint(0, self.n - 1)
            if j_candidate >= i:
                j_candidate += 1

            # Check if games are different
            game_i = self.base_dataset.samples[i].game
            game_j = self.base_dataset.samples[j_candidate].game

            if game_i != game_j:
                j = j_candidate
                break

        if j is None:
            # If we couldn't find a different game after max_attempts, use any j and log a warning
            j = rng.randint(0, self.n - 1)
            if j >= i:
                j += 1
            logger.warning(f"Could not find different game pair after {max_attempts} attempts for idx={idx}. Using same game pair.")

        blend_type = self.blend_type[
                    rng.randint(0, len(self.blend_type))
                ]
        
        level_data_a=self.base_dataset[i]
        level_data_b=self.base_dataset[j]

        return TextBlendedLevelData(
            level_data_a=level_data_a,
            level_data_b=level_data_b,
            blend_type=blend_type,
            level_id_a=i,
            level_id_b=j,
        )

    @staticmethod
    def collate(batch: List[TextBlendedLevelData]):
        levels_a = [item.level_data_a['level'] for item in batch]
        texts_a = [item.level_data_a['text'] for item in batch]
        game_a = [item.level_data_a['game'] for item in batch]
        level_id_a = [item.level_id_a for item in batch]

        levels_b = [item.level_data_b['level'] for item in batch]
        texts_b = [item.level_data_b['text'] for item in batch]
        game_b = [item.level_data_b['game'] for item in batch]
        level_id_b = [item.level_id_b for item in batch]

        blend_type = [item.blend_type for item in batch]

        return {
            "game_a": game_a,
            "level_a": torch.stack(levels_a),
            "text_a": texts_a,
            "level_id_a": level_id_a,

            "game_b": game_b,
            "level_b": torch.stack(levels_b),
            "text_b": texts_b,
            "level_id_b": level_id_b,

            "blend_type": blend_type,
        }

    
def make_text_blender_dataloader(
    data_loader: DataLoader,
    max_data_length: int = 1000,
    blend_type: List[str] = ['concat', 'mix', 'a_base', 'b_base'],
    batch_size: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    """
    Wrap an existing dataloader into a text blended-level dataloader
    without materializing all pair combinations.
    """

    base_dataset = data_loader.dataset

    blended_dataset = TextBlendedLevelDataset(
        base_dataset=base_dataset,
        blend_type=blend_type,
        max_data_length=max_data_length,
    )

    return DataLoader(
        blended_dataset,
        batch_size=batch_size or data_loader.batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=blended_dataset.collate,
    )
