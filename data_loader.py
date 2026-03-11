from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from os.path import join, abspath, dirname, exists

from transformers import CLIPTokenizer
from utils.logger import get_logger


DEFAULT_DATASET_ROOT = join(dirname(abspath(__file__)), "dataset", "processed_levels")


logger = get_logger(__file__)

class ClipCollator:
    def __init__(self, tokenizer_model: str, device: str = "cpu"):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model)
        self.device = device

    def __call__(self, batch):
        levels = torch.stack([b["level"] for b in batch]).to(self.device)
        texts = [b["text"] for b in batch]
        games = [b["game"] for b in batch]

        text_inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        output = {
            "level": levels,
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "raw_text": list(texts),
            "games": list(games),
        }

        if batch[0].get("embedding_idx") is not None:
            output["embedding_idx"] = torch.tensor(
                [b["embedding_idx"] for b in batch],
                dtype=torch.long,
                device=self.device,
            )

        if batch[0].get("difficulty") is not None:
            output["difficulty"] = torch.tensor(
                [b["difficulty"] for b in batch],
                dtype=torch.long,
                device=self.device,
            )

        return output

@dataclass
class LevelData:
    game: str
    level_size: int
    text: str
    level: np.ndarray
    level_path: str
    embedding_idx: Optional[int] = None
    difficulty: Optional[int] = None

class LevelDataset(Dataset):
    def __init__(
        self,
        instruct_df: pd.DataFrame,
        dataset_root: str = DEFAULT_DATASET_ROOT,
        sample_ratio: Optional[float] = None,
        active_losses: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.device = device

        self.samples = []
        self._cache = {}

        self.dataset_root = dataset_root
        self.sample_ratio = sample_ratio

        EMBEDDING_LOSSES = {"gen", "spec"}
        DIFFICULTY_LOSSES = {"diff"}
        self.use_embedding = bool(set(active_losses or []) & EMBEDDING_LOSSES)
        self.use_difficulty = bool(set(active_losses or []) & DIFFICULTY_LOSSES)

        self._build_samples(instruct_df)

    def _build_samples(self, df: pd.DataFrame):
        samples = []

        for _, row in df.iterrows():
            game = row["game"]
            level_size = int(row["data_size"])

            level_dir = join(
                self.dataset_root,
                row["game"],
                f"size_{level_size}",
            )
            level_path = join(level_dir, row["filename"] + ".npy")

            if not exists(level_path):
                continue

            level = np.load(level_path).astype(np.uint8)

            if level.ndim == 3:
                level = level[0]

            samples.append(
                LevelData(
                    game=game,
                    level_size=level_size,
                    text=row["instruction"].replace("_", " "),
                    level=level,
                    level_path=level_path,
                    embedding_idx=int(row["embedding_idx"]) if self.use_embedding else None,
                    difficulty=int(row["difficulty"]) if self.use_difficulty else None,
                )
            )

        # randomly subsample if specified
        if self.sample_ratio is not None and 0 < self.sample_ratio < 1.0:
            n_total = len(samples)
            n_select = int(n_total * self.sample_ratio)
            logger.info(f"Sampling {n_select}/{n_total} samples from dataset.")
            samples = np.random.choice(
                samples, size=n_select, replace=False
            ).tolist()

        self.samples = samples

    @property
    def num_classes(self):
        if hasattr(self, '_num_classes'):
            return self._num_classes

        stacked = np.stack([sample.level for sample in self.samples], axis=0)
        self._num_classes = int(stacked.max())

        return self._num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        level, text, game, embedding_idx = sample.level, sample.text, sample.game, sample.embedding_idx
        if self.use_embedding:
            embedding_idx = sample.embedding_idx
        if self.use_difficulty:
            difficulty = sample.difficulty

        if level.ndim == 3:
            level = level[0]
        assert level.ndim == 2, f"Expected level to be 2D, got {level.shape}"

        # one-hot encode
        onehot = np.eye(self.num_classes + 1, dtype=np.float32)[level]
        onehot = np.transpose(onehot, (2, 0, 1))
        # drop empty channel (0)
        onehot = onehot[1:]

        return {
            "level": torch.from_numpy(onehot),
            "text": text,
            "game": game,
            "embedding_idx": embedding_idx if self.use_embedding else None,
            "difficulty": difficulty if self.use_difficulty else None,
        }


class IndexedLevelDataset(Dataset):
    """Dataset that allows accessing specific game data by (game, index) tuple.

    Unlike LevelDataset which shuffles data randomly, this dataset organizes
    samples by game and allows deterministic access via game name and index.

    Usage:
        dataset = IndexedLevelDataset(df)
        # Get 5th sample from 'super_mario_bros' game
        sample = dataset.get_by_game('super_mario_bros', 5)
        # Or get total count for a game
        count = dataset.get_game_count('super_mario_bros')
    """

    def __init__(
        self,
        instruct_df: pd.DataFrame,
        dataset_root: str = DEFAULT_DATASET_ROOT,
        active_losses: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.dataset_root = dataset_root

        # Game-indexed storage: {game_name: [LevelData, ...]}
        self.game_samples = {}
        # Flat list for standard indexing
        self.samples = []
        # Mapping from flat index to (game, game_index)
        self.index_to_game = []

        EMBEDDING_LOSSES = {"gen", "spec"}
        DIFFICULTY_LOSSES = {"diff"}
        self.use_embedding = bool(set(active_losses or []) & EMBEDDING_LOSSES)
        self.use_difficulty = bool(set(active_losses or []) & DIFFICULTY_LOSSES)

        self._build_indexed_samples(instruct_df)

    def _build_indexed_samples(self, df: pd.DataFrame):
        """Build game-indexed structure from dataframe."""
        for _, row in df.iterrows():
            game = row["game"]
            level_size = int(row["data_size"])

            level_dir = join(
                self.dataset_root,
                row["game"],
                f"size_{level_size}",
            )
            level_path = join(level_dir, row["filename"] + ".npy")

            if not exists(level_path):
                continue

            level = np.load(level_path).astype(np.uint8)

            if level.ndim == 3:
                level = level[0]

            level_data = LevelData(
                game=game,
                level_size=level_size,
                text=row["instruction"],
                level=level,
                level_path=level_path,
                embedding_idx=int(row["embedding_idx"]) if self.use_embedding else None,
                difficulty=int(row["difficulty"]) if self.use_difficulty else None,
            )

            # Add to game-specific list
            if game not in self.game_samples:
                self.game_samples[game] = []
            game_index = len(self.game_samples[game])
            self.game_samples[game].append(level_data)

            # Add to flat list and maintain mapping
            self.samples.append(level_data)
            self.index_to_game.append((game, game_index))

        logger.info(f"IndexedLevelDataset loaded {len(self.samples)} samples across {len(self.game_samples)} games:")
        for game, samples in self.game_samples.items():
            logger.info(f"  {game}: {len(samples)} samples")

    @property
    def num_classes(self):
        if hasattr(self, '_num_classes'):
            return self._num_classes

        stacked = np.stack([sample.level for sample in self.samples], axis=0)
        self._num_classes = int(stacked.max())

        return self._num_classes

    @property
    def games(self):
        """Return list of available game names."""
        return list(self.game_samples.keys())

    def get_game_count(self, game: str) -> int:
        """Get number of samples for a specific game."""
        if game not in self.game_samples:
            raise ValueError(f"Game '{game}' not found. Available games: {self.games}")
        return len(self.game_samples[game])

    def get_by_game(self, game: str, index: int):
        """Get a specific sample by game name and index within that game.

        Args:
            game: Game name (e.g., 'super_mario_bros')
            index: Index within that game's samples (0-based)

        Returns:
            Dict with 'level', 'text', 'game', etc.
        """
        if game not in self.game_samples:
            raise ValueError(f"Game '{game}' not found. Available games: {self.games}")

        game_samples = self.game_samples[game]
        if index < 0 or index >= len(game_samples):
            raise IndexError(f"Index {index} out of range for game '{game}' (0-{len(game_samples)-1})")

        return self._process_sample(game_samples[index])

    def _process_sample(self, sample: LevelData):
        """Process a LevelData into the output format."""
        level = sample.level

        if level.ndim == 3:
            level = level[0]
        assert level.ndim == 2, f"Expected level to be 2D, got {level.shape}"

        # one-hot encode
        onehot = np.eye(self.num_classes + 1, dtype=np.float32)[level]
        onehot = np.transpose(onehot, (2, 0, 1))
        # drop empty channel (0)
        onehot = onehot[1:]

        return {
            "level": torch.from_numpy(onehot),
            "text": sample.text,
            "game": sample.game,
            "embedding_idx": sample.embedding_idx if self.use_embedding else None,
            "difficulty": sample.difficulty if self.use_difficulty else None,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Standard indexing - accesses flat list sequentially."""
        return self._process_sample(self.samples[idx])

def make_indexed_level_dataset(
    csv_path: str,
    game_list: list[str],
    level_size: int,
    active_losses: Optional[list[str]] = None,
    device: str = "cpu",
    dataset_root: str = DEFAULT_DATASET_ROOT,
) -> IndexedLevelDataset:
    """Create an IndexedLevelDataset from CSV file.

    Args:
        csv_path: Path to the instruction CSV file
        game_list: List of game names to include (e.g., ['super_mario_bros', 'zelda'])
        level_size: Size of levels to load (e.g., 14)
        active_losses: Optional list of active loss types (e.g., ['gen', 'spec'])
        device: Device to use ('cpu' or 'cuda')
        dataset_root: Root directory containing processed level data

    Returns:
        IndexedLevelDataset instance with game-indexed samples

    Example:
        dataset = make_indexed_level_dataset(
            csv_path='dataset/processed_levels/annotation.csv',
            game_list=['super_mario_bros', 'the_legend_of_zelda'],
            level_size=14,
            active_losses=['gen'],
            device='cuda'
        )
        # Access specific game data
        mario_sample = dataset.get_by_game('super_mario_bros', 5)
    """
    df = pd.read_csv(csv_path)
    df = df[(df["data_size"] == level_size) & (df["game"].isin(game_list))].reset_index(drop=True)

    logger.info(f"Creating IndexedLevelDataset with {len(df)} samples from {len(game_list)} games")

    dataset = IndexedLevelDataset(
        instruct_df=df,
        dataset_root=dataset_root,
        active_losses=active_losses,
        device=device,
    )

    return dataset

def make_dataloaders(
    csv_path: str,
    batch_size: int,
    clip_model: str,
    sample_ratio: Optional[float],
    game_list: list[str],
    level_size: int,
    active_losses: list[str],
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    trainset_game: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:

    df = pd.read_csv(csv_path)
    df = df[(df["data_size"] == level_size) & (df["game"].isin(game_list))].reset_index(drop=True)

    collator = ClipCollator(tokenizer_model=clip_model, device=device)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Optionally filter training split to a single target game
    if trainset_game is not None:
        before = len(train_df)
        train_df = train_df[train_df["game"] == trainset_game].reset_index(drop=True)
        logger.info(f"Filtered train set by target_game='{trainset_game}': {len(train_df)}/{before} samples kept")

    if trainset_game is not None:
        before = len(val_df)
        val_df = val_df[val_df["game"] == trainset_game].reset_index(drop=True)
        logger.info(f"Filtered val set by target_game='{trainset_game}': {len(val_df)}/{before} samples kept")

    train_ds = LevelDataset(
        instruct_df=train_df,
        sample_ratio=sample_ratio,
        active_losses=active_losses,
        device=device,
    )

    val_ds = LevelDataset(
        instruct_df=val_df,
        sample_ratio=1.0,
        active_losses=active_losses,
        device=device,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )

    logger.info(f"Dataset split: train={len(train_ds)} | val={len(val_ds)}")

    return train_loader, val_loader


# ============================================================
# FDM – Dataset / Collator (uses pre-computed text embeddings)
# ============================================================

# Default paths (same folder as annotation.csv = dataset/processed_levels/)
FDM_DATASET_ROOT    = DEFAULT_DATASET_ROOT
FDM_ANNOTATION_CSV  = join(DEFAULT_DATASET_ROOT, "general_annotation.csv")
FDM_EMBEDDING_PATH  = join(DEFAULT_DATASET_ROOT, "game_general_text_embeddings.pt")


@dataclass
class FDMLevelData:
    game: str
    level_size: int
    text: str
    level: np.ndarray
    level_path: str
    embedding_idx: int   # index into the pre-loaded embedding tensor


class FDMLevelDataset(Dataset):
    """Dataset for the Five Dollar Model.

    Uses pre-computed text embeddings stored in a ``.pt`` file
    (shape: ``(N, embed_dim)``).  Each sample's ``embedding_idx`` from
    ``general_annotation.csv`` is used to look up the embedding vector.

    The original FDM scales embeddings by ``embedding_scale``
    (default 6, matching Merino et al.).
    """

    def __init__(
        self,
        instruct_df: pd.DataFrame,
        embeddings: torch.Tensor,           # (N, D) pre-loaded tensor
        dataset_root: str = FDM_DATASET_ROOT,
        sample_ratio: Optional[float] = None,
        embedding_scale: float = 6.0,
    ) -> None:
        self.dataset_root    = dataset_root
        self.sample_ratio    = sample_ratio
        self.embedding_scale = embedding_scale
        self.embeddings      = embeddings   # shared across train/val

        self.samples: list[FDMLevelData] = []
        self._build_samples(instruct_df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        samples = []
        for _, row in df.iterrows():
            game       = row["game"]
            level_size = int(row["data_size"])

            level_path = join(
                self.dataset_root,
                game,
                f"size_{level_size}",
                row["filename"] + ".npy",
            )
            if not exists(level_path):
                continue

            level = np.load(level_path).astype(np.uint8)
            if level.ndim == 3:
                level = level[0]

            samples.append(FDMLevelData(
                game=game,
                level_size=level_size,
                text=row["instruction"].replace("_", " "),
                level=level,
                level_path=level_path,
                embedding_idx=int(row["embedding_idx"]),
            ))

        if self.sample_ratio is not None and 0 < self.sample_ratio < 1.0:
            n_select = int(len(samples) * self.sample_ratio)
            logger.info(f"FDM: sampling {n_select}/{len(samples)} samples.")
            samples = np.random.choice(samples, size=n_select, replace=False).tolist()

        self.samples = samples

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def num_classes(self) -> int:
        if hasattr(self, "_num_classes"):
            return self._num_classes
        self._num_classes = int(np.stack([s.level for s in self.samples]).max())
        return self._num_classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        level = s.level
        if level.ndim == 3:
            level = level[0]
        assert level.ndim == 2

        onehot = np.eye(self.num_classes + 1, dtype=np.float32)[level]
        onehot = np.transpose(onehot, (2, 0, 1))[1:]   # drop empty channel 0

        # look up pre-computed embedding and apply scale
        emb = self.embeddings[s.embedding_idx] * self.embedding_scale  # (D,)

        return {
            "level":     torch.from_numpy(onehot),
            "embedding": emb.float(),
            "text":      s.text,
            "game":      s.game,
        }


class FDMCollator:
    """Minimal collator for FDMLevelDataset – no tokeniser needed."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def __call__(self, batch: list[dict]) -> dict:
        return {
            "level":     torch.stack([b["level"]     for b in batch]).to(self.device),
            "embedding": torch.stack([b["embedding"] for b in batch]).to(self.device),
            "raw_text":  [b["text"] for b in batch],
            "games":     [b["game"] for b in batch],
        }


def make_fdm_dataloaders(
    batch_size: int,
    game_list: list[str],
    level_size: int,
    *,
    sample_ratio: float = 1.0,
    embedding_scale: float = 6.0,
    val_ratio: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
    dataset_root: str = FDM_DATASET_ROOT,
    annotation_csv: str = FDM_ANNOTATION_CSV,
    embedding_path: str = FDM_EMBEDDING_PATH,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for the Five Dollar Model.

    Reads ``general_annotation.csv`` (game / data_size / filename /
    instruction / embedding_idx) and looks up embeddings from
    ``game_general_text_embeddings.pt`` via ``embedding_idx``.
    """

    # ── load annotation ──────────────────────────────────────────────────────
    ann_df = pd.read_csv(annotation_csv)
    ann_df = ann_df[
        (ann_df["data_size"] == level_size) & (ann_df["game"].isin(game_list))
    ].reset_index(drop=True)

    logger.info(f"FDM: annotation rows after filter = {len(ann_df)}")

    # ── load pre-computed embeddings (N, D) ───────────────────────────────────
    embeddings: torch.Tensor = torch.load(embedding_path, map_location="cpu")
    logger.info(f"FDM: embedding tensor shape = {embeddings.shape}")

    train_df, val_df = train_test_split(
        ann_df, test_size=val_ratio, random_state=seed, shuffle=True
    )

    train_ds = FDMLevelDataset(
        train_df,
        embeddings=embeddings,
        dataset_root=dataset_root,
        sample_ratio=sample_ratio,
        embedding_scale=embedding_scale,
    )
    val_ds = FDMLevelDataset(
        val_df,
        embeddings=embeddings,
        dataset_root=dataset_root,
        sample_ratio=1.0,
        embedding_scale=embedding_scale,
    )

    collator = FDMCollator(device=device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, drop_last=True, collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, drop_last=False, collate_fn=collator,
    )

    logger.info(
        f"FDM dataset split: train={len(train_ds)} | val={len(val_ds)} "
        f"(embed_dim={train_ds.embed_dim})"
    )
    return train_loader, val_loader


