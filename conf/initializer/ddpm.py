"""conf/initializer/ddpm.py
==========================
Initialise DDPMConfig before training.
Mirrors the logic of conf/initializer/fdm.py.
"""

import os
import shutil
from datetime import datetime

from utils.dotenv import load_dotenv
from utils.logger import get_logger
import torch

DDPM_EXP_PREFIX = "ddpm"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

logger = get_logger(__file__)

GAME_NAME_MAP = {
    "dg": "dungeon",
    "lr": "lode_runner",
    "smb": "super_mario_bros",
    "tloz": "the_legend_of_zelda",
}

EXP_NAME_MAP = {
    "ddpm_T":        "T",
    "ddpm_base_ch":  "ch",
    "ddpm_n_levels": "lvl",
    "lr":            "lr",
    "batch_size":    "bs",
}


def format_exp_fields(config, field_map):
    parts = []
    for cfg_key, short in field_map.items():
        if not hasattr(config, cfg_key):
            continue
        value = getattr(config, cfg_key)
        if isinstance(value, (list, tuple)):
            value = "-".join(map(str, value))
        parts.append(f"{short}-{value}")
    return parts


def get_exp_group(config):
    parts = [DDPM_EXP_PREFIX]
    parts += format_exp_fields(config, EXP_NAME_MAP)
    return "_".join(parts)


def get_exp_name(config):
    return f"{config.exp_group}_s-{config.seed}"


def get_game_name(game_str: str) -> list[str]:
    keys = game_str.split("_")
    games = []
    for k in keys:
        if k not in GAME_NAME_MAP:
            raise ValueError(
                f"Unknown game name: {k}. Available: {list(GAME_NAME_MAP.keys())}"
            )
        games.append(GAME_NAME_MAP[k])
    return games


def init_config(config):
    """Initialise DDPMConfig for training."""
    if not torch.cuda.is_available() and config.device == "cuda":
        config.device = "cpu"
        logger.warning("CUDA is not available. Switching to CPU.")

    config.game_list = get_game_name(config.games)
    config.exp_group = get_exp_group(config)
    config.exp_name  = get_exp_name(config)
    config.exp_path  = os.path.join(config.saves_dir, config.exp_name)

    if config.overwrite:
        if os.path.exists(config.exp_path):
            shutil.rmtree(config.exp_path)

    if os.path.exists(config.exp_path):
        raise ValueError(
            f"Experiment path {config.exp_path} already exists. "
            "Use 'overwrite=True' to overwrite."
        )

    os.makedirs(config.exp_path)

    env_path = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    if config.wandb_project is None or os.environ.get("WANDB_API_KEY") is None:
        config.use_wandb = False
    elif config.use_wandb:
        now       = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
        config.wandb_id = f"{config.exp_name}-{timestamp}"

    if config.vit_batch_size is None:
        config.vit_batch_size = config.batch_size
        logger.info(
            f"vit_batch_size not specified, using batch_size: {config.vit_batch_size}"
        )

    # resolve relative paths to absolute (relative to project root)
    config.annotation_csv = os.path.join(ROOT_DIR, config.annotation_csv)
    config.embedding_path = os.path.join(ROOT_DIR, config.embedding_path)

    return config

