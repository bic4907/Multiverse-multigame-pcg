from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

@dataclass
class Config:
    exp_group: Optional[str] = None
    exp_name: str = "def"
    exp_path: Optional[str] = None
    saves_dir: str = "saves"

    overwrite: bool = False

    seed: int = 0
    dataset_split_seed: int = 42

    lr: float = 1.0e-4

    use_wandb: bool = True
    wandb_id: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = 'anonymous'
    wandb_resume: str = 'allow'
    wandb_key: Optional[str] = None

    device: str = "cuda"
    
    # dataset
    games: str = "smb_tloz_lr_dg"
    game_list: Optional[list[str]] = None
    level_size: int = 16

    instruction_csv: str = "scn-1_se-whole.csv"
    instruction_csv_path: Optional[str] = None

    vit_eval_freq: int = 200
    vit_score_single: bool = True  # for single instruction experiment
    vit_score_blend: bool = True  # for blending experiment
    vit_batch_size: Optional[int] = None # if None, use the default train batch size
    n_vit_blend_samples: int = 2000

    save_ckpt: bool = True
    save_keep: int = 10
    save_interval: int = 200

@dataclass
class LossConfig:
    active_losses: Optional[list[str]] = None
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "base": 0.0,
            "spec": 0.0,
            "gen": 0.0,
            "diff": 0.0,
        }
    )


@dataclass
class CLIPTrainConfig(Config, LossConfig):
    wandb_project: str = 'multigame_clip'
    instruction_csv: str = "annotation.csv"
    specific_text_embedding_path:str = "dataset/processed_levels/game_specific_text_embeddings.pt"
    general_text_embedding_path:str = "dataset/processed_levels/game_general_text_embeddings.pt"

    spec_threshold: float = 0.7
    gen_threshold: float = 0.7

    sample_ratio: float = 1.0
    n_epochs: int = 100
    lr: float = 5e-5
    batch_size: int = 256
    drop_rate: float = 0.1
    init_temperature: float = 0.14
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    val_freq: int = 10

    clip_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 128

    draw_tsne: bool = True
    tsne_interval: int = 200


@dataclass
class CVAETrainConfig(Config, LossConfig):
    wandb_project: str = 'multigame_cvae'
    instruction_csv: str = "annotation.csv"

    sample_ratio: float = 1.0
    n_epochs: int = 100
    lr: float = 4e-4
    val_ratio: float = 0.1
    batch_size: int = 512
    train_ratio: float = 0.8
    val_freq: int = 10

    clip_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 128

    # VQ params
    code_dim: int = 128  # encoder output channels == code dimension
    num_codes: int = 256  # codebook size (discrete vocab size)
    vq_beta: float = 0.5  # commitment loss weight
    vq_beta_min: float = 0.3

    vq_beta_coef: float = 1.0
    vq_beta_coef_min: float = 0.6
    vq_beta_start_decay: float = 100

    render_interval: int = 200
    n_render_samples: int = 20

    trainable_clip: bool = True

@dataclass
class TrainConfig(CLIPTrainConfig, CVAETrainConfig):
    wandb_project: str = 'multigame_train'

    n_epochs: int = 2000
    batch_size: int = 512

    vae_lr: float = 4e-4
    clip_lr: float = 5e-5


@dataclass
class EvalConfig(TrainConfig):
    reevaluate: bool = False

@dataclass
class TextBlendEvalConfig(EvalConfig):
    openai_model:str = "gpt-5.2"
    batch_id: Optional[str] = None
    blend_samples: int = 2000

    ckpt_path: str = ""
    output_dir: str = "eval_textblend"

    checkpoint_path: Optional[str] = None
    checkpoint_epoch: Optional[int] = None

    blended_instuction_csv: str = "evaluator/text_blend/blended_instruction.csv"

    overwrite: bool = False

@dataclass
class PreProcessConfig:
    # data_preprocessor
    text_path:str = "Super_Mario_Bros/Processed/"
    tile_json:str = "Super_Mario_Bros/smb.json"
    raw_levels_path:str = "dataset/raw_levels/super_mario_bros/"
    processed_levels_path:str = "dataset/processed_levels/super_mario_bros/size_32/"

    processing_size:int = 32
    stride:int = 16

    # data_annotator
    image_exts:str = ".png"
    image_path:str = "dataset/rendered_levels/dungeon/size_16/"
    annotation_csv:str = "dataset/processed_levels/general_annotation.csv"
    openai_model:str = "gpt-5.2"

    # text_embedder
    embedding_path:str = "dataset/processed_levels/game_general_text_embeddings.pt"
    embedding_model:str = "text-embedding-3-large"

    # text_generalizer
    specific_annotation_csv:str = "instruct/annotation.csv"
    general_annotation_csv:str = "instruct/general_annotation.csv"

@dataclass
class RolloutConfig(TrainConfig):
    ckpt_path: str = ""
    output_dir: str = "infer_single"

    checkpoint_path: Optional[str] = None
    checkpoint_epoch: Optional[int] = None

    overwrite: bool = False
    games: str = "smb_tloz_lr_dg"


@dataclass
class RolloutBlendConfig(RolloutConfig):
    output_dir: str = "infer_blend"

@dataclass
class RolloutTextBlendConfig(RolloutConfig):
    output_dir: str = "infer_textblend"
    openai_model:str = "gpt-5.2"

@dataclass
class TrainSingleConfig(TrainConfig):
    trainset_game: str = "smb"


@dataclass
class FDMConfig(Config):
    """Configuration for the Five Dollar Model (Merino et al., AIIDE 2023).

    Default hyper-parameters are set to match the original implementation:
        epochs=100, batch_size=256, lr=5e-4, kern_size=5,
        z_dim=5, filter_count=128, num_res_blocks=3, num_upsample=2

    Key differences vs. original:
        - embedding_model: sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (384-dim, free)
        - embedding_scale: 6.0 (same as original scaling_factor)
        - level_size: determined by dataset (16 by default)
    """
    wandb_project: str = 'multigame_fdm'

    # --- data paths (relative to project root) ---
    annotation_csv: str  = "dataset/processed_levels/general_annotation.csv"
    embedding_model: str = "multi-qa-MiniLM-L6-cos-v1"
    embedding_path: str  = "dataset/processed_levels/game_specific_text_embeddings_minilm.pt"

    # --- training (matches original: epochs=2000, batch=256, lr=5e-4) ---
    n_epochs: int    = 500
    batch_size: int  = 256
    val_ratio: float = 0.1
    sample_ratio: float = 1.0
    lr: float = 5e-4              # original: lr=0.0005

    # --- embedding (scaling_factor=6 from original) ---
    embedding_scale: float = 6.0

    # --- FDM model architecture (matches original __init__ defaults) ---
    fdm_z_dim: int          = 5      # z_dim=5
    fdm_filter_count: int   = 128    # filter_count=128
    fdm_kern_size: int      = 5      # kern_size=5  (original demo uses 5, not 7)
    fdm_num_res_blocks: int = 3      # num_res_blocks=3
    fdm_num_upsample: int   = 2      # 4→8→16 for level_size=16

    # --- eval ---
    vit_score_single: bool = True
    vit_score_blend: bool  = False
    vit_batch_size: Optional[int] = None
    n_vit_blend_samples: int = 2000
    vit_eval_freq: int = 10          # every 10 epochs (10% of total)

    render_interval: int = 20        # every 20 epochs
    n_render_samples: int = 20

    save_ckpt: bool  = True
    save_keep: int   = 5
    save_interval: int = 20

@dataclass
class DDPMConfig(Config):
    """Configuration for the DDPM level generator (Ho et al., NeurIPS 2020).

    Default hyper-parameters are chosen to be compatible with the project's
    16×16 tile-map setting and pre-computed text embeddings.
    """
    wandb_project: str = 'multigame_ddpm'

    # --- data paths (same as FDM) ---
    annotation_csv: str = "dataset/processed_levels/general_annotation.csv"
    embedding_path: str = "dataset/processed_levels/game_general_text_embeddings.pt"

    # --- training ---
    n_epochs: int    = 300
    batch_size: int  = 256
    val_ratio: float = 0.1
    sample_ratio: float = 1.0
    lr: float = 2e-4

    # --- embedding (same scaling as FDM) ---
    embedding_scale: float = 6.0

    # --- DDPM diffusion schedule ---
    ddpm_T: int             = 1000     # total diffusion steps
    ddpm_beta_start: float  = 1e-4
    ddpm_beta_end: float    = 2e-2

    # --- UNet architecture ---
    ddpm_base_ch: int  = 64            # base channel count
    ddpm_n_levels: int = 2             # encoder/decoder depth
    ddpm_time_dim: int = 128           # sinusoidal time embedding dim

    # --- eval ---
    vit_score_single: bool = True
    vit_score_blend: bool  = False
    vit_batch_size: Optional[int] = None
    n_vit_blend_samples: int = 2000
    vit_eval_freq: int = 10

    eval_freq: int = 5               # every N epochs run full sampling eval

    render_interval: int = 20
    n_render_samples: int = 20

    save_ckpt: bool  = True
    save_keep: int   = 5
    save_interval: int = 20


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_full", node=TrainConfig)
cs.store(name="train_clip", node=CLIPTrainConfig)
cs.store(name="train_cvae", node=CVAETrainConfig)
cs.store(name="train_single", node=TrainSingleConfig)
cs.store(name="train_fdm", node=FDMConfig)
cs.store(name="train_ddpm", node=DDPMConfig)
cs.store(name="eval_textblend", node=TextBlendEvalConfig)
cs.store(name="infer_single", node=RolloutConfig)
cs.store(name="infer_blend", node=RolloutBlendConfig)
cs.store(name="infer_textblend", node=RolloutTextBlendConfig)

