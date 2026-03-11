from conf.initializer.full import *
from utils.dotenv import load_dotenv
from utils.logger import get_logger

VAE_EXP_PREFIX = "vae"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

logger = get_logger(__file__)


def init_config(config):
    """Initialize configuration for training.

    Args:
        config: Configuration object containing training parameters.
    """
    # Set device
    if not torch.cuda.is_available() and config.device == "cuda":
        config.device = "cpu"
        logger.warning("CUDA is not available. Switching to CPU.")

    config.exp_group = get_exp_group(config)
    config.exp_name = get_exp_name(config)
    config.exp_path = os.path.join(config.saves_dir, config.exp_name)
    config.game_list = get_game_name(config.games)

    # import .env file if exists in the experiment path
    env_path = os.path.join(ROOT_DIR, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)

    # wandb setup
    if config.wandb_project is None or os.environ.get("WANDB_API_KEY") is None:
        config.use_wandb = False
    elif config.use_wandb:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
        config.wandb_id = f"{config.exp_name}-{timestamp}"
    else:
        pass

    if config.vit_batch_size is None:
        config.vit_batch_size = config.batch_size
        logger.info(f"vit_batch_size not specified, using train batch_size: {config.vit_batch_size}")

    return config