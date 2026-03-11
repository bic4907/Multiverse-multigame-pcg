import wandb
import hydra

from conf.config import TrainConfig
from conf.initializer.full import init_config, GAME_NAME_MAP
from train import train
from multiprocessing import Process, Queue
from analysis.tsne_visualize import tsne_worker
from utils.logger import get_logger

logger = get_logger(__file__)


@hydra.main(version_base=None, config_path="conf", config_name="train_single")
def main(config: TrainConfig):
    init_config(config)

    logger.info(f"Experiment Directory: {config.exp_path}")

    if config.trainset_game is not GAME_NAME_MAP.keys():
        config.trainset_game = GAME_NAME_MAP.get(config.trainset_game, None)

    assert config.trainset_game is not None, f"Unknown trainset game: {config.trainset_game}. " \

    logger.info(f"Trainset Game: {config.trainset_game}")

    if config.use_wandb:
        wandb.init(
            id=config.wandb_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.exp_name,
            config=dict(config),
        )
        wandb.define_metric("epoch")

    if config.draw_tsne:
        tsne_queue_in  = Queue(maxsize=8)
        tsne_queue_out = Queue(maxsize=8)
        tsne_process = Process(
        target=tsne_worker,
            args=(tsne_queue_in, tsne_queue_out),
            daemon=True
        )
        tsne_process.start()
        
        train(config, tsne_queue_in, tsne_queue_out)

    else:
        train(config)
    
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()