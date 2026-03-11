import os
import wandb
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__file__)

def upload_to_wandb(
    artifact_name: str,
    save_path: str,
    artifact_type: str = "evaluation_results",
):
    """
    CSV 파일을 wandb에 업로드하는 함수 (자동으로 Table로도 로깅)

    Args:
        artifact_name: wandb artifact 이름
        save_path: CSV 파일 경로
        artifact_type: artifact 타입 (기본값: "evaluation_results")
    """
    if not wandb.run:
        logger.warning("wandb run is not active. Skipping upload.")
        return

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File not found: {save_path}")

    # CSV 파일이면 wandb.Table로 로깅
    if save_path.endswith('.csv'):
        df = pd.read_csv(save_path)
        table = wandb.Table(dataframe=df)
        wandb.log({f"table/{artifact_name}": table})
        logger.info(f"Logged CSV as wandb.Table: table/{artifact_name}")

    # Artifact로 업로드
    wandb.save(save_path)
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type
    )
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    logger.info(f"Uploaded artifact: {artifact_name} ({save_path})")
