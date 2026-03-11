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
    Uploads a CSV file to wandb (also automatically logs as a Table)

    Args:
        artifact_name: wandb artifact name
        save_path: CSV file path
        artifact_type: artifact type (default: "evaluation_results")
    """
    if not wandb.run:
        logger.warning("wandb run is not active. Skipping upload.")
        return

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"File not found: {save_path}")

    # If CSV, log as wandb.Table
    if save_path.endswith('.csv'):
        df = pd.read_csv(save_path)
        table = wandb.Table(dataframe=df)
        wandb.log({f"table/{artifact_name}": table})
        logger.info(f"Logged CSV as wandb.Table: table/{artifact_name}")

    # Upload as Artifact
    wandb.save(save_path)
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type
    )
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    logger.info(f"Uploaded artifact: {artifact_name} ({save_path})")
