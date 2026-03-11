from typing import Optional

import torch

from .step import StepOutput


class BaseTrainer:
    def __init__(self, prefix: Optional[str]):
        self.prefix = f'{prefix}' if prefix else ''

    def train_step(self, batch, epoch: int) -> StepOutput:
        raise NotImplementedError

    @torch.no_grad()
    def val_step(self, batch, epoch: int) -> StepOutput:
        raise NotImplementedError

    def on_epoch_start(self, epoch: int):
        pass

    def on_epoch_end(self, epoch: int, logs: dict):
        pass
