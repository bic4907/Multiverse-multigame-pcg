import torch

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    extra: Dict[str, Any] | None = None