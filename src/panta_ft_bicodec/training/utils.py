import os

import numpy as np
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

def get_available_device() -> None:
    """ Get the available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int=42) -> None:
    """Sets the seed of the experiment for reproducibility."""
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # CPU
    torch.cuda.manual_seed(seed)          # GPU
    torch.cuda.manual_seed_all(seed)      # Multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scheduler_constructor(str_scheduler: str):
    match str_scheduler:
        case "StepLR":
            return torch.optim.lr_scheduler.StepLR
        case "CosineAnnealingLR":
            return CosineSchedulerWithInternalState
        case _:
            raise ValueError(f"Scheduler {str_scheduler} is not supported")


class RunningMetrics:
    """ Accumulate training metrics between two logs and return their means """

    def __init__(self) -> None:
        self.sums = {}
        self.counts = {}

    def add(self, metrics: dict) -> None:
        # Values stay on GPU until pop_means to avoid a sync at every step
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0) + value
            self.counts[name] = self.counts.get(name, 0) + 1

    def pop_means(self) -> dict:
        means = {name: float(self.sums[name] / self.counts[name]) for name in self.sums}
        self.sums = {}
        self.counts = {}
        return means


class CosineSchedulerWithInternalState(CosineLRScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inter_step = 0
    
    def step(self) -> None:
        self.inter_step += 1
        super().step(epoch=self.inter_step)