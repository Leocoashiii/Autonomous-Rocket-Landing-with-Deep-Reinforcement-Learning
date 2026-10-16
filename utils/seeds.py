"""Reproducibility helpers shared across training and evaluation."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and Torch in one place."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def episode_seed(base_seed: int, episode_index: int, offset: int = 0) -> int:
    """Return a deterministic per-episode seed."""

    return int(base_seed + offset + episode_index)
