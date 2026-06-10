from __future__ import annotations

import os
import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and Stable-Baselines3 helpers."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def configure_torch_threads(num_threads: int | None) -> None:
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(num_threads)

