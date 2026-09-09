"""Seeding and determinism helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional for data-only use
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Determinism limitation: some 3D conv kernels have no deterministic
        # implementation, so cuDNN benchmark is disabled instead of forcing
        # torch.use_deterministic_algorithms, which would raise at runtime.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
