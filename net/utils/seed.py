"""Seed every PRNG. Framework-agnostic: touches torch only if installed."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and (if installed) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)  # also seeds every CUDA device
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
