"""Helpers shared by the autoencoder and VAE training loops.

Reproducibility seeding and the detached-state-dict snapshot are identical for
both trainers; they live here so neither trainer imports the other.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Skips ``torch.use_deterministic_algorithms(True)`` because seeding
    alone is enough for run-to-run reproducibility on a fixed machine,
    and the deterministic flag forces slower kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU clone of ``model.state_dict()``; survives further training."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
