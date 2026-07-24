"""Smoke tests for the EDM denoiser training loop."""

from __future__ import annotations

import numpy as np
import torch

from paleoreco.data.cube import apply_anomaly, compute_zscore_stats
from paleoreco.models.diffusion import CircularUNet, EDMDenoiser
from paleoreco.training import trainer_diffusion as td
from paleoreco.training._common import set_seed


def _cube_norm(cube, valid):
    stats = compute_zscore_stats(cube, np.arange(len(cube)), valid)
    return apply_anomaly(cube, stats), stats["safe_valid"]


def _model():
    set_seed(0)   # deterministic init so two builds start from identical weights
    return EDMDenoiser(CircularUNet(base_channels=16, depth=2, grid_shape=(12, 12)))


def test_train_reduces_loss(cube, valid):
    cube_norm, safe = _cube_norm(cube, valid)
    out = td.train(_model(), cube_norm, safe, max_epochs=15, batch_size=8, lr=1e-3,
                   ema_decay=0.99, seed=0, verbose=False, progress=False)
    loss = out["history"]["train_loss"]
    assert loss[-1] < loss[0]
    assert out["epochs_trained"] == 15


def test_training_is_deterministic_under_fixed_seed(cube, valid):
    cube_norm, safe = _cube_norm(cube, valid)
    kw = dict(max_epochs=6, batch_size=8, lr=1e-3, ema_decay=None, seed=7,
              verbose=False, progress=False)
    a = td.train(_model(), cube_norm, safe, **kw)["history"]["train_loss"]
    b = td.train(_model(), cube_norm, safe, **kw)["history"]["train_loss"]
    assert np.allclose(a, b)


def test_checkpoint_roundtrips_config_and_weights(tmp_path, cube, valid):
    cube_norm, safe = _cube_norm(cube, valid)
    path = str(tmp_path / "diffusion.pt")
    scales = np.array([2.0, 1.0])
    td.train(_model(), cube_norm, safe, max_epochs=3, batch_size=8, ema_decay=None,
             checkpoint_path=path, channel_scales=scales, seed=0,
             verbose=False, progress=False)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["config"]["grid_shape"] == (12, 12)
    assert np.allclose(ckpt["channel_scales"], scales)
    reloaded = EDMDenoiser(CircularUNet(**ckpt["config"]), ckpt["sigma_data"])
    reloaded.load_state_dict(ckpt["state_dict"])   # keys match the saved denoiser
