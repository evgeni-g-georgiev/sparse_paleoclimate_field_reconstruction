"""Smoke tests for the EDM denoiser training loop."""

from __future__ import annotations

import numpy as np
import pytest
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


def _split(cube, valid, n_val=4):
    """Normalised cube split into (train, val, mask); the fixture has 12 states."""
    cube_norm, safe = _cube_norm(cube, valid)
    return cube_norm[n_val:], cube_norm[:n_val], safe


def test_fixed_grid_loss_is_deterministic(cube, valid):
    # The property early stopping depends on: the same weights must score identically
    # twice, which the random-sigma training loss does not.
    train, val, safe = _split(cube, valid)
    model = _model()
    mask = torch.as_tensor(safe, dtype=torch.float32)
    x = torch.as_tensor(val)
    sigmas = td.noise_level_grid(8, torch.device("cpu"))
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    a = td.fixed_grid_loss(model, x, mask, sigmas, noise)
    b = td.fixed_grid_loss(model, x, mask, sigmas, noise)
    assert a == b


def test_ema_debias_is_exact_after_one_update():
    # Undebiased, the first shadow would be (1 - decay) * weights, i.e. mostly the
    # zero seed; state() must divide that back out.
    model = _model()
    ema = td._EMA(model, 0.99)
    ema.update(model)
    state, live = ema.state(), model.state_dict()
    for k, v in live.items():
        if v.dtype.is_floating_point:
            assert torch.allclose(state[k], v, atol=1e-6), k


def _frozen(cube, valid, **kw):
    """Train with the weights pinned, so the held-out score cannot improve after epoch 0.

    ``lr=0`` freezes the live weights and ``ema_decay=None`` scores them directly, so
    the deterministic held-out loss repeats exactly and the patience counter is
    exercised on a known sequence rather than on whatever the data happens to do.
    """
    train, val, safe = _split(cube, valid)
    return td.train(_model(), train, safe, val_cube=val, lr=0.0, ema_decay=None,
                    batch_size=8, seed=0, verbose=False, progress=False, **kw)


def test_early_stopping_fires_and_reports_epochs(cube, valid):
    out = _frozen(cube, valid, patience=1, max_epochs=30)
    assert out["stopped_early"]
    assert out["stop_epoch"] < 30
    assert out["stop_epoch"] == len(out["history"]["train_loss"])
    assert out["stop_epoch"] == len(out["history"]["val_loss"])


def test_stop_epoch_is_best_plus_patience(cube, valid):
    # The convention the notebook's refit depends on: best_epoch indexes the history,
    # stop_epoch counts completed epochs, so a refit takes max_epochs=best_epoch + 1.
    patience = 3
    out = _frozen(cube, valid, patience=patience, max_epochs=40)
    assert out["stopped_early"]
    assert out["best_epoch"] == 0
    assert out["stop_epoch"] == out["best_epoch"] + 1 + patience


def test_patience_none_runs_to_max_epochs(cube, valid):
    out = _frozen(cube, valid, max_epochs=6)
    assert not out["stopped_early"]
    assert out["stop_epoch"] == out["epochs_trained"] == 6


def test_val_history_lengths_and_probe_cadence(cube, valid):
    train, val, safe = _split(cube, valid)
    out = td.train(_model(), train, safe, val_cube=val, max_epochs=12, probe_every=5,
                   batch_size=8, lr=1e-3, seed=0, verbose=False, progress=False)
    h = out["history"]
    assert len(h["val_loss"]) == len(h["val_loss_live"]) == len(h["train_loss"]) == 12
    assert h["train_probe_epoch"] == [0, 5, 10]
    assert len(h["train_probe_loss"]) == 3


def test_val_split_cannot_write_a_checkpoint(tmp_path, cube, valid):
    train, val, safe = _split(cube, valid)
    with pytest.raises(ValueError):
        td.train(_model(), train, safe, val_cube=val, max_epochs=2,
                 checkpoint_path=str(tmp_path / "x.pt"), verbose=False, progress=False)


def test_checkpoint_roundtrips_config_and_weights(tmp_path, cube, valid):
    cube_norm, safe = _cube_norm(cube, valid)
    path = str(tmp_path / "diffusion.pt")
    scales = np.array([2.0, 1.0])
    td.train(_model(), cube_norm, safe, max_epochs=3, batch_size=8, ema_decay=None,
             checkpoint_path=path, channel_scales=scales, seed=0,
             checkpoint_extra={"selection": {"best_epoch": 2}},
             verbose=False, progress=False)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["config"]["grid_shape"] == (12, 12)
    assert np.allclose(ckpt["channel_scales"], scales)
    # Provenance must survive the weights_only load the sampler uses.
    assert ckpt["selection"]["best_epoch"] == 2
    reloaded = EDMDenoiser(CircularUNet(**ckpt["config"]), ckpt["sigma_data"])
    reloaded.load_state_dict(ckpt["state_dict"])   # keys match the saved denoiser
