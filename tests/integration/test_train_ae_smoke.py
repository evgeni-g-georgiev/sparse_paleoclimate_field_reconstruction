"""Smoke test of the AE training loop (paleoreco.train_ae + models.autoencoder).

A couple of epochs on a tiny synthetic field set with a fixed seed: the loop must
run, return the documented history keys, drive the loss down, and stay
deterministic. Guards the trainer wiring across the _common helper extraction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from paleoreco.models.autoencoder import ConvAE
from paleoreco.training.trainer_ae import set_seed, train


def _loader(seed=0):
    rng = np.random.default_rng(seed)
    H, W = 8, 8
    fields = rng.normal(size=(16, 2, H, W)).astype(np.float32)
    mask = np.ones((H, W), dtype=np.float32)
    samples = np.concatenate(
        [fields, np.broadcast_to(mask, (16, 1, H, W))], axis=1
    ).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(samples))
    # The trainer expects each batch item to be the (3, H, W) tensor itself.
    return DataLoader(_Unwrap(ds), batch_size=4, shuffle=True), mask


class _Unwrap(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


def _build():
    return ConvAE(latent_dim=4, base_channels=4, depth=2, grid_shape=(8, 8))


def test_train_runs_and_reports_history():
    set_seed(0)
    loader, mask = _loader()
    model = _build()
    out = train(model, loader, val_loader=None, mask=mask,
                max_epochs=3, patience=None, seed=0, verbose=False, progress=False)

    for key in ("history", "best_state_dict", "best_epoch", "epochs_trained"):
        assert key in out
    assert "train_mse" in out["history"]
    assert len(out["history"]["train_mse"]) == 3
    # Loss should not increase end-to-start over a short run.
    losses = out["history"]["train_mse"]
    assert losses[-1] <= losses[0] + 1e-6


def test_training_is_deterministic_under_fixed_seed():
    def run():
        set_seed(0)              # seed before weight init so both runs match
        loader, mask = _loader(seed=0)
        model = _build()
        out = train(model, loader, val_loader=None, mask=mask, max_epochs=2,
                    patience=None, seed=0, verbose=False, progress=False)
        return out["history"]["train_mse"]

    a = run()
    b = run()
    assert np.allclose(a, b)
