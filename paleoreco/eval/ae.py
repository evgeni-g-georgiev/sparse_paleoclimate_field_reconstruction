"""Autoencoder-specific evaluation helpers.

Two functions live here because they have a hard dependency on
something AE-shaped:

* :func:`plot_loss_curves` reads the ``history`` dict shape produced
  by :mod:`paleoreco.training.trainer_ae`.
* :func:`reconstruct_split` assumes the model's forward returns
  ``(x_hat, z)`` (the AE contract).

Everything else under :mod:`paleoreco.eval` is generic and lives in
``shared``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Loss curves.
# ---------------------------------------------------------------------------
def plot_loss_curves(
    history: dict,
    best_epoch: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel loss curves: masked MSE (the optimisation target) on the
    left, RMSE on the right, both in °C anomaly units.

    Both panels overlay train (solid) and, when present, val (dashed)
    curves. If ``best_epoch`` is given, a vertical line marks it on both
    panels.
    """
    epochs = np.arange(len(history["train_mse"]))
    has_val = (
        "val_mse" in history
        and history["val_mse"] is not None
        and len(history["val_mse"]) > 0
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), squeeze=False, constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(epochs, history["train_mse"], label="train", lw=1.5)
    if has_val:
        ax.plot(epochs, history["val_mse"], label="val", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE (°C² anomaly)")
    ax.set_title("Loss curves: MSE (optimisation target)")
    ax.grid(True, alpha=0.3)
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="k", lw=1, alpha=0.4, label=f"best ep {best_epoch}")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, history["train_rmse"], label="train", lw=1.5)
    if has_val:
        ax.plot(epochs, history["val_rmse"], label="val", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE (°C anomaly)")
    ax.set_title("Loss curves: RMSE")
    ax.grid(True, alpha=0.3)
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="k", lw=1, alpha=0.4)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Forward-pass helper.
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct_split(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a ConvAE-shaped ``model`` on every sample of ``dataset``.

    Returns ``(truth, pred)`` numpy arrays of shape ``(N, 2, H, W)``
    in °C anomaly units, with ``N == len(dataset)``. Order matches the
    dataset's age-index order (``shuffle=False`` internally).

    Assumes the model's forward returns ``(x_hat, z)`` (the
    :class:`paleoreco.models.autoencoder.ConvAE` contract).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    truths, preds = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        x_hat, _ = model(batch)
        truths.append(batch[:, :2].cpu().numpy())
        preds.append(x_hat.cpu().numpy())
    return np.concatenate(truths, axis=0), np.concatenate(preds, axis=0)
