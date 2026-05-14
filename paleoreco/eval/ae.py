"""Autoencoder-specific evaluation helpers.

Two functions live here because they have a hard dependency on
something AE-shaped:

* :func:`plot_loss_curves` reads keys (``train_mse_z``, ``val_mse_z``,
  ``train_rmse_celsius``, ``val_rmse_celsius``) that come from the
  ``history`` dict produced by :mod:`paleoreco.train_ae`. A different
  training loop (e.g. diffusion model) produces a different dict shape.
* :func:`reconstruct_split` assumes the model's forward returns
  ``(x_hat, z)`` - the AE contract. Diffusion samplers / denoisers
  have a different signature.

Everything else under :mod:`paleoreco.eval` is generic and lives in
``shared``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Loss curves (artefact i).
# ---------------------------------------------------------------------------
def plot_loss_curves(
    history: dict,
    best_epoch: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel: z-score MSE on the left, °C RMSE on the right.

    Both panels overlay train (solid) and val (dashed) curves. If
    ``best_epoch`` is given, a vertical line marks it on both panels.

    The °C panel is omitted if ``train_rmse_celsius`` is missing from
    ``history`` (i.e. ``train_ae.train`` was called without
    ``zscore_std``).
    """
    epochs = np.arange(len(history["train_mse_z"]))
    has_celsius = (
        history.get("train_rmse_celsius") is not None
        and len(history["train_rmse_celsius"]) > 0
        and history["train_rmse_celsius"][0] is not None
    )
    n_panels = 2 if has_celsius else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels + 1, 4),
        squeeze=False, constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(epochs, history["train_mse_z"], label="train", lw=1.5)
    ax.plot(epochs, history["val_mse_z"], label="val", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE (z-score units)")
    ax.set_title("Loss curves — z-score units (optimisation target)")
    ax.grid(True, alpha=0.3)
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="k", lw=1, alpha=0.4, label=f"best ep {best_epoch}")
    ax.legend()

    if has_celsius:
        ax = axes[0, 1]
        ax.plot(epochs, history["train_rmse_celsius"], label="train", lw=1.5)
        ax.plot(epochs, history["val_rmse_celsius"], label="val", lw=1.5, ls="--")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE (°C)")
        ax.set_title("Loss curves — °C (human-readable)")
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

    Returns ``(truth_z, pred_z)`` numpy arrays of shape ``(N, 2, H, W)``
    in z-score units, with ``N == len(dataset)``. Order matches the
    dataset's age-index order (``shuffle=False`` internally).

    Assumes the model's forward returns ``(x_hat, z)`` - this is the
    contract :class:`paleoreco.models.autoencoder.ConvAE` follows. A
    model returning just ``x_hat`` won't unpack correctly.
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
