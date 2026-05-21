"""Beta-VAE specific evaluation helpers.

Three functions live here that depend on either the VAE forward contract
(``forward(x) -> (x_hat, mu, logvar, z)``) or on the ``history`` dict
produced by :mod:`paleoreco.train_vae`. Everything else under
:mod:`paleoreco.eval` is model-family-agnostic and lives in
:mod:`~paleoreco.eval.shared`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Forward-pass helper.
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct_split_vae(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a ConvBetaVAE-shaped ``model`` over every sample of ``dataset``.

    Returns ``(truth_z, pred_z, mu, logvar)`` numpy arrays in z-score units;
    ``truth_z`` and ``pred_z`` have shape ``(N, 2, H, W)``; ``mu`` and
    ``logvar`` have shape ``(N, latent_dim)``. Order matches the dataset's
    age-index order (``shuffle=False`` internally).

    Model is set to eval mode; the VAE's ``reparameterise()`` returns the
    posterior mean in that mode, so ``pred_z`` is the deterministic decode
    of ``mu``.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    truths, preds, mus, lvs = [], [], [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        x_hat, mu, logvar, _ = model(batch)
        truths.append(batch[:, :2].cpu().numpy())
        preds.append(x_hat.cpu().numpy())
        mus.append(mu.cpu().numpy())
        lvs.append(logvar.cpu().numpy())
    return (
        np.concatenate(truths, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(mus, axis=0),
        np.concatenate(lvs, axis=0),
    )


# ---------------------------------------------------------------------------
# Latent-distribution diagnostics.
# ---------------------------------------------------------------------------
def compute_vae_diagnostics(
    mu: np.ndarray, logvar: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Per-dim KL + aggregate KL + aggregate-posterior summary on a split.

    Parameters
    ----------
    mu, logvar : (N, d) float arrays
        Encoder outputs over a split (typically from
        :func:`reconstruct_split_vae`).

    Returns
    -------
    dict with keys:
        ``"kl_per_dim"`` : (d,) array. ``KL_k`` averaged over snapshots;
            a near-zero entry signals per-dim posterior collapse.
        ``"kl_total"`` : scalar. ``sum_k kl_per_dim``.
        ``"mu_norm"`` : scalar. ``||mean_n(mu)||_2``. Should be ~0 when
            the aggregate posterior is centred at the prior.
        ``"post_cov_diag_mean"`` : scalar. Mean over dims of the
            aggregate posterior variance per dim, ``Var_n(mu_k) +
            E_n(sigma_k^2)``. Should be ~1 when the aggregate posterior
            matches ``N(0, I)``.
    """
    kl_per_dim = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1.0).mean(axis=0)
    kl_total = float(kl_per_dim.sum())
    mu_norm = float(np.linalg.norm(mu.mean(axis=0)))
    post_var_per_dim = mu.var(axis=0, ddof=0) + np.exp(logvar).mean(axis=0)
    post_cov_diag_mean = float(post_var_per_dim.mean())
    return {
        "kl_per_dim": kl_per_dim,
        "kl_total": kl_total,
        "mu_norm": mu_norm,
        "post_cov_diag_mean": post_cov_diag_mean,
    }


# ---------------------------------------------------------------------------
# Loss curves.
# ---------------------------------------------------------------------------
def plot_loss_curves_vae(
    history: dict,
    best_epoch: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Loss-curve grid: total loss, recon MSE, KL, and the schedule.

    2x2 layout: total loss (the optimisation target, optional best-epoch
    marker), masked MSE in z-score units (recon-only quality), KL term
    in nats, and the beta-effective + LR schedule on twin axes. Val curves
    are overlaid where ``history`` carries the ``val_*`` keys.
    """
    epochs = np.arange(len(history["train_total_loss"]))
    has_val = (
        "val_total_loss" in history
        and history["val_total_loss"] is not None
        and len(history["val_total_loss"]) > 0
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    # Total loss.
    ax = axes[0, 0]
    ax.plot(epochs, history["train_total_loss"], label="train", lw=1.5)
    if has_val:
        ax.plot(epochs, history["val_total_loss"], label="val", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss (recon + beta * KL)")
    ax.set_title("Total loss (optimisation target)")
    ax.grid(True, alpha=0.3)
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="k", lw=1, alpha=0.4,
                   label=f"best ep {best_epoch}")
    ax.legend()

    # Recon MSE in z-score units.
    ax = axes[0, 1]
    ax.plot(epochs, history["train_mse_z"], label="train", lw=1.5)
    if has_val:
        ax.plot(epochs, history["val_mse_z"], label="val", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE (z-score units)")
    ax.set_title("Reconstruction MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # KL term in nats.
    ax = axes[1, 0]
    ax.plot(epochs, history["train_kl"], label="train", lw=1.5)
    if has_val:
        ax.plot(epochs, history["val_kl"], label="val", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL to N(0, I), nats")
    ax.set_title("KL term")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Beta + LR schedule on twin axes.
    ax = axes[1, 1]
    ax.plot(epochs, history["beta_effective_per_epoch"],
            color="C0", lw=1.5, label="beta")
    ax.set_xlabel("epoch")
    ax.set_ylabel("beta_effective", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title("Schedule")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(epochs, history["lr"], color="C3", lw=1.5, ls="--", label="lr")
    ax2.set_ylabel("learning rate", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig
