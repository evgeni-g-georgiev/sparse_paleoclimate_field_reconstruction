"""Training loop for the beta-VAE.

Trains a :class:`~paleoreco.models.autoencoder.ConvBetaVAE` with AdamW +
CosineAnnealingLR, optimising the ELBO (masked-MSE reconstruction +
``beta`` * KL to N(0, I)). ``beta`` is ramped linearly from 0 to its
target over the first ``kl_warmup_epochs`` epochs so the encoder can
develop an informative latent before the KL pressure kicks in.

Two modes:

* **With** ``val_loader``: per-epoch val metrics, best-val checkpointing
  on ``val_total_loss``, early stopping after ``patience`` epochs without
  improvement. Total loss is the monitor because the recon term
  advantages low-beta runs by construction. Best-tracking and the
  patience clock both skip the warmup window so the monitor is always
  evaluated at target beta.
* **Without** ``val_loader``: fixed-length training; ``best_state_dict``
  is the final-epoch state.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from paleoreco.losses import vae_elbo_loss
from paleoreco.train_ae import _snapshot_state_dict, set_seed


# ---------------------------------------------------------------------------
# Beta-warmup schedule.
# ---------------------------------------------------------------------------
def beta_at_epoch(
    epoch_idx: int, target_beta: float, kl_warmup_epochs: int
) -> float:
    """Linear ramp 0 -> ``target_beta`` over ``[0, kl_warmup_epochs)``, then constant.

    Epoch 0 gets ``beta_eff = 0`` (pure recon); epoch ``kl_warmup_epochs``
    and beyond get the full target. ``kl_warmup_epochs <= 0`` disables warmup.
    """
    if kl_warmup_epochs <= 0 or epoch_idx >= kl_warmup_epochs:
        return float(target_beta)
    return float(target_beta) * (epoch_idx / float(kl_warmup_epochs))


# ---------------------------------------------------------------------------
# Per-epoch helpers.
# ---------------------------------------------------------------------------
def _empty_metrics() -> dict[str, float | None]:
    nan = float("nan")
    return {
        "mse_z": nan, "rmse_z": nan, "rmse_celsius": None,
        "kl": nan, "total_loss": nan,
    }


def _train_one_epoch_vae(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mask: torch.Tensor,
    beta_eff: float,
    device: str | torch.device,
    zscore_std: torch.Tensor | None = None,
) -> dict[str, float | None]:
    """Run one training epoch at fixed ``beta_eff``.

    Returns ``{mse_z, rmse_z, rmse_celsius, kl, total_loss}``. Same
    in-progress-weights caveat as ``train_ae._train_one_epoch``: train
    metrics aggregate the optimisation trajectory; val metrics
    (evaluated with eval-mode forward) are one step ahead.
    """
    model.train()

    has_std = zscore_std is not None
    if has_std:
        std_sq = zscore_std.to(device).unsqueeze(0) ** 2

    sum_sq_z = 0.0
    sum_sq_c = 0.0
    sum_kl = 0.0
    sum_total = 0.0
    n_samples = 0
    n_terms = 0.0
    mask = mask.to(device)
    terms_per_sample = float(2 * mask.sum().item())

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch[:, :2]
        x_hat, mu, logvar, _ = model(batch)
        loss, _, kl = vae_elbo_loss(x_hat, target, mu, logvar, mask, beta_eff)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sq_err_z = (x_hat - target) ** 2 * mask
            sum_sq_z += sq_err_z.sum().item()
            if has_std:
                sum_sq_c += (sq_err_z * std_sq).sum().item()
            B = batch.shape[0]
            sum_kl += kl.item() * B
            sum_total += loss.item() * B
            n_samples += B
            n_terms += B * terms_per_sample

    if n_terms <= 0 or n_samples == 0:
        return _empty_metrics()
    mse_z = sum_sq_z / n_terms
    return {
        "mse_z":        mse_z,
        "rmse_z":       math.sqrt(mse_z),
        "rmse_celsius": math.sqrt(sum_sq_c / n_terms) if has_std else None,
        "kl":           sum_kl / n_samples,
        "total_loss":   sum_total / n_samples,
    }


@torch.no_grad()
def evaluate_vae(
    model: nn.Module,
    loader: DataLoader,
    mask: torch.Tensor,
    beta_eff: float,
    device: str | torch.device,
    zscore_std: torch.Tensor | None = None,
) -> dict[str, float | None]:
    """Recon metrics + KL + total loss over ``loader``.

    Model is set to eval mode internally; the VAE's reparameterise()
    returns the posterior mean in that mode, so the recon metrics are
    deterministic. ``beta_eff`` is the coefficient used to compute the
    total loss reported here.
    """
    model.eval()
    has_std = zscore_std is not None
    if has_std:
        std_sq = zscore_std.to(device).unsqueeze(0) ** 2

    sum_sq_z = 0.0
    sum_sq_c = 0.0
    sum_kl = 0.0
    sum_total = 0.0
    n_samples = 0
    n_terms = 0.0
    mask = mask.to(device)
    terms_per_sample = float(2 * mask.sum().item())

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch[:, :2]
        x_hat, mu, logvar, _ = model(batch)
        loss, _, kl = vae_elbo_loss(x_hat, target, mu, logvar, mask, beta_eff)
        sq_err_z = (x_hat - target) ** 2 * mask
        sum_sq_z += sq_err_z.sum().item()
        if has_std:
            sum_sq_c += (sq_err_z * std_sq).sum().item()
        B = batch.shape[0]
        sum_kl += kl.item() * B
        sum_total += loss.item() * B
        n_samples += B
        n_terms += B * terms_per_sample

    if n_terms <= 0 or n_samples == 0:
        return _empty_metrics()
    mse_z = sum_sq_z / n_terms
    return {
        "mse_z":        mse_z,
        "rmse_z":       math.sqrt(mse_z),
        "rmse_celsius": math.sqrt(sum_sq_c / n_terms) if has_std else None,
        "kl":           sum_kl / n_samples,
        "total_loss":   sum_total / n_samples,
    }


# ---------------------------------------------------------------------------
# Checkpoint + log helpers.
# ---------------------------------------------------------------------------
def _save_checkpoint(
    path: str,
    epoch: int,
    state: dict[str, torch.Tensor],
    val_total_loss: float | None,
) -> None:
    """Write ``{"epoch", "state_dict", "val_total_loss"}`` to ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"epoch": epoch, "state_dict": state, "val_total_loss": val_total_loss},
        path,
    )


def _format_epoch_line_vae(
    epoch: int,
    train_metrics: dict[str, float | None],
    val_metrics: dict[str, float | None] | None,
    *,
    lr: float,
    beta_eff: float,
    s_ep: float,
    improved: bool,
    has_val: bool,
) -> str:
    """One-line per-epoch summary; ``*`` marks val-total-loss improvements."""
    tail = f"beta={beta_eff:.2e}  lr={lr:.2e}  ({s_ep:.1f}s)"
    if has_val:
        assert val_metrics is not None
        star = " *" if improved else "  "
        return (
            f"epoch {epoch:3d}{star} "
            f"train_total={train_metrics['total_loss']:.4f}  "
            f"val_total={val_metrics['total_loss']:.4f}  "
            f"val_mse_z={val_metrics['mse_z']:.4f}  "
            f"val_kl={val_metrics['kl']:.3f}  " + tail
        )
    return (
        f"epoch {epoch:3d}   "
        f"train_total={train_metrics['total_loss']:.4f}  "
        f"train_mse_z={train_metrics['mse_z']:.4f}  "
        f"train_kl={train_metrics['kl']:.3f}  " + tail
    )


# ---------------------------------------------------------------------------
# Main training entry point.
# ---------------------------------------------------------------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    mask: torch.Tensor | np.ndarray | None = None,
    *,
    beta: float,
    kl_warmup_epochs: int = 30,
    zscore_std: torch.Tensor | np.ndarray | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 250,
    patience: int | None = 25,
    device: str | torch.device = "cpu",
    checkpoint_path: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 1,
    progress: bool = True,
    epoch_callback: Callable[[int, nn.Module], None] | None = None,
) -> dict[str, Any]:
    """Train a beta-VAE with AdamW + cosine LR + KL warmup; early-stops on
    ``val_total_loss``.

    Parameters mostly mirror :func:`paleoreco.train_ae.train`. New / changed:

    beta : float
        Target KL coefficient. ``beta`` is ramped linearly 0 -> ``beta``
        over the first ``kl_warmup_epochs`` epochs, then constant.
    kl_warmup_epochs : int
        Warmup length in epochs. 0 disables warmup. Must be strictly
        less than ``max_epochs``.

    Returns
    -------
    dict with the same shape as ``train_ae.train`` plus history keys
    ``train_kl``, ``val_kl``, ``train_total_loss``, ``val_total_loss``,
    ``beta_effective_per_epoch``. ``best_val_loss`` here is the lowest
    seen ``val_total_loss`` (not ``val_mse_z``).
    """
    if kl_warmup_epochs >= max_epochs:
        raise ValueError(
            f"kl_warmup_epochs={kl_warmup_epochs} must be < max_epochs="
            f"{max_epochs} so at least one epoch reaches target beta."
        )
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    if mask is None:
        raise ValueError("mask is required (the safe_valid mask used in the loss).")

    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
    std_t: torch.Tensor | None = None
    if zscore_std is not None:
        std_t = torch.as_tensor(zscore_std, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    has_val = val_loader is not None
    history: dict[str, list] = {
        "train_mse_z": [],
        "train_rmse_z": [],
        "train_rmse_celsius": [],
        "train_kl": [],
        "train_total_loss": [],
        "beta_effective_per_epoch": [],
        "lr": [],
        "epoch_seconds": [],
    }
    if has_val:
        history["val_mse_z"] = []
        history["val_rmse_z"] = []
        history["val_rmse_celsius"] = []
        history["val_kl"] = []
        history["val_total_loss"] = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0
    stopped_early = False
    early_stop_enabled = has_val and patience is not None

    epoch_iter: Any = range(max_epochs)
    pbar: tqdm | None = None
    if progress:
        latent = getattr(model, "latent_dim", "?")
        pbar = tqdm(
            range(max_epochs),
            desc=f"train_vae(d={latent},beta={beta:.0e})",
            unit="ep", leave=True, dynamic_ncols=True,
        )
        epoch_iter = pbar
    log = (lambda msg: tqdm.write(msg)) if progress else print

    for epoch in epoch_iter:
        t0 = time.perf_counter()
        beta_eff = beta_at_epoch(epoch, beta, kl_warmup_epochs)

        train_metrics = _train_one_epoch_vae(
            model, train_loader, optimizer, mask_t, beta_eff, device, std_t,
        )
        val_metrics = (
            evaluate_vae(model, val_loader, mask_t, beta_eff, device, std_t)
            if has_val
            else None
        )

        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["train_mse_z"].append(train_metrics["mse_z"])
        history["train_rmse_z"].append(train_metrics["rmse_z"])
        history["train_rmse_celsius"].append(train_metrics["rmse_celsius"])
        history["train_kl"].append(train_metrics["kl"])
        history["train_total_loss"].append(train_metrics["total_loss"])
        history["beta_effective_per_epoch"].append(beta_eff)
        if has_val:
            assert val_metrics is not None
            history["val_mse_z"].append(val_metrics["mse_z"])
            history["val_rmse_z"].append(val_metrics["rmse_z"])
            history["val_rmse_celsius"].append(val_metrics["rmse_celsius"])
            history["val_kl"].append(val_metrics["kl"])
            history["val_total_loss"].append(val_metrics["total_loss"])
        history["epoch_seconds"].append(time.perf_counter() - t0)

        # Skip best-tracking and patience during warmup so the monitor
        # comparison is always at target beta.
        in_warmup = epoch < kl_warmup_epochs
        improved = False
        if has_val and not in_warmup:
            assert val_metrics is not None
            val_loss = val_metrics["total_loss"]
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = _snapshot_state_dict(model)
                epochs_since_improve = 0
                if checkpoint_path is not None:
                    _save_checkpoint(checkpoint_path, epoch, best_state, val_loss)
            else:
                epochs_since_improve += 1

        if pbar is not None:
            postfix = (
                {
                    "val_total": f"{val_metrics['total_loss']:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "beta": f"{beta_eff:.0e}",
                    "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                }
                if has_val
                else {
                    "train_total": f"{train_metrics['total_loss']:.4f}",
                    "beta": f"{beta_eff:.0e}",
                    "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                }
            )
            pbar.set_postfix(postfix, refresh=False)

        if verbose and (epoch % log_every == 0 or improved):
            log(_format_epoch_line_vae(
                epoch, train_metrics, val_metrics,
                lr=history["lr"][-1],
                beta_eff=beta_eff,
                s_ep=history["epoch_seconds"][-1],
                improved=improved,
                has_val=has_val,
            ))

        if epoch_callback is not None:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    epoch_callback(epoch, model)
            finally:
                if was_training:
                    model.train()

        if early_stop_enabled and epochs_since_improve >= patience:
            stopped_early = True
            if verbose:
                log(
                    f"Early stopping at epoch {epoch} - no improvement "
                    f"in {patience} epochs (best val_total_loss="
                    f"{best_val_loss:.4f} at epoch {best_epoch})."
                )
            if pbar is not None:
                pbar.close()
            break

    if pbar is not None and not stopped_early:
        pbar.close()

    if not has_val:
        last_epoch = len(history["train_total_loss"]) - 1
        best_state = _snapshot_state_dict(model)
        best_epoch = last_epoch
        best_val_loss = float("nan")
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, last_epoch, best_state, None)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "stopped_early": stopped_early,
        "epochs_trained": len(history["train_total_loss"]),
    }
