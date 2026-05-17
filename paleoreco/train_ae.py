"""Training loop for the autoencoder.

Trains a :class:`~paleoreco.models.autoencoder.ConvAE` with AdamW +
CosineAnnealingLR, optimising masked-MSE in z-score units. Two modes:

* **With** ``val_loader``: per-epoch val metrics, best-val checkpointing,
  early stopping after ``patience`` epochs without improvement.
* **Without** ``val_loader``: fixed-length training with no model
  selection; ``best_state_dict`` is the final-epoch state.

The masked-MSE loss lives in z-score units (the model's working scale);
``evaluate`` additionally reports RMSE in °C when ``zscore_std`` is
given.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from paleoreco.losses import masked_mse


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Per-epoch helpers.
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mask: torch.Tensor,
    device: str | torch.device,
) -> float:
    """Run one training epoch. Returns mean masked-MSE (z-score units).

    The returned value uses in-progress weights so it lags the post-epoch
    evaluation by half a step. Diagnostic only; ``evaluate`` produces the
    authoritative train metric.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch[:, :2]                       # mtco_z, mtwa_z
        x_hat, _ = model(batch)
        loss = masked_mse(x_hat, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mask: torch.Tensor,
    device: str | torch.device,
    zscore_std: torch.Tensor | None = None,
) -> dict[str, float | None]:
    """Compute per-valid-cell MSE / RMSE on a loader.

    Parameters
    ----------
    model : nn.Module
        Set to ``eval`` mode internally.
    loader : DataLoader
        Batches are ``(B, 3, H, W)`` = ``(mtco_z, mtwa_z, valid_mask)``;
        target is the first two channels.
    mask : torch.Tensor of shape (H, W) or broadcastable
        ``safe_valid`` mask used to weight the loss.
    device : str or torch.device
    zscore_std : torch.Tensor or None, default None
        Per-cell std ``(2, H, W)``. If given, the returned dict's
        ``rmse_celsius`` is populated; otherwise it is ``None``.

    Returns
    -------
    dict with keys ``mse_z``, ``rmse_z``, ``rmse_celsius``.
    """
    model.eval()

    has_std = zscore_std is not None
    if has_std:
        # (1, 2, H, W) so it broadcasts against (B, 2, H, W) squared errors.
        std_sq = zscore_std.to(device).unsqueeze(0) ** 2

    sum_sq_z = 0.0
    sum_sq_c = 0.0
    n_terms = 0.0

    mask = mask.to(device)
    # 2 channels per sample, mask shared: each sample contributes
    # 2 * mask.sum() terms to the masked-MSE denominator.
    terms_per_sample = float(2 * mask.sum().item())

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch[:, :2]
        x_hat, _ = model(batch)

        # sq_err_z: (B, 2, H, W), already masked.
        sq_err_z = (x_hat - target) ** 2 * mask
        sum_sq_z += sq_err_z.sum().item()
        if has_std:
            sum_sq_c += (sq_err_z * std_sq).sum().item()
        n_terms += batch.shape[0] * terms_per_sample

    if n_terms <= 0:
        return {"mse_z": float("nan"), "rmse_z": float("nan"), "rmse_celsius": None}

    mse_z = sum_sq_z / n_terms
    out: dict[str, float | None] = {
        "mse_z": mse_z,
        "rmse_z": math.sqrt(mse_z),
        "rmse_celsius": math.sqrt(sum_sq_c / n_terms) if has_std else None,
    }
    return out


# ---------------------------------------------------------------------------
# Main training entry point.
# ---------------------------------------------------------------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    mask: torch.Tensor | np.ndarray | None = None,
    *,
    zscore_std: torch.Tensor | np.ndarray | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int | None = 30,
    device: str | torch.device = "cpu",
    checkpoint_path: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 1,
    progress: bool = True,
    epoch_callback: Callable[[int, nn.Module], None] | None = None,
) -> dict[str, Any]:
    """Train ``model`` with AdamW + cosine LR, optional val/early stopping.

    See module docstring for the two-modes summary.

    Parameters
    ----------
    model : nn.Module
        Must follow the AE forward contract: takes ``(B, 3, H, W)``,
        returns ``(x_hat, z)`` with ``x_hat`` of shape ``(B, 2, H, W)``.
    train_loader : DataLoader
    val_loader : DataLoader or None, default None
        ``None`` disables validation and early stopping.
    mask : (H, W) array
        ``safe_valid`` mask used by the loss and by ``evaluate``.
    zscore_std : (2, H, W) array, optional
        Per-cell std; if given, history includes ``rmse_celsius``.
    lr, weight_decay : float
        AdamW hyperparameters.
    max_epochs : int
        Upper bound on training; cosine LR schedules over this length.
    patience : int or None, default 30
        Epochs of no val-loss improvement before early stopping.
        ``None`` disables; ignored when ``val_loader`` is ``None``.
    device : str or torch.device
    checkpoint_path : str, optional
        With val: best-val state written here on each new best.
        Without val: final state written once at end.
    seed : int
    verbose, log_every : bool, int
        Per-epoch text logging cadence.
    progress : bool
        tqdm progress bar; independent of ``verbose`` so sweep loops can
        hide per-epoch text but keep a per-config bar.
    epoch_callback : Callable[[int, nn.Module], None] or None
        Hook called at end of every epoch with ``(epoch_index, model)``.
        Model is in ``eval`` mode under ``no_grad`` for the call and
        restored to ``train`` mode immediately after.

    Returns
    -------
    dict with keys:
        ``history``         : dict of per-epoch lists (always
            ``train_mse_z``, ``train_rmse_z``, ``train_rmse_celsius``,
            ``lr``, ``epoch_seconds``; with val also the ``val_*`` keys).
        ``best_val_loss``   : float or ``NaN`` (no val).
        ``best_epoch``      : int.
        ``best_state_dict`` : OrderedDict of CPU tensors. Best-val epoch
            with validation; final-epoch without.
        ``stopped_early``   : bool.
        ``epochs_trained``  : int.
    """
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

    # Val keys are only present when a val loader is given, so plot code
    # doesn't render misleading flat-line "val" curves.
    has_val = val_loader is not None
    history: dict[str, list] = {
        "train_mse_z": [],
        "train_rmse_z": [],
        "train_rmse_celsius": [],
        "lr": [],
        "epoch_seconds": [],
    }
    if has_val:
        history["val_mse_z"] = []
        history["val_rmse_z"] = []
        history["val_rmse_celsius"] = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0
    stopped_early = False
    early_stop_enabled = has_val and patience is not None

    # Use tqdm.write() so verbose prints don't clobber the progress bar;
    # fall back to regular print() if the bar is disabled.
    epoch_iter: Any = range(max_epochs)
    pbar: tqdm | None = None
    if progress:
        latent = getattr(model, "latent_dim", "?")
        pbar = tqdm(
            range(max_epochs), desc=f"train(d={latent})",
            unit="ep", leave=True, dynamic_ncols=True,
        )
        epoch_iter = pbar
    log = (lambda msg: tqdm.write(msg)) if progress else print

    for epoch in epoch_iter:
        t0 = time.perf_counter()

        _train_one_epoch(model, train_loader, optimizer, mask_t, device)
        train_metrics = evaluate(model, train_loader, mask_t, device, std_t)
        val_metrics = (
            evaluate(model, val_loader, mask_t, device, std_t)
            if has_val
            else None
        )

        # Log the LR before stepping, so the logged value matches what
        # the just-completed epoch actually trained with.
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["train_mse_z"].append(train_metrics["mse_z"])
        history["train_rmse_z"].append(train_metrics["rmse_z"])
        history["train_rmse_celsius"].append(train_metrics["rmse_celsius"])
        if has_val:
            assert val_metrics is not None
            history["val_mse_z"].append(val_metrics["mse_z"])
            history["val_rmse_z"].append(val_metrics["rmse_z"])
            history["val_rmse_celsius"].append(val_metrics["rmse_celsius"])
        history["epoch_seconds"].append(time.perf_counter() - t0)

        # Best-tracking only runs with a val loader; in the val-less
        # mode the "best" is set to the final-epoch state after the loop.
        improved = False
        if has_val:
            assert val_metrics is not None
            val_loss = val_metrics["mse_z"]
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_since_improve = 0
                if checkpoint_path is not None:
                    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": best_state,
                            "val_mse_z": val_loss,
                        },
                        checkpoint_path,
                    )
            else:
                epochs_since_improve += 1

        # Live progress-bar postfix: prefer val_mse_z when available,
        # fall back to train_mse_z + best (NaN displays sensibly).
        if pbar is not None:
            if has_val:
                pbar.set_postfix(
                    {
                        "val_mse_z": f"{val_metrics['mse_z']:.4f}",
                        "best": f"{best_val_loss:.4f}",
                        "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                    },
                    refresh=False,
                )
            else:
                pbar.set_postfix(
                    {
                        "train_mse_z": f"{train_metrics['mse_z']:.4f}",
                        "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                    },
                    refresh=False,
                )

        if verbose and (epoch % log_every == 0 or improved):
            c_train = (
                f"  train_rmse_C={train_metrics['rmse_celsius']:.3f}"
                if train_metrics["rmse_celsius"] is not None
                else ""
            )
            if has_val:
                c_val = (
                    f"  val_rmse_C={val_metrics['rmse_celsius']:.3f}"
                    if val_metrics["rmse_celsius"] is not None
                    else ""
                )
                star = " *" if improved else "  "
                log(
                    f"epoch {epoch:3d}{star} "
                    f"train_mse_z={train_metrics['mse_z']:.4f}  "
                    f"val_mse_z={val_metrics['mse_z']:.4f}{c_val}  "
                    f"lr={history['lr'][-1]:.2e}  "
                    f"({history['epoch_seconds'][-1]:.1f}s)"
                )
            else:
                log(
                    f"epoch {epoch:3d}   "
                    f"train_mse_z={train_metrics['mse_z']:.4f}{c_train}  "
                    f"lr={history['lr'][-1]:.2e}  "
                    f"({history['epoch_seconds'][-1]:.1f}s)"
                )

        # Switch to eval mode under no_grad so the callback can call
        # model.encode(...) cleanly, then restore train mode for the
        # next epoch.
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
                    f"in {patience} epochs (best val_mse_z={best_val_loss:.4f} "
                    f"at epoch {best_epoch})."
                )
            if pbar is not None:
                pbar.close()
            break

    # Make sure the bar is closed even if we ran the full max_epochs.
    if pbar is not None and not stopped_early:
        pbar.close()

    # Val-less mode has no model-selection step, so "best" is the
    # final-epoch state and best_val_loss is NaN.
    if not has_val:
        last_epoch = len(history["train_mse_z"]) - 1
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        best_epoch = last_epoch
        best_val_loss = float("nan")
        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": last_epoch,
                    "state_dict": best_state,
                    "val_mse_z": None,
                },
                checkpoint_path,
            )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "stopped_early": stopped_early,
        "epochs_trained": len(history["train_mse_z"]),
    }
