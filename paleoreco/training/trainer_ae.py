"""Training loop for the autoencoder.

Trains a :class:`~paleoreco.models.autoencoder.ConvAE` with AdamW +
CosineAnnealingLR, optimising masked-MSE in °C anomaly units. Two modes:

* **With** ``val_loader``: per-epoch val metrics, best-val checkpointing,
  early stopping after ``patience`` epochs without improvement.
* **Without** ``val_loader``: fixed-length training with no model
  selection; ``best_state_dict`` is the final-epoch state.

The masked-MSE loss lives in °C anomaly units (the model's working scale),
so ``rmse`` is already on the temperature scale.
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

from ._common import _snapshot_state_dict, set_seed
from .losses import masked_mse


# ---------------------------------------------------------------------------
# Per-epoch helpers.
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mask: torch.Tensor,
    device: str | torch.device,
) -> dict[str, float]:
    """Run one training epoch. Returns the same metrics dict as ``evaluate``.

    Metrics aggregate squared errors across the epoch's optimisation
    trajectory (in-progress weights), so train curves are not strictly
    apples-to-apples with val curves (which are measured on end-of-epoch
    weights via ``evaluate``). The asymmetry is invisible at typical
    training horizons and is the price of avoiding a second forward pass
    over the train loader per epoch.
    """
    model.train()

    sum_sq = 0.0
    n_terms = 0.0
    mask = mask.to(device)
    # 2 channels per sample, mask shared: each sample contributes
    # 2 * mask.sum() terms to the masked-MSE denominator.
    terms_per_sample = float(2 * mask.sum().item())

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch[:, :2]                       # mtco_anom, mtwa_anom
        x_hat, _ = model(batch)
        loss = masked_mse(x_hat, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sum_sq += ((x_hat - target) ** 2 * mask).sum().item()
            n_terms += batch.shape[0] * terms_per_sample

    if n_terms <= 0:
        return {"mse": float("nan"), "rmse": float("nan")}

    mse = sum_sq / n_terms
    return {"mse": mse, "rmse": math.sqrt(mse)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mask: torch.Tensor,
    device: str | torch.device,
) -> dict[str, float]:
    """Compute per-valid-cell MSE / RMSE on a loader.

    Parameters
    ----------
    model : nn.Module
        Set to ``eval`` mode internally.
    loader : DataLoader
        Batches are ``(B, 3, H, W)`` = ``(mtco_anom, mtwa_anom, valid_mask)``;
        target is the first two channels.
    mask : torch.Tensor of shape (H, W) or broadcastable
        ``safe_valid`` mask used to weight the loss.
    device : str or torch.device

    Returns
    -------
    dict with keys ``mse``, ``rmse`` (°C anomaly units).
    """
    model.eval()

    sum_sq = 0.0
    n_terms = 0.0

    mask = mask.to(device)
    # 2 channels per sample, mask shared: each sample contributes
    # 2 * mask.sum() terms to the masked-MSE denominator.
    terms_per_sample = float(2 * mask.sum().item())

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch[:, :2]
        x_hat, _ = model(batch)
        sum_sq += ((x_hat - target) ** 2 * mask).sum().item()
        n_terms += batch.shape[0] * terms_per_sample

    if n_terms <= 0:
        return {"mse": float("nan"), "rmse": float("nan")}

    mse = sum_sq / n_terms
    return {"mse": mse, "rmse": math.sqrt(mse)}


# ---------------------------------------------------------------------------
# Checkpoint helpers.
# ---------------------------------------------------------------------------
def _save_checkpoint(
    path: str,
    epoch: int,
    state: dict[str, torch.Tensor],
    val_mse: float | None,
) -> None:
    """Write a checkpoint ``{"epoch", "state_dict", "val_mse"}``; mkdir-p parent.

    ``val_mse`` is ``None`` in val-less mode.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": state, "val_mse": val_mse}, path)


# ---------------------------------------------------------------------------
# Log formatting.
# ---------------------------------------------------------------------------
def _format_epoch_line(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    *,
    lr: float,
    s_ep: float,
    improved: bool,
    has_val: bool,
) -> str:
    """Format one verbose-mode epoch summary. ``*`` marks val improvements."""
    tail = f"lr={lr:.2e}  ({s_ep:.1f}s)"
    if has_val:
        assert val_metrics is not None
        star = " *" if improved else "  "
        return (
            f"epoch {epoch:3d}{star} "
            f"train_mse={train_metrics['mse']:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  "
            f"val_rmse={val_metrics['rmse']:.3f}  " + tail
        )
    return (
        f"epoch {epoch:3d}   "
        f"train_mse={train_metrics['mse']:.4f}  "
        f"train_rmse={train_metrics['rmse']:.3f}  " + tail
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
            ``train_mse``, ``train_rmse``, ``lr``, ``epoch_seconds``;
            with val also the ``val_*`` keys).
        ``best_val_loss``   : float or ``NaN`` (no val).
        ``best_epoch``      : int.
        ``best_state_dict`` : OrderedDict of CPU tensors. Best-val epoch
            with validation; final-epoch without.
        ``stopped_early``   : bool.
        ``epochs_trained``  : int.
    """
    set_seed(seed) # set seed for reproducability 
    device = torch.device(device) 
    model = model.to(device) # move model to requested device (CPU or GPU)
    if mask is None:
        raise ValueError("mask is required (the safe_valid mask used in the loss).")

    # convert mask array to a PyTorch tensor on device
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)

    # create the optimiser and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # builds dict initiated with empty lists appended at end of each epoch
    # adds val_ keys only if val_loader is given to avoid flatline "val" plot
    has_val = val_loader is not None
    history: dict[str, list] = {
        "train_mse": [],
        "train_rmse": [],
        "lr": [],
        "epoch_seconds": [],
    }
    if has_val:
        history["val_mse"] = []
        history["val_rmse"] = []

    # best-state tracking
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

    for epoch in epoch_iter:     # main loop -> each iteration is one epoch
        t0 = time.perf_counter()

        train_metrics = _train_one_epoch(
            model, train_loader, optimizer, mask_t, device,
        ) # train one epoch
        val_metrics = (
            evaluate(model, val_loader, mask_t, device)
            if has_val
            else None
        ) # evaluate on val, if there is a val loader

        # Log the LR before stepping, so the logged value matches what
        # the just-completed epoch actually trained with.
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["train_mse"].append(train_metrics["mse"]) # append metrics to hist
        history["train_rmse"].append(train_metrics["rmse"])
        if has_val:
            assert val_metrics is not None
            history["val_mse"].append(val_metrics["mse"])
            history["val_rmse"].append(val_metrics["rmse"])
        history["epoch_seconds"].append(time.perf_counter() - t0)

        # Best-tracking only runs with a val loader; in the val-less
        # mode the "best" is set to the final-epoch state after the loop.
        improved = False
        if has_val:
            assert val_metrics is not None
            val_loss = val_metrics["mse"]
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

        # Live progress-bar postfix: prefer val_mse when available,
        # fall back to train_mse (NaN displays sensibly).
        if pbar is not None:
            postfix = (
                {
                    "val_mse": f"{val_metrics['mse']:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                }
                if has_val
                else {
                    "train_mse": f"{train_metrics['mse']:.4f}",
                    "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                }
            )
            pbar.set_postfix(postfix, refresh=False)

        if verbose and (epoch % log_every == 0 or improved):
            log(_format_epoch_line(
                epoch, train_metrics, val_metrics,
                lr=history["lr"][-1],
                s_ep=history["epoch_seconds"][-1],
                improved=improved,
                has_val=has_val,
            ))

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
                    f"in {patience} epochs (best val_mse={best_val_loss:.4f} "
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
        last_epoch = len(history["train_mse"]) - 1
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
        "epochs_trained": len(history["train_mse"]),
    }
