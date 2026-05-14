"""Training loop for the autoencoder.

What this module does
---------------------
Given :class:`~paleoreco.models.autoencoder.ConvAE`,
two PyTorch ``DataLoader``s (train and val), and the ``safe_valid`` mask
produced by :func:`paleoreco.data.compute_zscore_stats`, this module
trains the model according to the following recipe:

* AdamW (``lr=1e-3``, ``weight_decay=1e-4``)
* CosineAnnealingLR over ``max_epochs``
* Early stopping on validation masked-MSE (default patience 30 epochs)
* Best-val-loss checkpoint written to disk

The loop optimises masked-MSE **in z-score units** because that's the
scale the model lives in. For monitoring we additionally
report **RMSE in °C**.

Reproducibility
---------------
:func:`set_seed` seeds Python ``random``, NumPy, and PyTorch (CPU + CUDA
if present). It does *not* enable PyTorch's deterministic mode.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any

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

    Deliberately skips ``torch.use_deterministic_algorithms(True)``: it
    forces slower kernels, where seeding the RNGs
    already gives us run-to-run reproducibility on a fixed machine.
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
    """Run one training epoch. Returns mean masked-MSE in z-score units.

    The returned value is the running average of the per-batch loss with
    *in-progress* weights, so it lags the post-epoch evaluation by half a
    step. Used only for diagnostics; the authoritative train metric is
    produced by :func:`evaluate` after the epoch ends.
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
    """Compute the per-valid-cell MSE / RMSE on a loader.

    Parameters
    ----------
    model : nn.Module
        The autoencoder. Set to ``eval`` mode internally.
    loader : DataLoader
        Train, val, or test loader. Each batch is ``(B, 3, H, W)`` with
        channels ``(mtco_z, mtwa_z, valid_mask)``; the target is the
        first two channels.
    mask : torch.Tensor of shape (H, W) or broadcastable
        ``safe_valid`` mask used to weight the loss.
    device : str or torch.device
        Where to run forward passes.
    zscore_std : torch.Tensor or None, default None
        Per-cell std from :func:`paleoreco.data.compute_zscore_stats`,
        shape ``(2, H, W)``. If provided, the function also returns
        ``rmse_celsius``; otherwise that key is ``None``.

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
    # Number of (channel, valid-cell) terms per sample. The output has 2
    # channels and the mask is shared, so each sample contributes
    # 2 * mask.sum() terms regardless of batch index.
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
    val_loader: DataLoader,
    mask: torch.Tensor | np.ndarray,
    *,
    zscore_std: torch.Tensor | np.ndarray | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 30,
    device: str | torch.device = "cpu",
    checkpoint_path: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 1,
    progress: bool = True,
) -> dict[str, Any]:
    """Train ``model`` with AdamW + cosine LR + early stopping.

    Parameters
    ----------
    model : nn.Module
        ConvAE (or any module with the same I/O
        contract: takes ``(B, 3, H, W)``, returns ``(x_hat, z)`` where
        ``x_hat`` is ``(B, 2, H, W)``).
    train_loader, val_loader : DataLoader
        Wrap ``PaleoFieldDataset`` instances on disjoint age subsets.
    mask : (H, W) array
        ``safe_valid`` mask. Will be moved to ``device`` and used in the
        masked-MSE loss and in :func:`evaluate`.
    zscore_std : (2, H, W) array, optional
        Per-cell std for °C-unit reporting. ``rmse_celsius`` in the
        returned history will be ``None`` if not given.
    lr, weight_decay : float
        AdamW hyperparameters.
    max_epochs : int
        Upper bound on training; cosine LR schedules over this length.
        Real training usually stops earlier via early stopping.
    patience : int
        Number of consecutive epochs without val-loss improvement before
        early stopping triggers.
    device : str or torch.device
        ``"cpu"`` or ``"cuda"``.
    checkpoint_path : str, optional
        If given, the best-val-loss state dict is written here as a
        ``.pt`` file every time a new best is found.
    seed : int
        RNG seed.
    verbose : bool
        Print one line per ``log_every`` epochs (and on every best-epoch).
    log_every : int
        Logging cadence (epochs between progress prints).
    progress : bool
        Show a tqdm progress bar with ETA, current val_mse_z and best.
        Independent of ``verbose`` so the sweep loop can hide per-epoch
        text but still get a per-config progress bar.

    Returns
    -------
    dict with keys:
        ``history``           : dict of lists, one per epoch, keys
                                ``train_mse_z``, ``val_mse_z``,
                                ``train_rmse_z``, ``val_rmse_z``,
                                ``train_rmse_celsius``,
                                ``val_rmse_celsius``, ``lr``,
                                ``epoch_seconds``.
        ``best_val_loss``     : float (z-score MSE).
        ``best_epoch``        : int.
        ``best_state_dict``   : OrderedDict of CPU tensors at the best
                                epoch (suitable for ``model.load_state_dict``).
        ``stopped_early``     : bool.
        ``epochs_trained``    : int.
    """
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)

    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
    std_t: torch.Tensor | None = None
    if zscore_std is not None:
        std_t = torch.as_tensor(zscore_std, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    history: dict[str, list] = {
        "train_mse_z": [],
        "val_mse_z": [],
        "train_rmse_z": [],
        "val_rmse_z": [],
        "train_rmse_celsius": [],
        "val_rmse_celsius": [],
        "lr": [],
        "epoch_seconds": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0
    stopped_early = False

    # Wrap the epoch loop with tqdm if progress is requested. We use
    # tqdm.write() for the verbose per-epoch prints so they don't clobber
    # the bar; without progress, regular print() is used.
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
        val_metrics = evaluate(model, val_loader, mask_t, device, std_t)

        # Step the scheduler once per epoch; current LR is read *before*
        # the step so the logged value matches what the just-completed
        # epoch trained with.
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["train_mse_z"].append(train_metrics["mse_z"])
        history["val_mse_z"].append(val_metrics["mse_z"])
        history["train_rmse_z"].append(train_metrics["rmse_z"])
        history["val_rmse_z"].append(val_metrics["rmse_z"])
        history["train_rmse_celsius"].append(train_metrics["rmse_celsius"])
        history["val_rmse_celsius"].append(val_metrics["rmse_celsius"])
        history["epoch_seconds"].append(time.perf_counter() - t0)

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

        # Live progress-bar postfix: current val + best + sec/epoch.
        if pbar is not None:
            pbar.set_postfix(
                {
                    "val_mse_z": f"{val_loss:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "s/ep": f"{history['epoch_seconds'][-1]:.1f}",
                },
                refresh=False,
            )

        if verbose and (epoch % log_every == 0 or improved):
            c_str = (
                f"  val_rmse_C={val_metrics['rmse_celsius']:.3f}"
                if val_metrics["rmse_celsius"] is not None
                else ""
            )
            star = " *" if improved else "  "
            log(
                f"epoch {epoch:3d}{star} "
                f"train_mse_z={train_metrics['mse_z']:.4f}  "
                f"val_mse_z={val_loss:.4f}{c_str}  "
                f"lr={history['lr'][-1]:.2e}  "
                f"({history['epoch_seconds'][-1]:.1f}s)"
            )

        if epochs_since_improve >= patience:
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

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "stopped_early": stopped_early,
        "epochs_trained": len(history["train_mse_z"]),
    }
