"""Training loop for the score-based denoiser.

Trains an :class:`~paleoreco.models.diffusion.EDMDenoiser` by EDM
denoising-score-matching (Karras et al. 2022): draw a per-sample noise level
``sigma`` from a log-normal, corrupt the clean anomaly field, and regress the
denoiser back to it under the EDM loss weighting ``lambda(sigma)``. Masked cells
are held at zero on both the input and the target, so the denoiser never learns
to move them.

Training runs fixed-length, or against a held-out split with early stopping when
one is given. An EMA of the weights is kept because sampling quality reads the
averaged weights, not the last step, so the held-out score reads the EMA rather
than the live weights. The checkpoint carries the per-cell normalisation field
alongside the weights so the sampler reconstructs the model and its anomaly frame
together.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ._common import set_seed
from paleoreco.models.diffusion import P_MEAN, P_STD


class _EMA:
    """Exponential moving average of the model's floating-point tensors.

    Seeded at zero and debiased on read, so a checkpoint taken after few updates
    is not a blend with the random initialisation. Converges to the undebiased
    average once ``decay ** steps`` is negligible.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.steps = 0
        self.shadow = {k: (torch.zeros_like(v) if v.dtype.is_floating_point
                           else v.detach().clone())
                       for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state(self) -> dict[str, torch.Tensor]:
        """Debiased average: the raw shadow's weights sum to ``1 - decay ** steps``."""
        if self.steps == 0:
            return dict(self.shadow)
        scale = 1.0 / (1.0 - self.decay ** self.steps)
        return {k: (v * scale if v.dtype.is_floating_point else v)
                for k, v in self.shadow.items()}


def _sample_sigma(n: int, device) -> torch.Tensor:
    """Per-sample noise level ``ln(sigma) ~ N(P_mean, P_std^2)`` (EDM)."""
    return torch.exp(P_MEAN + P_STD * torch.randn(n, device=device))


def edm_loss(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """EDM denoising loss over valid cells, weighted so every noise level counts equally.

    ``mask`` is the ``(H, W)`` valid mask; the noised input is zeroed on masked cells
    so the network always sees zero there, matching the anomaly frame.
    """
    sigma = _sample_sigma(x.shape[0], x.device)
    noised = (x + sigma[:, None, None, None] * torch.randn_like(x)) * mask
    pred = model(noised, sigma)
    weight = (sigma ** 2 + model.sigma_data ** 2) / (sigma * model.sigma_data) ** 2
    sq = ((pred - x) ** 2 * mask).flatten(1).sum(1)
    n_valid = mask.sum() * x.shape[1]
    return (weight * sq / n_valid).mean()


def noise_level_grid(n: int, device) -> torch.Tensor:
    """``n`` log-spaced noise levels spanning +-2 sd of the training log-normal."""
    return torch.exp(torch.linspace(P_MEAN - 2.0 * P_STD, P_MEAN + 2.0 * P_STD,
                                    n, device=device))


@torch.no_grad()
def fixed_grid_loss(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor,
                    sigmas: torch.Tensor, noise: torch.Tensor,
                    batch_size: int = 32) -> float:
    """EDM loss over a fixed noise-level grid with fixed noise, averaged over the grid.

    :func:`edm_loss` draws one sigma per sample, and that sampling spread is an order
    of magnitude larger than the epoch-to-epoch change early stopping reads. Holding
    both the grid and the noise fixed makes successive scores differ only by the
    weights, at the cost of no longer matching the training loss in absolute value.

    The per-sample terms are summed and divided once, so the score does not depend on
    how the states happen to be batched.
    """
    n_valid = mask.sum() * x.shape[1]
    total = 0.0
    for sigma in sigmas:
        weight = (sigma ** 2 + model.sigma_data ** 2) / (sigma * model.sigma_data) ** 2
        for i in range(0, len(x), batch_size):
            xb, eps = x[i:i + batch_size], noise[i:i + batch_size]
            pred = model((xb + sigma * eps) * mask, sigma)
            sq = ((pred - xb) ** 2 * mask).flatten(1).sum(1)
            total += float((weight * sq / n_valid).sum())
    return total / (len(sigmas) * len(x))


def train(
    model: torch.nn.Module,
    cube_norm: np.ndarray,
    safe_valid: np.ndarray,
    *,
    val_cube: np.ndarray | None = None,
    patience: int | None = None,
    n_noise_levels: int = 8,
    train_probe_size: int = 64,
    probe_every: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    max_epochs: int = 2000,
    lr_schedule_epochs: int | None = None,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    checkpoint_path: str | None = None,
    scales: np.ndarray | None = None,
    checkpoint_extra: dict | None = None,
    ema_decay: float | None = 0.999,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 50,
    progress: bool = True,
) -> dict[str, Any]:
    """Train the denoiser on the normalised anomaly cube; return a history dict.

    ``cube_norm`` is ``(N, 2, H, W)`` in the per-cell-scaled anomaly frame with
    masked cells at zero. The saved ``state_dict`` is the EMA weights when
    ``ema_decay`` is set, else the final weights.

    ``val_cube`` turns on the held-out score (:func:`fixed_grid_loss` on the EMA
    weights, plus a live-weight copy and a training-split probe for diagnosis) and,
    with ``patience``, early stopping. It must be in the same normalised frame as
    ``cube_norm``, since the score is not comparable across frames. A held-out split
    and a ``checkpoint_path`` are mutually exclusive: an epoch-count search must not
    overwrite a production checkpoint.

    ``lr_schedule_epochs`` sets the cosine horizon independently of ``max_epochs``, so
    a shortened refit can follow the trajectory the search selected on.
    ``checkpoint_extra`` merges caller-owned provenance into the saved payload; values
    must be plain Python scalars to stay loadable under ``weights_only``.

    ``epoch_seconds`` covers the held-out evaluations as well as the training pass, so
    the reported rate matches wall clock.
    """
    if val_cube is not None and checkpoint_path is not None:
        raise ValueError("val_cube and checkpoint_path are mutually exclusive; the "
                         "search run must not write a production checkpoint")
    if val_cube is not None and len(val_cube) == 0:
        raise ValueError("val_cube is empty")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be >= 1; got {patience}")
    schedule_epochs = lr_schedule_epochs or max_epochs
    if max_epochs > schedule_epochs:
        raise ValueError(f"max_epochs={max_epochs} exceeds lr_schedule_epochs="
                         f"{schedule_epochs}; the cosine would turn back up")
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    mask = torch.as_tensor(np.asarray(safe_valid), dtype=torch.float32, device=device)

    data = torch.as_tensor(np.asarray(cube_norm, dtype=np.float32))
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs)
    ema = _EMA(model, ema_decay) if ema_decay is not None else None

    history: dict[str, list] = {"train_loss": [], "lr": [], "epoch_seconds": []}

    has_val = val_cube is not None
    if has_val:
        sigmas = noise_level_grid(n_noise_levels, device)
        # A CPU generator keeps the score reproducible across devices, and a dedicated
        # stream keeps the val configuration from perturbing the training trajectory.
        gen = torch.Generator().manual_seed(seed + 1)
        val_x = torch.as_tensor(np.asarray(val_cube, dtype=np.float32), device=device)
        val_noise = torch.randn(val_x.shape, generator=gen).to(device)
        # A fixed slice of the training split on the same grid, so the plotted train
        # and val curves measure one quantity and the gap between them is the overfit.
        probe_x = data[:train_probe_size].to(device)
        probe_noise = torch.randn(probe_x.shape, generator=gen).to(device)
        eval_model = copy.deepcopy(model).eval().requires_grad_(False)
        history["val_loss"] = []        # EMA weights, the curve early stopping reads
        history["val_loss_live"] = []   # live weights, diagnostic only
        history["train_probe_loss"] = []
        history["train_probe_epoch"] = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0
    stopped_early = False
    early_stop_enabled = has_val and patience is not None

    pbar = tqdm(range(max_epochs), desc="diffusion", unit="ep", leave=True,
                dynamic_ncols=True) if progress else None
    epoch_iter = pbar if pbar is not None else range(max_epochs)
    log = tqdm.write if progress else print

    for epoch in epoch_iter:
        t0 = time.perf_counter()
        model.train()
        running, n_batches = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            loss = edm_loss(model, batch, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running += loss.item()
            n_batches += 1

        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history["train_loss"].append(running / max(n_batches, 1))

        val_loss = float("nan")
        if has_val:
            val_live = fixed_grid_loss(model, val_x, mask, sigmas, val_noise, batch_size)
            eval_model.load_state_dict(
                ema.state() if ema is not None else model.state_dict())
            val_loss = fixed_grid_loss(eval_model, val_x, mask, sigmas, val_noise,
                                       batch_size)
            history["val_loss"].append(val_loss)
            history["val_loss_live"].append(val_live)
            if epoch % probe_every == 0:
                history["train_probe_loss"].append(
                    fixed_grid_loss(eval_model, probe_x, mask, sigmas, probe_noise,
                                    batch_size))
                history["train_probe_epoch"].append(epoch)
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in eval_model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        history["epoch_seconds"].append(time.perf_counter() - t0)

        if pbar is not None:
            s_ep = f"{history['epoch_seconds'][-1]:.1f}"
            train_loss = f"{history['train_loss'][-1]:.4f}"
            pbar.set_postfix(
                {"loss": train_loss, "val": f"{val_loss:.4f}",
                 "best": f"{best_val_loss:.4f}", "wait": f"{epochs_since_improve}",
                 "s/ep": s_ep} if has_val
                else {"loss": train_loss, "s/ep": s_ep}, refresh=False)
        if verbose and epoch % log_every == 0:
            tail = f"lr={history['lr'][-1]:.2e}  ({history['epoch_seconds'][-1]:.1f}s)"
            if has_val:
                log(f"epoch {epoch:4d}  loss={history['train_loss'][-1]:.4f}  "
                    f"val={val_loss:.4f}  best={best_val_loss:.4f}@{best_epoch}  {tail}")
            else:
                log(f"epoch {epoch:4d}  loss={history['train_loss'][-1]:.4f}  {tail}")

        if early_stop_enabled and epochs_since_improve >= patience:
            stopped_early = True
            if verbose:
                log(f"Early stopping at epoch {epoch} - no improvement in {patience} "
                    f"epochs (best val_loss={best_val_loss:.4f} at epoch {best_epoch}).")
            if pbar is not None:
                pbar.close()
            break

    if pbar is not None and not stopped_early:
        pbar.close()

    raw = ema.state() if ema is not None else model.state_dict()
    state = {k: v.detach().cpu().clone() for k, v in raw.items()}
    epochs_trained = len(history["train_loss"])
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        payload = {"epoch": epochs_trained - 1, "state_dict": state,
                   "config": model.config, "sigma_data": model.sigma_data}
        if scales is not None:
            # Store as a plain list so the checkpoint loads under weights_only; a numpy
            # array does not survive that load.
            payload["scales"] = np.asarray(scales, dtype=np.float64).tolist()
        if checkpoint_extra:
            payload.update(checkpoint_extra)
        # Write beside the target and rename, so an interrupted save cannot truncate
        # an existing checkpoint.
        tmp = f"{checkpoint_path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, checkpoint_path)

    out = {"history": history, "best_state_dict": best_state if best_state is not None else state,
           "epochs_trained": epochs_trained}
    if has_val:
        # best_epoch indexes the history lists; stop_epoch counts completed epochs, so
        # a refit takes max_epochs=best_epoch + 1.
        out.update({"best_val_loss": best_val_loss, "best_epoch": best_epoch,
                    "stop_epoch": epochs_trained, "stopped_early": stopped_early})
    return out
