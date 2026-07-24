"""Training loop for the score-based denoiser.

Trains an :class:`~paleoreco.models.diffusion.EDMDenoiser` by EDM
denoising-score-matching (Karras et al. 2022): draw a per-sample noise level
``sigma`` from a log-normal, corrupt the clean anomaly field, and regress the
denoiser back to it under the EDM loss weighting ``lambda(sigma)``. Masked cells
are held at zero on both the input and the target, so the denoiser never learns
to move them.

The prior is small (hundreds of states), so training is val-less and fixed
length; an EMA of the weights is kept because sampling quality reads the averaged
weights, not the last step. The checkpoint carries the per-channel normalisation
scalars alongside the weights so the sampler reconstructs the model and its
anomaly frame together.
"""

from __future__ import annotations

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
    """Exponential moving average of the model's floating-point tensors."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)


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


def train(
    model: torch.nn.Module,
    cube_norm: np.ndarray,
    safe_valid: np.ndarray,
    *,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    max_epochs: int = 2000,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    checkpoint_path: str | None = None,
    channel_scales: np.ndarray | None = None,
    ema_decay: float | None = 0.999,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 50,
    progress: bool = True,
) -> dict[str, Any]:
    """Train the denoiser on the normalised anomaly cube; return a history dict.

    ``cube_norm`` is ``(N, 2, H, W)`` in the per-channel-scaled anomaly frame with
    masked cells at zero. The saved ``state_dict`` is the EMA weights when
    ``ema_decay`` is set, else the final weights.
    """
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    mask = torch.as_tensor(np.asarray(safe_valid), dtype=torch.float32, device=device)

    data = torch.as_tensor(np.asarray(cube_norm, dtype=np.float32))
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    ema = _EMA(model, ema_decay) if ema_decay is not None else None

    history: dict[str, list] = {"train_loss": [], "lr": [], "epoch_seconds": []}
    pbar = tqdm(range(max_epochs), desc="diffusion", unit="ep", leave=True,
                dynamic_ncols=True) if progress else range(max_epochs)
    log = tqdm.write if progress else print

    for epoch in pbar:
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
        history["epoch_seconds"].append(time.perf_counter() - t0)

        if progress:
            pbar.set_postfix({"loss": f"{history['train_loss'][-1]:.4f}",
                              "s/ep": f"{history['epoch_seconds'][-1]:.1f}"}, refresh=False)
        if verbose and epoch % log_every == 0:
            log(f"epoch {epoch:4d}  loss={history['train_loss'][-1]:.4f}  "
                f"lr={history['lr'][-1]:.2e}  ({history['epoch_seconds'][-1]:.1f}s)")

    raw = ema.shadow if ema is not None else model.state_dict()
    state = {k: v.detach().cpu().clone() for k, v in raw.items()}
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        payload = {"epoch": max_epochs - 1, "state_dict": state, "config": model.config,
                   "sigma_data": model.sigma_data}
        if channel_scales is not None:
            # Store as a plain list so the checkpoint loads under weights_only.
            payload["channel_scales"] = np.asarray(channel_scales, dtype=np.float64).tolist()
        torch.save(payload, checkpoint_path)

    return {"history": history, "best_state_dict": state, "epochs_trained": max_epochs}
