"""Masked reconstruction losses for fields on the Prior grid.

Cells outside ``safe_valid`` are zero-filled in the input by
:func:`paleoreco.data.apply_anomaly`; the mask zeroes out their
contribution to the loss so a model isn't trained to predict 0 on
arbitrary cells.
"""

from __future__ import annotations

import torch


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean squared error averaged only over valid cells.

    Parameters
    ----------
    pred : torch.Tensor of shape (B, C, H, W)
    target : torch.Tensor of shape (B, C, H, W)
    mask : torch.Tensor broadcastable to (B, C, H, W)
        ``1`` (or ``True``) where the loss should contribute, ``0`` elsewhere.
        Typical shapes accepted: ``(H, W)``, ``(1, H, W)``, ``(1, 1, H, W)``,
        ``(B, 1, H, W)``, ``(B, C, H, W)``. The function lifts the mask up to
        ``pred``'s rank, so callers don't need to match dims exactly.
    reduction : {"mean", "sum", "none"}
        - ``"mean"``: sum of squared errors on valid terms, divided by the
          count of valid ``(sample, channel, cell)`` terms. This is the
          per-valid-cell MSE.
        - ``"sum"``: raw sum of squared errors on valid terms.
        - ``"none"``: element-wise masked squared error tensor.

    Returns
    -------
    torch.Tensor
        Scalar for ``"mean"``/``"sum"``; tensor matching ``pred`` for ``"none"``.

    Notes
    -----
    * The denominator counts mask terms *after* broadcasting against ``pred``
      so the per-cell mean is correct regardless of whether the caller passes
      a per-batch, per-channel, or shared mask.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} and target shape {tuple(target.shape)} must match"
        )

    # Promote the mask to pred's rank with leading singleton axes so callers
    # can pass an (H, W) mask without reshaping.
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(0)
    mask = mask.to(dtype=pred.dtype)

    sq_err = (pred - target) ** 2 * mask

    if reduction == "sum":
        return sq_err.sum()
    if reduction == "mean":
        # Count contributing terms with the same broadcast as the loss itself, so
        # the denominator is the actual number of valid (sample, channel, cell)
        # entries. clamp_min(1.0) avoids division by zero on an empty mask.
        n_terms = mask.expand_as(pred).sum().clamp_min(1.0)
        return sq_err.sum() / n_terms
    if reduction == "none":
        return sq_err

    raise ValueError(
        f"unknown reduction {reduction!r}; expected one of 'mean', 'sum', 'none'"
    )


def vae_elbo_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beta-VAE ELBO: masked-MSE reconstruction + ``beta`` * KL to N(0, I).

    The KL term is summed over latent dims and averaged over the batch. ``beta`` 
    here is the per-step coefficient; KL-annealing schedules are the trainer's 
    job, not this function's.

    Returns ``(loss, recon_term, kl_term)`` so the trainer can log the
    two terms separately even though only ``loss`` carries grad.
    """
    recon_term = masked_mse(x_hat, target, mask)
    # 0.5 * (mu^2 + sigma^2 - log sigma^2 - 1), summed over latent dims.
    kl_per_sample = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=1)
    kl_term = kl_per_sample.mean()
    loss = recon_term + beta * kl_term
    return loss, recon_term, kl_term
