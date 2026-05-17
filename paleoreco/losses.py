"""Masked MSE / RMSE for reconstruction on the Prior grid.

Cells outside ``safe_valid`` are zero-filled in the input by
:func:`paleoreco.data.apply_zscore`; the mask zeroes out their
contribution to the loss, so a model isn't trained to predict 0 on
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


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Square root of :func:`masked_mse` with the default ``"mean"`` reduction.

    Returned value is in z-score units.
    """
    return masked_mse(pred, target, mask, reduction="mean").sqrt()
