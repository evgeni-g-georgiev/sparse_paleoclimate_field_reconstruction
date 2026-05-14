"""Loss functions for the autoencoder.

The AE reconstructs the two temperature channels (``mtco``, ``mtwa``) on the
``safe_valid`` cells produced by :func:`paleoreco.data.compute_zscore_stats`.
Cells outside the mask are zero-filled by :func:`paleoreco.data.apply_zscore`
and would otherwise dominate the loss (asking the network to predict 0 on
arbitrary cells). The loss in this module multiplies the squared error by
the mask so masked cells contribute nothing to the gradient.
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
        Reconstructed fields.
    target : torch.Tensor of shape (B, C, H, W)
        Ground-truth fields.
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
    * Both inputs are assumed already z-scored. The loss therefore lives in
      dimensionless units; convert to °C by inverting the z-score before
      reporting human-readable error.
    * The denominator counts mask terms *after* broadcasting against ``pred``
      so the per-cell mean is correct regardless of whether the caller passes
      a per-batch, per-channel, or shared mask.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} and target shape {tuple(target.shape)} must match"
        )

    # Promote the mask to pred's rank with leading singleton axes. This keeps
    # the call site clean: the dataset hands us a (H, W) mask and we don't
    # require it to be reshaped to (1, 1, H, W) at every call.
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(0)
    mask = mask.to(dtype=pred.dtype)

    sq_err = (pred - target) ** 2 * mask

    if reduction == "sum":
        return sq_err.sum()
    if reduction == "mean":
        # Count contributing terms with the same broadcast as the loss itself,
        # so the denominator is exactly "number of (sample, channel, cell)
        # values that the mask let through". clamp_min(1.0) guards the
        # degenerate case where the mask is empty (would never happen in v1,
        # but keeps the function defensible if reused).
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
    """Root of :func:`masked_mse` with the default ``"mean"`` reduction.

    Useful for monitoring: RMSE in z-score units has a natural reading
    ("about how many standard deviations off the model is, per cell").
    """
    return masked_mse(pred, target, mask, reduction="mean").sqrt()
