"""Cross-validation drivers for the compressor sweeps.

Fold-loop mechanics shared by the AE and VAE architecture sweeps: per-fold
anomaly centring (the climatology is recomputed on each fold's train ages so
validation never leaks into it), dataset/loader construction, and aggregation
of whatever scalar metrics a caller's fit/eval closure returns. Model-specific
logic (building the net, calling the right ``train``, evaluating val) stays in
the caller's closure; this module only owns the fold loop and the final refit
on a chosen index pool.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from torch.utils.data import DataLoader

from ..data import PaleoFieldDataset, apply_anomaly, compute_zscore_stats


def cross_validate(
    cube: np.ndarray,
    valid: np.ndarray,
    folds: list[dict[str, np.ndarray]],
    fit_eval_fn: Callable[..., dict[str, float]],
    *,
    batch_size: int = 32,
) -> dict:
    """Run ``fit_eval_fn`` on every fold and aggregate its scalar outputs.

    ``folds`` is the ``"folds"`` list from :func:`paleoreco.data.splits.make_blocked_cv`.
    Each fold gets its own per-cell stats fit on ``fold["train"]`` only, so the
    validation ages stay out of the climatology. ``fit_eval_fn`` receives
    ``(train_loader, val_loader, mask, stats, val_ds)``, builds and trains a
    model, and returns a flat dict of scalar metrics (including ``"best_epoch"``).

    Returns ``{"<metric>_mean", "<metric>_std", ..., "n_folds", "per_fold"}``;
    every numeric key in the closure's output is averaged across folds.
    """
    per_fold: list[dict[str, float]] = []
    for fold in folds:
        stats = compute_zscore_stats(cube, fold["train"], valid)
        cube_anom = apply_anomaly(cube, stats)
        mask = stats["safe_valid"]
        train_ds = PaleoFieldDataset(cube_anom, mask, fold["train"])
        val_ds = PaleoFieldDataset(cube_anom, mask, fold["val"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        per_fold.append(fit_eval_fn(train_loader, val_loader, mask, stats, val_ds))

    keys = [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
    agg: dict = {"n_folds": len(per_fold), "per_fold": per_fold}
    for k in keys:
        vals = np.array([f[k] for f in per_fold], dtype=np.float64)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=0))
    return agg


def fit_on_indices(
    cube: np.ndarray,
    valid: np.ndarray,
    train_idx: np.ndarray,
    build_model: Callable[[], "object"],
    train_fn: Callable[..., dict],
    *,
    epochs: int,
    batch_size: int = 32,
    device: str = "cpu",
    **train_kwargs,
) -> tuple:
    """Refit a model on ``train_idx`` for a fixed epoch count, no early stopping.

    Used to retrain a chosen winner on the full non-test pool once the sweep has
    settled its architecture and a stopping epoch. ``epochs`` is the sweep's
    mean best-epoch for that config; ``patience=None`` and no val loader mean the
    model trains exactly that many epochs. Per-cell stats are fit on ``train_idx``.

    Returns ``(model, stats, mask)`` with the trained model loaded with its
    final-epoch weights.
    """
    stats = compute_zscore_stats(cube, train_idx, valid)
    cube_anom = apply_anomaly(cube, stats)
    mask = stats["safe_valid"]
    loader = DataLoader(
        PaleoFieldDataset(cube_anom, mask, train_idx),
        batch_size=batch_size, shuffle=True,
    )
    model = build_model()
    out = train_fn(
        model, loader, None,
        mask=mask,
        max_epochs=epochs, patience=None, device=device,
        **train_kwargs,
    )
    model.load_state_dict(out["best_state_dict"])
    return model, stats, mask


def pool_indices(folds: list[dict[str, np.ndarray]]) -> np.ndarray:
    """The non-test pool: every age that serves as validation in some fold.

    Each pool age is validated exactly once under
    :func:`paleoreco.data.splits.make_blocked_cv`, so the union of fold validations is
    the full train/val pool the winner is refit on.
    """
    return np.unique(np.concatenate([f["val"] for f in folds]))
