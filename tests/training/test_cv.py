"""Tests for the cross-validation harness (paleoreco.cv).

The fold loop and refit are exercised with a trivial closure / model so the test
stays fast; real training is covered by the AE smoke test.
"""

from __future__ import annotations

import numpy as np

from paleoreco.training.cv import cross_validate, pool_indices


def test_pool_indices_is_union_of_fold_vals():
    folds = [
        {"train": np.array([0, 1]), "val": np.array([2, 3])},
        {"train": np.array([2, 3]), "val": np.array([4, 5, 2])},
    ]
    assert np.array_equal(pool_indices(folds), np.array([2, 3, 4, 5]))


def test_cross_validate_aggregates_scalar_metrics(cube, valid):
    n = cube.shape[0]
    folds = [
        {"train": np.array([0, 1, 2, 3]), "val": np.array([4, 5])},
        {"train": np.array([6, 7, 8, 9]), "val": np.array([10, 11])},
    ]

    calls = {"n": 0}

    def fit_eval_fn(train_loader, val_loader, mask, stats, val_ds):
        # Confirm the harness handed us usable per-fold objects.
        assert mask.shape == valid.shape
        assert stats["mean"].shape[0] == 2
        calls["n"] += 1
        return {"best_epoch": calls["n"], "val_mse_z": float(calls["n"])}

    agg = cross_validate(cube, valid, folds, fit_eval_fn, batch_size=4)

    assert agg["n_folds"] == 2
    assert len(agg["per_fold"]) == 2
    # Mean of {1, 2} = 1.5 for both reported scalars.
    assert agg["val_mse_z_mean"] == 1.5
    assert agg["best_epoch_mean"] == 1.5
    assert "val_mse_z_std" in agg
