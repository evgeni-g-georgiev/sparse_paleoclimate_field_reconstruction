"""Tests for the resampling ensemble helpers (paleoreco.assim.montecarlo)."""

from __future__ import annotations

import numpy as np

from paleoreco.assim.montecarlo import FieldEnsembleAccumulator, random_site_partitions


def test_field_accumulator_matches_numpy_mean_std():
    rng = np.random.default_rng(2)
    stack = rng.normal(size=(50, 2, 3, 3))
    acc = FieldEnsembleAccumulator()
    for x in stack:
        acc.update(x)
    mean, std = acc.finalize()
    assert np.allclose(mean, stack.mean(axis=0))
    assert np.allclose(std, stack.std(axis=0, ddof=1))   # finalize uses ddof=1


def test_field_accumulator_single_sample_has_zero_spread():
    acc = FieldEnsembleAccumulator()
    acc.update(np.ones((2, 2)))
    mean, std = acc.finalize()
    assert np.allclose(mean, 1.0)
    assert np.allclose(std, 0.0)


def test_random_site_partitions_size_and_determinism():
    sites = np.arange(20)
    parts = random_site_partitions(sites, n_realizations=5, assim_frac=0.75, seed=0)
    assert len(parts) == 5
    for w in parts:
        assert len(w) == 5                      # 25% of 20 withheld
        assert len(set(w.tolist())) == len(w)   # drawn without replacement
    # Same seed -> identical draws.
    again = random_site_partitions(sites, n_realizations=5, assim_frac=0.75, seed=0)
    for a, b in zip(parts, again):
        assert np.array_equal(a, b)
