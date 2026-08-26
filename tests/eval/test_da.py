"""Tests for DA skill metrics (paleoreco.eval.da)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.eval import da


def test_coefficient_of_efficiency_bounds():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    ref = np.full_like(truth, truth.mean())
    assert da.coefficient_of_efficiency(truth, truth, ref) == pytest.approx(1.0)   # perfect
    assert da.coefficient_of_efficiency(truth, ref, ref) == pytest.approx(0.0)     # == ref


def test_pearson_rmse_rrmse():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = a + 1.0
    assert da.pearson_r(a, b) == pytest.approx(1.0)
    assert da.rmse(a, b) == pytest.approx(1.0)
    # RRMSE = rmse / std(truth).
    assert da.relative_rmse(a, b) == pytest.approx(1.0 / np.std(a))


def test_amplitude_ratio():
    truth = np.array([1.0, -1.0, 2.0, -2.0])
    assert da.amplitude_ratio(truth, truth) == pytest.approx(1.0)
    assert da.amplitude_ratio(truth, 0.5 * truth) == pytest.approx(0.5)   # too flat
    assert da.amplitude_ratio(truth, np.zeros_like(truth)) == pytest.approx(0.0)


def test_uncertainty_reduction():
    prior = np.array([4.0, 1.0])
    post = np.array([1.0, 1.0])
    red = da.uncertainty_reduction(prior, post)
    assert red[0] == pytest.approx(0.5)    # sd 2 -> 1
    assert red[1] == pytest.approx(0.0)    # unchanged


def test_skill_maps_over_stack():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(8, 2, 3))
    ref = truth.mean(axis=0)
    ce = da.ce_map(truth, truth, ref)
    assert np.allclose(ce, 1.0)            # perfect reconstruction -> CE 1 everywhere
    rm = da.rmse_map(truth, truth)
    assert np.allclose(rm, 0.0)


def test_nearest_obs_distance_zero_at_site():
    lats = np.array([-10.0, 0.0, 10.0])
    lons = np.array([0.0, 90.0])
    d = da.nearest_obs_distance(lats, lons, np.array([0.0]), np.array([0.0]))
    # The cell coincident with the observation has zero distance.
    assert d.min() == pytest.approx(0.0, abs=1e-6)
    assert d.shape == (lats.size * lons.size,)


# --- paired block bootstrap -------------------------------------------------
def _mean_gap(a, b):
    """difference(idx) for a constant offset between two paired series."""
    return lambda idx: float(np.mean(b[idx]) - np.mean(a[idx]))


def test_bootstrap_recovers_a_real_gap_and_excludes_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    b = a + 0.5                                   # paired, so the gap is exact per unit
    point, lo, hi = da.paired_block_bootstrap(_mean_gap(a, b), len(a), n_boot=200)
    assert point == pytest.approx(0.5)
    assert lo > 0 and hi < 1.0


def test_bootstrap_straddles_zero_when_the_two_agree():
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    point, lo, hi = da.paired_block_bootstrap(_mean_gap(a, a.copy()), len(a), n_boot=200)
    assert point == pytest.approx(0.0)
    assert lo <= 0.0 <= hi


def test_longer_blocks_widen_the_interval_on_a_correlated_series():
    """Serial correlation is exactly what block resampling has to stop hiding.

    Drawing a correlated run one unit at a time pretends it holds more independent
    information than it does, so the interval comes out too narrow to believe.
    """
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(size=600))        # neighbouring units nearly identical
    diff = lambda idx: float(np.mean(walk[idx]))  # noqa: E731

    width = []
    for block in (1, 50):
        _, lo, hi = da.paired_block_bootstrap(diff, len(walk), block=block,
                                              n_boot=300, seed=3)
        width.append(hi - lo)
    assert width[1] > width[0]


def test_bootstrap_covers_every_unit_exactly_once_in_the_point_estimate():
    """The point estimate must be the plain statistic, not a resampled draw."""
    a = np.arange(10.0)
    point, _, _ = da.paired_block_bootstrap(lambda idx: float(np.sum(a[idx])), len(a),
                                            block=3, n_boot=5)
    assert point == pytest.approx(a.sum())
