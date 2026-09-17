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


def test_lowpass_keeps_the_length_of_a_series_shorter_than_its_window():
    """A window wider than the series must not lengthen it.

    ``np.convolve`` in "same" mode returns the longer of its two arguments, so an
    unguarded kernel comes back longer than the series went in. Differencing two
    low-passes then fails on a shape nothing upstream ever set, which is how a short run
    breaks a band metric rather than simply having none.
    """
    stack = np.arange(6.0)[:, None, None] * np.ones((1, 3, 3))
    for window_yr in (25.0, 100.0, 1000.0):
        assert da.lowpass_time(stack, window_yr, 25.0).shape == stack.shape

    # Two windows that both exceed the series still difference cleanly.
    wide = da.lowpass_time(stack, 500.0, 25.0) - da.lowpass_time(stack, 1000.0, 25.0)
    assert wide.shape == stack.shape


def test_trim_drops_a_window_the_series_cannot_support():
    """The trim, not the filter, is what decides a window is unmeasurable.

    ``timescale_trim`` keeps the unclamped kernel, so a window wider than the series
    leaves nothing after trimming and the caller emits no row for it.
    """
    n, step = 43, 25.0
    assert n - 2 * da.timescale_trim(100.0, step) >= 3        # measurable
    assert n - 2 * da.timescale_trim(2000.0, step) < 3        # not, and dropped
