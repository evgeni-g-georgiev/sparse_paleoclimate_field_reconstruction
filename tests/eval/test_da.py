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


def test_rmse_and_rrmse():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = a + 1.0
    assert da.rmse(a, b) == pytest.approx(1.0)
    # RRMSE = rmse / std(truth).
    assert da.relative_rmse(a, b) == pytest.approx(1.0 / np.std(a))


def test_amplitude_ratio():
    truth = np.array([1.0, -1.0, 2.0, -2.0])
    assert da.amplitude_ratio(truth, truth) == pytest.approx(1.0)
    assert da.amplitude_ratio(truth, 0.5 * truth) == pytest.approx(0.5)   # too flat
    assert da.amplitude_ratio(truth, np.zeros_like(truth)) == pytest.approx(0.0)


def test_ce_map_over_stack():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(8, 2, 3))
    ref = truth.mean(axis=0)
    assert np.allclose(da.ce_map(truth, truth, ref), 1.0)   # perfect -> CE 1 everywhere
    assert np.allclose(da.ce_map(truth, np.broadcast_to(ref, truth.shape), ref), 0.0)


def test_nearest_obs_distance_zero_at_site():
    lats = np.array([-10.0, 0.0, 10.0])
    lons = np.array([0.0, 90.0])
    d = da.nearest_obs_distance(lats, lons, np.array([0.0]), np.array([0.0]))
    # The cell coincident with the observation has zero distance.
    assert d.min() == pytest.approx(0.0, abs=1e-6)
    assert d.shape == (lats.size * lons.size,)


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
