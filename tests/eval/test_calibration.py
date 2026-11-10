"""Tests for the calibration metrics (paleoreco.eval.calibration).

Each test pins a property the metric is used for: CRPS collapsing to absolute error
when the spread vanishes and matching a Monte Carlo estimate, and the RCRV dispersion
reporting a known miscalibration with the right sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.eval import calibration as cal


def _calibrated(n=20000, sd=2.0, seed=0):
    """Truths drawn from exactly the posterior that is handed to the metric."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=n)
    truth = mean + rng.normal(scale=sd, size=n)
    return truth, mean, np.full(n, sd ** 2)


def test_crps_gaussian_collapses_to_absolute_error():
    truth = np.array([1.0, -2.0, 0.5])
    mean = np.array([0.0, 0.0, 0.0])
    tiny = np.full(3, 1e-12)
    assert cal.crps_gaussian(truth, mean, tiny) == pytest.approx(np.abs(truth), abs=1e-5)


def test_crps_gaussian_matches_a_monte_carlo_estimate():
    """The energy form ``E|X - y| - E|X - X'| / 2`` over draws from the same Gaussian."""
    rng = np.random.default_rng(3)
    truth, mean, var = 0.7, 0.0, 4.0
    x = rng.normal(mean, np.sqrt(var), size=40000)
    energy = np.abs(x - truth).mean() - 0.5 * np.abs(x[:20000] - x[20000:]).mean()
    got = cal.crps_gaussian(np.array([truth]), np.array([mean]), np.array([var]))[0]
    assert got == pytest.approx(energy, rel=0.02)


def test_rcrv_honest_when_variance_is_right():
    truth, mean, var = _calibrated()
    assert cal.rcrv_dispersion(truth, mean, var) == pytest.approx(1.0, abs=0.02)


def test_rcrv_dispersion_flags_overconfidence():
    truth, mean, var = _calibrated()
    # Claiming a quarter of the true variance halves the stated sd, so errors read 2x.
    assert cal.rcrv_dispersion(truth, mean, var / 4.0) == pytest.approx(2.0, abs=0.05)

