"""Tests for the calibration metrics (paleoreco.eval.calibration).

Each test pins a property the metric is used for: CRPS collapsing to absolute error
when the spread vanishes, the Gaussian and ensemble estimators agreeing, and RCRV /
coverage / PIT reporting a known miscalibration with the right sign.
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


def test_crps_gaussian_matches_ensemble_estimator():
    rng = np.random.default_rng(3)
    truth, mean, var = np.array([0.7]), np.array([0.0]), np.array([4.0])
    samples = rng.normal(mean[0], np.sqrt(var[0]), size=(20000, 1))
    assert cal.crps_ensemble(truth, samples)[0] == pytest.approx(
        cal.crps_gaussian(truth, mean, var)[0], rel=0.02)


def test_crps_ensemble_single_member_is_absolute_error():
    truth, samples = np.array([2.0, -1.0]), np.array([[0.0, 0.0]])
    assert cal.crps_ensemble(truth, samples) == pytest.approx([2.0, 1.0])


def test_crpss_bounds():
    model, ref = np.array([1.0, 1.0]), np.array([2.0, 2.0])
    assert cal.crpss(model, ref) == pytest.approx(0.5)
    assert cal.crpss(ref, ref) == pytest.approx(0.0)      # no better than the reference
    assert cal.crpss(np.zeros(2), ref) == pytest.approx(1.0)


def test_rcrv_honest_when_variance_is_right():
    truth, mean, var = _calibrated()
    bias, dispersion = cal.rcrv(truth, mean, var)
    assert bias == pytest.approx(0.0, abs=0.02)
    assert dispersion == pytest.approx(1.0, abs=0.02)


def test_rcrv_dispersion_flags_overconfidence():
    truth, mean, var = _calibrated()
    # Claiming a quarter of the true variance halves the stated sd, so errors read 2x.
    _, dispersion = cal.rcrv(truth, mean, var / 4.0)
    assert dispersion == pytest.approx(2.0, abs=0.05)


def test_rcrv_bias_flags_systematic_offset():
    truth, mean, var = _calibrated(sd=1.0)
    bias, _ = cal.rcrv(truth, mean - 0.5, var)          # predictions run 0.5 too cold
    assert bias == pytest.approx(0.5, abs=0.03)


def test_coverage_matches_nominal_level():
    truth, mean, var = _calibrated()
    assert cal.coverage(truth, mean, var, level=0.9) == pytest.approx(0.9, abs=0.02)
    assert cal.coverage(truth, mean, var, level=0.5) == pytest.approx(0.5, abs=0.02)
    # Too-narrow intervals capture fewer truths than advertised.
    assert cal.coverage(truth, mean, var / 4.0, level=0.9) < 0.75
