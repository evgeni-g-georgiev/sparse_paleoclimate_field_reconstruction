"""Tests for DA skill and calibration metrics (paleoreco.eval.da)."""

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


def test_rcrv_calibrated_when_spread_matches_error():
    # z = (truth - mean)/sqrt(var); with unit error and mean 0, std(z) ~ 1, bias ~ 0.
    rng = np.random.default_rng(0)
    n = 20000
    truth = rng.normal(0.0, 1.0, n)
    mean = np.zeros(n)
    var = np.ones(n)
    bias, disp = da.rcrv(truth, mean, var)
    assert bias == pytest.approx(0.0, abs=0.05)
    assert disp == pytest.approx(1.0, abs=0.05)


def test_crps_gaussian_at_zero_z():
    # For truth == mean, z = 0, CRPS = sd * (2*phi(0) - 1/sqrt(pi)).
    mean = np.array([0.0])
    var = np.array([4.0])               # sd = 2
    expected = 2.0 * (2.0 / np.sqrt(2 * np.pi) - 1.0 / np.sqrt(np.pi))
    assert da.crps_gaussian(mean, mean, var)[0] == pytest.approx(expected)


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
