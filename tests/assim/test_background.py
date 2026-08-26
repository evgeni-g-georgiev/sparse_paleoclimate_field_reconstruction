"""Tests for background state and covariance (paleoreco.assim.background)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.background import (
    background_covariance,
    background_state,
    background_variance,
    temporal_structure_function,
)


def _smooth_cube(n_ages=16, n_lat=3, n_lon=4) -> np.ndarray:
    """A cube whose every cell drifts monotonically, so lags are strictly ordered."""
    t = np.arange(n_ages, dtype=np.float64)[:, None, None, None]
    # Every rate is non-zero, so no cell sits still and no variance is degenerate.
    space = np.linspace(0.5, 1.5, 2 * n_lat * n_lon).reshape(1, 2, n_lat, n_lon)
    return (t * space).astype(np.float32)


def test_background_covariance_matches_numpy(cube):
    idx = np.arange(cube.shape[0])
    B = background_covariance(cube, idx)
    X = cube[idx].reshape(len(idx), -1).astype(np.float64)
    assert np.allclose(B, np.cov(X, rowvar=False))   # both use ddof=1


def test_background_variance_is_diagonal(cube):
    B = background_covariance(cube, np.arange(cube.shape[0]))
    assert np.allclose(background_variance(B), np.diag(B))


def test_background_state_per_age_vs_climatological(cube):
    mean = cube.mean(axis=0)
    per_age = background_state(cube, mean, age_index=2, kind="per_age")
    clim = background_state(cube, mean, age_index=2, kind="climatological")
    assert np.allclose(per_age, cube[2].ravel())
    assert np.allclose(clim, mean.ravel())


def test_background_state_rejects_unknown_kind(cube):
    with pytest.raises(ValueError):
        background_state(cube, cube.mean(axis=0), age_index=0, kind="bogus")


def test_structure_function_variance_matches_covariance_diagonal(cube):
    """``rho = 1 - S / (2 var)`` is only a correlation if both come from the same ages.

    Pinning ``var`` to B's own diagonal is what makes a fully decorrelated observation
    carry exactly the climatological variance as noise.
    """
    idx = np.arange(cube.shape[0])
    _, var = temporal_structure_function(cube, idx, max_lag=3)
    assert np.allclose(var, np.diag(background_covariance(cube, idx)))


def test_structure_function_grows_with_lag_on_a_smooth_field():
    cube = _smooth_cube()
    S, _ = temporal_structure_function(cube, np.arange(cube.shape[0]), max_lag=5)
    assert np.allclose(S[0], 0.0)                       # a state differs from itself by nothing
    assert np.all(np.diff(S, axis=0) > 0)               # every further step costs more


def test_structure_function_clamps_max_lag_to_the_ages_it_has():
    """A lag with no pairs would be a mean over an empty slice, so NaN into R."""
    cube = _smooth_cube(n_ages=6)
    S, _ = temporal_structure_function(cube, np.arange(6), max_lag=99)
    assert S.shape[0] == 6                              # lags 0..5, the most 6 ages support
    assert np.isfinite(S).all()


def test_structure_function_reads_only_the_ages_it_is_given():
    """The leakage guard: ages outside the prior set must not reach the operator."""
    cube = _smooth_cube(n_ages=12)
    prior = np.arange(6)
    S, var = temporal_structure_function(cube, prior, max_lag=3)

    tampered = cube.copy()
    tampered[6:] += 100.0
    S_after, var_after = temporal_structure_function(tampered, prior, max_lag=3)
    assert np.allclose(S, S_after)
    assert np.allclose(var, var_after)
