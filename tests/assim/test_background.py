"""Tests for background state and covariance (paleoreco.assim.background)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.background import (
    background_covariance,
    background_state,
    background_variance,
)


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
