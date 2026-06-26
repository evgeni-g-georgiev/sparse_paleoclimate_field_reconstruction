"""Tests for grid lookup, innovation, and standardisation (paleoreco.assim.innovation)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.innovation import (
    nearest_age_index,
    nearest_lat_index,
    nearest_lon_index,
    obs_cell_index,
    obs_operator_scale,
    predicted_sd,
    r_diagonal,
    standardise,
)
from paleoreco.assim.scoring import ANOMALY, NORMALISED


def test_nearest_lat_index():
    lats = np.array([-10.0, 0.0, 10.0])
    assert list(nearest_lat_index(np.array([3.0, -9.0, 10.0]), lats)) == [1, 0, 2]


def test_nearest_lon_index_wraps_at_seam():
    lons = np.array([-180.0, -90.0, 0.0, 90.0])
    # 170 deg is 10 deg from the -180(=+180) cell, far from +90.
    assert nearest_lon_index(np.array([170.0]), lons)[0] == 0
    assert nearest_lon_index(np.array([5.0]), lons)[0] == 2


def test_nearest_age_index():
    ages = np.array([1000, 1025, 1050])
    assert list(nearest_age_index(np.array([1000, 1030, 1049]), ages)) == [0, 1, 2]


def test_obs_cell_index_flat_layout():
    lats = np.array([-10.0, 0.0, 10.0])      # n_lat = 3
    lons = np.array([0.0, 90.0])             # n_lon = 2
    n_cells = 6
    g = obs_cell_index(np.array([0.0]), np.array([90.0]),
                       np.array(["mtwa"]), lats, lons)
    # mtwa -> channel 1; lat 0.0 -> ilat 1; lon 90 -> ilon 1.
    assert g[0] == 1 * n_cells + 1 * 2 + 1


def test_r_diagonal_adds_representativeness():
    n_cells = 4
    gather = np.array([0, 5])                # channels 0 and 1
    sse = np.array([1.0, 2.0])
    rep_flat = np.arange(8, dtype=float)     # rep at cell index = the index
    c = np.array([0.5, 0.25])
    r = r_diagonal(gather, sse, rep_flat, c, n_cells)
    assert r[0] == pytest.approx(1.0 + 0.5 * 0.0)
    assert r[1] == pytest.approx(2.0 + 0.25 * 5.0)


def test_predicted_sd_and_standardise():
    prior_var = np.array([4.0])
    sse = np.array([5.0])
    sy = np.array([3.0])
    assert predicted_sd(prior_var, sse, sy, ANOMALY)[0] == pytest.approx(3.0)        # sqrt(9)
    assert predicted_sd(prior_var, sse, sy, NORMALISED)[0] == pytest.approx(1.0)     # 3 / sy
    assert standardise(np.array([6.0]), np.array([3.0]))[0] == pytest.approx(2.0)


def test_obs_operator_scale():
    prior_var = np.array([4.0])
    sy = np.array([2.0])
    assert obs_operator_scale(prior_var, sy, ANOMALY)[0] == pytest.approx(1.0)
    assert obs_operator_scale(prior_var, sy, NORMALISED)[0] == pytest.approx(1.0)   # sqrt(4)/2
