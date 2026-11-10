"""Tests for the nearest-cell observation operator (paleoreco.assim.innovation)."""

from __future__ import annotations

import numpy as np

from paleoreco.assim.innovation import (
    nearest_age_index,
    nearest_lat_index,
    nearest_lon_index,
    obs_cell_index,
)


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

