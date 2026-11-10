"""The observation operator on the prior grid: nearest-cell lookup.

H is selection: each observation reads the state at the grid cell nearest its site, so it
is carried as a gather index into the flattened ``(channel, lat, lon)`` state.
"""

from __future__ import annotations

import numpy as np

from paleoreco.data import VARS


def nearest_lat_index(lat: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Index of the closest latitude. Poles do not wrap, so plain distance."""
    return np.argmin(np.abs(lats[None, :] - np.asarray(lat)[:, None]), axis=1)


def nearest_lon_index(lon: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Index of the closest longitude under circular distance.

    Longitude wraps at +-180, so a site at -178.8 is one step from the +180 cell.
    """
    d = ((lons[None, :] - np.asarray(lon)[:, None] + 180.0) % 360.0) - 180.0
    return np.argmin(np.abs(d), axis=1)


def nearest_age_index(age: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Index of the closest age on the sorted ``ages`` axis."""
    age = np.asarray(age)
    pos = np.clip(np.searchsorted(ages, age), 1, len(ages) - 1)
    left, right = ages[pos - 1], ages[pos]
    return np.where(age - left <= right - age, pos - 1, pos)


def obs_cell_index(
    lat: np.ndarray,
    lon: np.ndarray,
    channel: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Gather index of each observation into the flattened ``(channel, lat, lon)`` state."""
    ic = np.array([VARS.index(c) for c in channel])
    ilat = nearest_lat_index(lat, lats)
    ilon = nearest_lon_index(lon, lons)
    n_lat, n_lon = len(lats), len(lons)
    return ic * (n_lat * n_lon) + ilat * n_lon + ilon
