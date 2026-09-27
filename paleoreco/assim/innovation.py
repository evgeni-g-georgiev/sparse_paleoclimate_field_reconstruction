"""Observation operator on the prior grid, innovations and their standardisation.

H is nearest-cell selection: each observation reads the state at the grid cell nearest its
site. Under ``normalised`` scoring H also carries a per-row factor ``sigma_x / sigma_y``, with
``sigma_x = sqrt(diag B)``.
"""

from __future__ import annotations

import numpy as np

from paleoreco.data import VARS
from paleoreco.assim.scoring import NORMALISED


# ---------------------------------------------------------------------------
# Grid lookup.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Innovation and standardisation.
# ---------------------------------------------------------------------------


def obs_operator_scale(prior_var_at_obs: np.ndarray, sy: np.ndarray, mode: str) -> np.ndarray:
    """Per-row factor applied by H to the selected state cell.

    ``1`` under raw/anomaly scoring, ``sigma_x / sigma_y`` under normalised.
    """
    if mode == NORMALISED:
        return np.sqrt(prior_var_at_obs) / sy
    return np.ones_like(prior_var_at_obs, dtype=float)


def innovation(
    y_scored: np.ndarray,
    state_scored: np.ndarray,
    gather: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Observation minus mapped background, ``y - scale * H(x_b)``, in the scored space."""
    return y_scored - scale * state_scored[gather]


def predicted_sd(
    prior_var_at_obs: np.ndarray,
    sse: np.ndarray,
    sy: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Predicted innovation sd ``sqrt((H B H^T)_rr + R_rr)`` under the given scoring.

    Under normalised scoring the sd is further divided by ``sigma_y``.
    """
    base = np.sqrt(prior_var_at_obs + sse)
    return base / sy if mode == NORMALISED else base


def standardise(d: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardised innovation ``z = d / sd``, distributed N(0,1) under the assumptions."""
    return d / sd
