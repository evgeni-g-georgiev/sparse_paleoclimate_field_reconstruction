"""Background error covariance B and the climatology, built from prior states.

B is the sample covariance of the prior anomalies. With ~800 states over
D = 2*n_lat*n_lon ~ 4000 cells its off-diagonals carry heavy sampling noise, so
this module exposes B regularization as a choice (raw / EOF truncation / diagonal
shrinkage / Gaspari-Cohn localization) rather than a fixed transform. The
variances (diagonal) are estimated well and every variant preserves them.

The state vector flattens one snapshot ``(2, n_lat, n_lon)`` in C order, channel
``mtco`` then ``mtwa``; distances between two state entries depend only on their
grid cell, so the localization taper is built per cell and shared across channels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from paleoreco.assim.background import background_covariance
from paleoreco.data import compute_zscore_stats

_EARTH_RADIUS_KM = 6371.0
RAW, EOF, SHRINKAGE, LOCALIZATION = "raw", "eof", "shrinkage", "localization"


@dataclass(frozen=True)
class Prior:
    """A built background: covariance, climatology, mask, and source ages.

    ``B`` is ``(D, D)``; ``clim_mean`` is ``(2, n_lat, n_lon)``; ``safe_valid``
    masks degenerate cells; ``ages`` are the prior ages (yr BP) the per-age
    background neighbour is drawn from. ``meta`` records the regularizer.
    """

    B: np.ndarray
    clim_mean: np.ndarray
    safe_valid: np.ndarray
    ages: np.ndarray
    meta: dict


# ---------------------------------------------------------------------------
# Construction.
# ---------------------------------------------------------------------------
def build_prior(
    cube: np.ndarray,
    ages: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    prior_age_indices: np.ndarray,
    valid: np.ndarray,
    *,
    b_reg: str = RAW,
    eof_rank: int | None = None,
    localization_km: float | None = None,
    shrinkage_lambda: float | None = None,
) -> Prior:
    """Build B and the climatology from ``prior_age_indices`` only.

    ``b_reg`` selects the regularizer; ``eof_rank``, ``localization_km`` and
    ``shrinkage_lambda`` are read by their respective variants
    (``shrinkage_lambda`` overrides the analytic shrinkage intensity when set).
    The climatology and mask come from the same prior ages so the anomaly frame is
    internally consistent.
    """
    prior_age_indices = np.asarray(prior_age_indices, dtype=np.int64)
    stats = compute_zscore_stats(cube, prior_age_indices, valid)
    clim_mean = stats["mean"]
    safe_valid = stats["safe_valid"]

    B = background_covariance(cube, prior_age_indices)
    if b_reg == RAW:
        pass
    elif b_reg == EOF:
        if eof_rank is None:
            raise ValueError("b_reg='eof' needs eof_rank")
        B = eof_truncate(B, eof_rank)
    elif b_reg == SHRINKAGE:
        X = cube[prior_age_indices].reshape(len(prior_age_indices), -1).astype(np.float64)
        B = shrink_to_diagonal(X, shrinkage_lambda)
    elif b_reg == LOCALIZATION:
        if localization_km is None:
            raise ValueError("b_reg='localization' needs localization_km")
        B = B * channel_taper(lats, lons, localization_km)
    else:
        raise ValueError(f"unknown b_reg {b_reg!r}")

    meta = {"b_reg": b_reg, "eof_rank": eof_rank, "localization_km": localization_km,
            "shrinkage_lambda": shrinkage_lambda, "n_prior_ages": int(len(prior_age_indices))}
    return Prior(B=B, clim_mean=clim_mean, safe_valid=safe_valid,
                 ages=np.asarray(ages, dtype=np.int64)[prior_age_indices], meta=meta)


# ---------------------------------------------------------------------------
# Regularizers.
# ---------------------------------------------------------------------------
def eof_truncate(B: np.ndarray, rank: int) -> np.ndarray:
    """Rank-``rank`` PSD approximation keeping B's leading eigenpairs.

    The leading modes carry the large-scale structure; dropping the tail removes
    the sampling-noise-dominated small scales.
    """
    w, V = np.linalg.eigh(B)
    keep = V[:, -rank:] * np.sqrt(np.clip(w[-rank:], 0.0, None))
    return keep @ keep.T


def shrink_to_diagonal(X: np.ndarray, lam: float | None = None) -> np.ndarray:
    """Schaefer-Strimmer (2005) shrinkage of correlations toward zero.

    Variances are kept exact; off-diagonals are scaled by ``1 - lambda``. ``lam``
    overrides the analytic optimal intensity when given. ``X`` is ``(n_states, D)``
    of raw states.
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc.T @ Xc                       # sum_k w_ijk
    s = G / (n - 1)                     # unbiased sample covariance

    if lam is None:
        sumsq = (Xc ** 2).T @ (Xc ** 2)     # sum_k w_ijk^2
        var_s = (n / (n - 1) ** 3) * (sumsq - G ** 2 / n)
        off = ~np.eye(len(s), dtype=bool)
        lam = var_s[off].sum() / (s[off] ** 2).sum()
    lam = float(np.clip(lam, 0.0, 1.0))

    B = (1.0 - lam) * s
    np.fill_diagonal(B, np.diag(s))
    return B


def gaspari_cohn(dist: np.ndarray, length_km: float) -> np.ndarray:
    """Gaspari-Cohn 5th-order taper, support ``[0, 2*length_km]``."""
    z = np.asarray(dist, dtype=np.float64) / length_km
    out = np.zeros_like(z)
    near = z <= 1.0
    mid = (z > 1.0) & (z <= 2.0)
    zn = z[near]
    out[near] = -0.25 * zn**5 + 0.5 * zn**4 + 0.625 * zn**3 - 5.0 / 3.0 * zn**2 + 1.0
    zm = z[mid]
    out[mid] = (zm**5 / 12.0 - 0.5 * zm**4 + 0.625 * zm**3 + 5.0 / 3.0 * zm**2
                - 5.0 * zm + 4.0 - 2.0 / (3.0 * zm))
    return out


def great_circle_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km) between cells given by lat/lon (deg)."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def channel_taper(lats: np.ndarray, lons: np.ndarray, length_km: float) -> np.ndarray:
    """Full ``(D, D)`` Schur taper: a per-cell spatial taper tiled over channels.

    The flattened spatial axis runs lat-major, lon-minor (cell s is lat ``s //
    n_lon``, lon ``s % n_lon``), matching the cube's C-order ravel.
    """
    lat_cell = np.repeat(lats, len(lons))
    lon_cell = np.tile(lons, len(lats))
    spatial = gaspari_cohn(great_circle_km(lat_cell, lon_cell), length_km)
    return np.block([[spatial, spatial], [spatial, spatial]])


# ---------------------------------------------------------------------------
# Per-age background neighbour (PPE timing-error offset).
# ---------------------------------------------------------------------------
def per_age_neighbour(truth_age: int, prior_ages: np.ndarray, offset_yr: int = 2000) -> int:
    """Closest prior age at least ``offset_yr`` from ``truth_age``.

    The offset stands in for proxy dating uncertainty and sits at the prior's
    temporal-decorrelation scale, so the neighbour is a genuine imperfect
    time-specific background rather than a near-duplicate of the truth.
    """
    d = np.abs(prior_ages - truth_age)
    eligible = d >= offset_yr
    if not eligible.any():
        eligible = d == d.max()
    cand = np.flatnonzero(eligible)
    return int(prior_ages[cand[np.argmin(d[cand])]])


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------
def save_prior(path: str, prior: Prior) -> None:
    """Save a built prior to ``path`` (npz)."""
    np.savez_compressed(path, B=prior.B, clim_mean=prior.clim_mean,
                        safe_valid=prior.safe_valid, ages=prior.ages,
                        meta=json.dumps(prior.meta))


def load_prior(path: str) -> Prior:
    """Load a prior saved by :func:`save_prior`."""
    with np.load(path, allow_pickle=False) as z:
        return Prior(B=z["B"], clim_mean=z["clim_mean"], safe_valid=z["safe_valid"],
                     ages=z["ages"], meta=json.loads(str(z["meta"])))
