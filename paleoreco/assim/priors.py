"""Background error covariance B and the climatology, built from prior states.

B is the sample covariance of the prior anomalies. With ~800 states over
D = 2*n_lat*n_lon ~ 4000 cells its off-diagonals carry heavy sampling noise, so
this module regularizes B with composable, diagonal-preserving Schur tapers:
Gaspari-Cohn localization, uniform shrinkage toward the diagonal, and a
cross-channel coupling taper. Each is a Hadamard mask with unit diagonal, so any
combination preserves the well-estimated variances and stays symmetric PSD.

The state vector flattens one snapshot ``(2, n_lat, n_lon)`` in C order, channel
``mtco`` then ``mtwa``; distances between two state entries depend only on their
grid cell, so the spatial tapers are built per cell and tiled across channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paleoreco.assim.background import background_covariance
from paleoreco.data import compute_zscore_stats

_EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Prior:
    """A built background: covariance, climatology, mask, and source ages.

    ``B`` is ``(D, D)``; ``clim_mean`` is ``(2, n_lat, n_lon)``; ``safe_valid``
    masks degenerate cells; ``ages`` are the prior ages (yr BP) the per-age
    background neighbour is drawn from. ``meta`` records the taper choice.
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
    localization_km: float | None = None,
    shrinkage_lambda: float = 0.0,
    alpha: float = 1.0,
) -> Prior:
    """Build B and the climatology from ``prior_age_indices`` only.

    ``localization_km``, ``shrinkage_lambda`` and ``alpha`` select composable,
    diagonal-preserving tapers of the sample covariance (see
    :func:`regularization_mask`); the defaults leave B as the raw sample
    covariance. The climatology and mask come from the same prior ages so the
    anomaly frame is internally consistent.
    """
    prior_age_indices = np.asarray(prior_age_indices, dtype=np.int64)
    stats = compute_zscore_stats(cube, prior_age_indices, valid)
    clim_mean = stats["mean"]
    safe_valid = stats["safe_valid"]

    B = background_covariance(cube, prior_age_indices)
    mask = regularization_mask(lats, lons, localization_km=localization_km,
                               shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    if mask is not None:
        B = B * mask

    meta = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
            "alpha": alpha, "n_prior_ages": int(len(prior_age_indices))}
    return Prior(B=B, clim_mean=clim_mean, safe_valid=safe_valid,
                 ages=np.asarray(ages, dtype=np.int64)[prior_age_indices], meta=meta)


# ---------------------------------------------------------------------------
# Regularizers: diagonal-preserving Schur (Hadamard) tapers of B.
# ---------------------------------------------------------------------------
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


def great_circle_km_between(
    lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray
) -> np.ndarray:
    """Great-circle distances (km) from each point of A to each of B, ``(n_a, n_b)``.

    Cell-to-site geometry needs only this rectangular block. Reaching it by taking the
    pairwise matrix over the concatenated sets also computes the cell-cell quadrant that
    is then discarded, which is nearly all of the work when the grid greatly outnumbers
    the observation network.
    """
    # Promote each coordinate pair to a common dtype before converting to radians: grid
    # axes are float32 and site coordinates float64, and rounding the grid to float32
    # radians first would shift the distances in their last bits.
    lat_t, lon_t = np.result_type(lat_a, lat_b), np.result_type(lon_a, lon_b)
    la = np.radians(np.asarray(lat_a, dtype=lat_t))
    lb = np.radians(np.asarray(lat_b, dtype=lat_t))
    lo_a = np.radians(np.asarray(lon_a, dtype=lon_t))
    lo_b = np.radians(np.asarray(lon_b, dtype=lon_t))
    dlat = la[:, None] - lb[None, :]
    dlon = lo_a[:, None] - lo_b[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(lb)[None, :] * np.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def great_circle_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km) between cells given by lat/lon (deg)."""
    return great_circle_km_between(lat, lon, lat, lon)


def localization_taper(lats: np.ndarray, lons: np.ndarray, length_km: float) -> np.ndarray:
    """Full ``(D, D)`` localization taper: a per-cell Gaspari-Cohn spatial taper
    tiled over channels.

    The flattened spatial axis runs lat-major, lon-minor (cell s is lat ``s //
    n_lon``, lon ``s % n_lon``), matching the cube's C-order ravel.
    """
    lat_cell = np.repeat(lats, len(lons))
    lon_cell = np.tile(lons, len(lats))
    spatial = gaspari_cohn(great_circle_km(lat_cell, lon_cell), length_km)
    return np.block([[spatial, spatial], [spatial, spatial]])


def coupling_taper(n_lat: int, n_lon: int, alpha: float) -> np.ndarray:
    """Full ``(D, D)`` cross-channel coupling taper: within-channel blocks are 1,
    the two cross-channel blocks scaled by ``alpha``.

    ``alpha=1`` is a no-op (full coupling), ``alpha=0`` decouples the channels
    (block-diagonal B). It is ``[[1, alpha], [alpha, 1]]`` tiled over cells, PSD
    for ``alpha`` in [-1, 1].
    """
    n = n_lat * n_lon
    ones = np.ones((n, n))
    return np.block([[ones, alpha * ones], [alpha * ones, ones]])


def shrinkage_taper(n_lat: int, n_lon: int, lam: float) -> np.ndarray:
    """Full ``(D, D)`` uniform shrinkage taper: off-diagonals scaled by ``1 - lam``,
    diagonal 1. PSD for ``lam`` in [0, 1].
    """
    d = 2 * n_lat * n_lon
    mask = np.full((d, d), 1.0 - lam)
    np.fill_diagonal(mask, 1.0)
    return mask


def regularization_mask(
    lats: np.ndarray, lons: np.ndarray, *,
    localization_km: float | None, shrinkage_lambda: float, alpha: float,
) -> np.ndarray | None:
    """Composite diagonal-preserving Schur taper, or ``None`` if all off.

    Hadamard-composes the active tapers (localization, channel coupling, uniform
    shrinkage). Each factor is PSD with unit diagonal, so the product is a valid
    correlation taper that preserves B's variances; the factors commute, so the
    result is order-independent. Returns ``None`` when no taper is active so the
    caller can skip the multiply.
    """
    if localization_km is None and shrinkage_lambda == 0.0 and alpha == 1.0:
        return None
    n_lat, n_lon = len(lats), len(lons)
    mask = None
    if localization_km is not None:
        mask = localization_taper(lats, lons, localization_km)
    if alpha != 1.0:
        c = coupling_taper(n_lat, n_lon, alpha)
        mask = c if mask is None else mask * c
    if shrinkage_lambda != 0.0:
        s = shrinkage_taper(n_lat, n_lon, shrinkage_lambda)
        mask = s if mask is None else mask * s
    return mask


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------
