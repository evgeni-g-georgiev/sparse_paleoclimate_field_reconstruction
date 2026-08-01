"""Constant-CO2 CCSM4 glacial run: load it, reduce its month axis, and put it on
another model's grid.

The run (Vettoretti et al. 2022) holds external forcing fixed and oscillates
spontaneously between stadial and interstadial states, so its equilibrated
snapshots sample one stationary glacial climate distribution. The file stores
monthly decadal-mean climatologies of shape ``(12, n_decades, n_lat, n_lon)`` in
Kelvin; each decade is one climate state.

The 12-month axis describes each state, it is not a sample axis: January and July
are fixed points of the deterministic seasonal cycle, not independent draws.
Reducing it to one field per decade removes that seasonal cycle and leaves the
decade-to-decade variability the analysis is about.

Two reductions to a cold/warm pair live here, and they are different statistics:

* *coldest month* and *warmest month*, the min and max over the 12 monthly means.
  These are the bioclimate MTCO and MTWA.
* *the prior's channels*, the solstitial 3-month means assigned by hemisphere.
  This is what the LOVECLIM ``mtco``/``mtwa`` fields actually hold, so it is the
  pair to build when the two models have to be comparable.
"""

from __future__ import annotations

import numpy as np

from .regrid import conservative_regrid

_KELVIN = 273.15

# The run is on a 365-day no-leap calendar, so these month lengths are exact.
_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
_DJF = (11, 0, 1)
_JJA = (5, 6, 7)

# Collapse the 12 monthly means to one field per decade.
_REDUCERS = {
    "annual_mean": lambda t: t.mean(axis=0),
    "coldest_month": lambda t: t.min(axis=0),
    "warmest_month": lambda t: t.max(axis=0),
}


def load_equilibrium_run(npz_path: str) -> dict:
    """Load the run's arrays and metadata from the extracted ``.npz``."""
    with np.load(npz_path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _monthly_degc(run: dict, var: str) -> np.ndarray:
    """Stored monthly climatologies in degC, ``(12, n_decades, n_lat, n_lon)``."""
    return np.asarray(run[var], dtype=np.float64) - _KELVIN


def state_fields(run: dict, var: str = "TREFHT", reduce: str = "annual_mean") -> np.ndarray:
    """Per-decade temperature field in degC, shape ``(n_decades, n_lat, n_lon)``.

    ``var`` selects a stored channel (``TREFHT`` is 2 m air temperature); ``reduce``
    collapses the 12-month axis (``annual_mean``, ``coldest_month``, ``warmest_month``).
    The month extremes are the bioclimate MTCO/MTWA, not the prior's channels;
    :func:`cube_on_grid` builds those.
    """
    if reduce not in _REDUCERS:
        raise ValueError(f"unknown reduce {reduce!r}; expected one of {tuple(_REDUCERS)}")
    return _REDUCERS[reduce](_monthly_degc(run, var)).astype(np.float32)


def seasonal_means(run: dict, var: str = "TREFHT") -> np.ndarray:
    """DJF and JJA means in degC, shape ``(2, n_decades, n_lat, n_lon)``.

    Weighted by month length rather than averaged flat: the two differ by around 5%
    of a per-cell anomaly standard deviation, which is not negligible against the
    decade-to-decade signal being measured. The season axis leads so the result
    feeds :func:`paleoreco.data.regrid.conservative_regrid` unchanged.
    """
    monthly = _monthly_degc(run, var)
    seasons = []
    for months in (_DJF, _JJA):
        weights = _MONTH_DAYS[list(months)]
        weights = weights / weights.sum()
        seasons.append(np.tensordot(weights, monthly[list(months)], axes=(0, 0)))
    return np.stack(seasons)


def assign_hemispheric_channels(seasonal: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """``(mtco, mtwa)`` state in degC, shape ``(n_decades, 2, n_lat, n_lon)``.

    The prior's cold channel is DJF north of the equator and JJA south of it, the
    warm channel the other way round. ``seasonal`` holds ``(DJF, JJA)`` on the grid
    ``lats`` describes.
    """
    seasonal = np.asarray(seasonal, dtype=np.float64)
    if seasonal.shape[0] != 2:
        raise ValueError(f"seasonal must lead with a (DJF, JJA) axis; got shape {seasonal.shape}")
    lats = np.asarray(lats)
    if len(lats) != seasonal.shape[-2]:
        raise ValueError(f"lats has {len(lats)} entries against {seasonal.shape[-2]} "
                         f"latitude cells in seasonal (shape {seasonal.shape})")
    djf, jja = seasonal
    north = (lats >= 0.0)[None, :, None]
    return np.stack([np.where(north, djf, jja), np.where(north, jja, djf)],
                    axis=1).astype(np.float32)


def cube_on_grid(
    run: dict, tgt_lat: np.ndarray, tgt_lon: np.ndarray, *, var: str = "TREFHT",
) -> np.ndarray:
    """Prior-channel cube in degC on the target grid, ``(n_decades, 2, n_lat, n_lon)``.

    The hemispheric assignment runs after the remap rather than before it. The cold
    channel steps discontinuously across the equator, so remapping it directly
    smears that step over the equatorial cells; DJF and JJA are each smooth and
    remap cleanly. Target coordinates are read as cell centres.
    """
    seasonal = seasonal_means(run, var)
    remapped = conservative_regrid(seasonal, run["lat"], run["lon"], tgt_lat, tgt_lon)
    return assign_hemispheric_channels(remapped, tgt_lat)
