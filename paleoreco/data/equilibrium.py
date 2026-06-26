"""Constant-CO2 CCSM4 glacial run: load it, reduce to per-decade temperature fields,
and build a truth cube on a target grid.

The run (Vettoretti et al. 2022) holds external forcing fixed and oscillates
spontaneously between stadial and interstadial states, so its equilibrated
snapshots sample one stationary glacial climate distribution. The file stores
monthly decadal-mean climatologies of shape ``(12, n_decades, n_lat, n_lon)`` in
Kelvin; each decade is one climate state.

The 12-month axis describes each state, it is not a sample axis: January and July
are fixed points of the deterministic seasonal cycle, not independent draws.
Reducing it to one field per decade removes that seasonal cycle and leaves the
decade-to-decade variability the analysis is about.
"""

from __future__ import annotations

import numpy as np

from .regrid import conservative_regrid

_KELVIN = 273.15

# Collapse the 12 monthly means to one field per decade. min/max recover the
# mean-temperature-of-coldest/warmest-month channels (MTCO/MTWA).
_REDUCERS = {
    "annual_mean": lambda t: t.mean(axis=0),
    "mtco": lambda t: t.min(axis=0),
    "mtwa": lambda t: t.max(axis=0),
}


def load_equilibrium_run(npz_path: str) -> dict:
    """Load the run's arrays and metadata from the extracted ``.npz``."""
    with np.load(npz_path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def state_fields(run: dict, var: str = "TREFHT", reduce: str = "annual_mean") -> np.ndarray:
    """Per-decade temperature field in degC, shape ``(n_decades, n_lat, n_lon)``.

    ``var`` selects a stored channel (``TREFHT`` is 2 m air temperature, the basis
    for MTCO/MTWA); ``reduce`` collapses the 12-month axis (``annual_mean``,
    ``mtco``, or ``mtwa``).
    """
    if reduce not in _REDUCERS:
        raise ValueError(f"unknown reduce {reduce!r}; expected one of {tuple(_REDUCERS)}")
    monthly = np.asarray(run[var], dtype=np.float64)  # (12, n_dec, n_lat, n_lon)
    return (_REDUCERS[reduce](monthly) - _KELVIN).astype(np.float32)


def truth_cube_on_grid(
    run: dict, tgt_lat: np.ndarray, tgt_lon: np.ndarray, *,
    var: str = "TREFHT", trim_decades: int = 100, stride: int = 25,
) -> np.ndarray:
    """MTCO/MTWA truth cube ``(n, 2, n_lat, n_lon)`` in degC on the target grid.

    Regrids the 12 monthly fields per decade before the min/max reduction, so the
    coldest/warmest-month channels are formed on the target grid the way a coarse
    model's own MTCO/MTWA would be. ``trim_decades`` drops the spin-up head and
    ``stride`` thins to near-independent decades; channel order is
    :data:`paleoreco.data.VARS`.
    """
    monthly = np.asarray(run[var], dtype=np.float64) - _KELVIN     # (12, n_dec, n_lat, n_lon)
    monthly = monthly[:, trim_decades::stride]                     # (12, n, n_lat, n_lon)
    regridded = conservative_regrid(monthly, run["lat"], run["lon"], tgt_lat, tgt_lon)
    mtco, mtwa = regridded.min(axis=0), regridded.max(axis=0)      # (n, n_lat, n_lon) each
    return np.stack([mtco, mtwa], axis=1).astype(np.float32)
