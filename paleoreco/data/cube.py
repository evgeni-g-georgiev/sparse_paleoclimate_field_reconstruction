"""Data loading, caching, and per-cell statistics for the Prior cube.

Turns ``Prior.csv`` (~1.6M rows in long format) into a dense
``(N_ages, 2, n_lat, n_lon)`` cube of ``[mtco, mtwa]`` channels and caches it as
``.npz`` for reuse. The Prior here is a LOVECLIM transient climate simulation;
the loader is agnostic to the engine.

Key invariants
--------------
* The Prior has no missing cells; the only mask that does anything is
  ``safe_valid``, which drops cells with degenerate std (e.g. permanent ice).
* Per-cell stats use **train ages only** to avoid leakage; ``mean`` centres
  the cube to anomaly, and ``std`` is kept for the assimilation's normalised path.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Constants describing the Prior grid and channel order.
# ----------------------------------------------------------------------------
GRID_SHAPE: tuple[int, int] = (32, 64)  # (n_lat, n_lon) at 5.625-degree resolution
VARS: tuple[str, str] = ("mtco", "mtwa")  # channel order convention used package-wide


# ----------------------------------------------------------------------------
# Prior cube construction.
# ----------------------------------------------------------------------------
def build_prior_cube(
    prior_csv: str = "data/Prior.csv",
    cache_path: str | None = "data/cache/prior_cube.npz",
    force_rebuild: bool = False,
) -> dict:
    """Pivot ``Prior.csv`` into a dense (N_ages, 2, n_lat, n_lon) cube.

    Parameters
    ----------
    prior_csv : str
        Path to the raw Prior CSV.
    cache_path : str or None
        Where to read/write the cached npz; ``None`` disables caching.
    force_rebuild : bool
        Rebuild from CSV even if a cache exists.

    Returns
    -------
    dict with keys:
        cube  : (N_ages, 2, n_lat, n_lon) float32, channels (mtco, mtwa).
        ages  : (N_ages,) int64, sorted ascending (yr BP).
        lats  : (n_lat,) float32, sorted ascending.
        lons  : (n_lon,) float32, sorted ascending.
        valid : (n_lat, n_lon) bool. True where the cube is finite for
                every age and both channels.
    """
    # Use the cache when it is available and no rebuild was asked for.
    if cache_path is not None and os.path.exists(cache_path) and not force_rebuild:
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    # usecols keeps memory bounded on the 1.6M-row CSV.
    df = pd.read_csv(prior_csv, usecols=["lon", "lat", "age", "mtco", "mtwa"])

    # Sorted unique axes define the cube's coordinate system.
    ages = np.sort(df["age"].unique())
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())

    # Vectorised index lookup via searchsorted into the sorted unique axes.
    age_idx = np.searchsorted(ages, df["age"].to_numpy())
    lat_idx = np.searchsorted(lats, df["lat"].to_numpy())
    lon_idx = np.searchsorted(lons, df["lon"].to_numpy())

    n_ages, n_lat, n_lon = len(ages), len(lats), len(lons)
    cube = np.full((n_ages, 2, n_lat, n_lon), np.nan, dtype=np.float32)
    cube[age_idx, 0, lat_idx, lon_idx] = df["mtco"].to_numpy(dtype=np.float32)
    cube[age_idx, 1, lat_idx, lon_idx] = df["mtwa"].to_numpy(dtype=np.float32)

    # Fail loud rather than silently zero-fill: a real 0 °C reading is
    # indistinguishable from a missing cell that defaulted to NaN -> 0.
    n_missing = int(np.isnan(cube).sum())
    if n_missing:
        raise ValueError(
            f"Prior cube has {n_missing} missing (age, channel, lat, lon) cells. "
            "Decide explicitly how to handle this before proceeding."
        )

    # A cell is geographically "valid" iff both channels are finite for every age.
    # For the Prior this is uniformly True (see verify_mask_constant_across_ages).
    valid = np.isfinite(cube).all(axis=(0, 1))

    result = {
        "cube": cube,
        "ages": ages.astype(np.int64),
        "lats": lats.astype(np.float32),
        "lons": lons.astype(np.float32),
        "valid": valid,
    }

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **result)

    return result


def compute_zscore_stats(
    cube: np.ndarray,
    train_age_indices: Sequence[int] | np.ndarray,
    valid: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """Compute per-cell mean and std from train ages only.

    Cells with degenerate std (below ``eps``) on either channel are
    excluded from the training mask. This handles the "permanent ice
    cell with no variability" case.

    Parameters
    ----------
    cube : (N_ages, 2, n_lat, n_lon) float32
    train_age_indices
        Indices into the N_ages axis (not ages themselves).
    valid : (n_lat, n_lon) bool
        Per-cell geographic validity from ``build_prior_cube``.
    eps : float
        Std threshold below which a cell is treated as degenerate.

    Returns
    -------
    dict with keys:
        mean       : (2, n_lat, n_lon) float32. Per-cell train mean.
        std        : (2, n_lat, n_lon) float32. Per-cell train std,
                     clamped to 1.0 on masked cells so division stays safe.
        safe_valid : (n_lat, n_lon) bool. The loss/mask channel used
                     downstream: geographically valid AND both channels
                     have non-degenerate variability.
    """
    train_age_indices = np.asarray(train_age_indices, dtype=np.int64)
    sub = cube[train_age_indices]  # (n_train, 2, n_lat, n_lon)
    mean = sub.mean(axis=0)         # (2, n_lat, n_lon)
    std = sub.std(axis=0)           # (2, n_lat, n_lon)

    # A cell is degenerate if EITHER channel has near-zero std on train ages.
    degenerate = (std < eps).any(axis=0)              # (n_lat, n_lon)
    safe_valid = valid & ~degenerate                  # (n_lat, n_lon)

    # Where the cell is masked, replace std with 1.0 so (x - mean) / std doesn't
    # explode. The mask channel will zero those cells out anyway.
    std_safe = np.where(safe_valid[None], std, 1.0).astype(np.float32)

    return {
        "mean": mean.astype(np.float32),
        "std": std_safe,
        "safe_valid": safe_valid,
    }
