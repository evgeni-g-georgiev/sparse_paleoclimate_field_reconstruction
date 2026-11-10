"""Loading, caching and per-cell statistics for the prior cube.

Turns ``Prior.csv`` (about 1.6M rows in long format) into a dense
``(N_ages, 2, n_lat, n_lon)`` cube of ``[mtco, mtwa]`` channels, cached as ``.npz``. The
prior has no missing cells; ``safe_valid`` only drops cells with degenerate variability.
Per-cell statistics are computed over a given subset of ages, so a held-out half never
informs them.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd


# Channel order, used package-wide.
VARS: tuple[str, str] = ("mtco", "mtwa")


def build_prior_cube(
    prior_csv: str = "data/Prior.csv",
    cache_path: str | None = "data/cache/prior_cube.npz",
    force_rebuild: bool = False,
) -> dict:
    """Pivot ``Prior.csv`` into a dense ``(N_ages, 2, n_lat, n_lon)`` cube.

    Returns a dict of ``cube`` (float32), ascending ``ages`` (yr BP), ``lats`` and ``lons``,
    and ``valid``, true where a cell is finite at every age. ``cache_path=None`` disables
    caching.
    """
    if cache_path is not None and os.path.exists(cache_path) and not force_rebuild:
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    df = pd.read_csv(prior_csv, usecols=["lon", "lat", "age", "mtco", "mtwa"])

    ages = np.sort(df["age"].unique())
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())

    age_idx = np.searchsorted(ages, df["age"].to_numpy())
    lat_idx = np.searchsorted(lats, df["lat"].to_numpy())
    lon_idx = np.searchsorted(lons, df["lon"].to_numpy())

    n_ages, n_lat, n_lon = len(ages), len(lats), len(lons)
    cube = np.full((n_ages, 2, n_lat, n_lon), np.nan, dtype=np.float32)
    cube[age_idx, 0, lat_idx, lon_idx] = df["mtco"].to_numpy(dtype=np.float32)
    cube[age_idx, 1, lat_idx, lon_idx] = df["mtwa"].to_numpy(dtype=np.float32)

    # Fail rather than zero-fill: a filled cell would be indistinguishable from 0 °C.
    n_missing = int(np.isnan(cube).sum())
    if n_missing:
        raise ValueError(
            f"Prior cube has {n_missing} missing (age, channel, lat, lon) cells. "
            "Decide explicitly how to handle this before proceeding."
        )

    # True everywhere for this prior, which has no missing cells.
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


def cell_climatology(
    cube: np.ndarray,
    age_indices: Sequence[int] | np.ndarray,
    valid: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """Per-cell ``mean`` and ``safe_valid`` over the ages at ``age_indices``.

    ``safe_valid`` drops cells whose std on either channel is below ``eps``, such as
    permanent ice, since a cell that never varies has no covariance to assimilate into.
    """
    sub = cube[np.asarray(age_indices, dtype=np.int64)]   # (n_ages, 2, n_lat, n_lon)
    degenerate = (sub.std(axis=0) < eps).any(axis=0)    # (n_lat, n_lon)
    return {"mean": sub.mean(axis=0).astype(np.float32), "safe_valid": valid & ~degenerate}
