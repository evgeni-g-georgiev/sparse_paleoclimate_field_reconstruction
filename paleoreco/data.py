"""Data loading, caching, and z-score normalisation for the Prior cube.

Turns ``Prior.csv`` (~1.6M rows in long format) into a dense
``(N_ages, 2, n_lat, n_lon)`` cube of ``[mtco, mtwa]`` channels, caches
the cube as ``.npz`` for reuse, and serves PyTorch tensors of shape
``(3, n_lat, n_lon)`` whose third channel is a binary valid-mask.

The Prior here is a LOVECLIM transient climate simulation; the loader
is agnostic to the engine.

Key invariants
--------------
* The Prior has no missing cells; the only mask that does anything is
  ``safe_valid``, which drops cells with degenerate std (e.g. permanent ice).
* Z-score stats use **train ages only** to avoid leakage; ``mean`` and
  ``std`` are returned so callers can invert at inference.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
    # Use the cache when it's available and the caller hasn't asked us to refresh.
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


def verify_mask_constant_across_ages(cube: np.ndarray) -> bool:
    """Assert per-age finite-masks all match age 0's.

    Tautological under the current contract: ``build_prior_cube`` already
    refuses to build a cube with any NaN. Kept as explicit documentation
    of the "mask is constant in time" assumption for future data sources.
    """
    per_age = np.isfinite(cube).all(axis=1)  # (N_ages, n_lat, n_lon)
    constant = bool((per_age == per_age[0:1]).all())
    if not constant:
        n_differing = int((per_age != per_age[0:1]).any(axis=(1, 2)).sum())
        print(
            f"WARNING: per-age valid mask differs from age 0 on "
            f"{n_differing} / {len(per_age)} ages; the constant-mask "
            "assumption needs revisiting."
        )
    return constant


# ----------------------------------------------------------------------------
# Per-cell z-score statistics.
# ----------------------------------------------------------------------------
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


def apply_zscore(cube: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score the cube per cell and zero out masked cells.

    Output shape matches ``cube``. Zeroing out masked cells means the
    AE never sees the raw values on degenerate/invalid cells, and the
    binary mask channel tells it where those zeros are real vs filled.
    """
    z = (cube - stats["mean"]) / stats["std"]
    mask = stats["safe_valid"].astype(np.float32)  # broadcasts (n_lat, n_lon) over leading dims
    return (z * mask).astype(np.float32)


def invert_zscore(cube_z: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of ``apply_zscore``: returns absolute °C (anomaly + mean).

    Masked cells return ``mean[c, i, j]`` because they were zero-filled.
    Always inspect outputs through the same mask.
    """
    return (cube_z * stats["std"] + stats["mean"]).astype(np.float32)


# ----------------------------------------------------------------------------
# PyTorch dataset.
# ----------------------------------------------------------------------------
class PaleoFieldDataset(Dataset):
    """Serves (3, n_lat, n_lon) tensors: [mtco_z, mtwa_z, valid_mask].

    Parameters
    ----------
    cube_z : np.ndarray, shape (N_ages, 2, n_lat, n_lon), float32
        Full z-scored cube (output of ``apply_zscore``).
    mask : np.ndarray, shape (n_lat, n_lon), bool
        ``safe_valid`` mask. Becomes the third input channel and is also
        used by ``masked_mse`` for the loss.
    age_indices : np.ndarray of int
        Indices into the N_ages axis that belong to this split
        (train / val / test).
    """

    def __init__(
        self,
        cube_z: np.ndarray,
        mask: np.ndarray,
        age_indices: Sequence[int] | np.ndarray,
    ) -> None:
        if cube_z.ndim != 4 or cube_z.shape[1] != 2:
            raise ValueError(
                f"cube_z must have shape (N_ages, 2, n_lat, n_lon); got {cube_z.shape}"
            )
        self.cube_z = np.ascontiguousarray(cube_z, dtype=np.float32)
        self.mask = np.ascontiguousarray(mask, dtype=bool)
        self.age_indices = np.asarray(age_indices, dtype=np.int64)
        # Pre-compute the mask tensor once; it's shared across samples in v1.
        self._mask_chan = torch.from_numpy(self.mask.astype(np.float32)).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.age_indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        a = int(self.age_indices[i])
        field = torch.from_numpy(self.cube_z[a])  # (2, n_lat, n_lon)
        return torch.cat([field, self._mask_chan], dim=0)  # (3, n_lat, n_lon)
