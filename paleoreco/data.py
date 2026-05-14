"""Data loading, caching, and z-score normalisation for the Prior cube.

This module turns ``Prior.csv`` (~1.6M rows in long format) into a dense
``(N_ages, 2, n_lat, n_lon)`` cube of ``[mtco, mtwa]`` channels, caches it
to ``.npz`` so we don't pay the pivot cost more than once, and serves
PyTorch tensors of shape ``(3, n_lat, n_lon)`` whose third channel is a
binary valid-mask.

Key design notes
----------------
* The Prior is a LOVECLIM simulation: temperatures are defined on every
  grid cell, so we do not expect missing values. The valid mask comes
  out of normalisation rather than from missing data — cells whose
  per-cell standard deviation is degenerate (e.g. constant) are masked
  to keep the loss well-defined.
* Per-cell z-score statistics are computed from **train ages only** to
  avoid leakage. ``mean`` and ``std`` are saved alongside model
  checkpoints so we can invert at inference time.
* The valid mask is assumed constant across ages for v1. A helper
  ``verify_mask_constant_across_ages`` exposes this assumption to the
  EDA notebook so we can confirm or flag it.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# Constants describing the LOVECLIM grid and channel order.
# ----------------------------------------------------------------------------
GRID_SHAPE: tuple[int, int] = (32, 64)  # (n_lat, n_lon) at 5.625-degree resolution
VARS: tuple[str, str] = ("mtco", "mtwa")  # channel order — must match elsewhere


# ----------------------------------------------------------------------------
# Prior cube construction.
# ----------------------------------------------------------------------------
def build_prior_cube(
    prior_csv: str = "data/Prior.csv",
    cache_path: str | None = "data/cache/prior_cube.npz",
    force_rebuild: bool = False,
) -> dict:
    """Pivot ``Prior.csv`` into a dense (N_ages, 2, n_lat, n_lon) cube.

    Pivoting one age at a time with a Python-level ``iterrows`` loop 
    scales poorly to all 804 ages (~1.6M rows). Instead we sort the 
    unique lat/lon/age values, look up each row's integer index with 
    ``np.searchsorted``, and assign into the cube with vectorised 
    advanced indexing.

    Parameters
    ----------
    prior_csv : str
        Path to the raw Prior CSV.
    cache_path : str or None
        Where to read/write the cached npz. ``None`` disables caching.
    force_rebuild : bool
        If True, rebuild the cube from CSV even if a cache exists.

    Returns
    -------
    dict with keys:
        cube  : (N_ages, 2, n_lat, n_lon) float32 - channels (mtco, mtwa).
        ages  : (N_ages,) int64, sorted ascending (yr BP).
        lats  : (n_lat,) float32, sorted ascending.
        lons  : (n_lon,) float32, sorted ascending.
        valid : (n_lat, n_lon) bool — True where the cube is finite for
                every age and both channels.
    """
    # Use the cache when it's available and the caller hasn't asked us to refresh.
    if cache_path is not None and os.path.exists(cache_path) and not force_rebuild:
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    # Only load the four columns we actually need — saves memory on the 1.6M-row CSV.
    df = pd.read_csv(prior_csv, usecols=["lon", "lat", "age", "mtco", "mtwa"])

    # Sorted unique axes define the cube's coordinate system.
    ages = np.sort(df["age"].unique())
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())

    # `searchsorted` against the sorted axes is O(N log K) and fully vectorised —
    # much faster than the per-row np.where pattern in provided notebooks/read_data.ipynb.
    age_idx = np.searchsorted(ages, df["age"].to_numpy())
    lat_idx = np.searchsorted(lats, df["lat"].to_numpy())
    lon_idx = np.searchsorted(lons, df["lon"].to_numpy())

    n_ages, n_lat, n_lon = len(ages), len(lats), len(lons)
    cube = np.full((n_ages, 2, n_lat, n_lon), np.nan, dtype=np.float32)
    cube[age_idx, 0, lat_idx, lon_idx] = df["mtco"].to_numpy(dtype=np.float32)
    cube[age_idx, 1, lat_idx, lon_idx] = df["mtwa"].to_numpy(dtype=np.float32)

    # Contract: Prior.csv must cover every (age, lat, lon) exactly once. The
    # LOVECLIM run has 804 * 32 * 64 = 1,646,592 rows, matching this. If a
    # future dataset has gaps, we want to fail loudly here rather than silently
    # zero-fill — a synthetic 0 is indistinguishable from a real 0 C reading.
    n_missing = int(np.isnan(cube).sum())
    if n_missing:
        raise ValueError(
            f"Prior cube has {n_missing} missing (age, channel, lat, lon) cells. "
            "Decide explicitly how to handle this before proceeding."
        )

    # A cell is geographically "valid" iff both channels are finite for every age.
    # For LOVECLIM we expect this to be uniformly True; EDA verifies.
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
    """Sanity check the assumption that the valid mask is constant in time.

    Computes a per-age finite-mask and compares each age's mask to age 0.
    Prints a diagnostic if any age disagrees. Used in 01_eda.ipynb.
    """
    per_age = np.isfinite(cube).all(axis=1)  # (N_ages, n_lat, n_lon)
    constant = bool((per_age == per_age[0:1]).all())
    if not constant:
        n_differing = int((per_age != per_age[0:1]).any(axis=(1, 2)).sum())
        print(
            f"WARNING: per-age valid mask differs from age 0 on "
            f"{n_differing} / {len(per_age)} ages — v1's constant-mask "
            "assumption (§3.8) needs revisiting."
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
    cell with no variability" case described in the design.

    Parameters
    ----------
    cube : (N_ages, 2, n_lat, n_lon) float32
        Full prior cube (all ages).
    train_age_indices : int sequence
        Indices into the N_ages axis defining the training split.
    valid : (n_lat, n_lon) bool
        Geographic validity mask from ``build_prior_cube``.
    eps : float
        Threshold below which a cell's std is treated as degenerate.

    Returns
    -------
    dict with keys:
        mean       : (2, n_lat, n_lon) float32 — per-cell train mean.
        std        : (2, n_lat, n_lon) float32 — per-cell train std,
                     clamped to 1.0 on masked cells so division stays safe.
        safe_valid : (n_lat, n_lon) bool — the loss/mask channel used
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
    """Inverse of ``apply_zscore``: returns °C-anomaly values.

    Note: masked cells will return ``mean[c, i, j]`` because they were
    zero-filled. Always inspect outputs through the same mask.
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
