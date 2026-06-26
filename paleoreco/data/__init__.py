"""Data substrate: the Prior cube, CV splits, regridding, and the truth run.

The cube loader and its z-score helpers are the heavily-used public surface, so
they are re-exported here: ``from paleoreco.data import build_prior_cube`` (and
the other cube symbols) resolves directly. The other concerns stay as explicit
submodules to keep this namespace focused:

* :mod:`paleoreco.data.cube`        - Prior.csv -> dense cube, z-score, Dataset.
* :mod:`paleoreco.data.splits`      - train/val/test and blocked CV over the age axis.
* :mod:`paleoreco.data.regrid`      - conservative rectilinear regridding.
* :mod:`paleoreco.data.equilibrium` - CCSM4 constant-CO2 truth cube on a target grid.
"""

from __future__ import annotations

from .cube import (
    GRID_SHAPE,
    VARS,
    PaleoFieldDataset,
    apply_zscore,
    build_prior_cube,
    compute_zscore_stats,
    invert_zscore,
    verify_mask_constant_across_ages,
)

__all__ = [
    "build_prior_cube",
    "compute_zscore_stats",
    "apply_zscore",
    "invert_zscore",
    "PaleoFieldDataset",
    "verify_mask_constant_across_ages",
    "VARS",
    "GRID_SHAPE",
]
