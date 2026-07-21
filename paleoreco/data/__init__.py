"""Data substrate: the Prior cube, CV splits, and the constant-CO2 model run.

The cube loader and its anomaly helpers are the heavily-used public surface, so
they are re-exported here: ``from paleoreco.data import build_prior_cube`` (and
the other cube symbols) resolves directly. The other concerns stay as explicit
submodules to keep this namespace focused:

* :mod:`paleoreco.data.cube`        - Prior.csv -> dense cube, anomaly, Dataset.
* :mod:`paleoreco.data.splits`      - train/val/test and blocked CV over the age axis.
* :mod:`paleoreco.data.equilibrium` - CCSM4 constant-CO2 run to per-decade fields.
"""

from __future__ import annotations

from .cube import (
    GRID_SHAPE,
    VARS,
    PaleoFieldDataset,
    apply_anomaly,
    build_prior_cube,
    compute_zscore_stats,
    verify_mask_constant_across_ages,
)

__all__ = [
    "build_prior_cube",
    "compute_zscore_stats",
    "apply_anomaly",
    "PaleoFieldDataset",
    "verify_mask_constant_across_ages",
    "VARS",
    "GRID_SHAPE",
]
