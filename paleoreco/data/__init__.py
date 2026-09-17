"""Data substrate: the Prior cube, CV splits, and the constant-CO2 model run.

The cube loader is the heavily-used public surface, so it is re-exported here:
``from paleoreco.data import build_prior_cube`` resolves directly. The split
utilities stay an explicit submodule to keep this namespace focused:

* :mod:`paleoreco.data.cube`        - Prior.csv -> dense cube and per-cell stats.
* :mod:`paleoreco.data.splits`      - train/val/test and blocked CV over the age axis.
"""

from __future__ import annotations

from .cube import (
    GRID_SHAPE,
    VARS,
    build_prior_cube,
    compute_zscore_stats,
)

__all__ = [
    "build_prior_cube",
    "compute_zscore_stats",
    "VARS",
    "GRID_SHAPE",
]
