"""The prior cube and splits of its age axis.

* :mod:`paleoreco.data.cube`   - Prior.csv to a dense cube, and per-cell statistics.
* :mod:`paleoreco.data.splits` - age-axis splits and the D-O event windows.
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
