"""The prior cube and its age axis.

* :mod:`paleoreco.data.cube`   - Prior.csv to a dense cube, and per-cell climatology.
* :mod:`paleoreco.data.splits` - the age axis: spacing, chronological split, D-O windows.
"""

from __future__ import annotations

from .cube import VARS, build_prior_cube, cell_climatology

__all__ = ["build_prior_cube", "cell_climatology", "VARS"]
