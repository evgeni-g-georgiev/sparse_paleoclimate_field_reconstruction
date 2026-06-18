"""Coordinate scorings for state and observation vectors.

Three consistent transforms, each a per-element shift and/or scale applied
identically to a prior state (per cell) or an observation vector (per site):

* ``raw``        : the original values.
* ``anomaly``    : subtract the mean (per-cell or per-site climatology).
* ``normalised`` : subtract the mean and divide by the std.

A constant shift leaves a covariance unchanged, so ``raw`` and ``anomaly`` share
the same B and R; ``normalised`` additionally rescales them. The shift also
removes a constant background bias, which the zero-mean innovation assumption of
3DVar requires.
"""

from __future__ import annotations

import numpy as np

RAW = "raw"
ANOMALY = "anomaly"
NORMALISED = "normalised"
SCORINGS: tuple[str, str, str] = (RAW, ANOMALY, NORMALISED)


def score(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Apply one scoring to aligned ``values``/``mean``/``std`` arrays.

    The same function serves prior cells (``mean``/``std`` are per-cell) and
    observations (per-site); the caller supplies the matching statistics.
    """
    if mode == RAW:
        return np.asarray(values, dtype=float)
    shifted = values - mean
    if mode == ANOMALY:
        return shifted
    if mode == NORMALISED:
        return shifted / std
    raise ValueError(f"unknown scoring {mode!r}; expected one of {SCORINGS}")
