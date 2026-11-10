"""The age axis: its spacing, its chronological split, and the D-O event windows.

The prior spans 29,100 to 49,175 yr BP, covering Greenland interstadials 5 to 12. Following
Liu et al. (2026) Sect. 2.4, each event's window runs from 300 yr before its onset to 600 yr
after. Onsets are from Rasmussen et al. (2014) Table 2 in b2k, and ``BP = b2k - 50``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# D-O event onsets in b2k (years before 2000 AD), from Rasmussen 2014
# Table 2 on the GICC05modelext timescale.
# ---------------------------------------------------------------------------
_GI_ONSET_B2K: dict[int, int] = {
    5:  32_500,
    6:  33_720,
    7:  35_470,
    8:  38_220,
    9:  40_160,
    10: 41_460,
    11: 43_340,
    12: 46_860,
}

# Liu 2026 analysis window: 300 yr before to 600 yr after each start.
_WINDOW_PRE: int = 300
_WINDOW_POST: int = 600

# D-O event analysis windows in yr BP.
DO_EVENT_WINDOWS: dict[int, tuple[int, int]] = {
    event: (b2k - 50 - _WINDOW_PRE, b2k - 50 + _WINDOW_POST)
    for event, b2k in _GI_ONSET_B2K.items()
}

# The onsets themselves in yr BP, which is what a composite aligned on the
# transition needs, as opposed to the padded analysis window around it.
DO_ONSET_BP: dict[int, int] = {event: b2k - 50 for event, b2k in _GI_ONSET_B2K.items()}


def age_step(ages: np.ndarray) -> float:
    """The single spacing of the age axis; the lag and timescale arithmetic assume it."""
    steps = np.unique(np.diff(np.asarray(ages, dtype=np.int64)))
    if len(steps) != 1:
        raise ValueError(f"the age axis must have a uniform step; found {steps}")
    return float(steps[0])


def chronological_half_split(
    ages: np.ndarray, *, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Split the age axis at its midpoint into ``(older_idx, younger_idx)``.

    ``ages`` is ascending yr BP, so the later indices are the older states. The
    younger half is thinned by ``stride``: neighbouring states at the cube's spacing
    are strongly autocorrelated, so a stride buys near-independent members.
    """
    n = len(np.asarray(ages))
    mid = n // 2
    return np.arange(mid, n), np.arange(0, mid)[::stride]
