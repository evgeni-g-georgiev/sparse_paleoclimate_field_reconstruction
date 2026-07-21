"""Train/val/test split utilities for the Prior cube.

Background on the D-O windows
-----------------------------
The Prior spans ~29,100 to ~49,175 yr BP, covering Greenland D-O events 5
through 12. Following Liu et al. (2026) Sect. 2.4, each D-O event is
defined by an analysis window of 300 years before to 600 years after its
onset. Onsets are taken from Rasmussen et al. 2014 (*Quat. Sci. Rev.* 106,
Table 2), reported in **b2k** (years before 2000 AD). The Prior is in
**yr BP** (years before 1950 AD), so ``BP = b2k - 50``.
``DO_EVENT_WINDOWS`` applies both conversions in one step.
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

# ---------------------------------------------------------------------------
# D-O event analysis windows in yr BP, built from _GI_ONSET_B2K
# minus 50 (b2k→BP) and padded by _WINDOW_PRE / _WINDOW_POST.
# ---------------------------------------------------------------------------
DO_EVENT_WINDOWS: dict[int, tuple[int, int]] = {
    event: (b2k - 50 - _WINDOW_PRE, b2k - 50 + _WINDOW_POST)
    for event, b2k in _GI_ONSET_B2K.items()
}

# Recognised D-O event indices, ascending. Order is load-bearing:
# overlapping windows in assign_event_label are resolved in this order.
DO_EVENT_NUMBERS: tuple[int, ...] = tuple(sorted(DO_EVENT_WINDOWS))


def assign_event_label(ages: np.ndarray) -> np.ndarray:
    """Label each age with its D-O event number (5..12), or 0 if between events.

    Parameters
    ----------
    ages : array-like of int
        Ages in yr BP. No sort requirement.

    Returns
    -------
    np.ndarray of int64, same shape as ``ages``.
        Element values: ``5..12`` for ages inside the corresponding event
        window (inclusive on both ends), ``0`` for ages outside every window.

    Notes
    -----
    If an age falls in two overlapping windows, the higher-numbered event
    wins (the loop assigns in ascending event order).
    """
    ages = np.asarray(ages, dtype=np.int64)
    labels = np.zeros_like(ages)
    for event in DO_EVENT_NUMBERS:
        lo, hi = DO_EVENT_WINDOWS[event]
        in_event = (ages >= lo) & (ages <= hi)
        labels[in_event] = event
    return labels


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


def make_blocked_cv(
    ages: np.ndarray,
    fold_years: int = 1000,
    embargo_years: int = 1000,
    holdout_test: bool = True,
    test_event: int = 8,
    test_fraction: float = 0.1,
) -> dict:
    """Purged blocked k-fold CV over the age axis, with an optional held-out test event.

    The timeline is cut into contiguous chunks of about ``fold_years``; each chunk is
    validation once while the rest trains, minus an ``embargo_years`` buffer on either
    side of the chunk. The embargo matters because snapshots sit 25 yr apart and the
    chunk width is near a D-O event width, so without it near-duplicate neighbours and
    the far half of a bisected event leak into training.

    When ``holdout_test`` is set, the ``test_fraction`` of snapshots closest to the
    ``test_event`` onset (that event plus surrounding non-event ages, never another
    event) form a stratified test block holding one full D-O transition, removed
    before folding. With it unset every snapshot is folded and no test block is built.

    Returns ``{"test": idx, "folds": [{"train": idx, "val": idx}, ...]}`` with int64
    indices into ``ages``; ``"test"`` is empty when ``holdout_test`` is false. Folds
    whose validation chunk is fully absorbed by the test block are dropped.
    """
    ages = np.asarray(ages, dtype=np.int64)
    n = ages.size
    if n == 0:
        raise ValueError("ages is empty")
    if fold_years <= 0 or embargo_years < 0:
        raise ValueError(
            f"need fold_years > 0 and embargo_years >= 0; got "
            f"fold_years={fold_years}, embargo_years={embargo_years}"
        )

    # Stratified test block: snapshots nearest the test event onset, drawn only
    # from that event and between-event ages so no other event is split across
    # the test boundary.
    test_idx = np.array([], dtype=np.int64)
    excluded = np.zeros(n, dtype=bool)   # test block plus its embargo band
    if holdout_test:
        if test_event not in DO_EVENT_WINDOWS:
            raise ValueError(
                f"test_event={test_event} not in {list(DO_EVENT_NUMBERS)}"
            )
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
        lo, hi = DO_EVENT_WINDOWS[test_event]
        center = (lo + hi) / 2.0
        labels = assign_event_label(ages)
        eligible = np.flatnonzero((labels == test_event) | (labels == 0))
        n_test = max(1, round(test_fraction * n))
        if n_test > eligible.size:
            raise ValueError(
                f"test_fraction={test_fraction} needs {n_test} ages but only "
                f"{eligible.size} are eligible around event {test_event}"
            )
        test_idx = np.sort(eligible[np.argsort(np.abs(ages[eligible] - center))][:n_test])
        t_lo, t_hi = int(ages[test_idx].min()), int(ages[test_idx].max())
        excluded = (ages >= t_lo - embargo_years) & (ages <= t_hi + embargo_years)

    pool = ~excluded   # snapshots available to train or validate across folds

    # Contiguous near-equal chunks so there is no tiny stub fold: fold_years sets
    # the target width, the span fixes the whole count.
    span = int(ages.max() - ages.min())
    n_folds = max(1, round(span / fold_years)) if span > 0 else 1
    edges = np.linspace(ages.min(), ages.max(), n_folds + 1)

    folds = []
    for f in range(n_folds):
        c_lo, c_hi = edges[f], edges[f + 1]
        if f == n_folds - 1:
            in_chunk = (ages >= c_lo) & (ages <= c_hi)
        else:
            in_chunk = (ages >= c_lo) & (ages < c_hi)
        val_mask = pool & in_chunk
        if not val_mask.any():
            continue
        near = (ages >= c_lo - embargo_years) & (ages <= c_hi + embargo_years)
        train_mask = pool & ~near
        folds.append({
            "train": np.flatnonzero(train_mask).astype(np.int64),
            "val":   np.flatnonzero(val_mask).astype(np.int64),
        })

    return {"test": test_idx, "folds": folds}
