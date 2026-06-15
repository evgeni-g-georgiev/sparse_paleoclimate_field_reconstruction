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


def split_ages_by_do_event(
    ages: np.ndarray,
    test_event: int = 8,
    val_event: int = 7,
) -> dict[str, np.ndarray]:
    """Partition ages into train / val / test by D-O event membership.

    Each age is assigned to exactly one bucket:
      * ``test``  - ages inside the ``test_event`` window.
      * ``val``   - ages inside the ``val_event`` window.
      * ``train`` - everything else, including ages inside *other* D-O
                    events and ages between events. Between-event ages
                    default to train rather than being dropped.

    Parameters
    ----------
    ages : (N,) array-like of int
        Ages in yr BP. Typically the ``ages`` field returned by
        :func:`paleoreco.data.build_prior_cube`.
    test_event : int in {5..12}, default 8.
    val_event : int in {5..12}, default 7. Must differ from ``test_event``.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``.
        Each value is an ``int64`` array of *indices into* ``ages`` (not
        the ages themselves). This is the form expected by
        :func:`paleoreco.data.compute_zscore_stats` and
        :class:`paleoreco.data.PaleoFieldDataset`.

    Raises
    ------
    ValueError
        If ``test_event == val_event`` or either is not in ``DO_EVENT_NUMBERS``.
    """
    if test_event == val_event:
        raise ValueError(
            f"test_event and val_event must differ; both equal {test_event}"
        )
    for name, ev in (("test_event", test_event), ("val_event", val_event)):
        if ev not in DO_EVENT_WINDOWS:
            raise ValueError(
                f"{name}={ev} is not a recognised event; "
                f"expected one of {list(DO_EVENT_NUMBERS)}"
            )

    labels = assign_event_label(ages)
    test_mask = labels == test_event
    val_mask = labels == val_event
    train_mask = ~(test_mask | val_mask)

    return {
        "train": np.flatnonzero(train_mask).astype(np.int64),
        "val":   np.flatnonzero(val_mask).astype(np.int64),
        "test":  np.flatnonzero(test_mask).astype(np.int64),
    }


def block_stride_split(
    n_ages: int,
    block_size: int = 40,
    test_stride: int = 10,
    test_offset: int = 0,
    val_stride: int = 10,
    val_offset: int = 5,
) -> dict[str, np.ndarray]:
    """Partition ``range(n_ages)`` into train / val / test by block stride.

    Block ``b`` covers ``ages[b*block_size : (b+1)*block_size]``. Test takes
    blocks with ``b % test_stride == test_offset``; val likewise. Train gets
    the rest plus any leftover at the end. Deterministic, no seed.

    Raises ``ValueError`` on bad sizes/offsets, ``n_ages < block_size``, or
    overlapping val/test block sets.
    """
    if block_size <= 0 or test_stride <= 0 or val_stride <= 0:
        raise ValueError(
            f"block_size, test_stride, val_stride must be positive; got "
            f"block_size={block_size}, test_stride={test_stride}, "
            f"val_stride={val_stride}"
        )
    if not 0 <= test_offset < test_stride:
        raise ValueError(
            f"test_offset={test_offset} outside [0, {test_stride})"
        )
    if not 0 <= val_offset < val_stride:
        raise ValueError(
            f"val_offset={val_offset} outside [0, {val_stride})"
        )
    if n_ages < block_size:
        raise ValueError(
            f"n_ages={n_ages} smaller than one block of size {block_size}"
        )

    n_blocks = n_ages // block_size
    block_ids = np.arange(n_blocks)
    test_blocks = set(block_ids[block_ids % test_stride == test_offset].tolist())
    val_blocks = set(block_ids[block_ids % val_stride == val_offset].tolist())

    overlap = test_blocks & val_blocks
    if overlap:
        raise ValueError(
            f"val and test block sets overlap on blocks {sorted(overlap)}; "
            f"adjust offsets/strides"
        )

    test_idx, val_idx, train_idx = [], [], []
    for b in range(n_blocks):
        start, stop = b * block_size, (b + 1) * block_size
        if b in test_blocks:
            test_idx.extend(range(start, stop))
        elif b in val_blocks:
            val_idx.extend(range(start, stop))
        else:
            train_idx.extend(range(start, stop))
    # Leftover ages past the last full block go to train; they sit at the
    # oldest end of the timeline and are deterministically out of val/test.
    train_idx.extend(range(n_blocks * block_size, n_ages))

    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val":   np.asarray(val_idx,   dtype=np.int64),
        "test":  np.asarray(test_idx,  dtype=np.int64),
    }


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


def summarize_split(ages: np.ndarray, split: dict[str, np.ndarray]) -> str:
    """One line per bucket: count, age range in yr BP, and per-D-O-event
    composition (via :func:`assign_event_label`)."""
    ages = np.asarray(ages, dtype=np.int64)
    lines = []
    for name in ("train", "val", "test"):
        idx = split[name]
        if len(idx) == 0:
            lines.append(f"{name:>5}: 0 ages")
            continue
        sub = ages[idx]
        labels = assign_event_label(sub)
        per_event = [
            f"DO{ev}={int((labels == ev).sum())}"
            for ev in DO_EVENT_NUMBERS
            if (labels == ev).any()
        ]
        between = int((labels == 0).sum())
        composition = ", ".join(per_event + [f"between={between}"])
        lines.append(
            f"{name:>5}: {len(idx):4d} ages, "
            f"range [{int(sub.min()):>5}, {int(sub.max()):>5}] yr BP "
            f"[{composition}]"
        )
    return "\n".join(lines)
