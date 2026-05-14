"""Dansgaard–Oeschger event-aware train/val/test split for the Prior cube.

Background
----------
The Prior is a LOVECLIM simulation spanning ~29,100 to ~49,175 yr BP, which
covers Greenland D–O events 5 through 12. Randomly splitting the 804 ages
into train/val/test would leak: neighbouring ages are strongly correlated,
so a held-out random sample is almost the same as a training sample. The
robust alternative (used here) is to hold out *entire* D–O events.

Following Liu et al. (2026) Sect. 2.4, each D–O event is identified with
its analysis window: 300 years before to 600 years after the "official"
Greenland D–O warming start date (Wolff et al. 2010, converted to AICC2012
timescale). The Liu 2026 main text and supplement do not tabulate the start
dates themselves; the canonical source for them is Rasmussen et al. 2014,
*Quat. Sci. Rev.* 106, Table 2, on the GICC05modelext timescale (which
agrees with AICC2012 to within ~100 yr across MIS 3). Wolff 2010 Table 1
cites the same pipeline.

Start ages are quoted there in **b2k** (years before 2000 AD). The Prior
is in **yr BP** (years before 1950 AD), so ``BP = b2k − 50``. The windows
in ``DO_EVENT_WINDOWS`` below are computed as ``[start_BP − 300,
start_BP + 600]``, giving a uniform 900-yr window per event, matching the
Liu 2026 main-text definition exactly.

The memorised b2k values may carry small errors (~10–50 yr per event)
relative to the printed Rasmussen 2014 table; this should be verified
against the original table. The 900-yr window is
wide enough that such errors do not affect which Prior ages fall into
which split bucket in practice.

Default split
-------------
* ``test_event = 8`` - a strong, well-resolved event.
* ``val_event = 7``  - adjacent to DO-8 and also extra-tropics-active.
* All other ages (DO-5, 6, 9–12, and "between-event" ages) go to train.

The choice of which events to hold out is to confirm with the supervisor.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# D–O event start ages on the GICC05modelext timescale, from Rasmussen 2014
# Table 2. The first column is the conventional b2k value; the second column
# converts to yr BP (b2k − 50) to match the Prior's age column.
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
# D–O event analysis windows in yr BP.
# Computed as [start_BP − 300, start_BP + 600] where start_BP = b2k − 50.
# ---------------------------------------------------------------------------
DO_EVENT_WINDOWS: dict[int, tuple[int, int]] = {
    event: (b2k - 50 - _WINDOW_PRE, b2k - 50 + _WINDOW_POST)
    for event, b2k in _GI_ONSET_B2K.items()
}

# All D–O event indices we recognise. Order matters for label-collision tests.
DO_EVENT_NUMBERS: tuple[int, ...] = tuple(sorted(DO_EVENT_WINDOWS))


def assign_event_label(ages: np.ndarray) -> np.ndarray:
    """Label each age with its D–O event number (5..12), or 0 if between events.

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
    If an age happens to fall in two overlapping windows, the *higher-numbered* event
    wins, because the loop assigns in ascending event order.
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
    """Partition ages into train / val / test by D–O event membership.

    Each age is assigned to exactly one bucket:
      * ``test``  - ages inside the ``test_event`` window.
      * ``val``   - ages inside the ``val_event`` window.
      * ``train`` - everything else, including ages inside *other* D–O
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


def summarize_split(ages: np.ndarray, split: dict[str, np.ndarray]) -> str:
    """Summary of a split.

    Reports the number of ages in each bucket and the min/max age covered.
    Use to print and visually sanity-check before training.
    """
    ages = np.asarray(ages, dtype=np.int64)
    lines = []
    for name in ("train", "val", "test"):
        idx = split[name]
        if len(idx) == 0:
            lines.append(f"{name:>5}: 0 ages")
            continue
        sub = ages[idx]
        lines.append(
            f"{name:>5}: {len(idx):4d} ages, "
            f"range [{int(sub.min()):>5}, {int(sub.max()):>5}] yr BP"
        )
    return "\n".join(lines)
