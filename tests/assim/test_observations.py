"""Unit tests for the representativeness-variance estimator (paleoreco.assim.observations).

``representativeness_variance`` pools co-cell, co-age proxy pairs into a per-channel
variance; these check it against a brute-force pair computation and pin the contract that
keeps the withholding estimate leakage-free (the ``sites`` filter).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim.observations import representativeness_variance


def _table(rows: list[tuple]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a (long, cell) pair from ``(site, channel, age, y, my, sse, cell)`` tuples."""
    cols = ["site", "channel", "age", "y", "my", "sse", "cell"]
    df = pd.DataFrame(rows, columns=cols)
    return df, df["cell"].to_numpy()


def _brute(long: pd.DataFrame, cell: np.ndarray, channel: str, sites=None) -> float:
    """Mean over co-(cell, age) pairs of ``(a_i-a_j)^2/2 - (sse_i+sse_j)/2`` for one channel."""
    df = long.assign(cell=cell, a=long["y"] - long["my"])
    df = df[df["channel"] == channel]
    if sites is not None:
        df = df[df["site"].isin(sites)]
    vals = []
    for _, g in df.groupby(["cell", "age"]):
        a, sse = g["a"].to_numpy(), g["sse"].to_numpy()
        vals += [0.5 * (a[i] - a[j]) ** 2 - 0.5 * (sse[i] + sse[j])
                 for i, j in combinations(range(len(g)), 2)]
    return float(np.mean(vals)) if vals else 0.0


# Two mtco cells (one observed at two ages) and one mtwa cell, all sse = 0.5.
_ROWS = [
    (1, "mtco", 100, 1.0, 0.0, 0.5, 10),
    (2, "mtco", 100, 3.0, 0.0, 0.5, 10),
    (1, "mtco", 200, 2.0, 0.0, 0.5, 10),
    (2, "mtco", 200, 0.0, 0.0, 0.5, 10),
    (3, "mtco", 100, 5.0, 0.0, 0.5, 20),
    (4, "mtco", 100, 1.0, 0.0, 0.5, 20),
    (5, "mtwa", 100, 0.0, 0.0, 0.5, 30),
    (6, "mtwa", 100, 2.0, 0.0, 0.5, 30),
]


def test_matches_brute_force_pair_computation():
    long, cell = _table(_ROWS)
    rep = representativeness_variance(long, cell)
    assert rep["mtco"] == pytest.approx(_brute(long, cell, "mtco"))
    assert rep["mtwa"] == pytest.approx(_brute(long, cell, "mtwa"))
    # Closed form against hand values: mtco pooled = mean(1.5, 1.5, 7.5), mtwa = 1.5.
    assert rep["mtco"] == pytest.approx(3.5)
    assert rep["mtwa"] == pytest.approx(1.5)


def test_sites_filter_excludes_a_sites_pairs():
    """Dropping a co-cell site removes its pairs, the leakage-clean contract."""
    long, cell = _table(_ROWS)
    rep = representativeness_variance(long, cell, sites={1, 3, 4, 5, 6})
    # Cell 10 now holds only site 1, so its pairs vanish; only cell 20's pair remains.
    assert rep["mtco"] == pytest.approx(7.5)
    assert rep["mtwa"] == pytest.approx(1.5)
    assert rep["mtco"] == pytest.approx(_brute(long, cell, "mtco", sites={1, 3, 4, 5, 6}))


def test_singleton_cells_give_zero():
    long, cell = _table(_ROWS)
    rep = representativeness_variance(long, cell, sites={1, 3, 5})  # one site per cell
    assert rep["mtco"] == 0.0
    assert rep["mtwa"] == 0.0


def test_clamps_to_zero_when_sse_exceeds_scatter():
    rows = [(s, c, a, y, m, 100.0, cell) for (s, c, a, y, m, _, cell) in _ROWS]
    long, cell = _table(rows)
    rep = representativeness_variance(long, cell)
    assert rep["mtco"] == 0.0
    assert rep["mtwa"] == 0.0


def test_channels_estimated_independently():
    long, cell = _table(_ROWS)
    rep = representativeness_variance(long, cell)
    assert rep["mtco"] != rep["mtwa"]
