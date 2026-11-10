"""Unit tests for the observation error model (paleoreco.assim.observations).

``representativeness_variance`` pools co-cell, co-age proxy pairs into a per-channel
variance; these check it against a brute-force pair computation and pin the contract that
keeps the withholding estimate leakage-free (the ``sites`` filter). The temporal terms are
checked against a structure function with a known lag correlation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim.observations import (
    ObservationModel,
    deflate,
    representativeness_variance,
    temporal_terms,
)


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


# --- temporal error ---------------------------------------------------------
# One cell whose structure function is the textbook 2*var*(1 - rho): var = 1, so
# S = [0, 0.5, 1.0, 2.0] is rho = [1, 0.75, 0.5, 0].
_S = np.array([[0.0], [0.5], [1.0], [2.0]])
_VAR = np.array([1.0])
_G = np.array([0])


def test_lag_rounds_half_away_from_zero():
    """A block centre is a midpoint, so half-step lags are half the real network.

    ``np.rint`` rounds half to even, which would send a 0.5-step lag to zero and leave
    those samples uncorrected while alternating the rest.
    """
    half_step = temporal_terms(_S, _VAR, _G, np.array([12.5]), 25.0)[0]
    assert half_step[0] == pytest.approx(0.75)      # lag index 1, not 0

    one_and_a_half = temporal_terms(_S, _VAR, _G, np.array([37.5]), 25.0)[0]
    assert one_and_a_half[0] == pytest.approx(0.5)  # lag index 2, as np.rint also gives


def test_lag_index_is_clipped_to_the_structure_function():
    """A network reaching past the prior's longest lag must damp, not raise."""
    rho, resid = temporal_terms(_S, _VAR, _G, np.array([10_000.0]), 25.0)
    assert rho[0] == pytest.approx(0.05)
    assert np.isfinite(resid).all()


def test_residual_variance_uses_the_floored_correlation():
    """A raw negative rho would understate the noise it is meant to charge for.

    At the floor the two moments share nothing, so the observation has to carry the
    whole climatological variance as error.
    """
    S_negative = np.array([[0.0], [3.0]])           # raw rho = -0.5
    _, resid = temporal_terms(S_negative, _VAR, _G, np.array([25.0]), 25.0)
    assert resid[0] == pytest.approx(1.0 * (1.0 - 0.05 ** 2))


def test_zero_lag_leaves_the_observation_untouched():
    y, sse = np.array([2.0]), np.array([0.5])
    rho, resid = temporal_terms(_S, _VAR, _G, np.array([0.0]), 25.0)
    yy, rr = deflate(y, sse, rho, resid)
    assert yy == pytest.approx(y)
    assert rr == pytest.approx(sse)


def test_deflation_undoes_the_attenuation():
    """A stale reading of a state that has decayed to ``rho`` times its value implies the
    full-size state, which is what the update is solving for; R pays for it in ``rho^2``.
    """
    rho, resid = temporal_terms(_S, _VAR, _G, np.array([50.0]), 25.0)   # rho = 0.5
    truth = np.array([4.0])
    y_def, r_def = deflate(rho * truth, np.array([0.1]), rho, resid)
    assert y_def == pytest.approx(truth)
    assert r_def == pytest.approx((0.1 + resid) / rho ** 2)


def test_a_stale_observation_is_trusted_less_the_further_it_sits():
    y, sse = np.array([1.0]), np.array([0.5])
    r = [deflate(y, sse, *temporal_terms(_S, _VAR, _G, np.array([lag]), 25.0))[1][0]
         for lag in (0.0, 25.0, 50.0)]
    assert r[0] < r[1] < r[2], r


def test_an_observation_model_is_named_by_the_terms_it_carries():
    assert ObservationModel(spatial=False, temporal=False).name == "sse"
    assert ObservationModel(spatial=True, temporal=False).name == "sse_spatial"
    assert ObservationModel(spatial=False, temporal=True).name == "sse_temporal"
    assert ObservationModel(spatial=True, temporal=True).name == "sse_spatial_temporal"
