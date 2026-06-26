"""Tests for the pairwise whitening diagnostic (paleoreco.assim.joint).

The whitened pair is z = L^-1 d where L is the lower Cholesky factor of the 2x2
predicted covariance Sigma_pair. We verify the transform exactly against a
numpy Cholesky solve for known innovations under anomaly (climatological) scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paleoreco.assim.joint import whitened_pair


def test_whitened_pair_equals_cholesky_solve():
    # Two components on cells 0 and 1; fixed prior covariance and obs error.
    diagB = np.array([2.0, 3.0])
    B = np.array([[2.0, 0.8], [0.8, 3.0]])
    sse = 0.5
    s11, s22, s21 = diagB[0] + sse, diagB[1] + sse, B[0, 1]
    Sigma = np.array([[s11, s21], [s21, s22]])
    L = np.linalg.cholesky(Sigma)

    ages = np.array([0, 10, 20, 30])
    dA = np.array([1.0, -2.0, 0.5, 3.0])
    dB = np.array([0.0, 1.0, -1.5, 2.0])

    rows = []
    for a, da_, db_ in zip(ages, dA, dB):
        rows.append({"site": 1, "channel": "mtco", "age": int(a),
                     "y": da_, "sse": sse, "my": 0.0})
        rows.append({"site": 2, "channel": "mtwa", "age": int(a),
                     "y": db_, "sse": sse, "my": 0.0})
    long = pd.DataFrame(rows)

    pairrow = pd.Series({"siteA": 1, "chanA": "mtco", "gA": 0,
                         "siteB": 2, "chanB": "mtwa", "gB": 1})

    z, shared = whitened_pair(
        pairrow, long=long, cube=np.zeros((1, 2)), ages=np.array([0]),
        mean_flat=np.zeros(2), diagB=diagB, B=B, kind="climatological",
    )

    assert np.array_equal(np.sort(shared), ages)
    # Expected lower-Cholesky whitening per age, matched by the returned age order.
    da_by_age = dict(zip(ages, dA))
    db_by_age = dict(zip(ages, dB))
    for row_i, a in enumerate(shared):
        d = np.array([da_by_age[a], db_by_age[a]])
        expected = np.linalg.solve(L, d)
        assert np.allclose(z[row_i], expected, atol=1e-9)
