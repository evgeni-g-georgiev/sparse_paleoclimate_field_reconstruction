"""Tests for the AnalysisResult container (paleoreco.assim.method)."""

from __future__ import annotations

import numpy as np

from paleoreco.assim.method import AnalysisResult


def test_predict_obs_gathers_flat_cells():
    anom = np.arange(8, dtype=float).reshape(2, 2, 2)
    res = AnalysisResult(mean_anom=anom)
    g = np.array([0, 3, 7])
    assert np.allclose(res.predict_obs(g), anom.ravel()[g])


def test_the_optional_spread_fields_default_to_absent():
    """A mean-only analysis must not be read as reporting a zero spread.

    This also pins that the two positional fields bind in the order every construction
    site passes them.
    """
    anom = np.zeros((2, 1, 2))
    var = np.ones((2, 1, 2))
    assert AnalysisResult(mean_anom=anom).posterior_var is None
    assert AnalysisResult(mean_anom=anom).posterior_cross_var is None
    assert AnalysisResult(anom, var).posterior_var is var
    assert AnalysisResult(anom, var).posterior_cross_var is None

