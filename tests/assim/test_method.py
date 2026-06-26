"""Tests for the AnalysisResult / Observations containers (paleoreco.assim.method)."""

from __future__ import annotations

import numpy as np

from paleoreco.assim.method import AnalysisResult, Observations


def test_to_celsius_adds_climatology():
    anom = np.array([[[1.0, -1.0]], [[2.0, 0.0]]])     # (2, 1, 2)
    clim = np.array([[[10.0, 10.0]], [[20.0, 20.0]]])
    res = AnalysisResult(mean_anom=anom)
    assert np.allclose(res.to_celsius(clim), anom + clim)


def test_predict_obs_gathers_flat_cells():
    anom = np.arange(8, dtype=float).reshape(2, 2, 2)
    res = AnalysisResult(mean_anom=anom)
    g = np.array([0, 3, 7])
    assert np.allclose(res.predict_obs(g), anom.ravel()[g])


def test_observations_is_frozen():
    obs = Observations(gather=np.array([0]), y_anom=np.array([1.0]), sse=np.array([1.0]))
    import dataclasses

    try:
        obs.gather = np.array([1])
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Observations should be frozen")
