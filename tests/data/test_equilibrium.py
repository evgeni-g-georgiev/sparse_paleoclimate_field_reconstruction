"""Tests for the CCSM4 equilibrium-run reducers (paleoreco.equilibrium)."""

from __future__ import annotations

import numpy as np

from paleoreco.data.equilibrium import state_fields

_KELVIN = 273.15


def _fake_run():
    # (12 months, 3 decades, 2 lat, 2 lon) in Kelvin.
    rng = np.random.default_rng(1)
    trefht = rng.normal(280.0, 5.0, size=(12, 3, 2, 2))
    return {"TREFHT": trefht}


def test_state_fields_reducers_and_kelvin_offset():
    run = _fake_run()
    monthly = run["TREFHT"]

    mean = state_fields(run, reduce="annual_mean")
    mtco = state_fields(run, reduce="mtco")
    mtwa = state_fields(run, reduce="mtwa")

    assert mean.shape == (3, 2, 2)
    # Coldest/warmest month = min/max over the month axis, in degC.
    assert np.allclose(mtco, monthly.min(axis=0) - _KELVIN)
    assert np.allclose(mtwa, monthly.max(axis=0) - _KELVIN)
    assert np.allclose(mean, monthly.mean(axis=0) - _KELVIN)
    # mtco never exceeds mtwa cell-wise.
    assert (mtco <= mtwa + 1e-6).all()


def test_state_fields_rejects_unknown_reduce():
    import pytest

    with pytest.raises(ValueError):
        state_fields(_fake_run(), reduce="median")
