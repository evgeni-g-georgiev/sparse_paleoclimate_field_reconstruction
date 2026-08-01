"""Tests for the CCSM4 equilibrium-run reducers (paleoreco.data.equilibrium).

The month-extreme reducers and the prior's solstitial channels are different
statistics, so the properties pinned here differ too: the extremes are ordered by
construction, the hemispheric channels are not.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.data.equilibrium import (
    assign_hemispheric_channels,
    cube_on_grid,
    seasonal_means,
    state_fields,
)

_KELVIN = 273.15
# Index 2 is exactly 0.0: the prior assigns the equator row to the northern
# convention, so the boundary is load-bearing rather than cosmetic.
_LATS = np.array([-45.0, -15.0, 0.0, 30.0])
_LONS = np.linspace(0.0, 315.0, 8)


def _fake_run(seed: int = 1):
    # (12 months, 3 decades, 4 lat, 8 lon) in Kelvin.
    rng = np.random.default_rng(seed)
    return {"TREFHT": rng.normal(280.0, 5.0, size=(12, 3, len(_LATS), len(_LONS))),
            "lat": _LATS, "lon": _LONS}


def _seasonal_run(djf: float, jja: float):
    """A run whose December-February mean is ``djf`` and June-August mean is ``jja``."""
    monthly = np.zeros((12, 2, len(_LATS), len(_LONS)))
    monthly[[11, 0, 1]] = djf + _KELVIN
    monthly[[5, 6, 7]] = jja + _KELVIN
    return {"TREFHT": monthly, "lat": _LATS, "lon": _LONS}


def test_state_fields_reducers_and_kelvin_offset():
    run = _fake_run()
    monthly = run["TREFHT"]

    mean = state_fields(run, reduce="annual_mean")
    cold = state_fields(run, reduce="coldest_month")
    warm = state_fields(run, reduce="warmest_month")

    assert mean.shape == (3, len(_LATS), len(_LONS))
    assert np.allclose(cold, monthly.min(axis=0) - _KELVIN)
    assert np.allclose(warm, monthly.max(axis=0) - _KELVIN)
    assert np.allclose(mean, monthly.mean(axis=0) - _KELVIN)
    # An extreme over the same axis is ordered by construction.
    assert (cold <= warm + 1e-6).all()


def test_state_fields_rejects_unknown_reduce():
    with pytest.raises(ValueError, match="unknown reduce"):
        state_fields(_fake_run(), reduce="median")


def test_seasonal_means_are_month_length_weighted():
    run = _fake_run()
    monthly = run["TREFHT"] - _KELVIN
    djf, jja = seasonal_means(run)

    # 365-day no-leap calendar: December, January, February are 31, 31, 28 days.
    w_djf = np.array([31.0, 31.0, 28.0]) / 90.0
    w_jja = np.array([30.0, 31.0, 31.0]) / 92.0
    assert np.allclose(djf, np.tensordot(w_djf, monthly[[11, 0, 1]], axes=(0, 0)))
    assert np.allclose(jja, np.tensordot(w_jja, monthly[[5, 6, 7]], axes=(0, 0)))
    # A flat mean is a different number, so the weighting is not decorative.
    assert not np.allclose(djf, monthly[[11, 0, 1]].mean(axis=0))


def test_assign_hemispheric_channels_swap_across_the_equator():
    cube = assign_hemispheric_channels(seasonal_means(_seasonal_run(djf=-8.0, jja=12.0)), _LATS)
    north = _LATS >= 0.0

    assert cube.shape == (2, 2, len(_LATS), len(_LONS))
    assert np.allclose(cube[:, 0][:, north], -8.0)      # cold channel is DJF in the north
    assert np.allclose(cube[:, 1][:, north], 12.0)
    assert np.allclose(cube[:, 0][:, ~north], 12.0)     # and JJA in the south
    assert np.allclose(cube[:, 1][:, ~north], -8.0)


def test_assign_hemispheric_channels_put_the_equator_row_north():
    """The prior assigns lat 0 to the northern convention, so the predicate is ``>= 0``."""
    cube = assign_hemispheric_channels(seasonal_means(_seasonal_run(djf=-3.0, jja=5.0)), _LATS)
    equator = int(np.flatnonzero(_LATS == 0.0)[0])
    assert np.allclose(cube[:, 0, equator], -3.0)


def test_assign_hemispheric_channels_reject_mismatched_latitudes():
    seasonal = seasonal_means(_fake_run())
    with pytest.raises(ValueError, match="latitude cells"):
        assign_hemispheric_channels(seasonal, _LATS[:-1])
    with pytest.raises(ValueError, match="DJF, JJA"):
        assign_hemispheric_channels(seasonal[0], _LATS)


def test_cube_on_grid_shape_and_uniform_field():
    pytest.importorskip("xarray_regrid")
    # Inside the source extent, with the middle row straddling the equator so a
    # remap-then-assign ordering would have to blend the two seasons there.
    tgt_lat, tgt_lon = np.array([-30.0, 0.0, 25.0]), np.linspace(-180.0, 90.0, 4)
    cube = cube_on_grid(_seasonal_run(djf=-8.0, jja=12.0), tgt_lat, tgt_lon)

    assert cube.shape == (2, 2, len(tgt_lat), len(tgt_lon))
    # Remapping a spatially uniform field returns that constant, in both channels.
    assert set(np.round(np.unique(cube), 4).tolist()) == {-8.0, 12.0}


def test_cube_on_grid_assigns_hemispheres_after_the_remap():
    """The cold channel steps across the equator, so remapping it first would blend.

    Every target cell must carry exactly the remapped DJF or the remapped JJA, never
    a mixture: the target row centred on the equator draws from source cells in both
    hemispheres, which is where the wrong order shows up.
    """
    regrid = pytest.importorskip("paleoreco.data.regrid")
    run = _seasonal_run(djf=-8.0, jja=12.0)
    # Inside the source extent, with the middle row straddling the equator so a
    # remap-then-assign ordering would have to blend the two seasons there.
    tgt_lat, tgt_lon = np.array([-30.0, 0.0, 25.0]), np.linspace(-180.0, 90.0, 4)

    cube = cube_on_grid(run, tgt_lat, tgt_lon)
    djf, jja = regrid.conservative_regrid(seasonal_means(run), run["lat"], run["lon"],
                                          tgt_lat, tgt_lon)
    north = (tgt_lat >= 0.0)[None, :, None]
    assert np.allclose(cube[:, 0], np.where(north, djf, jja), atol=1e-5)
    assert np.allclose(cube[:, 1], np.where(north, jja, djf), atol=1e-5)
