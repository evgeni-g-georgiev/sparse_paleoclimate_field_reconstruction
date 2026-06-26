"""Tests for conservative regridding (paleoreco.regrid)."""

from __future__ import annotations

import numpy as np
import pytest

# xarray-regrid is an optional-ish heavy dep; skip cleanly if absent.
pytest.importorskip("xarray_regrid")

from paleoreco.data.regrid import conservative_regrid  # noqa: E402


def test_uniform_field_is_preserved():
    src_lat = np.linspace(-60, 60, 8)
    src_lon = np.linspace(-150, 150, 8)
    tgt_lat = np.linspace(-60, 60, 4)
    tgt_lon = np.linspace(-150, 150, 4)

    field = np.full((src_lat.size, src_lon.size), 5.0)
    out = conservative_regrid(field, src_lat, src_lon, tgt_lat, tgt_lon)

    assert out.shape == (tgt_lat.size, tgt_lon.size)
    # Conservative remap of a constant field is that constant.
    assert np.allclose(out, 5.0, atol=1e-6)


def test_leading_axes_are_carried_through():
    src_lat = np.linspace(-60, 60, 6)
    src_lon = np.linspace(-150, 150, 6)
    tgt_lat = np.linspace(-60, 60, 3)
    tgt_lon = np.linspace(-150, 150, 3)

    field = np.full((4, 2, src_lat.size, src_lon.size), 2.0)
    out = conservative_regrid(field, src_lat, src_lon, tgt_lat, tgt_lon)
    assert out.shape == (4, 2, tgt_lat.size, tgt_lon.size)
    assert np.allclose(out, 2.0, atol=1e-6)
