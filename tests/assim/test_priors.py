"""Tests for B regularizer tapers and geometry helpers (paleoreco.assim.priors)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.priors import (
    build_prior,
    coupling_taper,
    gaspari_cohn,
    great_circle_km,
    localization_taper,
    regularization_mask,
    shrinkage_taper,
)

_R = 6371.0


def test_gaspari_cohn_endpoints_and_support():
    L = 1000.0
    d = np.array([0.0, L, 1.5 * L, 2.0 * L, 2.5 * L])
    gc = gaspari_cohn(d, L)
    assert gc[0] == pytest.approx(1.0)       # full weight at zero separation
    assert gc[3] == pytest.approx(0.0, abs=1e-9)   # zero at 2L
    assert gc[4] == pytest.approx(0.0)       # and beyond
    assert np.all(gc >= -1e-12)              # taper stays non-negative on this range
    assert gc[1] > gc[2] > gc[3]             # monotone decreasing


def test_great_circle_quarter_circumference():
    lat = np.array([0.0, 0.0])
    lon = np.array([0.0, 90.0])
    d = great_circle_km(lat, lon)
    assert d[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(d, d.T)
    assert d[0, 1] == pytest.approx(_R * np.pi / 2, rel=1e-6)


def test_localization_taper_block_structure():
    lats = np.array([-30.0, 0.0, 30.0])
    lons = np.array([0.0, 120.0, 240.0])
    n = lats.size * lons.size
    T = localization_taper(lats, lons, length_km=5000.0)
    assert T.shape == (2 * n, 2 * n)
    spatial = T[:n, :n]
    # All four channel blocks share the one spatial taper.
    assert np.allclose(T[n:, n:], spatial)
    assert np.allclose(T[:n, n:], spatial)
    assert np.allclose(T[n:, :n], spatial)
    assert np.allclose(np.diag(spatial), 1.0)            # self-correlation


def test_coupling_taper_scales_cross_channel_blocks():
    n_lat, n_lon = 3, 3
    n = n_lat * n_lon
    C = coupling_taper(n_lat, n_lon, alpha=0.4)
    assert C.shape == (2 * n, 2 * n)
    assert np.allclose(np.diag(C), 1.0)                  # variances untouched
    assert np.allclose(C[:n, :n], 1.0)                   # within-channel blocks intact
    assert np.allclose(C[n:, n:], 1.0)
    assert np.allclose(C[:n, n:], 0.4)                   # cross-channel scaled by alpha
    assert np.allclose(C[n:, :n], 0.4)
    # Endpoints: alpha=1 is a no-op, alpha=0 is block-diagonal.
    assert np.allclose(coupling_taper(n_lat, n_lon, 1.0), 1.0)
    block = coupling_taper(n_lat, n_lon, 0.0)
    assert np.allclose(block[:n, n:], 0.0) and np.allclose(block[:n, :n], 1.0)
    assert np.linalg.eigvalsh(C).min() > -1e-8           # PSD for alpha in [0, 1]


def test_shrinkage_taper_scales_offdiag_keeps_diag():
    n_lat, n_lon = 2, 3
    d = 2 * n_lat * n_lon
    S = shrinkage_taper(n_lat, n_lon, lam=0.25)
    assert S.shape == (d, d)
    assert np.allclose(np.diag(S), 1.0)                  # variances exact
    off = ~np.eye(d, dtype=bool)
    assert np.allclose(S[off], 0.75)                     # off-diag scaled by 1 - lam
    assert np.linalg.eigvalsh(S).min() > -1e-8           # PSD for lam in [0, 1]


def test_regularization_mask_none_when_off_and_composes_when_on():
    lats = np.array([-30.0, 0.0, 30.0])
    lons = np.array([0.0, 120.0, 240.0])
    assert regularization_mask(lats, lons, localization_km=None,
                               shrinkage_lambda=0.0, alpha=1.0) is None
    m = regularization_mask(lats, lons, localization_km=12500.0,
                            shrinkage_lambda=0.25, alpha=0.5)
    n = lats.size * lons.size
    assert m.shape == (2 * n, 2 * n)
    assert np.allclose(np.diag(m), 1.0)                  # diagonal preserved
    assert np.allclose(m, m.T)


def _sample_cube(n_ages=8, n_lat=4, n_lon=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_ages, 2, n_lat, n_lon)).astype(np.float32)


def test_build_prior_defaults_are_raw_sample_covariance():
    cube = _sample_cube()
    lats = np.linspace(-60, 60, 4).astype(np.float32)
    lons = np.linspace(-180, 120, 5).astype(np.float32)
    ages = np.arange(8, dtype=np.int64)
    valid = np.ones((4, 5), bool)
    idx = np.arange(8)
    from paleoreco.assim.background import background_covariance
    prior = build_prior(cube, ages, lats, lons, idx, valid)
    assert np.allclose(prior.B, background_covariance(cube, idx))
    assert prior.meta == {"localization_km": None, "shrinkage_lambda": 0.0,
                          "alpha": 1.0, "n_prior_ages": 8}


def test_build_prior_alpha_zero_decouples_channels():
    cube = _sample_cube()
    lats = np.linspace(-60, 60, 4).astype(np.float32)
    lons = np.linspace(-180, 120, 5).astype(np.float32)
    ages = np.arange(8, dtype=np.int64)
    valid = np.ones((4, 5), bool)
    prior = build_prior(cube, ages, lats, lons, np.arange(8), valid, alpha=0.0)
    n = 4 * 5
    assert np.allclose(prior.B[:n, n:], 0.0)             # cross-channel blocks zeroed
    assert np.allclose(prior.B[n:, :n], 0.0)
    assert np.allclose(prior.B, prior.B.T)               # stays symmetric
    assert np.linalg.eigvalsh(prior.B).min() > -1e-6     # stays PSD
