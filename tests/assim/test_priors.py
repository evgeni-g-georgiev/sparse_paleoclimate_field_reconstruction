"""Tests for B regularizers and geometry helpers (paleoreco.assim.priors)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.priors import (
    channel_taper,
    eof_truncate,
    gaspari_cohn,
    great_circle_km,
    per_age_neighbour,
    shrink_to_diagonal,
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


def test_eof_truncate_is_low_rank_psd_symmetric():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(10, 10))
    B = M @ M.T
    Bk = eof_truncate(B, rank=3)
    assert np.allclose(Bk, Bk.T)
    assert np.linalg.matrix_rank(Bk, tol=1e-8) <= 3
    w = np.linalg.eigvalsh(Bk)
    assert w.min() > -1e-8                   # PSD


def test_shrink_to_diagonal_preserves_variance_scales_offdiag():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 4))
    Xc = X - X.mean(axis=0, keepdims=True)
    s = (Xc.T @ Xc) / (X.shape[0] - 1)       # sample covariance
    B = shrink_to_diagonal(X, lam=0.5)
    assert np.allclose(np.diag(B), np.diag(s))           # variances exact
    off = ~np.eye(4, dtype=bool)
    assert np.allclose(B[off], 0.5 * s[off])             # off-diag scaled by 1 - lam


def test_channel_taper_block_structure():
    lats = np.array([-30.0, 0.0, 30.0])
    lons = np.array([0.0, 120.0, 240.0])
    n = lats.size * lons.size
    T = channel_taper(lats, lons, length_km=5000.0)
    assert T.shape == (2 * n, 2 * n)
    spatial = T[:n, :n]
    # All four channel blocks share the one spatial taper.
    assert np.allclose(T[n:, n:], spatial)
    assert np.allclose(T[:n, n:], spatial)
    assert np.allclose(T[n:, :n], spatial)
    assert np.allclose(np.diag(spatial), 1.0)            # self-correlation


def test_per_age_neighbour_respects_offset():
    prior_ages = np.array([1000, 2000, 3000, 5000])
    n = per_age_neighbour(3000, prior_ages, offset_yr=2000)
    assert abs(n - 3000) >= 2000
    # Fallback: when nothing satisfies the offset, take the farthest available.
    n2 = per_age_neighbour(3000, prior_ages, offset_yr=10_000)
    assert n2 in (1000, 5000)
