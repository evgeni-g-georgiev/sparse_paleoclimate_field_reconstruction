"""Tests for B regularizer tapers and geometry helpers (paleoreco.assim.priors)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.priors import (
    build_prior,
    chord_km,
    coupling_taper,
    gaspari_cohn,
    great_circle_km,
    localization_taper,
    regularization_mask,
    shrinkage_taper,
    taper_obs_blocks,
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


def test_chord_is_the_straight_line_through_the_sphere():
    """The chord saturates at the diameter, which is the whole reason the taper reads it.

    Arc length keeps growing to 20015 km between antipodes, far enough for a 20000 km
    Gaspari-Cohn support to wrap; the chord never exceeds 12742 km.
    """
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0, 90.0, 180.0])
    d = chord_km(lat, lon)
    assert np.allclose(np.diag(d), 0.0, atol=1e-6)
    assert np.allclose(d, d.T)
    assert d[0, 1] == pytest.approx(_R * np.sqrt(2.0), rel=1e-6)   # quarter circumference
    assert d[0, 2] == pytest.approx(2.0 * _R, rel=1e-6)            # antipodes: the diameter
    assert d.max() <= 2.0 * _R + 1e-9


def test_chord_and_great_circle_stay_consistent_through_the_half_angle():
    """One haversine feeds both readings, so neither can drift without the other noticing."""
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-90, 90, 40), rng.uniform(-180, 180, 40)
    arc = great_circle_km(lat, lon)
    assert np.allclose(chord_km(lat, lon), 2.0 * _R * np.sin(arc / (2.0 * _R)))


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


def test_the_localization_taper_is_read_on_the_chord_not_the_arc():
    """Two cells a quarter circumference apart: 10008 km of arc, 9010 km of chord.

    Both fall inside an 8000 km Gaspari-Cohn support, so the two readings give visibly
    different weights (0.130 against 0.075) rather than agreeing by accident. Nothing
    else in the suite would notice the metric reverting to arc length until a sweep
    reached a lengthscale where B goes indefinite again.
    """
    lats, lons = np.array([0.0]), np.array([0.0, 90.0])
    T = localization_taper(lats, lons, length_km=8000.0)
    assert T[0, 1] == pytest.approx(gaspari_cohn(np.array([_R * np.sqrt(2.0)]), 8000.0)[0])
    assert T[0, 1] != pytest.approx(gaspari_cohn(np.array([_R * np.pi / 2]), 8000.0)[0])


def _antipodal_grid(n_lat=8, n_lon=16):
    """A global grid whose cells reach exactly half the circumference apart.

    Positive definiteness of a distance taper is a property of the whole point set, and a
    regional grid never separates two cells far enough for the Gaspari-Cohn support to
    wrap, so it cannot see the defect at all.
    """
    return np.linspace(-78.75, 78.75, n_lat), np.linspace(-180.0, 157.5, n_lon)


def test_the_psd_grid_really_does_contain_an_antipodal_pair():
    """Guards the test below, which is toothless on a grid that stays within a hemisphere."""
    lats, lons = _antipodal_grid()
    d = great_circle_km(np.repeat(lats, len(lons)), np.tile(lons, len(lats)))
    assert d.max() == pytest.approx(np.pi * _R, rel=1e-9)


@pytest.mark.parametrize("length_km", [5000.0, 8000.0, 12500.0, 20000.0, 25000.0, 35000.0])
@pytest.mark.parametrize("shrinkage_lambda, alpha", [(0.0, 1.0), (0.5, 0.25)])
def test_the_regularization_mask_is_psd_at_every_swept_lengthscale(length_km,
                                                                  shrinkage_lambda, alpha):
    """A Schur taper leaves B a covariance only if the taper is PSD, at every swept value.

    Gaspari-Cohn is positive definite in R^3, but read on arc length it is positive
    definite on the sphere only while its support stays inside half the circumference
    (Gneiting 2013, Bernoulli 19(4), Thm 3), and the swept lengthscales run well past
    that: on this grid the arc-length reading gives the composed mask a minimum
    eigenvalue of -0.019 at 12500 km, -0.20 at 15000 km and -2.8 at 20000 km, so B stops
    being a covariance at the operating point rather than at some extrapolated one.
    Pure localization is the case that catches it, since shrinkage toward the diagonal
    adds enough to the spectrum to hide a mildly indefinite taper.
    """
    lats, lons = _antipodal_grid()
    mask = regularization_mask(lats, lons, localization_km=length_km,
                               shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    assert np.linalg.eigvalsh(mask).min() > -1e-8


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


@pytest.mark.parametrize("taper", [
    dict(localization_km=9000.0, shrinkage_lambda=0.0, alpha=1.0),
    dict(localization_km=None, shrinkage_lambda=0.4, alpha=1.0),
    dict(localization_km=None, shrinkage_lambda=0.0, alpha=0.25),
    dict(localization_km=9000.0, shrinkage_lambda=0.4, alpha=0.25),
    # Past the chord's saturation, where the taper reaches zero nowhere on the sphere.
    dict(localization_km=20000.0, shrinkage_lambda=0.0, alpha=1.0),
])
def test_taper_obs_blocks_match_the_full_mask(taper):
    """The reduced blocks must equal the corresponding slices of the (D, D) mask.

    They are what a per-analysis covariance is tapered with, so any drift between the
    two regularizes the analog and static covariances differently and silently.
    """
    lats = np.linspace(-60, 60, 4).astype(np.float32)
    lons = np.linspace(-180, 120, 5).astype(np.float32)
    # Two observations in one cell and a cross-channel pair, so the shrinkage diagonal
    # and the coupling block are both exercised on repeated indices.
    gather = np.array([0, 7, 7, 20 + 3, 20 + 7])

    full = regularization_mask(lats, lons, **taper)
    blocks = taper_obs_blocks(lats, lons, gather, **taper)
    assert blocks is not None
    t_dm, t_mm = blocks
    assert np.allclose(t_dm, full[:, gather])
    assert np.allclose(t_mm, full[np.ix_(gather, gather)])


def test_taper_obs_blocks_none_when_no_taper_active():
    lats = np.linspace(-60, 60, 4).astype(np.float32)
    lons = np.linspace(-180, 120, 5).astype(np.float32)
    assert taper_obs_blocks(lats, lons, np.array([0, 3]), localization_km=None,
                            shrinkage_lambda=0.0, alpha=1.0) is None
