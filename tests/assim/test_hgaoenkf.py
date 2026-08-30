"""Tests for the hybrid gain analog offline EnKF (paleoreco.assim.hgaoenkf).

Three algebraic identities carry most of the weight, because each says a piece is wired up
rather than merely running: ``hybrid_w = 0`` must reproduce Sun et al.'s AOEnKF-B, an analog
ensemble as large as the pool must reproduce the static analysis whatever the weight and
whatever the selection rule, and the evidence rule at zero background scale must reproduce
the misfit rule exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.analog import ANALOG_EVIDENCE, ANALOG_MISFIT, ANALOG_WINDOW
from paleoreco.assim.background import background_covariance
from paleoreco.assim.hgaoenkf import HGAOEnKF, make_hgaoenkf
from paleoreco.assim.method import Observations
from paleoreco.assim.priors import build_prior, regularization_mask
from paleoreco.assim.threedvar import ThreeDVar

B_SCALES = np.array([0.5, 1.0, 5.0])
N_POOL, N_LAT, N_LON = 30, 4, 5


@pytest.fixture
def grid():
    return (np.linspace(-60, 60, N_LAT).astype(np.float32),
            np.linspace(-180, 120, N_LON).astype(np.float32))


@pytest.fixture
def setup(grid):
    """A pool, the covariance built from it, and one observation network."""
    lats, lons = grid
    rng = np.random.default_rng(2)
    shape = (2, N_LAT, N_LON)
    D = int(np.prod(shape))
    cube = rng.normal(size=(N_POOL, *shape))
    pool = cube.reshape(N_POOL, -1) - cube.reshape(N_POOL, -1).mean(axis=0)
    taper = dict(localization_km=9000.0, shrinkage_lambda=0.25, alpha=0.5)
    B = (pool.T @ pool) / (N_POOL - 1) * regularization_mask(lats, lons, **taper)
    ages = (30_000 + 100 * np.arange(N_POOL)).astype(np.int64)
    gather = np.array([0, 7, 13, D // 2 + 2, D // 2 + 11])
    r = np.array([0.5, 1.0, 2.0, 0.75, 1.5])
    y = rng.normal(size=len(gather))
    return dict(lats=lats, lons=lons, shape=shape, D=D, pool=pool, B=B, ages=ages,
                gather=gather, r=r, y=y, taper=taper)


def _build(setup, **kw):
    kw.setdefault("k", 8)
    kw.setdefault("hybrid_w", 0.5)
    return HGAOEnKF(setup["pool"], setup["ages"], setup["B"], setup["shape"],
                    setup["lats"], setup["lons"], taper_meta=setup["taper"], **kw)


def _static_means(setup, background=None):
    """Mean analyses from the static covariance alone, over the ``b_scale`` sweep."""
    tv = ThreeDVar(setup["B"], setup["shape"])
    gain = tv.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    bg = np.zeros(setup["D"]) if background is None else background
    return [res.mean_anom for res in tv.apply_sweep(gain, setup["y"], bg)]


def test_hybrid_w_zero_is_aoenkf_b(setup):
    """At weight zero the gain is purely static, but the prior mean is still the analog one.

    That is Sun et al.'s AOEnKF-B, the corner of the family the hybrid weight interpolates
    from. Only the mean is asserted: 3DVar's posterior variance is the full-rank analytic
    diagonal, this one is the sample spread of k updated members.
    """
    m = _build(setup, hybrid_w=0.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    mu = m.pool[m.select(gain, setup["y"])].mean(axis=0)
    got = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    for res, expected in zip(got, _static_means(setup, mu)):
        assert np.allclose(res.mean_anom, expected, atol=1e-10)


@pytest.mark.parametrize("hybrid_w", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("selection", [ANALOG_MISFIT, ANALOG_EVIDENCE])
def test_full_pool_ensemble_reproduces_the_static_analysis(setup, hybrid_w, selection):
    """Sun et al.'s N = M identity: with every state selected there is no analog left.

    The re-centred pool covariance is exactly the covariance B was built from, and the
    analog mean is zero, so neither the weight nor the rule that ordered the pool matters.
    """
    m = _build(setup, k=N_POOL, hybrid_w=hybrid_w, selection=selection)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    got = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    for res, expected in zip(got, _static_means(setup)):
        assert np.allclose(res.mean_anom, expected, atol=1e-10)


def test_pool_covariance_is_the_background_covariance(setup):
    """Guards the identity above: the pool and B must describe the same states."""
    pool = setup["pool"]
    cube = pool.reshape(N_POOL, *setup["shape"])
    assert np.allclose((pool.T @ pool) / (N_POOL - 1),
                       background_covariance(cube, np.arange(N_POOL)))


def test_hybrid_weight_moves_the_analysis(setup):
    """A nonzero weight must actually change the mean, or the analog side is inert."""
    gains = {}
    for w in (0.0, 0.5):
        m = _build(setup, hybrid_w=w)
        gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
        gains[w] = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))[1].mean_anom
    assert not np.allclose(gains[0.0], gains[0.5])


def test_pure_analog_spread_shrinks_as_a_square_root_filter_must(setup):
    """At weight 1 the reduced gain matches the covariance the deviations came from.

    That is the square-root guarantee, and it is the only weight at which it holds: a
    blended gain reduces analog deviations by a partly static factor, so the mean and the
    spread come from different mixtures and the spread can grow slightly.
    """
    m = _build(setup, hybrid_w=1.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    dev = m.pool[m.select(gain, setup["y"])]
    prior_var = (dev - dev.mean(axis=0)).var(axis=0, ddof=1)
    for b, res in zip(B_SCALES, m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))):
        assert (res.posterior_var.ravel() <= b * prior_var + 1e-9).all()


def test_posterior_variance_is_finite_and_non_negative(setup):
    m = _build(setup)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    for res in m.apply_sweep(gain, setup["y"], np.zeros(setup["D"])):
        v = res.posterior_var.ravel()
        assert np.isfinite(v).all() and (v >= 0.0).all()


def test_estimator_is_deterministic(setup):
    """No random numbers anywhere, so a lane's own draws stay aligned across estimators."""
    m = _build(setup)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    first = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    second = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    for a, b in zip(first, second):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert np.array_equal(a.posterior_var, b.posterior_var)


def test_selection_uses_the_observations_and_is_reported(setup):
    m = _build(setup)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    sel = m.select(gain, setup["y"])
    assert sel.shape == (m.k,) and len(set(sel.tolist())) == m.k
    assert not np.array_equal(sel, m.select(gain, -setup["y"]))


def test_exclusion_band_needs_an_age_and_is_applied(setup):
    m = _build(setup, exclude_yr=500.0)
    age = float(setup["ages"][N_POOL // 2])
    # Without an age there is nothing to exclude, which is the split lanes' case.
    assert m.prepare_sweep(setup["gather"], setup["r"], B_SCALES).eligible is None
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES, age=age)
    assert np.all(np.abs(setup["ages"][m.select(gain, setup["y"])] - age) >= 500.0)


def test_evidence_at_zero_scale_is_the_misfit_estimator(setup):
    """End-to-end form of the analog-level identity, and the regression guard for the rule.

    Bit equality rather than a tolerance: the two paths must select the same members and
    then run the same arithmetic, so any drift in either is a real change.
    """
    misfit = _build(setup, selection=ANALOG_MISFIT)
    evidence = _build(setup, selection=ANALOG_EVIDENCE, evidence_scale=0.0)
    zero = np.zeros(setup["D"])
    for m in (misfit, evidence):
        m.gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    assert np.array_equal(misfit.select(misfit.gain, setup["y"]),
                          evidence.select(evidence.gain, setup["y"]))
    for a, b in zip(misfit.apply_sweep(misfit.gain, setup["y"], zero),
                    evidence.apply_sweep(evidence.gain, setup["y"], zero)):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert np.array_equal(a.posterior_var, b.posterior_var)


def test_evidence_selection_moves_the_ensemble_at_the_default_scale(setup):
    """A nonzero background scale must change which states are chosen."""
    misfit = _build(setup, selection=ANALOG_MISFIT)
    evidence = _build(setup, selection=ANALOG_EVIDENCE)
    gain_m = misfit.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    gain_e = evidence.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    assert not np.array_equal(misfit.select(gain_m, setup["y"]),
                              evidence.select(gain_e, setup["y"]))


def test_window_selection_ignores_the_observations(setup):
    m = _build(setup, selection=ANALOG_WINDOW)
    age = float(setup["ages"][N_POOL // 2])
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES, age=age)
    assert np.array_equal(m.select(gain, setup["y"]), m.select(gain, -setup["y"]))
    with pytest.raises(ValueError, match="needs the age"):
        m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)


def test_analyze_satisfies_the_method_contract(setup):
    m = _build(setup)
    obs = Observations(gather=setup["gather"], y_anom=setup["y"], sse=setup["r"])
    res = m.analyze(obs, np.zeros(setup["D"]))
    assert res.mean_anom.shape == setup["shape"]
    assert res.posterior_var.shape == setup["shape"]


def test_factory_pool_matches_the_prior_it_is_built_from(cube, ages, lats, lons, valid):
    """The factory recovers the pool from prior.ages, so the two cannot drift apart."""
    shape = (2, len(lats), len(lons))
    prior_idx = np.arange(0, len(ages), 2)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid, localization_km=9000.0)
    m = make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=0.5,
                      selection=ANALOG_EVIDENCE, evidence_scale=2.5)(prior, shape)
    assert m.pool.shape == (len(prior_idx), int(np.prod(shape)))
    assert np.array_equal(m.pool_ages, np.asarray(ages)[prior_idx])
    assert m.taper_meta["localization_km"] == 9000.0
    assert m.selection == ANALOG_EVIDENCE and m.evidence_scale == 2.5


def test_factory_rejects_invalid_arguments(cube, ages, lats, lons, valid):
    shape = (2, len(lats), len(lons))
    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid)
    with pytest.raises(ValueError, match="at least 2 members"):
        make_hgaoenkf(cube, ages, lats, lons, k=1, hybrid_w=0.5)(prior, shape)
    with pytest.raises(ValueError, match="hybrid_w"):
        make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=1.5)(prior, shape)
    with pytest.raises(ValueError, match="unknown selection"):
        make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=0.5, selection="nope")(prior, shape)
    with pytest.raises(ValueError, match="evidence_scale"):
        make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=0.5,
                      evidence_scale=-1.0)(prior, shape)
