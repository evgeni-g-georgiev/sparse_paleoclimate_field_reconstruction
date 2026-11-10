"""Tests for the hybrid gain analog offline EnKF (paleoreco.assim.hgaoenkf).

Algebraic identities carry most of the weight, because each says a piece is wired up rather
than merely running: ``hybrid_w = 0`` must reproduce the mean of Sun et al.'s AOEnKF-B; an
analog ensemble as large as the pool must reproduce the static analysis whatever the weight
and the selection rule; a zero tendency share must reproduce the estimator without MTA to
the bit; and every share must hold the whitened observation-space trace where it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.analog import ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN
from paleoreco.assim.background import background_covariance
from paleoreco.assim.ensrf import sqrt_gain_apply, whitened_block
from paleoreco.assim.hgaoenkf import HGAOEnKF, TendencyStack, make_hgaoenkf
from paleoreco.assim.priors import build_prior, regularization_mask
from paleoreco.assim.threedvar import ThreeDVar

B_SCALES = np.array([0.5, 1.0, 5.0])
N_POOL, N_LAT, N_LON = 30, 4, 5
STACK = TendencyStack(reference=100.0, first=(100.0, 200.0), second=(200.0,))
SHARES = (0.0, 0.25, 0.5, 0.9, 1.0)


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


def _sweep(m, setup, **kw):
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    return m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]), **kw)


def _whitened_trace(dev, setup):
    return float(((dev[:, setup["gather"]] ** 2) / setup["r"][None, :]).sum())


def _analog_posterior(m, gain, y, b):
    """``(dev, post)`` at pure analog weight, where ``post`` is Eq. 5's updated deviations."""
    _, dev = m.deviations(m.select(gain, y), gain)
    h_dev = dev[:, gain.gather].T
    P_obs = (dev.T @ h_dev.T) / (m.k - 1.0)
    S_obs = (h_dev @ h_dev.T) / (m.k - 1.0)
    if gain.taper is not None:
        P_obs, S_obs = P_obs * gain.taper[0], S_obs * gain.taper[1]
    block = whitened_block(P_obs, S_obs, gain.r_diag)
    return dev, dev.T - sqrt_gain_apply(block, b, h_dev)


def _static_means(setup, background=None):
    """Mean analyses from the static covariance alone, over the ``b_scale`` sweep."""
    tv = ThreeDVar(setup["B"], setup["shape"])
    gain = tv.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    bg = np.zeros(setup["D"]) if background is None else background
    return [res.mean_anom for res in tv.apply_sweep(gain, setup["y"], bg)]


# ---------------------------------------------------------------------------
# The hybrid gain.
# ---------------------------------------------------------------------------
def test_hybrid_w_zero_reproduces_the_aoenkf_b_mean(setup):
    """At weight zero the gain is purely static, but the prior mean is still the analog one.

    That reproduces the mean of Sun et al.'s AOEnKF-B, and only the mean: their AOEnKF-B
    carries the climatological perturbations, while Eq. 5 at alpha = 0 reduces the analog
    ones by a static gain.
    """
    m = _build(setup, hybrid_w=0.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    mu = m.pool[m.select(gain, setup["y"])].mean(axis=0)
    got = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    for res, expected in zip(got, _static_means(setup, mu)):
        assert np.allclose(res.mean_anom, expected, atol=1e-10)


@pytest.mark.parametrize("hybrid_w", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rule", [dict(), dict(kappa=1.0),
                                  dict(selection=ANALOG_CORRELATION),
                                  dict(selection=ANALOG_CORRELATION_PERCHAN)])
def test_full_pool_ensemble_reproduces_the_static_analysis(setup, hybrid_w, rule):
    """Sun et al.'s N = M identity: with every state selected there is no analog left.

    The re-centred pool covariance is exactly the covariance B was built from, and the
    analog mean is zero, so neither the weight nor the rule that ordered the pool matters.
    It also needs the two covariances to share a taper, which they do.
    """
    m = _build(setup, k=N_POOL, hybrid_w=hybrid_w, **rule)
    for res, expected in zip(_sweep(m, setup), _static_means(setup)):
        assert np.allclose(res.mean_anom, expected, atol=1e-10)


def test_pool_covariance_is_the_background_covariance(setup):
    """Guards the identity above: the pool and B must describe the same states."""
    pool = setup["pool"]
    cube = pool.reshape(N_POOL, *setup["shape"])
    assert np.allclose((pool.T @ pool) / (N_POOL - 1),
                       background_covariance(cube, np.arange(N_POOL)))


def test_hybrid_weight_moves_the_analysis(setup):
    """A nonzero weight must actually change the mean, or the analog side is inert."""
    means = [_sweep(_build(setup, hybrid_w=w), setup)[1].mean_anom for w in (0.0, 0.5)]
    assert not np.allclose(*means)


@pytest.mark.parametrize("mta", [{}, dict(stack=STACK, tendency_share=0.5)])
def test_pure_analog_spread_shrinks_as_a_square_root_filter_must(setup, mta):
    """At weight 1 the reduced gain matches the covariance the deviations came from.

    That is the square-root guarantee, and it is the only weight at which it holds. Both
    priors are normalised by the members, so a posterior counting the stacked rows instead
    would pass this by understating the spread rather than by shrinking it.
    """
    m = _build(setup, hybrid_w=1.0, **mta)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    _, dev = m.deviations(m.select(gain, setup["y"]), gain)
    prior_var = (dev ** 2).sum(axis=0) / (m.k - 1.0)
    for b, res in zip(B_SCALES, m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))):
        assert (res.posterior_var.ravel() <= b * prior_var + 1e-9).all()


def test_posterior_variance_is_finite_and_non_negative(setup):
    for res in _sweep(_build(setup, stack=STACK, tendency_share=0.5), setup):
        v = res.posterior_var.ravel()
        assert np.isfinite(v).all() and (v >= 0.0).all()


@pytest.mark.parametrize("mta", [{}, dict(stack=STACK, tendency_share=0.75)])
def test_the_mean_only_analysis_gives_the_full_analysis_mean(setup, mta):
    """A grid scan reads the mean alone, so skipping the spread must not move it at all."""
    m = _build(setup, kappa=1.0, **mta)
    for a, b in zip(_sweep(m, setup), _sweep(m, setup, mean_only=True)):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert b.posterior_var is None and b.posterior_cross_var is None


@pytest.mark.parametrize("mta", [{}, dict(kappa=1.0, stack=STACK, tendency_share=0.5)])
def test_estimator_is_deterministic(setup, mta):
    """No random numbers anywhere, so a lane's own draws stay aligned across estimators."""
    m = _build(setup, **mta)
    for a, b in zip(_sweep(m, setup), _sweep(m, setup)):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert np.array_equal(a.posterior_var, b.posterior_var)


# ---------------------------------------------------------------------------
# Selection.
# ---------------------------------------------------------------------------
def test_selection_uses_the_observations_and_is_reported(setup):
    m = _build(setup)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    sel = m.select(gain, setup["y"])
    assert sel.shape == (m.k,) and len(set(sel.tolist())) == m.k
    assert not np.array_equal(sel, m.select(gain, -setup["y"]))


def test_a_positive_kappa_moves_the_ensemble(setup):
    """A nonzero background scale must change which states are chosen."""
    gain = _build(setup).prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    assert not np.array_equal(_build(setup).select(gain, setup["y"]),
                              _build(setup, kappa=1.0).select(gain, setup["y"]))


@pytest.mark.parametrize("selection", [ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN])
def test_correlation_selection_reads_the_observations(setup, selection):
    m = _build(setup, selection=selection)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    sel = m.select(gain, setup["y"])
    assert sel.shape == (m.k,) and len(set(sel.tolist())) == m.k
    # Eq. 6 normalises, so reversing the sign of every observation is the one change it
    # does see: the best-correlated candidates become the worst.
    assert not np.array_equal(sel, m.select(gain, -setup["y"]))


def test_the_two_correlation_readings_differ_across_channels(setup):
    """The network spans both channels, so the pooled and per-channel forms must diverge."""
    pooled = _build(setup, selection=ANALOG_CORRELATION)
    per_chan = _build(setup, selection=ANALOG_CORRELATION_PERCHAN)
    gain = pooled.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    assert len(set(gain.obs_channel.tolist())) == 2
    assert not np.array_equal(pooled.select(gain, setup["y"]),
                              per_chan.select(gain, setup["y"]))


# ---------------------------------------------------------------------------
# The tendency stack.
# ---------------------------------------------------------------------------
def test_tendency_rows_difference_the_archive_across_the_lag(setup):
    """The lag is read against the age axis, so one archive step is one age step."""
    m = _build(setup, stack=TendencyStack(reference=100.0, first=(100.0,)))
    members = np.array([5, 6, 7])
    raw = 0.5 * (setup["pool"][members + 1] - setup["pool"][members - 1])
    assert np.allclose(m._tendency_rows(members), raw - raw.mean(axis=0))


def test_each_lag_adds_one_centred_block(setup):
    """Each block is the same construction at its own lag, stacked, not blended."""
    m = _build(setup, stack=TendencyStack(reference=100.0, first=(100.0, 200.0),
                                          second=(100.0,)))
    members = np.array([5, 6, 7])
    rows = m._tendency_rows(members)
    assert len(rows) == 3 * len(members)
    first = 0.5 * (setup["pool"][members + 1] - setup["pool"][members - 1])
    assert np.allclose(rows[:len(members)], first - first.mean(axis=0))
    curv = (setup["pool"][members + 1] - 2.0 * setup["pool"][members]
            + setup["pool"][members - 1])
    scale = m.tendency_scale[(100.0, True)]
    assert np.allclose(rows[2 * len(members):], scale * (curv - curv.mean(axis=0)))


def test_every_block_is_put_on_the_reference_lag_amplitude(setup):
    """Without it a stack is dominated by its longest lag."""
    m = _build(setup, stack=TendencyStack(reference=100.0, first=(100.0, 300.0)))
    every = np.arange(len(setup["pool"]))
    ref = m._block_rows(every, (100.0, False))
    far = m._block_rows(every, (300.0, False)) * m.tendency_scale[(300.0, False)]
    assert m.tendency_scale[(100.0, False)] == 1.0
    assert np.isclose(np.sqrt((far ** 2).mean()), np.sqrt((ref ** 2).mean()))


def test_a_clamped_neighbour_carries_the_block_amplitude(setup):
    """The archive's ends leave no room for a centred difference.

    A first difference keeps the member and is carried over the interval that exists. A
    second difference has no such reading, so the member leaves that block instead.
    """
    m = _build(setup, stack=TendencyStack(reference=100.0, first=(100.0,), second=(100.0,)))
    pool = setup["pool"]
    edge, interior = np.array([0]), np.array([5])
    assert np.allclose(m._block_rows(interior, (100.0, False)), 0.5 * (pool[6] - pool[4]))
    assert np.allclose(m._block_rows(edge, (100.0, False)), pool[1] - pool[0])
    assert len(m._block_rows(interior, (100.0, True))) == 1
    assert len(m._block_rows(edge, (100.0, True))) == 0


def test_a_zero_share_reproduces_the_estimator_without_mta_exactly(setup):
    """``w = 0`` is evidence-only HGAOEnKF, mean and spread, to the bit."""
    kw = dict(k=8, hybrid_w=0.75, kappa=1.0)
    plain, zero = _build(setup, **kw), _build(setup, **kw, stack=STACK, tendency_share=0.0)
    for a, b in zip(_sweep(plain, setup), _sweep(zero, setup)):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert np.array_equal(a.posterior_var, b.posterior_var)
        assert np.array_equal(a.posterior_cross_var, b.posterior_cross_var)


@pytest.mark.parametrize("share", SHARES)
def test_every_share_holds_the_whitened_observation_space_trace(setup, share):
    """``w`` reallocates variance between the members and the tendency rows, never adds it."""
    m = _build(setup, kappa=1.0, stack=STACK, tendency_share=share)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    selected = m.select(gain, setup["y"])
    members = m.pool[selected]
    _, dev = m.deviations(selected, gain)
    assert np.isclose(_whitened_trace(dev, setup),
                      _whitened_trace(members - members.mean(axis=0), setup))
    if 0.0 < share < 1.0:
        assert len(dev) > m.k


def test_a_unit_share_keeps_only_the_tendency_rows_about_the_analog_mean(setup):
    m = _build(setup, hybrid_w=1.0, stack=STACK, tendency_share=1.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    selected = m.select(gain, setup["y"])
    mu, dev = m.deviations(selected, gain)
    tend = m._tendency_rows(selected)
    assert np.allclose(mu, m.pool[selected].mean(axis=0))
    assert dev.shape == tend.shape
    assert np.allclose(dev / np.linalg.norm(dev), tend / np.linalg.norm(tend))


def test_with_no_tendency_variance_the_deviations_are_the_members_alone(setup):
    """A lag longer than the archive leaves every member the same row, which centring zeros."""
    m = _build(setup, stack=TendencyStack(reference=10_000.0, first=(10_000.0,)),
               tendency_share=0.5)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    selected = m.select(gain, setup["y"])
    members = m.pool[selected]
    _, dev = m.deviations(selected, gain)
    assert np.array_equal(dev, members - members.mean(axis=0))


def test_the_tendency_rows_move_the_analysis(setup):
    """A positive share must reach the covariance, or the term is inert."""
    kw = dict(hybrid_w=1.0, kappa=1.0)
    plain = _sweep(_build(setup, **kw), setup)[0].mean_anom
    mta = _sweep(_build(setup, **kw, stack=STACK, tendency_share=0.5), setup)[0].mean_anom
    assert not np.allclose(plain, mta)


def test_posterior_variance_counts_the_members_not_the_stacked_rows(setup):
    """The prior blocks divide by k - 1, so the posterior must divide by the same thing."""
    m = _build(setup, hybrid_w=1.0, stack=STACK, tendency_share=0.5)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    for b, res in zip(B_SCALES, m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))):
        dev, post = _analog_posterior(m, gain, setup["y"], b)
        assert dev.shape[0] > m.k                      # the stack is taller than the members
        assert np.allclose(res.posterior_var.ravel(),
                           b * (post ** 2).sum(axis=1) / (m.k - 1.0))
        assert not np.allclose(res.posterior_var.ravel(), b * post.var(axis=1, ddof=1))


def test_an_unstacked_posterior_variance_is_the_spread_over_the_members(setup):
    m = _build(setup, hybrid_w=1.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    for b, res in zip(B_SCALES, m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))):
        _, post = _analog_posterior(m, gain, setup["y"], b)
        assert np.allclose(res.posterior_var.ravel(), b * post.var(axis=1, ddof=1),
                           rtol=1e-12)


def test_posterior_cross_variance_is_the_covariance_between_the_two_channels(setup):
    """Being a covariance it takes either sign, so it cannot be a variance by mistake."""
    m = _build(setup, hybrid_w=1.0, stack=STACK, tendency_share=0.5)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    n_cells = N_LAT * N_LON
    for b, res in zip(B_SCALES, m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))):
        _, post = _analog_posterior(m, gain, setup["y"], b)
        cross = res.posterior_cross_var
        assert cross.shape == (N_LAT, N_LON)
        assert np.allclose(cross.ravel(),
                           b * (post[:n_cells] * post[n_cells:]).sum(axis=1) / (m.k - 1.0))
        assert (cross < 0).any() and (cross > 0).any()
        var = res.posterior_var.reshape(2, -1)
        assert (np.abs(cross.ravel()) <= np.sqrt(var[0] * var[1]) + 1e-12).all()


# ---------------------------------------------------------------------------
# Construction.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw, match", [
    (dict(reference=100.0, first=(100.0, 0.0)), "positive"),
    (dict(reference=100.0, first=(100.0,), second=(-50.0,)), "positive"),
    (dict(reference=100.0, first=(100.0, 100.0)), "repeat"),
    (dict(reference=100.0, first=(100.0,), second=(200.0, 200.0)), "repeat"),
    (dict(reference=300.0, first=(100.0, 200.0)), "reference"),
])
def test_a_malformed_stack_is_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        TendencyStack(**kw)


def test_invalid_estimator_arguments_are_rejected(setup):
    with pytest.raises(ValueError, match="kappa"):
        _build(setup, kappa=-1.0)
    with pytest.raises(ValueError, match="evidence rule"):
        _build(setup, selection=ANALOG_CORRELATION, kappa=1.0)
    with pytest.raises(ValueError, match="tendency_share"):
        _build(setup, stack=STACK, tendency_share=1.5)
    with pytest.raises(ValueError, match="stack"):
        _build(setup, tendency_share=0.5)


def test_factory_pool_matches_the_prior_it_is_built_from(cube, ages, lats, lons, valid):
    """The factory recovers the pool from prior.ages, so the two cannot drift apart."""
    shape = (2, len(lats), len(lons))
    prior_idx = np.arange(0, len(ages), 2)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid, localization_km=9000.0)
    m = make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=0.5, kappa=2.5)(prior, shape)
    assert m.pool.shape == (len(prior_idx), int(np.prod(shape)))
    assert np.array_equal(m.pool_ages, np.asarray(ages)[prior_idx])
    assert m.taper_meta["localization_km"] == 9000.0
    assert m.kappa == 2.5


def test_factory_rejects_invalid_arguments(cube, ages, lats, lons, valid):
    shape = (2, len(lats), len(lons))
    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid)
    with pytest.raises(ValueError, match="at least 2 members"):
        make_hgaoenkf(cube, ages, lats, lons, k=1, hybrid_w=0.5)(prior, shape)
    with pytest.raises(ValueError, match="hybrid_w"):
        make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=1.5)(prior, shape)
    with pytest.raises(ValueError, match="unknown selection"):
        make_hgaoenkf(cube, ages, lats, lons, k=4, hybrid_w=0.5, selection="nope")(prior, shape)
