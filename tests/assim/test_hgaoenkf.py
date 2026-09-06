"""Tests for the hybrid gain analog offline EnKF (paleoreco.assim.hgaoenkf).

Three algebraic identities carry most of the weight, because each says a piece is wired up
rather than merely running: ``hybrid_w = 0`` must reproduce the mean of Sun et al.'s
AOEnKF-B, an analog ensemble as large as the pool must reproduce the static analysis
whatever the weight and whatever the selection rule, and the evidence rule at zero
background scale must reproduce the misfit rule exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.analog import (
    ANALOG_CORRELATION,
    ANALOG_CORRELATION_PERCHAN,
    ANALOG_EVIDENCE,
    ANALOG_MISFIT,
    eligible_mask,
)
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


def test_hybrid_w_zero_reproduces_the_aoenkf_b_mean(setup):
    """At weight zero the gain is purely static, but the prior mean is still the analog one.

    That reproduces the mean of Sun et al.'s AOEnKF-B, the corner of the family the hybrid
    weight interpolates from, and only the mean: their AOEnKF-B carries the climatological
    perturbations, while Eq. 5 at alpha = 0 reduces the analog ones by a static gain. The
    spread is not asserted for a second reason too, that 3DVar's posterior variance is the
    full-rank analytic diagonal where this one is the sample spread of k updated members.
    """
    m = _build(setup, hybrid_w=0.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    mu = m.pool[m.select(gain, setup["y"])].mean(axis=0)
    got = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    for res, expected in zip(got, _static_means(setup, mu)):
        assert np.allclose(res.mean_anom, expected, atol=1e-10)


@pytest.mark.parametrize("hybrid_w", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("selection", [ANALOG_MISFIT, ANALOG_CORRELATION,
                                       ANALOG_CORRELATION_PERCHAN, ANALOG_EVIDENCE])
def test_full_pool_ensemble_reproduces_the_static_analysis(setup, hybrid_w, selection):
    """Sun et al.'s N = M identity: with every state selected there is no analog left.

    The re-centred pool covariance is exactly the covariance B was built from, and the
    analog mean is zero, so neither the weight nor the rule that ordered the pool matters.
    It also needs the two covariances to share a lengthscale, which is what Sun et al.
    Table 2 arranges: their flow-dependent lengthscale widens with the ensemble and reaches
    the static one exactly where N reaches M.
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


@pytest.mark.parametrize("terms", [
    {},
    {"tendency_theta": 1.0, "tendency_lag_yr": 200.0},
    {"selection": ANALOG_EVIDENCE, "redundancy_theta": 0.5},
])
def test_estimator_is_deterministic(setup, terms):
    """No random numbers anywhere, so a lane's own draws stay aligned across estimators."""
    m = _build(setup, **terms)
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


def test_exclusion_band_is_applied_only_with_an_age(setup):
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


def test_analog_localization_defaults_to_the_static_lengthscale(setup):
    """The regression guard: ``None`` is the static covariance's own lengthscale.

    Bit equality rather than a tolerance, since the two paths must build the same taper and
    then run the same arithmetic. Without this the parameter could silently change the
    analysis at its default.
    """
    inherited = _build(setup)
    explicit = _build(setup, analog_localization_km=setup["taper"]["localization_km"])
    zero = np.zeros(setup["D"])
    for a, b in zip(
            inherited.apply_sweep(inherited.prepare_sweep(setup["gather"], setup["r"],
                                                          B_SCALES), setup["y"], zero),
            explicit.apply_sweep(explicit.prepare_sweep(setup["gather"], setup["r"],
                                                        B_SCALES), setup["y"], zero)):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert np.array_equal(a.posterior_var, b.posterior_var)


def test_analog_localization_rejects_a_silencing_lengthscale(setup):
    """A lengthscale that cannot taper must raise rather than annihilate.

    Gaspari-Cohn compares ``dist / length``, and every comparison against a non-finite
    value is false, so the mask stays at its pre-allocated zeros: the analog covariance is
    multiplied by nothing rather than localized, the hybrid weight goes inert, and every
    number stays finite. A read of a swept lengthscale out of a table is one path to it.
    """
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="analog_localization_km"):
            _build(setup, analog_localization_km=bad)


def test_a_separate_analog_localization_moves_the_analysis(setup):
    """A different lengthscale must reach the analog blocks, or the parameter is inert."""
    shared = _build(setup)
    tighter = _build(setup, analog_localization_km=2000.0)
    zero = np.zeros(setup["D"])
    got = [m.apply_sweep(m.prepare_sweep(setup["gather"], setup["r"], B_SCALES),
                         setup["y"], zero)[1].mean_anom for m in (shared, tighter)]
    assert not np.allclose(*got)


def test_a_separate_analog_localization_breaks_the_full_pool_identity(setup):
    """The companion to the N = M identity: it holds because the tapers agree.

    Giving the analog covariance its own lengthscale makes the two blocks differently
    regularized, so the full-pool ensemble no longer reproduces the static analysis. That
    is what says the parameter reaches the analog block and only the analog block.
    """
    m = _build(setup, k=N_POOL, hybrid_w=1.0, analog_localization_km=2000.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    got = m.apply_sweep(gain, setup["y"], np.zeros(setup["D"]))
    assert not any(np.allclose(res.mean_anom, expected, atol=1e-10)
                   for res, expected in zip(got, _static_means(setup)))


def test_extension_terms_at_zero_reproduce_the_published_estimator(setup):
    """The identity both extension terms rest on, asserted to the bit.

    Not merely close: the tendency rows are exactly zero at zero weight and the greedy
    draw short-circuits to the ranking, so anything but an identical analysis means one of
    them has leaked into the published estimator.
    """
    kw = dict(k=8, hybrid_w=0.75, selection=ANALOG_EVIDENCE)
    published = _build(setup, **kw)
    extended = _build(setup, **kw, tendency_theta=0.0, tendency_lag_yr=0.0,
                      redundancy_theta=0.0)
    zero = np.zeros(setup["D"])
    for a, b in zip(published.apply_sweep(published.prepare_sweep(setup["gather"], setup["r"],
                                                                 B_SCALES), setup["y"], zero),
                    extended.apply_sweep(extended.prepare_sweep(setup["gather"], setup["r"],
                                                                B_SCALES), setup["y"], zero)):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert np.array_equal(a.posterior_var, b.posterior_var)


def test_tendency_rows_difference_the_archive_across_the_lag(setup):
    """The lag is read against the age axis, so one archive step is one age step."""
    m = _build(setup, tendency_theta=2.0, tendency_lag_yr=100.0)
    members = np.array([5, 6, 7])
    raw = 0.5 * (setup["pool"][members + 1] - setup["pool"][members - 1])
    assert np.allclose(m._tendency_rows(members, None), 2.0 * (raw - raw.mean(axis=0)))


def test_tendency_skips_a_member_whose_neighbour_falls_in_the_exclusion_band(setup):
    """The band keeps the analog step off the simulation's own state at the target age.

    A neighbour a lag away can sit inside the band while the member itself is eligible, so
    without this the tendency would walk straight back into what the band excludes. Only
    bites where the archive spans the age being reconstructed.
    """
    m = _build(setup, tendency_theta=1.0, tendency_lag_yr=100.0)
    ages = setup["ages"]
    eligible = eligible_mask(ages, float(ages[15]), 150.0)      # excludes 14, 15, 16
    members = np.array([13, 20, 25])                            # 13's neighbour is 14
    assert len(m._tendency_rows(members, None)) == 3
    assert len(m._tendency_rows(members, eligible)) == 2


def test_tendency_modes_move_the_analysis(setup):
    """A weighted tendency must reach the covariance, or the term is inert."""
    kw = dict(k=8, hybrid_w=1.0, selection=ANALOG_EVIDENCE)
    zero = np.zeros(setup["D"])

    def analysis(**extra):
        m = _build(setup, **kw, **extra)
        gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
        return m.apply_sweep(gain, setup["y"], zero)[0].mean_anom

    assert not np.allclose(analysis(), analysis(tendency_theta=1.0, tendency_lag_yr=100.0))


def test_redundancy_penalty_moves_the_selected_ensemble(setup):
    """The penalty must reach selection, and only the evidence rule forms its coordinates."""
    m = _build(setup, k=8, selection=ANALOG_EVIDENCE)
    penalised = _build(setup, k=8, selection=ANALOG_EVIDENCE, redundancy_theta=2.0)
    gain = m.prepare_sweep(setup["gather"], setup["r"], B_SCALES)
    assert not np.array_equal(m.select(gain, setup["y"]),
                              penalised.select(gain, setup["y"]))
    with pytest.raises(ValueError, match="evidence rule"):
        _build(setup, k=8, selection=ANALOG_MISFIT, redundancy_theta=0.5)


def test_extension_terms_reject_invalid_arguments(setup):
    with pytest.raises(ValueError, match="tendency_theta"):
        _build(setup, tendency_theta=-1.0)
    with pytest.raises(ValueError, match="tendency_lag_yr"):
        _build(setup, tendency_lag_yr=-1.0)
    with pytest.raises(ValueError, match="positive tendency_lag_yr"):
        _build(setup, tendency_theta=1.0)
    with pytest.raises(ValueError, match="redundancy_theta"):
        _build(setup, selection=ANALOG_EVIDENCE, redundancy_theta=-0.5)


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
                      selection=ANALOG_EVIDENCE, evidence_scale=2.5,
                      analog_localization_km=3000.0)(prior, shape)
    assert m.pool.shape == (len(prior_idx), int(np.prod(shape)))
    assert np.array_equal(m.pool_ages, np.asarray(ages)[prior_idx])
    assert m.taper_meta["localization_km"] == 9000.0
    # Only the lengthscale is overridden; the rest of the taper stays the prior's.
    assert m.analog_taper_meta["localization_km"] == 3000.0
    assert m.analog_taper_meta["alpha"] == m.taper_meta["alpha"]
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


def test_a_single_lag_stack_is_the_published_tendency_term(setup):
    """The extra blocks default to empty, so the shipped term must survive untouched.

    Normalising is a no-op against the reference lag itself, and preserving the trace is
    a no-op where there is one block whose amplitude already sets it, so both switches
    have to leave a single-lag tendency exactly where it was.
    """
    kw = dict(k=8, hybrid_w=1.0, selection=ANALOG_EVIDENCE,
              tendency_theta=1.0, tendency_lag_yr=100.0)
    plain = _build(setup, **kw)
    normalised = _build(setup, **kw, tendency_normalise=True)
    members = np.array([5, 6, 7, 12])
    assert np.allclose(plain._tendency_rows(members, None),
                       normalised._tendency_rows(members, None))


def test_extra_lags_add_one_block_of_rows_each(setup):
    """Each block is the same construction at its own lag, stacked, not blended."""
    m = _build(setup, tendency_theta=1.0, tendency_lag_yr=100.0,
               tendency_extra_lags_yr=(200.0,), tendency_curvature_yr=(100.0,))
    members = np.array([5, 6, 7])
    rows = m._tendency_rows(members, None)
    assert len(rows) == 3 * len(members)
    first = 0.5 * (setup["pool"][members + 1] - setup["pool"][members - 1])
    assert np.allclose(rows[:len(members)], first - first.mean(axis=0))
    curv = (setup["pool"][members + 1] - 2.0 * setup["pool"][members]
            + setup["pool"][members - 1])
    assert np.allclose(rows[2 * len(members):], curv - curv.mean(axis=0))


def test_normalising_puts_every_block_on_the_reference_lag_amplitude(setup):
    """Without it a stack is dominated by its longest lag, which is the failure it fixes."""
    m = _build(setup, tendency_theta=1.0, tendency_lag_yr=100.0,
               tendency_extra_lags_yr=(300.0,), tendency_normalise=True)
    every = np.arange(len(setup["pool"]))
    ref = m._block_rows(every, (100.0, False), None)
    far = m._block_rows(every, (300.0, False), None)
    scaled = far * (m._block_amplitude((100.0, False))
                    / m._block_amplitude((300.0, False)))
    assert np.isclose(np.sqrt((scaled ** 2).mean()), np.sqrt((ref ** 2).mean()))


def test_a_clamped_neighbour_carries_the_block_amplitude(setup):
    """The archive's ends leave no room for a centred difference, and a damped row there
    would be a quieter mode rather than the timescale the block names.

    A first difference keeps the member and is carried over the interval that exists. A
    second difference has no such reading, so the member leaves that block instead.
    """
    m = _build(setup, tendency_theta=1.0, tendency_lag_yr=100.0,
               tendency_curvature_yr=(100.0,))
    pool = setup["pool"]
    edge, interior = np.array([0]), np.array([5])
    assert np.allclose(m._block_rows(interior, (100.0, False), None),
                       0.5 * (pool[6] - pool[4]))
    assert np.allclose(m._block_rows(edge, (100.0, False), None), pool[1] - pool[0])
    assert len(m._block_rows(interior, (100.0, True), None)) == 1
    assert len(m._block_rows(edge, (100.0, True), None)) == 0


def test_preserving_the_trace_leaves_the_whitened_observation_trace_alone(setup):
    """b_scale only keeps its meaning if the extra rows change directions, not amplitude."""
    kw = dict(k=8, hybrid_w=1.0, selection=ANALOG_EVIDENCE)
    plain = _build(setup, **kw)
    stacked = _build(setup, **kw, tendency_theta=1.0, tendency_lag_yr=100.0,
                     tendency_extra_lags_yr=(200.0, 300.0), tendency_normalise=True,
                     preserve_obs_trace=True)
    g, r = setup["gather"], setup["r"]
    gain = stacked.prepare_sweep(g, r, B_SCALES)
    selected = stacked.select(gain, setup["y"])
    members = stacked.pool[selected]
    base = members - members.mean(axis=0)
    dev = np.vstack([base, stacked._tendency_rows(selected, None)])
    dev = dev * stacked._trace_rescale(base, dev, gain)
    assert np.isclose(((dev[:, g] ** 2) / r[None, :]).sum(),
                      ((base[:, g] ** 2) / r[None, :]).sum())
    # and the directions really did change, or the rescale is hiding an inert term
    zero = np.zeros(setup["D"])
    assert not np.allclose(
        plain.apply_sweep(plain.prepare_sweep(g, r, B_SCALES), setup["y"], zero)[0].mean_anom,
        stacked.apply_sweep(gain, setup["y"], zero)[0].mean_anom)


@pytest.mark.parametrize("kw", [
    dict(tendency_theta=1.0, tendency_lag_yr=100.0, tendency_extra_lags_yr=(0.0,)),
    dict(tendency_theta=1.0, tendency_lag_yr=100.0, tendency_curvature_yr=(-50.0,)),
    dict(tendency_theta=0.0, tendency_lag_yr=0.0, tendency_extra_lags_yr=(100.0,)),
])
def test_a_stack_that_would_be_silently_inert_is_rejected(setup, kw):
    """A zero lag differences a state against itself and a zero theta erases the stack."""
    with pytest.raises(ValueError):
        _build(setup, **kw)


@pytest.mark.parametrize("kw", [
    dict(tendency_extra_lags_yr=(100.0, 200.0)),          # repeats the reference lag
    dict(tendency_extra_lags_yr=(200.0, 200.0)),
    dict(tendency_curvature_yr=(200.0, 200.0)),
])
def test_a_repeated_lag_is_rejected_rather_than_stacked_twice(setup, kw):
    """Blocks are keyed by lag for lookup but stacked as a list, so a repeat doubles it.

    Nothing downstream would surface that: the covariance weights one timescale twice,
    which reads as a lag that matters more rather than as a configuration error.
    """
    with pytest.raises(ValueError, match="repeat"):
        _build(setup, tendency_theta=1.0, tendency_lag_yr=100.0, **kw)
