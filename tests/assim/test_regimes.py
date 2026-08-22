"""Tests for the regime-mixture prior (paleoreco.assim.regimes, regime_sampler).

Three properties carry the method: one regime reproduces 3DVar in closed form, the guided
reverse process reproduces it too within Monte Carlo error, and with several regimes the
exact and guided arms agree. The first pins the prior, the second the sampler, and the
third is what lets a difference from 3DVar be attributed to the prior rather than to the
inference.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.linalg import cho_factor, cho_solve

from paleoreco.assim.regimes import (
    build_regime_mixture,
    exact_posterior,
    exact_posterior_mean,
    partition_states,
    prior_sample,
)
from paleoreco.assim.regime_sampler import RegimeSampler
from paleoreco.assim.threedvar import ThreeDVar
from paleoreco.models.regime_score import MixtureDenoiser, build_regime_denoiser

# The residual network downsamples, so the test grid's axes have to be even.
SHAPE = (2, 4, 4)
D = int(np.prod(SHAPE))
GATHER = np.array([0, 3, 7, 11, 17, 20])


def _states(n=60, seed=0, separation=3.0):
    """Prior states with a genuinely separated regime, so a mixture has work to do."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D)) @ rng.standard_normal((D, D)) / np.sqrt(D)
    x[: n // 3] += separation * rng.standard_normal(D)
    return x


def _obs(seed=0):
    rng = np.random.default_rng(seed + 100)
    return GATHER, 0.4 + 0.1 * rng.random(len(GATHER)), rng.standard_normal(len(GATHER))


def _mixture(x, j, rho, seed=0):
    return build_regime_mixture(x, partition_states(x, j, seed=seed), SHAPE,
                                np.ones(SHAPE[1:], bool), rho=rho)


def _threedvar_mean(x, g, r, y, b):
    """The gain-form analysis this method must reproduce at one regime."""
    B = np.cov(x - x.mean(0), rowvar=False, ddof=1)
    chol = cho_factor(b * B[np.ix_(g, g)] + np.diag(r))
    return B, b * (B[:, g] @ cho_solve(chol, y))


# ---------------------------------------------------------------------------
# Exact nesting: one regime is 3DVar.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("b_scale", [0.5, 2.0, 5.0])
def test_one_regime_reproduces_3dvar_exactly(b_scale):
    """Against the shipped estimator, not a re-derivation of its formula."""
    x = _states()
    g, r, y = _obs()
    B = np.cov(x - x.mean(0), rowvar=False, ddof=1)
    ref = ThreeDVar(b_scale * B, SHAPE).analyze(
        type("O", (), {"gather": g, "y_anom": y, "sse": r})(), np.zeros(D)).mean_anom
    mix = _mixture(x, 1, rho=1.0)
    assert np.allclose(mix.mu[0], 0.0, atol=1e-10)          # nesting needs a zero mean
    assert np.allclose(exact_posterior_mean(mix, g, r, y, b_scale), ref.ravel(), atol=1e-9)


# ---------------------------------------------------------------------------
# The mixture denoiser.
# ---------------------------------------------------------------------------
def test_denoiser_limits():
    """``D -> x`` as the noise vanishes, and ``-> sum_j w_j m_j`` as it dominates."""
    x = _states()
    mix = _mixture(x, 3, rho=0.3)
    md = MixtureDenoiser(mix, b_scale=1.0).double()
    xt = torch.as_tensor(x[:5].reshape(5, *SHAPE), dtype=torch.float64)
    assert np.allclose(md(xt, 1e-5).reshape(5, -1).numpy(), x[:5], atol=1e-5)
    assert np.allclose(md(xt, 1e5).reshape(5, -1).numpy(),
                       (mix.w @ mix.mu)[None], atol=1e-4)


def test_denoiser_at_one_regime_is_the_analytic_gaussian_denoiser():
    x = _states()
    mix = _mixture(x, 1, rho=1.0)
    b, sigma = 2.0, 0.7
    md = MixtureDenoiser(mix, b_scale=b).double()
    xt = torch.as_tensor(x[:4].reshape(4, *SHAPE), dtype=torch.float64)
    got = md(xt, sigma).reshape(4, -1).numpy()
    C = b * mix.cov[0]
    ref = (C @ np.linalg.solve(C + sigma ** 2 * np.eye(D), x[:4].T)).T
    assert np.allclose(got, ref, atol=1e-8)


def test_denoiser_accepts_a_per_sample_noise_level():
    """The EDM training loop passes one sigma per sample; the reverse process a scalar."""
    x = _states()
    md = MixtureDenoiser(_mixture(x, 3, rho=0.3), b_scale=1.0).double()
    xt = torch.as_tensor(x[:4].reshape(4, *SHAPE), dtype=torch.float64)
    per_sample = md(xt, torch.full((4,), 0.5, dtype=torch.float64))
    assert torch.allclose(per_sample, md(xt, 0.5), atol=1e-10)


def test_untrained_residual_is_exactly_the_mixture():
    """A zero-initialised output layer makes the residual denoiser the mixture denoiser."""
    x = _states()
    mix = _mixture(x, 3, rho=0.3)
    model = build_regime_denoiser(mix, {"base_channels": 8, "depth": 1}).double()
    xt = torch.as_tensor(x[:3].reshape(3, *SHAPE), dtype=torch.float64)
    assert torch.allclose(model(xt, 0.6), model.mixture(xt, 0.6), atol=1e-12)


# ---------------------------------------------------------------------------
# The sampler: both arms, and the agreement between them.
# ---------------------------------------------------------------------------
def test_exact_ensemble_mean_matches_the_closed_form():
    x = _states()
    g, r, y = _obs()
    mix = _mixture(x, 3, rho=0.3)
    s = RegimeSampler(mix, b_scale=1.0, inference="exact", device="cpu")
    ens = s.sample_posterior(g, y, r, 1.0, 20000, seed=1).reshape(20000, -1)
    ref = exact_posterior_mean(mix, g, r, y, 1.0)
    tol = 4.0 * ens.std(0) / np.sqrt(len(ens))
    assert np.all(np.abs(ens.mean(0) - ref) < tol)


def test_guided_reverse_process_reproduces_3dvar_at_one_regime():
    """The reverse process and the guidance, gated separately from the prior."""
    x = _states()
    g, r, y = _obs()
    mix = _mixture(x, 1, rho=1.0)
    B, ref = _threedvar_mean(x, g, r, y, 1.0)
    s = RegimeSampler(mix, b_scale=1.0, inference="guided", n_steps=64, max_batch=4000,
                      device="cpu")
    ens = s.sample_posterior(g, y, r, 1.0, 4000, seed=1).reshape(4000, -1)

    chol = cho_factor(B[np.ix_(g, g)] + np.diag(r))
    post_sd = np.sqrt(np.diag(B - B[:, g] @ cho_solve(chol, B[:, g].T)))
    assert np.abs(ens.mean(0) - ref).max() < 0.10 * post_sd.mean()
    assert 0.95 < (ens.std(0) / post_sd).mean() < 1.05


def test_guided_agrees_with_exact_on_the_same_prior():
    """Both arms describe the same prior when the residual is off, so they must agree."""
    n_regimes, rho = 3, 0.3
    x = _states()
    g, r, y = _obs()
    mix = _mixture(x, n_regimes, rho=rho)
    ref = exact_posterior_mean(mix, g, r, y, 1.0)

    ex = RegimeSampler(mix, b_scale=1.0, inference="exact", device="cpu")
    gu = RegimeSampler(mix, b_scale=1.0, inference="guided", n_steps=64, max_batch=4000,
                       device="cpu")
    e = ex.sample_posterior(g, y, r, 1.0, 8000, seed=2).reshape(8000, -1)
    u = gu.sample_posterior(g, y, r, 1.0, 4000, seed=2).reshape(4000, -1)
    post_sd = e.std(0)
    assert np.abs(u.mean(0) - ref).max() < 0.10 * post_sd.mean()
    assert 0.95 < (u.std(0) / post_sd).mean() < 1.05


def test_temper_above_zero_forces_the_guided_arm():
    """Exact inference describes the mixture; with a residual on that is a different prior."""
    x = _states()
    mix = _mixture(x, 3, rho=0.3)
    model = build_regime_denoiser(mix, {"base_channels": 8, "depth": 1})
    s = RegimeSampler(mix, model, temper=0.5, inference="exact", device="cpu")
    assert s.inference == "guided"
    with pytest.raises(ValueError, match="temper"):
        RegimeSampler(mix, None, temper=0.5, device="cpu")


# ---------------------------------------------------------------------------
# Construction.
# ---------------------------------------------------------------------------
def test_prior_draws_carry_the_mixture_moments():
    x = _states()
    mix = _mixture(x, 3, rho=0.3)
    draws = prior_sample(mix, 1.0, 40000, np.random.default_rng(0))
    assert np.allclose(draws.mean(0), mix.w @ mix.mu, atol=0.1)
    within = sum(w * c for w, c in zip(mix.w, mix.cov))
    between = sum(w * np.outer(m - mix.w @ mix.mu, m - mix.w @ mix.mu)
                  for w, m in zip(mix.w, mix.mu))
    assert np.allclose(np.cov(draws, rowvar=False, ddof=1), within + between,
                       atol=0.35 * np.sqrt(np.trace(within + between) / D))


def test_shrinkage_moves_components_toward_the_pool():
    x = _states()
    lab = partition_states(x, 3, seed=0)
    spread = []
    for rho in (0.0, 0.5, 1.0):
        m = build_regime_mixture(x, lab, SHAPE, np.ones(SHAPE[1:], bool), rho=rho)
        spread.append(float(np.abs(m.cov[0] - m.cov[1]).max()))
    assert spread[0] > spread[1] > spread[2]
    assert spread[2] == pytest.approx(0.0, abs=1e-12)   # rho = 1: every component pooled


def test_time_partition_cuts_contiguous_windows():
    x = _states()
    ages = np.arange(len(x)) * 25
    lab = partition_states(x, 3, rule="time", ages=ages)
    assert set(np.unique(lab)) == {0, 1, 2}
    assert np.all(np.diff(lab) >= 0)                    # contiguous windows
    with pytest.raises(ValueError, match="ages"):
        partition_states(x, 3, rule="time")


def test_build_rejects_a_regime_too_small_to_estimate():
    x = _states(n=12)
    lab = np.zeros(len(x), dtype=np.int64)
    lab[0] = 1
    with pytest.raises(ValueError, match="too few"):
        build_regime_mixture(x, lab, SHAPE, np.ones(SHAPE[1:], bool), rho=0.3)


def test_sweep_matches_the_one_at_a_time_mean():
    """The gathered sweep is an optimisation, so it must change nothing."""
    from paleoreco.assim.regimes import exact_posterior_mean_sweep
    x = _states()
    g, r, y = _obs()
    mix = _mixture(x, 3, rho=0.3)
    b_scales = (0.5, 1.0, 2.0, 5.0)
    swept = exact_posterior_mean_sweep(mix, g, r, y, b_scales)
    one_by_one = np.stack([exact_posterior_mean(mix, g, r, y, b) for b in b_scales])
    assert np.allclose(swept, one_by_one, atol=1e-12)


def test_components_are_rebuilt_from_their_floored_eigendecomposition():
    """The closed-form arm reads ``cov`` and the reverse process reads ``U``/``lam``, so
    the two must describe the same covariance."""
    x = _states()
    mix = _mixture(x, 3, rho=0.3)
    for j in range(mix.n_regimes):
        assert np.allclose(mix.cov[j], (mix.U[j] * mix.lam[j]) @ mix.U[j].T, atol=1e-10)
    assert (mix.lam > 0).all()
    assert np.max(mix.lam[:, 0] / mix.lam[:, -1]) <= 1.0 / mix.meta["eig_floor"] + 1


# ---------------------------------------------------------------------------
# The noise schedule.
# ---------------------------------------------------------------------------
def test_schedule_tops_out_above_the_leading_mode():
    x = _states()
    mix = _mixture(x, 3, rho=0.3)
    s = RegimeSampler(mix, b_scale=4.0, inference="guided", device="cpu")
    lead = float(np.sqrt(4.0 * mix.lam.max()))
    assert s.sigmas[0] >= 8.0 * lead
    # and the sweep rebuilds it, so a larger amplitude gets a wider schedule
    assert s._schedule(16.0)[0] > s._schedule(1.0)[0]
    # an explicit value is still honoured
    assert RegimeSampler(mix, sigma_max=1234.0, device="cpu").sigmas[0] == pytest.approx(1234.0)
