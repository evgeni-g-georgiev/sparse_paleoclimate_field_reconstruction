"""Guided sampler correctness against the closed-form Gaussian posterior.

With an analytic linear denoiser whose implied prior is ``N(0, Sigma)``, the
unconditional sampler must recover ``Sigma`` and the guided posterior mean must
match the closed-form (Kalman / 3DVar) update. This validates the reverse process
and the observation guidance without a trained network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paleoreco.assim.generative import (
    GuidedSampler, HybridDenoiser, annealed_obs_cov, cell_scales, guidance_cov,
)
from paleoreco.assim.method import Observations


class _AnalyticDenoiser(torch.nn.Module):
    """Exact Tweedie denoiser for ``N(0, Sigma)``: ``D(x; s) = Sigma (Sigma + s^2 I)^-1 x``."""

    sigma_data = 0.5

    def __init__(self, Sigma):
        super().__init__()
        self.d = Sigma.shape[0]
        self.register_buffer("S", torch.tensor(Sigma, dtype=torch.float32))

    def forward(self, x, sigma):
        s = float(np.asarray(sigma).reshape(-1)[0])
        M = self.S @ torch.linalg.inv(self.S + s ** 2 * torch.eye(self.d))
        return (x.reshape(x.shape[0], -1) @ M.T).reshape(x.shape)


def _cov(Sigma, shape):
    # A scale field of sigma_data everywhere makes the normalised frame equal the
    # anomaly frame, so the guidance covariance is Sigma itself.
    scales = np.full(shape, _AnalyticDenoiser.sigma_data)
    return guidance_cov(Sigma, scales, np.ones(shape[1:], bool))


def _sampler(Sigma, shape, **kw):
    scales = np.full(shape, _AnalyticDenoiser.sigma_data)
    return GuidedSampler(_AnalyticDenoiser(Sigma), scales, np.ones(shape[1:], bool),
                         _cov(Sigma, shape), device="cpu", **kw)


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T / d + 0.3 * np.eye(d)


def test_guidance_cov_zeroes_masked_cells_and_clips():
    shape = (2, 2, 2)
    d = int(np.prod(shape))
    safe = np.ones(shape[1:], bool)
    safe[0, 0] = False
    cov = guidance_cov(_spd(d, 3), np.full(shape, 0.5), safe)
    assert cov.U.shape == (d, d) and cov.U.dtype == np.float32
    assert (cov.lam >= 0).all()
    Bn = (cov.U * cov.lam[None, :]) @ cov.U.T
    masked = ~np.broadcast_to(safe, shape).ravel()
    assert np.allclose(Bn[masked, :], 0.0, atol=1e-5)
    assert np.allclose(Bn[:, masked], 0.0, atol=1e-5)


def test_annealed_obs_cov_limits():
    d = 8
    Sigma = _spd(d, 0)
    lam_np, U_np = np.linalg.eigh(Sigma)
    g = np.array([0, 5])
    r = torch.tensor([0.2, 0.5], dtype=torch.float64)
    U_g = torch.as_tensor(U_np[g], dtype=torch.float64)
    lam = torch.as_tensor(lam_np, dtype=torch.float64)
    gamma = 2.0

    low = annealed_obs_cov(U_g, lam, r, 1e-5, gamma).numpy()
    assert np.allclose(low, np.diag(r.numpy()), atol=1e-8)
    high = annealed_obs_cov(U_g, lam, r, 1e5, gamma).numpy()
    assert np.allclose(high, gamma * Sigma[np.ix_(g, g)] + np.diag(r.numpy()), rtol=1e-6)


def test_cell_scales_are_per_cell_std():
    cube = np.zeros((5, 2, 4, 4))
    cube[:, 0, 1, 2] = np.arange(5)              # one cell of channel 0 varies
    s = cell_scales(cube, np.ones((4, 4), bool))
    assert s.shape == (2, 4, 4)
    assert s[0, 1, 2] == pytest.approx(np.arange(5).std())
    assert s[0, 0, 0] == 0.0 and (s[1] == 0.0).all()


def test_cell_scales_are_one_on_masked_cells():
    """Masked cells hold no variance, so they must not reach the sampler as a zero
    divisor."""
    cube = np.zeros((5, 2, 4, 4))
    safe = np.ones((4, 4), bool)
    safe[0] = False
    s = cell_scales(cube, safe)
    assert (s[:, ~safe] == 1.0).all()
    assert (s[:, safe] == 0.0).all()


def test_guided_sampler_rejects_scales_off_the_grid():
    shape = (2, 4, 4)
    d = int(np.prod(shape))
    with pytest.raises(ValueError, match="not a field over"):
        GuidedSampler(_AnalyticDenoiser(np.eye(d)), np.full(2, 0.5),
                      np.ones(shape[1:], bool), _cov(np.eye(d), shape),
                      n_steps=4, device="cpu")


def test_guided_sampler_rejects_cov_off_the_state():
    shape = (2, 4, 4)
    d = int(np.prod(shape))
    small = _cov(np.eye(8), (2, 2, 2))
    with pytest.raises(ValueError, match="does not match"):
        GuidedSampler(_AnalyticDenoiser(np.eye(d)), np.full(shape, 0.5),
                      np.ones(shape[1:], bool), small, n_steps=4, device="cpu")


def test_prior_sampling_recovers_gaussian_covariance():
    shape = (2, 2, 2)
    Sigma = _spd(np.prod(shape), 0)
    samp = _sampler(Sigma, shape, n_steps=128)
    x = samp.sample_prior(4000, seed=1).reshape(4000, -1)
    rel = np.linalg.norm(np.cov(x, rowvar=False) - Sigma) / np.linalg.norm(Sigma)
    assert np.abs(x.mean(0)).max() < 0.15
    assert rel < 0.4


def _kalman_rel(gamma, Sigma=None, gather=(0, 5), y=(1.2, -0.8)):
    """Relative error of the guided posterior mean against the closed-form update."""
    shape = (2, 2, 2)
    d = int(np.prod(shape))
    Sigma = _spd(d, 0) if Sigma is None else Sigma
    samp = _sampler(Sigma, shape, n_steps=128)

    gather = np.asarray(gather)
    y = np.asarray(y, dtype=np.float64)
    R = np.full(len(gather), 0.05)
    H = np.zeros((len(gather), d))
    H[np.arange(len(gather)), gather] = 1.0
    kalman = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + np.diag(R)) @ y

    ens = samp.sample_posterior(gather, y, R, gamma=gamma, n=4000, seed=2).reshape(4000, -1)
    return np.linalg.norm(ens.mean(0) - kalman) / np.linalg.norm(kalman)


def test_posterior_mean_matches_kalman():
    """``gamma = 1`` is exact moment matching for this prior: the guidance covariance
    equals the true annealed conditional covariance at every noise level, so the
    closed-form update must survive it."""
    assert _kalman_rel(1.0) < 0.1


def test_gamma_scales_posterior_spread_at_observed_cells():
    """Shrinking the prior block of S pins the observed cells (near-zero spread);
    growing it loosens them. The spread at the observed cells must rise with gamma."""
    shape = (2, 2, 2)
    Sigma = _spd(8, 0)
    samp = _sampler(Sigma, shape, n_steps=64)
    gather = np.array([0, 5])
    y = np.array([1.2, -0.8])
    R = np.full(2, 0.05)
    spread = []
    for gamma in (1e-3, 1.0, 4.0):
        ens = samp.sample_posterior(gather, y, R, gamma=gamma, n=500, seed=3)
        spread.append(ens.reshape(500, -1).var(axis=0)[gather].mean())
    assert spread[0] < spread[1] < spread[2]


def test_posterior_mean_matches_kalman_with_clustered_obs():
    """Adjacent observed cells of a strongly correlated prior.

    Correlated sites are the geometry where a per-site (diagonal) likelihood
    over-counts shared evidence; the full covariance must keep the closed form.
    """
    d = 8
    rng = np.random.default_rng(7)
    A = rng.standard_normal((d, 2))
    Sigma = A @ A.T + 0.1 * np.eye(d)            # long-range structure, 2 dominant modes
    assert _kalman_rel(1.0, Sigma=Sigma, gather=(0, 1, 2, 3),
                       y=(1.5, -1.0, 0.8, -1.2)) < 0.1


def test_masked_cells_stay_zero():
    shape = (2, 4, 4)
    d = int(np.prod(shape))
    safe = np.ones((4, 4), bool)
    safe[0] = False
    cov = guidance_cov(np.eye(d), np.full(shape, 0.5), safe)
    samp = GuidedSampler(_AnalyticDenoiser(np.eye(d)),
                         np.full(shape, 0.5), safe, cov, n_steps=8, device="cpu")
    x = samp.sample_prior(3, seed=0)
    assert np.allclose(x[:, :, ~safe], 0.0)


def _hybrid_sampler(Sigma, amp, sigma_switch, shape, **kw):
    # The wrapped net carries a deliberately wrong prior (identity), so any result
    # matching amp * Sigma proves the closed-form branch drove the trajectory.
    scales = np.full(shape, _AnalyticDenoiser.sigma_data)
    cov = _cov(Sigma, shape).scaled(amp)
    net = _AnalyticDenoiser(np.eye(int(np.prod(shape))))
    return GuidedSampler(HybridDenoiser(net, cov, sigma_switch), scales,
                         np.ones(shape[1:], bool), cov, device="cpu", **kw)


def test_scaled_cov_shares_the_eigenbasis():
    cov = _cov(_spd(8, 0), (2, 2, 2))
    doubled = cov.scaled(2.0)
    assert doubled.U is cov.U
    assert np.allclose(doubled.lam, 2.0 * cov.lam)
    assert doubled.meta["amp"] == 2.0 and "amp" not in cov.meta


def test_hybrid_denoiser_switches_between_net_and_closed_form():
    d = 8
    net = _AnalyticDenoiser(_spd(d, 1))
    cov = _cov(_spd(d, 2), (2, 2, 2))
    hyb = HybridDenoiser(net, cov, sigma_switch=1.5)
    x = torch.randn(3, 2, 2, 2)
    assert torch.equal(hyb(x, 0.5), net(x, 0.5))
    for s in (1.5, 4.0):                    # the boundary belongs to the closed form
        f = cov.lam / (cov.lam + s ** 2)
        want = (x.reshape(3, -1).numpy() @ cov.U) * f[None, :] @ cov.U.T
        assert np.allclose(hyb(x, s).reshape(3, -1).numpy(), want, atol=1e-5)
    assert hyb.sigma_data == net.sigma_data


def test_hybrid_prior_recovers_the_scaled_covariance():
    shape = (2, 2, 2)
    Sigma = _spd(np.prod(shape), 0)
    samp = _hybrid_sampler(Sigma, 2.0, 0.0, shape, n_steps=128)
    x = samp.sample_prior(4000, seed=1).reshape(4000, -1)
    rel = np.linalg.norm(np.cov(x, rowvar=False) - 2.0 * Sigma) / np.linalg.norm(2.0 * Sigma)
    assert np.abs(x.mean(0)).max() < 0.25
    assert rel < 0.4


def test_hybrid_posterior_mean_matches_kalman():
    """``gamma = 1`` with the shared scaled cov is exact moment matching for the
    hybrid's Gaussian prior, so the closed-form update for ``2 Sigma`` must survive."""
    shape = (2, 2, 2)
    d = int(np.prod(shape))
    Sigma = _spd(d, 0)
    samp = _hybrid_sampler(Sigma, 2.0, 0.0, shape, n_steps=128)

    gather = np.array([0, 5])
    y = np.array([1.2, -0.8])
    R = np.full(2, 0.05)
    H = np.zeros((2, d))
    H[np.arange(2), gather] = 1.0
    S2 = 2.0 * Sigma
    kalman = S2 @ H.T @ np.linalg.inv(H @ S2 @ H.T + np.diag(R)) @ y

    ens = samp.sample_posterior(gather, y, R, gamma=1.0, n=4000, seed=2).reshape(4000, -1)
    assert np.linalg.norm(ens.mean(0) - kalman) / np.linalg.norm(kalman) < 0.1


def test_hybrid_masked_cells_stay_zero():
    shape = (2, 4, 4)
    d = int(np.prod(shape))
    safe = np.ones((4, 4), bool)
    safe[0] = False
    scales = np.full(shape, 0.5)
    cov = guidance_cov(np.eye(d), scales, safe).scaled(2.0)
    # A switch inside the 8-step schedule, so both branches run during sampling.
    hyb = HybridDenoiser(_AnalyticDenoiser(np.eye(d)), cov, 1.0)
    samp = GuidedSampler(hyb, scales, safe, cov, n_steps=8, device="cpu")
    x = samp.sample_prior(3, seed=0)
    assert np.allclose(x[:, :, ~safe], 0.0)


def test_analyze_returns_mean_and_variance():
    shape = (2, 2, 2)
    # gamma away from the constructor default, so this also covers analyze reading it.
    samp = _sampler(_spd(8, 1), shape, n_steps=8, n_samples=6, gamma=2.0)
    obs = Observations(gather=np.array([0, 5]), y_anom=np.array([0.5, -0.5]),
                       sse=np.array([0.1, 0.1]))
    res = samp.analyze(obs, np.zeros(int(np.prod(shape))))
    assert res.mean_anom.shape == shape
    assert res.posterior_var is not None and (res.posterior_var >= 0).all()
