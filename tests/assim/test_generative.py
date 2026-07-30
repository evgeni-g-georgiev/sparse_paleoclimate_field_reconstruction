"""Guided sampler correctness against the closed-form Gaussian posterior.

With an analytic linear denoiser whose implied prior is ``N(0, Sigma)``, the
unconditional sampler must recover ``Sigma`` and the guided posterior mean must
match the closed-form (Kalman / 3DVar) update. This validates the reverse process
and the observation guidance without a trained network.
"""

from __future__ import annotations

import numpy as np
import torch

from paleoreco.assim.generative import GuidedSampler, channel_scales, likelihood_var
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


def _sampler(Sigma, shape, **kw):
    # channel_scales == sigma_data makes the normalised frame equal the anomaly frame.
    scales = np.full(shape[0], _AnalyticDenoiser.sigma_data)
    return GuidedSampler(_AnalyticDenoiser(Sigma), scales, np.ones(shape[1:], bool),
                         np.diag(Sigma).reshape(shape), device="cpu", **kw)


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T / d + 0.3 * np.eye(d)


def test_likelihood_var_brackets_r_and_r_plus_prior_var():
    r, b = np.array([0.2, 0.5]), np.array([1.0, 4.0])
    assert np.allclose(likelihood_var(r, b, 1e-4, 1.0), r, atol=1e-6)
    assert np.allclose(likelihood_var(r, b, 1e6, 1.0), r + b, rtol=1e-6)
    mid, high = likelihood_var(r, b, 0.5, 1.0), likelihood_var(r, b, 2.0, 1.0)
    assert (r < mid).all() and (mid < high).all() and (high < r + b).all()


def test_likelihood_var_is_r_where_the_prior_has_no_variance():
    assert likelihood_var(np.array([0.3]), np.array([0.0]), 1.0, 1.0) == 0.3


def test_channel_scales_are_per_channel_std():
    cube = np.zeros((5, 2, 4, 4))
    cube[:, 0] = np.arange(5)[:, None, None]     # channel 0 varies
    s = channel_scales(cube, np.ones((4, 4), bool))
    assert s[0] > 0 and s[1] == 0.0


def test_prior_sampling_recovers_gaussian_covariance():
    shape = (2, 2, 2)
    Sigma = _spd(np.prod(shape), 0)
    samp = _sampler(Sigma, shape, n_steps=128, n_correct=4, corrector_tau=0.3)
    x = samp.sample_prior(4000, seed=1).reshape(4000, -1)
    rel = np.linalg.norm(np.cov(x, rowvar=False) - Sigma) / np.linalg.norm(Sigma)
    assert np.abs(x.mean(0)).max() < 0.15
    assert rel < 0.4


def _kalman_rel(gamma):
    """Relative error of the guided posterior mean against the closed-form update."""
    shape = (2, 2, 2)
    d = int(np.prod(shape))
    Sigma = _spd(d, 0)
    samp = _sampler(Sigma, shape, n_steps=128, n_correct=4, corrector_tau=0.3)

    gather = np.array([0, 5])
    y = np.array([1.2, -0.8])
    R = np.array([0.05, 0.05])
    H = np.zeros((2, d))
    H[0, 0], H[1, 5] = 1.0, 1.0
    kalman = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + np.diag(R)) @ y

    ens = samp.sample_posterior(gather, y, R, gamma=gamma, n=4000, seed=2).reshape(4000, -1)
    return np.linalg.norm(ens.mean(0) - kalman) / np.linalg.norm(kalman)


def test_posterior_mean_matches_kalman():
    assert _kalman_rel(1e-3) < 0.1


def test_posterior_mean_matches_kalman_at_the_gaussian_gamma():
    """``gamma = 1`` carries the full prior-variance term, and the closed form survives
    it: taking only Sigma's diagonal while Sigma is correlated costs nothing here, so
    the same bound as the vanishing-gamma case holds.
    """
    assert _kalman_rel(1.0) < 0.1


def test_masked_cells_stay_zero():
    shape = (2, 4, 4)
    safe = np.ones((4, 4), bool)
    safe[0] = False
    samp = GuidedSampler(_AnalyticDenoiser(np.eye(np.prod(shape))),
                         np.full(2, 0.5), safe, np.ones(shape),
                         n_steps=8, n_correct=1, device="cpu")
    x = samp.sample_prior(3, seed=0)
    assert np.allclose(x[:, :, ~safe], 0.0)


def test_analyze_returns_mean_and_variance():
    shape = (2, 2, 2)
    # gamma away from the constructor default, so this also covers analyze reading it.
    samp = _sampler(_spd(8, 1), shape, n_steps=8, n_correct=1, n_samples=6, gamma=2.0)
    obs = Observations(gather=np.array([0, 5]), y_anom=np.array([0.5, -0.5]),
                       sse=np.array([0.1, 0.1]))
    res = samp.analyze(obs, np.zeros(int(np.prod(shape))))
    assert res.mean_anom.shape == shape
    assert res.posterior_var is not None and (res.posterior_var >= 0).all()
