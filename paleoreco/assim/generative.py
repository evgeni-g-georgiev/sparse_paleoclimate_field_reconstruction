"""Guided posterior sampling from a score-based prior (pixel-space score DA).

The unconditional diffusion prior of :mod:`paleoreco.models.diffusion` replaces
the Gaussian background of 3DVar. Reconstruction is posterior sampling: run the
EDM reverse process, but at every step add the observation-guidance score
``grad log p(y | x_sigma)`` from a Gaussian likelihood, so the ensemble is pulled
toward the proxies (Manshausen et al. 2025; Rozet and Louppe 2023; Chung et al.
2023). The likelihood covariance is ``R`` plus the denoiser's own conditional
variance at the observed cells (:func:`likelihood_var`), so it self-anneals per
site: early (large ``sigma``) the denoised estimate is unreliable and the
observations barely count, and as ``sigma -> 0`` each site is trusted at its own
error ``R``.

Sampling follows the predictor-corrector scheme: an EDM 2nd-order Heun predictor
(Karras 2022, Alg. 1) plus ``n_correct`` Langevin corrector steps per level
(Rozet 2023, Alg. 4, step ``delta = tau * dim / ||s||^2``). Fields live in the
per-cell-scaled anomaly frame during sampling and are returned in anomaly units,
so the network sees unit variance everywhere and the spatial variance pattern is
carried by the scale field rather than learned. Masked cells are held at zero
throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from paleoreco.assim.method import AnalysisResult, Method, Observations
from paleoreco.models.diffusion import SIGMA_DATA, SIGMA_MAX, SIGMA_MIN, RHO


def cell_scales(cube_anom: np.ndarray, safe_valid: np.ndarray) -> np.ndarray:
    """Per-cell anomaly std, the diffusion normalisation field, shape ``(C, H, W)``.

    Masked cells hold no variance and would divide by zero, so they carry one; the
    mask pins them to zero everywhere downstream regardless.
    """
    v = np.asarray(safe_valid, dtype=bool)
    x = np.asarray(cube_anom, dtype=np.float64)
    return np.where(v, x.std(axis=0), 1.0)


def _edm_sigmas(n_steps: int, sigma_min: float, sigma_max: float, rho: float) -> np.ndarray:
    """EDM noise schedule (Karras 2022, Eq. 5), descending, with a trailing zero."""
    i = np.arange(n_steps)
    r = 1.0 / rho
    sig = (sigma_max ** r + i / (n_steps - 1) * (sigma_min ** r - sigma_max ** r)) ** rho
    return np.concatenate([sig, [0.0]])


def likelihood_var(r, prior_var, sigma: float, gamma: float):
    """Diagonal of the observation likelihood covariance at one noise level.

    The proxy error ``r``, plus the variance of the denoiser's own estimate of the
    clean field at those cells. Under a Gaussian prior of variance ``prior_var`` that
    second term is exactly ``prior_var sigma^2 / (prior_var + sigma^2)``, running from
    ``prior_var`` (the analogue of ``H B H^T``) at high noise down to zero as
    ``sigma -> 0``, so ``gamma = 1`` is the Gaussian value and a tuned ``gamma``
    reports how far the prior's stated variance can be trusted. Numpy or torch.
    """
    return r + gamma * prior_var * sigma ** 2 / (prior_var + sigma ** 2)


@dataclass(frozen=True)
class PosteriorJob:
    """One posterior draw: an observation network, its operating point, and its seed.

    Carrying the seed on the job is what makes a batch of draws order-independent,
    so the same list yields the same ensembles however it is executed.
    """

    gather: np.ndarray
    y_anom: np.ndarray
    r_diag: np.ndarray
    gamma: float
    n: int
    seed: int


class GuidedSampler(Method):
    """Posterior sampler over a trained :class:`EDMDenoiser`; a :class:`Method`.

    ``analyze`` reduces an ensemble to a Gaussian ``(mean, variance)`` for the
    method-agnostic metrics; :meth:`sample_posterior` and :meth:`sample_prior`
    expose the raw ensembles the calibration and prior-vs-posterior diagnostics
    need. ``prior_var`` is the prior's per-cell variance in anomaly units, shaped
    like one field, which the likelihood covariance reads at the observed cells;
    ``gamma`` scales that term, 1 being its Gaussian value. ``n_samples`` sets the
    ensemble size that :meth:`analyze` draws.
    """

    def __init__(self, denoiser, scales: np.ndarray, safe_valid: np.ndarray,
                 prior_var: np.ndarray,
                 *, gamma: float = 1.0, n_samples: int = 16, n_steps: int = 64,
                 n_correct: int = 2, corrector_tau: float = 0.3,
                 sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX, rho: float = RHO,
                 device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                               else "mps" if torch.backends.mps.is_available()
                                               else "cpu"))
        self.denoiser = denoiser.to(self.device).eval()
        self.sd = float(getattr(denoiser, "sigma_data", SIGMA_DATA))
        self.scale = np.asarray(scales, dtype=np.float64)                  # (C, H, W)
        self.mult = self.sd / self.scale                                  # anomaly -> normalised
        self.safe_valid = np.asarray(safe_valid, dtype=bool)
        if self.scale.ndim != 3 or self.scale.shape[1:] != self.safe_valid.shape:
            raise ValueError(f"scales shape {self.scale.shape} is not a field over the "
                             f"{self.safe_valid.shape} grid")
        self.shape = self.scale.shape
        pv = np.asarray(prior_var, dtype=np.float64)
        if pv.shape != self.shape:
            raise ValueError(f"prior_var shape {pv.shape} does not match field shape "
                             f"{self.shape}")
        self.prior_var = pv
        self.gamma = gamma
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_correct = n_correct
        self.tau = corrector_tau
        self.sigmas = _edm_sigmas(n_steps, sigma_min, sigma_max, rho)

        self._mask = torch.as_tensor(self.safe_valid.astype(np.float32), device=self.device)
        self._mult_t = torch.as_tensor(self.mult, device=self.device, dtype=torch.float32)
        self._ndim = float(self.shape[0] * int(self.safe_valid.sum()))
        # Converted once: the likelihood is evaluated in the normalised frame at every
        # step of every draw.
        self._prior_var_norm = (pv * self.mult ** 2).ravel()

    # -- normalisation ---------------------------------------------------
    def _to_norm_obs(self, gather: np.ndarray, y_anom: np.ndarray, r_diag: np.ndarray):
        """Normalise observation values and error variance by their own cell's scale."""
        m = self.mult.ravel()[np.asarray(gather)]
        return (np.asarray(y_anom, dtype=np.float64) * m,
                np.asarray(r_diag, dtype=np.float64) * m ** 2)

    def _denorm(self, x: torch.Tensor) -> np.ndarray:
        """Normalised field batch to anomaly-unit numpy ``(n, 2, H, W)``."""
        x = x / self._mult_t[None]
        return x.detach().cpu().numpy().astype(np.float64)

    # -- score -----------------------------------------------------------
    def _score(self, x: torch.Tensor, sigma: float, guidance) -> torch.Tensor:
        """Prior score, plus the guidance score when ``guidance`` is given. Masked."""
        if guidance is None:
            with torch.no_grad():
                x_hat = self.denoiser(x, sigma)
            return (x_hat - x) / sigma ** 2 * self._mask
        xg = x.detach().requires_grad_(True)
        x_hat = self.denoiser(xg, sigma)
        loglik = guidance(x_hat, sigma)
        g = torch.autograd.grad(loglik, xg)[0]
        prior = (x_hat.detach() - xg.detach()) / sigma ** 2
        return (prior + g) * self._mask

    def _sample(self, n: int, guidance, seed: int) -> torch.Tensor:
        """EDM Heun predictor + Langevin corrector reverse loop; normalised fields."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed))

        def randn(shape):
            return torch.randn(shape, generator=gen, device=self.device) * self._mask

        x = self.sigmas[0] * randn((n, *self.shape))
        for i in range(self.n_steps):
            sig, sig_next = float(self.sigmas[i]), float(self.sigmas[i + 1])
            d = -sig * self._score(x, sig, guidance)
            x_pred = (x + (sig_next - sig) * d) * self._mask
            if sig_next <= 0.0:
                x = x_pred
                continue
            d2 = -sig_next * self._score(x_pred, sig_next, guidance)
            x = (x + (sig_next - sig) * 0.5 * (d + d2)) * self._mask
            for _ in range(self.n_correct):
                s = self._score(x, sig_next, guidance)
                snorm2 = s.flatten(1).pow(2).sum(1).clamp_min(1e-12)
                delta = (self.tau * self._ndim / snorm2)[:, None, None, None]
                x = (x + delta * s + torch.sqrt(2.0 * delta) * randn(x.shape)) * self._mask
        return x

    # -- public sampling -------------------------------------------------
    def sample_prior(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` unconditional prior fields in anomaly units, ``(n, 2, H, W)``."""
        return self._denorm(self._sample(n, None, seed))

    def sample_posterior(self, gather: np.ndarray, y_anom: np.ndarray, r_diag: np.ndarray,
                         gamma: float, n: int, seed: int = 0) -> np.ndarray:
        """``n`` posterior fields conditioned on one observation network, ``(n, 2, H, W)``."""
        y_norm, r_norm = self._to_norm_obs(gather, y_anom, r_diag)
        g = np.asarray(gather)
        idx = torch.as_tensor(g, device=self.device, dtype=torch.long)
        y_t = torch.as_tensor(y_norm, device=self.device, dtype=torch.float32)
        r_t = torch.as_tensor(r_norm, device=self.device, dtype=torch.float32)
        b_t = torch.as_tensor(self._prior_var_norm[g], device=self.device,
                              dtype=torch.float32)

        def guidance(x_hat: torch.Tensor, sigma: float) -> torch.Tensor:
            pred = x_hat.reshape(x_hat.shape[0], -1)[:, idx]
            v = likelihood_var(r_t, b_t, sigma, gamma)
            return -0.5 * ((y_t[None, :] - pred) ** 2 / v).sum()

        return self._denorm(self._sample(n, guidance, seed))

    def imap_posterior(self, jobs):
        """Posterior ensembles for a sequence of :class:`PosteriorJob`, in order."""
        for j in jobs:
            yield self.sample_posterior(j.gather, j.y_anom, j.r_diag, j.gamma,
                                        j.n, j.seed)

    # -- Method contract -------------------------------------------------
    @staticmethod
    def reduce(ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble ``(n, 2, H, W)`` to per-cell mean and variance.

        The variance is the unbiased estimate: scatter about the sample mean
        understates the spread by ``(n - 1) / n``, which at the ensemble sizes used
        here is a systematic few percent against an analytic posterior variance.
        """
        return ensemble.mean(axis=0), ensemble.var(axis=0, ddof=1)

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        ens = self.sample_posterior(obs.gather, obs.y_anom, obs.sse,
                                    self.gamma, self.n_samples)
        mean, var = self.reduce(ens)
        return AnalysisResult(mean_anom=mean, posterior_var=var)
