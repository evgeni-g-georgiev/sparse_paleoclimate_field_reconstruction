"""Guided posterior sampling from a score-based prior (pixel-space score DA).

The unconditional diffusion prior of :mod:`paleoreco.models.diffusion` replaces
the Gaussian background of 3DVar. Reconstruction is posterior sampling: run the
EDM reverse process, but at every step add the observation-guidance score
``grad log p(y | x_sigma)`` from a Gaussian likelihood (Manshausen et al. 2025;
Rozet and Louppe 2023; Chung et al. 2023). The likelihood covariance couples the
observations through the annealed background covariance,

    S(sigma) = gamma * H Sigma(sigma) H^T + R,

where ``Sigma(sigma)`` shares the background covariance B's eigenbasis with each
eigenvalue ``lam`` shrunk to ``lam sigma^2 / (lam + sigma^2)``, the exact
conditional covariance of a Gaussian prior at noise level ``sigma`` (Boys et al.
2023). Nearby sites therefore share their evidence instead of counting as
independent: S runs from ``gamma * H B H^T + R``, the 3DVar innovation
covariance, at high ``sigma`` down to ``R`` as ``sigma -> 0``. ``gamma`` scales
the prior block and is the tuned operating point.

Sampling is the EDM 2nd-order Heun predictor (Karras 2022, Alg. 1). Fields live
in the per-cell-scaled anomaly frame during sampling and are returned in anomaly
units, so the network sees unit variance everywhere and the spatial variance
pattern is carried by the scale field rather than learned. Masked cells are held
at zero throughout. The prior score comes from the trained net alone or from a
:class:`HybridDenoiser` that anchors the high-noise regime to the guidance
covariance.
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


@dataclass(frozen=True)
class GuidanceCov:
    """Eigendecomposition of a background covariance in the normalised frame.

    ``U`` is ``(D, D)`` float32, ``lam`` its eigenvalues clipped at zero; ``meta``
    records how the covariance was built (taper settings, source ages) so a run's
    config can state what the guidance believed. Holds numpy arrays only, so it
    pickles into sampler-pool workers.
    """

    U: np.ndarray
    lam: np.ndarray
    meta: dict

    def scaled(self, amp: float) -> "GuidanceCov":
        """The covariance of ``amp * B``: same eigenbasis, eigenvalues scaled."""
        return GuidanceCov(U=self.U, lam=amp * self.lam,
                           meta={**self.meta, "amp": float(amp)})


def guidance_cov(B: np.ndarray, scales: np.ndarray, safe_valid: np.ndarray,
                 *, sigma_data: float = SIGMA_DATA, meta: dict | None = None) -> GuidanceCov:
    """Transform a background covariance into the sampler's normalised frame.

    ``B`` is ``(D, D)`` in anomaly units; ``scales`` is the per-cell std field the
    checkpoint carries. Masked rows and columns are zeroed so masked cells can
    neither receive nor exert guidance. One eigh here saves a solve per noise
    level later: every S(sigma) shares the same eigenbasis.
    """
    scale = np.asarray(scales, dtype=np.float64)
    safe_flat = np.broadcast_to(np.asarray(safe_valid, dtype=bool), scale.shape).ravel()
    mult = np.where(safe_flat, sigma_data / scale.ravel(), 1.0)
    Bn = np.asarray(B, dtype=np.float64) * np.outer(mult, mult)
    Bn[~safe_flat, :] = 0.0
    Bn[:, ~safe_flat] = 0.0
    Bn = 0.5 * (Bn + Bn.T)
    lam, U = np.linalg.eigh(Bn)
    return GuidanceCov(U=U.astype(np.float32), lam=np.clip(lam, 0.0, None),
                       meta=dict(meta or {}))


def annealed_obs_cov(U_g: torch.Tensor, lam: torch.Tensor, r_norm: torch.Tensor,
                     sigma: float, gamma: float) -> torch.Tensor:
    """S(sigma) for one network: gamma-scaled annealed prior block plus diag R."""
    f = lam * sigma ** 2 / (lam + sigma ** 2)
    return gamma * ((U_g * f[None, :]) @ U_g.T) + torch.diag(r_norm)


class HybridDenoiser(torch.nn.Module):
    """Learned denoiser below ``sigma_switch``, closed-form Gaussian denoiser above.

    The net is essentially untrained at high noise (the EDM lognormal puts ~2% of
    its training mass above sigma ~3) yet the leading modes' variance is decided
    there, and learned scores match their Gaussian approximation at moderate-to-high
    noise anyway (arXiv 2412.09726). At or above the switch this applies the exact
    denoiser of ``N(0, B)`` for the covariance ``cov`` decomposes, so handing the
    sampler the same ``cov`` moment-matches prior and guidance by construction; an
    amplitude on B rides in ``cov`` itself (:meth:`GuidanceCov.scaled`). The
    closed-form branch stays autograd-friendly because the guidance score
    differentiates through the denoiser.
    """

    def __init__(self, net, cov: GuidanceCov, sigma_switch: float):
        super().__init__()
        self.net = net
        self.sigma_data = float(getattr(net, "sigma_data", SIGMA_DATA))
        self.switch = float(sigma_switch)
        self.register_buffer("U", torch.as_tensor(cov.U, dtype=torch.float32))
        self.register_buffer("lam", torch.as_tensor(cov.lam, dtype=torch.float32))

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        s = float(np.asarray(sigma).reshape(-1)[0])
        if s < self.switch:
            return self.net(x, sigma)
        f = self.lam / (self.lam + s ** 2)
        xf = x.reshape(x.shape[0], -1)
        return ((xf @ self.U) * f[None, :] @ self.U.T).reshape(x.shape)


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
    need. ``cov`` is the guidance covariance in the normalised frame
    (:func:`guidance_cov`); ``gamma`` scales its prior block, 1 being the
    Gaussian value. ``n_samples`` sets the ensemble size :meth:`analyze` draws.
    """

    def __init__(self, denoiser, scales: np.ndarray, safe_valid: np.ndarray,
                 cov: GuidanceCov,
                 *, gamma: float = 1.0, n_samples: int = 16, n_steps: int = 64,
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
        d = int(np.prod(self.shape))
        if cov.U.shape != (d, d):
            raise ValueError(f"cov.U shape {cov.U.shape} does not match the "
                             f"{d}-dim flattened state")
        self.cov = cov
        self.gamma = gamma
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.sigmas = _edm_sigmas(n_steps, sigma_min, sigma_max, rho)

        self._mask = torch.as_tensor(self.safe_valid.astype(np.float32), device=self.device)
        self._mult_t = torch.as_tensor(self.mult, device=self.device, dtype=torch.float32)

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
        """EDM Heun predictor reverse loop; normalised fields."""
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

        # Assemble and invert S per noise level in float64 on the CPU, then cast:
        # a float32 S^-1 held constant in the graph keeps the quadratic form's
        # gradient exact while the m x m inversions stay well conditioned.
        U_g = torch.as_tensor(self.cov.U[g], dtype=torch.float64)
        lam = torch.as_tensor(self.cov.lam, dtype=torch.float64)
        r_t = torch.as_tensor(r_norm, dtype=torch.float64)
        sinv = {}
        for s in self.sigmas[:-1]:
            S = annealed_obs_cov(U_g, lam, r_t, float(s), gamma)
            Si = torch.cholesky_inverse(torch.linalg.cholesky(S))
            Si = 0.5 * (Si + Si.T)
            sinv[float(s)] = Si.to(device=self.device, dtype=torch.float32)

        def guidance(x_hat: torch.Tensor, sigma: float) -> torch.Tensor:
            pred = x_hat.reshape(x_hat.shape[0], -1)[:, idx]
            r = y_t[None, :] - pred
            return -0.5 * torch.einsum("nm,mk,nk->", r, sinv[float(sigma)], r)

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
