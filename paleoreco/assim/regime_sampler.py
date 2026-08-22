"""Posterior sampling from a regime-mixture prior, exactly or by guided diffusion.

``exact``  The closed-form Gaussian-mixture posterior: one Kalman update per component,
           weights by marginal likelihood, draws by the perturbed-observation identity.
           Valid whenever the learned residual is off (``temper = 0``).

``guided`` The EDM reverse process. Required once the residual is on, since the posterior
           is then no longer a mixture.

At ``temper = 0`` both arms target the same distribution, so their difference measures the
inference error alone.

The guidance uses the mixture's own predictive rather than a DPS-style approximation.
``p(x_0 | x_sigma)`` is a Gaussian mixture, so ``p(y | x_sigma)`` is one too and is written
out as such: no Jacobian, no tuned guidance strength. Its covariance is held fixed in the
graph, as the annealed table is in :mod:`paleoreco.assim.generative`.
"""

from __future__ import annotations

import numpy as np
import torch

from paleoreco.assim.generative import _edm_sigmas
from paleoreco.assim.method import AnalysisResult, Method, Observations
from paleoreco.assim.regimes import (
    RegimeMixture, exact_posterior_mean, exact_posterior_sample, prior_sample,
)
from paleoreco.models.diffusion import RHO, SIGMA_DATA, SIGMA_MAX, SIGMA_MIN
from paleoreco.models.regime_score import MixtureDenoiser, RegimeDenoiser

_INFERENCE = frozenset({"exact", "guided"})

# How far above the prior's leading mode the noise schedule starts, when ``sigma_max`` is
# not given explicitly.
SIGMA_MAX_FACTOR = 8.0


class RegimeSampler(Method):
    """Posterior sampler over a regime mixture; a :class:`Method`.

    ``b_scale`` is the background amplitude, the same quantity 3DVar tunes. ``temper``
    scales the learned residual: at ``0`` the prior is the closed-form mixture and
    ``inference="exact"`` is available; above ``0`` the sampler falls back to ``guided``
    whatever is asked, since the exact arm would then describe a different prior.
    """

    def __init__(self, mixture: RegimeMixture, denoiser: RegimeDenoiser | None = None, *,
                 b_scale: float = 1.0, temper: float = 0.0, inference: str = "exact",
                 n_samples: int = 256, n_steps: int = 64, sigma_min: float = SIGMA_MIN,
                 sigma_max: float | None = None, rho: float = RHO,
                 max_batch: int | None = None, seed_offset: int = 0,
                 dtype: torch.dtype | None = None, device: str | None = None):
        if inference not in _INFERENCE:
            raise ValueError(f"inference must be one of {sorted(_INFERENCE)}; "
                             f"got {inference!r}")
        if max_batch is not None and max_batch < 1:
            raise ValueError(f"max_batch must be >= 1 or None; got {max_batch}")
        if temper != 0.0 and denoiser is None:
            raise ValueError("temper > 0 needs a trained residual denoiser")

        self.mixture = mixture
        # ``cov`` is the Method contract's name for whatever object describes the prior.
        self.cov = mixture
        self.b_scale = float(b_scale)
        self.temper = float(temper)
        self.inference = "guided" if self.temper != 0.0 else inference
        self.n_samples = int(n_samples)
        self.n_steps = int(n_steps)
        self.max_batch = max_batch
        self.seed_offset = int(seed_offset)
        self.shape = tuple(mixture.shape)
        self.sigma_min, self.rho = float(sigma_min), float(rho)
        self.sigma_max = sigma_max
        self.sigmas = self._schedule(self.b_scale)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # The reverse process integrates a stiff field over dozens of steps, so double
        # where it is affordable; a GPU's double throughput is a fraction of its single.
        self.dtype = dtype or (torch.float32 if self.device.type == "cuda"
                               else torch.float64)

        base = MixtureDenoiser(mixture, b_scale=self.b_scale)
        if denoiser is None:
            self.denoiser = base.to(device=self.device, dtype=self.dtype).eval()
            self.sd = float(base.sigma_data)
        else:
            model = RegimeDenoiser(base, denoiser.net, temper=self.temper)
            self.denoiser = model.to(device=self.device, dtype=self.dtype).eval()
            self.sd = float(model.sigma_data)
        self._mix_module = (self.denoiser.mixture if isinstance(self.denoiser, RegimeDenoiser)
                            else self.denoiser)
        self._mask = np.broadcast_to(mixture.safe_valid, self.shape)

    def _schedule(self, b_scale: float) -> np.ndarray:
        """EDM noise levels, with ``sigma_max`` taken from the prior unless it is given.

        The reverse process has to start above the prior's own scale, and this background's
        leading mode has standard deviation ``sqrt(b lam_max)``. The EDM default of 80 is
        calibrated for unit-variance images and does not clear that, so the top of the
        schedule is read off the spectrum instead.
        """
        if self.sigma_max is not None:
            top = float(self.sigma_max)
        else:
            top = SIGMA_MAX_FACTOR * float(np.sqrt(b_scale * self.mixture.lam.max()))
        return _edm_sigmas(self.n_steps, self.sigma_min, max(top, SIGMA_MAX), self.rho)

    # -- the exact arm ----------------------------------------------------
    def _exact(self, gather, r_diag, y, b_scale, n, seed) -> np.ndarray:
        rng = np.random.default_rng([int(seed) + self.seed_offset, 0xEA])
        draws = exact_posterior_sample(self.mixture, gather, r_diag, y, b_scale, n, rng)
        return draws.reshape(n, *self.shape) * self._mask

    def exact_mean(self, gather, r_diag, y, b_scale=None) -> np.ndarray:
        """Closed-form posterior mean, ``(C, H, W)``; no sampling anywhere."""
        k = self.b_scale if b_scale is None else float(b_scale)
        mean = exact_posterior_mean(self.mixture, gather, r_diag, y, k)
        return mean.reshape(self.shape) * self._mask

    # -- the guided arm ---------------------------------------------------
    def _guided(self, gather, r_diag, y, b_scale, n, seed) -> np.ndarray:
        dev, dt = self.device, self.dtype
        g = torch.as_tensor(np.asarray(gather), device=dev, dtype=torch.long)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), device=dev, dtype=dt)
        r_t = torch.as_tensor(np.asarray(r_diag, dtype=np.float64), device=dev, dtype=dt)
        model = (self.denoiser if b_scale == self.b_scale
                 else self._rescaled(b_scale))
        mix = model.mixture if isinstance(model, RegimeDenoiser) else model

        # The schedule is rebuilt per amplitude: sigma_max is read off b_scale * lam_max,
        # so a sweep over b_scale is a sweep over schedules too.
        sigmas = self._schedule(b_scale)
        # One Cholesky per component per noise level, shared by every sample: the
        # within-component conditional covariance does not depend on the state.
        chol = {float(s): self._obs_chol(mix, float(s), g, r_t) for s in sigmas[:-1]}

        # The reverse process is only exact if it starts from p_{sigma_max}. Starting from
        # pure noise instead assumes sigma_max swamps the data, which a prior with modes
        # this large does not licence. Drawing from the mixture and adding the noise is an
        # exact draw from p_{sigma_max} and costs nothing, the prior being closed form.
        rng = np.random.default_rng([int(seed) + self.seed_offset, 0xEC])
        base = prior_sample(self.mixture, b_scale, n, rng).reshape(n, *self.shape)
        gen = torch.Generator(device=dev).manual_seed(int(seed) + self.seed_offset)
        x = (torch.as_tensor(base, device=dev, dtype=dt)
             + sigmas[0] * torch.randn((n, *self.shape), generator=gen, device=dev,
                                       dtype=dt))
        for i in range(self.n_steps):
            sig, sig_next = float(sigmas[i]), float(sigmas[i + 1])
            d = -sig * self._score(model, mix, x, sig, g, y_t, chol[sig])
            x_pred = x + (sig_next - sig) * d
            if sig_next <= 0.0:
                x = x_pred
                continue
            d2 = -sig_next * self._score(model, mix, x_pred, sig_next, g, y_t, chol[sig_next])
            x = x + (sig_next - sig) * 0.5 * (d + d2)
        out = x.detach().cpu().numpy().astype(np.float64)
        return out * self._mask

    @staticmethod
    def _obs_chol(mix, sigma: float, g: torch.Tensor, r_t: torch.Tensor):
        """Cholesky and log-determinant of ``H Gamma_j H^T + R`` per component."""
        with torch.no_grad():
            S = mix.within_cov_obs_blocks(sigma, g)                   # (J, m, m)
            S = S + torch.diag_embed(r_t.expand(S.shape[0], -1))
            L = torch.linalg.cholesky(0.5 * (S + S.transpose(1, 2)))
        return L, 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)

    def _rescaled(self, b_scale: float):
        base = self.denoiser
        if isinstance(base, RegimeDenoiser):
            return RegimeDenoiser(base.mixture.at_scale(b_scale), base.net,
                                  base.temper).to(device=self.device, dtype=self.dtype).eval()
        return base.at_scale(b_scale).to(device=self.device, dtype=self.dtype).eval()

    def _score(self, model, mix, x, sigma, g, y_t, chol) -> torch.Tensor:
        """Prior score plus the mixture guidance score, at one noise level.

        The guidance term is ``grad log p(y | x_sigma)`` with the predictive written out as
        the mixture it is,

            log p(y | x_sigma) = logsumexp_j [ log r_j(x) + log N(y; H mu_j(x), S_j) ]

        ``S_j`` is state-free and factorised once per noise level; ``r_j`` and ``mu_j``
        carry the gradient. At one component this reduces to exact Gaussian guidance, which
        is why ``J = 1`` reproduces 3DVar. The covariance is held fixed in the graph, as the
        annealed table is in :mod:`paleoreco.assim.generative`. A trained residual enters
        through the prior term only; the guidance stays written on the mixture.
        """
        L, logdet = chol
        xg = x.detach().requires_grad_(True)
        x_hat = model(xg, sigma)
        log_r, hmu = mix.obs_predictive(xg, sigma, g)                 # (n, J), (n, J, m)
        resid = y_t[None, None, :] - hmu                              # (n, J, m)
        sol = torch.cholesky_solve(resid.permute(1, 2, 0), L)         # (J, m, n)
        quad = torch.einsum("njm,jmn->nj", resid, sol)
        loglik = torch.logsumexp(log_r - 0.5 * (logdet[None, :] + quad), dim=1).sum()
        grad = torch.autograd.grad(loglik, xg)[0]
        prior = (x_hat.detach() - xg.detach()) / sigma ** 2
        return prior + grad

    # -- public sampling ---------------------------------------------------
    def _chunks(self, n: int):
        if self.max_batch is None or n <= self.max_batch:
            return [(0, n)]
        return [(o, min(self.max_batch, n - o)) for o in range(0, n, self.max_batch)]

    @staticmethod
    def _chunk_seed(seed: int, chunk: int) -> int:
        """Chunk seeds sit far above the small consecutive integers the runners assign."""
        return int(seed) if chunk == 0 else (int(seed) << 32) + (1 << 31) + chunk

    def sample_prior(self, n: int, seed: int = 0, b_scale: float | None = None) -> np.ndarray:
        """``n`` unconditional prior fields in anomaly units, ``(n, C, H, W)``.

        Drawn from the mixture directly, so with a trained residual these are the mixture's
        samples rather than the full model's; the residual shows in the posterior.
        """
        k = self.b_scale if b_scale is None else float(b_scale)
        rng = np.random.default_rng([int(seed) + self.seed_offset, 0xEB])
        return prior_sample(self.mixture, k, n, rng).reshape(n, *self.shape) * self._mask

    def sample_posterior(self, gather: np.ndarray, y_anom: np.ndarray, r_diag: np.ndarray,
                         b_scale: float, n: int, seed: int = 0) -> np.ndarray:
        """``n`` posterior fields for one observation network, ``(n, C, H, W)``.

        Draws are i.i.d., so chunking a large ensemble changes nothing distributionally.
        """
        draw = self._exact if self.inference == "exact" else self._guided
        out = [draw(gather, r_diag, y_anom, b_scale, size, self._chunk_seed(seed, ci))
               for ci, (_, size) in enumerate(self._chunks(n))]
        return np.concatenate(out) if len(out) > 1 else out[0]

    def imap_posterior(self, jobs):
        """Posterior ensembles for a sequence of ``PosteriorJob``, in order.

        The job's ``gamma`` field carries the background amplitude for this method, not a
        guidance strength.
        """
        for j in jobs:
            yield self.sample_posterior(j.gather, j.y_anom, j.r_diag, j.gamma, j.n, j.seed)

    # -- Method contract ---------------------------------------------------
    @staticmethod
    def reduce(ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble ``(n, C, H, W)`` to per-cell mean and unbiased variance."""
        return ensemble.mean(axis=0), ensemble.var(axis=0, ddof=1)

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        """Assimilate one network. The prior carries its own means, so the background is unused."""
        ens = self.sample_posterior(obs.gather, obs.y_anom, obs.sse, self.b_scale,
                                    self.n_samples)
        mean, var = self.reduce(ens)
        return AnalysisResult(mean_anom=mean, posterior_var=var)
