"""Guided posterior sampling from a score prior confined to a whitened subspace.

The state splits along the leading ``k`` eigen-directions of the background
covariance B and their orthogonal complement,

    x = U_k a + r,      a = U_k' x  (k-dim),    r = (I - U_k U_k') x

and because those are *eigen*vectors the split is exact: ``B = U_k diag(lam_k) U_k'
+ C_perp`` with no cross terms. The learned prior lives on ``a`` alone; ``r`` stays
Gaussian with covariance ``C_perp`` and independent of ``a``. Three things follow,
and together they are the reason this exists.

**The Gaussian part is imposed, not learned.** In the whitened coordinate
``z = sigma_data U_k' x / sqrt(lam_k)`` the Gaussian model of the subspace is exactly
``N(0, sigma_data^2 I)``, which is what EDM's preconditioning is built for: its
``c_skip x`` term *is* that distribution's denoiser. A network output of zero
therefore reproduces the Gaussian prior, and the posterior mean of this sampler
reduces to the 3DVar analysis ``B H' (H B H' + R)^-1 y`` exactly. The method
contains its own baseline.

**Beyond k the prior is Gaussian by construction**, which is what the principal-
component evidence asks for: the LOVECLIM states are bimodal in their leading
components and unremarkable after that, and a Gaussian is the right model for the
tail rather than a concession.

**The complement is marginalised rather than approximated.** Since ``r`` is
genuinely Gaussian and independent of ``z``,

    y | z  ~  N(A z, H C_perp H' + R),     A = H U_k sqrt(lam_k) / sigma_data

is exact, so guidance only ever approximates the k-dimensional non-Gaussian part.
The annealed guidance covariance (Boys et al. 2023; Rozet and Louppe 2023) is
likewise exact for the anchor, because ``Cov(z | z_sigma)`` really is
``sigma_data^2 sigma^2 / (sigma_data^2 + sigma^2)`` times the identity. The
complement is then drawn from its exact conditional by the perturbed-observation
identity.

The tuned operating point is ``b_scale``, a genuine amplitude on B - the same
quantity 3DVar tunes, not a separate guidance strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from paleoreco.assim.generative import _edm_sigmas
from paleoreco.assim.method import AnalysisResult, Method, Observations
from paleoreco.models.diffusion import RHO, SIGMA_DATA, SIGMA_MAX, SIGMA_MIN

# How the likelihood's conditional covariance Cov(z | z_sigma) is obtained.
# ``anchor`` uses the Gaussian anchor's closed form, a scalar times the identity,
# which is exact only where the prior is Gaussian. ``jacobian`` uses the model's
# own, sigma^2 grad D, which is exact for whatever the network has learned and
# reduces to the anchor when the network output is zero.
_COND_COV = frozenset({"anchor", "jacobian"})


@dataclass(frozen=True)
class SpectralBasis:
    """The eigen-split of a background covariance, and the whitening it induces.

    ``U_K``/``lam_K`` are the leading eigenpairs of ``B`` (descending);
    ``half_perp`` factorises the complement, ``C_perp = half_perp half_perp'``,
    keeping only strictly positive modes so a rank-deficient B costs nothing.
    ``W`` maps a whitened coordinate to the subspace part of the state.
    ``B`` itself is retained because the observed blocks of ``C_perp`` are cheaper
    to take from it than to rebuild from ``half_perp``. ``meta`` records how the
    basis was built, so a run's config can state what the prior believed.
    """

    U_K: np.ndarray            # (D, K) float64, orthonormal
    lam_K: np.ndarray          # (K,) float64, strictly positive, descending
    W: np.ndarray              # (D, K) float64, U_K * sqrt(lam_K) / sigma_data
    half_perp: np.ndarray      # (D, R) float32, C_perp = half_perp half_perp'
    B: np.ndarray              # (D, D) float64
    shape: tuple[int, int, int]
    safe_valid: np.ndarray
    sigma_data: float
    meta: dict

    @property
    def n_dim(self) -> int:
        """Subspace dimension ``k``."""
        return int(self.lam_K.size)

    def whiten(self, states: np.ndarray) -> np.ndarray:
        """Anomaly states to whitened coordinates, ``(N, ...) -> (N, k)``.

        Under ``x ~ N(0, B)`` the result has covariance ``sigma_data^2 I``, which
        is the frame the denoiser is anchored to.
        """
        x = np.asarray(states, dtype=np.float64).reshape(len(states), -1)
        if x.shape[1] != self.U_K.shape[0]:
            raise ValueError(f"states flatten to {x.shape[1]} values, not the "
                             f"{self.U_K.shape[0]}-dim state this basis describes")
        return (x @ self.U_K) * (self.sigma_data / np.sqrt(self.lam_K))

    def unwhiten(self, z: np.ndarray) -> np.ndarray:
        """Whitened coordinates to the subspace part of the state, ``(N, k) -> (N, D)``."""
        z = np.asarray(z, dtype=np.float64).reshape(-1, self.n_dim)
        return z @ self.W.T


def build_spectral_basis(
    B: np.ndarray, shape: tuple[int, int, int], safe_valid: np.ndarray, k: int, *,
    sigma_data: float = SIGMA_DATA, eig_tol: float = 1e-10,
    meta: dict | None = None,
) -> SpectralBasis:
    """Eigen-split a background covariance into a leading-``k`` block and its complement.

    ``B`` is symmetrised and projected to the positive semi-definite cone before
    the split. That is not defensive tidying: the tuned pixel covariance is a Schur
    product of Gaspari-Cohn localisation with a cross-channel coupling factor, and
    great-circle Gaspari-Cohn is not guaranteed positive definite, so the tuned PPE
    B carries small negative eigenvalues. Gain-form 3DVar never notices, but
    whitening takes a square root and would.
    """
    Bm = np.asarray(B, dtype=np.float64)
    d = int(np.prod(shape))
    if Bm.ndim != 2 or Bm.shape[0] != Bm.shape[1]:
        raise ValueError(f"B must be square; got shape {Bm.shape}")
    if Bm.shape[0] != d:
        raise ValueError(f"B is {Bm.shape[0]}-dim but shape {shape} flattens to {d}")
    if not 1 <= int(k) < d:
        raise ValueError(f"k must satisfy 1 <= k < {d}; got {k}")
    k = int(k)

    lam, U = np.linalg.eigh(0.5 * (Bm + Bm.T))
    lam = np.ascontiguousarray(lam[::-1])
    U = np.ascontiguousarray(U[:, ::-1])
    n_clipped = int((lam < 0.0).sum())
    lam = np.clip(lam, 0.0, None)

    floor = eig_tol * max(float(lam[0]), 1.0)
    lam_K = lam[:k].copy()
    bad = np.flatnonzero(lam_K <= floor)
    if bad.size:
        raise ValueError(
            f"leading eigenvalue {int(bad[0]) + 1} of B is {lam_K[bad[0]]:.3e}, too "
            f"small to whiten against; use k <= {int(bad[0])}")
    U_K = U[:, :k].copy()
    W = U_K * (np.sqrt(lam_K) / float(sigma_data))

    rest_lam, rest_U = lam[k:], U[:, k:]
    keep = rest_lam > floor
    half_perp = (rest_U[:, keep] * np.sqrt(rest_lam[keep])).astype(np.float32)

    total = float(lam.sum())
    info = {
        "k": k,
        "sigma_data": float(sigma_data),
        "leading_variance_share": float(lam_K.sum() / total) if total > 0 else float("nan"),
        "n_negative_clipped": n_clipped,
        "n_complement_modes": int(keep.sum()),
        **(meta or {}),
    }
    return SpectralBasis(U_K=U_K, lam_K=lam_K, W=W, half_perp=half_perp, B=Bm,
                         shape=tuple(shape),
                         safe_valid=np.asarray(safe_valid, dtype=bool),
                         sigma_data=float(sigma_data), meta=info)


def match_subspace_covariance(
    basis: SpectralBasis, states: np.ndarray, *, max_condition: float = 1e8,
) -> tuple[np.ndarray, dict]:
    """Rescale states so their retained-subspace covariance equals B's leading block.

    The whitening of :class:`SpectralBasis` divides by B's eigenvalues, so the
    *anchor* carries B's covariance. The training states do not: their empirical
    variance in each retained direction differs from the matching eigenvalue, and
    they are correlated across directions, because ``U_K`` diagonalises the tapered
    B and not the sample covariance. Training therefore pulls the model off the
    tuned covariance and onto the empirical one, undoing the localisation and
    cross-channel taper in exactly the directions that carry the most weight.

    This restores the second moment before training. With ``a = U_K' x`` and
    ``Cd = Cov(a)``, the states are mapped through

        T = Cd^(-1/2) diag(sqrt(lam_K)),    so that Cov(a T) = diag(lam_K) exactly

    and only the subspace part moves; the complement is left alone because it is
    never learned and the sampler already draws it from B. ``T`` is an invertible
    linear map, so this fixes the covariance **without** touching the higher
    moments: skew, kurtosis and any multimodality survive it intact. It is not a
    step toward a Gaussian prior, it is what makes the learned prior's Gaussian part
    the one the lane's 3DVar baseline was tuned with.

    Returns the transformed states, in the shape they were given, and a dict of
    diagnostics for the caller to log or record in a run config.
    """
    x = np.asarray(states, dtype=np.float64).reshape(len(states), -1)
    k = basis.n_dim
    if x.shape[1] != basis.U_K.shape[0]:
        raise ValueError(f"states flatten to {x.shape[1]} values, not the "
                         f"{basis.U_K.shape[0]}-dim state this basis describes")
    if len(x) <= k:
        raise ValueError(f"need more than k={k} states to estimate the subspace "
                         f"covariance; got {len(x)}")

    a = x @ basis.U_K                                        # (N, k)
    mean_a = a.mean(axis=0)
    a = a - mean_a                                           # the anchor is zero-mean
    Cd = np.cov(a, rowvar=False, ddof=1)
    Cd = np.atleast_2d(0.5 * (Cd + Cd.T))

    ev, V = np.linalg.eigh(Cd)
    if ev.min() <= 0.0:
        raise ValueError(f"the subspace covariance is singular (smallest eigenvalue "
                         f"{ev.min():.3e}); use a smaller k than {k}")
    condition = float(ev.max() / ev.min())
    if condition > max_condition:
        raise ValueError(f"the subspace covariance is ill-conditioned "
                         f"({condition:.3e} > {max_condition:.3e}); use a smaller k "
                         f"than {k}")

    # The symmetric (Mahalanobis) inverse square root is the whitening closest to the
    # identity, so this is the least-rotating transform that can pin the covariance.
    # It still mixes the directions whenever ``Cd`` has off-diagonal structure, which
    # it does here because ``U_K`` diagonalises the tapered B and not the sample
    # covariance. The joint distribution is unchanged - ``transform`` is invertible, so
    # every projection of the original survives, just along a different axis - but the
    # per-coordinate marginals afterwards are blends and look more Gaussian than the
    # originals did. Read the shape of the result through ``transform``, not off the
    # new axes.
    T = (V * ev ** -0.5) @ V.T @ np.diag(np.sqrt(basis.lam_K))
    matched = x + (a @ T - (a + mean_a)) @ basis.U_K.T
    info = {
        "condition": condition,
        "sd_before": (a + mean_a).std(axis=0),
        "sd_after": (a @ T).std(axis=0),
        "mean_removed": float(np.abs(mean_a).max()),
        "transform": T,
    }
    return matched.reshape(np.shape(states)), info


class SpectralSampler(Method):
    """Posterior sampler over a subspace score prior with an exact Gaussian complement.

    ``b_scale`` is the background-covariance amplitude, the operating point the
    experiment runners sweep and record in the ``b_scale`` column; it scales the
    subspace block and the complement together, so the Gaussian limit of this
    sampler is 3DVar at that same amplitude. ``n_samples`` sets the ensemble
    :meth:`analyze` draws; ``max_batch`` caps how many fields one call materialises
    at once.

    ``cond_cov`` chooses the conditional covariance the likelihood is annealed with.
    ``anchor`` is the Gaussian anchor's closed form, a scalar times the identity;
    it is exact only where the prior is Gaussian, and so mis-weights the update in
    precisely the directions the learned prior departs from one. ``jacobian`` uses
    the model's own conditional covariance ``sigma^2 grad D`` (second-order
    Tweedie), which is exact for whatever the network has learned and reduces to
    ``anchor`` when the network output is zero. The Jacobian is affordable only
    because it is ``k x k``; in the full state it would not be.

    Unlike :class:`~paleoreco.assim.generative.GuidedSampler` there is no worker
    pool: the reverse process runs over ``k`` dimensions rather than 4096, so a
    draw costs a handful of small matrix products and the network is a few-layer
    MLP.
    """

    def __init__(self, denoiser, basis: SpectralBasis, *,
                 b_scale: float = 1.0, n_samples: int = 256, n_steps: int = 64,
                 sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX,
                 rho: float = RHO, max_batch: int | None = None,
                 cond_cov: str = "anchor", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.denoiser = denoiser.to(self.device).eval()
        self.basis = basis
        # The runners' provenance record reads ``sampler.cov.meta``; for this method
        # the object describing what the guidance believes is the basis itself.
        self.cov = basis
        self.sd = float(getattr(denoiser, "sigma_data", basis.sigma_data))
        if not np.isclose(self.sd, basis.sigma_data):
            raise ValueError(f"denoiser sigma_data {self.sd} does not match the basis's "
                             f"{basis.sigma_data}; the anchor and the whitening disagree")
        dim = getattr(denoiser, "config", {}).get("dim")
        if dim is not None and int(dim) != basis.n_dim:
            raise ValueError(f"denoiser is {dim}-dim but the basis is {basis.n_dim}-dim")
        if max_batch is not None and max_batch < 1:
            raise ValueError(f"max_batch must be >= 1 or None; got {max_batch}")
        if cond_cov not in _COND_COV:
            raise ValueError(f"cond_cov must be one of {sorted(_COND_COV)}; got {cond_cov!r}")

        self.cond_cov = cond_cov
        self._clipped, self._eig_total = 0, 0     # PSD-projection diagnostic
        self.b_scale = float(b_scale)
        self.n_samples = int(n_samples)
        self.n_steps = int(n_steps)
        self.max_batch = max_batch
        self.sigmas = _edm_sigmas(n_steps, sigma_min, sigma_max, rho)
        self.shape = basis.shape
        self.k = basis.n_dim
        self._mask = basis.safe_valid

    # -- score and reverse process ---------------------------------------
    @property
    def clip_rate(self) -> float:
        """Fraction of conditional-covariance eigenvalues clipped to zero so far.

        The model's Jacobian is only guaranteed to be a valid covariance where the
        network is a well-behaved score, so this reads as a health check: a rate far
        above zero means the learned prior is locally not log-concave and the
        guidance is leaning on the projection rather than on the model.
        """
        return self._clipped / self._eig_total if self._eig_total else 0.0

    def _conditional_cov(self, x_hat: torch.Tensor, xg: torch.Tensor,
                         sigma: float) -> torch.Tensor:
        """``Cov(z | z_sigma) = sigma^2 grad D``, the model's own, ``(n, k, k)``.

        Second-order Tweedie: ``D = z + sigma^2 grad log p`` gives ``grad D = I +
        sigma^2 grad^2 log p``, so ``sigma^2 grad D`` is exactly the conditional
        covariance. In ``k`` dimensions the Jacobian is ``k`` backward passes rather
        than ``n * k``, because the denoiser acts independently per sample and so the
        ``j``-th pass returns row ``j`` of every sample's Jacobian at once.

        The result is detached: it enters the likelihood as a covariance, not as
        something to differentiate through, exactly as the fixed ``S^-1`` table does
        in the anchor branch. Differentiating it would need second derivatives of the
        denoiser for no established benefit.
        """
        flat = x_hat.reshape(x_hat.shape[0], -1)
        rows = [torch.autograd.grad(flat[:, j].sum(), xg, retain_graph=True)[0]
                .reshape(flat.shape[0], -1)
                for j in range(self.k)]
        gam = sigma ** 2 * torch.stack(rows, dim=1).detach()      # [n, j, l]
        gam = 0.5 * (gam + gam.transpose(1, 2))   # a log-density Hessian is symmetric
        # Project to the PSD cone. Clipping to zero rather than to a floor is the
        # correct projection and is safe: the Woodbury form needs no inverse of gam,
        # so a zeroed direction simply drops the prior's contribution there.
        ev, V = torch.linalg.eigh(gam)
        self._clipped += int((ev < 0).sum())
        self._eig_total += ev.numel()
        ev = ev.clamp_min(0.0)
        return (V * ev[:, None, :]) @ V.transpose(1, 2)

    def _score(self, x: torch.Tensor, sigma: float, guidance) -> torch.Tensor:
        """Prior score, plus the guidance score when ``guidance`` is given."""
        if guidance is None:
            with torch.no_grad():
                x_hat = self.denoiser(x, sigma)
            return (x_hat - x) / sigma ** 2
        xg = x.detach().requires_grad_(True)
        x_hat = self.denoiser(xg, sigma)
        # One forward pass serves both the Jacobian and the denoised estimate, so the
        # conditional covariance is evaluated at exactly the point being guided.
        gam = (self._conditional_cov(x_hat, xg, sigma)
               if self.cond_cov == "jacobian" else None)
        loglik = guidance(x_hat, sigma, gam)
        g = torch.autograd.grad(loglik, xg)[0]
        prior = (x_hat.detach() - xg.detach()) / sigma ** 2
        return prior + g

    def _sample_z(self, n: int, guidance, seed: int) -> np.ndarray:
        """EDM Heun predictor over ``R^k``; whitened coordinates ``(n, k)``."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = self.sigmas[0] * torch.randn((n, self.k, 1, 1), generator=gen,
                                         device=self.device)
        for i in range(self.n_steps):
            sig, sig_next = float(self.sigmas[i]), float(self.sigmas[i + 1])
            d = -sig * self._score(x, sig, guidance)
            x_pred = x + (sig_next - sig) * d
            if sig_next <= 0.0:
                x = x_pred
                continue
            d2 = -sig_next * self._score(x_pred, sig_next, guidance)
            x = x + (sig_next - sig) * 0.5 * (d + d2)
        return x.detach().reshape(n, self.k).cpu().numpy().astype(np.float64)

    # -- the exact Gaussian complement -----------------------------------
    def _complement_blocks(self, gather: np.ndarray):
        """``(H C_perp H', C_perp H')`` for one network, without forming ``C_perp``."""
        b = self.basis
        Ug = b.U_K[gather]                                       # (m, k)
        block = b.B[np.ix_(gather, gather)] - (Ug * b.lam_K) @ Ug.T
        cols = b.B[:, gather] - b.U_K @ (b.lam_K[:, None] * Ug.T)
        return 0.5 * (block + block.T), cols

    def _draw_complement(self, rng, n: int, k_amp: float) -> np.ndarray:
        """``n`` unconditional complement draws ``r0 ~ N(0, k_amp C_perp)``, ``(n, D)``."""
        xi = rng.standard_normal((n, self.basis.half_perp.shape[1]), dtype=np.float32)
        return np.sqrt(k_amp) * (xi @ self.basis.half_perp.T).astype(np.float64)

    def _finish(self, z: np.ndarray, r_full: np.ndarray, k_amp: float) -> np.ndarray:
        """Assemble states from whitened coordinates and complement, ``(n, C, H, W)``."""
        x = np.sqrt(k_amp) * self.basis.unwhiten(z) + r_full
        return x.reshape(len(z), *self.shape) * self._mask

    @staticmethod
    def _chunks(n: int, max_batch: int | None):
        """``(offset, size)`` pairs covering ``n`` draws."""
        if max_batch is None or n <= max_batch:
            return [(0, n)]
        return [(o, min(max_batch, n - o)) for o in range(0, n, max_batch)]

    @staticmethod
    def _streams(seed: int, chunk: int):
        """Torch seed and an independent numpy generator for one chunk.

        Chunk seeds sit far above the small consecutive integers the experiment
        runners assign per job, so no chunk shares a stream with another job.
        """
        s = int(seed) if chunk == 0 else (int(seed) << 32) + (1 << 31) + chunk
        return s, np.random.default_rng([s, 0x5E])

    # -- public sampling --------------------------------------------------
    def sample_prior(self, n: int, seed: int = 0,
                     b_scale: float | None = None) -> np.ndarray:
        """``n`` unconditional prior fields in anomaly units, ``(n, 2, H, W)``.

        Defaults to the sampler's own operating point, so the spread diagnostics
        the runners store describe the amplitude the analysis actually used.
        """
        k_amp = self.b_scale if b_scale is None else float(b_scale)
        out = []
        for ci, (_, size) in enumerate(self._chunks(n, self.max_batch)):
            tseed, rng = self._streams(seed, ci)
            z = self._sample_z(size, None, tseed)
            out.append(self._finish(z, self._draw_complement(rng, size, k_amp), k_amp))
        return np.concatenate(out) if len(out) > 1 else out[0]

    def sample_posterior(self, gather: np.ndarray, y_anom: np.ndarray,
                         r_diag: np.ndarray, b_scale: float, n: int,
                         seed: int = 0) -> np.ndarray:
        """``n`` posterior fields conditioned on one observation network, ``(n, 2, H, W)``."""
        g = np.asarray(gather)
        y = np.asarray(y_anom, dtype=np.float64)
        r = np.asarray(r_diag, dtype=np.float64)
        k_amp = float(b_scale)
        b = self.basis

        # Exact pieces of the marginalised likelihood y | z ~ N(A z, C_y).
        A = np.sqrt(k_amp) * b.W[g]                              # (m, k)
        block, cols = self._complement_blocks(g)                 # (m, m), (D, m)
        C_y = k_amp * block + np.diag(r)
        chol_y = cho_factor(C_y)

        A_t = torch.as_tensor(A, device=self.device, dtype=torch.float32)
        y_t = torch.as_tensor(y, device=self.device, dtype=torch.float32)

        if self.cond_cov == "anchor":
            # S(sigma) inverted per noise level in float64, then held constant as
            # float32 in the graph: the quadratic form's gradient stays exact while
            # the m x m inversions stay well conditioned. One table serves every
            # sample because the anchor's conditional covariance is a scalar.
            AAt = A @ A.T
            sinv = {}
            for s in self.sigmas[:-1]:
                f = self.sd ** 2 * s ** 2 / (self.sd ** 2 + s ** 2)
                S = torch.as_tensor(f * AAt + C_y, dtype=torch.float64)
                Si = torch.cholesky_inverse(torch.linalg.cholesky(S))
                sinv[float(s)] = (0.5 * (Si + Si.T)).to(device=self.device,
                                                        dtype=torch.float32)

            def guidance(x_hat: torch.Tensor, sigma: float, gam) -> torch.Tensor:
                resid = y_t[None, :] - x_hat.reshape(x_hat.shape[0], -1) @ A_t.T
                return -0.5 * torch.einsum("nm,mk,nk->", resid, sinv[float(sigma)],
                                           resid)
        else:
            # The conditional covariance is now per sample, so S is too and a table
            # is no longer possible. S = C_y + A Gamma A' is a rank-k update of one
            # fixed matrix, so Woodbury turns the per-sample cost from an m x m
            # Cholesky into a k x k solve:
            #     S^-1 = Cy^-1 - M (I + Gamma G)^-1 Gamma M',
            #     M = Cy^-1 A,  G = A' Cy^-1 A
            # using (Gamma^-1 + G)^-1 = (I + Gamma G)^-1 Gamma, which needs no
            # inverse of Gamma and so tolerates the PSD projection zeroing it.
            Cy_inv = np.linalg.inv(C_y)
            Cy_inv = 0.5 * (Cy_inv + Cy_inv.T)
            M = Cy_inv @ A                                       # (m, k)
            G = A.T @ M                                          # (k, k)
            Cy_inv_t = torch.as_tensor(Cy_inv, device=self.device, dtype=torch.float32)
            M_t = torch.as_tensor(M, device=self.device, dtype=torch.float32)
            G_t = torch.as_tensor(0.5 * (G + G.T), device=self.device,
                                  dtype=torch.float32)
            eye_k = torch.eye(self.k, device=self.device, dtype=torch.float32)

            def guidance(x_hat: torch.Tensor, sigma: float, gam) -> torch.Tensor:
                resid = y_t[None, :] - x_hat.reshape(x_hat.shape[0], -1) @ A_t.T
                base = ((resid @ Cy_inv_t) * resid).sum(dim=1)          # r' Cy^-1 r
                # (I + Gamma G) has eigenvalues at least 1, so this solve is safe.
                W = torch.linalg.solve(eye_k + gam @ G_t, gam)
                W = 0.5 * (W + W.transpose(1, 2))
                q = resid @ M_t                                          # (n, k)
                low = torch.einsum("nk,nkl,nl->n", q, W, q)
                return -0.5 * (base - low).sum()

        out = []
        for ci, (_, size) in enumerate(self._chunks(n, self.max_batch)):
            tseed, rng = self._streams(seed, ci)
            z = self._sample_z(size, guidance, tseed)
            # Complement from its exact conditional, by the perturbed-observation
            # identity: r0 plus the gain applied to a perturbed innovation.
            r0 = self._draw_complement(rng, size, k_amp)
            e0 = rng.standard_normal((size, len(g))) * np.sqrt(r)
            pred = np.sqrt(k_amp) * (z @ b.W[g].T)               # A z
            innov = y[None, :] - pred - r0[:, g] - e0
            r_full = r0 + k_amp * (cho_solve(chol_y, innov.T).T @ cols.T)
            out.append(self._finish(z, r_full, k_amp))
        return np.concatenate(out) if len(out) > 1 else out[0]

    def imap_posterior(self, jobs):
        """Posterior ensembles for a sequence of ``PosteriorJob``, in order.

        The job's ``gamma`` field carries this method's operating point, which is
        the background amplitude rather than a guidance strength.
        """
        for j in jobs:
            yield self.sample_posterior(j.gather, j.y_anom, j.r_diag, j.gamma,
                                        j.n, j.seed)

    # -- Method contract --------------------------------------------------
    @staticmethod
    def reduce(ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble ``(n, 2, H, W)`` to per-cell mean and unbiased variance."""
        return ensemble.mean(axis=0), ensemble.var(axis=0, ddof=1)

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        """Assimilate one network. The prior is centred, so the background is unused."""
        ens = self.sample_posterior(obs.gather, obs.y_anom, obs.sse, self.b_scale,
                                    self.n_samples)
        mean, var = self.reduce(ens)
        return AnalysisResult(mean_anom=mean, posterior_var=var)
