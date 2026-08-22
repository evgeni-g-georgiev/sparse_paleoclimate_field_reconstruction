"""A regime mixture over the state, and its exact posterior.

    p(x) = sum_j w_j N(x; m_j, k C_j)

The components are climate regimes found in the prior states themselves, so the
observations both select a regime and refine within it. A Gaussian mixture under a
linear-Gaussian likelihood has a Gaussian-mixture posterior, so that analysis is closed
form: one Kalman update per component, weights by marginal likelihood.

At ``J = 1`` the component is ``N(0, k C_pool)``, which is 3DVar at background amplitude
``k`` when ``C_pool`` is built as :func:`~paleoreco.assim.priors.build_prior` builds B.

Covariances serve the closed-form posterior; their eigendecompositions serve the diffusion
arm, which needs a diagonal basis at every noise level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ``kmeans`` clusters the states by their leading principal scores. ``time`` cuts the
# record into contiguous age windows instead, which is the control for whether a partition
# has to be in state space or whether any split of the prior would do.
PARTITIONS = ("kmeans", "time")


def partition_states(states: np.ndarray, n_regimes: int, *, rule: str = "kmeans",
                     ages: np.ndarray | None = None, n_scores: int = 2,
                     seed: int = 0) -> np.ndarray:
    """Label each state with a regime index, ``(N,) -> (N,)`` ints in ``[0, n_regimes)``.

    ``kmeans`` reads the leading ``n_scores`` principal scores of ``states``; ``time``
    needs ``ages`` and cuts them into contiguous quantile windows. ``n_regimes = 1``
    returns all zeros under either rule.
    """
    x = np.asarray(states, dtype=np.float64).reshape(len(states), -1)
    j = int(n_regimes)
    if j < 1:
        raise ValueError(f"n_regimes must be >= 1; got {n_regimes}")
    if j == 1:
        return np.zeros(len(x), dtype=np.int64)
    if rule not in PARTITIONS:
        raise ValueError(f"rule must be one of {PARTITIONS}; got {rule!r}")

    if rule == "time":
        if ages is None:
            raise ValueError("the 'time' partition needs ages to cut the record on")
        a = np.asarray(ages, dtype=np.float64)
        edges = np.quantile(a, np.linspace(0.0, 1.0, j + 1))
        return np.clip(np.searchsorted(edges[1:-1], a), 0, j - 1).astype(np.int64)

    xc = x - x.mean(0, keepdims=True)
    v = np.linalg.svd(xc, full_matrices=False)[2][:max(n_scores, 1)]
    scores = xc @ v.T

    from sklearn.cluster import KMeans
    return KMeans(j, n_init=10, random_state=seed).fit_predict(scores).astype(np.int64)


@dataclass(frozen=True)
class RegimeMixture:
    """``sum_j w_j N(m_j, C_j)`` over the flattened state, plus its eigenbases.

    The amplitude ``k`` is not stored: it multiplies every covariance at use, so one
    mixture serves a whole ``b_scale`` sweep. Numpy only, so the object pickles.
    """

    w: np.ndarray                   # (J,)
    mu: np.ndarray                  # (J, D)
    cov: np.ndarray                 # (J, D, D)
    U: np.ndarray                   # (J, D, D), each orthonormal
    lam: np.ndarray                 # (J, D), non-negative
    shape: tuple[int, int, int]
    safe_valid: np.ndarray
    meta: dict

    @property
    def n_regimes(self) -> int:
        return int(len(self.w))

    @property
    def n_dim(self) -> int:
        return int(self.mu.shape[1])


def build_regime_mixture(
    states: np.ndarray, labels: np.ndarray, shape: tuple[int, int, int],
    safe_valid: np.ndarray, *, mask: np.ndarray | None = None, rho: float = 0.3,
    eig_floor: float = 1e-6, meta: dict | None = None,
) -> RegimeMixture:
    """Fit the mixture to labelled prior states.

    ``states`` are the prior anomalies the lane's B is built from, centred here on their
    pooled mean so that ``J = 1`` reproduces ``N(0, B)``. ``mask`` is the lane's Schur
    taper (:func:`~paleoreco.assim.priors.regularization_mask`), applied to every component
    so each is regularised as the baseline's B is. ``rho`` shrinks each component toward
    the pooled covariance, since a regime holds only a fraction of the states.
    """
    x = np.asarray(states, dtype=np.float64).reshape(len(states), -1)
    lab = np.asarray(labels, dtype=np.int64)
    if len(lab) != len(x):
        raise ValueError(f"got {len(lab)} labels for {len(x)} states")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1]; got {rho}")
    d = int(np.prod(shape))
    if x.shape[1] != d:
        raise ValueError(f"states flatten to {x.shape[1]} values, not the {d} that "
                         f"shape {shape} describes")

    x = x - x.mean(0, keepdims=True)          # the pooled component is zero-mean
    pooled = np.cov(x, rowvar=False, ddof=1)
    ids = np.unique(lab)

    w, mu, cov, sizes = [], [], [], []
    for j in ids:
        xi = x[lab == j]
        if len(xi) < 2:
            raise ValueError(f"regime {int(j)} holds {len(xi)} states, too few for a "
                             f"covariance; use fewer regimes")
        c = (1.0 - rho) * np.cov(xi, rowvar=False, ddof=1) + rho * pooled
        if mask is not None:
            c = c * mask
        w.append(len(xi) / len(x))
        mu.append(xi.mean(0))
        cov.append(0.5 * (c + c.T))
        sizes.append(len(xi))

    # Great-circle Gaspari-Cohn is not positive definite, so a tapered sample covariance
    # carries small negative eigenvalues, and one estimated from a few hundred states in
    # four thousand dimensions holds no information below the noise floor. The gain form
    # tolerates both; a reverse diffusion does not, since it takes the covariance apart.
    # Each component is therefore rebuilt from its floored eigendecomposition, so the
    # closed-form and diffusion arms read the same covariance rather than two that differ
    # in the tail.
    cov = np.stack(cov)
    U, lam = np.empty_like(cov), np.empty((len(cov), d))
    conds, n_clipped = [], 0
    for j, c in enumerate(cov):
        ev, vec = np.linalg.eigh(c)
        ev, vec = ev[::-1], np.ascontiguousarray(vec[:, ::-1])
        n_clipped += int((ev < 0.0).sum())
        floor = eig_floor * max(float(ev[0]), 0.0)
        lam[j], U[j] = np.clip(ev, floor, None), vec
        cov[j] = (U[j] * lam[j]) @ U[j].T
        conds.append(float(lam[j][0] / lam[j][-1]))

    info = {
        "n_regimes": int(len(ids)), "rho": float(rho),
        "sizes": [int(s) for s in sizes],
        "weights": [float(v) for v in w],
        "conditions": conds, "eig_floor": float(eig_floor),
        "n_negative_clipped": int(n_clipped),
        # How far apart the regimes sit, in units of the pooled spread.
        "mean_separation": float(np.max([np.linalg.norm(a - b) for a in mu for b in mu])
                                 / np.sqrt(max(np.trace(pooled), 1e-12))),
        **(meta or {}),
    }
    return RegimeMixture(w=np.asarray(w), mu=np.stack(mu), cov=cov, U=U, lam=lam,
                         shape=tuple(shape),
                         safe_valid=np.asarray(safe_valid, dtype=bool), meta=info)


# ---------------------------------------------------------------------------
# Exact inference: a Gaussian mixture under a linear-Gaussian likelihood.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Posterior:
    """The exact posterior of one assimilation: weights, means, and the gain per component."""

    p: np.ndarray                   # (J,) posterior weights
    means: np.ndarray               # (J, D)
    gains: list                     # per component, (cho_factor of S_j, k C_j H^T)
    gather: np.ndarray


def exact_posterior(mix: RegimeMixture, gather: np.ndarray, r_diag: np.ndarray,
                    y: np.ndarray, b_scale: float) -> _Posterior:
    """Closed-form posterior: a Kalman update per component, weights by marginal likelihood.

    ``H`` is nearest-cell selection, so ``H C_j H^T`` is the observed submatrix and the
    only inverse is the ``m x m`` one 3DVar already forms - once per component.
    """
    g = np.asarray(gather)
    r = np.asarray(r_diag, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = float(b_scale)

    logw = np.empty(mix.n_regimes)
    means = np.empty((mix.n_regimes, mix.n_dim))
    gains = []
    for j in range(mix.n_regimes):
        c = mix.cov[j]
        chol = cho_factor(k * c[np.ix_(g, g)] + np.diag(r), check_finite=False)
        cols = k * c[:, g]                                   # k C_j H^T, (D, m)
        d = y - mix.mu[j][g]
        sol = cho_solve(chol, d, check_finite=False)
        means[j] = mix.mu[j] + cols @ sol
        logw[j] = (np.log(mix.w[j])
                   - 0.5 * (2.0 * np.log(np.diag(chol[0])).sum() + d @ sol))
        gains.append((chol, cols))

    logw -= logw.max()
    p = np.exp(logw)
    return _Posterior(p=p / p.sum(), means=means, gains=gains, gather=g)


def exact_posterior_mean(mix: RegimeMixture, gather, r_diag, y, b_scale) -> np.ndarray:
    """Posterior mean under the mixture, ``(D,)``. The quantity the skill metrics read."""
    post = exact_posterior(mix, gather, r_diag, y, b_scale)
    return post.p @ post.means


def exact_posterior_mean_sweep(mix: RegimeMixture, gather, r_diag, y,
                               b_scales) -> np.ndarray:
    """Posterior means at every amplitude in ``b_scales``, ``(n_b, D)``.

    The same answer as :func:`exact_posterior_mean` per amplitude, with each component's
    observed columns gathered once rather than once per amplitude.
    """
    g = np.asarray(gather)
    r = np.asarray(r_diag, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    blocks = [(mix.cov[j][np.ix_(g, g)], mix.cov[j][:, g], y - mix.mu[j][g])
              for j in range(mix.n_regimes)]

    out = np.empty((len(b_scales), mix.n_dim))
    for bi, k in enumerate(b_scales):
        logw = np.empty(mix.n_regimes)
        means = np.empty((mix.n_regimes, mix.n_dim))
        for j, (cgg, cols, d) in enumerate(blocks):
            chol = cho_factor(float(k) * cgg + np.diag(r), check_finite=False)
            sol = cho_solve(chol, d, check_finite=False)
            means[j] = mix.mu[j] + float(k) * (cols @ sol)
            logw[j] = (np.log(mix.w[j])
                       - 0.5 * (2.0 * np.log(np.diag(chol[0])).sum() + d @ sol))
        logw -= logw.max()
        p = np.exp(logw)
        out[bi] = (p / p.sum()) @ means
    return out


def exact_posterior_sample(mix: RegimeMixture, gather, r_diag, y, b_scale, n: int,
                           rng) -> np.ndarray:
    """``n`` exact draws from the mixture posterior, ``(n, D)``.

    Draw a component from the posterior weights, then draw from that component's Gaussian
    posterior by the perturbed-observation identity, which avoids factorising
    ``(I - KH) k C_j``.
    """
    post = exact_posterior(mix, gather, r_diag, y, b_scale)
    g, k = post.gather, float(b_scale)
    r = np.asarray(r_diag, dtype=np.float64)
    counts = rng.multinomial(n, post.p)

    out = np.empty((n, mix.n_dim))
    at = 0
    for j, c in enumerate(counts):
        if c == 0:
            continue
        chol, cols = post.gains[j]
        xi = rng.standard_normal((c, mix.n_dim))
        # C_j = U diag(lam) U', so a draw is (xi * sqrt(lam)) U'. Scaling the sample before
        # projecting avoids materialising U diag(sqrt(lam)), a (D, D) temporary per call.
        draw = mix.mu[j] + np.sqrt(k) * ((xi * np.sqrt(mix.lam[j])) @ mix.U[j].T)
        e = rng.standard_normal((c, len(g))) * np.sqrt(r)
        innov = y[None, :] - draw[:, g] - e
        out[at:at + c] = draw + cho_solve(chol, innov.T, check_finite=False).T @ cols.T
        at += c
    return out[rng.permutation(n)]                           # unsort, so order carries nothing


def prior_sample(mix: RegimeMixture, b_scale: float, n: int, rng) -> np.ndarray:
    """``n`` unconditional draws from the mixture, ``(n, D)``."""
    counts = rng.multinomial(n, mix.w)
    out = np.empty((n, mix.n_dim))
    at = 0
    for j, c in enumerate(counts):
        if c == 0:
            continue
        xi = rng.standard_normal((c, mix.n_dim))
        out[at:at + c] = mix.mu[j] + np.sqrt(b_scale) * ((xi * np.sqrt(mix.lam[j]))
                                                         @ mix.U[j].T)
        at += c
    return out[rng.permutation(n)]
