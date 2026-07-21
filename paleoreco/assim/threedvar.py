"""3DVar analysis in the gain (dual) form.

With a linear observation operator (nearest-cell selection) the 3DVar cost has a
closed-form minimiser, so the analysis is computed directly as

    x_a = x_b + B H^T (H B H^T + R)^-1 (y - H x_b)

without ever inverting the dense D x D background covariance: the only inverse is
the m x m matrix over the m observations of one assimilation. The same factor
gives the posterior covariance A = (I - K H) B, whose diagonal is the posterior
variance map. A depends on B and the gain but not on x_b, so it is identical for
the climatological and per-age backgrounds.

Everything is in anomaly space; H is plain selection (scale one), so the
innovation is ``y_anom - x_b[gather]`` and ``H B H^T`` is the obs-cell submatrix
of B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from paleoreco.assim.method import AnalysisResult, Method, Observations


@dataclass(frozen=True)
class _Gain:
    """Per-network 3DVar operators that do not depend on y or the background.

    ``gather`` is H; ``B_obs`` is ``B H^T``; ``chol`` factorizes ``H B H^T + R``;
    ``post_var`` is the flat posterior-variance map. One network of observations
    reuses these across every innovation, so the analysis of many observation
    vectors costs one back-substitution each rather than a fresh solve.
    """

    gather: np.ndarray
    B_obs: np.ndarray
    chol: tuple
    post_var: np.ndarray


@dataclass(frozen=True)
class _SweepGain:
    """Per-network operators shared across a B-amplitude (``b_scale``) sweep.

    Scaling B by ``k`` holds R fixed and only reweights the spectrum of the
    R^-1/2-whitened obs block, so one eigendecomposition serves every ``k``: the
    D x m factor ``G`` is built once and each ``k`` is a matvec rather than a
    fresh solve. ``Lam``/``U`` diagonalize that whitened block; ``rinv_sqrt`` is
    R^-1/2 at the obs cells; ``G2`` is ``G**2``, reused for the posterior variance.
    """

    gather: np.ndarray
    Lam: np.ndarray
    U: np.ndarray
    G: np.ndarray
    G2: np.ndarray
    rinv_sqrt: np.ndarray
    diagB: np.ndarray
    b_scales: np.ndarray


class ThreeDVar(Method):
    """Gain-form 3DVar over a fixed background covariance."""

    def __init__(self, B: np.ndarray, shape: tuple[int, int, int]):
        # Column-major: every gain reads the obs columns ``B[:, gather]``, which is a
        # strided walk over the whole D x D array in row-major order and dominates the
        # per-network cost. Storing B column-major makes that gather contiguous for the
        # same values.
        self.B = np.asfortranarray(B, dtype=np.float64)
        self.diagB = np.diag(self.B).copy()
        self.shape = shape

    def prepare(self, gather: np.ndarray, sse: np.ndarray) -> _Gain:
        """Factorize the gain for one observation network (one solve over D rows)."""
        g = np.asarray(gather)
        B_obs = self.B[:, g]                               # B H^T, (D, m)
        S = self.B[np.ix_(g, g)] + np.diag(sse)           # H B H^T + R, (m, m)
        chol = cho_factor(S)

        # diag(K H B) = diag(B_obs S^-1 B_obs^T): per cell b_i^T S^-1 b_i.
        M = cho_solve(chol, B_obs.T)                      # (m, D)
        post_var = self.diagB - np.einsum("ik,ki->i", B_obs, M)
        return _Gain(gather=g, B_obs=B_obs, chol=chol, post_var=post_var)

    def apply(self, gain: _Gain, y_anom: np.ndarray, background_anom: np.ndarray) -> AnalysisResult:
        """Analysis for one innovation, reusing a factorized :class:`_Gain`."""
        x_b = np.asarray(background_anom, dtype=np.float64).ravel()
        d = np.asarray(y_anom, dtype=np.float64) - x_b[gain.gather]
        x_a = x_b + gain.B_obs @ cho_solve(gain.chol, d)
        return AnalysisResult(
            mean_anom=x_a.reshape(self.shape),
            posterior_var=gain.post_var.reshape(self.shape),
        )

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        return self.apply(self.prepare(obs.gather, obs.sse), obs.y_anom, background_anom)

    def analyze_many(
        self, obs: Observations, backgrounds: list[np.ndarray]
    ) -> list[AnalysisResult]:
        """One factorization, applied to each background; the gain is background-free."""
        gain = self.prepare(obs.gather, obs.sse)
        return [self.apply(gain, obs.y_anom, bg) for bg in backgrounds]

    # ------------------------------------------------------------------
    # B-amplitude sweep: one eigendecomposition reused across every b_scale.
    # ------------------------------------------------------------------
    def prepare_sweep(
        self, gather: np.ndarray, r_diag: np.ndarray, b_scales: np.ndarray
    ) -> _SweepGain:
        """Factorize the gain once for a network, reusable across all ``b_scales``.

        ``r_diag`` is the fixed observation-error variance per observation (R's
        diagonal); scaling B by ``b_scale`` does not touch it.
        """
        g = np.asarray(gather)
        r = np.asarray(r_diag, dtype=np.float64)
        rinv_sqrt = 1.0 / np.sqrt(r)
        A = self.B[np.ix_(g, g)]
        A_t = (A * rinv_sqrt[:, None]) * rinv_sqrt[None, :]   # R^-1/2 (H B H^T) R^-1/2
        A_t = 0.5 * (A_t + A_t.T)                             # exact symmetry for eigh
        Lam, U = np.linalg.eigh(A_t)
        G = (self.B[:, g] * rinv_sqrt[None, :]) @ U           # (D, m)
        return _SweepGain(gather=g, Lam=Lam, U=U, G=G, G2=G ** 2,
                          rinv_sqrt=rinv_sqrt, diagB=self.diagB,
                          b_scales=np.asarray(b_scales, dtype=np.float64))

    def apply_sweep(
        self, gain: _SweepGain, y_anom: np.ndarray, background_anom: np.ndarray
    ) -> list[AnalysisResult]:
        """Analysis at every ``b_scale`` for one innovation, one result per ``b_scale``."""
        x_b = np.asarray(background_anom, dtype=np.float64).ravel()
        d = np.asarray(y_anom, dtype=np.float64) - x_b[gain.gather]
        q = gain.U.T @ (gain.rinv_sqrt * d)                   # (m,)
        post_var = self.post_var_sweep(gain)
        out = []
        for ki, k in enumerate(gain.b_scales):
            x_a = x_b + k * (gain.G @ (q / (k * gain.Lam + 1.0)))
            out.append(AnalysisResult(mean_anom=x_a.reshape(self.shape),
                                      posterior_var=post_var[ki].reshape(self.shape)))
        return out

    def post_var_sweep(self, gain: _SweepGain) -> np.ndarray:
        """Flat posterior-variance map per ``b_scale``, ``(n_b_scale, D)``."""
        k = gain.b_scales[:, None]
        denom = k * gain.Lam[None, :] + 1.0                   # (n_k, m)
        return k * gain.diagB[None, :] - k ** 2 * ((1.0 / denom) @ gain.G2.T)
