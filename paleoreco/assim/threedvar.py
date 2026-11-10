"""3DVar analysis in the gain (dual) form.

With a linear observation operator (nearest-cell selection) the 3DVar cost has a
closed-form minimiser, so the analysis is computed directly as

    x_a = x_b + k B H^T (k H B H^T + R)^-1 (y - H x_b)

for background amplitude k, without ever inverting the dense D x D background covariance:
one eigendecomposition of the whitened m x m observation block serves every k (see
:mod:`paleoreco.assim.ensrf`). The same factor gives the posterior covariance
``A = (I - K H) k B``, whose diagonal is the posterior variance map. A depends on B and
the gain but not on x_b or y.

Everything is in anomaly space; H is plain selection, so ``H B H^T`` is the obs-cell
submatrix of B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paleoreco.assim.ensrf import WhitenedBlock, mean_gain_apply, whitened_block
from paleoreco.assim.method import AnalysisResult, Method


@dataclass(frozen=True)
class _SweepGain:
    """Per-network operators shared across a background-amplitude sweep."""

    gather: np.ndarray
    block: WhitenedBlock
    b_scales: np.ndarray


class ThreeDVar(Method):
    """Gain-form 3DVar over a fixed background covariance."""

    def __init__(self, B: np.ndarray, shape: tuple[int, int, int]):
        # Column-major: every gain reads the obs columns ``B[:, gather]``, which is a
        # strided walk over the whole D x D array in row-major order and dominates the
        # per-network cost. Storing B column-major makes that gather contiguous.
        self.B = np.asfortranarray(B, dtype=np.float64)
        self.diagB = np.diag(self.B).copy()
        self.shape = shape

    def prepare_sweep(self, gather: np.ndarray, r_diag: np.ndarray,
                      b_scales: np.ndarray) -> _SweepGain:
        """Factorize the gain once for a network, reusable across all ``b_scales``.

        ``r_diag`` is the observation-error variance (R's diagonal); scaling B by
        ``b_scale`` does not touch it.
        """
        g = np.asarray(gather)
        return _SweepGain(gather=g, block=whitened_block(self.B[:, g], self.B[np.ix_(g, g)],
                                                         r_diag),
                          b_scales=np.asarray(b_scales, dtype=np.float64))

    def apply_sweep(self, gain: _SweepGain, y_anom: np.ndarray, background_anom: np.ndarray,
                    *, mean_only: bool = False) -> list[AnalysisResult]:
        """Analysis at every ``b_scale`` for one innovation, one result per ``b_scale``."""
        x_b = np.asarray(background_anom, dtype=np.float64).ravel()
        d = np.asarray(y_anom, dtype=np.float64) - x_b[gain.gather]
        post_var = None if mean_only else self.post_var_sweep(gain)
        out = []
        for bj, b in enumerate(gain.b_scales):
            x_a = x_b + mean_gain_apply(gain.block, b, d)
            out.append(AnalysisResult(
                mean_anom=x_a.reshape(self.shape),
                posterior_var=None if mean_only else post_var[bj].reshape(self.shape)))
        return out

    def post_var_sweep(self, gain: _SweepGain) -> np.ndarray:
        """Flat posterior-variance map per ``b_scale``, ``(n_b_scale, D)``.

        ``diag(K H k B) = k^2 sum_j G_ij^2 / (k Lam_j + 1)`` with ``G = B H^T R^-1/2 U``.
        """
        wb = gain.block
        G2 = ((wb.P_obs * wb.rinv_sqrt[None, :]) @ wb.U) ** 2
        k = gain.b_scales[:, None]
        return k * self.diagB[None, :] - k ** 2 * ((1.0 / (k * wb.Lam[None, :] + 1.0)) @ G2.T)
