"""Ensemble square-root gains for one covariance against one observation network.

The ensemble square-root filter (Whitaker and Hamill 2002) updates the ensemble mean by
the Kalman gain K and the ensemble deviations by a reduced gain, so the updated deviations
carry the posterior covariance ``(I - K H) P`` rather than the too-small spread applying K
to them would give.

Both gains come from one eigendecomposition. Writing everything in whitened observation
coordinates, where R is the identity and the symmetric square root of ``H P H^T + R`` is
unambiguous, the two gains differ only in how they reweight the eigenvalues, and a
background-amplitude sweep is another reweighting rather than a fresh solve:

    R^-1/2 (H P H^T) R^-1/2 = U Lam U^T
    mean gain weights          1 / (k Lam + 1)
    square-root gain weights   1 / (k Lam + 1 + sqrt(k Lam + 1))

for background amplitude k. Each gain is applied as ``P H^T`` times an ``m``-row factor,
so the only ``(D, m)`` product is the final one. Nothing here is paleoclimate-specific:
the caller supplies ``P H^T`` and ``H P H^T`` already tapered, so a static covariance and
an ensemble-sampled one are handled identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WhitenedBlock:
    """One covariance's gain factors against one network, reusable across amplitudes.

    ``P_obs`` is ``P H^T``; ``Lam`` and ``U`` diagonalize the whitened observation block,
    and ``rinv_sqrt`` is R^-1/2 at the observation cells. All four are free of the
    observation values, so one factorization serves every innovation and amplitude.
    """

    P_obs: np.ndarray
    Lam: np.ndarray
    U: np.ndarray
    rinv_sqrt: np.ndarray


def whitened_block(P_obs: np.ndarray, S_obs: np.ndarray, r_diag: np.ndarray) -> WhitenedBlock:
    """Factorize a covariance against a network from its two observation blocks.

    ``P_obs`` is ``P H^T`` ``(D, m)`` and ``S_obs`` is ``H P H^T`` ``(m, m)``, both already
    tapered; ``r_diag`` is R's diagonal.
    """
    r = np.asarray(r_diag, dtype=np.float64)
    rinv_sqrt = 1.0 / np.sqrt(r)
    A = (np.asarray(S_obs, dtype=np.float64) * rinv_sqrt[:, None]) * rinv_sqrt[None, :]
    A = 0.5 * (A + A.T)                                   # exact symmetry for eigh
    Lam, U = np.linalg.eigh(A)
    return WhitenedBlock(P_obs=np.asarray(P_obs, dtype=np.float64), Lam=Lam, U=U,
                         rinv_sqrt=rinv_sqrt)


def _apply(wb: WhitenedBlock, b_scale: float, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``k P H^T R^-1/2 U diag(weight) U^T R^-1/2 x`` for a vector or a column stack."""
    rinv = wb.rinv_sqrt if x.ndim == 1 else wb.rinv_sqrt[:, None]
    w = weight if x.ndim == 1 else weight[:, None]
    q = wb.U.T @ (rinv * x)
    return b_scale * (wb.P_obs @ (rinv * (wb.U @ (q * w))))


def mean_gain_apply(wb: WhitenedBlock, b_scale: float, d: np.ndarray) -> np.ndarray:
    """``k P H^T (k H P H^T + R)^-1 d`` for amplitude ``k``, as a ``(D,)`` increment."""
    return _apply(wb, b_scale, np.asarray(d, dtype=np.float64), 1.0 / (b_scale * wb.Lam + 1.0))


def sqrt_gain_apply(wb: WhitenedBlock, b_scale: float, h_dev: np.ndarray) -> np.ndarray:
    """Reduced-gain increment for ensemble deviations, ``(D, n_members)``.

    ``h_dev`` is ``H X'`` ``(m, n_members)``. Subtracting the result from ``X'`` is the
    deviation half of the square-root update.
    """
    w = b_scale * wb.Lam + 1.0
    return _apply(wb, b_scale, np.asarray(h_dev, dtype=np.float64), 1.0 / (w + np.sqrt(w)))
