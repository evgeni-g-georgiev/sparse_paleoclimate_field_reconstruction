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

for background amplitude k. Nothing here is paleoclimate-specific: the caller supplies
``P H^T`` and ``H P H^T`` already tapered, so a static covariance and an ensemble-sampled
one are handled identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WhitenedBlock:
    """One covariance's gain factors against one network, reusable across amplitudes.

    ``G`` is ``P H^T R^-1/2 U``, the only ``(D, m)`` object either gain needs; ``Lam`` and
    ``U`` diagonalize the whitened observation block, and ``rinv_sqrt`` is R^-1/2 at the
    observation cells. All four are free of the observation values, so one factorization
    serves every innovation and every background amplitude.
    """

    G: np.ndarray
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
    G = (np.asarray(P_obs, dtype=np.float64) * rinv_sqrt[None, :]) @ U
    return WhitenedBlock(G=G, Lam=Lam, U=U, rinv_sqrt=rinv_sqrt)


def mean_gain_apply(wb: WhitenedBlock, b_scale: float, d: np.ndarray) -> np.ndarray:
    """``k P H^T (k H P H^T + R)^-1 d`` for amplitude ``k``, as a ``(D,)`` increment."""
    q = wb.U.T @ (wb.rinv_sqrt * np.asarray(d, dtype=np.float64))
    return b_scale * (wb.G @ (q / (b_scale * wb.Lam + 1.0)))


def sqrt_gain_apply(wb: WhitenedBlock, b_scale: float, h_dev: np.ndarray) -> np.ndarray:
    """Reduced-gain increment for ensemble deviations, ``(D, n_members)``.

    ``h_dev`` is ``H X'`` ``(m, n_members)``. Subtracting the result from ``X'`` is the
    deviation half of the square-root update.
    """
    q = wb.U.T @ (wb.rinv_sqrt[:, None] * np.asarray(h_dev, dtype=np.float64))
    w = b_scale * wb.Lam + 1.0
    return b_scale * (wb.G @ (q / (w + np.sqrt(w))[:, None]))
