"""Rules for choosing which prior states form an analog ensemble.

An analog ensemble is the subset of the prior archive that best matches one assimilation's
observations, so its covariance describes climates like the one being reconstructed
(Sun et al. 2022). Two rules are implemented:

* evidence: the candidate's marginal likelihood ``N(y; H x_j, kappa H B H^T + R)``. At
  ``kappa = 0`` it is the R-weighted squared distance to the observations, the misfit rule
  of Sun et al. (2025) Eq. 12 up to a monotone transform.
* correlation: spatial pattern correlation, Sun et al. (2022) Eq. 6 and Sun et al. (2024)
  Eq. 3. Both papers observe one variable, so the two-channel case has a pooled and a
  per-channel reading.

Every rule breaks ties by index, so selection is deterministic.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim.ensrf import WhitenedBlock

# The two correlation entries are the pooled and per-channel readings of one rule.
ANALOG_EVIDENCE = "evidence"
ANALOG_CORRELATION = "correlation"
ANALOG_CORRELATION_PERCHAN = "correlation_perchan"
ANALOG_RULES = (ANALOG_EVIDENCE, ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN)


def _rank(score: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` candidates with the smallest score, ties broken by index.

    A full stable sort keeps the ordering reproducible; the archive is small enough for it.
    """
    if len(score) < k:
        raise ValueError(f"only {len(score)} candidates for an ensemble of {k}")
    return np.argsort(np.asarray(score, dtype=np.float64), kind="stable")[:k]


def misfit_indices(pool_at_obs: np.ndarray, y_anom: np.ndarray, r_diag: np.ndarray,
                   k: int) -> np.ndarray:
    """Indices of the ``k`` prior states closest to the observations in R-weighted misfit.

    ``pool_at_obs`` is ``H`` applied to every candidate, ``(n_pool, m)``. Pass the same
    corrected ``y_anom`` and ``r_diag`` the update uses, so selection and update agree.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    r = np.asarray(r_diag, dtype=np.float64)
    misfit = (((y[None, :] - np.asarray(pool_at_obs, dtype=np.float64)) ** 2)
              / r[None, :]).sum(axis=1)
    return _rank(misfit, k)


def evidence_indices(pool_at_obs: np.ndarray, y_anom: np.ndarray, r_diag: np.ndarray,
                     whitened: WhitenedBlock, k: int, *, kappa: float) -> np.ndarray:
    """Indices of the ``k`` prior states whose marginal likelihood best explains ``y_anom``.

    ``whitened`` is the static covariance already factorized against this network, so the
    score ``d^T (kappa H B H^T + R)^-1 d`` needs no second decomposition. The
    log-determinant is the same for every candidate and is dropped. ``kappa = 0`` is
    computed as the misfit rule itself, so the two select identically.
    """
    if kappa == 0.0:
        return misfit_indices(pool_at_obs, y_anom, r_diag, k)
    d = np.asarray(y_anom, dtype=np.float64)[None, :] - np.asarray(pool_at_obs, dtype=np.float64)
    q = (d * whitened.rinv_sqrt[None, :]) @ whitened.U
    return _rank((q ** 2 / (float(kappa) * whitened.Lam + 1.0)[None, :]).sum(axis=1), k)


def _pearson(pool_at_obs: np.ndarray, y_anom: np.ndarray) -> np.ndarray:
    """Correlation of each candidate's predicted observations with ``y_anom``.

    A constant candidate or observation vector scores zero rather than NaN.
    """
    a = pool_at_obs - pool_at_obs.mean(axis=1, keepdims=True)
    b = y_anom - y_anom.mean()
    denom = np.sqrt((a ** 2).sum(axis=1)) * np.sqrt((b ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((a @ b) / denom, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_indices(pool_at_obs: np.ndarray, y_anom: np.ndarray, k: int, *,
                        channel: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``k`` prior states correlating best with the observations.

    Passing ``channel`` averages the per-channel correlations instead of pooling the
    vector, where the channel with the wider spread would otherwise dominate.
    """
    p = np.asarray(pool_at_obs, dtype=np.float64)
    y = np.asarray(y_anom, dtype=np.float64)
    if channel is None:
        corr = _pearson(p, y)
    else:
        ch = np.asarray(channel)
        # A correlation needs two points, so a channel carrying fewer contributes nothing.
        groups = [ch == c for c in np.unique(ch) if np.count_nonzero(ch == c) >= 2]
        corr = (np.mean([_pearson(p[:, g], y[g]) for g in groups], axis=0)
                if groups else np.zeros(len(p)))
    return _rank(-corr, k)
