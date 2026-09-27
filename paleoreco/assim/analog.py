"""Rules for choosing which prior states form an analog ensemble.

An analog ensemble is the subset of the prior archive that best matches one assimilation's
observations, so its covariance describes climates like the one being reconstructed
(Sun et al. 2022). Three rules are implemented:

* misfit: R-weighted squared distance to the observations, Sun et al. (2025) Eq. 12 up to a
  monotone transform.
* correlation: spatial pattern correlation, Sun et al. (2022) Eq. 6 and Sun et al. (2024)
  Eq. 3. Both papers observe one variable, so the two-channel case has a pooled and a
  per-channel reading.
* evidence: the candidate's marginal likelihood ``N(y; H x_j, c H B H^T + R)``, which
  reduces to the misfit rule at ``c = 0``.

An optional redundancy penalty turns the evidence ranking into a greedy set selection that
charges a candidate for resembling one already taken; at zero it returns the ranking. Every
rule breaks ties by index, so selection is deterministic.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim.ensrf import WhitenedBlock

# The two correlation entries are the pooled and per-channel readings of one rule.
ANALOG_MISFIT = "misfit"
ANALOG_CORRELATION = "correlation"
ANALOG_CORRELATION_PERCHAN = "correlation_perchan"
ANALOG_EVIDENCE = "evidence"
ANALOG_RULES = (ANALOG_MISFIT, ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN,
                ANALOG_EVIDENCE)
# Background amplitude in the evidence score, held at the value the update applies rather
# than tuned. Zero recovers the misfit rule.
EVIDENCE_SCALE = 1.0


def eligible_mask(pool_ages: np.ndarray, age: float, exclude_yr: float) -> np.ndarray | None:
    """Candidates at least ``exclude_yr`` from ``age``, or ``None`` when nothing is excluded.

    Where the archive spans the target age, this stops selection from picking the
    simulation's own state there. ``exclude_yr = 0`` excludes nothing.
    """
    if exclude_yr <= 0.0:
        return None
    return np.abs(np.asarray(pool_ages, dtype=np.float64) - float(age)) >= float(exclude_yr)


def _masked_score(score: np.ndarray, k: int, eligible: np.ndarray | None) -> np.ndarray:
    """``score`` with ineligible candidates at infinity, once there are enough to draw."""
    s = np.asarray(score, dtype=np.float64)
    if eligible is not None:
        n_ok = int(np.count_nonzero(eligible))
        if n_ok < k:
            raise ValueError(f"only {n_ok} eligible candidates for an ensemble of {k}")
        return np.where(eligible, s, np.inf)
    if len(s) < k:
        raise ValueError(f"only {len(s)} candidates for an ensemble of {k}")
    return s


def _rank(score: np.ndarray, k: int, eligible: np.ndarray | None) -> np.ndarray:
    """The ``k`` eligible candidates with the smallest score, ties broken by index.

    A full stable sort keeps the ordering reproducible; the archive is small enough for it.
    """
    return np.argsort(_masked_score(score, k, eligible), kind="stable")[:k]


def _greedy_rank(score: np.ndarray, signatures: np.ndarray, k: int, redundancy: float,
                 eligible: np.ndarray | None) -> np.ndarray:
    """The ``k`` candidates minimising ``score`` plus a penalty for resembling those taken.

    ``signatures`` are unit vectors whose inner product is the similarity charged for. The
    score is divided by its eligible mean so ``redundancy`` does not depend on its scale.
    """
    s = _masked_score(score, k, eligible)
    finite = np.isfinite(s)
    s = s / s[finite].mean()
    penalty = np.zeros(len(s))
    chosen = np.empty(k, dtype=np.int64)
    for i in range(k):
        adjusted = s + redundancy * penalty
        adjusted[chosen[:i]] = np.inf
        j = int(np.argmin(adjusted))
        chosen[i] = j
        penalty = np.maximum(penalty, np.abs(signatures @ signatures[j]))
    return chosen


def analog_indices(
    pool_at_obs: np.ndarray, y_anom: np.ndarray, r_diag: np.ndarray, k: int, *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of the ``k`` prior states closest to the observations in R-weighted misfit.

    ``pool_at_obs`` is ``H`` applied to every candidate, ``(n_pool, m)``. Pass the same
    corrected ``y_anom`` and ``r_diag`` the update uses, so selection and update agree.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    r = np.asarray(r_diag, dtype=np.float64)
    misfit = (((y[None, :] - np.asarray(pool_at_obs, dtype=np.float64)) ** 2) / r[None, :]).sum(axis=1)
    return _rank(misfit, k, eligible)


def _pearson(pool_at_obs: np.ndarray, y_anom: np.ndarray) -> np.ndarray:
    """Correlation of each candidate's predicted observations with ``y_anom``.

    A constant candidate or observation vector scores zero rather than NaN.
    """
    a = pool_at_obs - pool_at_obs.mean(axis=1, keepdims=True)
    b = y_anom - y_anom.mean()
    denom = np.sqrt((a ** 2).sum(axis=1)) * np.sqrt((b ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((a @ b) / denom, nan=0.0, posinf=0.0, neginf=0.0)


def correlation_indices(
    pool_at_obs: np.ndarray, y_anom: np.ndarray, k: int, *,
    channel: np.ndarray | None = None, eligible: np.ndarray | None = None,
) -> np.ndarray:
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
    return _rank(-corr, k, eligible)


def evidence_indices(
    pool_at_obs: np.ndarray, y_anom: np.ndarray, whitened: WhitenedBlock, k: int, *,
    eligible: np.ndarray | None = None, scale: float = EVIDENCE_SCALE,
    redundancy: float = 0.0,
) -> np.ndarray:
    """Indices of the ``k`` prior states whose marginal likelihood best explains ``y_anom``.

    ``whitened`` is the static covariance already factorized against this network, so the
    score ``d^T (scale H B H^T + R)^-1 d`` needs no second decomposition. The
    log-determinant is the same for every candidate and is dropped. ``redundancy > 0``
    selects greedily with the penalty described in the module docstring.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    d = y[None, :] - np.asarray(pool_at_obs, dtype=np.float64)          # (n_pool, m)
    q = (d * whitened.rinv_sqrt[None, :]) @ whitened.U
    weight = float(scale) * whitened.Lam + 1.0
    chi = (q ** 2 / weight[None, :]).sum(axis=1)
    if redundancy <= 0.0:
        return _rank(chi, k, eligible)
    # Each candidate's whitened predicted observations, recovered from q without a second
    # product over the pool.
    z = ((y * whitened.rinv_sqrt) @ whitened.U)[None, :] - q
    z /= np.sqrt(weight)[None, :]
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    # A zero vector has no direction, so it scores zero similarity rather than NaN.
    z = np.divide(z, norm, out=np.zeros_like(z), where=norm > 0.0)
    return _greedy_rank(chi, z, k, float(redundancy), eligible)
