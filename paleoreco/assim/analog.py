"""Choosing which prior states form an analog ensemble.

An analog ensemble is the subset of the prior archive that best matches the observations
of one assimilation, so the covariance built from it describes climates like the one being
reconstructed rather than the average of the whole run (Sun et al. 2022). Selection is the
only place the prior sees the data, and it is what makes the resulting covariance
flow-dependent.

Three rules live here, two of them published. The misfit rule ranks candidates by
R-weighted squared distance to the observations, which is Sun et al. (2025) Eq. 12 up to a
monotone transform; the same lineage ranks unweighted in Sun et al. (2022) Eq. 5 and Wu et
al. (2025) Eq. 5. Weighting by R is what lets a network of unequally trusted sites
contribute in proportion to what it knows, and it is what makes the score consistent with a
corrected observation pair, where dividing the observation by its attenuation and inflating
R by its square leaves the comparison between the raw observation and the attenuated
candidate.

The correlation rule is Sun et al. (2022) Eq. 6, carried into Sun et al. (2024) Eq. 3 as the
rule their hybrid gain filter selects by. It differs in kind rather than in weighting: it
removes the spatial mean and normalises amplitude, so it is blind to a uniform offset that
the misfit rule reads, and it discards R entirely. Both papers apply it to a single observed
variable, which leaves "the spatial average" ambiguous over an observation vector spanning
two channels, so the rule comes in a pooled transcription and a per-channel one.

The evidence rule ranks by the candidate's marginal likelihood ``N(y; H x_j, c H B H^T +
R)`` instead. A candidate is not the truth but the centre of a background with covariance
B, so the update that consumes the ensemble already carries that spread; measuring the
residual against R alone is the ``c = 0`` limit of the same score, which the misfit rule
reproduces exactly. Whitening by the prior predictive puts every direction on the scale the
prior says is normal for it and discounts observations the prior holds to be redundant with
each other. No published rule scores candidates this way.

Every rule scores a candidate on its own, so the best k can be near-duplicates: the archive
is a trajectory, and states close in time predict a sparse network almost identically. The
redundancy penalty makes the draw a set problem instead, charging a candidate for how much
it resembles one already taken, as a cosine in the same whitened coordinates the evidence
score uses. At zero penalty the greedy pass reproduces the ranking.

Every rule is deterministic to the tie, so an estimator built on them consumes no random
numbers and repeats exactly.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim.ensrf import WhitenedBlock

# How the k members of an analog ensemble are chosen. The two correlation entries are the
# same published rule under the two readings its spatial average admits here.
ANALOG_MISFIT = "misfit"
ANALOG_CORRELATION = "correlation"
ANALOG_CORRELATION_PERCHAN = "correlation_perchan"
ANALOG_EVIDENCE = "evidence"
ANALOG_RULES = (ANALOG_MISFIT, ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN,
                ANALOG_EVIDENCE)
# Amplitude of the background covariance in the evidence metric. Zero recovers the misfit
# rule. This is a tunable parameter held at the value that makes selection assume the same
# background the update applies; that is not where a sweep puts its optimum, so holding it
# here rather than tuning keeps the rule's reported margin conservative.
EVIDENCE_SCALE = 1.0


def eligible_mask(pool_ages: np.ndarray, age: float, exclude_yr: float) -> np.ndarray | None:
    """Candidates at least ``exclude_yr`` from ``age``, or ``None`` when nothing is excluded.

    Where the archive spans the age being reconstructed, the analog step can select the
    simulation's own state there and the method degenerates into a per-age background.
    Excluding a band around the target keeps it a climatological analog. ``exclude_yr = 0``
    excludes nothing, including the target age itself.
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

    A stable sort rather than a partial one: the archive is small enough that the full
    sort is free, and the ordering has to be reproducible for the estimator to be.
    """
    return np.argsort(_masked_score(score, k, eligible), kind="stable")[:k]


def _greedy_rank(score: np.ndarray, signatures: np.ndarray, k: int, redundancy: float,
                 eligible: np.ndarray | None) -> np.ndarray:
    """The ``k`` candidates minimising ``score`` plus a penalty for resembling those taken.

    ``signatures`` are unit vectors, one per candidate, whose inner product is the
    similarity charged for. The score is divided by its own eligible mean, so
    ``redundancy`` means the same thing whatever scale the network puts the score on.
    Ties break by index, as :func:`_rank` does, so the ranking stays reproducible.
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

    ``pool_at_obs`` is ``H`` applied to every candidate, ``(n_pool, m)``. The observation
    pair is taken as given, so whatever treatment produced ``y_anom`` and ``r_diag`` is the
    one selection scores; scoring a raw pair while the update consumes a corrected one is a
    disagreement about what the observations say that nothing else would surface.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    r = np.asarray(r_diag, dtype=np.float64)
    misfit = (((y[None, :] - np.asarray(pool_at_obs, dtype=np.float64)) ** 2) / r[None, :]).sum(axis=1)
    return _rank(misfit, k, eligible)


def _pearson(pool_at_obs: np.ndarray, y_anom: np.ndarray) -> np.ndarray:
    """Correlation of each candidate's predicted observations with ``y_anom``.

    A candidate whose predictions are constant across the network has no pattern to
    correlate, and neither does an observation vector that is; both score zero rather than
    a NaN so the ranking stays total and reproducible.
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

    Sun et al. (2022) Eq. 6, the rule Sun et al. (2024) Eq. 3 selects their analog ensemble
    by. Passing ``channel`` centres and normalises within each channel instead of over the
    whole vector, which is the same quantity as the mean of the per-channel correlations.
    The two agree where a network observes one variable, which is the case both papers
    treat; they diverge once a vector mixes channels of unequal spread, since the pooled
    form is then dominated by the wider one.
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

    ``whitened`` factorizes the static covariance against this network, so the score is
    ``d^T (scale H B H^T + R)^-1 d`` without a second decomposition; R enters through
    ``rinv_sqrt``, which is why no separate ``r_diag`` is taken. The log-determinant is the
    same for every candidate, so the Mahalanobis term alone orders them.

    ``redundancy`` charges a candidate for resembling one already taken, which makes the
    draw a set rather than a ranking; zero returns the ranking itself.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    d = y[None, :] - np.asarray(pool_at_obs, dtype=np.float64)          # (n_pool, m)
    q = (d * whitened.rinv_sqrt[None, :]) @ whitened.U
    weight = float(scale) * whitened.Lam + 1.0
    chi = (q ** 2 / weight[None, :]).sum(axis=1)
    if redundancy <= 0.0:
        return _rank(chi, k, eligible)
    # Each candidate's predicted observations, in the coordinates the score whitens by.
    # q already holds their residual against y, so subtracting it from the whitened
    # observations recovers them without a second product over the pool.
    z = ((y * whitened.rinv_sqrt) @ whitened.U)[None, :] - q
    z /= np.sqrt(weight)[None, :]
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    # A candidate the network cannot distinguish from the climatology has no direction to
    # be redundant in, so it scores zero similarity against everything rather than a NaN.
    z = np.divide(z, norm, out=np.zeros_like(z), where=norm > 0.0)
    return _greedy_rank(chi, z, k, float(redundancy), eligible)
