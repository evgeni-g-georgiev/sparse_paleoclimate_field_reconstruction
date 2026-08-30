"""Choosing which prior states form an analog ensemble.

An analog ensemble is the subset of the prior archive that best matches the observations
of one assimilation, so the covariance built from it describes climates like the one being
reconstructed rather than the average of the whole run (Sun et al. 2022). Selection is the
only place the prior sees the data, and it is what makes the resulting covariance
flow-dependent.

Three rules live here. The misfit rule ranks candidates by R-weighted squared distance to
the observations, which is the likelihood of a candidate being the truth. The same lineage
ranks three ways: Sun et al. (2024) Eq. 3 by spatial-pattern correlation, Wu et al. (2025)
Eq. 5 by plain RMSE, and Sun et al. (2025) Eq. 12 by RMSE normalised by the observation
error variance, which is this rule up to a monotone transform. Weighting by R is what lets a
network of unequally trusted sites contribute in proportion to what it knows, and it is what
makes the score consistent with a corrected observation pair, where dividing the observation
by its attenuation and inflating R by its square leaves the comparison between the raw
observation and the attenuated candidate. Eq. 3 differs in kind rather than in weighting: it
removes the spatial mean and normalises amplitude, so it is blind to a uniform offset that
the misfit rule reads.

The evidence rule ranks by the candidate's marginal likelihood ``N(y; H x_j, c H B H^T +
R)`` instead. A candidate is not the truth but the centre of a background with covariance
B, so the update that consumes the ensemble already carries that spread; measuring the
residual against R alone is the ``c = 0`` limit of the same score, which the misfit rule
reproduces exactly. Whitening by the prior predictive puts every direction on the scale the
prior says is normal for it and discounts observations the prior holds to be redundant with
each other.

The window rule ranks by nearness in time instead, which is how a running-window prior is
built (Osman et al. 2021; Erb et al. 2022). Window against the two data-driven rules
separates whether what matters is the epoch or the climate regime.

Every rule is deterministic to the tie, so an estimator built on them consumes no random
numbers and repeats exactly.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim.ensrf import WhitenedBlock

# How the k members of an analog ensemble are chosen.
ANALOG_MISFIT = "misfit"
ANALOG_WINDOW = "window"
ANALOG_EVIDENCE = "evidence"
ANALOG_RULES = (ANALOG_MISFIT, ANALOG_WINDOW, ANALOG_EVIDENCE)
# Amplitude of the background covariance in the evidence metric. Zero recovers the misfit
# rule, so this states which error model selection assumes rather than tuning a knob.
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


def _rank(score: np.ndarray, k: int, eligible: np.ndarray | None) -> np.ndarray:
    """The ``k`` eligible candidates with the smallest score, ties broken by index.

    A stable sort rather than a partial one: the archive is small enough that the full
    sort is free, and the ordering has to be reproducible for the estimator to be.
    """
    s = np.asarray(score, dtype=np.float64)
    if eligible is not None:
        n_ok = int(np.count_nonzero(eligible))
        if n_ok < k:
            raise ValueError(f"only {n_ok} eligible candidates for an ensemble of {k}")
        s = np.where(eligible, s, np.inf)
    elif len(s) < k:
        raise ValueError(f"only {len(s)} candidates for an ensemble of {k}")
    return np.argsort(s, kind="stable")[:k]


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


def evidence_indices(
    pool_at_obs: np.ndarray, y_anom: np.ndarray, whitened: WhitenedBlock, k: int, *,
    eligible: np.ndarray | None = None, scale: float = EVIDENCE_SCALE,
) -> np.ndarray:
    """Indices of the ``k`` prior states whose marginal likelihood best explains ``y_anom``.

    ``whitened`` factorizes the static covariance against this network, so the score is
    ``d^T (scale H B H^T + R)^-1 d`` without a second decomposition; R enters through
    ``rinv_sqrt``, which is why no separate ``r_diag`` is taken. The log-determinant is the
    same for every candidate, so the Mahalanobis term alone orders them.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    d = y[None, :] - np.asarray(pool_at_obs, dtype=np.float64)          # (n_pool, m)
    q = (d * whitened.rinv_sqrt[None, :]) @ whitened.U
    chi = (q ** 2 / (float(scale) * whitened.Lam + 1.0)[None, :]).sum(axis=1)
    return _rank(chi, k, eligible)


def window_indices(
    pool_ages: np.ndarray, age: float, k: int, *, eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of the ``k`` prior states nearest ``age`` in time."""
    return _rank(np.abs(np.asarray(pool_ages, dtype=np.float64) - float(age)), k, eligible)
