"""Calibration metrics: whether a posterior's stated uncertainty matches its errors.

Skill metrics (:mod:`paleoreco.eval.da`) ask how wrong a reconstruction is; these ask
whether it knew. Each function takes flat, aligned 1-D arrays and accepts either a
Gaussian posterior (``mean``/``var``) or an ensemble (``samples``), so a variational
method and a generative one score identically.

The variance passed in must match what the truth is: scoring against a *noisy
observation* requires the observation error to be added to the posterior variance
(``posterior_var + sse``), because the residual carries both; scoring against a
*noise-free truth* requires the posterior variance alone. Adding it in the wrong place
manufactures over- or under-confidence that is not in the analysis.

CRPS generalises absolute error to a distribution and collapses to it as the variance
goes to zero, so a sharp forecast is only rewarded when it is also right. CRPSS turns
that into a skill score against a reference, the role climatology plays for CE.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _standardise(truth: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Standardised residual ``(truth - mean) / sd``, N(0,1) under a calibrated posterior."""
    sd = np.sqrt(np.asarray(var, dtype=np.float64))
    return (np.asarray(truth, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / sd


# ---------------------------------------------------------------------------
# Continuous ranked probability score.
# ---------------------------------------------------------------------------
def crps_gaussian(truth: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-point CRPS of a Gaussian posterior, in the units of ``truth``.

    Closed form (Gneiting & Raftery 2007): with ``z`` the standardised residual,
    ``sd * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))``. At ``var = 0`` this is ``|truth -
    mean|``, so CRPS and absolute error are on one scale.
    """
    sd = np.sqrt(np.asarray(var, dtype=np.float64))
    z = _standardise(truth, mean, var)
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - _INV_SQRT_PI)


def crps_ensemble(truth: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Per-point CRPS of an ensemble ``(n_members, n_points)``.

    Energy form: member accuracy against the truth, discounted by the spread the members
    already spend among themselves. The pairwise term uses the ``1 / (n (n - 1))``
    normalisation (the fair estimator of Ferro 2007), which removes the small-ensemble
    bias that would otherwise reward drawing fewer members.
    """
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    n = x.shape[0]
    accuracy = np.abs(x - y[None, :]).mean(axis=0)
    if n < 2:
        return accuracy
    pairwise = np.abs(x[:, None, :] - x[None, :, :]).sum(axis=(0, 1))
    return accuracy - pairwise / (2.0 * n * (n - 1))


def crpss(crps_model: np.ndarray, crps_ref: np.ndarray) -> float:
    """CRPS skill score against a reference: 1 perfect, 0 no better than the reference."""
    ref = float(np.mean(crps_ref))
    if ref == 0.0:
        return float("nan")
    return float(1.0 - np.mean(crps_model) / ref)


# ---------------------------------------------------------------------------
# Reliability of the stated spread.
# ---------------------------------------------------------------------------
def rcrv(truth: np.ndarray, mean: np.ndarray, var: np.ndarray) -> tuple[float, float]:
    """Reduced centred random variable: ``(bias, dispersion)`` of the standardised residual.

    Bias 0 and dispersion 1 describe an honest posterior. Dispersion above 1 means the
    errors are larger than the stated uncertainty (overconfident), below 1 too cautious.
    Standardising per point before pooling is what makes this valid when the uncertainty
    varies across the field, which it does sharply between observed cells and voids.
    """
    z = _standardise(truth, mean, var)
    return float(np.mean(z)), float(np.std(z))


def coverage(truth: np.ndarray, mean: np.ndarray, var: np.ndarray,
             level: float = 0.9) -> float:
    """Fraction of truths inside the central ``level`` predictive interval.

    Should equal ``level``; below it the intervals are too narrow.
    """
    z_crit = norm.ppf(0.5 * (1.0 + level))
    return float(np.mean(np.abs(_standardise(truth, mean, var)) <= z_crit))
