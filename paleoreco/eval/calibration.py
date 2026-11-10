"""Calibration metrics: whether a posterior's stated uncertainty matches its errors.

Skill metrics (:mod:`paleoreco.eval.da`) ask how wrong a reconstruction is; these ask
whether it knew. Each function takes flat, aligned 1-D arrays and a Gaussian posterior
``mean``/``var``.

The variance passed in must match what the truth is: scoring against a *noisy
observation* requires the observation error to be added to the posterior variance,
because the residual carries both; scoring against a *noise-free truth* requires the
posterior variance alone. Adding it in the wrong place manufactures over- or
under-confidence that is not in the analysis.

CRPS generalises absolute error to a distribution and collapses to it as the variance
goes to zero, so a sharp forecast is only rewarded when it is also right.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _standardise(truth: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Standardised residual ``(truth - mean) / sd``, N(0,1) under a calibrated posterior."""
    sd = np.sqrt(np.asarray(var, dtype=np.float64))
    return (np.asarray(truth, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / sd


def crps_gaussian(truth: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-point CRPS of a Gaussian posterior, in the units of ``truth``.

    Closed form (Gneiting & Raftery 2007): with ``z`` the standardised residual,
    ``sd * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))``. At ``var = 0`` this is ``|truth -
    mean|``, so CRPS and absolute error are on one scale.
    """
    sd = np.sqrt(np.asarray(var, dtype=np.float64))
    z = _standardise(truth, mean, var)
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - _INV_SQRT_PI)


def rcrv_dispersion(truth: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Dispersion of the reduced centred random variable: the std of the standardised residual.

    1 describes an honest posterior. Above 1 the errors are larger than the stated
    uncertainty (overconfident), below 1 too cautious. Standardising per point before
    pooling is what makes this valid when the uncertainty varies across the field, which
    it does sharply between observed cells and voids.
    """
    return float(np.std(_standardise(truth, mean, var)))
