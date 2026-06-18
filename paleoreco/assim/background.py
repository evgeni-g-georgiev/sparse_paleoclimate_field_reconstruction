"""Background state and background-error covariance from the Prior cube.

The state vector is one snapshot ``cube[k]`` of shape ``(2, n_lat, n_lon)``
flattened in C order to length ``D = 2 * n_lat * n_lon``; channel ``mtco`` fills
the first ``n_lat * n_lon`` entries, ``mtwa`` the rest. B is the sample
covariance of the Prior's own anomalies, so the background inherits the model's
spatial covariance structure.
"""

from __future__ import annotations

import numpy as np


def background_state(
    cube: np.ndarray,
    mean: np.ndarray,
    age_index: int,
    kind: str,
) -> np.ndarray:
    """Flattened background state for one age.

    ``per_age`` returns the Prior snapshot at ``age_index``; ``climatological``
    returns the per-cell time mean (constant across ages, identically zero once
    anomaly-scored).
    """
    if kind == "per_age":
        return cube[age_index].ravel().astype(float)
    if kind == "climatological":
        return mean.ravel().astype(float)
    raise ValueError(f"unknown background kind {kind!r}; expected 'per_age' or 'climatological'")


def background_covariance(cube: np.ndarray, age_indices: np.ndarray) -> np.ndarray:
    """Full ``(D, D)`` sample covariance of Prior anomalies over the given ages.

    Computed in float64 so the diagonal matches per-cell variance to tight
    tolerance. Rank is bounded by ``len(age_indices) - 1``.
    """
    X = cube[age_indices].reshape(len(age_indices), -1).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    return (X.T @ X) / (len(age_indices) - 1)


def background_variance(cov: np.ndarray) -> np.ndarray:
    """Per-cell background variance, the diagonal of B the marginal test needs."""
    return np.diag(cov).copy()
