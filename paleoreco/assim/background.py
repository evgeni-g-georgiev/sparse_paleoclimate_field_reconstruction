"""Background state, error covariance, and temporal structure from the Prior cube.

The state vector is one snapshot ``cube[k]`` of shape ``(2, n_lat, n_lon)``
flattened in C order to length ``D = 2 * n_lat * n_lon``; channel ``mtco`` fills
the first ``n_lat * n_lon`` entries, ``mtwa`` the rest. B is the sample
covariance of the Prior's own anomalies, so the background inherits the model's
spatial covariance structure. The structure function is the same statistic along
the age axis instead of the spatial one.
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


def temporal_structure_function(
    cube: np.ndarray, age_indices: np.ndarray, *, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """``S[lag, cell]`` and per-cell variance over the given ages.

    ``S`` is the mean squared change across ``lag`` age steps, so ``1 - S / (2 var)``
    is the cell's lag correlation: how much of a state survives that many years. An
    observation whose sample sits away from the analysis age reports on a state that
    far off, and that correlation is what says how much it still knows.

    Both come from the same ages so the ratio is consistent, and the variance uses
    ``ddof=1`` to match :func:`background_covariance`'s diagonal. Pass the same ages
    the background was built from: on a lane whose truth is a model state, later ages
    would leak that truth into the operator that reconstructs it.
    """
    idx = np.asarray(age_indices, dtype=np.int64)
    X = cube[idx].reshape(len(idx), -1).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    var = X.var(axis=0, ddof=1)

    n_lag = max(min(int(max_lag), len(idx) - 1), 0)
    S = np.zeros((n_lag + 1, X.shape[1]))
    for lag in range(1, n_lag + 1):
        S[lag] = ((X[lag:] - X[:-lag]) ** 2).mean(axis=0)
    return S, var
