"""Background state and background-error covariance from the Prior cube.

The state vector is one snapshot ``cube[k]`` of shape ``(2, n_lat, n_lon)``
flattened in C order to length ``D = 2 * n_lat * n_lon``; channel ``mtco`` fills
the first ``n_lat * n_lon`` entries, ``mtwa`` the rest. B is the sample
covariance of the Prior's own anomalies, so the background inherits the model's
spatial covariance structure.

Optionally that covariance is taken over the band-passed states rather than the raw
ones, which drops the slow drift specific to the window the prior was cut from and
leaves the variability that transfers to the rest of the record.
"""

from __future__ import annotations

import numpy as np

from paleoreco.data.cube import highpass_states


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


def background_covariance(
    cube: np.ndarray,
    age_indices: np.ndarray,
    *,
    ages: np.ndarray | None = None,
    highpass_window: float | None = None,
) -> np.ndarray:
    """Full ``(D, D)`` sample covariance of Prior anomalies over the given ages.

    Computed in float64 so the diagonal matches per-cell variance to tight
    tolerance. Rank is bounded by ``len(age_indices) - 1``.

    ``highpass_window`` removes a running mean along the age axis first
    (:func:`~paleoreco.data.cube.highpass_states`), so B describes the fast component
    of the prior rather than the slow drift of the window the states came from. It
    needs ``ages`` to know where each state sits, and it is applied to the selected
    states alone, so a prior half never borrows the running mean of a held-out one.
    A constant per-cell climatology cancels in the running-mean difference, so the
    filter gives the same answer on the raw cube as on its anomalies.
    """
    idx = np.asarray(age_indices, dtype=np.int64)
    X = cube[idx].reshape(len(idx), -1).astype(np.float64)
    if highpass_window is not None:
        if ages is None:
            raise ValueError("highpass_window needs ages to place each state on the "
                             "age axis; pass ages=... alongside it")
        X = highpass_states(X, np.asarray(ages)[idx], highpass_window)
    X = X - X.mean(axis=0, keepdims=True)
    return (X.T @ X) / (len(idx) - 1)


def background_variance(cov: np.ndarray) -> np.ndarray:
    """Per-cell background variance, the diagonal of B the marginal test needs."""
    return np.diag(cov).copy()
