"""Common contract every reconstruction method implements.

A method factorizes its gain once per observation network over a sweep of background
amplitudes, then analyses one innovation against it, returning one
:class:`AnalysisResult` per amplitude. A scan that only reads the posterior mean asks
for ``mean_only`` and skips the spread, which is most of the cost of an ensemble update.

All fields are in anomaly space: state minus per-cell climatology, observations minus
their per-site climatology.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from paleoreco.assim.priors import Prior


@dataclass(frozen=True)
class AnalysisResult:
    """Posterior of one assimilation, in anomaly space.

    ``mean_anom`` is ``(2, n_lat, n_lon)``; ``posterior_var`` (same shape) is ``None`` for a
    mean-only analysis. ``posterior_cross_var`` is ``(n_lat, n_lon)``, the posterior
    covariance between the two channels at each cell, filled by the methods that sample
    it: the spread of a seasonal contrast is a difference of correlated channels, which
    per-channel variances alone cannot give.
    """

    mean_anom: np.ndarray
    posterior_var: np.ndarray | None = None
    posterior_cross_var: np.ndarray | None = None

    def predict_obs(self, gather: np.ndarray) -> np.ndarray:
        """Posterior-mean anomaly at the given flat cell indices (H applied)."""
        return self.mean_anom.ravel()[gather]

    def predict_obs_var(self, gather: np.ndarray) -> np.ndarray:
        """Posterior variance at the given flat cell indices."""
        return self.posterior_var.ravel()[gather]


class Method(ABC):
    """A reconstruction method: a background and observations to a posterior."""

    @abstractmethod
    def prepare_sweep(self, gather: np.ndarray, r_diag: np.ndarray, b_scales: np.ndarray):
        """Everything one network needs that does not depend on the observation values."""

    @abstractmethod
    def apply_sweep(self, gain, y_anom: np.ndarray, background_anom: np.ndarray, *,
                    mean_only: bool = False) -> list[AnalysisResult]:
        """The analysis at every amplitude of ``gain`` for one observation vector."""


# Builds a method from a prior and the field shape.
MethodFactory = Callable[["Prior", "tuple[int, int, int]"], Method]
