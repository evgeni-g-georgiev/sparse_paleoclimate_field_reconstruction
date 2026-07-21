"""Common contract every reconstruction method implements.

A method consumes a background and a set of observations for one assimilation and
returns an :class:`AnalysisResult`, which carries the posterior mean and, where the
method can produce one, the posterior variance.

All fields are in anomaly space (state minus per-cell climatology, observations
minus their per-cell climatology); :meth:`AnalysisResult.to_celsius` maps back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observations:
    """One assimilation's observations, reduced to what the update needs.

    ``gather`` indexes the flattened ``(2, n_lat, n_lon)`` state at each
    observation's grid cell (the action of H, nearest-cell selection); ``y_anom``
    is the observation in anomaly space; ``sse`` is its error variance (R's
    diagonal entry).
    """

    gather: np.ndarray
    y_anom: np.ndarray
    sse: np.ndarray


@dataclass(frozen=True)
class AnalysisResult:
    """Posterior of one assimilation, in anomaly space.

    ``mean_anom`` is ``(2, n_lat, n_lon)``; ``posterior_var`` (same shape) is filled
    by whichever method can produce it.
    """

    mean_anom: np.ndarray
    posterior_var: np.ndarray | None = None

    def to_celsius(self, clim_mean: np.ndarray) -> np.ndarray:
        """Posterior mean in degC: anomaly plus the per-cell climatology."""
        return self.mean_anom + clim_mean

    def predict_obs(self, gather: np.ndarray) -> np.ndarray:
        """Posterior-mean anomaly at the given flat cell indices (H applied)."""
        return self.mean_anom.ravel()[gather]

    def predict_obs_var(self, gather: np.ndarray) -> np.ndarray | None:
        """Posterior variance at those cells, or ``None`` if the method reports no spread.

        Scoring calibration in observation space needs the spread where the observations
        are, not the whole map.
        """
        if self.posterior_var is None:
            return None
        return self.posterior_var.ravel()[gather]


class Method(ABC):
    """A reconstruction method: background plus observations to a posterior."""

    @abstractmethod
    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        """Assimilate ``obs`` into ``background_anom`` (flattened state anomaly)."""

    def analyze_many(
        self, obs: Observations, backgrounds: list[np.ndarray]
    ) -> list[AnalysisResult]:
        """Analyses for several backgrounds that share one observation network.

        The default reuses :meth:`analyze`; a method whose operators do not depend
        on the background (a fixed-gain 3DVar) overrides this to factorize once and
        apply many, which is the common case when comparing first guesses.
        """
        return [self.analyze(obs, bg) for bg in backgrounds]
