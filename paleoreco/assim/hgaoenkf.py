"""Hybrid gain analog offline EnKF: a data-selected covariance blended with the static one.

Sun et al. (2024) build the prior ensemble from the archive states that best match the
observations of the assimilation at hand, so its covariance describes the climate regime
being reconstructed rather than the average of the whole run. A k-member covariance over
thousands of cells is badly undersampled, so the analysis blends it with the static
covariance through a hybrid gain (Penny 2014; Lei et al. 2021): one prior mean, one
innovation, two gains summed at weight ``hybrid_w``. Sun et al. Eq. 4 gives the mean
update and Eq. 5 the deviation update, both of which this implements.

Two departures from the paper, each measured rather than assumed:

* how candidates are ranked is a choice rather than Eq. 3's spatial-pattern correlation.
  The misfit rule scores the R-weighted residual, so a corrected observation pair is scored
  the way the update consumes it; the evidence rule scores the candidate's marginal
  likelihood under the same background covariance the update uses, and reduces to the
  misfit rule at ``evidence_scale = 0`` (see :mod:`paleoreco.assim.analog`);
* the prior mean is ``hybrid_w`` times the analog mean rather than the analog mean itself,
  which makes ``hybrid_w = 0`` reduce exactly to the static-covariance analysis rather than
  to Sun et al.'s AOEnKF-B.

The state and observation conventions are pixel 3DVar's: anomaly space throughout, H is
nearest-cell selection, and the returned :class:`AnalysisResult` is pixel-space. Both
covariances carry the same Schur taper, taken from the prior that built the static one, so
the blend never compares differently regularized objects.

One property of Eq. 5 is worth knowing when reading the posterior spread: the deviations
start from the analog ensemble alone but are reduced by a gain that is part static, so the
mean and the spread come from different mixtures. A square-root update guarantees the
spread shrinks only when the gain matches the covariance the deviations came from, which
here holds at ``hybrid_w = 1`` and is approached as the weight rises. At low weight the
per-cell variance can exceed the analog ensemble's own by a few per cent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paleoreco.assim.analog import (
    ANALOG_EVIDENCE,
    ANALOG_MISFIT,
    ANALOG_RULES,
    ANALOG_WINDOW,
    EVIDENCE_SCALE,
    analog_indices,
    eligible_mask,
    evidence_indices,
    window_indices,
)
from paleoreco.assim.ensrf import WhitenedBlock, mean_gain_apply, sqrt_gain_apply, whitened_block
from paleoreco.assim.method import AnalysisResult, Method, Observations
from paleoreco.assim.priors import Prior, taper_obs_blocks


@dataclass(frozen=True)
class _HybridSweepGain:
    """Per-network objects that do not depend on the observation values.

    The static block factorizes once here, as it does for 3DVar. The analog block cannot:
    it depends on which states the observations select, so it is built in
    :meth:`HGAOEnKF.apply_sweep`. What can be prepared is everything selection needs
    (``pool_at_obs``, ``eligible``, and ``static`` for the evidence rule) and the reduced
    taper the analog blocks are masked with.
    """

    gather: np.ndarray
    r_diag: np.ndarray
    b_scales: np.ndarray
    static: WhitenedBlock
    pool_at_obs: np.ndarray
    eligible: np.ndarray | None
    taper: tuple[np.ndarray, np.ndarray] | None
    age: float | None


class HGAOEnKF(Method):
    """Analog offline EnKF with a hybrid gain over a fixed background covariance.

    ``pool`` is the archive of candidate states as anomalies about the same mean ``B`` was
    built from, ``(n_pool, D)``; ``k`` members are drawn from it per assimilation.
    ``exclude_yr`` drops candidates within that many years of the age being reconstructed,
    which matters only where the archive spans that age. ``evidence_scale`` is the
    background amplitude the evidence rule scores against and is unused by the others.
    """

    def __init__(self, pool: np.ndarray, pool_ages: np.ndarray, B: np.ndarray,
                 shape: tuple[int, int, int], lats: np.ndarray, lons: np.ndarray, *,
                 k: int, hybrid_w: float, taper_meta: dict,
                 selection: str = ANALOG_MISFIT, exclude_yr: float = 0.0,
                 evidence_scale: float = EVIDENCE_SCALE):
        if selection not in ANALOG_RULES:
            raise ValueError(f"unknown selection rule {selection!r}; expected one of {ANALOG_RULES}")
        if k < 2:
            raise ValueError(f"an analog ensemble needs at least 2 members; got {k}")
        if not 0.0 <= hybrid_w <= 1.0:
            raise ValueError(f"hybrid_w must lie in [0, 1]; got {hybrid_w}")
        if evidence_scale < 0.0:
            raise ValueError(f"evidence_scale must be non-negative; got {evidence_scale}")
        self.pool = np.asarray(pool, dtype=np.float64)
        self.pool_ages = np.asarray(pool_ages, dtype=np.float64)
        self.B = np.asfortranarray(B, dtype=np.float64)
        self.diagB = np.diag(self.B).copy()
        self.shape = shape
        self.lats, self.lons = np.asarray(lats), np.asarray(lons)
        self.k = int(k)
        self.hybrid_w = float(hybrid_w)
        self.selection = selection
        self.exclude_yr = float(exclude_yr)
        self.evidence_scale = float(evidence_scale)
        self.taper_meta = {key: taper_meta[key]
                           for key in ("localization_km", "shrinkage_lambda", "alpha")}

    def prepare_sweep(self, gather: np.ndarray, r_diag: np.ndarray,
                      b_scales: np.ndarray, *, age: float | None = None) -> _HybridSweepGain:
        """Factorize the static gain and stage everything selection needs.

        ``age`` is the age being reconstructed; it drives the exclusion band and the window
        rule, and is unused where the archive is disjoint from the target in time.
        """
        g = np.asarray(gather)
        if self.selection == ANALOG_WINDOW and age is None:
            raise ValueError("window selection needs the age being reconstructed")
        static = whitened_block(self.B[:, g], self.B[np.ix_(g, g)], r_diag)
        return _HybridSweepGain(
            gather=g, r_diag=np.asarray(r_diag, dtype=np.float64),
            b_scales=np.asarray(b_scales, dtype=np.float64),
            static=static, pool_at_obs=self.pool[:, g],
            eligible=None if age is None else eligible_mask(self.pool_ages, age, self.exclude_yr),
            taper=taper_obs_blocks(self.lats, self.lons, g, **self.taper_meta),
            age=age)

    def select(self, gain: _HybridSweepGain, y_anom: np.ndarray) -> np.ndarray:
        """Pool indices of the analog ensemble for one observation vector.

        Exposed rather than kept inside the analysis so a driver can record which states
        were chosen; the selection is a pure function of the observations, so recomputing
        it costs one pass over the pool at the observation cells.
        """
        if self.selection == ANALOG_WINDOW:
            return window_indices(self.pool_ages, gain.age, self.k, eligible=gain.eligible)
        if self.selection == ANALOG_EVIDENCE:
            # The static block is already factorized for the gain, so the marginal
            # likelihood costs one product over the pool rather than a second solve.
            return evidence_indices(gain.pool_at_obs, y_anom, gain.static, self.k,
                                    eligible=gain.eligible, scale=self.evidence_scale)
        return analog_indices(gain.pool_at_obs, y_anom, gain.r_diag, self.k,
                              eligible=gain.eligible)

    def apply_sweep(self, gain: _HybridSweepGain, y_anom: np.ndarray,
                    background_anom: np.ndarray) -> list[AnalysisResult]:
        """Analysis at every ``b_scale`` for one innovation, one result per ``b_scale``.

        ``b_scale`` scales both covariances, so it keeps its meaning as the amplitude of
        the background relative to the observations; the ensemble deviations scale with its
        square root to match.
        """
        g = gain.gather
        members = self.pool[self.select(gain, y_anom)]
        mu = members.mean(axis=0)
        dev = members - mu                                    # (k, D)
        h_dev = dev[:, g].T                                   # (m, k)
        P_obs = (dev.T @ h_dev.T) / (self.k - 1.0)            # B_a H^T, never B_a itself
        S_obs = (h_dev @ h_dev.T) / (self.k - 1.0)            # H B_a H^T
        if gain.taper is not None:
            P_obs = P_obs * gain.taper[0]
            S_obs = S_obs * gain.taper[1]
        analog = whitened_block(P_obs, S_obs, gain.r_diag)

        w = self.hybrid_w
        x_b = np.asarray(background_anom, dtype=np.float64).ravel() + w * mu
        d = np.asarray(y_anom, dtype=np.float64) - x_b[g]

        out = []
        for b in gain.b_scales:
            x_a = x_b + (w * mean_gain_apply(analog, b, d)
                         + (1.0 - w) * mean_gain_apply(gain.static, b, d))
            post = dev.T - (w * sqrt_gain_apply(analog, b, h_dev)
                            + (1.0 - w) * sqrt_gain_apply(gain.static, b, h_dev))
            out.append(AnalysisResult(
                mean_anom=x_a.reshape(self.shape),
                posterior_var=(b * post.var(axis=1, ddof=1)).reshape(self.shape)))
        return out

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        gain = self.prepare_sweep(obs.gather, obs.sse, np.array([1.0]))
        return self.apply_sweep(gain, obs.y_anom, background_anom)[0]


def make_hgaoenkf(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray, *,
    k: int, hybrid_w: float, selection: str = ANALOG_MISFIT, exclude_yr: float = 0.0,
    evidence_scale: float = EVIDENCE_SCALE,
):
    """A method factory building :class:`HGAOEnKF` from a built prior.

    The candidate pool is recovered from ``prior.ages`` rather than passed alongside, so it
    is always the same states, in the same anomaly frame, that the prior's covariance was
    built from.
    """
    ages_i = np.asarray(ages, dtype=np.int64)

    def factory(prior: Prior, shape: tuple[int, int, int]) -> HGAOEnKF:
        idx = np.searchsorted(ages_i, prior.ages)
        if not np.array_equal(ages_i[idx], np.asarray(prior.ages, dtype=np.int64)):
            raise ValueError("prior ages are not a subset of the cube's ages")
        pool = cube[idx].reshape(len(idx), -1).astype(np.float64) - prior.clim_mean.ravel()
        return HGAOEnKF(pool, prior.ages, prior.B, shape, lats, lons,
                        k=k, hybrid_w=hybrid_w, taper_meta=prior.meta,
                        selection=selection, exclude_yr=exclude_yr,
                        evidence_scale=evidence_scale)

    return factory
