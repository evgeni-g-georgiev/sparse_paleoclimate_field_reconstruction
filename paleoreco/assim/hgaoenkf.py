"""Hybrid gain analog offline EnKF (HGAOEnKF), with multiscale tendency augmentation (MTA).

Sun et al. (2024) build the prior ensemble from the archive states that best match the
observations, then blend its covariance with the static B through a hybrid gain: one prior
mean, one innovation, and two gains summed at weight ``hybrid_w`` (their Eq. 4 for the
mean, Eq. 5 for the deviations). At ``hybrid_w = 0`` the gain is purely static, at 1 purely
analog; the prior mean is the analog mean at every weight. Selection rules live in
:mod:`paleoreco.assim.analog`.

MTA appends centred first differences of the archive around each selected member, at
``tendency_lag_yr`` and ``tendency_extra_lags_yr``, and second differences at
``tendency_curvature_yr``, to the analog deviations, weighted by ``tendency_theta``.
``tendency_normalise`` puts every block on the reference lag's amplitude, and
``preserve_obs_trace`` rescales the augmented ensemble to the unaugmented one's
observation-space trace, so MTA reallocates variance rather than inflating it and stays
separable from ``b_scale``. With ``tendency_theta = 0`` the estimator is the published one.

Conventions follow 3DVar: anomaly space, nearest-cell H, pixel-space results. Unlike Sun et
al., who taper the gain and assimilate serially, both ``P H^T`` and ``H P H^T`` are tapered
and the update is a batch one, so the taper must be PSD in its own right.
``analog_localization_km`` optionally gives the analog covariance its own lengthscale (Sun
et al. Table 2).

The deviations are reduced by a partly static gain, so the posterior spread is guaranteed to
shrink only at ``hybrid_w = 1``; at low weight the per-cell variance can exceed the analog
ensemble's own by a few per cent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paleoreco.assim.analog import (
    ANALOG_CORRELATION,
    ANALOG_CORRELATION_PERCHAN,
    ANALOG_EVIDENCE,
    ANALOG_MISFIT,
    ANALOG_RULES,
    EVIDENCE_SCALE,
    analog_indices,
    correlation_indices,
    eligible_mask,
    evidence_indices,
)
from paleoreco.assim.ensrf import WhitenedBlock, mean_gain_apply, sqrt_gain_apply, whitened_block
from paleoreco.assim.innovation import nearest_age_index
from paleoreco.assim.method import AnalysisResult, Method, Observations
from paleoreco.assim.priors import Prior, taper_obs_blocks


@dataclass(frozen=True)
class _HybridSweepGain:
    """Per-network objects that do not depend on the observation values.

    The static block factorizes once here. The analog block depends on which states are
    selected, so it is built in :meth:`HGAOEnKF.apply_sweep`; this stages what selection
    needs and the taper the analog blocks are masked with.
    """

    gather: np.ndarray
    r_diag: np.ndarray
    b_scales: np.ndarray
    static: WhitenedBlock
    pool_at_obs: np.ndarray
    eligible: np.ndarray | None
    obs_channel: np.ndarray
    taper: tuple[np.ndarray, np.ndarray] | None


class HGAOEnKF(Method):
    """Analog offline EnKF with a hybrid gain over a fixed background covariance.

    ``pool`` holds the candidate states, ``(n_pool, D)``, as anomalies about the mean ``B``
    was built from; ``k`` members are selected per assimilation. ``exclude_yr`` drops
    candidates within that many years of the target age. ``evidence_scale`` is used by the
    evidence rule only. ``analog_localization_km=None`` tapers the analog covariance like
    the static one. The ``tendency_*`` and ``preserve_obs_trace`` arguments configure MTA,
    as described in the module docstring; ``redundancy_theta`` is the evidence rule's
    redundancy penalty.
    """

    def __init__(self, pool: np.ndarray, pool_ages: np.ndarray, B: np.ndarray,
                 shape: tuple[int, int, int], lats: np.ndarray, lons: np.ndarray, *,
                 k: int, hybrid_w: float, taper_meta: dict,
                 selection: str = ANALOG_MISFIT, exclude_yr: float = 0.0,
                 evidence_scale: float = EVIDENCE_SCALE,
                 analog_localization_km: float | None = None,
                 tendency_theta: float = 0.0, tendency_lag_yr: float = 0.0,
                 tendency_extra_lags_yr: tuple[float, ...] = (),
                 tendency_curvature_yr: tuple[float, ...] = (),
                 tendency_normalise: bool = False,
                 preserve_obs_trace: bool = False,
                 redundancy_theta: float = 0.0):
        if selection not in ANALOG_RULES:
            raise ValueError(f"unknown selection rule {selection!r}; expected one of {ANALOG_RULES}")
        if k < 2:
            raise ValueError(f"an analog ensemble needs at least 2 members; got {k}")
        if not 0.0 <= hybrid_w <= 1.0:
            raise ValueError(f"hybrid_w must lie in [0, 1]; got {hybrid_w}")
        if evidence_scale < 0.0:
            raise ValueError(f"evidence_scale must be non-negative; got {evidence_scale}")
        if tendency_theta < 0.0:
            raise ValueError(f"tendency_theta must be non-negative; got {tendency_theta}")
        if tendency_lag_yr < 0.0:
            raise ValueError(f"tendency_lag_yr must be non-negative; got {tendency_lag_yr}")
        # A zero lag would difference a state against itself: a silent no-op.
        if tendency_theta > 0.0 and tendency_lag_yr <= 0.0:
            raise ValueError("a positive tendency_theta needs a positive tendency_lag_yr; "
                             f"got {tendency_lag_yr}")
        extra = tuple(float(v) for v in tendency_extra_lags_yr)
        curvature = tuple(float(v) for v in tendency_curvature_yr)
        if any(v <= 0.0 for v in extra + curvature):
            raise ValueError("every tendency lag must be positive; got extra "
                             f"{extra} and curvature {curvature}")
        # Extra blocks share theta, so with theta off they would silently vanish.
        if (extra or curvature) and tendency_theta <= 0.0:
            raise ValueError("extra or curvature tendency lags need a positive "
                             f"tendency_theta; got {tendency_theta}")
        # A repeated lag would be stacked twice, doubling its weight.
        if float(tendency_lag_yr) in extra or len(set(extra)) != len(extra):
            raise ValueError("tendency_extra_lags_yr must not repeat a lag or the "
                             f"reference {tendency_lag_yr}; got {extra}")
        if len(set(curvature)) != len(curvature):
            raise ValueError(f"tendency_curvature_yr must not repeat a lag; got {curvature}")
        if redundancy_theta < 0.0:
            raise ValueError(f"redundancy_theta must be non-negative; got {redundancy_theta}")
        # The penalty is defined in the evidence rule's whitened coordinates.
        if redundancy_theta > 0.0 and selection != ANALOG_EVIDENCE:
            raise ValueError("redundancy_theta is defined against the evidence rule's "
                             f"whitened score; selection is {selection!r}")
        # A non-positive or non-finite lengthscale would silently zero the analog taper.
        if analog_localization_km is not None and not (
                np.isfinite(analog_localization_km) and analog_localization_km > 0.0):
            raise ValueError("analog_localization_km must be positive and finite, or None "
                             f"to inherit the static covariance's; got {analog_localization_km}")
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
        # Only the lengthscale can differ; shrinkage and coupling are inherited to limit
        # tuning cost.
        self.analog_localization_km = analog_localization_km
        self.analog_taper_meta = dict(self.taper_meta)
        if analog_localization_km is not None:
            self.analog_taper_meta["localization_km"] = float(analog_localization_km)
        self.tendency_theta = float(tendency_theta)
        self.tendency_lag_yr = float(tendency_lag_yr)
        self.tendency_extra_lags_yr = extra
        self.tendency_curvature_yr = curvature
        self.tendency_normalise = bool(tendency_normalise)
        self.preserve_obs_trace = bool(preserve_obs_trace)
        self.redundancy_theta = float(redundancy_theta)
        # One (lag, is_second_order) block each; the reference lag leads the list.
        self._tendency_blocks = (
            tuple([(self.tendency_lag_yr, False)]
                  + [(lag, False) for lag in extra]
                  + [(lag, True) for lag in curvature])
            if self.tendency_theta > 0.0 else ())
        # Neighbours depend only on the age axis, so they are computed once.
        self._tendency_pair = {lag: self._tendency_neighbours(lag)
                               for lag, _ in self._tendency_blocks}
        # Without normalisation the longest lag, having the largest differences, dominates.
        self._tendency_scale = {}
        if self.tendency_normalise and self._tendency_blocks:
            amplitude = {block: self._block_amplitude(block) for block in self._tendency_blocks}
            reference = amplitude[self._tendency_blocks[0]]
            self._tendency_scale = {block: (reference / amp if amp > 0.0 else 0.0)
                                    for block, amp in amplitude.items()}

    def _tendency_neighbours(self, lag: float) -> tuple[np.ndarray, np.ndarray]:
        """Pool rows one lag either side of each candidate, clamped at the archive's ends."""
        order = np.argsort(self.pool_ages, kind="stable")
        sorted_ages = self.pool_ages[order]
        lo = order[nearest_age_index(self.pool_ages - lag, sorted_ages)]
        hi = order[nearest_age_index(self.pool_ages + lag, sorted_ages)]
        return lo, hi

    def _block_rows(self, members: np.ndarray, block: tuple[float, bool],
                    eligible: np.ndarray | None) -> np.ndarray:
        """Raw difference rows for one (lag, order) block, before centring or weighting.

        Neighbours clamp at the archive's ends. A first difference there is rescaled to the
        nominal interval; a second difference with a clamped arm is dropped.
        """
        lag, second = block
        lo, hi = self._tendency_pair[lag]
        a, b = lo[members], hi[members]
        keep = a != b            # coincident neighbours leave no interval to divide through
        if eligible is not None:
            keep = keep & eligible[a] & eligible[b]
        if second:
            # Tested on the requested ages, so a lag that is not a whole number of steps
            # does not empty the block.
            age = self.pool_ages[members]
            keep = (keep & (age - lag >= self.pool_ages.min())
                    & (age + lag <= self.pool_ages.max()))
        if not keep.any():
            return np.empty((0, self.pool.shape[1]))
        if second:
            return (self.pool[b[keep]] - 2.0 * self.pool[members[keep]]
                    + self.pool[a[keep]])
        # The factor is 1 for interior rows.
        span = (self.pool_ages[b] - self.pool_ages[a])[keep]
        return (0.5 * (self.pool[b[keep]] - self.pool[a[keep]])
                * (2.0 * lag / span)[:, None])

    def _block_amplitude(self, block: tuple[float, bool]) -> float:
        """Rms of one block's difference rows over the whole pool.

        A property of the archive, independent of the observations.
        """
        rows = self._block_rows(np.arange(len(self.pool)), block, None)
        return float(np.sqrt((rows ** 2).mean())) if len(rows) else 0.0

    def _tendency_rows(self, members: np.ndarray,
                       eligible: np.ndarray | None) -> np.ndarray:
        """Centred tendency deviations for the selected members, one block per lag.

        A member is dropped from a block when either neighbour falls in the exclusion band,
        when its neighbours coincide, or when a second difference runs off the archive.
        """
        rows = []
        for block in self._tendency_blocks:
            tend = self._block_rows(members, block, eligible)
            if not len(tend):
                continue
            scale = self._tendency_scale.get(block, 1.0)
            if scale <= 0.0:
                continue
            rows.append((self.tendency_theta * scale) * (tend - tend.mean(axis=0)))
        if not rows:
            return np.empty((0, self.pool.shape[1]))
        return np.vstack(rows)

    def _trace_rescale(self, base: np.ndarray, dev: np.ndarray,
                       gain: _HybridSweepGain) -> float:
        """Factor holding ``dev`` at ``base``'s trace in whitened observation space.

        Keeps ``b_scale`` meaning the same amplitude with and without the extra rows.
        """
        g, r = gain.gather, gain.r_diag
        t_base = float(((base[:, g] ** 2) / r[None, :]).sum())
        t_aug = float(((dev[:, g] ** 2) / r[None, :]).sum())
        if t_aug <= 0.0 or t_base <= 0.0:
            return 1.0
        return float(np.sqrt(t_base / t_aug))

    def prepare_sweep(self, gather: np.ndarray, r_diag: np.ndarray,
                      b_scales: np.ndarray, *, age: float | None = None) -> _HybridSweepGain:
        """Factorize the static gain and stage everything selection needs.

        ``age`` is the target age, used for the exclusion band.
        """
        g = np.asarray(gather)
        n_cells = self.shape[1] * self.shape[2]
        static = whitened_block(self.B[:, g], self.B[np.ix_(g, g)], r_diag)
        return _HybridSweepGain(
            gather=g, r_diag=np.asarray(r_diag, dtype=np.float64),
            b_scales=np.asarray(b_scales, dtype=np.float64),
            static=static, pool_at_obs=self.pool[:, g],
            eligible=None if age is None else eligible_mask(self.pool_ages, age, self.exclude_yr),
            obs_channel=g // n_cells,
            taper=taper_obs_blocks(self.lats, self.lons, g, **self.analog_taper_meta))

    def select(self, gain: _HybridSweepGain, y_anom: np.ndarray) -> np.ndarray:
        """Pool indices of the analog ensemble for one observation vector.

        Public so a driver can record which states were chosen.
        """
        if self.selection in (ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN):
            per_chan = self.selection == ANALOG_CORRELATION_PERCHAN
            return correlation_indices(gain.pool_at_obs, y_anom, self.k,
                                       channel=gain.obs_channel if per_chan else None,
                                       eligible=gain.eligible)
        if self.selection == ANALOG_EVIDENCE:
            # Reuses the static factorization, so no second solve.
            return evidence_indices(gain.pool_at_obs, y_anom, gain.static, self.k,
                                    eligible=gain.eligible, scale=self.evidence_scale,
                                    redundancy=self.redundancy_theta)
        return analog_indices(gain.pool_at_obs, y_anom, gain.r_diag, self.k,
                              eligible=gain.eligible)

    def apply_sweep(self, gain: _HybridSweepGain, y_anom: np.ndarray,
                    background_anom: np.ndarray) -> list[AnalysisResult]:
        """Analysis at every ``b_scale`` for one innovation, one result per ``b_scale``.

        ``b_scale`` scales both covariances.
        """
        g = gain.gather
        selected = self.select(gain, y_anom)
        members = self.pool[selected]
        mu = members.mean(axis=0)
        dev = base = members - mu                             # (k, D)
        if self._tendency_blocks:
            dev = np.vstack([base, self._tendency_rows(selected, gain.eligible)])
            if self.preserve_obs_trace:
                dev = dev * self._trace_rescale(base, dev, gain)
        h_dev = dev[:, g].T                                   # (m, n_dev)
        # Normalised by the k members, not the rows: tendency rows are extra directions on
        # a k-member ensemble, not extra members.
        P_obs = (dev.T @ h_dev.T) / (self.k - 1.0)            # B_a H^T, never B_a itself
        S_obs = (h_dev @ h_dev.T) / (self.k - 1.0)            # H B_a H^T
        if gain.taper is not None:
            P_obs = P_obs * gain.taper[0]
            S_obs = S_obs * gain.taper[1]
        analog = whitened_block(P_obs, S_obs, gain.r_diag)

        w = self.hybrid_w
        x_b = np.asarray(background_anom, dtype=np.float64).ravel() + mu
        d = np.asarray(y_anom, dtype=np.float64) - x_b[g]

        n_cells = self.shape[1] * self.shape[2]
        out = []
        for b in gain.b_scales:
            x_a = x_b + (w * mean_gain_apply(analog, b, d)
                         + (1.0 - w) * mean_gain_apply(gain.static, b, d))
            post = dev.T - (w * sqrt_gain_apply(analog, b, h_dev)
                            + (1.0 - w) * sqrt_gain_apply(gain.static, b, h_dev))
            # Normalised by k as P_obs is; every block is centred, so no mean is removed.
            scale = b / (self.k - 1.0)
            out.append(AnalysisResult(
                mean_anom=x_a.reshape(self.shape),
                posterior_var=(scale * (post ** 2).sum(axis=1)).reshape(self.shape),
                posterior_cross_var=(scale * (post[:n_cells] * post[n_cells:]).sum(axis=1)
                                     ).reshape(self.shape[1:])))
        return out

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        gain = self.prepare_sweep(obs.gather, obs.sse, np.array([1.0]))
        return self.apply_sweep(gain, obs.y_anom, background_anom)[0]


def make_hgaoenkf(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray, *,
    k: int, hybrid_w: float, selection: str = ANALOG_MISFIT, exclude_yr: float = 0.0,
    evidence_scale: float = EVIDENCE_SCALE, analog_localization_km: float | None = None,
    tendency_theta: float = 0.0, tendency_lag_yr: float = 0.0,
    tendency_extra_lags_yr: tuple[float, ...] = (),
    tendency_curvature_yr: tuple[float, ...] = (),
    tendency_normalise: bool = False, preserve_obs_trace: bool = False,
    redundancy_theta: float = 0.0,
):
    """A method factory building :class:`HGAOEnKF` from a built prior.

    The pool is taken from ``prior.ages``, so it is always the states B was built from.
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
                        evidence_scale=evidence_scale,
                        analog_localization_km=analog_localization_km,
                        tendency_theta=tendency_theta,
                        tendency_lag_yr=tendency_lag_yr,
                        tendency_extra_lags_yr=tendency_extra_lags_yr,
                        tendency_curvature_yr=tendency_curvature_yr,
                        tendency_normalise=tendency_normalise,
                        preserve_obs_trace=preserve_obs_trace,
                        redundancy_theta=redundancy_theta)

    return factory
