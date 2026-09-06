"""Hybrid gain analog offline EnKF: a data-selected covariance blended with the static one.

Sun et al. (2024) build the prior ensemble from the archive states that best match the
observations of the assimilation at hand, so its covariance describes the climate regime
being reconstructed rather than the average of the whole run. A k-member covariance over
thousands of cells is badly undersampled, so the analysis blends it with the static
covariance through a hybrid gain: one prior mean, one innovation, two gains summed at
weight ``hybrid_w``. Sun et al. Eq. 4 gives the mean update and Eq. 5 the deviation update,
both of which this implements. They reach that form through the ensemble framework of Lei
et al. (2021), citing Penny (2014) for the hybrid gain idea; Penny's own scheme applies the
two gains in sequence and leaves the deviations untouched, so it is the ancestor rather than
the algorithm here.

``hybrid_w`` is Sun et al.'s hybrid weight alpha: at 0 the gain is purely static, at 1
purely analog. The prior mean is the analog mean at every weight. Weight 0 reproduces the
mean of their AOEnKF-B but not its spread, since Eq. 5 reduces the analog deviations where
AOEnKF-B carries the climatological ones. At weight 1 the static gain leaves the update
entirely, and B reaches the analysis only through whatever the selection rule makes of it, so
the hybrid is a limit of the scheme rather than a fixed part of it.

How candidates are ranked is a choice the family leaves open; :mod:`paleoreco.assim.analog`
holds the rules, and ``selection`` names one.

Two terms extend that scheme, both zero by default and both nested: at zero the analysis is
the published one. The tendency term augments the analog deviations with the archive's local
rate of change around each selected member, differenced over ``tendency_lag_yr``. Selection
draws members that agree with the observations, so they agree with each other there too and
the covariance is flattest in the directions the gain leans on hardest; the tendency modes
never had to agree with anything, and they restore that spread out of the archive's time
ordering, which selection otherwise reads only to build the exclusion band. The redundancy
term is its selection-side counterpart and lives with the rules.

One lag samples one timescale. ``tendency_extra_lags_yr`` carries the same construction at
further lags and ``tendency_curvature_yr`` at second differences, so the prior gains the
archive's flow across the band of timescales D-O variability actually occupies rather than
at a single one. ``tendency_normalise`` puts every block on the amplitude of the reference
lag, without which a stack is dominated by its longest lag and is little more than a
rescaled copy of it. The reference carries no distinction beyond setting that amplitude:
choosing a different one rescales every block by one factor, which ``tendency_theta`` already
spans. Both default to the single-lag term exactly.

``preserve_obs_trace`` is what makes such a stack comparable to the estimator it extends.
``b_scale`` multiplies the covariance, so a constant factor on that covariance is a
relabelling of the amplitude a lane sweeps; the ``k - 1`` normalizer and the reference lag's
own amplitude are both such constants, and neither moves an analysis. Augmenting the
deviations is not constant: it inflates the prior by an amount that grows with
``tendency_theta``, so the amplitude that suits the stack moves with the weight and the two
cannot be read apart. Rescaling the augmented ensemble to the unaugmented one's
observation-space trace holds the amplitude where it was, which leaves weight and amplitude
separable and the comparison about which directions the prior carries.

The state and observation conventions are pixel 3DVar's: anomaly space throughout, H is
nearest-cell selection, and the returned :class:`AnalysisResult` is pixel-space. Both
covariances carry a Schur taper from the prior that built the static one, except that Sun et
al. Table 2 give the flow-dependent covariance its own localization lengthscale, tighter
than the static one as the ensemble shrinks; ``analog_localization_km`` is that lengthscale.
Their values are radians on another model, grid and network, so the schedule transfers but
the number does not.

One property of Eq. 5 is worth knowing when reading the posterior spread: the deviations
start from whatever built the analog covariance but are reduced by a gain that is part
static, so the mean and the spread come from different mixtures. A square-root update guarantees the
spread shrinks only when the gain matches the covariance the deviations came from, which
here holds at ``hybrid_w = 1`` and is approached as the weight rises. At low weight the
per-cell variance can exceed the analog ensemble's own by a few per cent.
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

    The static block factorizes once here, as it does for 3DVar. The analog block cannot:
    it depends on which states the observations select, so it is built in
    :meth:`HGAOEnKF.apply_sweep`. What can be prepared is everything selection needs
    (``pool_at_obs``, ``eligible``, ``obs_channel`` for the per-channel correlation rule,
    and ``static`` for the evidence rule) and the reduced taper the analog blocks are
    masked with.
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

    ``pool`` is the archive of candidate states as anomalies about the same mean ``B`` was
    built from, ``(n_pool, D)``; ``k`` members are drawn from it per assimilation.
    ``exclude_yr`` drops candidates within that many years of the age being reconstructed,
    which matters only where the archive spans that age. ``evidence_scale`` is the
    background amplitude the evidence rule scores against and is unused by the others.
    ``analog_localization_km`` localizes the analog covariance alone; ``None`` gives it the
    static covariance's lengthscale, so the two are tapered identically.
    ``tendency_theta`` weights the archive's local tendency modes into the analog
    covariance and ``tendency_lag_yr`` is the interval they are differenced over;
    ``tendency_extra_lags_yr`` differences at further intervals and
    ``tendency_curvature_yr`` at second differences, both empty by default;
    ``tendency_normalise`` rescales every block to the reference lag's amplitude, so which
    lag is the reference sets a scale ``tendency_theta`` already spans;
    ``preserve_obs_trace`` holds the augmented ensemble at the unaugmented one's
    observation-space trace, which keeps the flow weight and ``b_scale`` separable.
    ``redundancy_theta`` charges a candidate for resembling one already selected.
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
        # A weighted tendency with no lag differences a state against itself, which would
        # be a silent no-op rather than the term the caller asked for.
        if tendency_theta > 0.0 and tendency_lag_yr <= 0.0:
            raise ValueError("a positive tendency_theta needs a positive tendency_lag_yr; "
                             f"got {tendency_lag_yr}")
        extra = tuple(float(v) for v in tendency_extra_lags_yr)
        curvature = tuple(float(v) for v in tendency_curvature_yr)
        # A non-positive extra lag differences a state against itself exactly as a
        # non-positive reference lag would, so it is rejected for the same reason.
        if any(v <= 0.0 for v in extra + curvature):
            raise ValueError("every tendency lag must be positive; got extra "
                             f"{extra} and curvature {curvature}")
        # Blocks beyond the reference one are weighted by the same theta, so with the term
        # switched off they would silently be nothing rather than the stack asked for.
        if (extra or curvature) and tendency_theta <= 0.0:
            raise ValueError("extra or curvature tendency lags need a positive "
                             f"tendency_theta; got {tendency_theta}")
        # Blocks are keyed by lag when their neighbours are looked up but stacked as a
        # list, so a repeated first difference would be carried twice at full weight
        # rather than once, which reads as a lag that carries more weight than it was given.
        if float(tendency_lag_yr) in extra or len(set(extra)) != len(extra):
            raise ValueError("tendency_extra_lags_yr must not repeat a lag or the "
                             f"reference {tendency_lag_yr}; got {extra}")
        if len(set(curvature)) != len(curvature):
            raise ValueError(f"tendency_curvature_yr must not repeat a lag; got {curvature}")
        if redundancy_theta < 0.0:
            raise ValueError(f"redundancy_theta must be non-negative; got {redundancy_theta}")
        # The penalty is a cosine in the coordinates the evidence score whitens by, so it
        # has no meaning under a rule that never forms them.
        if redundancy_theta > 0.0 and selection != ANALOG_EVIDENCE:
            raise ValueError("redundancy_theta is defined against the evidence rule's "
                             f"whitened score; selection is {selection!r}")
        # A non-finite or non-positive lengthscale makes every Gaspari-Cohn comparison
        # false, so the taper returns its pre-allocated zeros and silences the analog
        # covariance instead of localizing it. The analysis stays finite, so nothing
        # downstream would surface it.
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
        # Only the lengthscale is separable. Shrinkage and the channel coupling regularize
        # the prior rather than answer Sun et al.'s ensemble-size question, so the analog
        # covariance inherits them and the comparison stays one of estimators.
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
        # One block per (lag, order). The reference lag leads, so it is the amplitude the
        # others are normalised against and the single-lag term is the head of the list.
        self._tendency_blocks = (
            tuple([(self.tendency_lag_yr, False)]
                  + [(lag, False) for lag in extra]
                  + [(lag, True) for lag in curvature])
            if self.tendency_theta > 0.0 else ())
        # The neighbours depend on the archive's age axis alone, so they are the same for
        # every network and every analysis this estimator runs.
        self._tendency_pair = {lag: self._tendency_neighbours(lag)
                               for lag, _ in self._tendency_blocks}
        # Every block is put on the reference block's amplitude, so each amplitude is measured
        # once over the pool rather than once per block it is compared against. A difference
        # over a longer lag is larger, and an unnormalised stack is dominated by its longest.
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

        The neighbour lookup clamps at the archive's ends, so a row there spans less than the
        nominal interval. Dividing a first difference through the interval actually differenced
        carries it at the block's own amplitude, leaving an end row a one-sided difference
        rather than a centred one shrunk towards zero. A second difference has no such reading:
        with one arm clamped it collapses towards a first difference, so those rows drop out.
        """
        lag, second = block
        lo, hi = self._tendency_pair[lag]
        a, b = lo[members], hi[members]
        keep = a != b            # coincident neighbours leave no interval to divide through
        if eligible is not None:
            keep = keep & eligible[a] & eligible[b]
        if second:
            # Tested on the age asked for rather than the interval returned: a lag that is not
            # a whole number of archive steps rounds one arm short for every member, which
            # would empty the block instead of trimming the ends the archive cannot span.
            age = self.pool_ages[members]
            keep = (keep & (age - lag >= self.pool_ages.min())
                    & (age + lag <= self.pool_ages.max()))
        if not keep.any():
            return np.empty((0, self.pool.shape[1]))
        if second:
            return (self.pool[b[keep]] - 2.0 * self.pool[members[keep]]
                    + self.pool[a[keep]])
        # The factor is 1 wherever both arms are whole, so an interior row is the centred
        # difference it always was.
        span = (self.pool_ages[b] - self.pool_ages[a])[keep]
        return (0.5 * (self.pool[b[keep]] - self.pool[a[keep]])
                * (2.0 * lag / span)[:, None])

    def _block_amplitude(self, block: tuple[float, bool]) -> float:
        """Rms of one block's difference rows over the whole pool.

        Measured on the pool rather than per analysis, so it is a property of the archive
        and does not move with the observations.
        """
        rows = self._block_rows(np.arange(len(self.pool)), block, None)
        return float(np.sqrt((rows ** 2).mean())) if len(rows) else 0.0

    def _tendency_rows(self, members: np.ndarray,
                       eligible: np.ndarray | None) -> np.ndarray:
        """Centred tendency deviations for the selected members, one block per lag.

        A member contributes nothing to a block where the exclusion band rules out either
        neighbour, which is what stops a lane whose archive spans the analysis age from
        reaching back inside that band through the tendency, nothing where the two
        neighbours coincide and there is no interval to difference, and nothing to a
        curvature block where the archive's end leaves it one short arm. The test is per
        block, so a long lag can drop a member that a short one keeps.
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

        The gain reads the ensemble only through ``H X'`` against R, so matching that
        trace is what leaves ``b_scale`` meaning the amplitude it meant before the extra
        rows were added. Returns 1.0 where the augmented ensemble is already the base one.
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

        ``age`` is the age being reconstructed; it drives the exclusion band, and is unused
        where the archive is disjoint from the target in time.
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

        Exposed rather than kept inside the analysis so a driver can record which states
        were chosen; the selection is a pure function of the observations, so recomputing
        it costs one pass over the pool at the observation cells.
        """
        if self.selection in (ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN):
            per_chan = self.selection == ANALOG_CORRELATION_PERCHAN
            return correlation_indices(gain.pool_at_obs, y_anom, self.k,
                                       channel=gain.obs_channel if per_chan else None,
                                       eligible=gain.eligible)
        if self.selection == ANALOG_EVIDENCE:
            # The static block is already factorized for the gain, so the marginal
            # likelihood costs one product over the pool rather than a second solve.
            return evidence_indices(gain.pool_at_obs, y_anom, gain.static, self.k,
                                    eligible=gain.eligible, scale=self.evidence_scale,
                                    redundancy=self.redundancy_theta)
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
        selected = self.select(gain, y_anom)
        members = self.pool[selected]
        mu = members.mean(axis=0)
        dev = base = members - mu                             # (k, D)
        if self._tendency_blocks:
            dev = np.vstack([base, self._tendency_rows(selected, gain.eligible)])
            if self.preserve_obs_trace:
                dev = dev * self._trace_rescale(base, dev, gain)
        h_dev = dev[:, g].T                                   # (m, n_dev)
        # The normalizer counts the members, not the rows: the tendency rows are extra
        # directions added to the same k-member ensemble, so at zero weight they contribute
        # nothing and the covariance is the published one exactly. Counting the rows instead
        # would scale both blocks by one constant, which b_scale already spans.
        P_obs = (dev.T @ h_dev.T) / (self.k - 1.0)            # B_a H^T, never B_a itself
        S_obs = (h_dev @ h_dev.T) / (self.k - 1.0)            # H B_a H^T
        if gain.taper is not None:
            P_obs = P_obs * gain.taper[0]
            S_obs = S_obs * gain.taper[1]
        analog = whitened_block(P_obs, S_obs, gain.r_diag)

        w = self.hybrid_w
        x_b = np.asarray(background_anom, dtype=np.float64).ravel() + mu
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
    evidence_scale: float = EVIDENCE_SCALE, analog_localization_km: float | None = None,
    tendency_theta: float = 0.0, tendency_lag_yr: float = 0.0,
    tendency_extra_lags_yr: tuple[float, ...] = (),
    tendency_curvature_yr: tuple[float, ...] = (),
    tendency_normalise: bool = False, preserve_obs_trace: bool = False,
    redundancy_theta: float = 0.0,
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
