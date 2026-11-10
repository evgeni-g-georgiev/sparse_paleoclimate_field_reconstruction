"""Hybrid gain analog offline EnKF (HGAOEnKF), with multiscale tendency augmentation (MTA).

Sun et al. (2024) build the prior ensemble from the archive states that best match the
observations, then blend its covariance with the static B through a hybrid gain: one prior
mean, one innovation, and two gains summed at weight ``hybrid_w`` (their Eq. 4 for the
mean, Eq. 5 for the deviations). At ``hybrid_w = 0`` the gain is purely static, at 1 purely
analog; the prior mean is the analog mean at every weight. Selection rules live in
:mod:`paleoreco.assim.analog`; the evidence rule at ``kappa = 0`` is Sun et al.'s misfit rule.

MTA appends centred first and second differences of the archive around each selected
member, at the lags of a :class:`TendencyStack`, to the analog deviations. With ``X'`` the
member deviations, ``T`` the stacked tendency blocks, and ``t_X``, ``t_T`` their traces in
whitened observation space at the observed cells,

    Z = [ sqrt(1 - w) X' ;  sqrt(w t_X / t_T) T ],    P = Z^T Z / (k - 1).

The tendency share ``w`` moves whitened observation-space variance from the member
deviations to the tendency rows without changing the total, so ``b_scale`` means the same
amplitude at every ``w``. ``w = 0`` is the estimator without MTA; ``w = 1`` keeps only the
tendency directions, about the same analog mean.

Conventions follow 3DVar: anomaly space, nearest-cell H, pixel-space results. Unlike Sun et
al., who taper the gain and assimilate serially, both ``P H^T`` and ``H P H^T`` are tapered,
with the static covariance's taper, and the update is a batch one, so the taper must be PSD
in its own right.

The deviations are reduced by a partly static gain, so the posterior spread is guaranteed to
shrink only at ``hybrid_w = 1``; at low weight the per-cell variance can exceed the analog
ensemble's own by a few per cent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paleoreco.assim.analog import (
    ANALOG_CORRELATION_PERCHAN,
    ANALOG_EVIDENCE,
    ANALOG_RULES,
    correlation_indices,
    evidence_indices,
)
from paleoreco.assim.ensrf import WhitenedBlock, mean_gain_apply, sqrt_gain_apply, whitened_block
from paleoreco.assim.innovation import nearest_age_index
from paleoreco.assim.method import AnalysisResult, Method
from paleoreco.assim.priors import Prior, taper_obs_blocks


@dataclass(frozen=True)
class TendencyStack:
    """The lags, in years, at which MTA differences the archive around each member.

    ``first`` are centred first differences and ``second`` second differences. Every block
    is put on the rms amplitude of the ``reference`` first difference, so a long lag does
    not dominate the stack just by having larger differences.
    """

    reference: float
    first: tuple[float, ...]
    second: tuple[float, ...] = ()

    def __post_init__(self):
        first = tuple(float(v) for v in self.first)
        second = tuple(float(v) for v in self.second)
        object.__setattr__(self, "reference", float(self.reference))
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        if any(v <= 0.0 for v in first + second):
            raise ValueError(f"every tendency lag must be positive; got {first} and {second}")
        # A repeated lag would be stacked twice, doubling its weight.
        if len(set(first)) != len(first) or len(set(second)) != len(second):
            raise ValueError(f"a tendency stack must not repeat a lag; got {first} and {second}")
        if self.reference not in first:
            raise ValueError(f"the reference lag {self.reference} must be a first difference; "
                             f"got {first}")

    @property
    def blocks(self) -> tuple[tuple[float, bool], ...]:
        """One ``(lag, is_second_order)`` pair per block, first differences first."""
        return (tuple((lag, False) for lag in self.first)
                + tuple((lag, True) for lag in self.second))


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
    obs_channel: np.ndarray
    taper: tuple[np.ndarray, np.ndarray] | None


class HGAOEnKF(Method):
    """Analog offline EnKF with a hybrid gain over a fixed background covariance.

    ``pool`` holds the candidate states, ``(n_pool, D)``, as anomalies about the mean ``B``
    was built from; ``k`` members are selected per assimilation by ``selection``, with
    ``kappa`` the evidence rule's background scale. ``stack`` and ``tendency_share``
    configure MTA, as described in the module docstring.
    """

    def __init__(self, pool: np.ndarray, pool_ages: np.ndarray, B: np.ndarray,
                 shape: tuple[int, int, int], lats: np.ndarray, lons: np.ndarray, *,
                 k: int, hybrid_w: float, taper_meta: dict,
                 selection: str = ANALOG_EVIDENCE, kappa: float = 0.0,
                 stack: TendencyStack | None = None, tendency_share: float = 0.0):
        if selection not in ANALOG_RULES:
            raise ValueError(f"unknown selection rule {selection!r}; "
                             f"expected one of {ANALOG_RULES}")
        if k < 2:
            raise ValueError(f"an analog ensemble needs at least 2 members; got {k}")
        if not 0.0 <= hybrid_w <= 1.0:
            raise ValueError(f"hybrid_w must lie in [0, 1]; got {hybrid_w}")
        if kappa < 0.0:
            raise ValueError(f"kappa must be non-negative; got {kappa}")
        if kappa > 0.0 and selection != ANALOG_EVIDENCE:
            raise ValueError(f"kappa is the evidence rule's background scale; "
                             f"selection is {selection!r}")
        if not 0.0 <= tendency_share <= 1.0:
            raise ValueError(f"tendency_share must lie in [0, 1]; got {tendency_share}")
        if tendency_share > 0.0 and stack is None:
            raise ValueError("a positive tendency_share needs a tendency stack")
        self.pool = np.asarray(pool, dtype=np.float64)
        self.pool_ages = np.asarray(pool_ages, dtype=np.float64)
        self.B = np.asfortranarray(B, dtype=np.float64)
        self.shape = shape
        self.lats, self.lons = np.asarray(lats), np.asarray(lons)
        self.k = int(k)
        self.hybrid_w = float(hybrid_w)
        self.selection = selection
        self.kappa = float(kappa)
        self.stack = stack
        self.tendency_share = float(tendency_share)
        self.taper_meta = {key: taper_meta[key]
                           for key in ("localization_km", "shrinkage_lambda", "alpha")}
        self._blocks = stack.blocks if stack is not None else ()
        # Neighbours depend only on the age axis, so they are computed once.
        self._neighbours = {lag: self._tendency_neighbours(lag) for lag, _ in self._blocks}
        # s_i per block: the reference lag's rms amplitude over this block's.
        self.tendency_scale = {}
        if self._blocks:
            amplitude = {block: self._block_amplitude(block) for block in self._blocks}
            reference = amplitude[(stack.reference, False)]
            self.tendency_scale = {block: (reference / amp if amp > 0.0 else 0.0)
                                   for block, amp in amplitude.items()}

    def _tendency_neighbours(self, lag: float) -> tuple[np.ndarray, np.ndarray]:
        """Pool rows one lag either side of each candidate, clamped at the archive's ends."""
        order = np.argsort(self.pool_ages, kind="stable")
        sorted_ages = self.pool_ages[order]
        lo = order[nearest_age_index(self.pool_ages - lag, sorted_ages)]
        hi = order[nearest_age_index(self.pool_ages + lag, sorted_ages)]
        return lo, hi

    def _block_rows(self, members: np.ndarray, block: tuple[float, bool]) -> np.ndarray:
        """Raw difference rows for one (lag, order) block, before centring or weighting.

        Neighbours clamp at the archive's ends. A first difference there is rescaled to the
        nominal interval; a second difference with a clamped arm is dropped, as is a member
        whose two neighbours coincide.
        """
        lag, second = block
        lo, hi = self._neighbours[lag]
        a, b = lo[members], hi[members]
        keep = a != b
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
        rows = self._block_rows(np.arange(len(self.pool)), block)
        return float(np.sqrt((rows ** 2).mean())) if len(rows) else 0.0

    def _tendency_rows(self, members: np.ndarray) -> np.ndarray:
        """The stacked tendency blocks ``T`` for the selected members.

        Each block is centred on the rows it keeps and weighted by its ``s_i``. A block whose
        rows are all the same, as when a lag outreaches the archive and every member clamps
        to its two ends, has no direction to add and is left out: centring it would leave
        only rounding noise for the trace rescaling to amplify.
        """
        rows = []
        for block in self._blocks:
            scale = self.tendency_scale[block]
            if scale <= 0.0:
                continue
            tend = self._block_rows(members, block)
            if len(tend) and np.ptp(tend, axis=0).any():
                rows.append(scale * (tend - tend.mean(axis=0)))
        return np.vstack(rows) if rows else np.empty((0, self.pool.shape[1]))

    @staticmethod
    def _whitened_trace(rows: np.ndarray, gain: _HybridSweepGain) -> float:
        """``sum over rows of row[gather]^2 / r``: the rows' trace in whitened observation space."""
        return float(((rows[:, gain.gather] ** 2) / gain.r_diag[None, :]).sum())

    def deviations(self, selected: np.ndarray,
                   gain: _HybridSweepGain) -> tuple[np.ndarray, np.ndarray]:
        """``(analog mean, Z)`` for the selected members, ``Z`` as in the module docstring.

        Without tendency rows to share with (``w = 0``, or none survive) ``Z`` is ``X'``.
        """
        members = self.pool[selected]
        mu = members.mean(axis=0)
        base = members - mu                                   # (k, D)
        if self.tendency_share == 0.0:
            return mu, base
        tend = self._tendency_rows(selected)
        t_tend = self._whitened_trace(tend, gain)
        if t_tend <= 0.0:
            return mu, base
        w = self.tendency_share
        rows = [np.sqrt(w * self._whitened_trace(base, gain) / t_tend) * tend]
        if w < 1.0:
            rows.insert(0, np.sqrt(1.0 - w) * base)
        return mu, np.vstack(rows)

    def _analog_block(self, dev: np.ndarray, gain: _HybridSweepGain) -> WhitenedBlock:
        """The tapered analog covariance factorized against the network.

        Normalised by the ``k`` members, not the rows: tendency rows are extra directions on
        a ``k``-member ensemble, not extra members.
        """
        h_dev = dev[:, gain.gather]                           # (n_dev, m)
        P_obs = (dev.T @ h_dev) / (self.k - 1.0)              # B_a H^T, never B_a itself
        S_obs = (h_dev.T @ h_dev) / (self.k - 1.0)            # H B_a H^T
        if gain.taper is not None:
            P_obs = P_obs * gain.taper[0]
            S_obs = S_obs * gain.taper[1]
        return whitened_block(P_obs, S_obs, gain.r_diag)

    def prepare_sweep(self, gather: np.ndarray, r_diag: np.ndarray,
                      b_scales: np.ndarray) -> _HybridSweepGain:
        """Factorize the static gain and stage everything selection needs."""
        g = np.asarray(gather)
        n_cells = self.shape[1] * self.shape[2]
        static = whitened_block(self.B[:, g], self.B[np.ix_(g, g)], r_diag)
        return _HybridSweepGain(
            gather=g, r_diag=np.asarray(r_diag, dtype=np.float64),
            b_scales=np.asarray(b_scales, dtype=np.float64),
            static=static, pool_at_obs=self.pool[:, g], obs_channel=g // n_cells,
            taper=taper_obs_blocks(self.lats, self.lons, g, **self.taper_meta))

    def select(self, gain: _HybridSweepGain, y_anom: np.ndarray) -> np.ndarray:
        """Pool indices of the analog ensemble for one observation vector.

        Public so a driver can record which states were chosen.
        """
        if self.selection == ANALOG_EVIDENCE:
            # Reuses the static factorization, so no second solve.
            return evidence_indices(gain.pool_at_obs, y_anom, gain.r_diag, gain.static,
                                    self.k, kappa=self.kappa)
        per_chan = self.selection == ANALOG_CORRELATION_PERCHAN
        return correlation_indices(gain.pool_at_obs, y_anom, self.k,
                                   channel=gain.obs_channel if per_chan else None)

    def apply_sweep(self, gain: _HybridSweepGain, y_anom: np.ndarray,
                    background_anom: np.ndarray, *,
                    mean_only: bool = False) -> list[AnalysisResult]:
        """Analysis at every ``b_scale`` for one innovation, one result per ``b_scale``.

        ``b_scale`` scales both covariances. A gain at zero weight is never built.
        """
        g = gain.gather
        mu, dev = self.deviations(self.select(gain, y_anom), gain)
        w = self.hybrid_w
        gains = []
        if w > 0.0:
            gains.append((w, self._analog_block(dev, gain)))
        if w < 1.0:
            gains.append((1.0 - w, gain.static))

        x_b = np.asarray(background_anom, dtype=np.float64).ravel() + mu
        d = np.asarray(y_anom, dtype=np.float64) - x_b[g]
        h_dev = dev[:, g].T                                   # (m, n_dev)
        n_cells = self.shape[1] * self.shape[2]
        out = []
        for b in gain.b_scales:
            x_a = x_b + sum(weight * mean_gain_apply(block, b, d) for weight, block in gains)
            if mean_only:
                out.append(AnalysisResult(mean_anom=x_a.reshape(self.shape)))
                continue
            post = dev.T - sum(weight * sqrt_gain_apply(block, b, h_dev)
                               for weight, block in gains)
            # Normalised by k as P_obs is; every block is centred, so no mean is removed.
            scale = b / (self.k - 1.0)
            out.append(AnalysisResult(
                mean_anom=x_a.reshape(self.shape),
                posterior_var=(scale * (post ** 2).sum(axis=1)).reshape(self.shape),
                posterior_cross_var=(scale * (post[:n_cells] * post[n_cells:]).sum(axis=1)
                                     ).reshape(self.shape[1:])))
        return out


def make_hgaoenkf(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray, *,
    k: int, hybrid_w: float, selection: str = ANALOG_EVIDENCE, kappa: float = 0.0,
    stack: TendencyStack | None = None, tendency_share: float = 0.0,
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
                        selection=selection, kappa=kappa, stack=stack,
                        tendency_share=tendency_share)

    return factory
