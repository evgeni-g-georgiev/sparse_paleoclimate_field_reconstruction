"""Monte Carlo proxy-resampling ensemble for offline reconstruction.

The verification protocol of Hakim et al. (2016): with the background covariance
learned once, assimilate a random fraction of proxy sites and withhold the rest,
repeat over many independent draws, and read the spread across draws as the
reconstruction's sensitivity to which proxies were used. Whole sites move
together, so a site's every age and both channels are assimilated or withheld as
one unit.

The harness is method-agnostic: :func:`assimilate_withheld` drives any
:class:`~paleoreco.assim.method.Method`, so a deterministic 3DVar and an ensemble
filter run through the same loop. ``AnalysisResult`` already carries posterior
samples, so a method-internal ensemble composes with this outer resampling
ensemble without changing the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from paleoreco.assim.background import background_state
from paleoreco.assim.innovation import obs_cell_index, r_diagonal
from paleoreco.assim.method import Method, Observations
from paleoreco.assim.observations import observations_at_age
from paleoreco.assim.threedvar import ThreeDVar

BACKGROUNDS = ("climatological", "per_age")


# ---------------------------------------------------------------------------
# Resampling geometry.
# ---------------------------------------------------------------------------
def random_site_partitions(
    sites: np.ndarray, n_realizations: int, assim_frac: float = 0.75, seed: int = 0
) -> list[np.ndarray]:
    """Withheld site ids for each realization: independent draws, not a partition.

    Each realization withholds the same fraction ``1 - assim_frac`` of sites, but
    the draws overlap, so a site is withheld in some realizations and assimilated
    in others. This is the resampling that gives the ensemble its spread.
    """
    sites = np.asarray(sites)
    n_withheld = int(round((1.0 - assim_frac) * len(sites)))
    rng = np.random.default_rng(seed)
    return [rng.choice(sites, size=n_withheld, replace=False) for _ in range(n_realizations)]


# ---------------------------------------------------------------------------
# Field ensemble accumulation.
# ---------------------------------------------------------------------------
@dataclass
class FieldEnsembleAccumulator:
    """Running mean and spread of the reconstructed field over realizations.

    Welford accumulation keeps memory at one mean and one sum-of-squares array
    rather than the full stack of realizations. ``finalize`` returns the
    ensemble mean and the across-realization standard deviation (ddof 1).
    """

    n: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros_like(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else np.zeros_like(self.mean)
        return self.mean, std


# ---------------------------------------------------------------------------
# One realization's assimilation.
# ---------------------------------------------------------------------------
@dataclass
class WithheldSweep:
    """One realization's withheld predictions and reconstructed fields.

    ``actual``/``site``/``channel``/``age``/``sse``/``post_var`` describe the
    withheld observations and are shared across backgrounds (``post_var``, the
    analysis variance at the withheld cell, does not depend on the first guess);
    ``pred``, ``prior_pred`` and ``field`` are keyed by background. ``prior_pred``
    is H applied to the background (the first guess at the withheld cell), the
    departure baseline the analysis is compared against. ``field`` is
    ``(n_obs_ages, n_chan, n_lat, n_lon)`` aligned to ``obs_ages`` and is filled
    only when requested.
    """

    actual: np.ndarray
    site: np.ndarray
    channel: np.ndarray
    age: np.ndarray
    sse: np.ndarray
    post_var: np.ndarray
    pred: dict[str, np.ndarray]
    prior_pred: dict[str, np.ndarray] = field(default_factory=dict)
    field: dict[str, np.ndarray] = field(default_factory=dict)


def assimilate_withheld(
    method: Method,
    long: pd.DataFrame,
    obs_ages: np.ndarray,
    ages: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    safe_flat: np.ndarray,
    clim_flat: np.ndarray,
    cube: np.ndarray,
    clim_mean: np.ndarray,
    withheld: np.ndarray,
    *,
    backgrounds: tuple[str, ...] = BACKGROUNDS,
    want_field: bool = True,
) -> WithheldSweep:
    """Assimilate the kept sites age by age, predicting the withheld sites.

    ``long`` carries the per-site climatology ``my``: observations enter in anomaly
    space ``y - my`` so the proxy-vs-model offset cancels. At each age the kept
    (assimilated) and withheld sites are the present rows not in / in ``withheld``;
    an age with no kept or no withheld observation contributes no prediction but
    still yields a reconstructed field from whatever was assimilated.
    """
    shape = (len(clim_mean), len(lats), len(lons))
    n_chan, n_lat, n_lon = shape
    wset = set(np.asarray(withheld).tolist())

    actual, site, channel, age_col, sse_col, post_var_col = [], [], [], [], [], []
    pred = {bg: [] for bg in backgrounds}
    prior_pred = {bg: [] for bg in backgrounds}
    fields = {bg: np.zeros((len(obs_ages), *shape)) for bg in backgrounds} if want_field else {}

    for ai_obs, age in enumerate(obs_ages):
        o = observations_at_age(long, int(age))
        gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
        keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
        is_w = np.array([s in wset for s in o["site"]])
        kept = keep & ~is_w
        wkeep = keep & is_w

        ai = int(np.searchsorted(ages, age))
        bg_anom = {bg: background_state(cube, clim_mean, ai, bg).astype(np.float64) - clim_flat
                   for bg in backgrounds}

        if kept.sum() > 0:
            y_anom = (o["y"][kept] - o["my"][kept]).astype(np.float64)
            obs = Observations(gather=gather[kept], y_anom=y_anom,
                               sse=o["sse"][kept].astype(np.float64))
            # One gain factorization shared across backgrounds (same network).
            res_list = method.analyze_many(obs, [bg_anom[bg] for bg in backgrounds])
            results = dict(zip(backgrounds, res_list))
        else:
            results = None

        if want_field:
            for bg in backgrounds:
                fields[bg][ai_obs] = (results[bg].mean_anom if results is not None
                                      else bg_anom[bg].reshape(shape))

        if results is None or wkeep.sum() == 0:
            continue
        gw = gather[wkeep]
        actual.append((o["y"][wkeep] - o["my"][wkeep]).astype(np.float64))
        site.append(o["site"][wkeep])
        channel.append(gw // (n_lat * n_lon))
        age_col.append(np.full(int(wkeep.sum()), int(age)))
        sse_col.append(o["sse"][wkeep].astype(np.float64))
        post_var_col.append(results[backgrounds[0]].posterior_var.ravel()[gw])
        for bg in backgrounds:
            pred[bg].append(results[bg].predict_obs(gw))
            prior_pred[bg].append(bg_anom[bg][gw])

    cat = lambda parts: np.concatenate(parts) if parts else np.array([])
    return WithheldSweep(
        actual=cat(actual), site=cat(site), channel=cat(channel), age=cat(age_col),
        sse=cat(sse_col), post_var=cat(post_var_col),
        pred={bg: cat(pred[bg]) for bg in backgrounds},
        prior_pred={bg: cat(prior_pred[bg]) for bg in backgrounds},
        field={bg: fields[bg] for bg in backgrounds} if want_field else {},
    )


# ---------------------------------------------------------------------------
# One realization's assimilation across a B-amplitude sweep.
# ---------------------------------------------------------------------------
@dataclass
class WithheldSweepB:
    """One realization's withheld predictions and fields across a b_scale sweep.

    The withheld-observation descriptors are background- and b_scale-independent;
    ``post_var`` and ``pred`` carry a leading b_scale axis, ``prior_pred`` is the
    background-only first guess. ``rep`` is the representativeness variance at the
    withheld cell, kept because it is absent from the assimilated R yet needed for
    the predictive variance.
    """

    actual: np.ndarray              # (n_w,)
    site: np.ndarray
    channel: np.ndarray
    age: np.ndarray
    sse: np.ndarray
    rep: np.ndarray
    b_scales: np.ndarray            # (n_k,)
    post_var: np.ndarray            # (n_k, n_w)
    pred: dict[str, np.ndarray]     # bg -> (n_k, n_w)
    prior_pred: dict[str, np.ndarray] = field(default_factory=dict)  # bg -> (n_w,)
    field: dict[str, np.ndarray] = field(default_factory=dict)       # bg -> (n_k, n_obs_ages, *shape)


def assimilate_withheld_sweep(
    tv: ThreeDVar,
    long: pd.DataFrame,
    obs_ages: np.ndarray,
    ages: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    safe_flat: np.ndarray,
    clim_flat: np.ndarray,
    cube: np.ndarray,
    clim_mean: np.ndarray,
    withheld: np.ndarray,
    rep_flat: np.ndarray,
    c: np.ndarray,
    b_scales: np.ndarray,
    *,
    backgrounds: tuple[str, ...] = BACKGROUNDS,
    want_field: bool = True,
) -> WithheldSweepB:
    """Assimilate the kept sites age by age across every ``b_scale``, predicting withheld.

    One eig-shared factorization per age serves all ``b_scales``; R is the
    augmented ``sse + c*rep_var`` at the kept cells. Mirrors
    :func:`assimilate_withheld` but is tied to :class:`ThreeDVar` for the sweep.
    """
    shape = (len(clim_mean), len(lats), len(lons))
    n_lat, n_lon = shape[1:]
    n_cells = n_lat * n_lon
    n_k = len(b_scales)
    c = np.asarray(c, dtype=np.float64)
    wset = set(np.asarray(withheld).tolist())

    actual, site, channel, age_col, sse_col, rep_col, post_var_col = [], [], [], [], [], [], []
    pred = {bg: [] for bg in backgrounds}
    prior_pred = {bg: [] for bg in backgrounds}
    fields = {bg: np.zeros((n_k, len(obs_ages), *shape)) for bg in backgrounds} if want_field else {}

    for ai_obs, age in enumerate(obs_ages):
        o = observations_at_age(long, int(age))
        gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
        keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
        is_w = np.array([s in wset for s in o["site"]])
        kept = keep & ~is_w
        wkeep = keep & is_w

        ai = int(np.searchsorted(ages, age))
        bg_anom = {bg: background_state(cube, clim_mean, ai, bg).astype(np.float64) - clim_flat
                   for bg in backgrounds}

        if kept.sum() > 0:
            y_anom = (o["y"][kept] - o["my"][kept]).astype(np.float64)
            r_diag = r_diagonal(gather[kept], o["sse"][kept].astype(np.float64),
                                rep_flat, c, n_cells)
            gain = tv.prepare_sweep(gather[kept], r_diag, b_scales)
            results = {bg: tv.apply_sweep(gain, y_anom, bg_anom[bg]) for bg in backgrounds}
        else:
            results = None

        if want_field:
            for bg in backgrounds:
                if results is not None:
                    for ki in range(n_k):
                        fields[bg][ki, ai_obs] = results[bg][ki].mean_anom
                else:
                    fields[bg][:, ai_obs] = bg_anom[bg].reshape(shape)

        if results is None or wkeep.sum() == 0:
            continue
        gw = gather[wkeep]
        chan_w = gw // n_cells
        actual.append((o["y"][wkeep] - o["my"][wkeep]).astype(np.float64))
        site.append(o["site"][wkeep])
        channel.append(chan_w)
        age_col.append(np.full(int(wkeep.sum()), int(age)))
        sse_col.append(o["sse"][wkeep].astype(np.float64))
        rep_col.append(c[chan_w] * rep_flat[gw])
        post_var_col.append(np.stack([results[backgrounds[0]][ki].posterior_var.ravel()[gw]
                                      for ki in range(n_k)]))
        for bg in backgrounds:
            pred[bg].append(np.stack([results[bg][ki].predict_obs(gw) for ki in range(n_k)]))
            prior_pred[bg].append(bg_anom[bg][gw])

    cat = lambda parts: np.concatenate(parts) if parts else np.array([])
    cat_k = lambda parts: np.concatenate(parts, axis=1) if parts else np.zeros((n_k, 0))
    return WithheldSweepB(
        actual=cat(actual), site=cat(site), channel=cat(channel), age=cat(age_col),
        sse=cat(sse_col), rep=cat(rep_col), b_scales=np.asarray(b_scales, dtype=np.float64),
        post_var=cat_k(post_var_col),
        pred={bg: cat_k(pred[bg]) for bg in backgrounds},
        prior_pred={bg: cat(prior_pred[bg]) for bg in backgrounds},
        field={bg: fields[bg] for bg in backgrounds} if want_field else {},
    )
