"""The reconstruction: one analysis per age of the archive, from the real proxy network.

There is no truth, split or metric here. Alongside the fields the output records how many
observations each age assimilated, and innovation diagnostics that check the assumed
observation error against the residuals.

Observations are anomalies about each site's own climatology and the state about the
prior's, so a proxy's offset from the model cancels. The background is climatological at
every age; an analog estimator updates its ensemble mean instead. The observation model is
the full one: each sample's stated error, the spatial term estimated from every site, and
the temporal term from a structure function over every state.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from paleoreco.data import VARS
from paleoreco.data.splits import age_step
from paleoreco.assim.background import temporal_structure_function
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.lanes import report_progress
from paleoreco.assim.method import MethodFactory
from paleoreco.assim.observations import (
    FULL_MODEL,
    deflate,
    max_block_lag,
    observations_at_age,
    representativeness_variance,
    sample_block_centres,
    temporal_terms,
)
from paleoreco.assim.priors import build_prior
from paleoreco.assim.threedvar import ThreeDVar

LANE_RECONSTRUCTION = "reconstruction"
# An amplitude small enough that the analysis returns its own background, added to every
# sweep so the prior mean comes from the same ensemble as the analysis.
PRIOR_LIMIT = 1e-9
NO_ANALOG = -1


def _network_at_age(long: pd.DataFrame, age: int, lats: np.ndarray, lons: np.ndarray,
                    safe_flat: np.ndarray, n_cells: int, rep_lookup: np.ndarray,
                    min_obs: int) -> dict | None:
    """Assimilable observations at one age, or ``None`` where the network is too thin.

    An observation is usable where its cell is valid, its error variance is positive and
    its site has a climatology; a missing climatology would spread NaN through the field.
    """
    o = observations_at_age(long, int(age))
    if not len(o.get("age", [])):
        return None
    gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
    keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
    if int(keep.sum()) < min_obs:
        return None
    g = gather[keep]
    return {
        "gather": g,
        "y_anom": (o["y"][keep] - o["my"][keep]).astype(np.float64),
        # A point proxy does not measure its cell's mean, so R carries that scatter before
        # the temporal term is applied to the pair.
        "r": o["sse"][keep].astype(np.float64) + rep_lookup[g // n_cells],
        "lag": np.abs(o["age"][keep] - o["centre"][keep]).astype(np.float64),
        "n_sites": int(len(np.unique(o["site"][keep]))),
    }


def _desroziers(d_b: np.ndarray, d_a: np.ndarray, r: np.ndarray) -> float:
    """Desroziers (2005) ratio: ``E[d_a d_b'] = R`` holds where the assumed R is right.

    Needs no truth, so it can check a reconstruction from real proxies.
    """
    return float((d_a * d_b).mean() / r.mean())


def _reduced_chi2(d: np.ndarray, r: np.ndarray) -> float:
    """Innovation whitened by R, which is the norm the gain actually shrinks.

    Whitening makes the value comparable across the amplitude sweep when R varies.
    """
    return float(((d ** 2) / r).mean())


def run_reconstruction(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *, taper: dict,
    make_method: MethodFactory | None = None, b_scales: tuple[float, ...] = (1.0,),
    record: dict | None = None, min_obs: int = 1, progress_every: int | None = None,
) -> dict:
    """Assimilate the real network at every age and write the reconstruction.

    ``make_method`` defaults to 3DVar. The prior, climatology and temporal structure
    function use every age, since no model state is held out. ``b_scales`` are published
    side by side, since nothing here can choose between them. Ages with fewer than
    ``min_obs`` observations carry the climatology, with ``b_scale`` times the prior
    variance, and are flagged in ``prior_only``. ``record`` describes the estimator in the
    config: its name and settings, which the fields cannot say.
    """
    if min_obs < 1:
        raise ValueError(f"an analysis needs at least one observation; got {min_obs}")
    ages_i = np.asarray(ages, dtype=np.int64)
    step_yr = age_step(ages_i)
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    b_scales = tuple(float(b) for b in b_scales)
    n_b, n_ages = len(b_scales), len(ages_i)

    prior_idx = np.arange(n_ages)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid, **taper)
    safe_valid = prior.safe_valid
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    tv = ThreeDVar(prior.B, shape) if make_method is None else make_method(prior, shape)
    zero_bg = np.zeros(int(np.prod(shape)))
    prior_var = np.diag(prior.B).reshape(shape)
    # The prior's own cross-channel covariance at each cell, which is what an age with no
    # observations reports in place of a posterior one.
    cell = np.arange(n_cells)
    prior_cross = prior.B[cell, cell + n_cells].reshape(shape[1:])

    long = sample_block_centres(long)
    cell_all = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons)
    rep = representativeness_variance(long, cell_all)
    rep_lookup = np.array([rep.get(v, 0.0) for v in VARS])
    max_lag = max_block_lag(long, step_yr)
    S, prior_var_cell = temporal_structure_function(cube, prior_idx, max_lag=max_lag)

    sweep = np.asarray(b_scales + (PRIOR_LIMIT,))
    mean_anom = np.zeros((n_b, n_ages, *shape), dtype=np.float32)
    post_var = np.zeros((n_b, n_ages, *shape), dtype=np.float32)
    post_cross_var = np.zeros((n_b, n_ages, *shape[1:]), dtype=np.float32)
    prior_mean_anom = np.zeros((n_ages, *shape), dtype=np.float32)
    n_obs = np.zeros(n_ages, dtype=np.int64)
    n_sites = np.zeros(n_ages, dtype=np.int64)
    prior_only = np.zeros(n_ages, dtype=bool)
    desroziers_r = np.full((n_b, n_ages), np.nan)
    chi2_an = np.full((n_b, n_ages), np.nan)
    chi2_bg = np.full(n_ages, np.nan)
    innov_r_mean = np.full(n_ages, np.nan)
    analog_index = None
    has_cross = False

    t0 = time.time()
    for i, age in enumerate(ages_i):
        geom = _network_at_age(long, int(age), lats, lons, safe_flat, n_cells,
                               rep_lookup, min_obs)
        if geom is None:
            # Must come before the gain is prepared: an empty network would still return a
            # plausible field.
            prior_only[i] = True
            for bj, b in enumerate(b_scales):
                post_var[bj, i] = b * prior_var
                post_cross_var[bj, i] = b * prior_cross
            continue

        g = geom["gather"]
        rho, resid = temporal_terms(S, prior_var_cell, g, geom["lag"], step_yr)
        yv, r = deflate(geom["y_anom"], geom["r"], rho, resid)
        gain = tv.prepare_sweep(g, r, sweep)
        res = tv.apply_sweep(gain, yv, zero_bg)

        d_b = yv - res[-1].predict_obs(g)
        prior_mean_anom[i] = res[-1].mean_anom
        n_obs[i], n_sites[i] = len(g), geom["n_sites"]
        chi2_bg[i], innov_r_mean[i] = _reduced_chi2(d_b, r), float(r.mean())
        for bj in range(n_b):
            d_a = yv - res[bj].predict_obs(g)
            mean_anom[bj, i] = res[bj].mean_anom
            post_var[bj, i] = res[bj].posterior_var
            if res[bj].posterior_cross_var is not None:
                has_cross = True
                post_cross_var[bj, i] = res[bj].posterior_cross_var
            desroziers_r[bj, i] = _desroziers(d_b, d_a, r)
            chi2_an[bj, i] = _reduced_chi2(d_a, r)

        if hasattr(tv, "select"):
            drawn = tv.select(gain, yv)
            if analog_index is None:
                analog_index = np.full((n_ages, len(drawn)), NO_ANALOG, dtype=np.int64)
            analog_index[i] = drawn
        if progress_every and (i + 1) % progress_every == 0:
            report_progress("reconstruction age", i + 1, n_ages, t0)

    npz_arrays = {
        "ages": ages_i, "lats": np.asarray(lats), "lons": np.asarray(lons),
        "safe_valid": safe_valid, "clim_mean": prior.clim_mean.astype(np.float64),
        "prior_var": prior_var, "b_scales": np.asarray(b_scales),
        "mean_anom": mean_anom, "post_var": post_var,
        "prior_mean_anom": prior_mean_anom,
        "n_obs": n_obs, "n_sites": n_sites, "prior_only": prior_only,
        "desroziers_r": desroziers_r, "chi2_bg": chi2_bg,
        "chi2_an": chi2_an, "innov_r_mean": innov_r_mean,
    }
    # Written only where the estimator produced them.
    if has_cross:
        npz_arrays["post_cross_var"] = post_cross_var
    if analog_index is not None:
        npz_arrays["analog_index"] = analog_index

    config = {
        "lane": LANE_RECONSTRUCTION, **(record or {}), "taper": dict(taper),
        "b_scales": list(b_scales), "prior_limit": PRIOR_LIMIT, "prior_meta": prior.meta,
        "observation_model": asdict(FULL_MODEL), "min_obs": int(min_obs),
        "background": "climatological", "step_yr": step_yr, "max_block_lag_steps": max_lag,
        "rep_var_full": {v: float(rep.get(v, 0.0)) for v in VARS},
        "n_ages": int(n_ages),
        "n_prior_only_ages": int(prior_only.sum()),
        "prior_only_ages": [int(a) for a in ages_i[prior_only]],
        # Observation ages off the archive's axis, which the loop never reaches.
        "n_obs_ages_off_axis": int(len(np.setdiff1d(long["age"].unique(), ages_i))),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{LANE_RECONSTRUCTION}_fields.npz"),
                        **npz_arrays)
    with open(os.path.join(out_dir, f"{LANE_RECONSTRUCTION}_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config
