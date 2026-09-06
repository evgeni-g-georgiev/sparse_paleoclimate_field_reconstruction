"""Reconstruction of a full age axis from a real proxy network.

The evaluation lanes score an estimator against a truth. This produces the reconstruction
itself: one analysis per age, with no truth, no split and no metrics. What rides alongside
the fields is what a reader needs to interpret them, since the network varies from nothing
to the whole archive of sites: the count assimilated at each age, and innovation
diagnostics that say whether the assumed observation error matches the residuals.

Observations enter in anomaly space against each site's own climatology, and the state in
anomaly space against the prior's, so a proxy's offset from the model cancels. The
background is climatological at every age. Where the estimator draws an analog ensemble
that ensemble's mean is the background the analysis updates, so a per-age background would
hand the analysis the archive's own state at the target age, which is what an exclusion
band exists to prevent.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from paleoreco.data import VARS
from paleoreco.assim.background import temporal_structure_function
from paleoreco.assim.experiments import (
    ESTIMATOR_3DVAR,
    MethodFactory,
    _age_step,
    _max_block_lag,
    _report_progress,
)
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.observations import (
    TEMPORAL_DEFLATE,
    apply_temporal_error,
    observations_at_age,
    representativeness_variance,
    sample_block_centres,
    temporal_terms,
)
from paleoreco.assim.priors import build_prior
from paleoreco.assim.threedvar import ThreeDVar

LANE_RECONSTRUCTION = "reconstruction"
# The amplitude at which the gain vanishes, so the analysis returns its own background. It
# rides along with every sweep rather than being solved for separately, which is what makes
# the prior mean and the analysis share one ensemble and so one innovation.
PRIOR_LIMIT = 1e-9
NO_ANALOG = -1


def _network_at_age(long: pd.DataFrame, age: int, lats: np.ndarray, lons: np.ndarray,
                    safe_flat: np.ndarray, n_cells: int, rep_lookup: np.ndarray,
                    min_obs: int) -> dict | None:
    """Assimilable observations at one age, or ``None`` where the network is too thin.

    An observation is usable where its cell survives the prior's mask, its stated error
    variance is positive, and its site has a climatology to take an anomaly against. A
    missing climatology would otherwise carry a NaN through the gain to the whole field.
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
        "sse": o["sse"][keep].astype(np.float64) + rep_lookup[g // n_cells],
        "lag": np.abs(o["age"][keep] - o["centre"][keep]).astype(np.float64),
        "n_sites": int(len(np.unique(o["site"][keep]))),
    }


def _desroziers(d_b: np.ndarray, d_a: np.ndarray, r: np.ndarray) -> float:
    """Desroziers (2005) ratio: ``E[d_a d_b'] = R`` holds where the assumed R is right.

    Needs no truth, so with the innovation chi-squares it is the only check on the amplitude
    balance a reconstruction from real proxies has.
    """
    return float((d_a * d_b).mean() / r.mean())


def _reduced_chi2(d: np.ndarray, r: np.ndarray) -> float:
    """Innovation whitened by R, which is the norm the gain actually shrinks.

    The unweighted residual is not monotone in the background amplitude once R varies
    between observations, so a per-observation weighting is what makes the two columns
    comparable across the sweep.
    """
    return float(((d ** 2) / r).mean())


def _json_scalar(value):
    """A numpy scalar as something ``json.dump`` will emit as valid JSON.

    A column that does not apply to an estimator arrives as NaN, which serialises to a bare
    ``NaN`` token that only Python reads back.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value = float(value)
    return value if np.isfinite(value) else None


def _write_reconstruction_artifacts(out_dir: str, npz_arrays: dict, config: dict) -> None:
    """Persist the fields and the configuration that produced them.

    No metrics CSV: nothing here is scored, so there are no rows to append.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{LANE_RECONSTRUCTION}_fields.npz"),
                        **npz_arrays)
    with open(os.path.join(out_dir, f"{LANE_RECONSTRUCTION}_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def run_reconstruction(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    make_method: MethodFactory | None = None, b_scales: tuple[float, ...] = (1.0,),
    temporal_mode: str = TEMPORAL_DEFLATE,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
    min_obs: int = 1, progress_every: int | None = None,
) -> dict:
    """Assimilate the real network at every age and write the reconstruction.

    The prior, its climatology and the temporal structure function all come from every age:
    there is no held-out model state here, so the operator and the covariance describe the
    same archive the analysis draws on. ``b_scales`` are reported side by side rather than
    selected between, since nothing here can score them.

    Ages whose network is thinner than ``min_obs`` carry the prior instead of an analysis,
    flagged so a reader can tell a reconstruction from a climatology.
    """
    if min_obs < 1:
        raise ValueError(f"an analysis needs at least one observation; got {min_obs}")
    ages_i = np.asarray(ages, dtype=np.int64)
    step_yr = _age_step(ages_i)
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    b_scales = tuple(float(b) for b in b_scales)
    n_b, n_ages = len(b_scales), len(ages_i)

    prior_idx = np.arange(n_ages)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    safe_valid = prior.safe_valid
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    tv = ThreeDVar(prior.B, shape) if make_method is None else make_method(prior, shape)
    zero_bg = np.zeros(int(np.prod(shape)))
    prior_var = tv.diagB.reshape(shape)
    # The prior's own cross-channel covariance at each cell, which is what an age with no
    # observations reports in place of a posterior one.
    cell = np.arange(n_cells)
    prior_cross = prior.B[cell, cell + n_cells].reshape(shape[1:])

    long = sample_block_centres(long)
    cell_all = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons)
    rep = representativeness_variance(long, cell_all)
    rep_lookup = np.array([rep.get(v, 0.0) for v in VARS])
    S, prior_var_cell = temporal_structure_function(
        cube, prior_idx, max_lag=_max_block_lag(long, step_yr))

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
            # Selection reads the observations, so with none there is no ensemble to draw
            # and no analysis to form. This branch has to come before the gain is prepared:
            # an empty network factorizes without complaint and would return a plausible
            # field built on whichever candidates the ranking left first.
            prior_only[i] = True
            for bj, b in enumerate(b_scales):
                post_var[bj, i] = b * prior_var
                post_cross_var[bj, i] = b * prior_cross
            continue

        g = geom["gather"]
        rho, resid = temporal_terms(S, prior_var_cell, g, geom["lag"], step_yr)
        yv, r = apply_temporal_error(geom["y_anom"], geom["sse"], rho, resid, temporal_mode)
        # The age reaches the estimator because the archive spans it, so an analog step
        # could otherwise select the simulation's own state at the target.
        gain = tv.prepare_sweep(g, r, sweep, age=int(age))
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
            _report_progress("reconstruction age", i + 1, n_ages, t0)

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
    # Both ride along only where the estimator produced one, so a reader cannot mistake a
    # filled default for a reported quantity.
    if has_cross:
        npz_arrays["post_cross_var"] = post_cross_var
    if analog_index is not None:
        npz_arrays["analog_index"] = analog_index

    config = {
        "lane": LANE_RECONSTRUCTION, "space": "pixel",
        "localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
        "alpha": alpha, "b_scales": list(b_scales), "prior_limit": PRIOR_LIMIT,
        "prior_meta": prior.meta, "estimator": estimator,
        "temporal_mode": temporal_mode, "min_obs": int(min_obs),
        "background": "climatological", "step_yr": step_yr,
        "max_block_lag_steps": _max_block_lag(long, step_yr),
        "rep_var_full": {v: float(rep.get(v, 0.0)) for v in VARS},
        "n_ages": int(n_ages),
        "n_prior_only_ages": int(prior_only.sum()),
        "prior_only_ages": [int(a) for a in ages_i[prior_only]],
        # An observation age off the archive's axis is never reached by the loop, and
        # nothing in the fields would say so.
        "n_obs_ages_off_axis": int(len(np.setdiff1d(long["age"].unique(), ages_i))),
    }
    if method_cols:
        config.update({key: _json_scalar(v) for key, v in method_cols.items()})
    _write_reconstruction_artifacts(out_dir, npz_arrays, config)
    return config
