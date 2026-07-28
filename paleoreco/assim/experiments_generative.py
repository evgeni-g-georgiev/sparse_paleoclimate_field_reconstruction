"""Experiment runners for the generative (score-based) prior.

The same two evaluation lanes as :mod:`paleoreco.assim.experiments`, but the
Gaussian-gain analysis is replaced by guided posterior sampling from a
:class:`paleoreco.assim.generative.GuidedSampler`. The observation geometry,
truth draws, site folds, representativeness variance, metric rows, and tidy CSV
schema are reused unchanged, so the generative method drops into notebook 09's
cross-method comparison with no special-casing.

``gamma`` (the likelihood inflation) is the tuned operating point, the analogue
of 3DVar ``b_scale``: swept on the held-out selection split over a reduced
workload, then the winner is run once on the full test split. Rows carry
``method="generative"``, ``space="generative"``, and ``b_scale=gamma`` so the
selection rule (:func:`select_best_config`) and the reported test rows match the
classical lanes. Calibration is scored both as a Gaussian summary of the
ensemble (directly comparable to 3DVar) and natively over the ensemble.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from paleoreco.data import VARS
from paleoreco.data.splits import chronological_half_split
from paleoreco.assim.observations import observations_at_age, representativeness_variance
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.priors import build_prior
from paleoreco.eval import calibration
from paleoreco.assim.experiments import (
    _obs_geometry, _draw_usable_ages, _naive_geometry, _naive_apply,
    _naive_obs_predictions, _pad_obs, _skill_rows, _ssim_rows, _field_calibration_rows,
    _withholding_rows, _calibration_rows, _obs_channel_groups, _append_csv,
    _report_progress, select_best_config, _NAN_REG, LANE_PPE, SEL_TOL,
)

GAMMA_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
_GEN = {"method": "generative", "space": "generative", **_NAN_REG}


# ---------------------------------------------------------------------------
# Native ensemble calibration (over and above the Gaussian-summary rows).
# ---------------------------------------------------------------------------
def _native_calibration_rows(truth: np.ndarray, ens: np.ndarray, base: dict) -> list[dict]:
    """ECR, sharpness, native CRPS/coverage over a ``(K, C, N)`` ensemble, pooled and per channel.

    ``truth`` is ``(C, N)`` over valid cells; ``ens`` is ``(K, C, N)``. Native CRPS
    and coverage read the samples directly, so a non-Gaussian spread is scored on
    its own terms; ECR and sharpness read the two moments.
    """
    rows = []
    channels = [("pooled", None)] + [(name, c) for c, name in enumerate(VARS)]
    for chan_name, c in channels:
        t = truth.ravel() if c is None else truth[c]
        e = ens.reshape(ens.shape[0], -1) if c is None else ens[:, c]
        mean, var = e.mean(axis=0), e.var(axis=0, ddof=1)
        for metric, value in (
            ("ecr", calibration.ecr(t, mean, var)),
            ("sharpness", calibration.sharpness(var)),
            ("crps_ens", float(np.mean(calibration.crps_ensemble(t, e)))),
            ("coverage90_ens", calibration.coverage_ensemble(t, e, 0.9)),
        ):
            rows.append({**base, "do_event": "all", "channel": chan_name,
                         "metric": metric, "value": value})
    return rows


# ---------------------------------------------------------------------------
# Same-model pseudo-proxy lane.
# ---------------------------------------------------------------------------
def run_ppe_generative(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long, out_dir: str, *, sampler,
    gamma_grid: tuple[float, ...] = GAMMA_GRID, n_samples: int = 16,
    n_samples_select: int = 8, n_shapes: int = 5, n_select: int = 4,
    n_noise: int = 5, truth_stride: int = 10, sel_subsample_truths: int | None = 12,
    n_prior_samples: int = 32, sel_tol: float = SEL_TOL, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Same-model PPE with a generative prior; guided sampling replaces the gain.

    Truths are the younger chronological half; each borrows ``n_shapes`` real
    network geometries, the first ``n_select`` for selection and the last for the
    test. ``gamma`` is swept on the selection shapes over ``sel_subsample_truths``
    truths (``None`` uses all), then the winner runs on the full test split.
    Pseudo-obs are the truth at each site plus the mean of ``n_noise`` ``N(0, sse)``
    draws (shared across ``gamma``); ``R = diag(sse)``.

    Averaging the draws matches the classical lane, which averages ``n_noise``
    analyses and, being affine in the observations, thereby assimilates the same
    mean; without it the two lanes would assimilate observations of different
    effective precision. Consuming the shared ``rng`` one draw at a time also keeps
    the two lanes' streams aligned, so both draw the same network shapes per truth.
    """
    ages_i = np.asarray(ages, dtype=np.int64)
    prior_idx, truth_idx = chronological_half_split(ages_i, stride=truth_stride)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid)
    safe_valid = prior.safe_valid
    shape = (len(VARS), len(lats), len(lons))
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    prior_var = np.diag(prior.B).reshape(shape)                 # CE/CRPSS reference

    truth_cube = cube[truth_idx].astype(np.float64)
    truth_clim = truth_cube.mean(axis=0)
    truth_anoms = truth_cube - truth_clim
    T = len(truth_anoms)
    events = np.zeros(T, dtype=np.int64)

    rng = np.random.default_rng(seed)
    geoms, drawn_ages, noise = [], np.zeros((T, n_shapes), np.int64), []
    for ti in range(T):
        shape_ages = _draw_usable_ages(long, rng, lats, lons, safe_flat, n_shapes)
        drawn_ages[ti] = shape_ages
        row_g, row_n = [], []
        for k_age in shape_ages:
            g = _obs_geometry(observations_at_age(long, int(k_age)), lats, lons, safe_flat)
            row_g.append(g)
            row_n.append(np.mean([rng.normal(0.0, np.sqrt(g["sse"]))
                                  for _ in range(n_noise)], axis=0))
        geoms.append(row_g)
        noise.append(row_n)

    def _obs(ti, si):
        g = geoms[ti][si]
        y = truth_anoms[ti].ravel()[g["gather"]] + noise[ti][si]
        return g, y

    def _post_mean(ti, si, gamma, n):
        g, y = _obs(ti, si)
        ens = sampler.sample_posterior(g["gather"], y, g["sse"], gamma, n,
                                       seed=seed + ti * n_shapes + si)
        return g, y, ens

    # -- selection: sweep gamma on selection shapes over a truth subset ----
    sel_ti = (np.arange(T) if sel_subsample_truths is None
              else np.linspace(0, T - 1, min(sel_subsample_truths, T)).round().astype(int))
    sel_ti = np.unique(sel_ti)
    rows: list[dict] = []
    t0 = time.time()
    for gi, gamma in enumerate(gamma_grid):
        recon, truth_pool = [], []
        for ti in sel_ti:
            for si in range(n_select):
                _, _, ens = _post_mean(ti, si, gamma, n_samples_select)
                recon.append(ens.mean(axis=0))
                truth_pool.append(truth_anoms[ti])
        base = {**_GEN, "lane": LANE_PPE, "fold": -1, "b_scale": float(gamma),
                "background": "climatological", "split": "selection"}
        rows += _skill_rows(np.stack(truth_pool), np.stack(recon), safe_valid,
                            np.zeros(len(recon), np.int64), base)
        if progress_every:
            _report_progress("gamma", gi + 1, len(gamma_grid), t0)
    gamma_star = select_best_config(_sel_rrmse(rows), sel_tol=sel_tol)["b_scale"]

    # -- test: run the winner on the held-out shape, all truths ------------
    recon_test = np.zeros((T, *shape))
    post_test = np.zeros((T, *shape))
    naive_test = {"nearest": np.zeros((T, *shape)), "idw": np.zeros((T, *shape))}
    ens_valid, test_obs = [], []
    t0 = time.time()
    for ti in range(T):
        g, y, ens = _post_mean(ti, n_select, gamma_star, n_samples)
        recon_test[ti] = ens.mean(axis=0)
        post_test[ti] = ens.var(axis=0, ddof=1)
        ens_valid.append(ens[:, :, safe_valid])                 # (K, 2, n_valid)
        naive_geom = _naive_geometry(lats, lons, g, len(VARS))
        for kind in naive_test:
            naive_test[kind][ti] = _naive_apply(kind, naive_geom, y, shape)
        test_obs.append({"lat": g["lat"], "lon": g["lon"],
                         "val": truth_anoms[ti].ravel()[g["gather"]],
                         "chan": g["gather"] // (len(lats) * len(lons))})
        if progress_every and (ti + 1) % progress_every == 0:
            _report_progress("truth", ti + 1, T, t0)

    base = {**_GEN, "lane": LANE_PPE, "fold": -1, "b_scale": float(gamma_star),
            "background": "climatological", "split": "test"}
    rows += _skill_rows(truth_anoms, recon_test, safe_valid, events, base)
    rows += _ssim_rows(truth_anoms, recon_test, safe_valid, events, base)
    rows += _field_calibration_rows(truth_anoms, recon_test, post_test, prior_var,
                                    safe_valid, events, base)
    truth_valid = truth_anoms[:, :, safe_valid].transpose(1, 0, 2).reshape(len(VARS), -1)
    ens_all = np.concatenate([e.reshape(e.shape[0], len(VARS), -1) for e in ens_valid], axis=2)
    rows += _native_calibration_rows(truth_valid, ens_all, base)

    for kind in naive_test:
        nb = {"method": kind, "space": "generative", **_NAN_REG, "lane": LANE_PPE,
              "fold": -1, "b_scale": 1.0, "background": "none", "split": "test"}
        rows += _skill_rows(truth_anoms, naive_test[kind], safe_valid, events, nb)
        rows += _ssim_rows(truth_anoms, naive_test[kind], safe_valid, events, nb)

    prior_ens = sampler.sample_prior(n_prior_samples, seed=seed + 1_000_000)
    prior_ens_var = prior_ens.var(axis=0, ddof=1)
    obs_lat, obs_lon, obs_val, obs_chan, obs_n = _pad_obs(test_obs, T)
    npz = {
        "truth_anom": truth_anoms, "clim_mean": prior.clim_mean.astype(np.float64),
        "safe_valid": safe_valid, "prior_var": prior_var, "prior_ens_var": prior_ens_var,
        "post_var": post_test[None], "recon_climatological": recon_test[None],
        "b_scales": np.asarray([gamma_star]), "lats": np.asarray(lats), "lons": np.asarray(lons),
        "drawn_ages": drawn_ages, "truth_clim": truth_clim,
        "naive_nearest": naive_test["nearest"], "naive_idw": naive_test["idw"],
        "obs_lat": obs_lat, "obs_lon": obs_lon, "obs_val": obs_val,
        "obs_chan": obs_chan, "obs_n": obs_n,
    }
    config = {"lane": LANE_PPE, "space": "generative", "selected": {"b_scale": gamma_star},
              "gamma_grid": [float(g) for g in gamma_grid], "n_samples": n_samples,
              "n_samples_select": n_samples_select, "n_shapes": n_shapes, "n_select": n_select,
              "n_noise": n_noise, "n_truths": int(T), "truth_stride": truth_stride,
              "sel_subsample_truths": sel_subsample_truths,
              "n_prior_samples": n_prior_samples, "sel_tol": sel_tol,
              "seed": seed, "sampler": _sampler_meta(sampler),
              "prior_meta": prior.meta}
    _write(out_dir, LANE_PPE, "analysis", rows, npz, config)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-proxy withholding lane.
# ---------------------------------------------------------------------------
def run_withholding_generative(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long, out_dir: str, *, sampler,
    gamma_grid: tuple[float, ...] = GAMMA_GRID, n_samples: int = 16,
    n_samples_select: int = 8, k_folds: int = 5, fold_kind: str = "random",
    age_stride: int = 6, sel_tol: float = SEL_TOL, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Nested-CV site withholding with a generative prior; age-subsampled for cost.

    Two passes over one ``k_folds`` site partition: selection (assimilate 3 folds,
    predict the val fold, pooled over the rotation) picks ``gamma``; the test pass
    (assimilate 4 folds, predict the 5th) reports it. ``R = diag(sse + rep_var)``
    with ``rep_var`` estimated per fold from the assimilated sites only. Ages are
    thinned by ``age_stride`` on both passes to bound the sampling cost; the stride
    is recorded in the config.
    """
    from paleoreco.assim.experiments import _site_folds

    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid)
    safe_valid = prior.safe_valid
    shape = (len(VARS), len(lats), len(lons))
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    n_cells = len(lats) * len(lons)
    diagB = np.diag(prior.B)
    lane = f"withholding_{fold_kind}"

    obs_ages = np.intersect1d(long["age"].unique(), np.asarray(ages))[::age_stride]
    fold_sets = [set(f.tolist()) for f in _site_folds(long, k_folds, fold_kind, seed)]
    all_sites = set(long["site"].unique().tolist())
    cell_all = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons)

    def rep_lookup(sites):
        rv = representativeness_variance(long, cell_all, sites=sites)
        return np.array([rv.get(v, 0.0) for v in VARS])

    def predict(assim_sites, target_sites, gamma, n):
        """Assimilate ``assim_sites``, return withheld-site predictions pooled over ages."""
        rep = rep_lookup(assim_sites)
        actual, channel, sse, rep_out = [], [], [], []
        pred, post_var, prior_var, dist, ens_cols = [], [], [], [], []
        naive = {"nearest": [], "idw": []}
        for age in obs_ages:
            o = observations_at_age(long, int(age))
            gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
            keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
            kept = keep & np.array([s in assim_sites for s in o["site"]])
            wk = keep & np.array([s in target_sites for s in o["site"]])
            if kept.sum() == 0 or wk.sum() == 0:
                continue
            y_anom = (o["y"][kept] - o["my"][kept]).astype(np.float64)
            gk, gw = gather[kept], gather[wk]
            r = o["sse"][kept].astype(np.float64) + rep[gk // n_cells]
            ens = sampler.sample_posterior(gk, y_anom, r, gamma, n, seed=seed + int(age))
            ens_w = ens.reshape(n, -1)[:, gw]                   # (n, n_w)
            actual.append((o["y"][wk] - o["my"][wk]).astype(np.float64))
            channel.append(gw // n_cells)
            sse.append(o["sse"][wk].astype(np.float64))
            rep_out.append(rep[gw // n_cells])
            pred.append(ens_w.mean(axis=0))
            post_var.append(ens_w.var(axis=0, ddof=1))
            prior_var.append(diagB[gw])
            ens_cols.append(ens_w)
            nv, d = _naive_obs_predictions(
                {"lat": o["lat"][kept], "lon": o["lon"][kept], "y": y_anom, "chan": gk // n_cells},
                {"lat": o["lat"][wk], "lon": o["lon"][wk], "chan": gw // n_cells}, len(VARS))
            for kind in naive:
                naive[kind].append(nv[kind])
            dist.append(d)
        if not actual:
            return None
        return {"actual": np.concatenate(actual), "channel": np.concatenate(channel),
                "sse": np.concatenate(sse), "rep_var": np.concatenate(rep_out),
                "pred": np.concatenate(pred), "post_var": np.concatenate(post_var),
                "prior_var": np.concatenate(prior_var), "ens": np.concatenate(ens_cols, axis=1),
                "distance_km": np.concatenate(dist),
                "naive": {k: np.concatenate(v) for k, v in naive.items()}}

    def skill_and_calib(tp, gamma, split, fold):
        base = {**_GEN, "lane": lane, "fold": fold, "b_scale": float(gamma),
                "background": "climatological", "split": split}
        out = _withholding_rows(tp["actual"], tp["pred"], tp["channel"], base)
        groups = _obs_channel_groups(tp["channel"])
        out += _calibration_rows(tp["actual"], tp["pred"], tp["post_var"] + tp["sse"] + tp["rep_var"],
                                 gamma * tp["prior_var"] + tp["sse"] + tp["rep_var"], groups, base)
        return out

    # -- selection: rotate the val fold, pool, pick gamma -----------------
    rows: list[dict] = []
    t0 = time.time()
    for gi, gamma in enumerate(gamma_grid):
        parts = []
        for i in range(k_folds):
            assim = all_sites - fold_sets[(i + 1) % k_folds] - fold_sets[i]
            tp = predict(assim, fold_sets[(i + 1) % k_folds], gamma, n_samples_select)
            if tp is not None:
                parts.append(tp)
        if parts:
            rows += skill_and_calib(_concat(parts), gamma, "selection", -1)
        if progress_every:
            _report_progress(f"gamma ({lane})", gi + 1, len(gamma_grid), t0)
    gamma_star = select_best_config(_sel_rrmse(rows, lane), sel_tol=sel_tol)["b_scale"]

    # -- test: assimilate 4 folds, predict the 5th ------------------------
    pooled = []
    t0 = time.time()
    for i in range(k_folds):
        tp = predict(all_sites - fold_sets[i], fold_sets[i], gamma_star, n_samples)
        if tp is None:
            continue
        rows += skill_and_calib(tp, gamma_star, "test", i)
        pooled.append(tp)
        if progress_every:
            _report_progress(f"test-fold ({lane})", i + 1, k_folds, t0)

    predictions = {"b_scales": np.asarray([gamma_star]), "rep_var_full": rep_lookup(all_sites)}
    if pooled:
        tp = _concat(pooled)
        rows += skill_and_calib(tp, gamma_star, "test", -1)
        native_base = {**_GEN, "lane": lane, "fold": -1, "b_scale": float(gamma_star),
                       "background": "climatological", "split": "test"}
        rows += _native_obs_calibration_rows(tp, native_base)
        for kind, pred in tp["naive"].items():
            rows += _withholding_rows(tp["actual"], pred, tp["channel"],
                                      {"method": kind, "space": "generative", **_NAN_REG,
                                       "lane": lane, "fold": -1, "b_scale": 1.0,
                                       "background": "none", "split": "test"})
        predictions.update({
            "actual": tp["actual"], "channel": tp["channel"],
            "climatological_pred": tp["pred"][None], "post_var_pred": tp["post_var"][None],
            "prior_var_pred": tp["prior_var"], "sse": tp["sse"], "rep_var": tp["rep_var"],
            "distance_km": tp["distance_km"], "naive_nearest": tp["naive"]["nearest"],
            "naive_idw": tp["naive"]["idw"]})

    config = {"lane": lane, "space": "generative", "selected": {"b_scale": gamma_star},
              "gamma_grid": [float(g) for g in gamma_grid], "n_samples": n_samples,
              "n_samples_select": n_samples_select, "k_folds": k_folds, "fold_kind": fold_kind,
              "age_stride": age_stride, "n_obs_ages": int(len(obs_ages)), "seed": seed,
              "sampler": _sampler_meta(sampler), "prior_meta": prior.meta,
              "rep_var_full": {VARS[i]: float(v) for i, v in enumerate(predictions["rep_var_full"])}}
    _write(out_dir, lane, "predictions", rows, predictions, config)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _native_obs_calibration_rows(tp: dict, base: dict) -> list[dict]:
    """ECR/sharpness/native CRPS/coverage at withheld sites, pooled and per channel.

    The predictive spread carries the proxy error and representativeness variance,
    which the residual also holds, matched to the Gaussian-summary rows.
    """
    extra = tp["sse"] + tp["rep_var"]
    rows = []
    groups = [("pooled", np.ones(len(tp["actual"]), bool))]
    groups += [(name, tp["channel"] == c) for c, name in enumerate(VARS)]
    for name, sel in groups:
        if sel.sum() < 2:
            continue
        a, e = tp["actual"][sel], tp["ens"][:, sel]
        var = e.var(axis=0, ddof=1) + extra[sel]
        for metric, value in (
            ("ecr", calibration.ecr(a, e.mean(axis=0), var)),
            ("sharpness", calibration.sharpness(var)),
            ("crps_ens", float(np.mean(calibration.crps_ensemble(a, e)))),
            ("coverage90_ens", calibration.coverage_ensemble(a, e, 0.9)),
        ):
            rows.append({**base, "do_event": "all", "channel": name,
                         "metric": metric, "value": value})
    return rows


def _concat(parts: list[dict]) -> dict:
    """Pool several folds' withheld-site predictions."""
    out = {k: np.concatenate([p[k] for p in parts], axis=(1 if k == "ens" else 0))
           for k in ("actual", "channel", "sse", "rep_var", "pred", "post_var",
                     "prior_var", "ens", "distance_km")}
    out["naive"] = {k: np.concatenate([p["naive"][k] for p in parts])
                    for k in parts[0]["naive"]}
    return out


def _sel_rrmse(rows: list[dict], lane: str = LANE_PPE) -> pd.DataFrame:
    """Pooled selection-split RRMSE rows, the surface ``gamma`` is chosen on."""
    M = pd.DataFrame(rows)
    return M[(M.lane == lane) & (M.split == "selection") & (M.channel == "pooled")
             & (M.do_event == "all") & (M.metric == "rrmse")]


def _sampler_meta(sampler) -> dict:
    """JSON-safe record of the sampler configuration."""
    return {"n_steps": sampler.n_steps, "n_correct": sampler.n_correct,
            "corrector_tau": sampler.tau, "sigma_data": sampler.sd}


def _write(out_dir: str, lane: str, npz_kind: str, rows: list[dict],
           npz: dict, config: dict) -> None:
    """Persist metrics CSV (appended), the lane npz, and the config json."""
    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_{npz_kind}.npz"), **npz)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
