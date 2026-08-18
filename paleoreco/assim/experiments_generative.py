"""Experiment runners for the generative (score-based) prior.

The same two evaluation lanes as :mod:`paleoreco.assim.experiments`, but the
Gaussian-gain analysis is replaced by guided posterior sampling from a
:class:`paleoreco.assim.generative.GuidedSampler`. The observation geometry,
truth draws, site folds, representativeness variance, metric rows, and tidy CSV
schema are reused unchanged, so the generative method drops into notebook 09's
cross-method comparison with no special-casing.

``gamma`` scales the prior block of the guidance covariance and is the tuned
operating point, the analogue of 3DVar ``b_scale``: swept on the held-out
selection split over a reduced workload, then the winner is run once on the full
test split. Rows carry ``method="generative"``, ``space="generative"``, and
``b_scale=gamma`` so the selection rule (:func:`select_best_config`) and the
reported test rows match the classical lanes. Calibration is scored both as a
Gaussian summary of the ensemble (directly comparable to 3DVar) and natively over
the ensemble.

The hybrid runners tune ``(amp, sigma_switch, gamma)`` jointly on the same
selection split. ``amp`` and ``sigma_switch`` change the denoiser itself, so each
grid point takes its own sampler from a caller-supplied factory; rows carry
``method="generative_hybrid"`` plus ``hybrid_amp`` / ``hybrid_switch`` columns and
the winner is the argmin of the debiased selection RRMSE.

Each pass builds its whole list of posterior draws before executing it, so a sampler
that spreads draws over worker processes stays fed and the progress line quotes an
ETA over the pass rather than over one gamma.
"""

from __future__ import annotations

import glob
import itertools
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
from paleoreco.assim.generative import PosteriorJob
from paleoreco.eval import calibration
from paleoreco.assim.experiments import (
    _obs_geometry, _draw_usable_ages, _naive_geometry, _naive_apply,
    _naive_obs_predictions, _pad_obs, _skill_rows, _ssim_rows, _field_calibration_rows,
    _withholding_rows, _calibration_rows, _obs_channel_groups, _append_csv,
    _report_progress, select_best_config, _NAN_REG, LANE_PPE, SEL_TOL,
)

GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
AMP_GRID = (1.0, 2.0)
SWITCH_GRID = (0.5, 1.0, 2.0, 4.0)
HYBRID_GAMMA_GRID = (0.5, 1.0, 2.0)
_GEN = {"method": "generative", "space": "generative", **_NAN_REG}
# _append_csv appends positionally, so both hybrid runners must spell their columns
# in this one order; the nan placeholders pin where the overrides land.
_HYB = {"method": "generative_hybrid", "space": "generative", **_NAN_REG,
        "hybrid_amp": np.nan, "hybrid_switch": np.nan}


def _stream(sampler, jobs, label: str, progress_every: int | None):
    """Yield each job's ensemble in order, reporting progress over the whole list.

    Every draw costs the same (``2 * n_steps - 1`` network evaluations, independent
    of how many observations it carries), so the linear-rate ETA
    :func:`_report_progress` prints is accurate rather than indicative.
    """
    t0 = time.time()
    for k, ens in enumerate(sampler.imap_posterior(jobs)):
        yield ens
        if progress_every and (k + 1) % progress_every == 0:
            _report_progress(label, k + 1, len(jobs), t0)


# ---------------------------------------------------------------------------
# Selection surface and native ensemble calibration.
# ---------------------------------------------------------------------------
def _ens_rrmse_row(truth: np.ndarray, pred: np.ndarray, post_var: np.ndarray,
                   n: int, base: dict) -> dict:
    """Pooled RRMSE with the ``n``-member sampling error removed.

    The mean of ``n`` posterior draws carries an extra ``mean(post_var) / n`` of
    squared error, and the posterior spread grows with ``gamma``, so the raw
    ensemble RRMSE tilts selection toward small ``gamma``. Subtracting it compares
    operating points at their large-ensemble value.
    """
    mse = float(np.mean((pred - truth) ** 2)) - float(np.mean(post_var)) / n
    sd = float(np.std(truth))
    value = float(np.sqrt(max(mse, 0.0)) / sd) if sd > 0 else float("nan")
    return {**base, "do_event": "all", "channel": "pooled",
            "metric": "rrmse_ens", "value": value}


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
def _ppe_setup(cube, ages, lats, lons, valid, long, *, n_shapes: int, n_noise: int,
               truth_stride: int, seed: int) -> dict:
    """Truths, network geometries and pseudo-obs shared by every PPE operating point.

    One seeded rng draws the geometries and noise, consumed one draw at a time so
    the stream stays aligned with the classical lane and both draw the same network
    shapes per truth. The ``job`` closure carries the per-draw seed, so a job list
    reproduces however it executes.
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

    def obs(ti, si):
        g = geoms[ti][si]
        y = truth_anoms[ti].ravel()[g["gather"]] + noise[ti][si]
        return g, y

    def job(ti, si, gamma, n):
        g, y = obs(ti, si)
        return PosteriorJob(g["gather"], y, g["sse"], gamma, n,
                            seed + ti * n_shapes + si)

    return dict(prior=prior, safe_valid=safe_valid, shape=shape, prior_var=prior_var,
                truth_anoms=truth_anoms, truth_clim=truth_clim, T=T,
                events=np.zeros(T, dtype=np.int64), drawn_ages=drawn_ages,
                lats=lats, lons=lons, obs=obs, job=job)


def _sel_truths(T: int, sel_subsample_truths: int | None) -> np.ndarray:
    """Truth indices the selection sweep runs on (``None`` uses all)."""
    if sel_subsample_truths is None:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, min(sel_subsample_truths, T))
                     .round().astype(int))


def _ppe_selection_rows(sampler, S: dict, gamma_grid, sel_ti, n_select: int,
                        n_samples_select: int, base: dict,
                        progress_every: int | None, label: str) -> list[dict]:
    """Selection-split skill rows for one sampler swept over ``gamma_grid``.

    ``base`` carries the row identity (method and operating-point columns) with a
    ``b_scale`` placeholder each gamma overrides in place, keeping the tidy CSV's
    column order independent of who built the rows.
    """
    truth_anoms, safe_valid = S["truth_anoms"], S["safe_valid"]
    sel_keys = [(gamma, ti) for gamma in gamma_grid for ti in sel_ti
                for _ in range(n_select)]
    sel_jobs = [S["job"](ti, si, gamma, n_samples_select)
                for gamma in gamma_grid for ti in sel_ti for si in range(n_select)]

    means, varis = [], []
    for ens in _stream(sampler, sel_jobs, label, progress_every):
        means.append(ens.mean(axis=0))
        varis.append(ens.var(axis=0, ddof=1))

    rows: list[dict] = []
    for gamma in gamma_grid:
        sel = [k for k, (gm, _) in enumerate(sel_keys) if gm == gamma]
        recon = np.stack([means[k] for k in sel])
        post = np.stack([varis[k] for k in sel])
        truth_pool = np.stack([truth_anoms[sel_keys[k][1]] for k in sel])
        gbase = {**base, "b_scale": float(gamma)}
        rows += _skill_rows(truth_pool, recon, safe_valid,
                            np.zeros(len(recon), np.int64), gbase)
        rows.append(_ens_rrmse_row(truth_pool[:, :, safe_valid], recon[:, :, safe_valid],
                                   post[:, :, safe_valid], n_samples_select, gbase))
    return rows


def _ppe_test_pass(sampler, S: dict, gamma_star: float, n_samples: int, n_select: int,
                   base: dict, prior_ens_var: np.ndarray,
                   progress_every: int | None, label: str) -> tuple[list[dict], dict]:
    """Winner-only test split: the full metric rows plus the analysis npz payload."""
    truth_anoms, safe_valid, shape = S["truth_anoms"], S["safe_valid"], S["shape"]
    T, events, lats, lons = S["T"], S["events"], S["lats"], S["lons"]
    recon_test = np.zeros((T, *shape))
    post_test = np.zeros((T, *shape))
    naive_test = {"nearest": np.zeros((T, *shape)), "idw": np.zeros((T, *shape))}
    ens_valid, test_obs = [], []
    test_jobs = [S["job"](ti, n_select, gamma_star, n_samples) for ti in range(T)]
    for ti, ens in enumerate(_stream(sampler, test_jobs, label, progress_every)):
        g, y = S["obs"](ti, n_select)
        recon_test[ti] = ens.mean(axis=0)
        post_test[ti] = ens.var(axis=0, ddof=1)
        ens_valid.append(ens[:, :, safe_valid])                 # (K, 2, n_valid)
        naive_geom = _naive_geometry(lats, lons, g, len(VARS))
        for kind in naive_test:
            naive_test[kind][ti] = _naive_apply(kind, naive_geom, y, shape)
        test_obs.append({"lat": g["lat"], "lon": g["lon"],
                         "val": truth_anoms[ti].ravel()[g["gather"]],
                         "chan": g["gather"] // (len(lats) * len(lons))})

    tbase = {**base, "b_scale": float(gamma_star)}
    rows = _skill_rows(truth_anoms, recon_test, safe_valid, events, tbase)
    rows.append(_ens_rrmse_row(truth_anoms[:, :, safe_valid],
                               recon_test[:, :, safe_valid],
                               post_test[:, :, safe_valid], n_samples, tbase))
    rows += _ssim_rows(truth_anoms, recon_test, safe_valid, events, tbase)
    rows += _field_calibration_rows(truth_anoms, recon_test, post_test, S["prior_var"],
                                    safe_valid, events, tbase)
    truth_valid = truth_anoms[:, :, safe_valid].transpose(1, 0, 2).reshape(len(VARS), -1)
    ens_all = np.concatenate([e.reshape(e.shape[0], len(VARS), -1) for e in ens_valid], axis=2)
    rows += _native_calibration_rows(truth_valid, ens_all, tbase)

    for kind in naive_test:
        nb = {"method": kind, "space": "generative", **_NAN_REG, "lane": LANE_PPE,
              "fold": -1, "b_scale": 1.0, "background": "none", "split": "test"}
        rows += _skill_rows(truth_anoms, naive_test[kind], safe_valid, events, nb)
        rows += _ssim_rows(truth_anoms, naive_test[kind], safe_valid, events, nb)

    obs_lat, obs_lon, obs_val, obs_chan, obs_n = _pad_obs(test_obs, T)
    npz = {
        "truth_anom": truth_anoms, "clim_mean": S["prior"].clim_mean.astype(np.float64),
        "safe_valid": safe_valid, "prior_var": S["prior_var"],
        # Per-cell spread of the sampled prior, for the prior-vs-posterior figures;
        # the guidance itself reads the full covariance, not this diagonal.
        "prior_ens_var": np.asarray(prior_ens_var, dtype=np.float64),
        "post_var": post_test[None], "recon_climatological": recon_test[None],
        "b_scales": np.asarray([gamma_star]), "lats": np.asarray(lats), "lons": np.asarray(lons),
        "drawn_ages": S["drawn_ages"], "truth_clim": S["truth_clim"],
        "naive_nearest": naive_test["nearest"], "naive_idw": naive_test["idw"],
        "obs_lat": obs_lat, "obs_lon": obs_lon, "obs_val": obs_val,
        "obs_chan": obs_chan, "obs_n": obs_n,
    }
    return rows, npz


def run_ppe_generative(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long, out_dir: str, *, sampler, prior_ens_var: np.ndarray,
    gamma_grid: tuple[float, ...] = GAMMA_GRID, n_samples: int = 32,
    n_samples_select: int = 16, n_shapes: int = 5, n_select: int = 4,
    n_noise: int = 5, truth_stride: int = 10, sel_subsample_truths: int | None = 12,
    sel_tol: float = SEL_TOL, seed: int = 0, method: str = "generative",
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

    ``prior_ens_var`` is the sampled prior's per-cell variance, stored in the
    analysis npz for the prior-vs-posterior spread diagnostics.
    ``progress_every`` counts posterior draws.

    ``method`` labels the rows, so a second generative prior writing the same
    schema into its own directory appears in notebook 09 as its own row rather
    than colliding with this one. It overrides a value in the row base without
    changing the key order the tidy CSV is appended on.
    """
    S = _ppe_setup(cube, ages, lats, lons, valid, long, n_shapes=n_shapes,
                   n_noise=n_noise, truth_stride=truth_stride, seed=seed)
    sel_ti = _sel_truths(S["T"], sel_subsample_truths)
    base = {**_GEN, "method": method, "lane": LANE_PPE, "fold": -1,
            "b_scale": np.nan, "background": "climatological"}

    rows = _ppe_selection_rows(sampler, S, gamma_grid, sel_ti, n_select,
                               n_samples_select, {**base, "split": "selection"},
                               progress_every, "ppe selection draws")
    gamma_star = select_best_config(_sel_rrmse(rows), sel_tol=sel_tol)["b_scale"]

    test_rows, npz = _ppe_test_pass(sampler, S, gamma_star, n_samples, n_select,
                                    {**base, "split": "test"}, prior_ens_var,
                                    progress_every, "ppe test draws")
    rows += test_rows

    config = {"lane": LANE_PPE, "space": "generative", "method": method,
              "selected": {"b_scale": gamma_star},
              "gamma_grid": [float(g) for g in gamma_grid], "n_samples": n_samples,
              "n_samples_select": n_samples_select, "n_shapes": n_shapes, "n_select": n_select,
              "n_noise": n_noise, "n_truths": int(S["T"]), "truth_stride": truth_stride,
              "sel_subsample_truths": sel_subsample_truths, "sel_tol": sel_tol,
              "seed": seed, "sampler": _sampler_meta(sampler),
              "prior_meta": S["prior"].meta}
    _write(out_dir, LANE_PPE, "analysis", rows, npz, config)
    return pd.DataFrame(rows)


def _prior_var(sampler, n: int, seed: int) -> np.ndarray:
    """Per-cell variance of ``n`` unconditional draws, batched to bound memory."""
    ens = [sampler.sample_prior(min(64, n - k), seed=seed + k // 64)
           for k in range(0, n, 64)]
    return np.concatenate(ens).var(axis=0, ddof=1)


def run_ppe_generative_hybrid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long, out_dir: str, *, make_sampler,
    amp_grid: tuple[float, ...] = AMP_GRID,
    switch_grid: tuple[float, ...] = SWITCH_GRID,
    gamma_grid: tuple[float, ...] = HYBRID_GAMMA_GRID, n_samples: int = 32,
    n_samples_select: int = 16, n_shapes: int = 5, n_select: int = 4,
    n_noise: int = 5, truth_stride: int = 10, sel_subsample_truths: int | None = 12,
    n_prior_var: int = 512, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Same-model PPE for the hybrid denoiser, tuned over (amp, sigma_switch, gamma).

    ``make_sampler(amp, sigma_switch)`` returns a context manager yielding a sampler
    whose denoiser and guidance share the amp-scaled covariance; the combos run
    sequentially, so one grid point's workers hold the GPU at a time. Every combo
    shares one set of truths, geometries and pseudo-obs (the ``seed`` protocol of
    :func:`run_ppe_generative`), the winner is the argmin of the debiased selection
    RRMSE, and only the winner runs the test split. ``prior_ens_var`` cannot be an
    argument here because it belongs to the winner, so ``n_prior_var`` unconditional
    winner draws stand in for it.
    """
    S = _ppe_setup(cube, ages, lats, lons, valid, long, n_shapes=n_shapes,
                   n_noise=n_noise, truth_stride=truth_stride, seed=seed)
    sel_ti = _sel_truths(S["T"], sel_subsample_truths)
    base = {**_HYB, "lane": LANE_PPE, "fold": -1, "b_scale": np.nan,
            "background": "climatological"}

    rows: list[dict] = []
    for amp, switch in itertools.product(amp_grid, switch_grid):
        cbase = {**base, "hybrid_amp": float(amp), "hybrid_switch": float(switch),
                 "split": "selection"}
        with make_sampler(amp, switch) as smp:
            rows += _ppe_selection_rows(
                smp, S, gamma_grid, sel_ti, n_select, n_samples_select, cbase,
                progress_every,
                f"hybrid selection draws (amp={amp:g}, switch={switch:g})")

    sel = _sel_rrmse(rows)
    win = sel.loc[sel["value"].idxmin()]
    amp_s, switch_s = float(win["hybrid_amp"]), float(win["hybrid_switch"])
    gamma_s = float(win["b_scale"])

    with make_sampler(amp_s, switch_s) as smp:
        prior_ens_var = _prior_var(smp, n_prior_var, seed)
        test_rows, npz = _ppe_test_pass(
            smp, S, gamma_s, n_samples, n_select,
            {**base, "hybrid_amp": amp_s, "hybrid_switch": switch_s, "split": "test"},
            prior_ens_var, progress_every, "hybrid test draws")
        sampler_meta = _sampler_meta(smp)
    rows += test_rows

    config = {"lane": LANE_PPE, "space": "generative", "method": "generative_hybrid",
              "selected": {"b_scale": gamma_s, "hybrid_amp": amp_s,
                           "hybrid_switch": switch_s},
              "amp_grid": [float(a) for a in amp_grid],
              "switch_grid": [float(s) for s in switch_grid],
              "gamma_grid": [float(g) for g in gamma_grid],
              "n_samples": n_samples, "n_samples_select": n_samples_select,
              "n_shapes": n_shapes, "n_select": n_select, "n_noise": n_noise,
              "n_truths": int(S["T"]), "truth_stride": truth_stride,
              "sel_subsample_truths": sel_subsample_truths, "n_prior_var": n_prior_var,
              "seed": seed, "sampler": sampler_meta, "prior_meta": S["prior"].meta}
    _write(out_dir, LANE_PPE, "analysis", rows, npz, config)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-proxy withholding lane.
# ---------------------------------------------------------------------------
def _withholding_setup(cube, ages, lats, lons, valid, long, *, k_folds: int,
                       fold_kind: str, age_stride: int, seed: int) -> dict:
    """Prior, site folds and observation-split closures shared by every operating point."""
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

    def fold_split(assim_sites, target_sites):
        """Per-age observation split for one assim/target pair, free of ``gamma``.

        The split is what the posterior draws are built from, and it does not depend
        on the operating point, so one build serves the whole grid.
        """
        rep = rep_lookup(assim_sites)
        out = []
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
            nv, d = _naive_obs_predictions(
                {"lat": o["lat"][kept], "lon": o["lon"][kept], "y": y_anom, "chan": gk // n_cells},
                {"lat": o["lat"][wk], "lon": o["lon"][wk], "chan": gw // n_cells}, len(VARS))
            out.append({
                "age": int(age), "gk": gk, "gw": gw, "y_anom": y_anom,
                "r": o["sse"][kept].astype(np.float64) + rep[gk // n_cells],
                "actual": (o["y"][wk] - o["my"][wk]).astype(np.float64),
                "channel": gw // n_cells, "sse": o["sse"][wk].astype(np.float64),
                "rep_var": rep[gw // n_cells], "prior_var": diagB[gw],
                "naive": nv, "distance_km": d})
        return out

    def fold_jobs(splits, gamma, n):
        return [PosteriorJob(s["gk"], s["y_anom"], s["r"], gamma, n, seed + s["age"])
                for s in splits]

    return dict(prior=prior, lane=lane, k_folds=k_folds, obs_ages=obs_ages,
                fold_sets=fold_sets, all_sites=all_sites, rep_lookup=rep_lookup,
                fold_split=fold_split, fold_jobs=fold_jobs)


def _collect(splits, cols):
    """Withheld-site predictions for one fold, from its withheld-column ensembles."""
    if not splits:
        return None
    keys = ("actual", "channel", "sse", "rep_var", "prior_var", "distance_km")
    out = {k: np.concatenate([s[k] for s in splits]) for k in keys}
    out["pred"] = np.concatenate([c.mean(axis=0) for c in cols])
    out["post_var"] = np.concatenate([c.var(axis=0, ddof=1) for c in cols])
    out["ens"] = np.concatenate(cols, axis=1)
    out["naive"] = {k: np.concatenate([s["naive"][k] for s in splits])
                    for k in splits[0]["naive"]}
    return out


def _withholding_skill_rows(tp: dict, base: dict, n: int) -> list[dict]:
    """Skill, Gaussian-summary calibration and debiased-RRMSE rows for one pool."""
    out = _withholding_rows(tp["actual"], tp["pred"], tp["channel"], base)
    groups = _obs_channel_groups(tp["channel"])
    # The CRPSS reference is the do-nothing forecast N(0, diag B), matching the PPE
    # lane. gamma scales the likelihood, not the prior, so it does not enter here.
    out += _calibration_rows(tp["actual"], tp["pred"], tp["post_var"] + tp["sse"] + tp["rep_var"],
                             tp["prior_var"] + tp["sse"] + tp["rep_var"], groups, base)
    out.append(_ens_rrmse_row(tp["actual"], tp["pred"], tp["post_var"], n, base))
    return out


def _withholding_selection_rows(sampler, W: dict, gamma_grid, n_samples_select: int,
                                base: dict, progress_every: int | None,
                                label: str) -> list[dict]:
    """Selection-split rows for one sampler: rotate the val fold, pool per gamma."""
    k_folds, fold_sets, all_sites = W["k_folds"], W["fold_sets"], W["all_sites"]
    sel_splits = [W["fold_split"](all_sites - fold_sets[(i + 1) % k_folds] - fold_sets[i],
                                  fold_sets[(i + 1) % k_folds]) for i in range(k_folds)]
    sel_flat = [s for _ in gamma_grid for sp in sel_splits for s in sp]
    sel_jobs = [j for gamma in gamma_grid for sp in sel_splits
                for j in W["fold_jobs"](sp, gamma, n_samples_select)]
    sel_cols = [ens.reshape(n_samples_select, -1)[:, s["gw"]]
                for s, ens in zip(sel_flat, _stream(sampler, sel_jobs, label,
                                                    progress_every))]

    rows: list[dict] = []
    at = 0
    for gamma in gamma_grid:
        parts = []
        for sp in sel_splits:
            tp = _collect(sp, sel_cols[at:at + len(sp)])
            at += len(sp)
            if tp is not None:
                parts.append(tp)
        if parts:
            rows += _withholding_skill_rows(_concat(parts),
                                            {**base, "b_scale": float(gamma)},
                                            n_samples_select)
    return rows


def _withholding_test_pass(sampler, W: dict, gamma_star: float, n_samples: int,
                           base: dict, progress_every: int | None,
                           label: str) -> tuple[list[dict], dict]:
    """Winner-only test split: per-fold and pooled rows plus the predictions payload."""
    k_folds, fold_sets, all_sites = W["k_folds"], W["fold_sets"], W["all_sites"]
    test_splits = [W["fold_split"](all_sites - fold_sets[i], fold_sets[i])
                   for i in range(k_folds)]
    test_flat = [s for sp in test_splits for s in sp]
    test_jobs = [j for sp in test_splits
                 for j in W["fold_jobs"](sp, gamma_star, n_samples)]
    test_cols = [ens.reshape(n_samples, -1)[:, s["gw"]]
                 for s, ens in zip(test_flat, _stream(sampler, test_jobs, label,
                                                      progress_every))]

    tbase = {**base, "b_scale": float(gamma_star)}
    rows: list[dict] = []
    pooled = []
    at = 0
    for i, sp in enumerate(test_splits):
        tp = _collect(sp, test_cols[at:at + len(sp)])
        at += len(sp)
        if tp is None:
            continue
        rows += _withholding_skill_rows(tp, {**tbase, "fold": i}, n_samples)
        pooled.append(tp)

    predictions = {"b_scales": np.asarray([gamma_star]),
                   "rep_var_full": W["rep_lookup"](all_sites)}
    if pooled:
        tp = _concat(pooled)
        rows += _withholding_skill_rows(tp, tbase, n_samples)
        rows += _native_obs_calibration_rows(tp, tbase)
        for kind, pred in tp["naive"].items():
            rows += _withholding_rows(tp["actual"], pred, tp["channel"],
                                      {"method": kind, "space": "generative", **_NAN_REG,
                                       "lane": W["lane"], "fold": -1, "b_scale": 1.0,
                                       "background": "none", "split": "test"})
        predictions.update({
            "actual": tp["actual"], "channel": tp["channel"],
            "climatological_pred": tp["pred"][None], "post_var_pred": tp["post_var"][None],
            "prior_var_pred": tp["prior_var"], "sse": tp["sse"], "rep_var": tp["rep_var"],
            "distance_km": tp["distance_km"], "naive_nearest": tp["naive"]["nearest"],
            "naive_idw": tp["naive"]["idw"]})
    return rows, predictions


def run_withholding_generative(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long, out_dir: str, *, sampler,
    gamma_grid: tuple[float, ...] = GAMMA_GRID, n_samples: int = 32,
    n_samples_select: int = 16, k_folds: int = 5, fold_kind: str = "random",
    age_stride: int = 6, sel_tol: float = SEL_TOL, seed: int = 0,
    method: str = "generative", progress_every: int | None = None,
) -> pd.DataFrame:
    """Nested-CV site withholding with a generative prior; age-subsampled for cost.

    Two passes over one ``k_folds`` site partition: selection (assimilate 3 folds,
    predict the val fold, pooled over the rotation) picks ``gamma``; the test pass
    (assimilate 4 folds, predict the 5th) reports it. ``R = diag(sse + rep_var)``
    with ``rep_var`` estimated per fold from the assimilated sites only. Ages are
    thinned by ``age_stride`` on both passes to bound the sampling cost; the stride
    is recorded in the config.

    ``progress_every`` counts posterior draws. ``method`` labels the rows, as in
    :func:`run_ppe_generative`.
    """
    W = _withholding_setup(cube, ages, lats, lons, valid, long, k_folds=k_folds,
                           fold_kind=fold_kind, age_stride=age_stride, seed=seed)
    lane = W["lane"]
    base = {**_GEN, "method": method, "lane": lane, "fold": -1, "b_scale": np.nan,
            "background": "climatological"}

    rows = _withholding_selection_rows(sampler, W, gamma_grid, n_samples_select,
                                       {**base, "split": "selection"}, progress_every,
                                       f"withholding selection draws ({lane})")
    gamma_star = select_best_config(_sel_rrmse(rows, lane), sel_tol=sel_tol)["b_scale"]

    test_rows, predictions = _withholding_test_pass(
        sampler, W, gamma_star, n_samples, {**base, "split": "test"}, progress_every,
        f"withholding test draws ({lane})")
    rows += test_rows

    config = {"lane": lane, "space": "generative", "method": method,
              "selected": {"b_scale": gamma_star},
              "gamma_grid": [float(g) for g in gamma_grid], "n_samples": n_samples,
              "n_samples_select": n_samples_select, "k_folds": k_folds, "fold_kind": fold_kind,
              "age_stride": age_stride, "n_obs_ages": int(len(W["obs_ages"])), "seed": seed,
              "sampler": _sampler_meta(sampler), "prior_meta": W["prior"].meta,
              "rep_var_full": {VARS[i]: float(v) for i, v in enumerate(predictions["rep_var_full"])}}
    _write(out_dir, lane, "predictions", rows, predictions, config)
    return pd.DataFrame(rows)


def run_withholding_generative_hybrid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long, out_dir: str, *, make_sampler,
    amp_grid: tuple[float, ...] = AMP_GRID,
    switch_grid: tuple[float, ...] = SWITCH_GRID,
    gamma_grid: tuple[float, ...] = HYBRID_GAMMA_GRID, n_samples: int = 32,
    n_samples_select: int = 16, k_folds: int = 5, fold_kind: str = "random",
    age_stride: int = 6, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Site withholding for the hybrid denoiser, tuned over (amp, sigma_switch, gamma).

    The nested-CV protocol of :func:`run_withholding_generative`: all combos share
    one fold partition and observation split, run sequentially through
    ``make_sampler`` (a context-manager factory, see
    :func:`run_ppe_generative_hybrid`), and the argmin of the debiased selection
    RRMSE runs the test rotation.
    """
    W = _withholding_setup(cube, ages, lats, lons, valid, long, k_folds=k_folds,
                           fold_kind=fold_kind, age_stride=age_stride, seed=seed)
    lane = W["lane"]
    base = {**_HYB, "lane": lane, "fold": -1, "b_scale": np.nan,
            "background": "climatological"}

    rows: list[dict] = []
    for amp, switch in itertools.product(amp_grid, switch_grid):
        cbase = {**base, "hybrid_amp": float(amp), "hybrid_switch": float(switch),
                 "split": "selection"}
        with make_sampler(amp, switch) as smp:
            rows += _withholding_selection_rows(
                smp, W, gamma_grid, n_samples_select, cbase, progress_every,
                f"hybrid selection draws ({lane}, amp={amp:g}, switch={switch:g})")

    sel = _sel_rrmse(rows, lane)
    win = sel.loc[sel["value"].idxmin()]
    amp_s, switch_s = float(win["hybrid_amp"]), float(win["hybrid_switch"])
    gamma_s = float(win["b_scale"])

    with make_sampler(amp_s, switch_s) as smp:
        test_rows, predictions = _withholding_test_pass(
            smp, W, gamma_s, n_samples,
            {**base, "hybrid_amp": amp_s, "hybrid_switch": switch_s, "split": "test"},
            progress_every, f"hybrid test draws ({lane})")
        sampler_meta = _sampler_meta(smp)
    rows += test_rows

    config = {"lane": lane, "space": "generative", "method": "generative_hybrid",
              "selected": {"b_scale": gamma_s, "hybrid_amp": amp_s,
                           "hybrid_switch": switch_s},
              "amp_grid": [float(a) for a in amp_grid],
              "switch_grid": [float(s) for s in switch_grid],
              "gamma_grid": [float(g) for g in gamma_grid],
              "n_samples": n_samples, "n_samples_select": n_samples_select,
              "k_folds": k_folds, "fold_kind": fold_kind, "age_stride": age_stride,
              "n_obs_ages": int(len(W["obs_ages"])), "seed": seed,
              "sampler": sampler_meta, "prior_meta": W["prior"].meta,
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
    """Sampling-error-corrected selection-split RRMSE, the surface ``gamma`` is chosen on."""
    M = pd.DataFrame(rows)
    return M[(M.lane == lane) & (M.split == "selection") & (M.channel == "pooled")
             & (M.do_event == "all") & (M.metric == "rrmse_ens")]


def _sampler_meta(sampler) -> dict:
    """JSON-safe record of the sampler configuration.

    ``cond_cov`` records how the likelihood's conditional covariance was obtained;
    a sampler that offers no choice is described as ``anchor``, which is what its
    Gaussian annealing amounts to.
    """
    return {"n_steps": sampler.n_steps, "sigma_data": sampler.sd,
            "cond_cov": getattr(sampler, "cond_cov", "anchor"),
            "guidance_cov": dict(sampler.cov.meta)}


def clear_lane(out_dir: str, lane: str) -> None:
    """Remove one lane's artefacts and its rows from the shared metrics CSV.

    The runners append, so re-running a lane into a directory that already holds it
    would double its rows. Clearing per lane rather than per directory lets one lane be
    re-run without disturbing another that shares the CSV.
    """
    if not os.path.isdir(out_dir):
        return
    for pattern in (f"{lane}_*.npz", f"{lane}_*.json"):
        for path in glob.glob(os.path.join(out_dir, pattern)):
            os.remove(path)

    csv_path = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return
    rows = pd.read_csv(csv_path)
    kept = rows[rows["lane"] != lane]
    if kept.empty:
        os.remove(csv_path)
    else:
        kept.to_csv(csv_path, index=False)


def _write(out_dir: str, lane: str, npz_kind: str, rows: list[dict],
           npz: dict, config: dict) -> None:
    """Persist metrics CSV (appended), the lane npz, and the config json."""
    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_{npz_kind}.npz"), **npz)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
