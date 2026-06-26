"""Experiment runners for data-assimilation reconstruction.

Two evaluation lanes, both method-agnostic through the :class:`Method` contract:

* :func:`run_ppe` - pseudo-proxy experiments. Hold out every Nth age as truth,
  build the prior from the rest, sample noisy pseudo-observations from each truth
  at the real network's locations, reconstruct, and score against the held-out
  truth field. The held-out truths are the only states never seen by the prior.
* :func:`run_withholding` - withhold whole proxy sites, assimilate the rest, and
  score the predicted observations at the withheld sites against the real values.

Both write a tidy long-format metrics CSV (one row per method/background/B_reg/
lane/fold/rep_K/b_scale/do_event/channel/metric), the analysis fields as npz, and
a config.json. ``rep_K`` is the PPE observation-noise multiplier (0 elsewhere);
``b_scale`` is the background-covariance amplitude the analysis used. R is the
augmented ``diag(sse) + diag(c*rep_var)``. Old metrics CSVs predate the
``b_scale`` column and must be regenerated, not merged. Scoring is in anomaly
space; CE, RMSE, and correlation are shift-invariant so the anomaly-space values
equal their degC counterparts.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from paleoreco.data import VARS
from paleoreco.data.splits import assign_event_label
from paleoreco.assim.observations import observations_at_age
from paleoreco.assim.innovation import obs_cell_index, r_diagonal
from paleoreco.assim.montecarlo import (
    FieldEnsembleAccumulator,
    assimilate_withheld_sweep,
    random_site_partitions,
)
from paleoreco.assim.priors import build_prior, per_age_neighbour
from paleoreco.assim.threedvar import ThreeDVar
from paleoreco.eval import da

BACKGROUNDS = ("climatological", "per_age")
C_REP = (0.06, 0.18)                            # per-channel representativeness scale, VARS order
B_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
LANE_IMPERFECT = "ppe_imperfect"


# ---------------------------------------------------------------------------
# Hold-out and observation geometry.
# ---------------------------------------------------------------------------
def every_nth_holdout(n_ages: int, stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Held-out (truth) and prior age indices: every ``stride``-th age is held out."""
    held = np.arange(stride - 1, n_ages, stride, dtype=np.int64)
    prior = np.setdiff1d(np.arange(n_ages, dtype=np.int64), held)
    return held, prior


def _obs_geometry(o: dict, lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray) -> dict:
    """Gather indices, error variance, and site coords for usable observations."""
    gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
    keep = safe_flat[gather] & (o["sse"] > 0)
    return {
        "gather": gather[keep],
        "sse": o["sse"][keep].astype(np.float64),
        "lat": o["lat"][keep],
        "lon": o["lon"][keep],
    }


def _draw_usable_age(long: pd.DataFrame, rng: np.random.Generator,
                     lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray) -> int:
    """A real proxy age whose borrowed network has at least one usable cell.

    Drawn from the shared ``rng`` so an external-truth run is reproducible from its
    seed; the age contributes only the network geometry, not any climate time.
    """
    ages_avail = long["age"].unique()
    for _ in range(len(ages_avail)):
        a = int(rng.choice(ages_avail))
        o = observations_at_age(long, a)
        if len(o.get("age", [])) and len(_obs_geometry(o, lats, lons, safe_flat)["gather"]):
            return a
    raise ValueError("no proxy age has a usable network on this grid")


def representativeness_variance(clim_mean: np.ndarray, safe_valid: np.ndarray) -> np.ndarray:
    """Per-cell subgrid variance: the spread of the climatology over its 3x3 block.

    Stands in for the point-vs-cell mismatch a proxy carries (large over steep
    gradients, small over homogeneous regions). Longitude wraps, latitude clamps,
    masked neighbours are excluded; cells with fewer than two valid neighbours get
    zero. Built from the prior climatology so it holds no truth-specific signal.
    """
    n_lat, n_lon = clim_mean.shape[1:]
    li, lj = np.arange(n_lat), np.arange(n_lon)
    vals, valids = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = np.clip(li + di, 0, n_lat - 1)
            jj = (lj + dj) % n_lon
            vals.append(clim_mean[:, ii][:, :, jj].astype(np.float64))
            valids.append(safe_valid[ii][:, jj])
    vals = np.stack(vals, axis=0)                      # (9, n_chan, n_lat, n_lon)
    m = np.stack(valids, axis=0)[:, None]              # (9, 1, n_lat, n_lon)
    cnt = m.sum(axis=0)
    mean = np.where(m, vals, 0.0).sum(axis=0) / np.clip(cnt, 1, None)
    var = np.where(m, (vals - mean) ** 2, 0.0).sum(axis=0) / np.clip(cnt, 1, None)
    return np.where((cnt >= 2) & safe_valid[None], var, 0.0)


# ---------------------------------------------------------------------------
# Naive baselines (no B, no model).
# ---------------------------------------------------------------------------
def _naive_geometry(lats: np.ndarray, lons: np.ndarray, geom: dict, n_chan: int) -> list:
    """Per-channel nearest index and IDW weights mapping obs values to a field.

    The distances depend only on the network, not the observed values, so they are
    built once and applied to every noisy/corrupted observation vector by
    :func:`_naive_apply`. Channels with no observations get ``None``.
    """
    n_lat, n_lon = len(lats), len(lons)
    lat_cell = np.repeat(lats, n_lon)
    lon_cell = np.tile(lons, n_lat)
    chan = geom["gather"] // (n_lat * n_lon)
    out = []
    for c in range(n_chan):
        sel = chan == c
        if not sel.any():
            out.append(None)
            continue
        d = da.great_circle_km(
            np.concatenate([lat_cell, geom["lat"][sel]]),
            np.concatenate([lon_cell, geom["lon"][sel]]),
        )[: len(lat_cell), len(lat_cell):]
        w = 1.0 / np.clip(d, 1.0, None) ** 2
        out.append({"sel": sel, "nearest": np.argmin(d, axis=1),
                    "weights": w / w.sum(axis=1, keepdims=True)})
    return out


def _naive_apply(kind: str, naive_geom: list, y_anom: np.ndarray,
                 shape: tuple[int, int, int]) -> np.ndarray:
    """Prior-free interpolated anomaly field for one observation vector.

    ``nearest`` copies each cell's nearest observation; ``idw`` is
    inverse-square-distance weighted. Channels with no observations stay zero.
    """
    field = np.zeros(shape, dtype=np.float64)
    for c, gc in enumerate(naive_geom):
        if gc is None:
            continue
        yv = y_anom[gc["sel"]]
        vals = yv[gc["nearest"]] if kind == "nearest" else gc["weights"] @ yv
        field[c] = vals.reshape(shape[1], shape[2])
    return field


# ---------------------------------------------------------------------------
# Metric assembly.
# ---------------------------------------------------------------------------
def _flatten(truth_anom: np.ndarray, recon_anom: np.ndarray, safe_valid: np.ndarray,
             truth_sel: np.ndarray, channel: int | None) -> tuple[np.ndarray, np.ndarray]:
    """1-D truth/recon anomalies over valid cells, a truth subset, and a channel."""
    cells = safe_valid
    t = truth_anom[truth_sel]
    r = recon_anom[truth_sel]
    if channel is None:
        t = t[:, :, cells]
        r = r[:, :, cells]
    else:
        t = t[:, channel, cells]
        r = r[:, channel, cells]
    return t.ravel(), r.ravel()


def _skill_rows(truth_anom: np.ndarray, recon_anom: np.ndarray, safe_valid: np.ndarray,
                events: np.ndarray, base: dict) -> list[dict]:
    """CE / correlation / RMSE rows, pooled and per channel, for all and per event."""
    rows = []
    groups = [("all", np.ones(len(events), dtype=bool))]
    for ev in sorted(set(events[events > 0])):
        groups.append((str(int(ev)), events == ev))
    channels = [("pooled", None)] + [(name, c) for c, name in enumerate(VARS)]

    for do_event, tsel in groups:
        for chan_name, c in channels:
            t, r = _flatten(truth_anom, recon_anom, safe_valid, tsel, c)
            zero = np.zeros_like(t)
            for metric, value in (
                ("ce", da.coefficient_of_efficiency(t, r, zero)),
                ("corr", da.pearson_r(t, r)),
                ("rmse", da.rmse(t, r)),
                ("rrmse", da.relative_rmse(t, r)),
            ):
                rows.append({**base, "do_event": do_event, "channel": chan_name,
                             "metric": metric, "value": value})
    return rows


def _ssim_rows(truth_anom: np.ndarray, recon_anom: np.ndarray, safe_valid: np.ndarray,
               events: np.ndarray, base: dict) -> list[dict]:
    """Masked SSIM rows, per channel and pooled, for all ages and per event.

    Field space only: SSIM is a 2-D structural metric with no observation-space
    analogue. ``data_range`` is per channel over the valid cells of the whole truth
    stack, fixed so per-truth SSIMs share the stabilising constants and average.
    The ``pooled`` channel is the mean of the per-channel SSIMs (multichannel SSIM),
    not the concatenation pooling used for CE/RMSE.
    """
    rows = []
    groups = [("all", np.ones(len(events), dtype=bool))]
    for ev in sorted(set(events[events > 0])):
        groups.append((str(int(ev)), events == ev))

    dr = [float(truth_anom[:, c, safe_valid].max() - truth_anom[:, c, safe_valid].min())
          for c in range(len(VARS))]

    for do_event, tsel in groups:
        per_chan = []
        for c, name in enumerate(VARS):
            s = float(np.mean([da.masked_ssim(truth_anom[ti, c], recon_anom[ti, c], safe_valid, dr[c])
                               for ti in np.flatnonzero(tsel)]))
            per_chan.append(s)
            rows.append({**base, "do_event": do_event, "channel": name,
                         "metric": "ssim", "value": s})
        rows.append({**base, "do_event": do_event, "channel": "pooled",
                     "metric": "ssim", "value": float(np.mean(per_chan))})
    return rows


def _append_csv(path: str, rows: list[dict]) -> None:
    """Append metric rows to the tidy CSV, writing the header once."""
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ---------------------------------------------------------------------------
# PPE.
# ---------------------------------------------------------------------------
def run_ppe(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    b_reg: str = "raw", eof_rank: int | None = None, localization_km: float | None = None,
    shrinkage_lambda: float | None = None,
    stride: int = 10, offset_yr: int = 2000, n_noise: int = 20,
    rep_levels: tuple[float, ...] = (0.0,), c_rep: tuple[float, ...] = C_REP,
    b_scales: tuple[float, ...] = B_SCALES,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
) -> pd.DataFrame:
    """Run the pseudo-proxy experiment for one B variant; persist artifacts.

    ``rep_levels`` sweeps the observation-noise multiplier k: the pseudo-obs is
    ``H(x_true) + k*(e + w)`` with ``e ~ N(0, sse)`` and ``w ~ N(0, c*rep_var)``.
    R is held at the augmented ``diag(sse) + diag(c*rep_var)``, so k=1 matches the
    generating error (well-specified) and k>1 stresses it. ``b_scales`` scales B by
    each value at assimilation time, shared through one eig factorization per
    network. Returns the metrics rows and writes analysis fields, skill-vs-distance
    curves, and config under ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    rep_levels = tuple(float(k) for k in rep_levels)
    b_scales = tuple(float(b) for b in b_scales)
    c = np.asarray(c_rep, dtype=np.float64)
    n_k, n_b = len(rep_levels), len(b_scales)

    held, prior_idx = every_nth_holdout(len(ages), stride)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        b_reg=b_reg, eof_rank=eof_rank, localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda)
    clim_mean = prior.clim_mean.astype(np.float64)
    clim_flat = clim_mean.ravel()
    safe_valid = prior.safe_valid
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    s_flat = representativeness_variance(clim_mean, safe_valid).ravel()
    tv = ThreeDVar(prior.B, shape)

    truth_anoms, post_var = [], []
    recon = {bg: [] for bg in BACKGROUNDS}              # each item (n_k, n_b, 2, n_lat, n_lon)
    naive = {"nearest": [], "idw": []}                 # each item (n_k, 2, n_lat, n_lon)
    truth_ages, dist_per_truth = [], []
    bg_per_age = []                                     # the per-age first guess per truth

    for ti in held:
        truth_age = int(ages[ti])
        o = observations_at_age(long, truth_age)
        if len(o.get("age", [])) == 0:
            continue
        geom = _obs_geometry(o, lats, lons, safe_flat)
        g = geom["gather"]
        if len(g) == 0:
            continue

        truth = cube[ti].astype(np.float64)
        truth_anom = truth - clim_mean
        truth_at_obs = truth.ravel()[g] - clim_flat[g]
        rep_obs = c[g // n_cells] * s_flat[g]            # w variance, c*rep_var per obs

        backgrounds = {
            "climatological": np.zeros(int(np.prod(shape))),
            "per_age": cube[int(np.searchsorted(ages, per_age_neighbour(truth_age, prior.ages, offset_yr)))].ravel() - clim_flat,
        }
        r_diag = r_diagonal(g, geom["sse"], s_flat, c, n_cells)
        gain = tv.prepare_sweep(g, r_diag, b_scales)
        naive_geom = _naive_geometry(lats, lons, geom, len(VARS))

        sum_recon = {bg: np.zeros((n_k, n_b, *shape)) for bg in BACKGROUNDS}
        sum_naive = {kind: np.zeros((n_k, *shape)) for kind in naive}
        for _ in range(n_noise):
            e = rng.normal(0.0, np.sqrt(geom["sse"]))
            w = rng.normal(0.0, np.sqrt(rep_obs))
            for ki, k in enumerate(rep_levels):
                y_k = truth_at_obs + k * (e + w)
                for bg, bg_anom in backgrounds.items():
                    res = tv.apply_sweep(gain, y_k, bg_anom)
                    for bj in range(n_b):
                        sum_recon[bg][ki, bj] += res[bj].mean_anom
                for kind in naive:
                    sum_naive[kind][ki] += _naive_apply(kind, naive_geom, y_k, shape)

        truth_anoms.append(truth_anom)
        post_var.append(tv.post_var_sweep(gain).reshape(n_b, *shape))   # (n_b, ...), rep-level free
        truth_ages.append(truth_age)
        bg_per_age.append(backgrounds["per_age"].reshape(shape))
        for bg in BACKGROUNDS:
            recon[bg].append(sum_recon[bg] / n_noise)
        for kind in naive:
            naive[kind].append(sum_naive[kind] / n_noise)
        dist_per_truth.append(da.nearest_obs_distance(lats, lons, geom["lat"], geom["lon"]))

    truth_anoms = np.stack(truth_anoms)                # (T, 2, n_lat, n_lon)
    post_var = np.stack(post_var)                      # (T, n_b, 2, n_lat, n_lon)
    truth_ages = np.asarray(truth_ages)
    events = assign_event_label(truth_ages)
    recon = {bg: np.stack(v) for bg, v in recon.items()}   # (T, n_k, n_b, 2, n_lat, n_lon)
    naive = {kind: np.stack(v) for kind, v in naive.items()}  # (T, n_k, 2, n_lat, n_lon)
    bg_per_age = np.stack(bg_per_age)                       # (T, 2, n_lat, n_lon)

    # Metric rows: 3DVar over (rep_level, b_scale); naive baselines are B-free.
    rows = []
    tv_safe = truth_anoms[:, :, safe_valid]
    for ki, k in enumerate(rep_levels):
        for bj, kb in enumerate(b_scales):
            meta = {"method": "3dvar", "B_reg": b_reg, "lane": "ppe", "fold": -1,
                    "rep_K": k, "b_scale": kb}
            pv_safe = np.broadcast_to(post_var[:, bj][:, :, safe_valid].clip(1e-12, None),
                                      tv_safe.shape)
            for bg in BACKGROUNDS:
                base = {**meta, "background": bg}
                rows += _skill_rows(truth_anoms, recon[bg][:, ki, bj], safe_valid, events, base)
                rows += _ssim_rows(truth_anoms, recon[bg][:, ki, bj], safe_valid, events, base)
                bias, disp = da.rcrv(tv_safe, recon[bg][:, ki, bj][:, :, safe_valid], pv_safe)
                rows.append({**base, "do_event": "all", "channel": "pooled",
                             "metric": "rcrv_bias", "value": bias})
                rows.append({**base, "do_event": "all", "channel": "pooled",
                             "metric": "rcrv_disp", "value": disp})
        for kind in naive:
            base = {"method": kind, "B_reg": "none", "lane": "ppe", "fold": -1,
                    "rep_K": k, "b_scale": 1.0, "background": "none"}
            rows += _skill_rows(truth_anoms, naive[kind][:, ki], safe_valid, events, base)
            rows += _ssim_rows(truth_anoms, naive[kind][:, ki], safe_valid, events, base)

    # Skill vs distance to nearest observation, per background, rep_level, and b_scale.
    if dist_edges_km is None:
        dist_edges_km = np.array([0, 500, 1000, 2000, 3000, 5000, 8000, 20000], dtype=float)
    dist = np.stack(dist_per_truth)                          # (T, n_lat*n_lon)
    cell_mask = safe_valid.ravel()
    dist_pool = np.tile(dist[:, cell_mask], (1, len(VARS))).ravel()
    t_pool = tv_safe.reshape(len(truth_ages), -1).ravel()
    curves = {}
    for bg in BACKGROUNDS:
        ce = np.array([[da.skill_vs_distance(
            t_pool, recon[bg][:, ki, bj][:, :, safe_valid].reshape(len(truth_ages), -1).ravel(),
            np.zeros_like(t_pool), dist_pool, dist_edges_km)["ce"]
            for bj in range(n_b)] for ki in range(n_k)])
        curves[bg] = ce                                      # (n_k, n_b, n_bins)

    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(
        os.path.join(out_dir, "ppe_analysis.npz"),
        truth_anom=truth_anoms, clim_mean=clim_mean, safe_valid=safe_valid,
        post_var=np.moveaxis(post_var, 0, 1), prior_var=tv.diagB.reshape(shape),  # (n_b, T, ...)
        background_per_age=bg_per_age, truth_ages=truth_ages, events=events,
        rep_levels=np.asarray(rep_levels), b_scales=np.asarray(b_scales),
        **{f"recon_{bg}": np.moveaxis(recon[bg], 0, 2) for bg in BACKGROUNDS},  # (n_k, n_b, T, ...)
        **{f"naive_{k}": np.moveaxis(naive[k], 1, 0) for k in naive},           # (n_k, T, ...)
    )
    np.savez_compressed(os.path.join(out_dir, "ppe_skill_vs_distance.npz"),
                        edges=dist_edges_km, rep_levels=np.asarray(rep_levels),
                        b_scales=np.asarray(b_scales),
                        **{bg: curves[bg] for bg in BACKGROUNDS})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"lane": "ppe", "b_reg": b_reg, "eof_rank": eof_rank,
                   "localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                   "stride": stride, "offset_yr": offset_yr, "n_noise": n_noise,
                   "rep_levels": list(rep_levels), "c_rep": list(c_rep),
                   "b_scales": list(b_scales), "seed": seed,
                   "prior_meta": prior.meta}, f, indent=2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Imperfect-model PPE: out-of-distribution truths from another model.
# ---------------------------------------------------------------------------
def run_ppe_imperfect(
    truth_cube: np.ndarray, cube: np.ndarray, ages: np.ndarray,
    lats: np.ndarray, lons: np.ndarray, valid: np.ndarray,
    long: pd.DataFrame, out_dir: str, *,
    b_reg: str = "raw", eof_rank: int | None = None, localization_km: float | None = None,
    shrinkage_lambda: float | None = None,
    rep_levels: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    c_rep: tuple[float, ...] = C_REP, b_scales: tuple[float, ...] = B_SCALES,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
) -> pd.DataFrame:
    """PPE scoring a LOVECLIM prior against out-of-distribution truths from another model.

    ``truth_cube`` is an external ``(n, 2, n_lat, n_lon)`` degC stack on the prior
    grid. B and the climatology come from the full ``cube`` with no holdout (the
    truths are not LOVECLIM states); each truth is anomalised by its OWN mean so the
    cross-model offset cancels and the test is whether the prior covariance
    reconstructs the truth's variability. Each truth borrows a randomly drawn real
    proxy age's network geometry (cells and sse); the obs values come from the truth.
    Single climatological background. Persists the same artifacts as :func:`run_ppe`.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    rep_levels = tuple(float(k) for k in rep_levels)
    b_scales = tuple(float(b) for b in b_scales)
    c = np.asarray(c_rep, dtype=np.float64)
    n_k, n_b = len(rep_levels), len(b_scales)

    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid,
                        b_reg=b_reg, eof_rank=eof_rank, localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda)
    clim_mean = prior.clim_mean.astype(np.float64)
    safe_valid = prior.safe_valid
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    s_flat = representativeness_variance(clim_mean, safe_valid).ravel()
    tv = ThreeDVar(prior.B, shape)

    truth_cube = np.asarray(truth_cube, dtype=np.float64)
    ccsm4_clim = truth_cube.mean(axis=0)                 # own-mean frame, cross-model bias removed
    truth_anom_cube = truth_cube - ccsm4_clim
    zero_bg = np.zeros(int(np.prod(shape)))

    truth_anoms, post_var = [], []
    recon = []                                           # each item (n_k, n_b, 2, n_lat, n_lon)
    naive = {"nearest": [], "idw": []}
    dist_per_truth, drawn_ages = [], []

    for truth_anom in truth_anom_cube:
        k_age = _draw_usable_age(long, rng, lats, lons, safe_flat)
        o = observations_at_age(long, k_age)
        geom = _obs_geometry(o, lats, lons, safe_flat)
        g = geom["gather"]
        truth_at_obs = truth_anom.ravel()[g]             # obs values from the truth, H = nearest cell
        rep_obs = c[g // n_cells] * s_flat[g]
        r_diag = r_diagonal(g, geom["sse"], s_flat, c, n_cells)
        gain = tv.prepare_sweep(g, r_diag, b_scales)
        naive_geom = _naive_geometry(lats, lons, geom, len(VARS))

        e = rng.normal(0.0, np.sqrt(geom["sse"]))
        w = rng.normal(0.0, np.sqrt(rep_obs))
        sum_recon = np.zeros((n_k, n_b, *shape))
        sum_naive = {kind: np.zeros((n_k, *shape)) for kind in naive}
        for ki, k in enumerate(rep_levels):
            y_k = truth_at_obs + k * (e + w)
            res = tv.apply_sweep(gain, y_k, zero_bg)
            for bj in range(n_b):
                sum_recon[ki, bj] = res[bj].mean_anom
            for kind in naive:
                sum_naive[kind][ki] = _naive_apply(kind, naive_geom, y_k, shape)

        truth_anoms.append(truth_anom)
        post_var.append(tv.post_var_sweep(gain).reshape(n_b, *shape))
        recon.append(sum_recon)
        for kind in naive:
            naive[kind].append(sum_naive[kind])
        dist_per_truth.append(da.nearest_obs_distance(lats, lons, geom["lat"], geom["lon"]))
        drawn_ages.append(k_age)

    truth_anoms = np.stack(truth_anoms)                  # (T, 2, n_lat, n_lon)
    post_var = np.stack(post_var)                        # (T, n_b, 2, n_lat, n_lon)
    recon = np.stack(recon)                              # (T, n_k, n_b, 2, n_lat, n_lon)
    naive = {kind: np.stack(v) for kind, v in naive.items()}
    events = np.zeros(len(truth_anoms), dtype=np.int64)  # no DO-event labels for external truths

    rows = []
    tv_safe = truth_anoms[:, :, safe_valid]
    for ki, k in enumerate(rep_levels):
        for bj, kb in enumerate(b_scales):
            base = {"method": "3dvar", "B_reg": b_reg, "lane": LANE_IMPERFECT, "fold": -1,
                    "rep_K": k, "b_scale": kb, "background": "climatological"}
            pv_safe = np.broadcast_to(post_var[:, bj][:, :, safe_valid].clip(1e-12, None),
                                      tv_safe.shape)
            rows += _skill_rows(truth_anoms, recon[:, ki, bj], safe_valid, events, base)
            rows += _ssim_rows(truth_anoms, recon[:, ki, bj], safe_valid, events, base)
            bias, disp = da.rcrv(tv_safe, recon[:, ki, bj][:, :, safe_valid], pv_safe)
            rows.append({**base, "do_event": "all", "channel": "pooled",
                         "metric": "rcrv_bias", "value": bias})
            rows.append({**base, "do_event": "all", "channel": "pooled",
                         "metric": "rcrv_disp", "value": disp})
        for kind in naive:
            base = {"method": kind, "B_reg": "none", "lane": LANE_IMPERFECT, "fold": -1,
                    "rep_K": k, "b_scale": 1.0, "background": "none"}
            rows += _skill_rows(truth_anoms, naive[kind][:, ki], safe_valid, events, base)
            rows += _ssim_rows(truth_anoms, naive[kind][:, ki], safe_valid, events, base)

    if dist_edges_km is None:
        dist_edges_km = np.array([0, 500, 1000, 2000, 3000, 5000, 8000, 20000], dtype=float)
    dist = np.stack(dist_per_truth)
    cell_mask = safe_valid.ravel()
    dist_pool = np.tile(dist[:, cell_mask], (1, len(VARS))).ravel()
    t_pool = tv_safe.reshape(len(truth_anoms), -1).ravel()
    curve = np.array([[da.skill_vs_distance(
        t_pool, recon[:, ki, bj][:, :, safe_valid].reshape(len(truth_anoms), -1).ravel(),
        np.zeros_like(t_pool), dist_pool, dist_edges_km)["ce"]
        for bj in range(n_b)] for ki in range(n_k)])     # (n_k, n_b, n_bins)

    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(
        os.path.join(out_dir, "ppe_imperfect_analysis.npz"),
        truth_anom=truth_anoms, clim_mean=clim_mean, ccsm4_clim=ccsm4_clim,
        safe_valid=safe_valid, post_var=np.moveaxis(post_var, 0, 1),       # (n_b, T, ...)
        prior_var=tv.diagB.reshape(shape), drawn_ages=np.asarray(drawn_ages),
        rep_levels=np.asarray(rep_levels), b_scales=np.asarray(b_scales),
        recon_climatological=np.moveaxis(recon, 0, 2),                     # (n_k, n_b, T, ...)
        **{f"naive_{k}": np.moveaxis(naive[k], 1, 0) for k in naive},      # (n_k, T, ...)
    )
    np.savez_compressed(os.path.join(out_dir, "ppe_imperfect_skill_vs_distance.npz"),
                        edges=dist_edges_km, rep_levels=np.asarray(rep_levels),
                        b_scales=np.asarray(b_scales), climatological=curve)
    with open(os.path.join(out_dir, "ppe_imperfect_config.json"), "w") as f:
        json.dump({"lane": LANE_IMPERFECT, "b_reg": b_reg, "eof_rank": eof_rank,
                   "localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                   "n_truths": int(len(truth_anoms)), "rep_levels": list(rep_levels),
                   "c_rep": list(c_rep), "b_scales": list(b_scales), "seed": seed,
                   "prior_meta": prior.meta}, f, indent=2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-proxy withholding.
# ---------------------------------------------------------------------------
def _site_folds(long: pd.DataFrame, k: int, kind: str, seed: int) -> list[np.ndarray]:
    """Partition sites into ``k`` folds, randomly or by spatial cluster."""
    sites = long.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    ids = sites["site"].to_numpy()
    if kind == "random":
        rng = np.random.default_rng(seed)
        return [f for f in np.array_split(rng.permutation(ids), k)]
    if kind == "spatial":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(
            sites[["lat", "lon"]].to_numpy()
        )
        return [ids[labels == c] for c in range(k)]
    raise ValueError(f"unknown fold kind {kind!r}")


def _withholding_rows(actual: np.ndarray, pred: np.ndarray, channel: np.ndarray,
                      base: dict) -> list[dict]:
    """CE / correlation / RMSE in observation space, pooled and per channel."""
    rows = []
    groups = [("pooled", np.ones(len(actual), dtype=bool))]
    groups += [(name, channel == c) for c, name in enumerate(VARS)]
    for chan_name, sel in groups:
        if sel.sum() < 2:
            continue
        a, p = actual[sel], pred[sel]
        for metric, value in (
            ("ce", da.coefficient_of_efficiency(a, p, np.zeros_like(a))),
            ("corr", da.pearson_r(a, p)),
            ("rmse", da.rmse(a, p)),
            ("rrmse", da.relative_rmse(a, p)),
        ):
            rows.append({**base, "do_event": "all", "channel": chan_name,
                         "metric": metric, "value": value})
    return rows


def run_withholding(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    b_reg: str = "raw", eof_rank: int | None = None, localization_km: float | None = None,
    shrinkage_lambda: float | None = None,
    k_folds: int = 5, fold_kind: str = "random", c_rep: tuple[float, ...] = C_REP,
    b_scales: tuple[float, ...] = B_SCALES, seed: int = 0,
) -> pd.DataFrame:
    """Withhold whole sites, assimilate the rest, score predictions at withheld sites.

    ``long`` must carry per-site climatology ``my`` (real observations enter in
    anomaly space ``y - my`` so the proxy-vs-model offset cancels). The prior uses
    all ages: the held-out quantity is real proxies, not model states. R is the
    augmented ``diag(sse) + diag(c*rep_var)``; ``b_scales`` scales B per value.
    """
    os.makedirs(out_dir, exist_ok=True)
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    b_scales = tuple(float(b) for b in b_scales)
    c = np.asarray(c_rep, dtype=np.float64)
    n_b = len(b_scales)
    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid,
                        b_reg=b_reg, eof_rank=eof_rank, localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda)
    clim_flat = prior.clim_mean.astype(np.float64).ravel()
    safe_flat = np.broadcast_to(prior.safe_valid, shape).ravel()
    rep_flat = representativeness_variance(prior.clim_mean.astype(np.float64), prior.safe_valid).ravel()
    tv = ThreeDVar(prior.B, shape)

    obs_ages = np.intersect1d(long["age"].unique(), ages)
    folds = _site_folds(long, k_folds, fold_kind, seed)
    lane = f"withholding_{fold_kind}"

    rows = []
    pooled = {"actual": [], "channel": [], "pred": {bg: [] for bg in BACKGROUNDS}}
    for f, withheld in enumerate(folds):
        wset = set(withheld.tolist())
        per_fold = {"actual": [], "channel": [], "pred": {bg: [] for bg in BACKGROUNDS}}
        for age in obs_ages:
            o = observations_at_age(long, int(age))
            gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
            keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
            is_w = np.array([s in wset for s in o["site"]])
            kept = keep & ~is_w
            wkeep = keep & is_w
            if kept.sum() == 0 or wkeep.sum() == 0:
                continue

            y_anom = (o["y"][kept] - o["my"][kept]).astype(np.float64)
            r_diag = r_diagonal(gather[kept], o["sse"][kept].astype(np.float64),
                                rep_flat, c, n_cells)
            gain = tv.prepare_sweep(gather[kept], r_diag, b_scales)
            ai = int(np.searchsorted(ages, age))
            backgrounds = {
                "climatological": np.zeros(int(np.prod(shape))),
                "per_age": cube[ai].ravel() - clim_flat,
            }
            gw = gather[wkeep]
            per_fold["actual"].append((o["y"][wkeep] - o["my"][wkeep]).astype(np.float64))
            per_fold["channel"].append(gw // n_cells)
            for bg, bg_anom in backgrounds.items():
                res = tv.apply_sweep(gain, y_anom, bg_anom)
                per_fold["pred"][bg].append(np.stack([res[bj].predict_obs(gw) for bj in range(n_b)]))

        a = np.concatenate(per_fold["actual"])
        ch = np.concatenate(per_fold["channel"])
        for bg in BACKGROUNDS:
            p = np.concatenate(per_fold["pred"][bg], axis=1)          # (n_b, n_w)
            for bj, kb in enumerate(b_scales):
                rows += _withholding_rows(a, p[bj], ch, {"method": "3dvar", "B_reg": b_reg,
                                                         "lane": lane, "fold": f, "rep_K": 0.0,
                                                         "b_scale": kb, "background": bg})
            pooled["pred"][bg].append(p)
        pooled["actual"].append(a)
        pooled["channel"].append(ch)

    a = np.concatenate(pooled["actual"])
    ch = np.concatenate(pooled["channel"])
    dumps = {"actual": a, "channel": ch, "b_scales": np.asarray(b_scales)}
    for bg in BACKGROUNDS:
        p = np.concatenate(pooled["pred"][bg], axis=1)                # (n_b, n_w)
        for bj, kb in enumerate(b_scales):
            rows += _withholding_rows(a, p[bj], ch, {"method": "3dvar", "B_reg": b_reg,
                                                     "lane": lane, "fold": -1, "rep_K": 0.0,
                                                     "b_scale": kb, "background": bg})
        dumps[f"{bg}_pred"] = p

    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_predictions.npz"), **dumps)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as fh:
        json.dump({"lane": lane, "b_reg": b_reg, "eof_rank": eof_rank,
                   "localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                   "k_folds": k_folds, "fold_kind": fold_kind, "c_rep": list(c_rep),
                   "b_scales": list(b_scales), "seed": seed,
                   "prior_meta": prior.meta}, fh, indent=2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LMR-style Monte Carlo proxy-resampling ensemble.
# ---------------------------------------------------------------------------
def _mean_posterior_var_field(
    tv: ThreeDVar, long: pd.DataFrame, obs_ages: np.ndarray,
    lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray, shape: tuple[int, int, int],
    rep_flat: np.ndarray, c: np.ndarray, b_scales: tuple[float, ...],
) -> np.ndarray:
    """Full-network posterior variance per b_scale, averaged over the observed ages.

    The analysis variance depends only on the network, B, and R, not on the values
    assimilated, so this is the analysis uncertainty a single realization sees
    before any sites are withheld. Returns ``(n_b_scale, *shape)``.
    """
    n_cells = shape[1] * shape[2]
    acc = np.zeros((len(b_scales), *shape))
    n = 0
    for age in obs_ages:
        o = observations_at_age(long, int(age))
        gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
        keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
        if keep.sum() == 0:
            continue
        r_diag = r_diagonal(gather[keep], o["sse"][keep].astype(np.float64), rep_flat, c, n_cells)
        gain = tv.prepare_sweep(gather[keep], r_diag, b_scales)
        acc += tv.post_var_sweep(gain).reshape(len(b_scales), *shape)
        n += 1
    return acc / max(n, 1)


def run_montecarlo_withholding(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    b_reg: str = "raw", eof_rank: int | None = None, localization_km: float | None = None,
    shrinkage_lambda: float | None = None,
    n_realizations: int = 100, assim_frac: float = 0.75, c_rep: tuple[float, ...] = C_REP,
    b_scales: tuple[float, ...] = B_SCALES, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Resample the proxy network many times, assimilate each draw, verify withheld.

    Each realization assimilates a random ``assim_frac`` of sites and predicts the
    rest, all sharing one B (Hakim et al. 2016). The metrics rows carry the
    per-realization skill (``fold`` 0..n-1) and the skill and calibration of the
    ensemble-mean prediction (``fold`` -1); the npz artifacts carry the
    ensemble-mean reconstruction with its across-realization spread and the pooled
    per-observation ensemble. R is the augmented ``diag(sse) + diag(c*rep_var)``;
    every product carries a leading b_scale axis. ``long`` must carry ``my``.
    """
    os.makedirs(out_dir, exist_ok=True)
    shape = (len(VARS), len(lats), len(lons))
    b_scales = tuple(float(b) for b in b_scales)
    c = np.asarray(c_rep, dtype=np.float64)
    n_b = len(b_scales)
    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid,
                        b_reg=b_reg, eof_rank=eof_rank, localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda)
    clim_mean = prior.clim_mean.astype(np.float64)
    clim_flat = clim_mean.ravel()
    safe_flat = np.broadcast_to(prior.safe_valid, shape).ravel()
    rep_flat = representativeness_variance(clim_mean, prior.safe_valid).ravel()
    tv = ThreeDVar(prior.B, shape)

    obs_ages = np.intersect1d(long["age"].unique(), ages)
    partitions = random_site_partitions(long["site"].unique(), n_realizations, assim_frac, seed)
    lane = "withholding_montecarlo"

    rows, pooled = [], []
    accum = {(bg, bj): FieldEnsembleAccumulator() for bg in BACKGROUNDS for bj in range(n_b)}
    for r, withheld in enumerate(partitions):
        if progress_every and (r + 1) % progress_every == 0:
            print(f"  realization {r + 1}/{n_realizations}", flush=True)
        sweep = assimilate_withheld_sweep(tv, long, obs_ages, ages, lats, lons,
                                          safe_flat, clim_flat, cube, clim_mean, withheld,
                                          rep_flat, c, b_scales,
                                          backgrounds=BACKGROUNDS, want_field=True)
        for bg in BACKGROUNDS:
            for bj, kb in enumerate(b_scales):
                accum[(bg, bj)].update(sweep.field[bg][bj])
                base = {"method": "3dvar", "B_reg": b_reg, "lane": lane, "fold": r,
                        "rep_K": 0.0, "b_scale": kb, "background": bg}
                rows += _withholding_rows(sweep.actual, sweep.pred[bg][bj], sweep.channel, base)
        df = pd.DataFrame({"site": sweep.site, "channel": sweep.channel, "age": sweep.age,
                           "actual": sweep.actual, "sse": sweep.sse, "rep": sweep.rep})
        for bj in range(n_b):
            df[f"post_var_b{bj}"] = sweep.post_var[bj]
        for bg in BACKGROUNDS:
            df[f"prior_pred_{bg}"] = sweep.prior_pred[bg]
            for bj in range(n_b):
                df[f"pred_{bg}_b{bj}"] = sweep.pred[bg][bj]
        pooled.append(df)

    # Per-observation ensemble: pool the draws that withheld each (site, channel, age).
    pooled = pd.concat(pooled, ignore_index=True)
    grp = pooled.groupby(["site", "channel", "age"])
    agg = grp.agg(actual=("actual", "first"), sse=("sse", "first"), rep=("rep", "first"),
                  n=("actual", "size")).reset_index()
    for bj in range(n_b):
        agg[f"post_var_b{bj}"] = grp[f"post_var_b{bj}"].mean().to_numpy()
    for bg in BACKGROUNDS:
        # The background first guess at a withheld cell does not vary across draws.
        agg[f"prior_pred_{bg}"] = grp[f"prior_pred_{bg}"].first().to_numpy()
        for bj in range(n_b):
            agg[f"ens_mean_{bg}_b{bj}"] = grp[f"pred_{bg}_b{bj}"].mean().to_numpy()
            agg[f"ens_std_{bg}_b{bj}"] = grp[f"pred_{bg}_b{bj}"].std(ddof=0).to_numpy()

    actual = agg["actual"].to_numpy()
    channel = agg["channel"].to_numpy()
    obs_err = (agg["sse"] + agg["rep"]).to_numpy()                    # sse + c*rep_var at withheld cell
    post_v = np.stack([agg[f"post_var_b{bj}"].to_numpy() for bj in range(n_b)])
    dumps = {"n_withheld_per_obs": agg["n"].to_numpy(), "actual": actual,
             "channel": channel, "b_scales": np.asarray(b_scales)}
    for bg in BACKGROUNDS:
        ens_mean = np.stack([agg[f"ens_mean_{bg}_b{bj}"].to_numpy() for bj in range(n_b)])
        ens_std = np.stack([agg[f"ens_std_{bg}_b{bj}"].to_numpy() for bj in range(n_b)])
        total_var = (ens_std ** 2 + post_v + obs_err[None, :]).clip(1e-12, None)
        for bj, kb in enumerate(b_scales):
            base = {"method": "3dvar", "B_reg": b_reg, "lane": lane, "fold": -1,
                    "rep_K": 0.0, "b_scale": kb, "background": bg}
            rows += _withholding_rows(actual, ens_mean[bj], channel, base)
            bias, disp = da.rcrv(actual, ens_mean[bj], total_var[bj])
            rows.append({**base, "do_event": "all", "channel": "pooled", "metric": "rcrv_bias", "value": bias})
            rows.append({**base, "do_event": "all", "channel": "pooled", "metric": "rcrv_disp", "value": disp})
        dumps.update({f"{bg}_ens_mean_pred": ens_mean, f"{bg}_ens_std_pred": ens_std,
                      f"{bg}_prior_pred": agg[f"prior_pred_{bg}"].to_numpy(),
                      f"{bg}_total_var": total_var})

    ens = {}
    for bg in BACKGROUNDS:
        fin = [accum[(bg, bj)].finalize() for bj in range(n_b)]
        ens[f"recon_mean_{bg}"] = np.stack([f[0] for f in fin])      # (n_b, *shape)
        ens[f"recon_std_{bg}"] = np.stack([f[1] for f in fin])
    post_var_field = _mean_posterior_var_field(tv, long, obs_ages, lats, lons, safe_flat,
                                               shape, rep_flat, c, b_scales)

    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, "montecarlo_ensemble.npz"),
                        obs_ages=obs_ages, clim_mean=clim_mean, safe_valid=prior.safe_valid,
                        post_var=post_var_field, prior_var=tv.diagB.reshape(shape),
                        b_scales=np.asarray(b_scales), **ens)
    np.savez_compressed(os.path.join(out_dir, "montecarlo_predictions.npz"), **dumps)
    with open(os.path.join(out_dir, "montecarlo_config.json"), "w") as fh:
        json.dump({"lane": lane, "b_reg": b_reg, "eof_rank": eof_rank,
                   "localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                   "n_realizations": n_realizations, "assim_frac": assim_frac,
                   "c_rep": list(c_rep), "b_scales": list(b_scales), "seed": seed,
                   "prior_meta": prior.meta}, fh, indent=2)
    return pd.DataFrame(rows)
