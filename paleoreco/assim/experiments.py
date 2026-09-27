"""Runners for the three evaluation lanes, and the grids that tune over them.

* :func:`run_ppe`: the pseudo-proxy snapshot lane. The older half of the archive builds the
  prior; each younger-half state is a truth, observed through two real network geometries,
  one selecting the operating point and one reporting it.
* :func:`run_trajectory`: the pseudo-proxy time series lane. The same split, reconstructed in
  sequence, with each observation read at its own sample's offset in time, so skill can be
  resolved by timescale.
* :func:`run_withholding`: the real-proxy lane. Nested cross-validation over proxy sites.

The ``*_pixel_grid`` wrappers tune the covariance taper jointly with ``b_scale``; the
``run_hgaoenkf_*`` wrappers tune the analog ensemble size and hybrid weight at an inherited
taper. Both keep fields for the winning configuration only.

Every lane writes a long-format ``metrics.csv`` (one row per configuration, ``b_scale``,
split, event, channel and metric), an analysis npz and a config.json. ``split`` is
``selection`` (used to choose the operating point) or ``test`` (reported). A row's ``method``
combines the estimator with its treatment of observation staleness (:func:`method_label`).
Scoring is in anomaly space. Prior-free ``nearest`` and ``idw`` reference rows are tagged
``background="none"``.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from paleoreco.data import VARS
from paleoreco.data.splits import chronological_half_split
from paleoreco.assim.background import temporal_structure_function
from paleoreco.assim.method import Method
from paleoreco.assim.observations import (
    TEMPORAL_ADD,
    TEMPORAL_DEFLATE,
    TEMPORAL_MODES,
    TEMPORAL_OFF,
    apply_temporal_error,
    observations_at_age,
    representativeness_variance,
    sample_block_centres,
    temporal_terms,
)
from paleoreco.assim.analog import (
    ANALOG_CORRELATION,
    ANALOG_CORRELATION_PERCHAN,
    ANALOG_EVIDENCE,
    ANALOG_MISFIT,
    EVIDENCE_SCALE,
)
from paleoreco.assim.hgaoenkf import make_hgaoenkf
from paleoreco.assim.innovation import nearest_age_index, obs_cell_index
from paleoreco.assim.priors import Prior, build_prior, great_circle_km_between
from paleoreco.assim.threedvar import ThreeDVar
from paleoreco.eval import calibration, da

# Builds an estimator from a prior and the field shape. The estimator must provide
# prepare_sweep and apply_sweep and return AnalysisResult objects.
MethodFactory = Callable[[Prior, "tuple[int, int, int]"], Method]

B_SCALES = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
# Taper grid (localization, shrinkage, channel coupling); ``None`` means no localization.
# Lengthscales are chordal, so the relevant scale is the 12742 km diameter.
LOCALIZATION_KM_GRID = (None, 12500.0, 20000.0)
SHRINKAGE_GRID = (0.0, 0.25, 0.5)
ALPHA_GRID = (0.0, 0.5, 1.0)
SEL_TOL = 0.0   # 0 = pure argmin of selection RRMSE; >0 prefers the simpler config within this relative band
# Analog grid: ensemble size and hybrid weight. The weight spans Sun et al.'s AOEnKF-B (0)
# to AOEnKF (1).
K_GRID = (20, 40, 60, 100)
HYBRID_W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# Grids for the extension terms. Zero on every axis is the published estimator.
TENDENCY_THETA_GRID = (0.0, 1.0)
TENDENCY_LAG_YR_GRID = (200.0, 400.0)
# The redundancy penalty is held at zero rather than swept.
REDUNDANCY_THETA_GRID = (0.0,)
# The MTA lags, fixed a priori rather than tuned; only the weight theta is gridded.
MT_REFERENCE_LAG_YR = 400.0
MT_EXTRA_LAGS_YR = (200.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0)
MT_CURVATURE_YR = (400.0, 1600.0)
MT_THETA_GRID = (0.0, 1.0, 2.0)
# Stacks the sensitivity pass compares, as ``(extra lags, curvature)``. ``1lag`` is the
# single first difference.
MT_STACKS = {
    "1lag": ((), ()),
    "3lag": ((800.0, 1600.0), ()),
    "7lag": (MT_EXTRA_LAGS_YR, ()),
    "7lag_curv": (MT_EXTRA_LAGS_YR, MT_CURVATURE_YR),
}
# Exclusion band around the target age, and the widths the sensitivity pass sweeps.
EXCLUDE_YR = 1000.0
EXCLUDE_YR_GRID = (0.0, 1000.0, 2000.0)
# Lengthscales for the analog covariance's own taper; ``None`` inherits the static one.
ANALOG_LOCALIZATION_GRID_PPE = (None, 5000.0, 8000.0, 12500.0, 20000.0)
# Minimum observations for a network to be drawn.
MIN_OBS = 10
# Attempts to find a borrowed network whose offsets stay on the archive at one age.
_MAX_SHAPE_DRAWS = 20
LANE_PPE = "ppe"
LANE_TRAJECTORY = "trajectory"

# Estimator tags, combined with the staleness treatment by :func:`method_label`.
ESTIMATOR_3DVAR = "3dvar"
ESTIMATOR_HGAOENKF = "hgaoenkf"
# MTA-HGAOEnKF.
ESTIMATOR_HGAOENKF_MT = f"{ESTIMATOR_HGAOENKF}_mt"
_TEMPORAL_SUFFIX = {TEMPORAL_OFF: "", TEMPORAL_ADD: "_temporal_add",
                    TEMPORAL_DEFLATE: "_temporal_deflate"}
# Estimator tag per selection rule; the published misfit rule keeps the bare name.
_HGAOENKF_TAG = {
    ANALOG_MISFIT: ESTIMATOR_HGAOENKF,
    ANALOG_EVIDENCE: f"{ESTIMATOR_HGAOENKF}_evidence",
    ANALOG_CORRELATION: f"{ESTIMATOR_HGAOENKF}_correlation",
    ANALOG_CORRELATION_PERCHAN: f"{ESTIMATOR_HGAOENKF}_correlation_perchan",
}


def method_label(estimator: str, mode: str = TEMPORAL_OFF) -> str:
    """The metrics-CSV ``method`` for one (estimator, staleness treatment) pair."""
    return f"{estimator}{_TEMPORAL_SUFFIX[mode]}"


def hgaoenkf_estimator(selection: str) -> str:
    """Estimator tag for one analog selection rule."""
    return _HGAOENKF_TAG[selection]


METHOD_BASE = method_label(ESTIMATOR_3DVAR)
TEMPORAL_METHODS = {method_label(ESTIMATOR_3DVAR, mode): mode for mode in TEMPORAL_MODES}

# Timescales the trajectory lane resolves. A low-pass window keeps everything slower than
# it; a band is the difference of two low-passes and isolates one timescale.
LOWPASS_WINDOWS = (25, 100, 250, 500, 1000, 2000)
BANDS = ((25, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000))
# Metrics the RRMSE selection never reads, so a grid scan computes them for the winner only.
_FULL_METRICS = frozenset({"ssim", "crps", "crpss", "rcrv_bias", "rcrv_dispersion",
                           "coverage90"})

# Taper columns for prior-free (naive) rows, which carry no regularizer.
_NAN_REG = {"localization_km": np.nan, "shrinkage_lambda": np.nan, "alpha": np.nan}
# The extension terms, named once so row columns match the estimator keywords.
TERM_KEYS = ("tendency_theta", "tendency_lag_yr", "redundancy_theta")
# MTA switches recorded on each row; not in TERM_KEYS because they are not tuned.
_STACK_KEYS = ("tendency_normalise", "preserve_obs_trace")
# Analog-ensemble columns for rows from an estimator that draws no analog ensemble.
_NAN_ANALOG = {"analog_k": np.nan, "hybrid_w": np.nan,
               **{key: np.nan for key in TERM_KEYS + _STACK_KEYS}}


def analog_cols(k: int, hybrid_w: float, *, tendency_theta: float = 0.0,
                tendency_lag_yr: float = 0.0, redundancy_theta: float = 0.0,
                tendency_normalise: bool = False,
                preserve_obs_trace: bool = False) -> dict:
    """The analog columns for a row, as :func:`_NAN_REG` does for the taper.

    The extension terms default to off. The two switches are stored as floats so every
    analog column shares one type.
    """
    return {"analog_k": float(k), "hybrid_w": float(hybrid_w),
            "tendency_theta": float(tendency_theta),
            "tendency_lag_yr": float(tendency_lag_yr),
            "redundancy_theta": float(redundancy_theta),
            "tendency_normalise": float(tendency_normalise),
            "preserve_obs_trace": float(preserve_obs_trace)}


# ---------------------------------------------------------------------------
# Observation geometry.
# ---------------------------------------------------------------------------
def _obs_geometry(o: dict, lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray) -> dict:
    """Gather indices, error variance, site coords and block centres for usable observations.

    ``keep`` is the filter mask, so a caller can apply it to other columns of ``o``.
    ``centre`` is each sample's dating-block midpoint, present only if the caller attached it.
    """
    gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
    keep = safe_flat[gather] & (o["sse"] > 0)
    geom = {
        "gather": gather[keep],
        "sse": o["sse"][keep].astype(np.float64),
        "lat": o["lat"][keep],
        "lon": o["lon"][keep],
        "keep": keep,
    }
    if "centre" in o:
        geom["centre"] = o["centre"][keep].astype(np.float64)
    return geom


def _draw_shapes(long: pd.DataFrame, rng: np.random.Generator,
                 lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray,
                 n: int, min_obs: int) -> list[tuple[int, dict]]:
    """``n`` distinct proxy ages and their networks, each holding at least ``min_obs`` rows.

    An age contributes only its network geometry. The source age is returned because a
    borrowed sample's time offset is relative to its own network's age.
    """
    picked = []
    for a in rng.permutation(long["age"].unique()):
        a = int(a)
        o = observations_at_age(long, a)
        if not len(o.get("age", [])):
            continue
        geom = _obs_geometry(o, lats, lons, safe_flat)
        if len(geom["gather"]) >= min_obs:
            picked.append((a, geom))
            if len(picked) == n:
                return picked
    raise ValueError(
        f"fewer than {n} proxy ages carry {min_obs} usable observations on this grid")


def _pad_obs(test_obs: list[dict], T: int) -> tuple[np.ndarray, ...]:
    """Ragged per-truth test-shape obs to padded ``(T, max_obs)`` arrays for npz.

    ``obs_n`` is the real count per truth, so a reader can drop the padding.
    """
    max_obs = max((len(o["val"]) for o in test_obs), default=0)
    obs_lat = np.full((T, max_obs), np.nan)
    obs_lon = np.full((T, max_obs), np.nan)
    obs_val = np.full((T, max_obs), np.nan)
    obs_chan = np.full((T, max_obs), -1, dtype=np.int64)
    obs_n = np.zeros(T, dtype=np.int64)
    for ti, o in enumerate(test_obs):
        m = len(o["val"])
        obs_n[ti] = m
        obs_lat[ti, :m] = o["lat"]
        obs_lon[ti, :m] = o["lon"]
        obs_val[ti, :m] = o["val"]
        obs_chan[ti, :m] = o["chan"]
    return obs_lat, obs_lon, obs_val, obs_chan, obs_n


# ---------------------------------------------------------------------------
# Naive baselines (no B, no model).
# ---------------------------------------------------------------------------
def _naive_geometry(lats: np.ndarray, lons: np.ndarray, geom: dict, n_chan: int) -> list:
    """Per-channel nearest index and IDW weights mapping obs values to a field.

    Depends only on the network, so it is built once per network. Channels with no
    observations get ``None``.
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
        d = great_circle_km_between(lat_cell, lon_cell, geom["lat"][sel], geom["lon"][sel])
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


def _naive_obs_predictions(assim: dict, target: dict, n_chan: int) -> tuple[dict, np.ndarray]:
    """Prior-free predictions at withheld sites, and each one's distance to the nearest
    assimilated site.

    No field is built: withholding only reads predictions at the withheld sites. A channel
    with no assimilated observation predicts zero.
    """
    n_t = len(target["lat"])
    out = {"nearest": np.zeros(n_t), "idw": np.zeros(n_t)}
    dist = np.full(n_t, np.nan)
    for c in range(n_chan):
        tsel, asel = target["chan"] == c, assim["chan"] == c
        if not tsel.any() or not asel.any():
            continue
        n_ts = int(tsel.sum())
        d = great_circle_km_between(target["lat"][tsel], target["lon"][tsel],
                                    assim["lat"][asel], assim["lon"][asel])
        y = assim["y"][asel]
        nearest = np.argmin(d, axis=1)
        dist[tsel] = d[np.arange(n_ts), nearest]
        out["nearest"][tsel] = y[nearest]
        w = 1.0 / np.clip(d, 1.0, None) ** 2
        out["idw"][tsel] = (w / w.sum(axis=1, keepdims=True)) @ y
    return out, dist


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


def _event_groups(events: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """``(label, truth mask)`` for all truths and then each D-O event present."""
    groups = [("all", np.ones(len(events), dtype=bool))]
    for ev in sorted(set(events[events > 0])):
        groups.append((str(int(ev)), events == ev))
    return groups


def _skill_rows(truth_anom: np.ndarray, recon_anom: np.ndarray, safe_valid: np.ndarray,
                events: np.ndarray, base: dict) -> list[dict]:
    """CE / correlation / RMSE rows, pooled and per channel, for all and per event."""
    rows = []
    groups = _event_groups(events)
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
                ("amplitude", da.amplitude_ratio(t, r)),
            ):
                rows.append({**base, "do_event": do_event, "channel": chan_name,
                             "metric": metric, "value": value})
    return rows


def _calibration_rows(truth: np.ndarray, mean: np.ndarray, var: np.ndarray,
                      ref_var: np.ndarray, groups: list, base: dict) -> list[dict]:
    """CRPS / CRPSS / RCRV / coverage rows over flat, aligned arrays.

    ``groups`` is ``(do_event, channel_name, mask)`` triples. ``var`` is the predictive
    variance: the posterior variance for a noise-free truth, plus the observation error
    when the truth is a measurement. The CRPSS reference is the prior ``N(0, ref_var)``.
    """
    crps_model = calibration.crps_gaussian(truth, mean, var)
    crps_ref = calibration.crps_gaussian(truth, np.zeros_like(truth), ref_var)
    rows = []
    for do_event, chan_name, sel in groups:
        if sel.sum() < 2:
            continue
        bias, dispersion = calibration.rcrv(truth[sel], mean[sel], var[sel])
        for metric, value in (
            ("crps", float(np.mean(crps_model[sel]))),
            ("crpss", calibration.crpss(crps_model[sel], crps_ref[sel])),
            ("rcrv_bias", bias),
            ("rcrv_dispersion", dispersion),
            ("coverage90", calibration.coverage(truth[sel], mean[sel], var[sel], 0.9)),
        ):
            rows.append({**base, "do_event": do_event, "channel": chan_name,
                         "metric": metric, "value": value})
    return rows


def _field_calibration_rows(truth_anom: np.ndarray, recon_anom: np.ndarray,
                            post_var: np.ndarray, prior_var: np.ndarray,
                            safe_valid: np.ndarray, events: np.ndarray,
                            base: dict) -> list[dict]:
    """Calibration rows in field space, for one ``b_scale`` of a PPE lane.

    The truth is a model state with no observation error, so the predictive variance is
    the posterior variance alone.
    """
    t = truth_anom[:, :, safe_valid]
    r = recon_anom[:, :, safe_valid]
    v = post_var[:, :, safe_valid]
    ref = np.broadcast_to(prior_var[:, safe_valid][None], t.shape)
    chan = np.broadcast_to(np.arange(len(VARS))[None, :, None], t.shape).ravel()

    groups = []
    for do_event, esel in _event_groups(events):
        emask = np.broadcast_to(esel[:, None, None], t.shape).ravel()
        groups.append((do_event, "pooled", emask))
        groups += [(do_event, name, emask & (chan == c)) for c, name in enumerate(VARS)]
    return _calibration_rows(t.ravel(), r.ravel(), v.ravel(), ref.ravel(), groups, base)


def _ssim_rows(truth_anom: np.ndarray, recon_anom: np.ndarray, safe_valid: np.ndarray,
               events: np.ndarray, base: dict) -> list[dict]:
    """Masked SSIM rows, per channel and pooled, for all ages and per event.

    ``data_range`` is fixed per channel over the whole truth stack, so per-truth SSIMs are
    comparable. ``pooled`` is the mean of the per-channel SSIMs.
    """
    rows = []
    groups = _event_groups(events)

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


def _timescale_metric_rows(truth_f: np.ndarray, recon_f: np.ndarray, safe_valid: np.ndarray,
                           trim: int, suffix: str, base: dict) -> list[dict]:
    """Median per-cell corr / CE / amplitude of one filtered truth-recon pair.

    A median over cells, since after a wide filter a pooled statistic has few independent
    points.
    """
    n = len(truth_f)
    if n - 2 * trim < 3:
        return []
    t = truth_f[trim:n - trim]
    r = recon_f[trim:n - trim]
    maps = {"corr": da.corr_map(t, r),
            "ce": da.ce_map(t, r, np.zeros_like(t[0])),
            "amp": da.amplitude_map(t, r)}
    rows = []
    for chan_name, c in [("pooled", None)] + [(name, i) for i, name in enumerate(VARS)]:
        for metric, m in maps.items():
            cells = m[:, safe_valid] if c is None else m[c, safe_valid]
            rows.append({**base, "do_event": "all", "channel": chan_name,
                         "metric": f"{metric}_{suffix}",
                         "value": float(np.nanmedian(cells))})
    return rows


def _timescale_rows(truth_anom: np.ndarray, recon_anom: np.ndarray, safe_valid: np.ndarray,
                    base: dict, *, step_yr: float, windows, bands) -> list[dict]:
    """Skill by timescale for a consecutive run of states.

    Emits ``{corr,ce,amp}_lp{window}`` for each low-pass series and ``..._bp{a}_{b}`` for
    each band.
    """
    lp_t = {w: da.lowpass_time(truth_anom, w, step_yr) for w in set(windows) | {b for ab in bands for b in ab}}
    lp_r = {w: da.lowpass_time(recon_anom, w, step_yr) for w in lp_t}

    rows = []
    for w in windows:
        rows += _timescale_metric_rows(lp_t[w], lp_r[w], safe_valid,
                                       da.timescale_trim(w, step_yr), f"lp{int(w)}", base)
    for lo, hi in bands:
        rows += _timescale_metric_rows(lp_t[lo] - lp_t[hi], lp_r[lo] - lp_r[hi], safe_valid,
                                       da.timescale_trim(hi, step_yr),
                                       f"bp{int(lo)}_{int(hi)}", base)
    return rows


def _append_csv(path: str, rows: list[dict]) -> None:
    """Append metric rows to the tidy CSV, reconciling a file written to a narrower schema.

    Rows with a column the file lacks would otherwise be misaligned, so the file is
    rewritten with the wider header. When the headers agree only the header is read.
    """
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != list(df.columns):
            # Ordered by the incoming rows so the rewrite happens once. Written to a
            # temporary file and moved, so an interrupted rewrite loses nothing.
            cols = list(df.columns) + [c for c in header if c not in df.columns]
            merged = pd.concat([pd.read_csv(path), df], ignore_index=True)[cols]
            tmp = f"{path}.tmp"
            merged.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ---------------------------------------------------------------------------
# Pseudo-proxy PPE lane: same-model chronological split.
# ---------------------------------------------------------------------------
def _report_progress(label: str, done: int, total: int, t0: float) -> None:
    """One-line progress update with a linear-rate ETA over the loop so far."""
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    print(f"  {label} {done}/{total} ({100 * done / total:.0f}%, "
          f"elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)


def _score_ppe_lane(
    truth_anoms: np.ndarray, prior: Prior, long: pd.DataFrame,
    lats: np.ndarray, lons: np.ndarray, *,
    lane: str, make_method: MethodFactory | None, space: str, reg_cols: dict,
    b_scales: tuple[float, ...],
    dist_edges_km: np.ndarray | None, seed: int, full_metrics: bool = True,
    npz_extra: dict | None = None, progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
    min_obs: int = MIN_OBS,
) -> tuple[list[dict], dict, dict]:
    """Score a built prior against a stack of truth anomalies (no file writes).

    Returns ``(rows, npz_arrays, skill)``, which :func:`_write_ppe_artifacts` persists.

    Each truth draws two real network shapes with independent noise: the first is the
    ``selection`` split, the second the ``test`` split, so the operating point is chosen on
    observations the reported analysis never saw. Each pseudo-observation is the truth at
    its nearest cell plus ``N(0, sse)`` noise, and R is ``diag(sse)``. ``make_method``
    defaults to :class:`ThreeDVar`; ``space``, ``reg_cols``, ``estimator`` and
    ``method_cols`` tag the rows. ``full_metrics=False`` skips SSIM, calibration and skill
    against distance, which the rRMSE selection does not need. Calibration is scored on the
    test split only.
    """
    rng = np.random.default_rng(seed)
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    b_scales = tuple(float(b) for b in b_scales)
    n_b = len(b_scales)
    npz_extra = npz_extra or {}
    method_cols = method_cols or _NAN_ANALOG

    clim_mean = prior.clim_mean.astype(np.float64)
    safe_valid = prior.safe_valid
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    tv = ThreeDVar(prior.B, shape) if make_method is None else make_method(prior, shape)

    zero_bg = np.zeros(int(np.prod(shape)))
    T = len(truth_anoms)

    splits = ("selection", "test")
    recon = {split: np.zeros((n_b, T, *shape)) for split in splits}
    post_test = np.zeros((n_b, T, *shape))
    naive_test = {"nearest": np.zeros((T, *shape)), "idw": np.zeros((T, *shape))}
    dist_test, test_obs = [], []
    drawn_ages = np.zeros((T, len(splits)), dtype=np.int64)

    t0 = time.time()
    for ti, truth_anom in enumerate(truth_anoms):
        shapes = _draw_shapes(long, rng, lats, lons, safe_flat, len(splits), min_obs)
        drawn_ages[ti] = [age for age, _ in shapes]
        for split, (_, geom) in zip(splits, shapes):
            g = geom["gather"]
            truth_at_obs = truth_anom.ravel()[g]         # H = nearest cell
            y = truth_at_obs + rng.normal(0.0, np.sqrt(geom["sse"]))
            res = tv.apply_sweep(tv.prepare_sweep(g, geom["sse"], b_scales), y, zero_bg)
            for bj in range(n_b):
                recon[split][bj, ti] = res[bj].mean_anom
            if split != "test":
                continue
            for bj in range(n_b):
                # Taken from the analyses: an analog estimator's spread depends on y.
                post_test[bj, ti] = res[bj].posterior_var
            naive_geom = _naive_geometry(lats, lons, geom, len(VARS))
            for kind in naive_test:
                naive_test[kind][ti] = _naive_apply(kind, naive_geom, y, shape)
            dist_test.append(da.nearest_obs_distance(lats, lons, geom["lat"], geom["lon"]))
            test_obs.append({"lat": geom["lat"], "lon": geom["lon"],
                             "val": truth_at_obs, "chan": g // n_cells})
        if progress_every and (ti + 1) % progress_every == 0:
            _report_progress("truth", ti + 1, T, t0)

    events = np.zeros(T, dtype=np.int64)                 # PPE truths carry no DO-event labels

    rows = []
    for bj, kb in enumerate(b_scales):
        base = {"method": method_label(estimator), "space": space, **reg_cols, **method_cols,
                "lane": lane, "fold": -1, "b_scale": kb,
                "background": "climatological", "split": "test"}
        base_sel = {**base, "split": "selection"}
        rows += _skill_rows(truth_anoms, recon["test"][bj], safe_valid, events, base)
        rows += _skill_rows(truth_anoms, recon["selection"][bj], safe_valid, events, base_sel)
        if full_metrics:
            rows += _ssim_rows(truth_anoms, recon["test"][bj], safe_valid, events, base)
            rows += _ssim_rows(truth_anoms, recon["selection"][bj], safe_valid, events,
                               base_sel)
            rows += _field_calibration_rows(truth_anoms, recon["test"][bj], post_test[bj],
                                            kb * tv.diagB.reshape(shape), safe_valid,
                                            events, base)
    for kind in naive_test:
        base = {"method": kind, "space": space, **_NAN_REG, **_NAN_ANALOG,
                "lane": lane, "fold": -1, "b_scale": 1.0,
                "background": "none", "split": "test"}
        rows += _skill_rows(truth_anoms, naive_test[kind], safe_valid, events, base)
        if full_metrics:
            rows += _ssim_rows(truth_anoms, naive_test[kind], safe_valid, events, base)

    skill = {}
    if full_metrics:
        if dist_edges_km is None:
            dist_edges_km = np.array([0, 500, 1000, 2000, 3000, 5000, 8000, 20000],
                                     dtype=float)
        dist = np.stack(dist_test)
        dist_pool = np.tile(dist[:, safe_valid.ravel()], (1, len(VARS))).ravel()
        t_pool = truth_anoms[:, :, safe_valid].reshape(T, -1).ravel()
        curves = [da.skill_vs_distance(
            t_pool, recon["test"][bj][:, :, safe_valid].reshape(T, -1).ravel(),
            np.zeros_like(t_pool), dist_pool, dist_edges_km) for bj in range(n_b)]
        skill = {"edges": dist_edges_km, "b_scales": np.asarray(b_scales),
                 # The count says which bins hold enough cells to read.
                 "count": curves[0]["count"],
                 **{key: np.array([c[key] for c in curves]) for key in ("ce", "rmse")}}

    obs_lat, obs_lon, obs_val, obs_chan, obs_n = _pad_obs(test_obs, T)
    npz_arrays = {
        "truth_anom": truth_anoms, "clim_mean": clim_mean,
        "safe_valid": safe_valid, "post_var": post_test, "prior_var": tv.diagB.reshape(shape),
        "lats": np.asarray(lats), "lons": np.asarray(lons),
        "drawn_ages": drawn_ages, "b_scales": np.asarray(b_scales),
        "recon_climatological": recon["test"],           # (n_b, T, 2, n_lat, n_lon)
        "naive_nearest": naive_test["nearest"], "naive_idw": naive_test["idw"],
        "obs_lat": obs_lat, "obs_lon": obs_lon, "obs_val": obs_val,
        "obs_chan": obs_chan, "obs_n": obs_n,
        **npz_extra,
    }
    return rows, npz_arrays, skill


def _write_ppe_artifacts(out_dir: str, lane: str, rows: list[dict],
                         npz_arrays: dict, skill: dict, config: dict,
                         b_scale: float | None = None) -> None:
    """Persist a scored PPE lane: metrics CSV (appended), analysis npz, skill npz, config.

    ``b_scale`` keeps the fields at that amplitude only, in float32.
    """
    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_analysis.npz"),
                        **_selected_fields(npz_arrays, b_scale))
    if skill:
        np.savez_compressed(os.path.join(out_dir, f"{lane}_skill_vs_distance.npz"), **skill)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as f:
        json.dump(config, f, indent=2)


# Analysis fields carrying a leading b_scale axis, which the selected amplitude indexes.
_SWEPT_FIELDS = ("recon_climatological", "post_var")


def _selected_fields(npz_arrays: dict, b_scale: float | None) -> dict:
    """``npz_arrays`` with the swept fields cut to one ``b_scale`` and cast to float32."""
    if b_scale is None:
        return npz_arrays
    scales = np.asarray(npz_arrays["b_scales"], dtype=np.float64)
    bj = int(np.argmin(np.abs(scales - float(b_scale))))
    out = dict(npz_arrays)
    out["selected_b_scale"] = np.asarray(scales[bj])
    for key in _SWEPT_FIELDS:
        if key in out:
            out[key] = np.asarray(out[key])[bj].astype(np.float32)
    for key in ("naive_nearest", "naive_idw", "truth_anom"):
        if key in out:
            out[key] = np.asarray(out[key]).astype(np.float32)
    return out


def run_ppe(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    make_method: MethodFactory | None = None, space: str = "pixel",
    b_scales: tuple[float, ...] = B_SCALES, truth_stride: int = 1,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
    progress_every: int | None = None, sel_tol: float = SEL_TOL,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """Same-model PPE for one taper config: truths are a held-out chronological chunk.

    The older half of the age axis builds B and the climatology; the younger half, every
    ``truth_stride`` states, supplies truths anomalised about their own mean. Scoring follows
    :func:`_score_ppe_lane`.
    """
    ages_i = np.asarray(ages, dtype=np.int64)
    prior_idx, truth_idx = chronological_half_split(ages_i, stride=truth_stride)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km, shrinkage_lambda=shrinkage_lambda,
                        alpha=alpha)
    truth_cube = cube[truth_idx].astype(np.float64)
    truth_clim = truth_cube.mean(axis=0)
    truth_anoms = truth_cube - truth_clim
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}
    rows, npz_arrays, skill = _score_ppe_lane(
        truth_anoms, prior, long, lats, lons,
        lane=LANE_PPE, make_method=make_method, space=space, reg_cols=reg_cols,
        b_scales=b_scales, dist_edges_km=dist_edges_km, seed=seed,
        npz_extra={"truth_clim": truth_clim}, min_obs=min_obs,
        progress_every=progress_every, estimator=estimator, method_cols=method_cols)
    win = select_best_config(_selection_rrmse(rows, LANE_PPE, method_label(estimator)),
                             sel_tol=sel_tol)
    config = _ppe_config(LANE_PPE, space, prior, len(truth_anoms), b_scales, seed, reg_cols,
                         _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride),
                         _PPE_NETWORKS,
                         extra={"selected": win, "sel_tol": sel_tol, "min_obs": min_obs})
    _write_ppe_artifacts(out_dir, LANE_PPE, rows, npz_arrays, skill, config,
                         b_scale=win["b_scale"])
    return pd.DataFrame(rows)


def _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride):
    """Split-provenance keys for a same-model PPE config.json."""
    return {"split": "chronological_midpoint", "prior_half": "older",
            "split_index": int(prior_idx[0]), "truth_stride": int(truth_stride),
            "chunk_a_ages": [int(ages_i[prior_idx].min()), int(ages_i[prior_idx].max())],
            "chunk_b_ages": [int(ages_i[truth_idx].min()), int(ages_i[truth_idx].max())]}


# Where each arm's observation network comes from, recorded in the config.
DRAWN_NETWORK = "a proxy age drawn at random"
OWN_NETWORK = "the age's own proxy network"
_PPE_NETWORKS = {"selection": DRAWN_NETWORK, "test": DRAWN_NETWORK}


def _ppe_config(lane, space, prior, n_truths, b_scales, seed, reg_cols, split_meta,
                networks, extra=None):
    """Assemble a same-model lane config.json dict (single-config or grid winner)."""
    cfg = {"lane": lane, "space": space, **reg_cols,
           "n_truths": int(n_truths), "networks": dict(networks), "n_noise": 1,
           "b_scales": [float(b) for b in b_scales], "seed": seed,
           "prior_meta": prior.meta, **split_meta}
    if extra:
        cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Trajectory lane: a consecutive run of states, scored by timescale.
# ---------------------------------------------------------------------------
def _age_step(ages: np.ndarray) -> float:
    """The single spacing of the age axis; the timescale filters assume it is uniform."""
    steps = np.unique(np.diff(np.asarray(ages, dtype=np.int64)))
    if len(steps) != 1:
        raise ValueError(f"trajectory scoring needs a uniform age step; found {steps}")
    return float(steps[0])


def _max_block_lag(long: pd.DataFrame, step_yr: float) -> int:
    """Widest gap in age steps between an age and its own sample's block centre.

    Sizes the structure function so no observation needs a clipped lag.
    """
    lag = (long["age"] - long["centre"]).abs().to_numpy(dtype=np.float64)
    return int(np.floor(lag.max() / step_yr + 0.5)) if len(lag) else 0


def _network_at_age(long: pd.DataFrame, age: int, lats: np.ndarray, lons: np.ndarray,
                    safe_flat: np.ndarray, min_obs: int) -> dict | None:
    """The proxy network an age actually carries, or ``None`` where it is too thin.

    The caller records those ages rather than borrowing a network for them.
    """
    o = observations_at_age(long, int(age))
    if not len(o.get("age", [])):
        return None
    geom = _obs_geometry(o, lats, lons, safe_flat)
    return geom if len(geom["gather"]) >= min_obs else None


def transplanted_source(centre: np.ndarray, shape_age: int, age: int,
                        ages_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Archive index each observation reads, and which ones land on the axis.

    A network borrowed from ``shape_age`` reports on ``age + (centre - shape_age)``: the
    time offset is relative to the network's own age. Offsets that run off the archive are
    dropped rather than clamped to its ends.
    """
    src_age = float(age) + (np.asarray(centre, dtype=np.float64) - float(shape_age))
    on_axis = (src_age >= ages_i[0]) & (src_age <= ages_i[-1])
    return nearest_age_index(src_age[on_axis], ages_i), on_axis


def run_trajectory(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    make_method: MethodFactory | None = None, space: str = "pixel",
    b_scales: tuple[float, ...] = B_SCALES,
    lowpass_windows: tuple[int, ...] = LOWPASS_WINDOWS,
    bands: tuple[tuple[int, int], ...] = BANDS,
    min_obs: int = MIN_OBS, sel_tol: float = SEL_TOL, seed: int = 0,
    progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
    temporal_modes: tuple[str, ...] = TEMPORAL_MODES,
) -> pd.DataFrame:
    """Reconstruct a consecutive run of states and score skill by timescale.

    The split is :func:`_score_ppe_lane`'s, but every younger-half state is a truth, so the
    analyses form a time series. Each observation reads the state at its own sample's
    offset in time, while scoring is against the state at the analysis age.

    The test arm assimilates each age's own network, which changes little between
    neighbouring ages. The selection arm borrows a network from another age, so the
    operating point is chosen on observations the reported analysis never saw. Ages with
    no usable network are recorded and skipped.

    ``temporal_modes`` chooses how staleness is treated (see
    :func:`paleoreco.assim.observations.apply_temporal_error`); the structure function
    comes from the prior ages only. The taper is inherited, and only ``b_scale`` is swept.
    The selection arm is scored on skill alone, so the timescale metrics are never tuned on.
    """
    ages_i = np.asarray(ages, dtype=np.int64)
    step_yr = _age_step(ages_i)
    prior_idx, truth_idx = chronological_half_split(ages_i, stride=1)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km, shrinkage_lambda=shrinkage_lambda,
                        alpha=alpha)
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}

    shape = (len(VARS), len(lats), len(lons))
    safe_valid = prior.safe_valid
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    tv = ThreeDVar(prior.B, shape) if make_method is None else make_method(prior, shape)
    b_scales = tuple(float(b) for b in b_scales)
    n_b = len(b_scales)
    zero_bg = np.zeros(int(np.prod(shape)))

    truth_cube = cube[truth_idx].astype(np.float64)
    truth_clim = truth_cube.mean(axis=0)
    truth_anoms = truth_cube - truth_clim
    # Observations are read from the whole run, since a block centre can fall in the older
    # half; B still sees prior ages only.
    all_anoms = cube.reshape(len(ages_i), -1).astype(np.float64) - truth_clim.ravel()
    long = sample_block_centres(long)
    S, prior_var_cell = temporal_structure_function(
        cube, prior_idx, max_lag=_max_block_lag(long, step_yr))

    method_cols = method_cols or _NAN_ANALOG
    labels = {method_label(estimator, mode): mode for mode in temporal_modes}
    base_label = next(iter(labels))
    splits = ("selection", "test")

    rng = np.random.default_rng(seed)
    truth_ages = ages_i[truth_idx]
    # float32 keeps the field stacks to a manageable size.
    recon = {split: {m: [] for m in labels} for split in splits}
    post = {m: [] for m in labels}
    naive = {"nearest": [], "idw": []}
    covered, skipped = [], []
    analog_index, obs_n, shape_ages = [], [], []
    t0 = time.time()
    for ti, age in enumerate(truth_ages):
        # Test arm: the age's own network. A freshly drawn network each age would add
        # site turnover to the fastest band.
        own = _network_at_age(long, int(age), lats, lons, safe_flat, min_obs)
        if own is None:
            skipped.append(int(age))
            continue
        sources = {"test": (int(age), own, *transplanted_source(own["centre"], int(age),
                                                               age, ages_i))}
        # Selection arm: a network borrowed from another age, redrawn if too many of its
        # offsets run off the archive.
        for _ in range(_MAX_SHAPE_DRAWS):
            (shape_age, geom), = _draw_shapes(long, rng, lats, lons, safe_flat, 1, min_obs)
            src, on_axis = transplanted_source(geom["centre"], shape_age, age, ages_i)
            if on_axis.sum() >= min_obs:
                sources["selection"] = (shape_age, geom, src, on_axis)
                break
        else:
            raise ValueError(
                f"no borrowed network kept {min_obs} observations on the archive at age "
                f"{int(age)} in {_MAX_SHAPE_DRAWS} draws")

        drawn = {}
        for split in splits:
            shape_age, geom, src, on_axis = sources[split]
            g = geom["gather"][on_axis]
            sse = geom["sse"][on_axis]
            stale = all_anoms[src, g] + rng.normal(0.0, np.sqrt(sse))
            # The lag is taken from the age actually read, after rounding.
            rho, resid = temporal_terms(S, prior_var_cell, g,
                                        np.abs(ages_i[src] - age), step_yr)
            drawn[split] = (shape_age, geom, on_axis, g, sse, stale, rho, resid)
        covered.append(ti)

        for split, (shape_age, geom, on_axis, g, sse, stale, rho, resid) in drawn.items():
            for method, mode in labels.items():
                yv, r = apply_temporal_error(stale, sse, rho, resid, mode)
                gain = tv.prepare_sweep(g, r, b_scales)
                res = tv.apply_sweep(gain, yv, zero_bg)
                recon[split][method].append(
                    np.stack([x.mean_anom for x in res]).astype(np.float32))
                if split != "test":
                    continue
                post[method].append(
                    np.stack([x.posterior_var for x in res]).astype(np.float32))
                if method != base_label:
                    continue
                # The selected analog members, where the estimator selects them.
                if hasattr(tv, "select"):
                    analog_index.append(tv.select(gain, yv))
                naive_geom = _naive_geometry(
                    lats, lons, {"gather": g, "lat": geom["lat"][on_axis],
                                 "lon": geom["lon"][on_axis]}, len(VARS))
                for kind in naive:
                    naive[kind].append(_naive_apply(kind, naive_geom, yv, shape))
                obs_n.append(len(g))
                shape_ages.append(int(shape_age))
        if progress_every and (ti + 1) % progress_every == 0:
            _report_progress("trajectory age", ti + 1, len(truth_idx), t0)

    if len(covered) < 4:
        raise ValueError(f"only {len(covered)} ages carried a usable network; "
                         "nothing to score")
    covered = np.asarray(covered)
    # The timescale filters assume contiguous ages; skipped ages may only sit at the end.
    if np.any(np.diff(covered) != 1):
        raise ValueError("the scored ages are not consecutive; the timescale filters "
                         "assume a uniform step")

    covered_ages = truth_ages[covered]
    # The anomaly frame is the whole younger half, not just the scored ages.
    truth_run = truth_anoms[covered]
    recon = {split: {m: np.stack(v, axis=1) for m, v in methods.items()}
             for split, methods in recon.items()}          # (n_b, n_covered, C, H, W)
    post = {m: np.stack(v, axis=1) for m, v in post.items()}
    naive = {k: np.stack(v) for k, v in naive.items()}
    events = np.zeros(len(covered), dtype=np.int64)

    rows: list[dict] = []
    for bj, kb in enumerate(b_scales):
        for method in labels:
            for split in splits:
                base = {"method": method, "space": space, **reg_cols, **method_cols,
                        "lane": LANE_TRAJECTORY, "fold": -1, "b_scale": kb,
                        "background": "climatological", "split": split}
                field = recon[split][method][bj]
                rows += _skill_rows(truth_run, field, safe_valid, events, base)
                # Pooled RRMSE is all the selection shape is read for.
                if split != "test":
                    continue
                rows += _ssim_rows(truth_run, field, safe_valid, events, base)
                rows += _timescale_rows(truth_run, field, safe_valid, base,
                                        step_yr=step_yr, windows=lowpass_windows, bands=bands)
                rows += _field_calibration_rows(
                    truth_run, field, post[method][bj],
                    kb * tv.diagB.reshape(shape), safe_valid, events, base)
    for kind, field in naive.items():
        base = {"method": kind, "space": space, **_NAN_REG, **_NAN_ANALOG,
                "lane": LANE_TRAJECTORY, "fold": -1, "b_scale": 1.0,
                "background": "none", "split": "test"}
        rows += _skill_rows(truth_run, field, safe_valid, events, base)
        rows += _ssim_rows(truth_run, field, safe_valid, events, base)
        rows += _timescale_rows(truth_run, field, safe_valid, base,
                                step_yr=step_yr, windows=lowpass_windows, bands=bands)

    win = select_best_config(_selection_rrmse(rows, LANE_TRAJECTORY, base_label),
                             sel_tol=sel_tol)
    bw = int(np.argmin(np.abs(np.asarray(b_scales) - win["b_scale"])))
    # Fields for the selected b_scale only.
    npz_arrays = {
        "truth_anom": truth_run.astype(np.float32), "truth_clim": truth_clim,
        "clim_mean": prior.clim_mean.astype(np.float64), "safe_valid": safe_valid,
        "lats": lats, "lons": lons, "ages": covered_ages,
        "b_scales": np.asarray(b_scales), "selected_b_scale": np.asarray(b_scales[bw]),
        "step_yr": np.asarray(step_yr),
        "recon_realistic": recon["test"][base_label][bw],
        "naive_nearest": naive["nearest"].astype(np.float32),
        "naive_idw": naive["idw"].astype(np.float32),
        "post_var": post[base_label][bw],
        "prior_var": tv.diagB.reshape(shape),
        "obs_n": np.asarray(obs_n),
        # The age each test network came from, needed to rebuild it later.
        "shape_ages": np.asarray(shape_ages, dtype=np.int64),
        "skipped_ages": np.asarray(skipped, dtype=np.int64),
    }
    if analog_index:
        npz_arrays["analog_index"] = np.stack(analog_index)
    # The temporal variants, stored at the baseline's b_scale so they can be differenced.
    for method in labels:
        if method == base_label:
            continue
        npz_arrays[f"recon_{method}"] = recon["test"][method][bw]
        npz_arrays[f"post_var_{method}"] = post[method][bw]
    config = _ppe_config(
        LANE_TRAJECTORY, space, prior, len(covered), b_scales, seed, reg_cols,
        split_meta=_chronological_split_meta(ages_i, prior_idx, truth_idx, 1),
        networks={"selection": DRAWN_NETWORK, "test": OWN_NETWORK},
        extra={"selected": win, "sel_tol": sel_tol, "estimator": estimator,
               "selected_b_scale_by_method": _b_scale_by_method(rows, LANE_TRAJECTORY,
                                                                sel_tol, tuple(labels)),
               "max_block_lag_steps": _max_block_lag(long, step_yr),
               "lowpass_windows": [int(w) for w in lowpass_windows],
               "bands": [[int(a), int(b)] for a, b in bands],
               "step_yr": step_yr, "min_obs": min_obs,
               "n_covered_ages": len(covered), "n_skipped_ages": len(skipped),
               "skipped_ages": [int(a) for a in skipped],
               "scored_ages": [int(covered_ages[0]), int(covered_ages[-1])]})
    _write_trajectory_artifacts(out_dir, LANE_TRAJECTORY, rows, npz_arrays, config)
    return pd.DataFrame(rows)


def _write_trajectory_artifacts(out_dir: str, lane: str, rows: list[dict],
                                npz_arrays: dict, config: dict) -> None:
    """Persist a scored trajectory lane: metrics CSV (appended), analysis npz, config.

    No skill-vs-distance npz, since the network changes with every age.
    """
    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_analysis.npz"), **npz_arrays)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Real-proxy withholding.
# ---------------------------------------------------------------------------
def _site_folds(long: pd.DataFrame, k: int, kind: str, seed: int) -> list[np.ndarray]:
    """Partition sites into ``k`` folds."""
    sites = long.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    ids = sites["site"].to_numpy()
    if kind == "random":
        rng = np.random.default_rng(seed)
        return [f for f in np.array_split(rng.permutation(ids), k)]
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
            ("amplitude", da.amplitude_ratio(a, p)),
        ):
            rows.append({**base, "do_event": "all", "channel": chan_name,
                         "metric": metric, "value": value})
    return rows


def _obs_channel_groups(channel: np.ndarray) -> list:
    """``(do_event, channel_name, mask)`` triples for observation-space calibration.

    Real proxies carry no D-O event label, so the event is always ``all``.
    """
    groups = [("all", "pooled", np.ones(len(channel), dtype=bool))]
    return groups + [("all", name, channel == c) for c, name in enumerate(VARS)]


@dataclass(frozen=True)
class _TargetPredictions:
    """Withheld-site predictions for one assim/target split, pooled over ages.

    ``pred`` and ``post_var`` map a method label to an ``(n_b, N)`` array; the other fields
    are ``(N,)`` and describe the withheld observation. ``prior_var`` is the CRPSS
    reference, ``distance_km`` the distance to the nearest assimilated site, ``rep_var`` and
    ``resid_var`` the spatial and temporal error terms at the target, and ``site`` the unit
    a resampling test must redraw whole.
    """

    actual: np.ndarray
    channel: np.ndarray
    sse: np.ndarray
    pred: dict
    post_var: dict
    prior_var: np.ndarray
    naive: dict
    distance_km: np.ndarray
    rep_var: np.ndarray
    resid_var: np.ndarray
    site: np.ndarray

    def __len__(self) -> int:
        return len(self.actual)


def _predict_targets(
    tv: Method, long: pd.DataFrame, obs_ages: np.ndarray,
    lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray, clim_flat: np.ndarray,
    assim_sites: set, target_sites: set, b_scales: tuple[float, ...],
    n_b: int, n_cells: int, rep_lookup: np.ndarray,
    S: np.ndarray, prior_var_cell: np.ndarray, step_yr: float,
    labels: dict[str, str],
) -> _TargetPredictions:
    """Assimilate the assim-set sites age by age, predict the target-set sites.

    Observations enter as anomalies ``y - my`` against a zero background, with
    ``R = diag(sse + rep_var)`` before the temporal term. A withheld observation is scored
    only at ages that also carry an assimilated one.

    Each temporal method runs its own analysis. Every prediction, including the uncorrected
    method's, is multiplied by the withheld sample's own ``rho``, so all methods are scored
    on the same attenuated scale; the prior-free baselines already sit on it.
    """
    bg_zero = np.zeros(len(clim_flat))
    actual, channel, sse, prior_var, dist, rep, resid, site = [], [], [], [], [], [], [], []
    pred = {m: [] for m in labels}
    post_var = {m: [] for m in labels}
    naive = {"nearest": [], "idw": []}
    for age in obs_ages:
        o = observations_at_age(long, int(age))
        gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
        keep = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
        kept = keep & np.array([s in assim_sites for s in o["site"]])
        wkeep = keep & np.array([s in target_sites for s in o["site"]])
        if kept.sum() == 0 or wkeep.sum() == 0:
            continue
        y_anom = (o["y"][kept] - o["my"][kept]).astype(np.float64)
        gk, gw = gather[kept], gather[wkeep]
        r_kept = o["sse"][kept].astype(np.float64) + rep_lookup[gk // n_cells]
        lag = np.abs(o["age"] - o["centre"])
        rho_k, resid_k = temporal_terms(S, prior_var_cell, gk, lag[kept], step_yr)
        rho_w, resid_w = temporal_terms(S, prior_var_cell, gw, lag[wkeep], step_yr)

        for method, mode in labels.items():
            yv, r = apply_temporal_error(y_anom, r_kept, rho_k, resid_k, mode)
            # The age drives the analog exclusion band, since this prior spans it.
            res = tv.apply_sweep(tv.prepare_sweep(gk, r, b_scales, age=int(age)), yv, bg_zero)
            pred[method].append(
                np.stack([rho_w * res[bj].predict_obs(gw) for bj in range(n_b)]))
            post_var[method].append(
                np.stack([rho_w ** 2 * res[bj].predict_obs_var(gw) for bj in range(n_b)]))

        actual.append((o["y"][wkeep] - o["my"][wkeep]).astype(np.float64))
        channel.append(gw // n_cells)
        sse.append(o["sse"][wkeep].astype(np.float64))
        rep.append(rep_lookup[gw // n_cells])
        resid.append(resid_w)
        site.append(o["site"][wkeep])
        prior_var.append(rho_w ** 2 * tv.diagB[gw])
        nv, d = _naive_obs_predictions(
            {"lat": o["lat"][kept], "lon": o["lon"][kept], "y": y_anom,
             "chan": gk // n_cells},
            {"lat": o["lat"][wkeep], "lon": o["lon"][wkeep], "chan": gw // n_cells},
            len(VARS))
        for kind in naive:
            naive[kind].append(nv[kind])
        dist.append(d)

    if not actual:
        def empty():
            return np.array([])
        return _TargetPredictions(
            empty(), empty(), empty(), {m: np.zeros((n_b, 0)) for m in pred},
            {m: np.zeros((n_b, 0)) for m in post_var}, empty(),
            {k: empty() for k in naive}, empty(), empty(), empty(), empty())
    return _TargetPredictions(
        actual=np.concatenate(actual), channel=np.concatenate(channel),
        sse=np.concatenate(sse),
        pred={m: np.concatenate(v, axis=1) for m, v in pred.items()},
        post_var={m: np.concatenate(v, axis=1) for m, v in post_var.items()},
        prior_var=np.concatenate(prior_var),
        naive={k: np.concatenate(v) for k, v in naive.items()},
        distance_km=np.concatenate(dist), rep_var=np.concatenate(rep),
        resid_var=np.concatenate(resid), site=np.concatenate(site))


def _concat_targets(parts: list[_TargetPredictions]) -> _TargetPredictions:
    """Pool several folds' predictions, keeping the ``(n_b, N)`` sweep axis leading."""
    return _TargetPredictions(
        actual=np.concatenate([p.actual for p in parts]),
        channel=np.concatenate([p.channel for p in parts]),
        sse=np.concatenate([p.sse for p in parts]),
        pred={m: np.concatenate([p.pred[m] for p in parts], axis=1) for m in parts[0].pred},
        post_var={m: np.concatenate([p.post_var[m] for p in parts], axis=1)
                  for m in parts[0].post_var},
        prior_var=np.concatenate([p.prior_var for p in parts]),
        naive={k: np.concatenate([p.naive[k] for p in parts]) for k in parts[0].naive},
        distance_km=np.concatenate([p.distance_km for p in parts]),
        rep_var=np.concatenate([p.rep_var for p in parts]),
        resid_var=np.concatenate([p.resid_var for p in parts]),
        site=np.concatenate([p.site for p in parts]))


def _score_withholding_lane(
    prior: Prior, long: pd.DataFrame, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    *, cube: np.ndarray, prior_age_indices: np.ndarray,
    make_method: MethodFactory | None, space: str, reg_cols: dict,
    k_folds: int, fold_kind: str, b_scales: tuple[float, ...], seed: int,
    temporal_modes: tuple[str, ...] = TEMPORAL_MODES,
    use_rep_var: bool = True,
    progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
) -> tuple[str, list[dict], dict]:
    """Nested-CV site withholding for one built prior (no file writes).

    Returns ``(lane, rows, predictions)``. Two passes over one partition of the sites into
    ``k_folds`` folds:

    * selection: for each fold ``i``, assimilate every fold except ``i`` and ``i + 1`` and
      predict fold ``i + 1``, pooling over the rotation;
    * test: for each fold, assimilate the other ``k_folds - 1`` and predict it.

    The representativeness variance is estimated from the assimilated sites alone, so a
    withheld site never informs the analysis that scores it. ``temporal_modes`` chooses the
    staleness treatments scored; ``use_rep_var=False`` drops the spatial term from R and
    from the predictive spread. Calibration adds the proxy's own error terms to the
    posterior spread. ``predictions`` keeps what is needed to rescore without a re-run.
    """
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    b_scales = tuple(float(b) for b in b_scales)
    n_b = len(b_scales)
    clim_flat = prior.clim_mean.astype(np.float64).ravel()
    safe_flat = np.broadcast_to(prior.safe_valid, shape).ravel()
    tv = ThreeDVar(prior.B, shape) if make_method is None else make_method(prior, shape)

    obs_ages = np.intersect1d(long["age"].unique(), ages)
    fold_sets = [set(f.tolist()) for f in _site_folds(long, k_folds, fold_kind, seed)]
    all_sites = set(long["site"].unique().tolist())
    lane = f"withholding_{fold_kind}"
    method_cols = method_cols or _NAN_ANALOG
    labels = {method_label(estimator, mode): mode for mode in temporal_modes}
    base_label = next(iter(labels))

    # Block centres, and the structure function over the same ages B was built from.
    long = sample_block_centres(long)
    step_yr = _age_step(np.asarray(ages, dtype=np.int64))
    S, prior_var_cell = temporal_structure_function(
        cube, prior_age_indices, max_lag=_max_block_lag(long, step_yr))

    # rep_var is re-estimated per fold from the assimilated sites only.
    cell_all = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons)

    def _rep_lookup(sites) -> np.ndarray:
        if not use_rep_var:
            return np.zeros(len(VARS))
        rv = representativeness_variance(long, cell_all, sites=sites)
        return np.array([rv.get(v, 0.0) for v in VARS])

    def _rows(tp: _TargetPredictions, fold: int, split: str) -> list[dict]:
        """Skill and calibration rows over the b_scale sweep for every temporal method."""
        out = []
        groups = _obs_channel_groups(tp.channel)
        # The truth is a measurement, so its error terms join both the posterior spread
        # and the prior reference; only the background part scales with b_scale.
        obs_var = tp.sse + tp.rep_var + tp.resid_var
        for method in tp.pred:
            for bj, kb in enumerate(b_scales):
                base = {"method": method, "space": space, **reg_cols, **method_cols,
                        "lane": lane, "fold": fold, "b_scale": kb,
                        "background": "climatological", "split": split}
                out += _withholding_rows(tp.actual, tp.pred[method][bj], tp.channel, base)
                out += _calibration_rows(tp.actual, tp.pred[method][bj],
                                         tp.post_var[method][bj] + obs_var,
                                         kb * tp.prior_var + obs_var, groups, base)
        return out

    def _naive_rows(tp: _TargetPredictions) -> list[dict]:
        """Prior-free reference rows, emitted once: they carry no b_scale and no spread."""
        out = []
        for kind, pred in tp.naive.items():
            out += _withholding_rows(tp.actual, pred, tp.channel, {
                "method": kind, "space": space, **_NAN_REG, **_NAN_ANALOG, "lane": lane,
                "fold": -1, "b_scale": 1.0, "background": "none", "split": "test"})
        return out

    rows = []

    # Selection: hold out folds i and i + 1, predict i + 1; pool across the rotation.
    sel = []
    t0 = time.time()
    for i in range(k_folds):
        assim = all_sites - fold_sets[(i + 1) % k_folds] - fold_sets[i]
        tp = _predict_targets(tv, long, obs_ages, lats, lons, safe_flat,
                              clim_flat, assim, fold_sets[(i + 1) % k_folds],
                              b_scales, n_b, n_cells, _rep_lookup(assim),
                              S, prior_var_cell, step_yr, labels)
        if len(tp):
            sel.append(tp)
        if progress_every and (i + 1) % progress_every == 0:
            _report_progress("sel-fold", i + 1, k_folds, t0)
    if sel:
        rows += _rows(_concat_targets(sel), -1, "selection")

    # Test: hold out fold i and predict it; per fold and pooled.
    pooled = []
    t0 = time.time()
    for i in range(k_folds):
        assim = all_sites - fold_sets[i]
        tp = _predict_targets(tv, long, obs_ages, lats, lons, safe_flat,
                              clim_flat, assim, fold_sets[i],
                              b_scales, n_b, n_cells, _rep_lookup(assim),
                              S, prior_var_cell, step_yr, labels)
        if not len(tp):
            continue
        rows += _rows(tp, i, "test")
        pooled.append(tp)
        if progress_every and (i + 1) % progress_every == 0:
            _report_progress("test-fold", i + 1, k_folds, t0)

    predictions = {"b_scales": np.asarray(b_scales), "rep_var_full": _rep_lookup(all_sites)}
    if pooled:
        tp = _concat_targets(pooled)
        rows += _rows(tp, -1, "test")
        rows += _naive_rows(tp)
        predictions.update({
            "actual": tp.actual, "channel": tp.channel, "site": tp.site,
            "climatological_pred": tp.pred[base_label],
            "post_var_pred": tp.post_var[base_label], "prior_var_pred": tp.prior_var,
            "sse": tp.sse, "distance_km": tp.distance_km, "rep_var": tp.rep_var,
            "resid_var": tp.resid_var,
            "naive_nearest": tp.naive["nearest"], "naive_idw": tp.naive["idw"]})
        # The temporal variants under their own keys; the uncorrected one keeps the bare
        # names.
        for method in tp.pred:
            if method == base_label:
                continue
            predictions[f"climatological_pred_{method}"] = tp.pred[method]
            predictions[f"post_var_pred_{method}"] = tp.post_var[method]
    return lane, rows, predictions


def _write_withholding_artifacts(out_dir: str, lane: str, rows: list[dict],
                                 predictions: dict, config: dict) -> None:
    """Persist a scored withholding lane: metrics CSV (appended), predictions npz, config."""
    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_predictions.npz"), **predictions)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)


def _withholding_config(lane, space, prior, k_folds, fold_kind, b_scales, seed, reg_cols,
                        extra=None):
    """Assemble a withholding lane config.json dict (single-config or grid winner)."""
    cfg = {"lane": lane, "space": space, **reg_cols, "k_folds": k_folds,
           "fold_kind": fold_kind, "b_scales": [float(b) for b in b_scales],
           "background": "climatological", "seed": seed, "prior_meta": prior.meta}
    if extra:
        cfg.update(extra)
    return cfg


def _rep_var_full(predictions: dict) -> dict:
    """Full-network rep_var per channel, JSON-ready for the config record."""
    return {VARS[i]: float(v) for i, v in enumerate(predictions["rep_var_full"])}


def run_withholding(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    make_method: MethodFactory | None = None, space: str = "pixel",
    k_folds: int = 5, fold_kind: str = "random",
    b_scales: tuple[float, ...] = B_SCALES, seed: int = 0,
    progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
    temporal_modes: tuple[str, ...] = TEMPORAL_MODES, use_rep_var: bool = True,
) -> pd.DataFrame:
    """Nested-CV site withholding for one taper config.

    ``long`` must carry the per-site climatology ``my``. The prior uses all ages, since
    what is held out is real proxies, not model states.
    """
    prior_idx = np.arange(len(ages))
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km, shrinkage_lambda=shrinkage_lambda,
                        alpha=alpha)
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}
    lane, rows, predictions = _score_withholding_lane(
        prior, long, ages, lats, lons, cube=cube, prior_age_indices=prior_idx,
        make_method=make_method, space=space,
        reg_cols=reg_cols, k_folds=k_folds, fold_kind=fold_kind, b_scales=b_scales,
        seed=seed, progress_every=progress_every, estimator=estimator,
        method_cols=method_cols, temporal_modes=temporal_modes, use_rep_var=use_rep_var)
    config = _withholding_config(lane, space, prior, k_folds, fold_kind, b_scales, seed,
                                 reg_cols, extra={"use_rep_var": use_rep_var,
                                                  "rep_var_full": _rep_var_full(predictions)})
    _write_withholding_artifacts(out_dir, lane, rows, predictions, config)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pixel regularizer tuning: joint (localization, shrinkage, alpha, b_scale) selection.
# ---------------------------------------------------------------------------
def select_best_config(sel_rows: pd.DataFrame, *, sel_tol: float = SEL_TOL) -> dict:
    """Winner ``{localization_km, shrinkage_lambda, alpha, b_scale}`` on the selection split.

    ``sel_rows`` holds the pooled selection-split rRMSE rows of one method and lane. Among
    rows within ``sel_tol`` (relative) of the minimum, picks the fewest active tapers, then
    the ``b_scale`` closest to 1, then the lowest rRMSE. With ``sel_tol = 0`` it is the
    argmin.
    """
    df = sel_rows.dropna(subset=["value"])
    if df.empty:
        raise ValueError("no finite selection-split RRMSE to select from")
    best = float(df["value"].min())
    band = df[df["value"] <= best * (1.0 + sel_tol)]

    def key(r):
        loc_active = 0 if pd.isna(r["localization_km"]) else 1
        n_active = loc_active + int(float(r["shrinkage_lambda"]) > 0) + int(float(r["alpha"]) < 1.0)
        return (n_active, abs(np.log10(float(r["b_scale"]))), float(r["value"]))

    winner = min((r for _, r in band.iterrows()), key=key)
    return {
        "localization_km": None if pd.isna(winner["localization_km"]) else float(winner["localization_km"]),
        "shrinkage_lambda": float(winner["shrinkage_lambda"]),
        "alpha": float(winner["alpha"]),
        "b_scale": float(winner["b_scale"]),
    }


def select_analog_config(sel_rows: pd.DataFrame) -> dict:
    """Winner over the analog axes and ``b_scale`` on the selection split.

    A plain argmin: unlike the taper axes, none of these has a simpler end to prefer.
    """
    df = sel_rows.dropna(subset=["value"])
    if df.empty:
        raise ValueError("no finite selection-split RRMSE to select from")
    win = df.loc[df["value"].idxmin()]
    extras = {col: 0.0 if pd.isna(win.get(col, np.nan)) else float(win[col])
              for col in TERM_KEYS}
    return {"analog_k": int(win["analog_k"]), "hybrid_w": float(win["hybrid_w"]),
            **extras, "b_scale": float(win["b_scale"])}


def _pixel_grid_configs(localization_grid, shrinkage_grid, alpha_grid):
    """The (localization_km, shrinkage_lambda, alpha) grid points, and a JSON-safe record."""
    configs = list(itertools.product(localization_grid, shrinkage_grid, alpha_grid))
    record = {"localization_grid": [None if g is None else float(g) for g in localization_grid],
              "shrinkage_grid": [float(g) for g in shrinkage_grid],
              "alpha_grid": [float(g) for g in alpha_grid]}
    return configs, record


def _selection_rrmse(rows: list[dict], lane: str, method: str = METHOD_BASE) -> pd.DataFrame:
    """Pooled selection-split RRMSE rows for one method, the surface tuned over."""
    M = pd.DataFrame(rows)
    return M[(M.method == method) & (M.lane == lane) & (M.split == "selection")
             & (M.channel == "pooled") & (M.do_event == "all") & (M.metric == "rrmse")]


def _b_scale_by_method(rows: list[dict], lane: str, sel_tol: float,
                       methods: tuple[str, ...] = tuple(TEMPORAL_METHODS)) -> dict[str, float]:
    """Each temporal variant's own selection-split ``b_scale``.

    Inflating R shifts the balance between background and observations, so each variant
    is reported at its own selected amplitude.
    """
    out = {}
    for method in methods:
        sel = _selection_rrmse(rows, lane, method)
        if len(sel):
            out[method] = float(select_best_config(sel, sel_tol=sel_tol)["b_scale"])
    return out


def run_ppe_pixel_grid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_grid=LOCALIZATION_KM_GRID, shrinkage_grid=SHRINKAGE_GRID,
    alpha_grid=ALPHA_GRID, b_scales: tuple[float, ...] = B_SCALES, sel_tol: float = SEL_TOL,
    truth_stride: int = 1, dist_edges_km: np.ndarray | None = None, seed: int = 0,
    progress_every: int | None = None, min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """Same-model PPE tuned over the taper grid: full-grid metrics, winner-only fields.

    Scores every taper configuration on skill alone, selects jointly with ``b_scale`` via
    :func:`select_best_config`, then re-runs the winner with every metric and saves its
    fields.
    """
    ages_i = np.asarray(ages, dtype=np.int64)
    prior_idx, truth_idx = chronological_half_split(ages_i, stride=truth_stride)
    truth_cube = cube[truth_idx].astype(np.float64)
    truth_clim = truth_cube.mean(axis=0)
    truth_anoms = truth_cube - truth_clim
    configs, grid_record = _pixel_grid_configs(localization_grid, shrinkage_grid, alpha_grid)

    all_rows: list[dict] = []
    t0 = time.time()
    for ci, (loc, lam, a) in enumerate(configs):
        prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                            localization_km=loc, shrinkage_lambda=lam, alpha=a)
        reg_cols = {"localization_km": loc, "shrinkage_lambda": lam, "alpha": a}
        rows, _, _ = _score_ppe_lane(
            truth_anoms, prior, long, lats, lons,
            lane=LANE_PPE, make_method=None, space="pixel", reg_cols=reg_cols,
            b_scales=b_scales, dist_edges_km=dist_edges_km, seed=seed, full_metrics=False,
            npz_extra={"truth_clim": truth_clim}, min_obs=min_obs)
        all_rows += [r for r in rows if r["method"] == "3dvar"]   # naive added once, from winner
        _report_progress("pixel-grid config", ci + 1, len(configs), t0)

    win = select_best_config(_selection_rrmse(all_rows, LANE_PPE), sel_tol=sel_tol)
    reg_w = {"localization_km": win["localization_km"], "shrinkage_lambda": win["shrinkage_lambda"],
             "alpha": win["alpha"]}
    prior_w = build_prior(cube, ages, lats, lons, prior_idx, valid, **reg_w)
    win_rows, npz_arrays, skill = _score_ppe_lane(
        truth_anoms, prior_w, long, lats, lons,
        lane=LANE_PPE, make_method=None, space="pixel", reg_cols=reg_w,
        b_scales=b_scales, dist_edges_km=dist_edges_km, seed=seed, full_metrics=True,
        npz_extra={"truth_clim": truth_clim}, progress_every=progress_every, min_obs=min_obs)
    all_rows += [r for r in win_rows
                 if r["method"] == "3dvar" and r["metric"] in _FULL_METRICS]
    all_rows += [r for r in win_rows if r["method"] != "3dvar"]

    extra = {"selected": win, "sel_tol": sel_tol, "min_obs": min_obs, **grid_record}
    config = _ppe_config(LANE_PPE, "pixel", prior_w, len(truth_anoms), b_scales, seed, reg_w,
                         _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride),
                         _PPE_NETWORKS, extra=extra)
    _write_ppe_artifacts(out_dir, LANE_PPE, all_rows, npz_arrays, skill, config,
                         b_scale=win["b_scale"])
    return pd.DataFrame(all_rows)


def run_withholding_pixel_grid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_grid=LOCALIZATION_KM_GRID, shrinkage_grid=SHRINKAGE_GRID,
    alpha_grid=ALPHA_GRID, k_folds: int = 5, fold_kind: str = "random",
    b_scales: tuple[float, ...] = B_SCALES, sel_tol: float = SEL_TOL, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Withholding lane tuned over the taper grid: full-grid metrics, winner-only predictions.

    Scores every taper configuration, selects jointly with ``b_scale``, then re-runs the
    winner to save its predictions.
    """
    configs, grid_record = _pixel_grid_configs(localization_grid, shrinkage_grid, alpha_grid)
    lane = f"withholding_{fold_kind}"

    all_rows: list[dict] = []
    t0 = time.time()
    prior_idx = np.arange(len(ages))
    for ci, (loc, lam, a) in enumerate(configs):
        prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                            localization_km=loc, shrinkage_lambda=lam, alpha=a)
        reg_cols = {"localization_km": loc, "shrinkage_lambda": lam, "alpha": a}
        _, rows, _ = _score_withholding_lane(
            prior, long, ages, lats, lons, cube=cube, prior_age_indices=prior_idx,
            make_method=None, space="pixel",
            reg_cols=reg_cols, k_folds=k_folds, fold_kind=fold_kind, b_scales=b_scales,
            seed=seed, temporal_modes=(TEMPORAL_OFF,))
        # The grid runs the uncorrected method only; the winner pass adds the temporal
        # variants and the prior-free rows.
        all_rows += [r for r in rows if r["method"] == METHOD_BASE]
        _report_progress(f"pixel-grid config ({lane})", ci + 1, len(configs), t0)

    win = select_best_config(_selection_rrmse(all_rows, lane), sel_tol=sel_tol)
    reg_w = {"localization_km": win["localization_km"], "shrinkage_lambda": win["shrinkage_lambda"],
             "alpha": win["alpha"]}
    prior_w = build_prior(cube, ages, lats, lons, prior_idx, valid, **reg_w)
    _, win_rows, predictions = _score_withholding_lane(
        prior_w, long, ages, lats, lons, cube=cube, prior_age_indices=prior_idx,
        make_method=None, space="pixel", reg_cols=reg_w,
        k_folds=k_folds, fold_kind=fold_kind, b_scales=b_scales, seed=seed,
        progress_every=progress_every)
    # Keep only the new rows; the baseline's are already in from the grid pass.
    all_rows += [r for r in win_rows if r["method"] != METHOD_BASE]

    config = _withholding_config(lane, "pixel", prior_w, k_folds, fold_kind, b_scales, seed,
                                 reg_w, extra={"selected": win, "sel_tol": sel_tol,
                                               "rep_var_full": _rep_var_full(predictions),
                                               **grid_record})
    _write_withholding_artifacts(out_dir, lane, all_rows, predictions, config)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Analog-ensemble tuning: (k, hybrid_w, b_scale) at an inherited taper.
# ---------------------------------------------------------------------------
def analog_term_states(tendency_theta_grid, tendency_lag_yr_grid, redundancy_theta_grid):
    """Distinct ``(tendency_theta, tendency_lag_yr, redundancy_theta)`` states.

    The lag is irrelevant at zero tendency weight, so those combinations collapse to one.
    """
    states = []
    for theta in tendency_theta_grid:
        lags = tendency_lag_yr_grid if theta > 0.0 else (0.0,)
        for lag in lags:
            for redundancy in redundancy_theta_grid:
                states.append((float(theta), float(lag), float(redundancy)))
    return states


def _analog_grid_configs(k_grid, hybrid_w_grid,
                         tendency_theta_grid=(0.0,), tendency_lag_yr_grid=(0.0,),
                         redundancy_theta_grid=(0.0,)):
    """The analog grid points, and a JSON-safe record of the axes.

    The extension-term grids default to off, giving the published estimator's grid.
    """
    states = analog_term_states(tendency_theta_grid, tendency_lag_yr_grid,
                                redundancy_theta_grid)
    configs = [(k, w, *state) for k, w in itertools.product(k_grid, hybrid_w_grid)
               for state in states]
    record = {"k_grid": [int(k) for k in k_grid],
              "hybrid_w_grid": [float(w) for w in hybrid_w_grid],
              "tendency_theta_grid": [float(t) for t in tendency_theta_grid],
              "tendency_lag_yr_grid": [float(g) for g in tendency_lag_yr_grid],
              "redundancy_theta_grid": [float(r) for r in redundancy_theta_grid]}
    return configs, record


def _mt_kwargs(extra_lags_yr, curvature_yr, normalise, preserve_trace):
    """The flow-stack settings split into the row columns and the estimator keywords.

    The lag tuples go to the config JSON rather than the row columns.
    """
    switches = {"tendency_normalise": bool(normalise),
                "preserve_obs_trace": bool(preserve_trace)}
    stack = {"tendency_extra_lags_yr": tuple(float(v) for v in extra_lags_yr),
             "tendency_curvature_yr": tuple(float(v) for v in curvature_yr), **switches}
    return switches, stack


def _mt_record(stack: dict) -> dict:
    """JSON-safe record of the MTA stack an analog grid held fixed."""
    return {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in stack.items()}


def run_hgaoenkf_ppe_grid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    k_grid=K_GRID, hybrid_w_grid=HYBRID_W_GRID,
    tendency_theta_grid=(0.0,), tendency_lag_yr_grid=(0.0,), redundancy_theta_grid=(0.0,),
    selection: str = ANALOG_MISFIT, evidence_scale: float = EVIDENCE_SCALE,
    analog_localization_km: float | None = None,
    tendency_extra_lags_yr: tuple[float, ...] = (),
    tendency_curvature_yr: tuple[float, ...] = (),
    tendency_normalise: bool = False, preserve_obs_trace: bool = False,
    estimator: str | None = None,
    b_scales: tuple[float, ...] = B_SCALES, truth_stride: int = 1,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
    progress_every: int | None = None, min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """Same-model PPE tuned over the analog grid: full-grid metrics, winner-only fields.

    The taper is inherited from 3DVar, so the comparison is between estimators rather than
    between regularisations. ``selection`` names the analog rule; ``estimator`` overrides
    the row tag, e.g. for MTA-HGAOEnKF, which shares a rule with another estimator.
    """
    ages_i = np.asarray(ages, dtype=np.int64)
    prior_idx, truth_idx = chronological_half_split(ages_i, stride=truth_stride)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    truth_cube = cube[truth_idx].astype(np.float64)
    truth_clim = truth_cube.mean(axis=0)
    truth_anoms = truth_cube - truth_clim
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}
    if estimator is None:
        estimator = hgaoenkf_estimator(selection)
    label = method_label(estimator)
    switches, stack = _mt_kwargs(tendency_extra_lags_yr, tendency_curvature_yr,
                                 tendency_normalise, preserve_obs_trace)
    configs, grid_record = _analog_grid_configs(
        k_grid, hybrid_w_grid, tendency_theta_grid, tendency_lag_yr_grid,
        redundancy_theta_grid)

    def score(k, w, tendency_theta, tendency_lag_yr, redundancy_theta,
              full_metrics, progress=None):
        terms = dict(tendency_theta=tendency_theta, tendency_lag_yr=tendency_lag_yr,
                     redundancy_theta=redundancy_theta)
        # At zero weight the stack is dropped, since the estimator rejects a stack it
        # cannot use.
        on = tendency_theta > 0.0
        return _score_ppe_lane(
            truth_anoms, prior, long, lats, lons,
            lane=LANE_PPE, space="pixel", reg_cols=reg_cols,
            make_method=make_hgaoenkf(cube, ages, lats, lons, k=k, hybrid_w=w,
                                      selection=selection, evidence_scale=evidence_scale,
                                      analog_localization_km=analog_localization_km,
                                      **terms, **(stack if on else {})),
            estimator=estimator,
            method_cols=analog_cols(k, w, **terms, **(switches if on else {})),
            b_scales=b_scales, dist_edges_km=dist_edges_km, seed=seed,
            full_metrics=full_metrics, min_obs=min_obs,
            npz_extra={"truth_clim": truth_clim}, progress_every=progress)

    all_rows: list[dict] = []
    t0 = time.time()
    for ci, config in enumerate(configs):
        rows, _, _ = score(*config, full_metrics=False)
        all_rows += [r for r in rows if r["method"] == label]
        _report_progress("analog-grid config", ci + 1, len(configs), t0)

    win = select_analog_config(_selection_rrmse(all_rows, LANE_PPE, label))
    win_rows, npz_arrays, skill = score(
        win["analog_k"], win["hybrid_w"], win["tendency_theta"], win["tendency_lag_yr"],
        win["redundancy_theta"], full_metrics=True, progress=progress_every)
    all_rows += [r for r in win_rows if r["method"] == label and r["metric"] in _FULL_METRICS]
    all_rows += [r for r in win_rows if r["method"] != label]

    config = _ppe_config(
        LANE_PPE, "pixel", prior, len(truth_anoms), b_scales, seed, reg_cols,
        _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride),
        _PPE_NETWORKS,
        extra={"estimator": estimator, "selection": selection,
               "evidence_scale": float(evidence_scale),
               "analog_localization_km": analog_localization_km, "min_obs": min_obs,
               **_mt_record(stack), "selected": win, **grid_record})
    _write_ppe_artifacts(out_dir, LANE_PPE, all_rows, npz_arrays, skill, config,
                         b_scale=win["b_scale"])
    return pd.DataFrame(all_rows)


def run_hgaoenkf_withholding_grid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    k_grid=K_GRID, hybrid_w_grid=HYBRID_W_GRID,
    tendency_theta_grid=(0.0,), tendency_lag_yr_grid=(0.0,), redundancy_theta_grid=(0.0,),
    exclude_yr: float = EXCLUDE_YR,
    selection: str = ANALOG_MISFIT, evidence_scale: float = EVIDENCE_SCALE,
    analog_localization_km: float | None = None,
    tendency_extra_lags_yr: tuple[float, ...] = (),
    tendency_curvature_yr: tuple[float, ...] = (),
    tendency_normalise: bool = False, preserve_obs_trace: bool = False,
    estimator: str | None = None,
    temporal_mode: str = TEMPORAL_DEFLATE,
    report_temporal_modes: tuple[str, ...] = (TEMPORAL_DEFLATE,),
    k_folds: int = 5, fold_kind: str = "random",
    b_scales: tuple[float, ...] = B_SCALES, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Withholding lane tuned over the analog grid: full-grid metrics, winner-only predictions.

    This lane's prior spans the target age, so ``exclude_yr`` stops selection from picking
    the simulation's own state there. ``temporal_mode`` is fixed during the grid;
    ``report_temporal_modes`` are the treatments the winner is then scored under.
    ``selection`` and ``estimator`` are as in :func:`run_hgaoenkf_ppe_grid`.
    """
    prior_idx = np.arange(len(ages))
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}
    if estimator is None:
        estimator = hgaoenkf_estimator(selection)
    label = method_label(estimator, temporal_mode)
    switches, stack = _mt_kwargs(tendency_extra_lags_yr, tendency_curvature_yr,
                                 tendency_normalise, preserve_obs_trace)
    configs, grid_record = _analog_grid_configs(
        k_grid, hybrid_w_grid, tendency_theta_grid, tendency_lag_yr_grid,
        redundancy_theta_grid)

    def score(k, w, tendency_theta, tendency_lag_yr, redundancy_theta, modes,
              progress=None):
        terms = dict(tendency_theta=tendency_theta, tendency_lag_yr=tendency_lag_yr,
                     redundancy_theta=redundancy_theta)
        on = tendency_theta > 0.0
        return _score_withholding_lane(
            prior, long, ages, lats, lons, cube=cube, prior_age_indices=prior_idx,
            space="pixel", reg_cols=reg_cols,
            make_method=make_hgaoenkf(cube, ages, lats, lons, k=k, hybrid_w=w,
                                      exclude_yr=exclude_yr, selection=selection,
                                      evidence_scale=evidence_scale,
                                      analog_localization_km=analog_localization_km,
                                      **terms, **(stack if on else {})),
            estimator=estimator,
            method_cols=analog_cols(k, w, **terms, **(switches if on else {})),
            temporal_modes=modes, k_folds=k_folds, fold_kind=fold_kind,
            b_scales=b_scales, seed=seed, progress_every=progress)

    all_rows: list[dict] = []
    t0 = time.time()
    for ci, config in enumerate(configs):
        lane, rows, _ = score(*config, (temporal_mode,))
        all_rows += [r for r in rows if r["method"] == label]
        _report_progress(f"analog-grid config ({lane})", ci + 1, len(configs), t0)

    win = select_analog_config(_selection_rrmse(all_rows, lane, label))
    _, win_rows, predictions = score(
        win["analog_k"], win["hybrid_w"], win["tendency_theta"], win["tendency_lag_yr"],
        win["redundancy_theta"], report_temporal_modes, progress=progress_every)
    # The re-run adds the other treatments, the prior-free rows and the predictions npz.
    all_rows += [r for r in win_rows if r["method"] != label]

    config = _withholding_config(
        lane, "pixel", prior, k_folds, fold_kind, b_scales, seed, reg_cols,
        extra={"estimator": estimator, "selection": selection,
               "evidence_scale": float(evidence_scale),
               "analog_localization_km": analog_localization_km,
               **_mt_record(stack), "selected": win,
               "exclude_yr": exclude_yr, "temporal_mode": temporal_mode,
               "report_temporal_modes": list(report_temporal_modes),
               "rep_var_full": _rep_var_full(predictions), **grid_record})
    _write_withholding_artifacts(out_dir, lane, all_rows, predictions, config)
    return pd.DataFrame(all_rows)


def analog_variant_estimator(rule: str, exclude_yr: float) -> str:
    """Estimator tag for one width of the exclusion band."""
    return f"{ESTIMATOR_HGAOENKF}_{rule}_excl{int(exclude_yr)}"


def analog_localization_estimator(rule: str, km: float | None) -> str:
    """Estimator tag for one lengthscale of the analog covariance's own taper.

    ``None`` inherits the static lengthscale and is tagged ``static``.
    """
    suffix = "static" if km is None else f"{int(km)}"
    return f"{hgaoenkf_estimator(rule)}_loc{suffix}"


def evidence_scale_estimator(scale: float) -> str:
    """Estimator tag for one value of the evidence rule's background scale."""
    return f"{hgaoenkf_estimator(ANALOG_EVIDENCE)}_c{scale:g}"


def mt_stack_estimator(name: str) -> str:
    """Estimator tag for one MTA stack, named by its entry in :data:`MT_STACKS`."""
    return f"{ESTIMATOR_HGAOENKF_MT}_{name}"


def run_hgaoenkf_withholding_variants(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    k: int, hybrid_w: float, variants: tuple[tuple[str, dict], ...],
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    evidence_scale: float = EVIDENCE_SCALE,
    temporal_mode: str = TEMPORAL_DEFLATE,
    k_folds: int = 5, fold_kind: str = "random",
    b_scales: tuple[float, ...] = B_SCALES, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Score variations on how the analog covariance is built, at a fixed ``(k, hybrid_w)``.

    ``variants`` pairs an estimator tag with the :func:`make_hgaoenkf` keywords it changes.
    These are sensitivity passes, not tuning; their rows are written to their own directory.
    """
    prior_idx = np.arange(len(ages))
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}

    all_rows: list[dict] = []
    t0 = time.time()
    for vi, (tag, kw) in enumerate(variants):
        lane, rows, _ = _score_withholding_lane(
            prior, long, ages, lats, lons, cube=cube, prior_age_indices=prior_idx,
            space="pixel", reg_cols=reg_cols,
            make_method=make_hgaoenkf(cube, ages, lats, lons, k=k, hybrid_w=hybrid_w,
                                      evidence_scale=evidence_scale, **kw),
            estimator=tag,
            # Record any extension term or MTA setting the variant changes.
            method_cols=analog_cols(k, hybrid_w, **{key: kw[key]
                                                    for key in TERM_KEYS + _STACK_KEYS
                                                    if key in kw}),
            temporal_modes=(temporal_mode,),
            k_folds=k_folds, fold_kind=fold_kind, b_scales=b_scales, seed=seed)
        # The prior-free rows are already written by the grid pass.
        all_rows += [r for r in rows if r["background"] != "none"]
        _report_progress("analog variant", vi + 1, len(variants), t0)

    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), all_rows)
    config = {"lane": lane, "space": "pixel", **reg_cols, "analog_k": int(k),
              "hybrid_w": float(hybrid_w), "temporal_mode": temporal_mode,
              "variants": {tag: {key: val for key, val in kw.items()}
                           for tag, kw in variants},
              "evidence_scale": float(evidence_scale),
              "k_folds": k_folds, "fold_kind": fold_kind, "seed": seed,
              "b_scales": [float(b) for b in b_scales], "prior_meta": prior.meta}
    with open(os.path.join(out_dir, f"{lane}_variants_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    return pd.DataFrame(all_rows)
