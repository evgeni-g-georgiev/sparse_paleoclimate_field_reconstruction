"""Experiment runners for data-assimilation reconstruction.

Three evaluation lanes, all method-agnostic through the :class:`Method` contract:

* :func:`run_ppe` - same-model pseudo-proxy experiments: the prior cube is split
  chronologically, B and the climatology come from one half, and truths are drawn from
  the other. Each truth borrows several real-proxy network shapes; ``b_scale`` is
  selected on held-out shapes and scored on a disjoint one.
* :func:`run_withholding` - nested cross-validation over proxy sites: select
  ``b_scale`` on a held-out fold, report on a fresh fold.
* :func:`run_trajectory` - a consecutive run of states reconstructed with the real
  network at each age, each observation carrying the state at its own sample's block
  centre rather than at the analysis age. Scores skill by timescale, which the other
  two lanes cannot: they grade one snapshot at a time, and the PPE lane deliberately
  borrows network geometry without its climate time.

:func:`run_ppe_pixel_grid` and :func:`run_withholding_pixel_grid` wrap these over a
coarse localization/shrinkage/coupling grid, jointly selecting the operating point with
``b_scale`` on the held-out selection split and persisting only the winning config's
fields. :func:`run_hgaoenkf_ppe_grid` and :func:`run_hgaoenkf_withholding_grid` do the
same over the analog ensemble size and hybrid weight, at a taper those grids already
settled and under one analog selection rule, which names the estimator through
:func:`hgaoenkf_estimator`; :func:`run_hgaoenkf_withholding_variants` scores how the
ensemble is chosen.

A row's ``method`` composes the estimator with its treatment of observation staleness
(:func:`method_label`), so several estimators share one schema without colliding and every
row says which pair of choices produced it. ``analog_k`` and ``hybrid_w`` carry the analog
parameters and are NaN for an estimator that draws no ensemble.

Each writes a tidy long-format metrics CSV (one row per method/space/localization_km/
shrinkage_lambda/alpha/analog_k/hybrid_w/lane/fold/b_scale/background/split/do_event/
channel/metric), the analysis fields as npz, and a config.json. ``b_scale`` is the background-covariance
amplitude the analysis used; ``split`` is ``selection`` (used to pick the operating
point) or ``test`` (reported). The withholding lane uses ``R = diag(sse + rep_var)``,
adding the per-channel representativeness variance so a point proxy is not trusted to
resolve its grid cell exactly; the pseudo-proxy lane keeps ``diag(sse)`` because its
synthetic observations sample the exact grid cell. Scoring is in anomaly space; CE,
RMSE, and correlation are shift-invariant so the anomaly-space values equal their degC
counterparts.

The withholding and trajectory lanes also carry the temporal twin of ``rep_var``: a
sample is replicated across every age in its block, so at most of them it reports on a
state some way off in time. ``temporal_modes`` chooses which treatments of that to score,
from leaving it alone through charging it as extra observation error to also correcting the
attenuation a lagged state carries, so the treatments sit side by side in one schema and
any two are paired over the same ages. A lane comparing estimators holds the treatment
fixed instead, which keeps the difference between them the estimator. The PPE lane stays
out of it: it borrows network geometry without its climate time, so a sample's distance
from the analysis age measures nothing there.

Every lane emits skill metrics (``ce``, ``corr``, ``rmse``, ``rrmse``, ``amplitude``,
plus field-only ``ssim``) and calibration metrics (``crps``, ``crpss``, ``rcrv_bias``,
``rcrv_dispersion``, ``coverage90``), scored against the prior ``N(0, b_scale diag B)``
as the CRPSS reference so it matches CE's climatology baseline. Every lane also carries
prior-free ``nearest``/``idw`` reference rows, tagged ``background="none"``, which is the
context a bare CE cannot supply. The trajectory lane adds timescale-resolved metrics,
named ``{corr,ce,amp}_lp{window}`` for the low-pass series and ``..._bp{a}_{b}`` for a
band, keeping the timescale in the metric name so the row schema is unchanged.
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

# A method factory turns a built prior and the field shape into an estimator, so a
# lane can run pixel 3DVar or a latent method behind one call. The estimator must
# expose the sweep surface (prepare_sweep / apply_sweep / post_var_sweep) and return
# pixel-space AnalysisResult, like ThreeDVar.
MethodFactory = Callable[[Prior, "tuple[int, int, int]"], Method]

B_SCALES = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
# Coarse tuning grid for the pixel regularizer (localization / shrinkage / channel
# coupling); ``None`` localization is "off", the raw sample covariance.
LOCALIZATION_KM_GRID = (None, 7500.0, 12500.0)
SHRINKAGE_GRID = (0.0, 0.25, 0.5)
ALPHA_GRID = (0.0, 0.5, 1.0)
SEL_TOL = 0.0   # 0 = pure argmin of selection RRMSE; >0 prefers the simpler config within this relative band
# Analog-ensemble tuning grid: how many prior states form the ensemble, and how much of
# the gain their covariance carries against the static one. The weight spans Sun et al.'s
# whole family, from their AOEnKF-B at 0 to their AOEnKF at 1, so the published ablation
# ladder falls out of the sweep rather than needing separate runs.
K_GRID = (20, 40, 60, 100)
HYBRID_W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# The two terms that extend the analog scheme, each swept from off. The tendency weight
# scales the archive's local rate of change into the analog covariance and the lag is the
# interval it is differenced over; the redundancy penalty charges a candidate for
# resembling one already selected. Zero on both axes is the published estimator, so the
# ablation is a set of rows in the same grid rather than a separate run.
TENDENCY_THETA_GRID = (0.0, 1.0)
TENDENCY_LAG_YR_GRID = (200.0, 400.0)
REDUNDANCY_THETA_GRID = (0.0, 0.25)
# Width of the band around the analysis age dropped from the analog pool, and the widths
# a sensitivity pass sweeps. Only bites where the prior spans the age being reconstructed.
EXCLUDE_YR = 1000.0
EXCLUDE_YR_GRID = (0.0, 1000.0, 2000.0)
# Lengthscales for the analog covariance's own taper, swept at a fixed (k, hybrid_w).
# ``None`` is whatever the static covariance carries, which differs by lane, so the two
# grids bracket both that value and the tighter one Sun et al. (2024) Table 2 predicts.
ANALOG_LOCALIZATION_GRID_PPE = (None, 5000.0, 7500.0, 10000.0, 20000.0)
ANALOG_LOCALIZATION_GRID_WH = (None, 5000.0, 10000.0, 15000.0)
LANE_PPE = "ppe"
LANE_TRAJECTORY = "trajectory"

# Which estimator produced a row. The method label composes it with the treatment of
# observation staleness, so two estimators can share the metrics CSV without colliding and
# every row still says which pair of choices it came from.
ESTIMATOR_3DVAR = "3dvar"
ESTIMATOR_HGAOENKF = "hgaoenkf"
_TEMPORAL_SUFFIX = {TEMPORAL_OFF: "", TEMPORAL_ADD: "_temporal_add",
                    TEMPORAL_DEFLATE: "_temporal_deflate"}
# Estimator tag per analog selection rule. The map is explicit rather than derived from the
# rule string so a tag can be renamed without renaming the rule it selects by: the baseline
# is the published one and holds the bare name, and the rule with no published counterpart
# is tagged for what it is rather than for how it scores.
_HGAOENKF_TAG = {
    ANALOG_MISFIT: ESTIMATOR_HGAOENKF,
    ANALOG_EVIDENCE: f"{ESTIMATOR_HGAOENKF}_plus",
    ANALOG_CORRELATION: f"{ESTIMATOR_HGAOENKF}_correlation",
    ANALOG_CORRELATION_PERCHAN: f"{ESTIMATOR_HGAOENKF}_correlation_perchan",
}


def method_label(estimator: str, mode: str = TEMPORAL_OFF) -> str:
    """The metrics-CSV ``method`` for one (estimator, staleness treatment) pair."""
    return f"{estimator}{_TEMPORAL_SUFFIX[mode]}"


def hgaoenkf_estimator(selection: str) -> str:
    """Estimator tag for one analog selection rule."""
    return _HGAOENKF_TAG[selection]


def ceiling_label(estimator: str) -> str:
    """The label for a run with observations taken at the analysis age.

    Not a treatment of staleness but the bound on what it costs, so it sits outside
    :func:`method_label`'s composition.
    """
    return f"{estimator}_ceiling"


# One method label per treatment, so the three sit side by side in the shared row schema
# and every comparison between them is paired over the same ages.
METHOD_BASE = method_label(ESTIMATOR_3DVAR)
METHOD_CEILING = ceiling_label(ESTIMATOR_3DVAR)
TEMPORAL_METHODS = {method_label(ESTIMATOR_3DVAR, mode): mode for mode in TEMPORAL_MODES}
TRAJECTORY_METHODS = tuple(TEMPORAL_METHODS) + (METHOD_CEILING,)

# Timescales the trajectory lane resolves. Low-pass keeps everything slower than the
# window, so it is cumulative and reads high wherever the slow components carry the
# variance; a band subtracts one low-pass from another and is the honest per-timescale
# view. Both are reported.
# The widest band is bounded by the split: edge trimming costs a full window at each end,
# so a band wider than the reported block leaves nothing to score.
LOWPASS_WINDOWS = (25, 100, 250, 500, 1000, 2000)
BANDS = ((25, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000))
# Metrics the RRMSE selection never reads, so a grid scan computes them for the winner only.
_FULL_METRICS = frozenset({"ssim", "crps", "crpss", "rcrv_bias", "rcrv_dispersion",
                           "coverage90"})

# Taper columns for prior-free (naive) rows, which carry no regularizer.
_NAN_REG = {"localization_km": np.nan, "shrinkage_lambda": np.nan, "alpha": np.nan}
# The extension terms, named once so a row's columns cannot drift from the keywords the
# estimator was built with.
TERM_KEYS = ("tendency_theta", "tendency_lag_yr", "redundancy_theta")
# Analog-ensemble columns for rows from an estimator that draws no analog ensemble.
_NAN_ANALOG = {"analog_k": np.nan, "hybrid_w": np.nan,
               **{key: np.nan for key in TERM_KEYS}}


def analog_cols(k: int, hybrid_w: float, *, tendency_theta: float = 0.0,
                tendency_lag_yr: float = 0.0, redundancy_theta: float = 0.0) -> dict:
    """The analog columns for a row, as :func:`_NAN_REG` does for the taper.

    The extension terms default to off, which is what an estimator that does not carry
    them is: a row saying zero and a row from before they existed mean the same thing.
    """
    return {"analog_k": float(k), "hybrid_w": float(hybrid_w),
            "tendency_theta": float(tendency_theta),
            "tendency_lag_yr": float(tendency_lag_yr),
            "redundancy_theta": float(redundancy_theta)}


# ---------------------------------------------------------------------------
# Observation geometry.
# ---------------------------------------------------------------------------
def _obs_geometry(o: dict, lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray) -> dict:
    """Gather indices, error variance, and site coords for usable observations.

    ``keep`` is the mask the other fields were filtered by, so a caller can carry an
    extra column of ``o`` through the same filter without restating the rule.
    """
    gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
    keep = safe_flat[gather] & (o["sse"] > 0)
    return {
        "gather": gather[keep],
        "sse": o["sse"][keep].astype(np.float64),
        "lat": o["lat"][keep],
        "lon": o["lon"][keep],
        "keep": keep,
    }


def _draw_usable_ages(long: pd.DataFrame, rng: np.random.Generator,
                      lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray,
                      n: int) -> list[int]:
    """``n`` distinct real proxy ages, each borrowing a usable network on the grid.

    Drawn from the shared ``rng`` so a run is reproducible from its seed; an age
    contributes only network geometry, not any climate time. The distinct shapes give
    one truth several sparsity patterns, so ``b_scale`` can be selected on some shapes
    and scored on a held-out other.
    """
    picked = []
    for a in rng.permutation(long["age"].unique()):
        a = int(a)
        o = observations_at_age(long, a)
        if len(o.get("age", [])) and len(_obs_geometry(o, lats, lons, safe_flat)["gather"]):
            picked.append(a)
            if len(picked) == n:
                return picked
    raise ValueError(f"fewer than {n} proxy ages have a usable network on this grid")


def _pad_obs(test_obs: list[dict], T: int) -> tuple[np.ndarray, ...]:
    """Ragged per-truth test-shape obs to padded ``(T, max_obs)`` arrays for npz.

    ``obs_n`` gives the real count per truth so the field gallery can drop the padding
    when overlaying the assimilated sites.
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

    No field is built, unlike :func:`_naive_geometry`: withholding only ever reads the
    reconstruction at the withheld sites, so one per-channel great-circle block between
    target and assimilated sites yields both baselines and the void distance. A channel
    with no assimilated observation keeps the climatological zero.
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

    ``groups`` is ``(do_event, channel_name, mask)`` triples, so a field lane can group by
    event and channel while an observation lane groups by channel alone. ``var`` is the
    predictive variance the residual is scored against: the posterior variance for a
    noise-free truth, plus the observation error when the truth is itself a measurement.
    The CRPSS reference is the prior ``N(0, ref_var)``, the same do-nothing forecast CE
    scores against.
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

    Flattens over (truth, channel, valid cell) once and masks per group so the event and
    channel breakdowns share one CRPS evaluation. The truth is a model state carrying no
    observation error, so the predictive variance is the posterior variance alone.
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

    Field space only: SSIM is a 2-D structural metric with no observation-space
    analogue. ``data_range`` is per channel over the valid cells of the whole truth
    stack, fixed so per-truth SSIMs share the stabilising constants and average.
    The ``pooled`` channel is the mean of the per-channel SSIMs (multichannel SSIM),
    not the concatenation pooling used for CE/RMSE.
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

    The median over cells rather than a pooled or regional statistic: after a wide
    filter a regional index retains few independent points and its correlation swings
    on the handful that remain.
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

    Emits ``{corr,ce,amp}_lp{window}`` from the low-pass series and ``..._bp{a}_{b}``
    from the difference of two, which is the only view that separates skill at one
    timescale from skill inherited off a slower component carrying most of the variance.
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
    """Append metric rows to the tidy CSV, writing the header once."""
    df = pd.DataFrame(rows)
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
    b_scales: tuple[float, ...], n_shapes: int, n_select: int, n_noise: int,
    dist_edges_km: np.ndarray | None, seed: int, full_metrics: bool = True,
    npz_extra: dict | None = None, progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
) -> tuple[list[dict], dict, dict]:
    """Score a built prior against a stack of truth anomalies (no file writes).

    Returns ``(rows, npz_arrays, skill)``: the tidy metric rows, the analysis-npz dict,
    and the skill-vs-distance dict; :func:`_write_ppe_artifacts` persists them.

    Held-out selection: each truth draws ``n_shapes`` distinct real-proxy network shapes;
    the first ``n_select`` form the ``selection`` split, the last the ``test`` split.
    Metric rows for both splits span the whole ``b_scales`` sweep (``split`` column), so
    the operating point is chosen on the selection shapes and reported on the disjoint
    test shape. R is ``diag(sse)``; each pseudo-obs is the truth at its nearest cell plus
    ``N(0, sse)`` noise, averaged over ``n_noise`` draws. Single climatological
    background. ``make_method`` selects the estimator (default pixel :class:`ThreeDVar`);
    ``space``/``reg_cols`` tag the rows, as do ``estimator`` (which names the rows' method)
    and ``method_cols`` (the analog columns, absent for a static-covariance estimator).
    ``full_metrics=False`` skips the field SSIM and calibration rows (neither feeds the
    RRMSE selection), the free saving the grid scan takes.

    Calibration is scored on the test shape only, where the posterior variance is kept,
    and against the noise-free truth so the predictive variance is the posterior alone.
    Averaging the analysis over ``n_noise`` draws makes it mildly conservative relative to
    a single realization; ``n_noise=1`` recovers the single-realization calibration.
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

    # Selection shapes are pooled over shape x truth; the test shape is kept per truth
    # for scoring and the field gallery.
    recon_sel = np.zeros((n_select, n_b, T, *shape))
    recon_test = np.zeros((n_b, T, *shape))
    post_test = np.zeros((n_b, T, *shape))
    naive_test = {"nearest": np.zeros((T, *shape)), "idw": np.zeros((T, *shape))}
    dist_test, test_obs = [], []
    drawn_ages = np.zeros((T, n_shapes), dtype=np.int64)

    t0 = time.time()
    for ti, truth_anom in enumerate(truth_anoms):
        shape_ages = _draw_usable_ages(long, rng, lats, lons, safe_flat, n_shapes)
        drawn_ages[ti] = shape_ages
        for si, k_age in enumerate(shape_ages):
            o = observations_at_age(long, int(k_age))
            geom = _obs_geometry(o, lats, lons, safe_flat)
            g = geom["gather"]
            truth_at_obs = truth_anom.ravel()[g]         # obs values from the truth, H = nearest cell
            gain = tv.prepare_sweep(g, geom["sse"], b_scales)   # R = diag(sse)
            naive_geom = _naive_geometry(lats, lons, geom, len(VARS))

            sum_recon = np.zeros((n_b, *shape))
            sum_post = np.zeros((n_b, *shape))
            sum_naive = {kind: np.zeros(shape) for kind in naive_test}
            for _ in range(n_noise):
                y = truth_at_obs + rng.normal(0.0, np.sqrt(geom["sse"]))
                res = tv.apply_sweep(gain, y, zero_bg)
                for bj in range(n_b):
                    sum_recon[bj] += res[bj].mean_anom
                    sum_post[bj] += res[bj].posterior_var
                for kind in sum_naive:
                    sum_naive[kind] += _naive_apply(kind, naive_geom, y, shape)
            sum_recon /= n_noise

            if si < n_select:
                recon_sel[si, :, ti] = sum_recon
            else:
                recon_test[:, ti] = sum_recon
                # Read off the analyses rather than asked for separately: an estimator
                # whose covariance depends on the observations has no value-free spread.
                post_test[:, ti] = sum_post / n_noise
                for kind in naive_test:
                    naive_test[kind][ti] = sum_naive[kind] / n_noise
                dist_test.append(da.nearest_obs_distance(lats, lons, geom["lat"], geom["lon"]))
                test_obs.append({"lat": geom["lat"], "lon": geom["lon"],
                                 "val": truth_at_obs, "chan": g // n_cells})
        if progress_every and (ti + 1) % progress_every == 0:
            _report_progress("truth", ti + 1, T, t0)

    events = np.zeros(T, dtype=np.int64)                 # PPE truths carry no DO-event labels
    truth_sel = np.tile(truth_anoms, (n_select, 1, 1, 1))
    events_sel = np.zeros(n_select * T, dtype=np.int64)

    rows = []
    for bj, kb in enumerate(b_scales):
        base = {"method": method_label(estimator), "space": space, **reg_cols, **method_cols,
                "lane": lane, "fold": -1, "b_scale": kb,
                "background": "climatological", "split": "test"}
        rows += _skill_rows(truth_anoms, recon_test[bj], safe_valid, events, base)
        base_sel = {**base, "split": "selection"}
        recon_pool = recon_sel[:, bj].reshape(n_select * T, *shape)
        rows += _skill_rows(truth_sel, recon_pool, safe_valid, events_sel, base_sel)
        if full_metrics:
            rows += _ssim_rows(truth_anoms, recon_test[bj], safe_valid, events, base)
            rows += _ssim_rows(truth_sel, recon_pool, safe_valid, events_sel, base_sel)
            rows += _field_calibration_rows(truth_anoms, recon_test[bj], post_test[bj],
                                            kb * tv.diagB.reshape(shape), safe_valid,
                                            events, base)
    for kind in naive_test:
        base = {"method": kind, "space": space, **_NAN_REG, **_NAN_ANALOG,
                "lane": lane, "fold": -1, "b_scale": 1.0,
                "background": "none", "split": "test"}
        rows += _skill_rows(truth_anoms, naive_test[kind], safe_valid, events, base)
        if full_metrics:
            rows += _ssim_rows(truth_anoms, naive_test[kind], safe_valid, events, base)

    if dist_edges_km is None:
        dist_edges_km = np.array([0, 500, 1000, 2000, 3000, 5000, 8000, 20000], dtype=float)
    dist = np.stack(dist_test)
    cell_mask = safe_valid.ravel()
    dist_pool = np.tile(dist[:, cell_mask], (1, len(VARS))).ravel()
    tv_safe = truth_anoms[:, :, safe_valid]
    t_pool = tv_safe.reshape(T, -1).ravel()
    curve = np.array([da.skill_vs_distance(
        t_pool, recon_test[bj][:, :, safe_valid].reshape(T, -1).ravel(),
        np.zeros_like(t_pool), dist_pool, dist_edges_km)["ce"]
        for bj in range(n_b)])                           # (n_b, n_bins)

    obs_lat, obs_lon, obs_val, obs_chan, obs_n = _pad_obs(test_obs, T)
    npz_arrays = {
        "truth_anom": truth_anoms, "clim_mean": clim_mean,
        "safe_valid": safe_valid, "post_var": post_test, "prior_var": tv.diagB.reshape(shape),
        "lats": np.asarray(lats), "lons": np.asarray(lons),
        "drawn_ages": drawn_ages, "b_scales": np.asarray(b_scales),
        "recon_climatological": recon_test,              # (n_b, T, 2, n_lat, n_lon)
        "naive_nearest": naive_test["nearest"], "naive_idw": naive_test["idw"],
        "obs_lat": obs_lat, "obs_lon": obs_lon, "obs_val": obs_val,
        "obs_chan": obs_chan, "obs_n": obs_n,
        **npz_extra,
    }
    skill = {"edges": dist_edges_km, "b_scales": np.asarray(b_scales), "climatological": curve}
    return rows, npz_arrays, skill


def _write_ppe_artifacts(out_dir: str, lane: str, rows: list[dict],
                         npz_arrays: dict, skill: dict, config: dict) -> None:
    """Persist a scored PPE lane: metrics CSV (appended), analysis npz, skill npz, config."""
    os.makedirs(out_dir, exist_ok=True)
    _append_csv(os.path.join(out_dir, "metrics.csv"), rows)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_analysis.npz"), **npz_arrays)
    np.savez_compressed(os.path.join(out_dir, f"{lane}_skill_vs_distance.npz"), **skill)
    with open(os.path.join(out_dir, f"{lane}_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def run_ppe(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    make_method: MethodFactory | None = None, space: str = "pixel",
    b_scales: tuple[float, ...] = B_SCALES,
    n_shapes: int = 5, n_select: int = 4, n_noise: int = 20, truth_stride: int = 10,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
    progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
) -> pd.DataFrame:
    """Same-model PPE for one taper config: truths are a held-out chronological chunk.

    The ascending age axis splits at its midpoint; the older half builds B and the
    climatology, the younger half (subsampled every ``truth_stride`` states) supplies
    truths, each anomalised by the subsampled set's own mean so the inter-chunk offset
    cancels. Scoring and held-out selection follow :func:`_score_ppe_lane`; rows and
    files are tagged ``ppe``. :func:`run_ppe_pixel_grid` wraps this over the taper grid.
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
        b_scales=b_scales, n_shapes=n_shapes, n_select=n_select, n_noise=n_noise,
        dist_edges_km=dist_edges_km, seed=seed, npz_extra={"truth_clim": truth_clim},
        progress_every=progress_every, estimator=estimator, method_cols=method_cols)
    config = _ppe_config(LANE_PPE, space, prior, len(truth_anoms), n_shapes, n_select,
                         n_noise, b_scales, seed, reg_cols,
                         _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride))
    _write_ppe_artifacts(out_dir, LANE_PPE, rows, npz_arrays, skill, config)
    return pd.DataFrame(rows)


def _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride):
    """Split-provenance keys for a same-model PPE config.json."""
    return {"split": "chronological_midpoint", "prior_half": "older",
            "split_index": int(prior_idx[0]), "truth_stride": int(truth_stride),
            "chunk_a_ages": [int(ages_i[prior_idx].min()), int(ages_i[prior_idx].max())],
            "chunk_b_ages": [int(ages_i[truth_idx].min()), int(ages_i[truth_idx].max())]}


def _ppe_config(lane, space, prior, n_truths, n_shapes, n_select, n_noise,
                b_scales, seed, reg_cols, split_meta, extra=None):
    """Assemble a PPE lane config.json dict (single-config or grid winner)."""
    cfg = {"lane": lane, "space": space, **reg_cols,
           "n_truths": int(n_truths), "n_shapes": n_shapes, "n_select": n_select,
           "n_noise": n_noise, "b_scales": [float(b) for b in b_scales], "seed": seed,
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

    Sizes the structure function to the network that will use it, so no observation
    has to fall back on a clipped lag.
    """
    lag = (long["age"] - long["centre"]).abs().to_numpy(dtype=np.float64)
    return int(np.floor(lag.max() / step_yr + 0.5)) if len(lag) else 0


def _trajectory_observations(long: pd.DataFrame, age: int, lats: np.ndarray, lons: np.ndarray,
                             safe_flat: np.ndarray, min_obs: int) -> dict | None:
    """Usable observations at one age, with each sample's own position on the age axis.

    ``None`` when the network is too thin to assimilate, which the caller records so
    the count of scored ages is auditable.
    """
    o = observations_at_age(long, int(age))
    if not len(o.get("age", [])):
        return None
    geom = _obs_geometry(o, lats, lons, safe_flat)
    if len(geom["gather"]) < min_obs:
        return None
    geom["centre"] = o["centre"][geom["keep"]]
    return geom


def run_trajectory(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    make_method: MethodFactory | None = None, space: str = "pixel",
    b_scales: tuple[float, ...] = B_SCALES,
    lowpass_windows: tuple[int, ...] = LOWPASS_WINDOWS,
    bands: tuple[tuple[int, int], ...] = BANDS,
    min_obs: int = 10, sel_tol: float = SEL_TOL, seed: int = 0,
    progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
    temporal_modes: tuple[str, ...] = TEMPORAL_MODES,
) -> pd.DataFrame:
    """Reconstruct a consecutive run of states and score skill by timescale.

    The older half of the age axis builds B and the climatology; every state of the
    younger half is a truth, so the analyses form a time series rather than a set of
    independent snapshots. At each age the observation network is the real one at that
    age, and each pseudo-observation carries the truth at its own sample's block centre:
    a sample describes its own moment, not the moment being assimilated. The ceiling rows
    repeat the run with observations taken at the analysis age, which bounds what the
    staleness costs.

    Between those two sit the temporal variants, which answer the staleness rather than
    remove it: one charges it as extra observation error, the other also corrects the
    attenuation a lagged state carries. Their structure function comes from the prior
    ages alone, so the truth half never informs the operator that reconstructs it.
    ``temporal_modes`` chooses which of them to run, so a lane comparing estimators can
    hold the treatment fixed at one rather than re-deciding it per estimator.

    One noise draw per age, shared by every variant. Averaging draws as the PPE lane
    does would smooth the reconstructed series and flatter exactly the high-frequency
    skill the band metrics exist to measure, and sharing the draw leaves the treatment
    of observation timing as the only difference between the variants.

    The tapers are inputs, not a grid: they answer a spatial question the PPE lane
    already settles, and re-gridding them over a 400-state run is not affordable. Only
    ``b_scale`` is swept, selected on the earlier half of the run by pooled RRMSE, the
    same criterion the other lanes use, and reported on the later half. Timescale
    metrics are therefore never optimised against.
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
    # A block centre near the chunk boundary can land in the older half, so observations
    # are read from the whole run. Centring on the truth climatology keeps one anomaly
    # frame; B still sees prior ages only, so nothing about the truth half leaks into it.
    all_anoms = cube.reshape(len(ages_i), -1).astype(np.float64) - truth_clim.ravel()
    long = sample_block_centres(long)
    S, prior_var_cell = temporal_structure_function(
        cube, prior_idx, max_lag=_max_block_lag(long, step_yr))

    method_cols = method_cols or _NAN_ANALOG
    labels = {method_label(estimator, mode): mode for mode in temporal_modes}
    base_label = next(iter(labels))
    ceiling = ceiling_label(estimator)

    rng = np.random.default_rng(seed)
    covered, skipped = [], []
    recon = {m: [] for m in (*labels, ceiling)}
    naive = {"nearest": [], "idw": []}
    post = {m: [] for m in labels}
    analog_index = []
    obs_n = []
    t0 = time.time()
    for ti, age in enumerate(ages_i[truth_idx]):
        geom = _trajectory_observations(long, int(age), lats, lons, safe_flat, min_obs)
        if geom is None:
            skipped.append(int(age))
            continue
        g, sse = geom["gather"], geom["sse"]
        noise = rng.normal(0.0, np.sqrt(sse))
        src = nearest_age_index(geom["centre"], ages_i)
        stale = all_anoms[src, g] + noise
        # The lag is read off the age the observation was actually drawn from, not off
        # the raw block centre, so the correction answers the staleness this lane built
        # rather than a rounding of it.
        rho, resid = temporal_terms(S, prior_var_cell, g,
                                    np.abs(ages_i[src] - age), step_yr)

        obs = {}
        for method, mode in labels.items():
            yv, r = apply_temporal_error(stale, sse, rho, resid, mode)
            obs[method] = (yv, tv.prepare_sweep(g, r, b_scales))
        # Observations at the analysis age carry no staleness, so the ceiling assimilates
        # them under the uncorrected R.
        obs[ceiling] = (truth_anoms[ti].ravel()[g] + noise, tv.prepare_sweep(g, sse, b_scales))

        for method, (yv, gain) in obs.items():
            res = tv.apply_sweep(gain, yv, zero_bg)
            recon[method].append(np.stack([r.mean_anom for r in res]))
            if method in post:
                post[method].append(np.stack([r.posterior_var for r in res]))
        # Which prior states the analysis drew on, where the estimator chooses them. The
        # set turns over between neighbouring ages, which is a high-frequency source a
        # fixed-covariance analysis does not have, so the band metrics need it alongside.
        if hasattr(tv, "select"):
            analog_index.append(tv.select(obs[base_label][1], obs[base_label][0]))
        naive_geom = _naive_geometry(lats, lons, geom, len(VARS))
        for kind in naive:
            naive[kind].append(_naive_apply(kind, naive_geom, obs[base_label][0], shape))
        covered.append(ti)
        obs_n.append(len(g))
        if progress_every and (len(covered) % progress_every == 0):
            _report_progress("trajectory age", ti + 1, len(truth_idx), t0)

    if len(covered) < 4:
        raise ValueError(f"only {len(covered)} ages had a usable network; nothing to score")

    covered = np.asarray(covered)
    truth_run = truth_anoms[covered]
    recon = {k: np.stack(v, axis=1) for k, v in recon.items()}   # (n_b, n_covered, C, H, W)
    naive = {k: np.stack(v) for k, v in naive.items()}
    post = {k: np.stack(v, axis=1) for k, v in post.items()}
    mid = len(covered) // 2
    blocks = [("selection", slice(0, mid)), ("test", slice(mid, len(covered)))]

    rows: list[dict] = []
    for bj, kb in enumerate(b_scales):
        for method in recon:
            for split, sl in blocks:
                base = {"method": method, "space": space, **reg_cols, **method_cols,
                        "lane": LANE_TRAJECTORY, "fold": -1, "b_scale": kb,
                        "background": "climatological", "split": split}
                events = np.zeros(len(truth_run[sl]), dtype=np.int64)
                rows += _skill_rows(truth_run[sl], recon[method][bj, sl], safe_valid, events, base)
                rows += _ssim_rows(truth_run[sl], recon[method][bj, sl], safe_valid, events, base)
                rows += _timescale_rows(truth_run[sl], recon[method][bj, sl], safe_valid, base,
                                        step_yr=step_yr, windows=lowpass_windows, bands=bands)
                if method in post:
                    rows += _field_calibration_rows(
                        truth_run[sl], recon[method][bj, sl], post[method][bj, sl],
                        kb * tv.diagB.reshape(shape), safe_valid, events, base)
    for kind, field in naive.items():
        for split, sl in blocks:
            base = {"method": kind, "space": space, **_NAN_REG, **_NAN_ANALOG,
                    "lane": LANE_TRAJECTORY, "fold": -1, "b_scale": 1.0,
                    "background": "none", "split": split}
            events = np.zeros(len(truth_run[sl]), dtype=np.int64)
            rows += _skill_rows(truth_run[sl], field[sl], safe_valid, events, base)
            rows += _ssim_rows(truth_run[sl], field[sl], safe_valid, events, base)
            rows += _timescale_rows(truth_run[sl], field[sl], safe_valid, base,
                                    step_yr=step_yr, windows=lowpass_windows, bands=bands)

    win = select_best_config(_selection_rrmse(rows, LANE_TRAJECTORY, base_label),
                             sel_tol=sel_tol)
    bw = int(np.argmin(np.abs(np.asarray(b_scales) - win["b_scale"])))
    covered_ages = ages_i[truth_idx][covered]
    # Fields for the winner only, float32: the whole sweep over a 400-state run is two
    # orders of magnitude larger than the PPE lane's and nothing reads the losing scales.
    npz_arrays = {
        "truth_anom": truth_run.astype(np.float32), "truth_clim": truth_clim,
        "clim_mean": prior.clim_mean.astype(np.float64), "safe_valid": safe_valid,
        "lats": lats, "lons": lons, "ages": covered_ages,
        "b_scales": np.asarray(b_scales), "selected_b_scale": np.asarray(b_scales[bw]),
        "split_index": np.asarray(mid), "step_yr": np.asarray(step_yr),
        "recon_realistic": recon[base_label][bw].astype(np.float32),
        "recon_ceiling": recon[ceiling][bw].astype(np.float32),
        "naive_nearest": naive["nearest"].astype(np.float32),
        "naive_idw": naive["idw"].astype(np.float32),
        "post_var": post[base_label][bw].astype(np.float32),
        "prior_var": tv.diagB.reshape(shape),
        "obs_n": np.asarray(obs_n), "skipped_ages": np.asarray(skipped, dtype=np.int64),
    }
    if analog_index:
        npz_arrays["analog_index"] = np.stack(analog_index)
    # The corrected variants ride alongside under their own keys, at the same b_scale as
    # the baseline so a paired difference over ages is what a reader gets by subtracting.
    for method in labels:
        if method == base_label:
            continue
        npz_arrays[f"recon_{method}"] = recon[method][bw].astype(np.float32)
        npz_arrays[f"post_var_{method}"] = post[method][bw].astype(np.float32)
    config = _ppe_config(
        LANE_TRAJECTORY, space, prior, len(covered), n_shapes=None, n_select=None,
        n_noise=1, b_scales=b_scales, seed=seed, reg_cols=reg_cols,
        split_meta=_chronological_split_meta(ages_i, prior_idx, truth_idx, 1),
        extra={"selected": win, "sel_tol": sel_tol, "estimator": estimator,
               "selected_b_scale_by_method": _b_scale_by_method(rows, LANE_TRAJECTORY,
                                                                sel_tol, tuple(labels)),
               "max_block_lag_steps": _max_block_lag(long, step_yr),
               "lowpass_windows": [int(w) for w in lowpass_windows],
               "bands": [[int(a), int(b)] for a, b in bands],
               "step_yr": step_yr, "min_obs": min_obs,
               "n_covered_ages": len(covered), "n_skipped_ages": len(skipped),
               "skipped_ages": skipped,
               "selection_ages": [int(covered_ages[0]), int(covered_ages[mid - 1])],
               "test_ages": [int(covered_ages[mid]), int(covered_ages[-1])]})
    _write_trajectory_artifacts(out_dir, LANE_TRAJECTORY, rows, npz_arrays, config)
    return pd.DataFrame(rows)


def _write_trajectory_artifacts(out_dir: str, lane: str, rows: list[dict],
                                npz_arrays: dict, config: dict) -> None:
    """Persist a scored trajectory lane: metrics CSV (appended), analysis npz, config.

    No skill-vs-distance npz: the lane keeps one network per age rather than a repeated
    geometry, so the distance curve the PPE lane builds has no counterpart here.
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

    Real proxies carry no D-O event label, so the event axis stays ``all`` and only the
    channel breakdown varies.
    """
    groups = [("all", "pooled", np.ones(len(channel), dtype=bool))]
    return groups + [("all", name, channel == c) for c, name in enumerate(VARS)]


@dataclass(frozen=True)
class _TargetPredictions:
    """Withheld-site predictions for one assim/target split, pooled over ages.

    ``pred`` and ``post_var`` map a method label to an ``(n_b, N)`` array, one row per
    ``b_scale``; the rest are ``(N,)`` and shared, because they describe the withheld
    observation rather than the analysis that predicts it. ``prior_var`` is the
    background variance at each target cell, which is the CRPSS reference;
    ``distance_km`` is how far the site sits from the nearest assimilated one, the axis
    that separates interpolation from extrapolation. ``rep_var`` is the per-channel
    representativeness variance carried to the target sites so the predictive spread can
    include the proxy's deviation from its cell mean, and ``resid_var`` is its temporal
    twin: what the withheld sample's own distance in time leaves unexplained. ``site``
    is the cluster a resampling scheme has to respect, since a sample repeats across
    every age in its block.
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

    Climatological background (zero anomaly), ``R = diag(sse + rep_var)`` before the
    temporal term, where ``rep_lookup`` holds the per-channel representativeness
    variance. Observations enter
    in anomaly space ``y - my`` so the proxy-vs-model offset cancels. A withheld
    observation is scored only when its own age also carries an assimilated one, so an age
    holding just one side of the split contributes nothing.

    One analysis per temporal method, each with its own R and so its own factorization.
    The withheld observation is attenuated by its own distance in time exactly as the
    assimilated ones are, so predicting it means applying that attenuation rather than
    undoing it: the prediction carries ``rho``, and ``actual`` is left alone. That factor
    is a property of the withheld sample, not of the analysis, so it applies to every
    method including the uncorrected one; scoring one attenuated and another not would
    settle the comparison by the scoring rule. The prior-free baselines keep the raw
    observations, since they predict an observation from observations and already sit on
    the attenuated scale.
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
            # The age reaches the estimator here because this lane's prior spans it, so an
            # analog step could otherwise select the simulation's own state at the target.
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
    progress_every: int | None = None,
    estimator: str = ESTIMATOR_3DVAR, method_cols: dict | None = None,
) -> tuple[str, list[dict], dict]:
    """Nested-CV site withholding for one built prior (no file writes).

    Returns ``(lane, rows, predictions)``. Two passes over the same ``k_folds`` site
    partition, tagged with a ``split`` column so the operating point is chosen on held-out
    predictions and reported on a disjoint fold. The selection pass (``split='selection'``)
    assimilates 3 folds and predicts a 4th (the val fold, rotated so every fold is val
    once), pooling those predictions across the rotation. The reporting pass
    (``split='test'``) assimilates 4 folds and predicts the 5th, per fold and pooled. A
    fold serves as a val target in one rotation and a test target in another, so for a
    single global operating point a mild dependence remains; the reported test predictions
    are never scored during selection. Observations enter in anomaly space ``y - my``;
    ``R = diag(sse + rep_var)`` before the temporal term, with the representativeness
    variance estimated per fold from the assimilated sites alone, so a withheld site never
    informs the update or the spread that scores it; the background is climatological.
    ``temporal_modes`` chooses which treatments of observation staleness to score, so a
    grid scan can spend its passes on the taper and leave the comparison between
    treatments to the winner. ``space``/``reg_cols``/``estimator``/``method_cols`` tag the
    rows.

    Alongside the skill rows each ``b_scale`` carries calibration scored against the
    proxy, whose own error and representativeness variance join the posterior spread since
    the residual holds both. The pooled test set also yields prior-free ``nearest``/``idw``
    rows, emitted once because they depend on neither the taper nor ``b_scale``, and the
    predictions dict keeps the posterior and prior variance, the proxy error and
    representativeness variance, and the distance to the nearest assimilated site so
    calibration and skill-vs-void can be rebuilt without a re-run.
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

    # Each sample's own position on the age axis, and the structure function that says
    # what sitting away from it costs. The ages are the ones the background was built
    # from, so the correction and B describe the same prior.
    long = sample_block_centres(long)
    step_yr = _age_step(np.asarray(ages, dtype=np.int64))
    S, prior_var_cell = temporal_structure_function(
        cube, prior_age_indices, max_lag=_max_block_lag(long, step_yr))

    # Estimate rep_var per fold from the assimilated sites only; the flattened cell index
    # of every row is fixed, so build it once and slice it by site set per fold.
    cell_all = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons)

    def _rep_lookup(sites) -> np.ndarray:
        rv = representativeness_variance(long, cell_all, sites=sites)
        return np.array([rv.get(v, 0.0) for v in VARS])

    def _rows(tp: _TargetPredictions, fold: int, split: str) -> list[dict]:
        """Skill and calibration rows over the b_scale sweep for every temporal method."""
        out = []
        groups = _obs_channel_groups(tp.channel)
        # The truth is itself a measurement that deviates from its grid cell and from the
        # analysis age, so its error, representativeness variance and temporal residual
        # all join the posterior spread; the prior reference carries the same terms to
        # stay comparable, and only the background half of it scales with b_scale.
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

    # Selection: assimilate 3 folds, predict the val fold; pool across the rotation.
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

    # Reporting: assimilate 4 folds, predict the test fold; per fold and pooled.
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
        # The corrected variants ride alongside under their own keys; the uncorrected one
        # keeps the bare names so anything reading the lane's predictions still finds it.
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
    temporal_modes: tuple[str, ...] = TEMPORAL_MODES,
) -> pd.DataFrame:
    """Nested-CV site withholding for one taper config.

    ``long`` must carry per-site climatology ``my``; the prior uses all ages, as the
    held-out quantity is real proxies not model states. Scoring follows
    :func:`_score_withholding_lane`; :func:`run_withholding_pixel_grid` wraps this over
    the taper grid.
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
        method_cols=method_cols, temporal_modes=temporal_modes)
    config = _withholding_config(lane, space, prior, k_folds, fold_kind, b_scales, seed,
                                 reg_cols, extra={"rep_var_full": _rep_var_full(predictions)})
    _write_withholding_artifacts(out_dir, lane, rows, predictions, config)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pixel regularizer tuning: joint (localization, shrinkage, alpha, b_scale) selection.
# ---------------------------------------------------------------------------
def select_best_config(sel_rows: pd.DataFrame, *, sel_tol: float = SEL_TOL) -> dict:
    """Winner ``{localization_km, shrinkage_lambda, alpha, b_scale}`` on the selection split.

    ``sel_rows`` is pre-filtered to one (model, lane, split=selection, channel=pooled,
    do_event=all, metric=rrmse) and carries the four config columns plus ``value``. Among
    rows within ``sel_tol`` (relative) of the minimum RRMSE, picks the most raw-like by
    ``(n_active_knobs, |log10(b_scale)|, rrmse)`` where a knob is active when localization
    is set, shrinkage > 0, or alpha < 1. This banks the coarse grid's noise headroom
    rather than chasing a spuriously-complex argmin.
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

    A plain argmin, unlike :func:`select_best_config`, whose tie-break prefers the config
    closest to a raw sample covariance. None of these axes has a "simpler" end to lean
    toward: the ensemble size and the hybrid weight bracket the static analysis, one at a
    full-pool ensemble and the other at zero weight, and the extension terms bracket the
    published estimator at zero. Rows written before an axis existed read as zero on it.
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

    Inflating R shifts the background-to-observation balance the analysis wants, so a
    variant is reported at the amplitude its own selection split prefers rather than at
    the uncorrected one's.
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
    n_shapes: int = 5, n_select: int = 4, n_noise: int = 20, truth_stride: int = 10,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Same-model PPE tuned over the taper grid: full-grid metrics, winner-only fields.

    Scores every ``(localization, shrinkage, alpha)`` config (SSIM and calibration skipped,
    the RRMSE selection needs neither), jointly selects the operating point with ``b_scale``
    on the selection split via :func:`select_best_config`, then re-runs the winner with the
    full metric set to persist its analysis fields. metrics.csv holds the full grid's skill
    rows plus the winner's SSIM and calibration; the npz holds only the winner.
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
            b_scales=b_scales, n_shapes=n_shapes, n_select=n_select, n_noise=n_noise,
            dist_edges_km=dist_edges_km, seed=seed, full_metrics=False,
            npz_extra={"truth_clim": truth_clim})
        all_rows += [r for r in rows if r["method"] == "3dvar"]   # naive added once, from winner
        _report_progress("pixel-grid config", ci + 1, len(configs), t0)

    win = select_best_config(_selection_rrmse(all_rows, LANE_PPE), sel_tol=sel_tol)
    reg_w = {"localization_km": win["localization_km"], "shrinkage_lambda": win["shrinkage_lambda"],
             "alpha": win["alpha"]}
    prior_w = build_prior(cube, ages, lats, lons, prior_idx, valid, **reg_w)
    win_rows, npz_arrays, skill = _score_ppe_lane(
        truth_anoms, prior_w, long, lats, lons,
        lane=LANE_PPE, make_method=None, space="pixel", reg_cols=reg_w,
        b_scales=b_scales, n_shapes=n_shapes, n_select=n_select, n_noise=n_noise,
        dist_edges_km=dist_edges_km, seed=seed, full_metrics=True,
        npz_extra={"truth_clim": truth_clim}, progress_every=progress_every)
    all_rows += [r for r in win_rows
                 if r["method"] == "3dvar" and r["metric"] in _FULL_METRICS]
    all_rows += [r for r in win_rows if r["method"] != "3dvar"]

    extra = {"selected": win, "sel_tol": sel_tol, **grid_record}
    config = _ppe_config(LANE_PPE, "pixel", prior_w, len(truth_anoms), n_shapes, n_select,
                         n_noise, b_scales, seed, reg_w,
                         _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride),
                         extra=extra)
    _write_ppe_artifacts(out_dir, LANE_PPE, all_rows, npz_arrays, skill, config)
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

    Scores every config's nested-CV, jointly selects the operating point with ``b_scale``
    on the selection split, then re-runs the winner to persist its predictions npz. The
    prior-free reference rows do not depend on the taper, so they are kept from the winner
    pass alone rather than replicated once per config.
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
        # The grid answers a spatial question, so it runs the uncorrected method alone and
        # the winner pass adds the temporal comparison; the prior-free rows come from
        # there too, since they depend on neither the taper nor b_scale.
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
    # The winner pass adds the temporal variants and the prior-free rows; the baseline's
    # own rows are already in from the grid pass at this same config.
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

    The lag says nothing where the tendency weight is zero, so those combinations collapse
    to one state rather than running the same estimator once per lag.
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

    The extension-term grids default to off, so a caller that names only ``k`` and
    ``hybrid_w`` gets the published estimator's grid.
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


def run_hgaoenkf_ppe_grid(
    cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    valid: np.ndarray, long: pd.DataFrame, out_dir: str, *,
    localization_km: float | None = None, shrinkage_lambda: float = 0.0, alpha: float = 1.0,
    k_grid=K_GRID, hybrid_w_grid=HYBRID_W_GRID,
    tendency_theta_grid=(0.0,), tendency_lag_yr_grid=(0.0,), redundancy_theta_grid=(0.0,),
    selection: str = ANALOG_MISFIT, evidence_scale: float = EVIDENCE_SCALE,
    analog_localization_km: float | None = None,
    b_scales: tuple[float, ...] = B_SCALES,
    n_shapes: int = 5, n_select: int = 4, n_noise: int = 20, truth_stride: int = 10,
    dist_edges_km: np.ndarray | None = None, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Same-model PPE tuned over the analog grid: full-grid metrics, winner-only fields.

    The taper is an input rather than an axis. It answers a spatial question about the
    static covariance that the pixel grid settles once, and holding it fixed is also what
    keeps the comparison against 3DVar a comparison of estimators rather than of two
    differently regularized covariances.

    ``selection`` names the analog rule and so the estimator the rows are tagged with, which
    is what lets two rules share a metrics CSV and still be compared row for row.
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
    estimator = hgaoenkf_estimator(selection)
    label = method_label(estimator)
    configs, grid_record = _analog_grid_configs(
        k_grid, hybrid_w_grid, tendency_theta_grid, tendency_lag_yr_grid,
        redundancy_theta_grid)

    def score(k, w, tendency_theta, tendency_lag_yr, redundancy_theta,
              full_metrics, progress=None):
        terms = dict(tendency_theta=tendency_theta, tendency_lag_yr=tendency_lag_yr,
                     redundancy_theta=redundancy_theta)
        return _score_ppe_lane(
            truth_anoms, prior, long, lats, lons,
            lane=LANE_PPE, space="pixel", reg_cols=reg_cols,
            make_method=make_hgaoenkf(cube, ages, lats, lons, k=k, hybrid_w=w,
                                      selection=selection, evidence_scale=evidence_scale,
                                      analog_localization_km=analog_localization_km,
                                      **terms),
            estimator=estimator, method_cols=analog_cols(k, w, **terms),
            b_scales=b_scales, n_shapes=n_shapes, n_select=n_select, n_noise=n_noise,
            dist_edges_km=dist_edges_km, seed=seed, full_metrics=full_metrics,
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
        LANE_PPE, "pixel", prior, len(truth_anoms), n_shapes, n_select, n_noise,
        b_scales, seed, reg_cols,
        _chronological_split_meta(ages_i, prior_idx, truth_idx, truth_stride),
        extra={"estimator": estimator, "selection": selection,
               "evidence_scale": float(evidence_scale),
               "analog_localization_km": analog_localization_km,
               "selected": win, **grid_record})
    _write_ppe_artifacts(out_dir, LANE_PPE, all_rows, npz_arrays, skill, config)
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
    temporal_mode: str = TEMPORAL_DEFLATE,
    report_temporal_modes: tuple[str, ...] = (TEMPORAL_DEFLATE,),
    k_folds: int = 5, fold_kind: str = "random",
    b_scales: tuple[float, ...] = B_SCALES, seed: int = 0,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Withholding lane tuned over the analog grid: full-grid metrics, winner-only predictions.

    This lane's prior spans every age including the one being reconstructed, so
    ``exclude_yr`` drops a band around the target from the analog pool. Without it the
    analog step can select the simulation's own state there, which both degenerates the
    method into a per-age background and inherits whatever alignment the proxy chronology
    was given against that simulation.

    ``temporal_mode`` is fixed rather than swept: which treatment of observation staleness
    to use is settled on the 3DVar lanes, and varying it here would confound the analog
    parameters with it. ``report_temporal_modes`` are the treatments the winner is then
    scored under, so the staleness ladder is read at one operating point rather than
    re-selecting the analog parameters per treatment. ``selection`` names the analog rule
    and so the estimator the rows are tagged with.
    """
    prior_idx = np.arange(len(ages))
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid,
                        localization_km=localization_km,
                        shrinkage_lambda=shrinkage_lambda, alpha=alpha)
    reg_cols = {"localization_km": localization_km, "shrinkage_lambda": shrinkage_lambda,
                "alpha": alpha}
    estimator = hgaoenkf_estimator(selection)
    label = method_label(estimator, temporal_mode)
    configs, grid_record = _analog_grid_configs(
        k_grid, hybrid_w_grid, tendency_theta_grid, tendency_lag_yr_grid,
        redundancy_theta_grid)

    def score(k, w, tendency_theta, tendency_lag_yr, redundancy_theta, modes,
              progress=None):
        terms = dict(tendency_theta=tendency_theta, tendency_lag_yr=tendency_lag_yr,
                     redundancy_theta=redundancy_theta)
        return _score_withholding_lane(
            prior, long, ages, lats, lons, cube=cube, prior_age_indices=prior_idx,
            space="pixel", reg_cols=reg_cols,
            make_method=make_hgaoenkf(cube, ages, lats, lons, k=k, hybrid_w=w,
                                      exclude_yr=exclude_yr, selection=selection,
                                      evidence_scale=evidence_scale,
                                      analog_localization_km=analog_localization_km,
                                      **terms),
            estimator=estimator, method_cols=analog_cols(k, w, **terms),
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
    # The winner's rows under the selection treatment are already in from the grid pass at
    # this same config; what the re-run adds is the other treatments, the prior-free
    # references and the predictions npz.
    all_rows += [r for r in win_rows if r["method"] != label]

    config = _withholding_config(
        lane, "pixel", prior, k_folds, fold_kind, b_scales, seed, reg_cols,
        extra={"estimator": estimator, "selection": selection,
               "evidence_scale": float(evidence_scale),
               "analog_localization_km": analog_localization_km, "selected": win,
               "exclude_yr": exclude_yr, "temporal_mode": temporal_mode,
               "report_temporal_modes": list(report_temporal_modes),
               "rep_var_full": _rep_var_full(predictions), **grid_record})
    _write_withholding_artifacts(out_dir, lane, all_rows, predictions, config)
    return pd.DataFrame(all_rows)


def analog_variant_estimator(rule: str, exclude_yr: float) -> str:
    """Estimator tag for one width of the exclusion band.

    The width is a modelling choice rather than a tuned parameter, so it names the
    estimator and composes into the method label the same way a treatment of staleness
    does, instead of adding a column to the shared row schema.
    """
    return f"{ESTIMATOR_HGAOENKF}_{rule}_excl{int(exclude_yr)}"


def analog_localization_estimator(rule: str, km: float | None) -> str:
    """Estimator tag for one lengthscale of the analog covariance's own taper.

    ``None`` is the lengthscale the static covariance carries, so it tags as ``static``
    rather than as a number that would differ between lanes.
    """
    suffix = "static" if km is None else f"{int(km)}"
    return f"{hgaoenkf_estimator(rule)}_loc{suffix}"


def evidence_scale_estimator(scale: float) -> str:
    """Estimator tag for one value of the evidence rule's background scale.

    A sensitivity pass, not a tuning axis: the scale is held at one value everywhere else,
    so it names the estimator rather than joining the analog grid.
    """
    return f"{hgaoenkf_estimator(ANALOG_EVIDENCE)}_c{scale:g}"


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

    ``variants`` pairs an estimator tag with the :func:`make_hgaoenkf` keywords that vary,
    so one pass can sweep the exclusion width, the analog lengthscale, or the selection
    rule without any of them joining the tuning grid. None of them is a knob to optimise:
    the exclusion width says how much of the skill comes from the analog step being allowed
    to select the simulation's own neighbourhood of the target age, and the lengthscale
    answers the ensemble-size question Sun et al. (2024) Table 2 poses.

    Rows land in their own directory under estimator-tagged method labels, so the
    head-to-head comparison is not cluttered by variants that are not candidates for it.
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
            # A variant that varies an extension term has to say so in its rows, or the
            # sweep is labelled as though the term were off.
            method_cols=analog_cols(k, hybrid_w,
                                    **{key: kw[key] for key in TERM_KEYS if key in kw}),
            temporal_modes=(temporal_mode,),
            k_folds=k_folds, fold_kind=fold_kind, b_scales=b_scales, seed=seed)
        # The prior-free references do not depend on any of this, and the grid pass has
        # already written them.
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
