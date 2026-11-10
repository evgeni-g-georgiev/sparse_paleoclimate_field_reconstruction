"""Tuning grids, the estimators they define, and the drivers that write each lane's artefacts.

Selection everywhere is the argmin of the pooled selection-split rRMSE. Only the snapshot
lane tunes anything beyond the time-series amplitude:

* 3DVar: the covariance taper jointly with the amplitude ``c``.
* At that taper, one analog grid over the evidence scale ``kappa``, ensemble size ``k``,
  hybrid weight ``alpha`` and ``c``. Its ``kappa = 0`` slice is HGAOEnKF, with the misfit
  rule; its best point over ``kappa > 0`` is evidence-only HGAOEnKF.
* MTA-HGAOEnKF: at evidence-only's ``kappa``, a grid over the tendency share ``w``, ``k``,
  ``alpha`` and ``c`` with the fixed tendency stack :data:`MTA_STACK`.

The time-series lane inherits each estimator's snapshot taper and analog parameters and
selects ``c`` on its own selection arm. The withholding lane inherits them too and selects
nothing: it reports every ``c``.

A run directory holds one estimator on one lane at its operating point: ``metrics.csv``
(the amplitude sweep's skill rows, then the full metrics at the selected amplitude), the
fields or predictions, and ``<lane>_config.json``. A grid directory holds a scan's
``metrics.csv`` and ``grid_config.json``.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paleoreco import paths
from paleoreco.data import VARS
from paleoreco.assim.analog import (
    ANALOG_CORRELATION,
    ANALOG_CORRELATION_PERCHAN,
    ANALOG_EVIDENCE,
)
from paleoreco.assim.hgaoenkf import TendencyStack, make_hgaoenkf
from paleoreco.assim.lanes import (
    report_progress,
    scan_pseudo_proxy,
    score_snapshot,
    score_trajectory,
    score_withholding,
    snapshot_lane,
    trajectory_lane,
    withholding_lane,
)
from paleoreco.assim.method import Method, MethodFactory
from paleoreco.assim.observations import FULL_MODEL, ObservationModel
from paleoreco.assim.priors import Prior, build_prior
from paleoreco.assim.threedvar import ThreeDVar

# Background amplitude c, every lane.
B_SCALES = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)
# The amplitudes the product is published at, side by side.
PRODUCT_B_SCALES = (0.3, 1.0, 2.0, 5.0, 10.0)
# The covariance taper: Gaspari-Cohn lengthscale (``None`` is no localization), shrinkage
# toward the diagonal, and channel coupling. The lengthscales are chordal, so they span
# what the taper does across a 12742 km diameter: 5000 reaches zero inside the sphere,
# 12500 leaves 0.19 between antipodes, and 35000 leaves 0.82.
TAPER_GRID = dict(
    localization_grid=(None, 5000.0, 8000.0, 12500.0, 20000.0, 25000.0, 35000.0),
    shrinkage_grid=(0.0, 0.25, 0.5, 0.75),
    alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
)
# The analog axes. The hybrid weight spans Sun et al.'s AOEnKF-B (0) to AOEnKF (1).
K_GRID = (40, 60, 100, 150, 200)
# MTA's ensemble sizes run to 400, just under the snapshot lane's 402-state pool.
MTA_K_GRID = K_GRID + (300, 400)
HYBRID_W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
KAPPA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TENDENCY_SHARE_GRID = (0.25, 0.5, 0.75, 0.9, 0.97, 1.0)
# The evidence scale of the ablation's fixed-kappa row.
ABLATION_KAPPA = 1.0

# Tendency stacks: first differences, and second differences, at fixed lags. 400 yr is the
# reference every block's amplitude is normalised to.
_SEVEN_LAGS = (200.0, 400.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0)
MTA_STACKS = {
    "1lag": TendencyStack(reference=400.0, first=(400.0,)),
    "3lag": TendencyStack(reference=400.0, first=(400.0, 800.0, 1600.0)),
    "7lag": TendencyStack(reference=400.0, first=_SEVEN_LAGS),
    "7lag_curv": TendencyStack(reference=400.0, first=_SEVEN_LAGS, second=(400.0, 1600.0)),
}
MTA_STACK = MTA_STACKS["7lag_curv"]

ESTIMATOR_3DVAR = "3dvar"
ESTIMATOR_HGAOENKF = "hgaoenkf"
ESTIMATOR_EVIDENCE = "hgaoenkf_evidence"
ESTIMATOR_MTA = "hgaoenkf_mt"
CORRELATION_RULES = {ANALOG_CORRELATION: "hgaoenkf_correlation",
                     ANALOG_CORRELATION_PERCHAN: "hgaoenkf_correlation_perchan"}

# Grid and ablation directory names.
GRID_TAPER = "3dvar_taper"
GRID_EVIDENCE = "evidence"
GRID_MTA = "mta"
ABLATION_CORRELATION = "correlation_rules"
ABLATION_LADDER = "mta_ladder"
ABLATION_OBS_MODEL = "obs_model"
# The 3DVar withholding arms beyond the full model, which the withholding run itself is.
OBS_MODEL_ARMS = (ObservationModel(spatial=False, temporal=False),
                  ObservationModel(spatial=True, temporal=False),
                  ObservationModel(spatial=False, temporal=True))
# The stacks the ablation scans over ``w``; the full stack's scan is the MTA grid.
LADDER_SCANNED = ("1lag", "3lag", "7lag")

TAPER_COLS = ("localization_km", "shrinkage_lambda", "alpha")
ANALOG_COLS = ("analog_k", "hybrid_w", "kappa", "tendency_share")
METRICS_CSV = "metrics.csv"
GRID_CONFIG = "grid_config.json"


def ladder_estimator(name: str) -> str:
    """Estimator tag for MTA with one stack of :data:`MTA_STACKS`."""
    return f"{ESTIMATOR_MTA}_{name}"


# ---------------------------------------------------------------------------
# Operating points.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalogPoint:
    """One analog estimator's settings: what the grids tune and what the other lanes inherit."""

    k: int
    hybrid_w: float
    kappa: float = 0.0
    tendency_share: float = 0.0
    selection: str = ANALOG_EVIDENCE
    stack: TendencyStack | None = None

    def factory(self, cube, ages, lats, lons) -> MethodFactory:
        return make_hgaoenkf(cube, ages, lats, lons, k=self.k, hybrid_w=self.hybrid_w,
                             selection=self.selection, kappa=self.kappa, stack=self.stack,
                             tendency_share=self.tendency_share)

    def columns(self) -> dict:
        return {"analog_k": float(self.k), "hybrid_w": float(self.hybrid_w),
                "kappa": float(self.kappa), "tendency_share": float(self.tendency_share)}

    def record(self) -> dict:
        """JSON-ready, and read back by :func:`point_from_record`."""
        return {"k": int(self.k), "hybrid_w": float(self.hybrid_w), "kappa": float(self.kappa),
                "tendency_share": float(self.tendency_share), "selection": self.selection,
                "stack": None if self.stack is None else asdict(self.stack)}


def point_from_record(record: dict | None) -> AnalogPoint | None:
    """The :class:`AnalogPoint` a config recorded, or ``None`` for 3DVar."""
    if record is None:
        return None
    stack = record["stack"]
    return AnalogPoint(k=int(record["k"]), hybrid_w=float(record["hybrid_w"]),
                       kappa=float(record["kappa"]),
                       tendency_share=float(record["tendency_share"]),
                       selection=record["selection"],
                       stack=None if stack is None else TendencyStack(**stack))


def _columns(taper: dict, point: AnalogPoint | None) -> dict:
    """A metrics row's configuration columns, with the analog ones NaN for 3DVar."""
    analog = point.columns() if point else {key: np.nan for key in ANALOG_COLS}
    return {**{key: taper[key] for key in TAPER_COLS}, **analog}


def _method(prior: Prior, cube, ages, lats, lons, point: AnalogPoint | None) -> Method:
    """3DVar over the prior's B, or the analog estimator ``point`` describes."""
    shape = (len(VARS), len(lats), len(lons))
    if point is None:
        return ThreeDVar(prior.B, shape)
    return point.factory(cube, ages, lats, lons)(prior, shape)


# ---------------------------------------------------------------------------
# Selection.
# ---------------------------------------------------------------------------
def selection_rrmse(rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled selection-split rRMSE rows, the surface every selection reads."""
    return rows[(rows["split"] == "selection") & (rows["channel"] == "pooled")
                & (rows["metric"] == "rrmse")]


def select(rows: pd.DataFrame) -> pd.Series:
    """The selection row with the lowest rRMSE among ``rows``."""
    sel = selection_rrmse(rows).dropna(subset=["value"])
    if sel.empty:
        raise ValueError("no finite selection-split rRMSE to select from")
    return sel.loc[sel["value"].idxmin()]


def taper_from_row(row: pd.Series) -> dict:
    """The taper a metrics row was scored under, keyed as :func:`build_prior` expects."""
    loc = row["localization_km"]
    return {"localization_km": None if pd.isna(loc) else float(loc),
            "shrinkage_lambda": float(row["shrinkage_lambda"]), "alpha": float(row["alpha"])}


def point_from_row(row: pd.Series, *, selection: str = ANALOG_EVIDENCE,
                   stack: TendencyStack | None = None) -> AnalogPoint:
    """The :class:`AnalogPoint` a metrics row was scored at, given what no grid tunes."""
    return AnalogPoint(k=int(row["analog_k"]), hybrid_w=float(row["hybrid_w"]),
                       kappa=float(row["kappa"]), tendency_share=float(row["tendency_share"]),
                       selection=selection, stack=stack)


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------
def _json_default(value):
    """``json.dump`` fallback for numpy scalars and arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, record: dict) -> None:
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=_json_default)


def read_metrics(directory: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / METRICS_CSV)


def _write_run(out_dir: str | Path, lane: str, rows: list[dict], config: dict,
               arrays: dict[str, dict]) -> None:
    """``metrics.csv``, ``<lane>_config.json`` and one npz per entry of ``arrays``."""
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out_dir) / METRICS_CSV, index=False)
    _write_json(Path(out_dir) / f"{lane}_config.json", config)
    for name, content in arrays.items():
        np.savez_compressed(Path(out_dir) / f"{lane}_{name}.npz", **content)


def _lane_meta(meta: dict) -> dict:
    """A lane's provenance for a config, without its per-truth arrays."""
    return {key: value for key, value in meta.items() if not isinstance(value, np.ndarray)}


def _record(estimator: str, taper: dict, point: AnalogPoint | None, b_scales,
            prior: Prior) -> dict:
    return {"estimator": estimator, "taper": dict(taper),
            "point": None if point is None else point.record(),
            "b_scales": [float(b) for b in b_scales], "prior_meta": prior.meta}


# ---------------------------------------------------------------------------
# Grid scans, snapshot lane.
# ---------------------------------------------------------------------------
def run_taper_grid(cube, ages, lats, lons, valid, long, out_dir, *,
                   localization_grid, shrinkage_grid, alpha_grid,
                   b_scales=B_SCALES) -> pd.DataFrame:
    """3DVar over every taper and amplitude, mean-only, written as one grid."""
    lane = snapshot_lane(cube, ages, lats, lons, valid, long)
    tapers = [dict(localization_km=loc, shrinkage_lambda=lam, alpha=a)
              for loc, lam, a in itertools.product(localization_grid, shrinkage_grid,
                                                   alpha_grid)]
    rows: list[dict] = []
    t0 = time.time()
    for ci, taper in enumerate(tapers):
        prior = build_prior(cube, ages, lats, lons, lane.prior_idx, valid, **taper)
        rows += scan_pseudo_proxy(lane, _method(prior, cube, ages, lats, lons, None), b_scales,
                                  {"method": ESTIMATOR_3DVAR, **_columns(taper, None)})
        report_progress("taper", ci + 1, len(tapers), t0)
    record = {"lane": paths.LANE_PPE, "estimator": ESTIMATOR_3DVAR,
              "localization_grid": list(localization_grid),
              "shrinkage_grid": list(shrinkage_grid), "alpha_grid": list(alpha_grid),
              "b_scales": [float(b) for b in b_scales], **_lane_meta(lane.meta)}
    return _write_grid(out_dir, rows, record)


def run_analog_grid(cube, ages, lats, lons, valid, long, out_dir, *, taper: dict,
                    points, b_scales=B_SCALES) -> pd.DataFrame:
    """Every ``(tag, AnalogPoint)`` of ``points`` at every amplitude, mean-only, as one grid."""
    lane = snapshot_lane(cube, ages, lats, lons, valid, long)
    prior = build_prior(cube, ages, lats, lons, lane.prior_idx, valid, **taper)
    rows: list[dict] = []
    t0 = time.time()
    for ci, (tag, point) in enumerate(points):
        rows += scan_pseudo_proxy(lane, _method(prior, cube, ages, lats, lons, point),
                                  b_scales, {"method": tag, **_columns(taper, point)})
        report_progress("analog point", ci + 1, len(points), t0)
    record = {"lane": paths.LANE_PPE, "taper": dict(taper),
              "points": [{"method": tag, **point.record()} for tag, point in points],
              "b_scales": [float(b) for b in b_scales], **_lane_meta(lane.meta)}
    return _write_grid(out_dir, rows, record)


def _write_grid(out_dir, rows: list[dict], record: dict) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / METRICS_CSV, index=False)
    _write_json(Path(out_dir) / GRID_CONFIG, record)
    return df


# ---------------------------------------------------------------------------
# One estimator on one lane at its operating point.
# ---------------------------------------------------------------------------
def run_snapshot(cube, ages, lats, lons, valid, long, out_dir, *, estimator: str,
                 taper: dict, point: AnalogPoint | None = None,
                 b_scales=B_SCALES) -> dict:
    """The snapshot lane at one configuration: the amplitude sweep, then the selected ``c`` in full.

    The sweep repeats the grid's own scan of this configuration, so the amplitude it selects
    is the one the grid's argmin chose.
    """
    lane = snapshot_lane(cube, ages, lats, lons, valid, long)
    prior = build_prior(cube, ages, lats, lons, lane.prior_idx, valid, **taper)
    method = _method(prior, cube, ages, lats, lons, point)
    base = {"method": estimator, **_columns(taper, point)}
    rows = scan_pseudo_proxy(lane, method, b_scales, base)
    best = select(pd.DataFrame(rows))
    full, fields, skill = score_snapshot(lane, method, float(best["b_scale"]), base,
                                         progress="snapshot truth")
    config = {**_record(estimator, taper, point, b_scales, prior), "lane": paths.LANE_PPE,
              "observation_model": asdict(ObservationModel(spatial=False, temporal=False)),
              "b_scale": float(best["b_scale"]), "selection_rrmse": float(best["value"]),
              **_lane_meta(lane.meta)}
    _write_run(out_dir, paths.LANE_PPE, rows + full, config,
               {"analysis": fields, "skill_vs_distance": skill})
    return config


def run_trajectory(cube, ages, lats, lons, valid, long, out_dir, *, estimator: str,
                   taper: dict, point: AnalogPoint | None = None,
                   b_scales=B_SCALES) -> dict:
    """The time-series lane: ``c`` selected on the borrowed-network arm, then reported in full."""
    lane = trajectory_lane(cube, ages, lats, lons, valid, long)
    prior = build_prior(cube, ages, lats, lons, lane.prior_idx, valid, **taper)
    method = _method(prior, cube, ages, lats, lons, point)
    base = {"method": estimator, **_columns(taper, point)}
    rows = scan_pseudo_proxy(lane, method, b_scales, base, progress="time-series age")
    best = select(pd.DataFrame(rows))
    full, fields = score_trajectory(lane, method, float(best["b_scale"]), base,
                                    progress="time-series age")
    config = {**_record(estimator, taper, point, b_scales, prior),
              "lane": paths.LANE_TRAJECTORY,
              "observation_model": asdict(ObservationModel(spatial=False, temporal=True)),
              "b_scale": float(best["b_scale"]), "selection_rrmse": float(best["value"]),
              **_lane_meta(lane.meta),
              "scored_ages": [int(lane.meta["ages"][0]), int(lane.meta["ages"][-1])]}
    _write_run(out_dir, paths.LANE_TRAJECTORY, rows + full, config, {"analysis": fields})
    return config


def run_withholding(cube, ages, lats, lons, valid, long, out_dir, *, estimator: str,
                    taper: dict, point: AnalogPoint | None = None,
                    model: ObservationModel = FULL_MODEL, b_scales=B_SCALES) -> dict:
    """The withholding lane at every amplitude, under one observation model.

    ``long`` must carry the per-site climatology ``my``. Nothing is selected here.
    """
    lane = withholding_lane(cube, ages, lats, lons, valid, long)
    prior = build_prior(cube, ages, lats, lons, np.arange(len(ages)), valid, **taper)
    method = _method(prior, cube, ages, lats, lons, point)
    rows, predictions = score_withholding(lane, method, b_scales,
                                          {"method": estimator, **_columns(taper, point)},
                                          model, progress="withholding batch")
    config = {**_record(estimator, taper, point, b_scales, prior),
              "lane": paths.LANE_WITHHOLDING, "observation_model": asdict(model),
              **lane.meta}
    _write_run(out_dir, paths.LANE_WITHHOLDING, rows, config, {"predictions": predictions})
    return config
