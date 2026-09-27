"""Readers for the raw inputs and for the stored lane, ablation and product artefacts.

A lane's ``metrics.csv`` holds a whole grid of configurations, so most reads here are filters
that pin every configuration column to one value and leave a single row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.observations import (
    attach_site_stats,
    collapse_to_samples,
    load_observations,
    observation_site_stats,
)
from paleoreco.data import build_prior_cube

TAPER_COLS = ("localization_km", "shrinkage_lambda", "alpha")
POOLED_FOLD = -1


@dataclass
class Inputs:
    """The prior archive, the proxy tables and the stored product, loaded once."""

    cube: np.ndarray
    ages: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    valid: np.ndarray
    long: pd.DataFrame        # replicated over dating blocks, with each site's climatology
    raw: pd.DataFrame         # the observation file as published
    recon: dict
    recon_cfg: dict


def load_inputs() -> Inputs:
    """Read the prior cube, both proxy tables and the product."""
    p = build_prior_cube(prior_csv=str(paths.PRIOR_CSV), cache_path=str(paths.PRIOR_CACHE))
    long = load_observations(str(paths.OBSERVATION_CSV))
    long = attach_site_stats(long, observation_site_stats(collapse_to_samples(long)))
    d = paths.product_dir()
    return Inputs(cube=p["cube"], ages=p["ages"], lats=p["lats"], lons=p["lons"],
                  valid=p["valid"], long=long,
                  raw=pd.read_csv(str(paths.OBSERVATION_CSV)),
                  recon=np.load(d / "reconstruction_fields.npz"),
                  recon_cfg=json.load(open(d / "reconstruction_config.json")))


def run_artifact(estimator: str, lane: str, name: str):
    """One stored file of one (estimator, lane): a dict for ``.json``, an npz otherwise."""
    path = paths.run_dir(estimator, lane) / name
    return json.load(open(path)) if name.endswith(".json") else np.load(path)


def config(estimator: str, lane: str) -> dict:
    """The stored config of one (estimator, lane)."""
    name = {paths.LANE_PPE: "ppe_config.json",
            paths.LANE_TRAJECTORY: "trajectory_config.json",
            paths.LANE_WITHHOLDING: "withholding_random_config.json"}[lane]
    return run_artifact(estimator, lane, name)


def inherited_taper(estimator: str, lane: str) -> dict:
    """The covariance taper one lane ran under, keyed as the prior builder expects."""
    selected = config(estimator, lane)["selected"]
    return {k: selected[k] for k in TAPER_COLS}


def metrics(estimator: str, lane: str) -> pd.DataFrame:
    """The metrics table of one (estimator, lane)."""
    return pd.read_csv(paths.run_dir(estimator, lane) / "metrics.csv")


def ablation_metrics(name: str) -> pd.DataFrame:
    """The metrics table of one ablation."""
    return pd.read_csv(paths.ablation_dir(name) / "metrics.csv")


def pooled(M: pd.DataFrame) -> pd.DataFrame:
    """Rows scored over both channels, every event and, where folds exist, all folds."""
    return M[(M.channel == "pooled") & (M.do_event == "all") & (M.fold == POOLED_FOLD)]


def pick(M: pd.DataFrame, **cols) -> pd.DataFrame:
    """Rows where each named column equals its value; ``None`` matches a missing entry."""
    m = np.ones(len(M), dtype=bool)
    for key, value in cols.items():
        if value is None:
            m &= M[key].isna().to_numpy()
        elif isinstance(value, str):
            m &= (M[key] == value).to_numpy()
        else:
            m &= np.isclose(M[key], value)
    return M[m]


def value(M: pd.DataFrame, metric: str, **cols) -> float:
    """The single value of ``metric`` among the rows matching ``cols``."""
    rows = pick(M, metric=metric, **cols)
    if len(rows) != 1:
        raise ValueError(f"{len(rows)} rows match {metric} {cols}, expected one")
    return float(rows["value"].iloc[0])


def selected_by_amplitude(M: pd.DataFrame, **cols) -> tuple[float, float]:
    """``(c, selection rRMSE)`` at the amplitude the selection split scores lowest."""
    sel = pick(M, split="selection", metric="rrmse", **cols)
    best = sel.loc[sel["value"].idxmin()]
    return float(best["b_scale"]), float(best["value"])


def operating_point(estimator: str, lane: str, split: str = "test") -> pd.Series:
    """Pooled metrics of one (estimator, lane) at the configuration its config selected.

    The snapshot lane scores the analysis as run; the two lanes whose observations are displaced
    in time score it with the temporal term deflated, the treatment every estimator carries.
    """
    cfg = config(estimator, lane)
    method = (estimator if lane == paths.LANE_PPE
              else ex.method_label(estimator, ex.TEMPORAL_DEFLATE))
    M = pick(pooled(metrics(estimator, lane)), method=method, split=split)
    return M[at_selected(M, cfg, cfg["selected"]["b_scale"])].set_index("metric")["value"]


def lane_rows(estimator, lane, file_lane, *, method, split="test", **fixed):
    """Pooled, all-event metric rows of one lane, filtered to one method and split."""
    M = pd.read_csv(paths.run_dir(estimator, lane) / "metrics.csv")
    m = ((M.method == method) & (M.split == split) & (M.channel == "pooled")
         & (M.do_event == "all") & (M.lane == file_lane))
    for key, v in fixed.items():
        m &= np.isclose(M[key], v)
    return M[m]


def at_selected(M, cfg, b_scale):
    """Mask of the rows a lane's config records as its selected configuration.

    The taper is read from ``selected`` where the lane searched it and from the config's own
    columns where it was inherited. A key absent from ``selected`` was never varied.
    """
    sel = cfg["selected"]
    m = np.isclose(M["b_scale"], b_scale)
    for key in TAPER_COLS:
        v = sel.get(key, cfg.get(key))
        m &= M[key].isna() if v is None else np.isclose(M[key], v)
    for key in ("analog_k", "hybrid_w") + ex.TERM_KEYS:
        if key in sel:
            m &= np.isclose(M[key], sel[key])
    return m
