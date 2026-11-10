"""Readers for the raw inputs and for the stored lane, grid, ablation and product artefacts.

A run's ``metrics.csv`` holds one configuration over the amplitude sweep and a grid's holds a
whole scan, so most reads here are filters that pin columns to values and leave one row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
    """Read the prior cube, both proxy tables and the product.

    The cube is cut to the ages the product was run on, which is every age of a full run.
    """
    p = build_prior_cube(prior_csv=str(paths.PRIOR_CSV), cache_path=str(paths.PRIOR_CACHE))
    long = load_observations(str(paths.OBSERVATION_CSV))
    long = attach_site_stats(long, observation_site_stats(collapse_to_samples(long)))
    recon = np.load(paths.PRODUCT / "reconstruction_fields.npz")
    keep = np.isin(p["ages"], recon["ages"])
    return Inputs(cube=p["cube"][keep], ages=p["ages"][keep], lats=p["lats"], lons=p["lons"],
                  valid=p["valid"], long=long, raw=pd.read_csv(str(paths.OBSERVATION_CSV)),
                  recon=recon,
                  recon_cfg=json.load(open(paths.PRODUCT / "reconstruction_config.json")))


def run_artifact(estimator: str, lane: str, name: str):
    """One stored file of one (estimator, lane): a dict for ``.json``, an npz otherwise."""
    path = paths.run_dir(estimator, lane) / name
    return json.load(open(path)) if name.endswith(".json") else np.load(path)


def config(estimator: str, lane: str) -> dict:
    """The stored config of one (estimator, lane)."""
    return run_artifact(estimator, lane, f"{lane}_config.json")


def point(estimator: str) -> ex.AnalogPoint | None:
    """The analog settings an estimator ran at on the snapshot lane, ``None`` for 3DVar."""
    return ex.point_from_record(config(estimator, paths.LANE_PPE)["point"])


def metrics(estimator: str, lane: str) -> pd.DataFrame:
    """The metrics table of one (estimator, lane)."""
    return ex.read_metrics(paths.run_dir(estimator, lane))


def grid_metrics(name: str) -> pd.DataFrame:
    """The metrics table of one tuning grid."""
    return ex.read_metrics(paths.grid_dir(name))


def grid_config(name: str) -> dict:
    """The axes and settings one tuning grid scanned."""
    return json.load(open(paths.grid_dir(name) / ex.GRID_CONFIG))


def ablation_metrics(name: str | Path) -> pd.DataFrame:
    """The metrics table of one ablation."""
    return ex.read_metrics(paths.ablation_dir(str(name)))


def pooled(M: pd.DataFrame) -> pd.DataFrame:
    """Rows scored over both channels."""
    return M[M.channel == "pooled"]


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


def selected(M: pd.DataFrame, **cols) -> pd.Series:
    """The pooled selection row with the lowest rRMSE among the rows matching ``cols``."""
    return ex.select(pick(pooled(M), **cols))


def test_at(M: pd.DataFrame, row: pd.Series) -> pd.Series:
    """Pooled test metrics at the configuration and amplitude of a selection row."""
    keys = [key for key in ("method", *ex.TAPER_COLS, *ex.ANALOG_COLS, "b_scale")
            if key in M.columns]
    cols = {key: (None if pd.isna(row[key]) else row[key]) for key in keys}
    return pick(pooled(M), split="test", **cols).set_index("metric")["value"]


def operating_point(estimator: str, lane: str, split: str = "test") -> pd.Series:
    """Pooled metrics of one (estimator, lane) at the amplitude its config selected."""
    M = pick(pooled(metrics(estimator, lane)), method=estimator, split=split,
             b_scale=config(estimator, lane)["b_scale"])
    return M.set_index("metric")["value"]


def sweep(M: pd.DataFrame, metric: str, **cols) -> pd.Series:
    """One pooled metric against ``c`` among the rows matching ``cols``."""
    return pick(pooled(M), metric=metric, **cols).set_index("b_scale")["value"].sort_index()
