"""Grid-tuning drivers: full-grid metrics, winner-only fields, and the selection rule.

``run_ppe_pixel_grid`` / ``run_withholding_pixel_grid`` score every (localization,
shrinkage, alpha) config, jointly select the operating point with ``b_scale`` on the
selection split, and persist only the winner's fields. These guard the full-grid metrics
schema, the winner-only artifacts, and that notebook 09 re-derives the same winner via
``select_best_config`` on the persisted CSV.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from paleoreco.assim import experiments as ex

TINY_GRID = dict(localization_grid=(None,), shrinkage_grid=(0.0, 0.5), alpha_grid=(1.0, 0.0))


def _n_configs(grid):
    return (len(grid["localization_grid"]) * len(grid["shrinkage_grid"])
            * len(grid["alpha_grid"]))


def _selection_rows(tdv, lane):
    return tdv[(tdv["lane"] == lane) & (tdv["split"] == "selection")
              & (tdv["channel"] == "pooled") & (tdv["do_event"] == "all")
              & (tdv["metric"] == "rrmse")]


def test_run_ppe_pixel_grid_full_grid_and_winner(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out = tmp_path / "pixel"
    df = ex.run_ppe_pixel_grid(
        cube, ages, lats, lons, valid, obs_long, str(out),
        b_scales=(0.5, 1.0), n_shapes=3, n_select=2, n_noise=1, truth_stride=1, seed=0,
        **TINY_GRID,
    )
    tdv = df[df["method"] == "3dvar"]

    # New schema, full grid in metrics.csv.
    assert {"localization_km", "shrinkage_lambda", "alpha"}.issubset(df.columns)
    assert "B_reg" not in df.columns
    combos = tdv[["localization_km", "shrinkage_lambda", "alpha"]].drop_duplicates()
    assert len(combos) == _n_configs(TINY_GRID)

    # Winner-only field npz; config records a selected config drawn from the grid.
    assert os.path.exists(out / "ppe_analysis.npz")
    cfg = json.load(open(out / "ppe_config.json"))
    win = cfg["selected"]
    assert win["localization_km"] in TINY_GRID["localization_grid"]
    assert win["shrinkage_lambda"] in TINY_GRID["shrinkage_grid"]
    assert win["alpha"] in TINY_GRID["alpha_grid"]

    # SSIM is reported for the winning config only (the RRMSE scan skips it).
    ssim_combos = (tdv[tdv["metric"] == "ssim"][["localization_km", "shrinkage_lambda", "alpha"]]
                   .drop_duplicates())
    assert len(ssim_combos) == 1

    # Naive baselines are written once, not per config.
    assert len(df[df["method"].isin(["idw", "nearest"])]) > 0

    # nb09 re-derives the same winner from the persisted metrics.
    assert ex.select_best_config(_selection_rows(tdv, "ppe")) == win


def _varying_obs(obs_long):
    """The conftest obs_long has constant y (degenerate RRMSE); vary it per site/channel/age."""
    df = obs_long.copy()
    chan = df["channel"].map({"mtco": 0, "mtwa": 1}).to_numpy()
    df["y"] = df["site"].to_numpy() * 2.0 + chan + 0.5 * np.sin(df["age"].to_numpy() / 500.0)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    return df


def test_run_withholding_pixel_grid_full_grid_and_winner(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out = tmp_path / "pixel"
    df = ex.run_withholding_pixel_grid(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(out),
        k_folds=3, fold_kind="random", b_scales=(0.5, 1.0), seed=0, **TINY_GRID,
    )
    tdv = df[df["method"] == "3dvar"]
    combos = tdv[["localization_km", "shrinkage_lambda", "alpha"]].drop_duplicates()
    assert len(combos) == _n_configs(TINY_GRID)
    assert os.path.exists(out / "withholding_random_predictions.npz")

    # The prior-free references do not depend on the taper, so the winner pass emits them
    # once; one row per (method, channel, metric) rather than one per grid config.
    naive = df[df["method"].isin(["nearest", "idw"])]
    assert not naive.empty
    assert not naive.duplicated(subset=["method", "channel", "metric"]).any()

    cfg = json.load(open(out / "withholding_random_config.json"))
    assert ex.select_best_config(_selection_rows(tdv, "withholding_random")) == cfg["selected"]


def test_select_best_config_prefers_simpler_within_tolerance():
    # Near-tie: the raw-like config (all off, b=1) is within tol of the complex best.
    rows = pd.DataFrame([
        {"localization_km": np.nan, "shrinkage_lambda": 0.0, "alpha": 1.0, "b_scale": 1.0, "value": 0.501},
        {"localization_km": 12500.0, "shrinkage_lambda": 0.5, "alpha": 0.0, "b_scale": 5.0, "value": 0.500},
    ])
    assert ex.select_best_config(rows, sel_tol=0.01) == {
        "localization_km": None, "shrinkage_lambda": 0.0, "alpha": 1.0, "b_scale": 1.0}

    # Clear winner outside tolerance is taken despite being complex.
    rows2 = pd.DataFrame([
        {"localization_km": np.nan, "shrinkage_lambda": 0.0, "alpha": 1.0, "b_scale": 1.0, "value": 0.60},
        {"localization_km": 12500.0, "shrinkage_lambda": 0.5, "alpha": 0.0, "b_scale": 5.0, "value": 0.50},
    ])
    win = ex.select_best_config(rows2, sel_tol=0.01)
    assert win["shrinkage_lambda"] == 0.5 and win["alpha"] == 0.0
