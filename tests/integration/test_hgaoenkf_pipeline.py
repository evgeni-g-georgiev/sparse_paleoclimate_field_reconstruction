"""End-to-end checks on HGAOEnKF through the three evaluation lanes.

The lanes reach an estimator through the sweep surface rather than the Method ABC, so
these guard that HGAOEnKF is reachable that way, that its rows carry the composed method
label and the analog columns, and that the two grid drivers persist a winner a reader can
re-derive from the CSV.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex
from paleoreco.assim.hgaoenkf import make_hgaoenkf
from paleoreco.assim.observations import TEMPORAL_DEFLATE

SCHEMA = {"method", "space", "localization_km", "shrinkage_lambda", "alpha",
          "analog_k", "hybrid_w", "lane", "fold", "b_scale", "background", "split",
          "do_event", "channel", "metric", "value"}
K, W = 4, 0.5
B_SCALES = (0.5, 1.0)
TINY_GRID = dict(k_grid=(3, 4), hybrid_w_grid=(0.25, 0.5))


@pytest.fixture
def factory(cube, ages, lats, lons):
    return make_hgaoenkf(cube, ages, lats, lons, k=K, hybrid_w=W)


def _varying_obs(obs_long):
    """The conftest obs_long has constant y, which makes RRMSE degenerate."""
    df = obs_long.copy()
    chan = df["channel"].map({"mtco": 0, "mtwa": 1}).to_numpy()
    df["y"] = df["site"].to_numpy() * 2.0 + chan + 0.5 * np.sin(df["age"].to_numpy() / 500.0)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    return df


def test_ppe_lane_runs_and_tags_its_rows(tmp_path, cube, ages, lats, lons, valid,
                                         obs_long, factory):
    df = ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(tmp_path),
                    make_method=factory, estimator=ex.ESTIMATOR_HGAOENKF,
                    method_cols=ex.analog_cols(K, W), b_scales=B_SCALES,
                    n_shapes=3, n_select=2, n_noise=1, truth_stride=1, seed=0)

    assert SCHEMA.issubset(set(df.columns))
    hg = df[df["method"] == "hgaoenkf"]
    assert len(hg) and (hg["analog_k"] == K).all() and (hg["hybrid_w"] == W).all()
    # The prior-free references carry no ensemble, so their analog columns stay empty.
    assert df[df["method"] == "idw"]["analog_k"].isna().all()
    assert np.isfinite(hg[hg["metric"] == "rrmse"]["value"]).all()
    assert os.path.exists(tmp_path / "ppe_analysis.npz")


def test_trajectory_lane_runs_one_treatment_and_records_the_selection(
    tmp_path, cube, ages, lats, lons, valid, obs_long, factory
):
    """Fixing the treatment is what keeps the analog parameters unconfounded with it."""
    df = ex.run_trajectory(cube, ages, lats, lons, valid, obs_long, str(tmp_path),
                           make_method=factory, estimator=ex.ESTIMATOR_HGAOENKF,
                           method_cols=ex.analog_cols(K, W),
                           temporal_modes=(TEMPORAL_DEFLATE,), b_scales=B_SCALES,
                           lowpass_windows=(500,), bands=((500, 1000),), min_obs=2, seed=0)

    assert set(df["method"]) == {"hgaoenkf_temporal_deflate", "hgaoenkf_ceiling",
                                 "nearest", "idw"}
    z = np.load(tmp_path / "trajectory_analysis.npz")
    assert "analog_index" in z.files
    assert z["analog_index"].shape == (len(z["ages"]), K)
    # Every selected index has to name a real prior state.
    assert z["analog_index"].min() >= 0
    cfg = json.load(open(tmp_path / "trajectory_config.json"))
    assert cfg["estimator"] == ex.ESTIMATOR_HGAOENKF
    assert set(cfg["selected_b_scale_by_method"]) == {"hgaoenkf_temporal_deflate"}


def test_withholding_lane_runs_with_an_exclusion_band(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    factory = make_hgaoenkf(cube, ages, lats, lons, k=K, hybrid_w=W, exclude_yr=600.0)
    df = ex.run_withholding(cube, ages, lats, lons, valid, _varying_obs(obs_long),
                            str(tmp_path), make_method=factory,
                            estimator=ex.ESTIMATOR_HGAOENKF,
                            method_cols=ex.analog_cols(K, W),
                            temporal_modes=(TEMPORAL_DEFLATE,), k_folds=3,
                            b_scales=B_SCALES, seed=0)
    hg = df[df["method"] == "hgaoenkf_temporal_deflate"]
    assert len(hg) and np.isfinite(hg[hg["metric"] == "ce"]["value"]).all()
    assert os.path.exists(tmp_path / "withholding_random_predictions.npz")


def test_ppe_grid_persists_a_winner_the_csv_re_derives(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out = tmp_path / "hg"
    df = ex.run_hgaoenkf_ppe_grid(cube, ages, lats, lons, valid, obs_long, str(out),
                                  b_scales=B_SCALES, n_shapes=3, n_select=2, n_noise=1,
                                  truth_stride=1, seed=0, **TINY_GRID)

    hg = df[df["method"] == "hgaoenkf"]
    combos = hg[["analog_k", "hybrid_w"]].drop_duplicates()
    assert len(combos) == len(TINY_GRID["k_grid"]) * len(TINY_GRID["hybrid_w_grid"])
    # SSIM comes from the winner pass alone, as it does for the taper grid.
    assert len(hg[hg["metric"] == "ssim"][["analog_k", "hybrid_w"]].drop_duplicates()) == 1

    cfg = json.load(open(out / "ppe_config.json"))
    sel = ex._selection_rrmse(df.to_dict("records"), ex.LANE_PPE, "hgaoenkf")
    assert ex.select_analog_config(sel) == cfg["selected"]
    assert cfg["selected"]["analog_k"] in TINY_GRID["k_grid"]
    assert cfg["selected"]["hybrid_w"] in TINY_GRID["hybrid_w_grid"]


def test_withholding_grid_persists_a_winner_the_csv_re_derives(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out = tmp_path / "hg"
    df = ex.run_hgaoenkf_withholding_grid(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(out),
        exclude_yr=600.0, k_folds=3, b_scales=B_SCALES, seed=0, **TINY_GRID)

    label = "hgaoenkf_temporal_deflate"
    hg = df[df["method"] == label]
    assert len(hg[["analog_k", "hybrid_w"]].drop_duplicates()) == 4
    cfg = json.load(open(out / "withholding_random_config.json"))
    sel = ex._selection_rrmse(df.to_dict("records"), "withholding_random", label)
    assert ex.select_analog_config(sel) == cfg["selected"]
    assert cfg["exclude_yr"] == 600.0
    # Prior-free references come once, from the winner pass.
    naive = df[df["background"] == "none"]
    assert not naive.duplicated(subset=["method", "channel", "metric"]).any()


def test_variants_lane_labels_every_ablation_distinctly(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    df = ex.run_hgaoenkf_withholding_variants(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(tmp_path),
        k=K, hybrid_w=W, misfit_exclude=(0.0, 600.0), window_exclude=(0.0, 600.0),
        evidence_exclude=(0.0, 600.0), k_folds=3, b_scales=B_SCALES, seed=0)

    assert set(df["method"]) == {
        "hgaoenkf_misfit_excl0_temporal_deflate",
        "hgaoenkf_misfit_excl600_temporal_deflate",
        "hgaoenkf_window_excl0_temporal_deflate",
        "hgaoenkf_window_excl600_temporal_deflate",
        "hgaoenkf_evidence_excl0_temporal_deflate",
        "hgaoenkf_evidence_excl600_temporal_deflate",
    }
    # The window rule ignores the observations, so banding it must still change the result.
    def ce(method):
        sub = df[(df.method == method) & (df.metric == "ce") & (df.channel == "pooled")
                 & (df.split == "test") & (df.fold == -1)]
        return sub["value"].to_numpy()

    assert not np.allclose(ce("hgaoenkf_window_excl0_temporal_deflate"),
                           ce("hgaoenkf_window_excl600_temporal_deflate"))
    assert os.path.exists(tmp_path / "withholding_random_variants_config.json")


def test_3dvar_rows_are_unchanged_by_the_estimator_parameters(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    """The default path must still produce exactly the rows it did before.

    Every 3DVar result on disk was written by these runners, so a drift in their defaults
    would silently unpair the comparison the new estimator exists for.
    """
    kw = dict(b_scales=B_SCALES, n_shapes=3, n_select=2, n_noise=1, truth_stride=1, seed=0)
    a = ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(tmp_path / "a"), **kw)
    b = ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(tmp_path / "b"),
                   estimator=ex.ESTIMATOR_3DVAR, method_cols=None, **kw)
    assert set(a["method"]) == {"3dvar", "nearest", "idw"}
    assert a["analog_k"].isna().all()
    pd.testing.assert_frame_equal(a, b)
