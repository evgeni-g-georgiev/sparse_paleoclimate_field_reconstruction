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
from paleoreco.assim.analog import ANALOG_MISFIT
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


def test_trajectory_lane_runs_the_whole_staleness_ladder(
    tmp_path, cube, ages, lats, lons, valid, obs_long, factory
):
    """The ladder is a property of the observation layer, so one estimator carries it all.

    ``recon_realistic`` holds whichever treatment came first in the tuple, so the deflated
    run has to be reachable by name; reading the alias would silently return the
    uncorrected one.
    """
    df = ex.run_trajectory(cube, ages, lats, lons, valid, obs_long, str(tmp_path),
                           make_method=factory, estimator=ex.ESTIMATOR_HGAOENKF,
                           method_cols=ex.analog_cols(K, W),
                           temporal_modes=ex.TEMPORAL_MODES, b_scales=B_SCALES,
                           lowpass_windows=(500,), bands=((500, 1000),), min_obs=2, seed=0)

    assert set(df["method"]) == {"hgaoenkf", "hgaoenkf_temporal_add",
                                 "hgaoenkf_temporal_deflate", "hgaoenkf_ceiling",
                                 "nearest", "idw"}
    z = np.load(tmp_path / "trajectory_analysis.npz")
    for key in ("recon_realistic", "recon_ceiling", "recon_hgaoenkf_temporal_add",
                "recon_hgaoenkf_temporal_deflate", "analog_index"):
        assert key in z.files
    # The alias is the uncorrected run, so it must not be the deflated one.
    assert not np.allclose(z["recon_realistic"], z["recon_hgaoenkf_temporal_deflate"])


def test_trajectory_naive_rows_do_not_depend_on_the_estimator(
    tmp_path, cube, ages, lats, lons, valid, obs_long, factory
):
    """Prior-free references are built from the observations, so every run must agree.

    They come from whichever treatment leads the tuple, so a run that led with a corrected
    treatment would interpolate corrected observations and quietly disagree with a run that
    did not. Notebook 09 pools directories and drops duplicates, which would hide it.
    """
    common = dict(b_scales=B_SCALES, lowpass_windows=(500,), bands=((500, 1000),),
                  min_obs=2, seed=0, temporal_modes=ex.TEMPORAL_MODES)
    base = ex.run_trajectory(cube, ages, lats, lons, valid, obs_long,
                             str(tmp_path / "3dvar"), **common)
    analog = ex.run_trajectory(cube, ages, lats, lons, valid, obs_long,
                               str(tmp_path / "hg"), make_method=factory,
                               estimator=ex.ESTIMATOR_HGAOENKF,
                               method_cols=ex.analog_cols(K, W), **common)

    def naive(df):
        sub = df[df["background"] == "none"]
        return sub.set_index(["method", "channel", "metric", "split"])["value"].sort_index()

    pd.testing.assert_series_equal(naive(base), naive(analog))


def test_withholding_grid_reports_the_ladder_at_the_selected_point(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    """The grid selects under one treatment; the winner is then scored under all of them.

    Selecting per treatment would confound the analog parameters with the observation
    model, so the ladder has to be a re-scoring of one operating point rather than three.
    """
    out = tmp_path / "hg"
    df = ex.run_hgaoenkf_withholding_grid(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(out),
        exclude_yr=600.0, k_folds=3, b_scales=B_SCALES, seed=0,
        report_temporal_modes=ex.TEMPORAL_MODES, **TINY_GRID)

    label = "hgaoenkf_temporal_deflate"
    assert {"hgaoenkf", "hgaoenkf_temporal_add", label} <= set(df["method"])
    # Only the treatment the grid ran carries the whole grid; the others come from the
    # winner pass alone and so sit at one config.
    assert len(df[df.method == label][["analog_k", "hybrid_w"]].drop_duplicates()) == 4
    for other in ("hgaoenkf", "hgaoenkf_temporal_add"):
        cfg = json.load(open(out / "withholding_random_config.json"))
        sub = df[df.method == other][["analog_k", "hybrid_w"]].drop_duplicates()
        assert len(sub) == 1
        assert sub.iloc[0]["analog_k"] == cfg["selected"]["analog_k"]
        assert sub.iloc[0]["hybrid_w"] == cfg["selected"]["hybrid_w"]


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
    variants = (
        (ex.analog_variant_estimator(ANALOG_MISFIT, 0.0),
         dict(selection=ANALOG_MISFIT, exclude_yr=0.0)),
        (ex.analog_variant_estimator(ANALOG_MISFIT, 600.0),
         dict(selection=ANALOG_MISFIT, exclude_yr=600.0)),
        (ex.analog_localization_estimator(ANALOG_MISFIT, None),
         dict(selection=ANALOG_MISFIT, exclude_yr=600.0, analog_localization_km=None)),
        (ex.analog_localization_estimator(ANALOG_MISFIT, 3000.0),
         dict(selection=ANALOG_MISFIT, exclude_yr=600.0, analog_localization_km=3000.0)),
    )
    df = ex.run_hgaoenkf_withholding_variants(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(tmp_path),
        k=K, hybrid_w=W, variants=variants, k_folds=3, b_scales=B_SCALES, seed=0)

    assert set(df["method"]) == {
        "hgaoenkf_misfit_excl0_temporal_deflate",
        "hgaoenkf_misfit_excl600_temporal_deflate",
        "hgaoenkf_locstatic_temporal_deflate",
        "hgaoenkf_loc3000_temporal_deflate",
    }

    def ce(method):
        sub = df[(df.method == method) & (df.metric == "ce") & (df.channel == "pooled")
                 & (df.split == "test") & (df.fold == -1)]
        return sub["value"].to_numpy()

    # Both axes have to bite, or the pass is scoring one estimator under several names.
    assert not np.allclose(ce("hgaoenkf_misfit_excl0_temporal_deflate"),
                           ce("hgaoenkf_misfit_excl600_temporal_deflate"))
    assert not np.allclose(ce("hgaoenkf_locstatic_temporal_deflate"),
                           ce("hgaoenkf_loc3000_temporal_deflate"))
    # A null lengthscale is the static covariance's own, so that variant is the banded
    # misfit run under a second tag and must score identically to it.
    assert np.array_equal(ce("hgaoenkf_locstatic_temporal_deflate"),
                          ce("hgaoenkf_misfit_excl600_temporal_deflate"))
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
