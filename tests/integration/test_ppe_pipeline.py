"""End-to-end smoke test of the same-model PPE runner (paleoreco.assim.experiments).

``run_ppe`` splits the prior cube chronologically: the older half builds B and the
climatology, the younger half supplies own-mean-anomalised truths. These tests guard
the split wiring, the ``ppe`` tags/artifacts, and the method-agnostic contract.
``truth_stride=1`` keeps all chunk-B states as truths (the tiny fixture would otherwise
leave a single, zero-variance truth).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from paleoreco.assim import experiments as ex
from paleoreco.assim.threedvar import ThreeDVar

SKILL_METRICS = {"ce", "corr", "rmse", "rrmse", "amplitude", "ssim"}
CALIBRATION_METRICS = {"crps", "crpss", "rcrv_bias", "rcrv_dispersion", "coverage90"}


def test_run_ppe_writes_metrics_and_artifacts(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out_dir = tmp_path / "ppe"
    df = ex.run_ppe(
        cube, ages, lats, lons, valid, obs_long, str(out_dir),
        n_shapes=3, n_select=2, n_noise=1, b_scales=(0.5, 1.0),
        truth_stride=1, seed=0,
    )

    assert isinstance(df, pd.DataFrame)
    expected_cols = {"method", "space", "localization_km", "shrinkage_lambda", "alpha",
                     "lane", "fold", "b_scale", "background", "split", "do_event",
                     "channel", "metric", "value"}
    assert expected_cols.issubset(df.columns)
    assert (df["lane"] == "ppe").all()

    assert os.path.exists(out_dir / "metrics.csv")
    assert os.path.exists(out_dir / "ppe_analysis.npz")
    assert os.path.exists(out_dir / "ppe_skill_vs_distance.npz")
    assert os.path.exists(out_dir / "ppe_config.json")

    # Both splits are written; skill and calibration are both emitted, nothing else.
    assert set(df.loc[df["method"] == "3dvar", "split"]) == {"selection", "test"}
    assert set(df["metric"]).issubset(SKILL_METRICS | CALIBRATION_METRICS)
    assert CALIBRATION_METRICS.issubset(set(df["metric"]))
    # Calibration needs the posterior variance, which is only kept for the test shape.
    cal_rows = df[df["metric"].isin(CALIBRATION_METRICS)]
    assert set(cal_rows["split"]) == {"test"}

    ce = df[(df["method"] == "3dvar") & (df["metric"] == "ce")
            & (df["channel"] == "pooled") & (df["split"] == "test")]
    assert len(ce) > 0
    assert np.isfinite(ce["value"].to_numpy()).all()


def test_run_ppe_posterior_var_within_prior(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out_dir = tmp_path / "ppe_pv"
    ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(out_dir),
               n_shapes=3, n_select=2, n_noise=1,
               b_scales=(0.5, 1.0, 2.0, 10.0), truth_stride=1, seed=0)
    with np.load(out_dir / "ppe_analysis.npz") as z:
        post_var = z["post_var"]       # (n_b, T, 2, n_lat, n_lon)
        prior_var = z["prior_var"]     # (2, n_lat, n_lon), the b_scale=1 background
        b_scales = z["b_scales"]       # (n_b,)
    scaled_prior = b_scales[:, None, None, None, None] * prior_var[None, None]
    assert np.all(post_var <= scaled_prior + 1e-6)


def test_run_ppe_prior_uses_chunk_a(tmp_path, cube, ages, lats, lons, valid, obs_long):
    out_dir = tmp_path / "ppe_cfg"
    ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(out_dir),
               n_shapes=3, n_select=2, n_noise=1, b_scales=(1.0,),
               truth_stride=1, seed=0)
    with open(out_dir / "ppe_config.json") as fh:
        cfg = json.load(fh)

    mid = len(ages) // 2
    assert cfg["prior_meta"]["n_prior_ages"] == len(ages) - mid   # older half only
    assert cfg["split_index"] == mid
    assert cfg["prior_half"] == "older"
    # Older prior chunk and younger truth chunk share no ages.
    assert cfg["chunk_a_ages"][0] > cfg["chunk_b_ages"][1]


def test_run_ppe_make_method_default_matches_explicit(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    common = dict(n_shapes=3, n_select=2, n_noise=2,
                  b_scales=(0.5, 1.0), truth_stride=1, seed=0)
    df_default = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                            str(tmp_path / "default"), **common)
    df_factory = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                            str(tmp_path / "factory"),
                            make_method=lambda prior, shape: ThreeDVar(prior.B, shape),
                            **common)

    pd.testing.assert_frame_equal(df_default, df_factory)
    assert (df_default["space"] == "pixel").all()
