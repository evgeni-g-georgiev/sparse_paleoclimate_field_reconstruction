"""End-to-end smoke test of the same-model PPE runner (paleoreco.assim.experiments).

``run_ppe`` splits the prior cube chronologically: the older half builds B and the
climatology, the younger half supplies own-mean-anomalised truths. Each truth draws two
network shapes, one selecting the operating point and one reporting it. These tests guard
the split wiring, the ``ppe`` tags/artifacts, and the method-agnostic contract.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex
from paleoreco.assim.threedvar import ThreeDVar

SKILL_METRICS = {"ce", "corr", "rmse", "rrmse", "amplitude", "ssim"}
CALIBRATION_METRICS = {"crps", "crpss", "rcrv_bias", "rcrv_dispersion", "coverage90"}
# The fixture network is six sites over two channels, so every age clears this.
MIN_OBS = 4


def test_run_ppe_writes_metrics_and_artifacts(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out_dir = tmp_path / "ppe"
    df = ex.run_ppe(
        cube, ages, lats, lons, valid, obs_long, str(out_dir),
        b_scales=(0.5, 1.0), min_obs=MIN_OBS, seed=0,
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
    # The prior-free references are the context a bare CE cannot supply.
    assert {"nearest", "idw"} <= set(df["method"])

    ce = df[(df["method"] == "3dvar") & (df["metric"] == "ce")
            & (df["channel"] == "pooled") & (df["split"] == "test")]
    assert len(ce) > 0
    assert np.isfinite(ce["value"].to_numpy()).all()


def test_run_ppe_persists_the_selected_scale_only(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    """Fields are kept at the selected ``b_scale`` alone, which is what consumers read.

    Keeping the whole sweep is two orders of magnitude larger over a few hundred truths,
    and a stray leading axis would silently shift every field a reader indexes.
    """
    out_dir = tmp_path / "ppe_pv"
    b_scales = (0.5, 1.0, 2.0, 10.0)
    ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(out_dir),
               b_scales=b_scales, min_obs=MIN_OBS, seed=0)
    with np.load(out_dir / "ppe_analysis.npz") as z:
        post_var = z["post_var"]       # (T, 2, n_lat, n_lon)
        recon = z["recon_climatological"]
        prior_var = z["prior_var"]     # (2, n_lat, n_lon), the b_scale=1 background
        selected = float(z["selected_b_scale"])
        truth = z["truth_anom"]

    assert selected in b_scales
    assert post_var.shape == truth.shape
    assert recon.shape == truth.shape
    assert recon.dtype == np.float32
    assert np.all(post_var <= selected * prior_var[None] + 1e-6)


def test_run_ppe_prior_uses_chunk_a(tmp_path, cube, ages, lats, lons, valid, obs_long):
    out_dir = tmp_path / "ppe_cfg"
    ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(out_dir),
               b_scales=(1.0,), min_obs=MIN_OBS, seed=0)
    with open(out_dir / "ppe_config.json") as fh:
        cfg = json.load(fh)

    mid = len(ages) // 2
    assert cfg["prior_meta"]["n_prior_ages"] == len(ages) - mid   # older half only
    assert cfg["split_index"] == mid
    assert cfg["prior_half"] == "older"
    # Both arms borrow geometry here; the trajectory lane reports on the age's own.
    assert cfg["networks"] == {"selection": ex.DRAWN_NETWORK, "test": ex.DRAWN_NETWORK}
    assert cfg["n_noise"] == 1
    # Older prior chunk and younger truth chunk share no ages.
    assert cfg["chunk_a_ages"][0] > cfg["chunk_b_ages"][1]


def test_run_ppe_make_method_default_matches_explicit(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    common = dict(b_scales=(0.5, 1.0), min_obs=MIN_OBS, seed=0)
    df_default = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                            str(tmp_path / "default"), **common)
    df_factory = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                            str(tmp_path / "factory"),
                            make_method=lambda prior, shape: ThreeDVar(prior.B, shape),
                            **common)

    pd.testing.assert_frame_equal(df_default, df_factory)
    assert (df_default["space"] == "pixel").all()


def test_drawn_shapes_clear_the_observation_floor(lats, lons, valid, obs_long):
    """A shape below ``min_obs`` is re-drawn rather than assimilated.

    The thinnest proxy ages carry a handful of sites; one such draw inside a consecutive
    run would put a near-empty analysis into the series the timescale metrics read.
    """
    shape = (2, len(lats), len(lons))
    safe_flat = np.broadcast_to(valid, shape).ravel()
    thin_age = int(obs_long["age"].min())
    # Leave one age holding a single site, well under the floor.
    thinned = obs_long[(obs_long["age"] != thin_age) | (obs_long["site"] == 1)]

    rng = np.random.default_rng(0)
    for _ in range(20):
        drawn = ex._draw_shapes(thinned, rng, lats, lons, safe_flat, 2, min_obs=4)
        assert all(len(geom["gather"]) >= 4 for _, geom in drawn)
        assert thin_age not in [age for age, _ in drawn]

    only_thin = obs_long[(obs_long["age"] == thin_age) & (obs_long["site"] == 1)]
    with pytest.raises(ValueError, match="usable observations"):
        ex._draw_shapes(only_thin, rng, lats, lons, safe_flat, 2, min_obs=4)
