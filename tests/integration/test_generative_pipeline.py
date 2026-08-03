"""End-to-end smoke test of the generative runners (experiments_generative).

A tiny untrained sampler drives both lanes on the synthetic fixtures. The tests
guard the tidy-CSV schema, the npz keys notebook 09 reads, the two-split
structure, and the generative row labels, so the method drops into the
comparison notebook unchanged.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from paleoreco.assim import experiments as ex
from paleoreco.assim import experiments_generative as g
from paleoreco.assim.generative import GuidedSampler, cell_scales
from paleoreco.data.cube import apply_anomaly, compute_zscore_stats
from paleoreco.models.diffusion import CircularUNet, EDMDenoiser

SKILL = {"ce", "corr", "rmse", "rrmse", "amplitude", "ssim"}
GAUSS_CAL = {"crps", "crpss", "rcrv_bias", "rcrv_dispersion", "coverage90"}
NATIVE_CAL = {"ecr", "sharpness", "crps_ens", "coverage90_ens"}
# The sampling-error-corrected RRMSE, which is the surface gamma is selected on.
SELECTION = {"rrmse_ens"}


def _sampler(cube, valid):
    stats = compute_zscore_stats(cube, np.arange(len(cube)), valid)
    anom = apply_anomaly(cube, stats)
    scales = cell_scales(anom, stats["safe_valid"])
    net = EDMDenoiser(CircularUNet(base_channels=16, depth=2, grid_shape=cube.shape[2:]))
    return GuidedSampler(net, scales, stats["safe_valid"], anom.var(axis=0, ddof=1),
                         n_steps=4, n_correct=1, device="cpu")


def _varying_obs(obs_long):
    df = obs_long.copy()
    chan = df["channel"].map({"mtco": 0, "mtwa": 1}).to_numpy()
    df["y"] = df["site"].to_numpy() * 2.0 + chan + 0.5 * np.sin(df["age"].to_numpy() / 500.0)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    return df


def test_run_ppe_generative_schema_and_npz(tmp_path, cube, ages, lats, lons, valid, obs_long):
    out = tmp_path / "generative"
    df = g.run_ppe_generative(
        cube, ages, lats, lons, valid, obs_long, str(out), sampler=_sampler(cube, valid),
        gamma_grid=(0.5, 1.0), n_samples=4, n_samples_select=2, n_shapes=3, n_select=2,
        truth_stride=1, sel_subsample_truths=None, seed=0)

    assert (df["method"] == "generative").any()
    gen = df[df["method"] == "generative"]
    assert (gen["space"] == "generative").all()
    assert set(gen["split"]) == {"selection", "test"}
    assert (SKILL | GAUSS_CAL | NATIVE_CAL | SELECTION).issubset(set(gen["metric"]))
    # gamma is the operating point, recorded as b_scale.
    cfg = json.load(open(out / "ppe_config.json"))
    assert cfg["selected"]["b_scale"] in (0.5, 1.0)
    assert set(gen.loc[gen["split"] == "selection", "b_scale"]) == {0.5, 1.0}

    z = np.load(out / "ppe_analysis.npz")
    for k in ("truth_anom", "safe_valid", "b_scales", "recon_climatological", "prior_var",
              "post_var", "obs_n", "obs_chan", "obs_lat", "obs_lon", "lats", "lons",
              "naive_idw", "naive_nearest", "prior_ens_var"):
        assert k in z.files, k
    assert z["recon_climatological"].shape[0] == 1
    assert np.isclose(z["b_scales"][0], cfg["selected"]["b_scale"])


def test_ppe_lanes_draw_identical_networks(tmp_path, cube, ages, lats, lons, valid, obs_long):
    """Both PPE lanes must borrow the same network shapes per truth.

    They share one seeded rng, so the streams only stay aligned while both consume it
    in the same order: one age permutation per truth, then one draw per noise
    realisation. Averaging the draws in a single vectorised call would desynchronise
    them and silently unpair the comparison.
    """
    kw = dict(n_shapes=3, n_select=2, truth_stride=1, seed=0)
    g.run_ppe_generative(
        cube, ages, lats, lons, valid, obs_long, str(tmp_path / "gen"),
        sampler=_sampler(cube, valid), gamma_grid=(1.0,), n_samples=2,
        n_samples_select=2, n_noise=5, sel_subsample_truths=None, **kw)
    ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(tmp_path / "pixel"),
               b_scales=(1.0,), n_noise=5, **kw)

    gen = np.load(tmp_path / "gen" / "ppe_analysis.npz")["drawn_ages"]
    pixel = np.load(tmp_path / "pixel" / "ppe_analysis.npz")["drawn_ages"]
    assert np.array_equal(gen, pixel)


def test_run_withholding_generative_schema_and_npz(tmp_path, cube, ages, lats, lons, valid, obs_long):
    out = tmp_path / "generative"
    df = g.run_withholding_generative(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(out),
        sampler=_sampler(cube, valid), gamma_grid=(0.5, 1.0), n_samples=4,
        n_samples_select=2, k_folds=3, age_stride=1, seed=0)

    gen = df[df["method"] == "generative"]
    assert set(gen["split"]) == {"selection", "test"}
    rmse = gen[(gen["metric"] == "rmse") & (gen["channel"] == "pooled")]
    assert len(rmse) and np.isfinite(rmse["value"].to_numpy()).all()

    z = np.load(out / "withholding_random_predictions.npz")
    for k in ("b_scales", "actual", "climatological_pred", "channel", "post_var_pred",
              "sse", "rep_var"):
        assert k in z.files, k
    assert z["climatological_pred"].shape[0] == 1
    assert os.path.exists(out / "withholding_random_config.json")
