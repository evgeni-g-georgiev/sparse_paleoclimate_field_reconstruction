"""End-to-end smoke test of the subspace score prior through the generative runners.

A tiny untrained :class:`~paleoreco.assim.spectral.SpectralSampler` drives both
lanes on the synthetic fixtures. The tests guard the tidy-CSV schema, the npz keys
notebook 09 reads, and the new ``method`` label, so ``generative_spectral`` drops
into the comparison alongside the pixel-space generative runs with no
special-casing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from paleoreco.assim import experiments_generative as g
from paleoreco.assim.priors import build_prior
from paleoreco.assim.spectral import SpectralSampler, build_spectral_basis
from paleoreco.data.splits import chronological_half_split
from paleoreco.models.spectral_score import build_spectral_denoiser

SKILL = {"ce", "corr", "rmse", "rrmse", "amplitude", "ssim"}
GAUSS_CAL = {"crps", "crpss", "rcrv_bias", "rcrv_dispersion", "coverage90"}
NATIVE_CAL = {"ecr", "sharpness", "crps_ens", "coverage90_ens"}
SELECTION = {"rrmse_ens"}
K = 3


def _sampler(cube, ages, lats, lons, valid, prior_idx, *, n_steps=4,
             cond_cov="anchor"):
    """Tiny spectral sampler on the fixture cube, plus the prior-variance field.

    The network is untrained, so its output layer is still zero-initialised and the
    sampler is exactly the Gaussian prior. That is the right smoke test: it
    exercises the whole path (whitening, guidance, complement completion) while
    keeping the numbers well behaved.
    """
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid)
    basis = build_spectral_basis(prior.B, (2, *cube.shape[2:]), prior.safe_valid, K,
                                 meta={"prior": "test"})
    net = build_spectral_denoiser(dim=K, width=16, depth=1)
    sampler = SpectralSampler(net, basis, n_steps=n_steps, cond_cov=cond_cov,
                              device="cpu")
    return sampler, sampler.sample_prior(8, seed=0).var(axis=0, ddof=1)


def _varying_obs(obs_long):
    df = obs_long.copy()
    chan = df["channel"].map({"mtco": 0, "mtwa": 1}).to_numpy()
    df["y"] = df["site"].to_numpy() * 2.0 + chan + 0.5 * np.sin(df["age"].to_numpy() / 500.0)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    return df


@pytest.mark.parametrize("cond_cov", ["anchor", "jacobian"])
def test_run_ppe_spectral_schema_and_npz(tmp_path, cube, ages, lats, lons, valid,
                                         obs_long, cond_cov):
    out = tmp_path / "generative_spectral"
    prior_idx, _ = chronological_half_split(np.asarray(ages, np.int64), stride=1)
    sampler, prior_ens_var = _sampler(cube, ages, lats, lons, valid, prior_idx,
                                      cond_cov=cond_cov)

    df = g.run_ppe_generative(
        cube, ages, lats, lons, valid, obs_long, str(out), sampler=sampler,
        prior_ens_var=prior_ens_var, method="generative_spectral",
        gamma_grid=(1.0, 2.0), n_samples=4, n_samples_select=2, n_shapes=3,
        n_select=2, truth_stride=1, sel_subsample_truths=None, seed=0)

    spec = df[df["method"] == "generative_spectral"]
    assert len(spec)
    assert (spec["space"] == "generative").all()
    assert set(spec["split"]) == {"selection", "test"}
    assert (SKILL | GAUSS_CAL | NATIVE_CAL | SELECTION).issubset(set(spec["metric"]))
    # The operating point is the background amplitude, recorded as b_scale.
    assert set(spec.loc[spec["split"] == "selection", "b_scale"]) == {1.0, 2.0}
    rmse = spec[(spec["metric"] == "rmse") & (spec["channel"] == "pooled")]
    assert len(rmse) and np.isfinite(rmse["value"].to_numpy()).all()

    cfg = json.load(open(out / "ppe_config.json"))
    assert cfg["method"] == "generative_spectral"
    assert cfg["selected"]["b_scale"] in (1.0, 2.0)
    assert cfg["sampler"]["guidance_cov"]["k"] == K
    # The run must record which conditional covariance produced it, or the anchor
    # and jacobian arms would be indistinguishable after the fact.
    assert cfg["sampler"]["cond_cov"] == cond_cov

    z = np.load(out / "ppe_analysis.npz")
    for k in ("truth_anom", "safe_valid", "b_scales", "recon_climatological", "prior_var",
              "post_var", "obs_n", "obs_chan", "obs_lat", "obs_lon", "lats", "lons",
              "naive_idw", "naive_nearest", "prior_ens_var"):
        assert k in z.files, k
    assert z["recon_climatological"].shape[0] == 1
    assert np.isclose(z["b_scales"][0], cfg["selected"]["b_scale"])


def test_run_withholding_spectral_schema_and_npz(tmp_path, cube, ages, lats, lons,
                                                 valid, obs_long):
    out = tmp_path / "generative_spectral"
    all_idx = np.arange(len(ages))
    sampler, _ = _sampler(cube, ages, lats, lons, valid, all_idx)

    df = g.run_withholding_generative(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(out),
        sampler=sampler, method="generative_spectral", gamma_grid=(1.0, 2.0),
        n_samples=4, n_samples_select=2, k_folds=3, age_stride=1, seed=0)

    spec = df[df["method"] == "generative_spectral"]
    assert set(spec["split"]) == {"selection", "test"}
    rmse = spec[(spec["metric"] == "rmse") & (spec["channel"] == "pooled")]
    assert len(rmse) and np.isfinite(rmse["value"].to_numpy()).all()

    z = np.load(out / "withholding_random_predictions.npz")
    for k in ("b_scales", "actual", "climatological_pred", "channel", "post_var_pred",
              "sse", "rep_var"):
        assert k in z.files, k
    assert z["climatological_pred"].shape[0] == 1
    cfg = json.load(open(out / "withholding_random_config.json"))
    assert cfg["method"] == "generative_spectral"


def test_method_label_defaults_to_generative(tmp_path, cube, ages, lats, lons,
                                             valid, obs_long):
    """The new ``method`` kwarg must not change what existing callers emit."""
    out = tmp_path / "generative"
    prior_idx, _ = chronological_half_split(np.asarray(ages, np.int64), stride=1)
    sampler, prior_ens_var = _sampler(cube, ages, lats, lons, valid, prior_idx)

    df = g.run_ppe_generative(
        cube, ages, lats, lons, valid, obs_long, str(out), sampler=sampler,
        prior_ens_var=prior_ens_var, gamma_grid=(1.0,), n_samples=4,
        n_samples_select=2, n_shapes=3, n_select=2, truth_stride=1,
        sel_subsample_truths=None, seed=0)

    assert set(df.loc[df["space"] == "generative", "method"]) <= {
        "generative", "nearest", "idw"}
    assert (df["method"] == "generative").any()
    assert json.load(open(out / "ppe_config.json"))["method"] == "generative"
