"""End-to-end for the regime lane: the runner, its schema, and its artefacts.

Mirrors ``test_spectral_pipeline``. ``generative_regime`` must write the same tidy rows
every other method writes, and the three columns naming its operating point must survive
into the CSV and the config.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from contextlib import contextmanager

from paleoreco.assim import experiments_generative as gex
from paleoreco.assim.regimes import build_regime_mixture, partition_states
from paleoreco.assim.regime_sampler import RegimeSampler
from paleoreco.data.cube import apply_anomaly, compute_zscore_stats
from paleoreco.data.splits import chronological_half_split

REGIME_GRID = ((1, 1.0), (2, 0.5))
TEMPER_GRID = (0.0,)                 # no trained residual in the smoke lane


def _factory(cube, ages, valid, shape):
    """``make_sampler(J, rho, temper)`` over the prior half."""
    prior_idx, _ = chronological_half_split(np.asarray(ages, np.int64), stride=1)
    stats = compute_zscore_stats(cube, prior_idx, valid)
    states = apply_anomaly(cube, stats)[prior_idx].reshape(len(prior_idx), -1)

    @contextmanager
    def make(n_regimes, rho, temper):
        mix = build_regime_mixture(states, partition_states(states, n_regimes, seed=0),
                                   shape, stats["safe_valid"], rho=rho)
        yield RegimeSampler(mix, b_scale=1.0, inference="exact", n_steps=8,
                            max_batch=64, device="cpu")

    return make


def test_run_ppe_generative_regime_schema_and_npz(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out = tmp_path / "generative_regime"
    shape = (cube.shape[1], len(lats), len(lons))
    df = gex.run_ppe_generative_regime(
        cube, ages, lats, lons, valid, obs_long, str(out),
        make_sampler=_factory(cube, ages, valid, shape),
        regime_grid=REGIME_GRID, temper_grid=TEMPER_GRID, gamma_grid=(1.0, 2.0),
        n_samples=8, n_samples_select=8, n_shapes=3, n_select=2, n_noise=1,
        truth_stride=1, sel_subsample_truths=2, n_prior_var=16, seed=0)

    assert (df["method"] == "generative_regime").any()
    assert {"regime_j", "regime_rho", "regime_temper"}.issubset(df.columns)

    # Every grid point is scored on the selection split; only the winner on the test split.
    sel = df[(df.split == "selection") & (df.method == "generative_regime")]
    combos = sel[["regime_j", "regime_rho", "regime_temper"]].drop_duplicates()
    assert len(combos) == len(REGIME_GRID) * len(TEMPER_GRID)
    test = df[(df.split == "test") & (df.method == "generative_regime")]
    assert len(test[["regime_j", "regime_rho", "regime_temper"]].drop_duplicates()) == 1

    # The artefacts notebook 09 and the diagnostics read.
    assert os.path.exists(out / "ppe_analysis.npz")
    z = np.load(out / "ppe_analysis.npz")
    assert {"truth_anom", "recon_climatological", "post_var", "prior_ens_var",
            "safe_valid"}.issubset(set(z.files))

    cfg = json.load(open(out / "ppe_config.json"))
    assert cfg["method"] == "generative_regime"
    assert set(cfg["selected"]) == {"b_scale", "regime_j", "regime_rho", "regime_temper"}
    assert (cfg["selected"]["regime_j"], cfg["selected"]["regime_rho"]) in [
        tuple(p) for p in cfg["regime_grid"]
    ] or [cfg["selected"]["regime_j"], cfg["selected"]["regime_rho"]] in cfg["regime_grid"]

    # The prior-free reference rows come along, so the lane is readable on its own.
    assert set(df["method"].unique()) >= {"generative_regime", "nearest", "idw"}


def test_regime_rows_carry_nan_taper_like_every_generative_method(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    """The taper columns describe a row's B; a generative row has none to describe."""
    out = tmp_path / "generative_regime"
    shape = (cube.shape[1], len(lats), len(lons))
    df = gex.run_ppe_generative_regime(
        cube, ages, lats, lons, valid, obs_long, str(out),
        make_sampler=_factory(cube, ages, valid, shape),
        regime_grid=((1, 1.0),), temper_grid=(0.0,), gamma_grid=(1.0,),
        n_samples=8, n_samples_select=8, n_shapes=3, n_select=2, n_noise=1,
        truth_stride=1, sel_subsample_truths=2, n_prior_var=8, seed=0,
        highpass_window=None)
    reg = df[df.method == "generative_regime"]
    for col in ("localization_km", "shrinkage_lambda", "alpha", "highpass_window"):
        assert reg[col].isna().all()

def _varying_obs(obs_long):
    """Site- and age-varying measurements, so the withholding folds are not degenerate."""
    df = obs_long.copy()
    chan = df["channel"].map({"mtco": 0, "mtwa": 1}).to_numpy()
    df["y"] = df["site"].to_numpy() * 2.0 + chan + 0.5 * np.sin(df["age"].to_numpy() / 500.0)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    return df


def test_run_withholding_generative_regime_schema_and_npz(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    """The withholding twin writes the same schema and names the same operating point."""
    out = tmp_path / "generative_regime"
    shape = (cube.shape[1], len(lats), len(lons))
    df = gex.run_withholding_generative_regime(
        cube, ages, lats, lons, valid, _varying_obs(obs_long), str(out),
        make_sampler=_factory(cube, ages, valid, shape),
        regime_grid=REGIME_GRID, temper_grid=TEMPER_GRID, gamma_grid=(1.0, 2.0),
        n_samples=4, n_samples_select=2, k_folds=3, age_stride=1, seed=0)

    reg = df[df.method == "generative_regime"]
    assert set(reg["split"]) == {"selection", "test"}
    rmse = reg[(reg.metric == "rmse") & (reg.channel == "pooled")]
    assert len(rmse) and np.isfinite(rmse["value"].to_numpy()).all()

    # Every combo on the selection split, only the winner on the test rotation.
    sel = reg[reg.split == "selection"][["regime_j", "regime_rho", "regime_temper"]]
    assert len(sel.drop_duplicates()) == len(REGIME_GRID) * len(TEMPER_GRID)
    test = reg[reg.split == "test"][["regime_j", "regime_rho", "regime_temper"]]
    assert len(test.drop_duplicates()) == 1

    z = np.load(out / "withholding_random_predictions.npz")
    for k in ("b_scales", "actual", "climatological_pred", "channel", "post_var_pred",
              "sse", "rep_var"):
        assert k in z.files, k
    cfg = json.load(open(out / "withholding_random_config.json"))
    assert cfg["method"] == "generative_regime"
    assert set(cfg["selected"]) == {"b_scale", "regime_j", "regime_rho", "regime_temper"}
    assert (cfg["selected"]["regime_j"], cfg["selected"]["regime_rho"]) in [
        tuple(p) for p in REGIME_GRID
    ]
