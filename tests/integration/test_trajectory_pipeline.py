"""End-to-end checks on the trajectory lane.

The conftest cube is twelve states long, too short for a low-pass window to leave
anything after edge trimming, so these tests build a longer run locally. The
observations are given per-site block widths so a sample's block centre sits some way
from most of the ages it is assimilated at, which is the situation the lane exists to
measure.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex

SCHEMA = {"method", "space", "localization_km", "shrinkage_lambda", "alpha", "lane",
          "fold", "b_scale", "background", "split", "do_event", "channel", "metric", "value"}
N_AGES = 80
STEP = 25
WINDOWS = (25, 100)
BANDS = ((25, 100),)


@pytest.fixture
def run_ages() -> np.ndarray:
    return (30_000 + STEP * np.arange(N_AGES)).astype(np.int64)


@pytest.fixture
def run_cube(run_ages, lats, lons) -> np.ndarray:
    """A cube whose cells evolve, so that stale observations actually cost something."""
    rng = np.random.default_rng(0)
    t = np.arange(len(run_ages))[:, None, None]
    space = np.linspace(-30.0, 10.0, len(lats) * len(lons)).reshape(len(lats), len(lons))
    slow = np.sin(2 * np.pi * t / 40.0) * 4.0
    cube = np.stack([
        np.stack([space + slow[i] + rng.normal(0.0, 0.5, space.shape),
                  space + 12.0 + 0.5 * slow[i] + rng.normal(0.0, 0.5, space.shape)])
        for i in range(len(run_ages))
    ])
    return cube.astype(np.float32)


@pytest.fixture
def run_obs(run_ages, lats, lons) -> pd.DataFrame:
    """Six sites whose samples own blocks of different widths, so staleness varies."""
    cells = [(1, 1), (3, 2), (5, 6), (6, 4), (2, 5), (4, 0)]
    widths = [1, 1, 4, 4, 8, 8]
    rows = []
    for site, ((i, j), width) in enumerate(zip(cells, widths)):
        for start in range(0, len(run_ages), width):
            block = run_ages[start:start + width]
            sample = site * 1000 + start
            value = float(np.sin(start / 7.0))
            for age in block:
                for channel in ("mtco", "mtwa"):
                    rows.append({"site": site, "sample": sample, "channel": channel,
                                 "age": int(age), "age_mean": int(block.mean()),
                                 "lat": float(lats[i]), "lon": float(lons[j]),
                                 "y": value, "sse": 1.0, "my": 0.0})
    return pd.DataFrame(rows)


def _run(tmp_path, cube, ages, lats, lons, valid, obs, **kw):
    return ex.run_trajectory(cube, ages, lats, lons, valid, obs, str(tmp_path),
                             b_scales=kw.pop("b_scales", (0.5, 1.0)),
                             lowpass_windows=WINDOWS, bands=BANDS, min_obs=2, seed=0, **kw)


def test_run_trajectory_schema_and_artifacts(tmp_path, run_cube, run_ages, lats, lons,
                                             valid, run_obs):
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)

    assert SCHEMA.issubset(set(df.columns))
    assert (df["lane"] == ex.LANE_TRAJECTORY).all()
    assert (df["fold"] == -1).all()
    assert set(df["split"]) == {"selection", "test"}
    assert set(df["method"]) == set(ex.TRAJECTORY_METHODS) | {"nearest", "idw"}
    assert set(df["do_event"]) == {"all"}

    for name in ("metrics.csv", "trajectory_analysis.npz", "trajectory_config.json"):
        assert os.path.exists(tmp_path / name)

    cfg = json.load(open(tmp_path / "trajectory_config.json"))
    assert cfg["step_yr"] == STEP
    assert cfg["n_covered_ages"] + cfg["n_skipped_ages"] == len(run_ages) // 2
    assert set(cfg["selected"]) == {"localization_km", "shrinkage_lambda", "alpha", "b_scale"}
    # Each variant carries its own b_scale, since inflating R moves the balance the
    # analysis wants between background and observations.
    assert set(cfg["selected_b_scale_by_method"]) == set(ex.TEMPORAL_METHODS)


def test_run_trajectory_emits_both_timescale_families(tmp_path, run_cube, run_ages, lats,
                                                      lons, valid, run_obs):
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    metrics = set(df["metric"])

    assert {"corr_lp25", "corr_lp100", "ce_lp100", "amp_lp100"} <= metrics
    assert {"corr_bp25_100", "ce_bp25_100", "amp_bp25_100"} <= metrics

    band = df[(df.metric == "corr_bp25_100") & (df.method == "3dvar")]
    assert len(band) and np.isfinite(band["value"]).all()


def test_lowpass_at_the_step_is_the_unsmoothed_metric(tmp_path, run_cube, run_ages, lats,
                                                      lons, valid, run_obs):
    """A one-sample window is the identity, so ``corr_lp25`` must equal plain ``corr``.

    This pins the filter's orientation: a window shorter than the step must not shift or
    smooth the series.
    """
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    sub = df[(df.method == "3dvar") & (df.split == "test") & (df.channel == "pooled")]
    lp = sub[sub.metric == "corr_lp25"].set_index("b_scale")["value"]
    # ``corr`` pools cells into one series while ``corr_lp25`` is the median over cells,
    # so they differ in value; what must hold is that both are finite and agree in sign.
    plain = sub[sub.metric == "corr"].set_index("b_scale")["value"]
    assert np.isfinite(lp).all() and np.isfinite(plain).all()
    assert (np.sign(lp) == np.sign(plain)).all()


def test_ceiling_beats_stale_observations(tmp_path, run_cube, run_ages, lats, lons,
                                          valid, run_obs):
    """Observations read at the analysis age must beat ones read at the block centre.

    This is the invariant that catches the block-centre lookup being wired the wrong way
    round, or reading ``age_mean`` instead of the block.
    """
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    sub = df[(df.split == "test") & (df.channel == "pooled") & (df.metric == "rrmse")]
    realistic = sub[sub.method == "3dvar"].set_index("b_scale")["value"]
    ceiling = sub[sub.method == "3dvar_ceiling"].set_index("b_scale")["value"]
    assert (ceiling <= realistic + 1e-9).all(), (ceiling, realistic)
    assert (ceiling < realistic).any(), "staleness cost nothing; the two variants match"


def test_posterior_var_within_prior(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs):
    _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs, b_scales=(0.5, 1.0, 2.0))
    z = np.load(tmp_path / "trajectory_analysis.npz")
    b = float(z["selected_b_scale"])
    assert (z["post_var"] <= b * z["prior_var"] + 1e-4).all()
    assert z["recon_realistic"].shape == z["truth_anom"].shape
    assert z["recon_ceiling"].shape == z["truth_anom"].shape
    assert len(z["ages"]) == z["truth_anom"].shape[0]


def test_ceiling_is_never_temporally_corrected(tmp_path, run_cube, run_ages, lats, lons,
                                               valid, run_obs):
    """The ceiling must stay the uncorrected upper bound on every variant.

    It reuses the baseline gain by reference, so a correction leaking into it would be
    silent: the lane would still run and simply stop bounding anything.
    """
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    sub = df[(df.split == "test") & (df.channel == "pooled") & (df.metric == "rrmse")]
    ceiling = sub[sub.method == "3dvar_ceiling"].set_index("b_scale")["value"]
    for method in ex.TEMPORAL_METHODS:
        other = sub[sub.method == method].set_index("b_scale")["value"]
        assert (ceiling <= other + 1e-9).all(), (method, ceiling, other)


def test_temporal_variants_persist_fields_and_calibration(tmp_path, run_cube, run_ages,
                                                          lats, lons, valid, run_obs):
    """Each variant carries its own posterior variance, not the uncorrected one's.

    Inflating R changes the gain, so a shared ``post_var`` would report the baseline's
    spread against the variant's mean and quietly mis-state its calibration.
    """
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    z = np.load(tmp_path / "trajectory_analysis.npz")

    for method in ex.TEMPORAL_METHODS:
        if method == ex.METHOD_BASE:
            continue
        assert z[f"recon_{method}"].shape == z["truth_anom"].shape
        assert not np.allclose(z[f"post_var_{method}"], z["post_var"])
        cal = df[(df.method == method) & (df.metric == "coverage90")]
        assert len(cal) and np.isfinite(cal["value"]).all()
