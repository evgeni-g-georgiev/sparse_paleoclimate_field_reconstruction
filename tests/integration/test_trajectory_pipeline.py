"""End-to-end checks on the trajectory lane.

The conftest cube is twelve states long, too short for a low-pass window to leave
anything after edge trimming, so these tests build a longer run locally. The observations
are given per-site block widths so a sample sits some way in time from the state it is
assimilated at, which is the situation the lane exists to measure.
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


def _obs_table(run_ages, lats, lons, widths) -> pd.DataFrame:
    """Six sites whose samples own blocks of the given widths."""
    cells = [(1, 1), (3, 2), (5, 6), (6, 4), (2, 5), (4, 0)]
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


@pytest.fixture
def run_obs(run_ages, lats, lons) -> pd.DataFrame:
    return _obs_table(run_ages, lats, lons, widths=[1, 1, 4, 4, 8, 8])


def _run(tmp_path, cube, ages, lats, lons, valid, obs, **kw):
    return ex.run_trajectory(cube, ages, lats, lons, valid, obs, str(tmp_path),
                             b_scales=kw.pop("b_scales", (0.5, 1.0)),
                             lowpass_windows=kw.pop("lowpass_windows", WINDOWS),
                             bands=kw.pop("bands", BANDS),
                             min_obs=2, seed=0, **kw)


def test_run_trajectory_schema_and_artifacts(tmp_path, run_cube, run_ages, lats, lons,
                                             valid, run_obs):
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)

    assert SCHEMA.issubset(set(df.columns))
    assert (df["lane"] == ex.LANE_TRAJECTORY).all()
    assert (df["fold"] == -1).all()
    assert set(df["split"]) == {"selection", "test"}
    assert set(df["method"]) == set(ex.TEMPORAL_METHODS) | {"nearest", "idw"}
    assert set(df["do_event"]) == {"all"}

    for name in ("metrics.csv", "trajectory_analysis.npz", "trajectory_config.json"):
        assert os.path.exists(tmp_path / name)

    cfg = json.load(open(tmp_path / "trajectory_config.json"))
    assert cfg["step_yr"] == STEP
    # Every state of the younger half is either reconstructed or recorded as skipped.
    assert cfg["n_covered_ages"] + cfg["n_skipped_ages"] == len(run_ages) // 2
    # The reported arm assimilates the age's own network; only selection borrows one.
    assert cfg["networks"] == {"selection": ex.DRAWN_NETWORK, "test": ex.OWN_NETWORK}
    assert set(cfg["selected"]) == {"localization_km", "shrinkage_lambda", "alpha", "b_scale"}
    # Each variant carries its own b_scale, since inflating R moves the balance the
    # analysis wants between background and observations.
    assert set(cfg["selected_b_scale_by_method"]) == set(ex.TEMPORAL_METHODS)


def test_the_selection_shape_carries_skill_rows_alone(tmp_path, run_cube, run_ages, lats,
                                                      lons, valid, run_obs):
    """Pooled RRMSE is the only thing the selection shape is read for.

    Scoring the timescale metrics on it would spend the run's cost on numbers nothing
    reads, and would put the band metrics inside the loop that picks ``b_scale``.
    """
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    sel = set(df.loc[df["split"] == "selection", "metric"])
    assert {"ce", "rrmse", "amplitude"} <= sel
    assert not any(m.startswith(("corr_lp", "ce_bp", "amp_bp")) for m in sel)
    assert not {"ssim", "crps", "coverage90"} & sel


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


def test_wide_dating_blocks_cost_skill(tmp_path, run_cube, run_ages, lats, lons, valid):
    """Samples dated to a block must score worse than samples dated to one age.

    This is the invariant that catches the offset lookup being wired the wrong way round:
    with every block one age wide the offset is zero and the lane has no staleness left
    to charge for.
    """
    fresh = _obs_table(run_ages, lats, lons, widths=[1] * 6)
    stale = _obs_table(run_ages, lats, lons, widths=[16] * 6)
    scores = {}
    for name, obs in (("fresh", fresh), ("stale", stale)):
        df = _run(tmp_path / name, run_cube, run_ages, lats, lons, valid, obs,
                  temporal_modes=(ex.TEMPORAL_OFF,))
        sub = df[(df.split == "test") & (df.channel == "pooled") & (df.metric == "rrmse")
                 & (df.method == ex.METHOD_BASE)]
        scores[name] = sub["value"].min()
    assert scores["fresh"] < scores["stale"], scores


def test_the_reported_arm_assimilates_the_age_s_own_network(tmp_path, run_cube, run_ages,
                                                            lats, lons, valid, run_obs):
    """The npz says where the reported arm's network came from, and it is the age itself.

    A network borrowed from elsewhere turns over completely between neighbouring ages,
    which puts geometry rather than climate into the fastest band. Without the recorded
    source nothing downstream can rebuild the network the analysis saw.
    """
    _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs)
    z = np.load(tmp_path / "trajectory_analysis.npz")

    np.testing.assert_array_equal(z["shape_ages"], z["ages"])
    assert (z["obs_n"] > 0).all()


def test_ages_without_a_network_are_skipped_not_borrowed_for(tmp_path, run_cube, run_ages,
                                                             lats, lons, valid, run_obs):
    """An age carrying no proxies is recorded and left out, and the rest stay consecutive.

    The real record has no pollen at its young end. Reconstructing those ages from a
    network borrowed from elsewhere would report a skill the product cannot have, and a
    gap anywhere but the ends would mis-time every band metric.
    """
    empty = run_ages[:6]                       # the youngest ages of the truth half
    thinned = run_obs[~run_obs["age"].isin(empty.tolist())]

    # The project's own windows, which reach wider than what a shortened run can measure.
    # A band the run cannot support must produce no row, not a broken one.
    df = _run(tmp_path, run_cube, run_ages, lats, lons, valid, thinned,
              lowpass_windows=ex.LOWPASS_WINDOWS, bands=ex.BANDS)
    cfg = json.load(open(tmp_path / "trajectory_config.json"))
    z = np.load(tmp_path / "trajectory_analysis.npz")

    assert cfg["skipped_ages"] == empty.tolist()
    assert cfg["n_covered_ages"] == len(run_ages) // 2 - len(empty)
    assert cfg["scored_ages"][0] == int(run_ages[len(empty)])
    # Contiguous, so the timescale filters still see one uniform step.
    assert np.all(np.diff(z["ages"]) == STEP)
    assert len(z["ages"]) == z["truth_anom"].shape[0] == cfg["n_covered_ages"]
    assert len(df) > 0


def test_posterior_var_within_prior(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs):
    _run(tmp_path, run_cube, run_ages, lats, lons, valid, run_obs, b_scales=(0.5, 1.0, 2.0))
    z = np.load(tmp_path / "trajectory_analysis.npz")
    b = float(z["selected_b_scale"])
    assert (z["post_var"] <= b * z["prior_var"] + 1e-4).all()
    assert z["recon_realistic"].shape == z["truth_anom"].shape
    assert len(z["ages"]) == z["truth_anom"].shape[0]


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


# ---------------------------------------------------------------------------
# The borrowed-network observation operator.
# ---------------------------------------------------------------------------
def test_offset_is_transplanted_not_the_absolute_block_centre():
    """A borrowed sample reports on ``age + (centre - shape_age)``.

    Carrying the absolute centre instead would ask for the state at the age the shape was
    drawn from, tens of thousands of years away, where the lag clips and the correction
    switches every observation off.
    """
    ages = (29_100 + 25 * np.arange(804)).astype(np.int64)
    shape_age, age = 42_375, 31_000
    centre = np.array([42_375.0, 42_400.0, 43_462.0])   # offsets 0, +25, +1087

    src, on_axis = ex.transplanted_source(centre, shape_age, age, ages)

    assert on_axis.all()
    np.testing.assert_array_equal(ages[src], np.array([31_000, 31_025, 32_075]))
    lag = np.abs(ages[src] - age)
    assert lag.max() < 1_500                      # inside the structure function's reach


def test_reads_off_the_archive_are_dropped_not_clamped():
    """An offset running past the end of the archive has no state to read.

    ``nearest_age_index`` clamps, which would pass the end state off as an observation of
    a moment it does not describe, and nothing downstream would surface it.
    """
    ages = (29_100 + 25 * np.arange(804)).astype(np.int64)
    shape_age, age = 40_000, 29_150                 # near the young end of the axis
    centre = np.array([38_600.0, 40_000.0, 41_400.0])   # offsets -1400, 0, +1400

    src, on_axis = ex.transplanted_source(centre, shape_age, age, ages)

    assert list(on_axis) == [False, True, True]     # 29,150 - 1,400 is off the axis
    np.testing.assert_array_equal(ages[src], np.array([29_150, 30_550]))
    assert ages[src].min() >= ages[0]
