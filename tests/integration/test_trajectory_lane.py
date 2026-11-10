"""End-to-end checks on the time-series lane (paleoreco.assim.lanes).

The conftest cube is twelve states long, too short for a low-pass window to leave
anything after edge trimming, so these tests build a longer run locally. The observations
are given per-site block widths so a sample sits some way in time from the state it is
assimilated at, which is the situation the lane exists to measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex
from paleoreco.assim import lanes
from paleoreco.assim.priors import build_prior
from paleoreco.assim.threedvar import ThreeDVar

N_AGES = 80
STEP = 25
B_SCALES = (0.5, 1.0, 2.0)
BASE = {"method": "3dvar"}


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


def _lane(cube, ages, lats, lons, valid, obs):
    return lanes.trajectory_lane(cube, ages, lats, lons, valid, obs, min_obs=2)


def _method(cube, ages, lats, lons, valid, lane):
    prior = build_prior(cube, ages, lats, lons, lane.prior_idx, valid)
    return ThreeDVar(prior.B, lane.shape)


@pytest.fixture
def lane(run_cube, run_ages, lats, lons, valid, run_obs):
    return _lane(run_cube, run_ages, lats, lons, valid, run_obs)


@pytest.fixture
def method(run_cube, run_ages, lats, lons, valid, lane):
    return _method(run_cube, run_ages, lats, lons, valid, lane)


def test_the_selection_arm_carries_skill_rows_alone(lane, method):
    """Pooled rRMSE is the only thing the selection arm is read for.

    Scoring the timescale metrics on it would put the band metrics inside the loop that
    picks ``b_scale``.
    """
    rows = lanes.scan_pseudo_proxy(lane, method, B_SCALES, BASE)
    assert {r["split"] for r in rows} == set(lanes.SPLITS)
    assert {r["metric"] for r in rows} == {"ce", "rrmse", "amplitude"}


def test_the_full_pass_emits_both_timescale_families(lane, method):
    rows, fields = lanes.score_trajectory(lane, method, 1.0, BASE)
    metrics = {r["metric"] for r in rows if r["method"] == "3dvar"}
    assert {"ssim", "crps", "rcrv_dispersion"} <= metrics
    assert {"ce_lp25", "ce_lp100", "ce_lp250", "ce_bp25_100"} <= metrics
    band = [r["value"] for r in rows if r["metric"] == "ce_bp25_100" and r["method"] == "3dvar"]
    assert band and np.isfinite(band).all()
    assert {r["split"] for r in rows} == {"test"}
    assert fields["recon"].shape == fields["truth_anom"].shape


def test_the_observations_are_deflated_and_the_references_read_them_raw(lane):
    """The update consumes the corrected pair; the prior-free fields interpolate the sample."""
    stale = [net for nets in lane.networks for net in nets.values()]
    assert any(not np.allclose(net.y, net.y_read) for net in stale)
    for net in stale:
        assert np.all(net.r >= 1.0 - 1e-12)                  # sse = 1, only ever inflated


def test_wide_dating_blocks_cost_skill(run_cube, run_ages, lats, lons, valid):
    """Samples dated to a block must score worse than samples dated to one age.

    This catches the offset lookup being wired the wrong way round: with every block one
    age wide the offset is zero and the lane has no staleness left to charge for.
    """
    scores = {}
    for name, widths in (("fresh", [1] * 6), ("stale", [16] * 6)):
        obs = _obs_table(run_ages, lats, lons, widths=widths)
        lane = _lane(run_cube, run_ages, lats, lons, valid, obs)
        rows = pd.DataFrame(lanes.scan_pseudo_proxy(
            lane, _method(run_cube, run_ages, lats, lons, valid, lane), B_SCALES, BASE))
        sub = rows[(rows.split == "test") & (rows.channel == "pooled") & (rows.metric == "rrmse")]
        scores[name] = sub["value"].min()
    assert scores["fresh"] < scores["stale"], scores


def test_the_reported_arm_assimilates_the_age_s_own_network(lane, method):
    """A network borrowed from elsewhere turns over completely between neighbouring ages,
    which puts geometry rather than climate into the fastest band.
    """
    _, fields = lanes.score_trajectory(lane, method, 1.0, BASE)
    np.testing.assert_array_equal(fields["shape_ages"], fields["ages"])
    assert np.all(np.diff(fields["ages"]) == STEP)


def test_ages_without_a_network_are_skipped_not_borrowed_for(run_cube, run_ages, lats, lons,
                                                             valid, run_obs):
    """An age carrying no proxies is recorded and left out, and the rest stay consecutive.

    The real record has no pollen at its young end. Reconstructing those ages from a
    borrowed network would report a skill the product cannot have.
    """
    empty = run_ages[:6]                       # the youngest ages of the truth half
    thinned = run_obs[~run_obs["age"].isin(empty.tolist())]
    lane = _lane(run_cube, run_ages, lats, lons, valid, thinned)
    assert lane.meta["skipped_ages"] == empty.tolist()
    assert lane.meta["n_truths"] == len(run_ages) // 2 - len(empty)
    assert lane.meta["ages"][0] == int(run_ages[len(empty)])
    assert len(lane.truth_anom) == lane.meta["n_truths"]
    # A band the shortened run cannot support produces no row rather than a broken one.
    rows, _ = lanes.score_trajectory(lane, _method(run_cube, run_ages, lats, lons, valid, lane),
                                     1.0, BASE)
    assert not any(r["metric"] == "ce_bp1000_2000" for r in rows)


def test_posterior_var_within_prior(run_cube, run_ages, lats, lons, valid, lane, method):
    _, fields = lanes.score_trajectory(lane, method, 2.0, BASE)
    prior = build_prior(run_cube, run_ages, lats, lons, lane.prior_idx, valid)
    assert (fields["post_var"] <= 2.0 * np.diag(prior.B).reshape(lane.shape) + 1e-4).all()


def test_the_run_driver_selects_on_the_selection_arm(tmp_path, run_cube, run_ages, lats, lons,
                                                     valid, run_obs):
    cfg = ex.run_trajectory(run_cube, run_ages, lats, lons, valid, run_obs, tmp_path,
                            estimator="3dvar", taper=dict(localization_km=None,
                                                          shrinkage_lambda=0.0, alpha=1.0),
                            b_scales=B_SCALES)
    M = ex.read_metrics(tmp_path)
    assert cfg["b_scale"] == float(ex.select(M)["b_scale"])
    assert cfg["observation_model"] == {"spatial": False, "temporal": True}
    full = M[M.metric == "ssim"]
    assert set(full["b_scale"].dropna()) == {cfg["b_scale"]}
    assert (tmp_path / "trajectory_analysis.npz").exists()


# ---------------------------------------------------------------------------
# The borrowed-network observation operator.
# ---------------------------------------------------------------------------
def test_offset_is_transplanted_not_the_absolute_block_centre():
    """A borrowed sample reports on ``age + (centre - shape_age)``."""
    ages = (29_100 + 25 * np.arange(804)).astype(np.int64)
    shape_age, age = 42_375, 31_000
    centre = np.array([42_375.0, 42_400.0, 43_462.0])   # offsets 0, +25, +1087
    src, on_axis = lanes.transplanted_source(centre, shape_age, age, ages)
    assert on_axis.all()
    np.testing.assert_array_equal(ages[src], np.array([31_000, 31_025, 32_075]))


def test_reads_off_the_archive_are_dropped_not_clamped():
    """An offset running past the end of the archive has no state to read."""
    ages = (29_100 + 25 * np.arange(804)).astype(np.int64)
    shape_age, age = 40_000, 29_150                 # near the young end of the axis
    centre = np.array([38_600.0, 40_000.0, 41_400.0])   # offsets -1400, 0, +1400
    src, on_axis = lanes.transplanted_source(centre, shape_age, age, ages)
    assert list(on_axis) == [False, True, True]     # 29,150 - 1,400 is off the axis
    np.testing.assert_array_equal(ages[src], np.array([29_150, 30_550]))
