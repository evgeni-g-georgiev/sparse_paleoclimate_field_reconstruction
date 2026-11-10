"""End-to-end checks on the snapshot lane (paleoreco.assim.lanes).

The lane is built once and scored many times, so these guard what it draws, that a scan
emits only the skill rows a selection reads, and that the full pass adds everything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim import lanes
from paleoreco.assim.priors import build_prior
from paleoreco.assim.threedvar import ThreeDVar
from paleoreco.eval.rows import field_skill_rows

# The fixture network is six sites over two channels, so every age clears this.
MIN_OBS = 4
B_SCALES = (0.5, 1.0, 2.0)
BASE = {"method": "3dvar"}


@pytest.fixture
def lane(cube, ages, lats, lons, valid, obs_long):
    return lanes.snapshot_lane(cube, ages, lats, lons, valid, obs_long, min_obs=MIN_OBS)


@pytest.fixture
def method(cube, ages, lats, lons, valid, lane):
    prior = build_prior(cube, ages, lats, lons, lane.prior_idx, valid)
    return ThreeDVar(prior.B, lane.shape)


def test_the_older_half_builds_the_prior_and_the_younger_supplies_truths(lane, ages):
    mid = len(ages) // 2
    assert np.array_equal(lane.prior_idx, np.arange(mid, len(ages)))
    assert lane.truth_anom.shape[0] == mid
    # Truths are anomalies about their own half's mean.
    assert np.allclose(lane.truth_anom.mean(axis=0), 0.0, atol=1e-5)
    assert lane.meta["prior_ages"][0] > lane.meta["truth_ages"][1]


def test_each_truth_is_observed_by_two_networks_with_independent_noise(lane):
    for truth, nets in zip(lane.truth_anom, lane.networks):
        assert set(nets) == set(lanes.SPLITS)
        for net in nets.values():
            noise = net.y - truth.ravel()[net.gather]
            assert len(net.gather) >= MIN_OBS and np.any(noise != 0.0)
            assert (net.r > 0).all()
        sel, test = nets["selection"], nets["test"]
        assert not (np.array_equal(sel.gather, test.gather) and np.allclose(sel.y, test.y))


def test_the_same_seed_draws_the_same_lane(cube, ages, lats, lons, valid, obs_long, lane):
    """Every configuration a grid scans must meet the same networks and noise."""
    again = lanes.snapshot_lane(cube, ages, lats, lons, valid, obs_long, min_obs=MIN_OBS)
    for a, b in zip(lane.networks, again.networks):
        for split in lanes.SPLITS:
            assert np.array_equal(a[split].gather, b[split].gather)
            assert np.array_equal(a[split].y, b[split].y)


def test_a_scan_emits_skill_rows_on_both_splits_at_every_amplitude(lane, method):
    rows = lanes.scan_pseudo_proxy(lane, method, B_SCALES, BASE)
    assert {r["split"] for r in rows} == set(lanes.SPLITS)
    assert {r["b_scale"] for r in rows} == set(B_SCALES)
    assert {r["metric"] for r in rows} == {"ce", "rrmse", "amplitude"}
    assert {r["channel"] for r in rows} == {"pooled", "mtco", "mtwa"}
    assert all(np.isfinite(r["value"]) for r in rows)


def test_the_full_pass_adds_ssim_calibration_and_the_references(lane, method):
    rows, fields, skill = lanes.score_snapshot(lane, method, 1.0, BASE)
    own = [r for r in rows if r["method"] == "3dvar"]
    assert {r["metric"] for r in own} == {"ssim", "crps", "rcrv_dispersion"}
    assert {r["split"] for r in rows} == {"test"}
    for kind in ("nearest", "idw"):
        ref = [r for r in rows if r["method"] == kind]
        assert {r["metric"] for r in ref} == {"ce", "rrmse", "amplitude", "ssim"}
        assert all(np.isnan(r["b_scale"]) for r in ref)
    assert fields["recon"].shape == fields["truth_anom"].shape == fields["post_var"].shape
    assert fields["recon"].dtype == np.float32
    assert len(skill["ce"]) == len(skill["edges"]) - 1 == len(skill["count"])


def test_the_full_pass_mean_is_the_scan_mean(lane, method):
    """The full pass at the selected amplitude must score the analysis the scan selected."""
    scan = lanes.scan_pseudo_proxy(lane, method, (1.0,), BASE)
    _, fields, _ = lanes.score_snapshot(lane, method, 1.0, BASE)
    again = field_skill_rows(lane.truth_anom, fields["recon"].astype(np.float64),
                             lane.safe_valid, {**BASE, "b_scale": 1.0, "split": "test"})
    tested = {(r["channel"], r["metric"]): r["value"] for r in scan if r["split"] == "test"}
    for r in again:
        assert r["value"] == pytest.approx(tested[(r["channel"], r["metric"])], rel=1e-5)


def test_drawn_shapes_clear_the_observation_floor(lats, lons, valid, obs_long):
    """A shape below ``min_obs`` is re-drawn rather than assimilated."""
    shape = (2, len(lats), len(lons))
    safe_flat = np.broadcast_to(valid, shape).ravel()
    thin_age = int(obs_long["age"].min())
    # Leave one age holding a single site, well under the floor.
    thinned = obs_long[(obs_long["age"] != thin_age) | (obs_long["site"] == 1)]
    rng = np.random.default_rng(0)
    for _ in range(20):
        drawn = lanes._draw_shapes(thinned, rng, lats, lons, safe_flat, 2, min_obs=4)
        assert all(len(geom["gather"]) >= 4 for _, geom in drawn)
        assert thin_age not in [age for age, _ in drawn]
    only_thin = obs_long[(obs_long["age"] == thin_age) & (obs_long["site"] == 1)]
    with pytest.raises(ValueError, match="usable observations"):
        lanes._draw_shapes(only_thin, rng, lats, lons, safe_flat, 2, min_obs=4)
