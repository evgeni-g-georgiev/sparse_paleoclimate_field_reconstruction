"""End-to-end checks on the withholding lane (paleoreco.assim.lanes).

Two stub estimators make the lane's bookkeeping visible. One marks the cells it assimilated,
so a withheld row reads one exactly where its own cell was assimilated; the other returns a
known mean and spread, so every arm's prediction and predictive variance can be checked
against its own observation model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex
from paleoreco.assim import lanes
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.method import AnalysisResult, Method
from paleoreco.assim.observations import (
    FULL_MODEL,
    ObservationModel,
    representativeness_variance,
)

K_FOLDS = 3
B_SCALES = (0.5, 1.0)
ARMS = (ObservationModel(spatial=False, temporal=False),
        ObservationModel(spatial=True, temporal=False),
        ObservationModel(spatial=False, temporal=True), FULL_MODEL)
POST_VAR = 2.0
TAPER = dict(localization_km=None, shrinkage_lambda=0.0, alpha=1.0)


class _Indicator(Method):
    """Mean 1 at the cells it assimilated and 0 elsewhere, at every amplitude."""

    def __init__(self, shape):
        self.shape = shape

    def prepare_sweep(self, gather, r_diag, b_scales):
        return np.asarray(gather), len(b_scales)

    def apply_sweep(self, gain, y_anom, background_anom, *, mean_only=False):
        gather, n_b = gain
        mean = np.zeros(int(np.prod(self.shape)))
        mean[gather] = 1.0
        return [AnalysisResult(mean.reshape(self.shape), np.full(self.shape, POST_VAR))
                for _ in range(n_b)]


class _Constant(_Indicator):
    """Mean 1 everywhere, so a prediction shows only what the scoring does to it."""

    def apply_sweep(self, gain, y_anom, background_anom, *, mean_only=False):
        _, n_b = gain
        return [AnalysisResult(np.ones(self.shape), np.full(self.shape, POST_VAR))
                for _ in range(n_b)]


def _paired_blocks(ages, lats, lons) -> pd.DataFrame:
    """Six sites in three shared cells, each sample dated to a block of three ages.

    Co-cell pairs give a nonzero spatial term, and the blocks put most rows away from their
    sample's centre, so the temporal term has something to correct.
    """
    nodes = [(lats[1], lons[1]), (lats[5], lons[6]), (lats[9], lons[9])]
    rows = []
    for sid in range(1, 7):
        cell, pos = (sid - 1) // 2, (sid - 1) % 2
        la, lo = nodes[cell]
        for t, a in enumerate(ages):
            for chan in ("mtco", "mtwa"):
                block = t // 3
                amp = (0.5, 2.0, 8.0)[cell] * (1.7 if chan == "mtwa" else 1.0)
                rows.append({"site": sid, "sample": sid * 100 + block, "channel": chan,
                             "age": int(a), "age_mean": int(a), "lat": float(la),
                             "lon": float(lo), "sse": 0.5,
                             "y": float(amp * np.sin(0.7 * block + 1.2 * pos))})
    df = pd.DataFrame(rows)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    return df


@pytest.fixture
def varying(obs_long):
    """The conftest network with values that vary by site and age, and their climatology."""
    long = obs_long.assign(y=np.sin(obs_long["age"] / 700.0) + obs_long["site"])
    long["my"] = long.groupby(["site", "channel"])["y"].transform("mean")
    return long


@pytest.fixture
def lane(cube, ages, lats, lons, valid, varying):
    return lanes.withholding_lane(cube, ages, lats, lons, valid, varying, k_folds=K_FOLDS)


@pytest.fixture
def paired_lane(cube, ages, lats, lons, valid):
    long = _paired_blocks(ages, lats, lons)
    return long, lanes.withholding_lane(cube, ages, lats, lons, valid, long, k_folds=K_FOLDS)


def test_one_pass_predicts_every_site_exactly_once_from_an_analysis_without_it(lane, varying):
    """The fixture's sites sit on distinct cells, so the indicator reads 1 at a withheld row
    exactly when that row's own site was assimilated by the analysis that predicted it.
    """
    model = ObservationModel(spatial=False, temporal=False)
    _, pred = lanes.score_withholding(lane, _Indicator(lane.shape), B_SCALES, {}, model)
    assert np.all(pred["pred"] == 0.0)
    assert not pred["cell_observed"].any()

    rows = pd.DataFrame({"site": pred["site"], "age": pred["age"], "channel": pred["channel"],
                         "fold": pred["fold"]})
    assert not rows.duplicated(["site", "age", "channel"]).any()
    assert set(rows["site"]) == set(varying["site"])
    # Each site sits in one fold, and the folds partition the sites.
    assert (rows.groupby("site")["fold"].nunique() == 1).all()
    assert sorted(rows["fold"].unique()) == list(range(K_FOLDS))
    # Every usable row of the network is predicted, so the pass is the whole lane.
    assert len(rows) == len(varying)


@pytest.mark.parametrize("model", ARMS, ids=lambda m: m.name)
def test_each_arm_is_scored_with_its_own_operator_and_variance(paired_lane, model):
    long, lane = paired_lane
    _, pred = lanes.score_withholding(lane, _Constant(lane.shape), B_SCALES, {}, model)
    rho, resid, rep, sse = pred["rho"], pred["resid_var"], pred["rep_var"], pred["sse"]
    assert (rho < 1.0).any() and (rep > 0.0).any()       # both terms have something to do
    noise = sse + (rep if model.spatial else 0.0)
    if model.temporal:
        want_pred, want_var = rho, rho ** 2 * POST_VAR + noise + resid
    else:
        want_pred, want_var = np.ones_like(rho), POST_VAR + noise
    for bj in range(len(B_SCALES)):
        assert np.allclose(pred["pred"][bj], want_pred)
        assert np.allclose(pred["pred_var"][bj], want_var)


@pytest.mark.parametrize("model", ARMS, ids=lambda m: m.name)
def test_each_arm_assimilates_under_its_own_model(paired_lane, model):
    _, lane = paired_lane
    rows = lane.batches[0].assim
    y, r = rows.assimilated(model)
    base = rows.sse + (rows.rep if model.spatial else 0.0)
    if model.temporal:
        assert np.allclose(y, rows.y / rows.rho)
        assert np.allclose(r, (base + rows.resid) / rows.rho ** 2)
    else:
        assert np.array_equal(y, rows.y) and np.allclose(r, base)


def test_the_spatial_term_is_estimated_per_fold_from_the_assimilated_sites(
        paired_lane, lats, lons):
    long, lane = paired_lane
    cell = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                          long["channel"].to_numpy(), lats, lons)
    every = set(long["site"])
    folds = lanes.site_folds(long, K_FOLDS, 0)
    per_fold = [representativeness_variance(long, cell, sites=every - set(f.tolist()))
                for f in folds]
    got = lane.meta["rep_var_by_fold"]
    for name in ("mtco", "mtwa"):
        assert got[name] == pytest.approx([pf[name] for pf in per_fold])
    # The estimate genuinely varies, so the check is not trivially the global value.
    assert len({round(pf["mtco"], 9) for pf in per_fold}) > 1
    for batch in lane.batches:
        expect = np.array([per_fold[batch.fold][v] for v in ("mtco", "mtwa")])
        assert np.allclose(batch.assim.rep, expect[batch.assim.gather // (len(lats) * len(lons))])


def test_the_references_do_not_depend_on_the_arm(paired_lane):
    _, lane = paired_lane

    def references(model):
        rows, _ = lanes.score_withholding(lane, _Constant(lane.shape), B_SCALES, {}, model)
        df = pd.DataFrame(rows)
        return df[df["method"].isin(["nearest", "idw"])].reset_index(drop=True)

    pd.testing.assert_frame_equal(references(ARMS[0]), references(FULL_MODEL))


def test_the_run_driver_reports_every_amplitude_and_selects_nothing(tmp_path, cube, ages, lats,
                                                                     lons, valid, varying):
    cfg = ex.run_withholding(cube, ages, lats, lons, valid, varying, tmp_path,
                             estimator="3dvar", taper=TAPER, b_scales=B_SCALES)
    M = ex.read_metrics(tmp_path)
    assert "split" not in M.columns and "b_scale" not in cfg
    own = M[M["method"] == "3dvar"]
    assert set(own["b_scale"]) == set(B_SCALES)
    assert set(own["metric"]) == {"ce", "rrmse", "amplitude", "crps", "rcrv_dispersion"}
    assert cfg["observation_model"] == {"spatial": True, "temporal": True}
    with np.load(tmp_path / "withholding_predictions.npz") as z:
        assert z["pred"].shape == z["pred_var"].shape == (len(B_SCALES), len(z["actual"]))
