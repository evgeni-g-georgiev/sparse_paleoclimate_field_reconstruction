"""The tuning grids and the run drivers (paleoreco.assim.experiments).

A grid is a mean-only scan; a run is one configuration's sweep followed by the selected
amplitude in full. These guard that the two agree, that what a run records is what the
next stage reads back, and that the estimators the evidence grid defines are its slices.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex
from paleoreco.assim.analog import ANALOG_CORRELATION
from paleoreco.assim.hgaoenkf import TendencyStack

B_SCALES = (0.5, 1.0, 2.0)
TAPER = dict(localization_km=None, shrinkage_lambda=0.0, alpha=1.0)
STACK = TendencyStack(reference=500.0, first=(500.0, 1000.0), second=(500.0,))


@pytest.fixture
def varying(obs_long):
    long = obs_long.assign(y=np.sin(obs_long["age"] / 700.0) + obs_long["site"])
    long["my"] = long.groupby(["site", "channel"])["y"].transform("mean")
    return long


def _data(cube, ages, lats, lons, valid):
    return cube, ages, lats, lons, valid


def _at_taper(rows: pd.DataFrame, taper: dict) -> pd.DataFrame:
    loc = taper["localization_km"]
    m = rows["localization_km"].isna() if loc is None else np.isclose(rows["localization_km"], loc)
    return rows[m & np.isclose(rows["shrinkage_lambda"], taper["shrinkage_lambda"])
                & np.isclose(rows["alpha"], taper["alpha"])]


def test_the_taper_grid_scans_every_taper_and_the_run_repeats_the_winner(
        tmp_path, cube, ages, lats, lons, valid, obs_long):
    data = _data(cube, ages, lats, lons, valid)
    grid = dict(localization_grid=(None, 9000.0), shrinkage_grid=(0.0, 0.5), alpha_grid=(1.0,))
    rows = ex.run_taper_grid(*data, obs_long, tmp_path / "grid", b_scales=B_SCALES, **grid)
    assert len(rows[["localization_km", "shrinkage_lambda", "alpha"]].drop_duplicates()) == 4
    assert set(rows["metric"]) == {"ce", "rrmse", "amplitude"}
    assert json.load(open(tmp_path / "grid" / ex.GRID_CONFIG))["shrinkage_grid"] == [0.0, 0.5]

    best = ex.select(rows)
    taper = ex.taper_from_row(best)
    cfg = ex.run_snapshot(*data, obs_long, tmp_path / "run", estimator="3dvar", taper=taper,
                          b_scales=B_SCALES)
    assert cfg["taper"] == taper and cfg["b_scale"] == float(best["b_scale"])
    assert cfg["selection_rrmse"] == pytest.approx(float(best["value"]))
    # The run's sweep is the grid's own scan of that taper, row for row.
    run = ex.read_metrics(tmp_path / "run")
    keys = ["b_scale", "split", "channel", "metric"]
    swept = run[(run["method"] == "3dvar") & run["metric"].isin(["ce", "rrmse", "amplitude"])]
    pd.testing.assert_series_equal(swept.set_index(keys)["value"].sort_index(),
                                   _at_taper(rows, taper).set_index(keys)["value"].sort_index())
    for name in ("ppe_analysis.npz", "ppe_skill_vs_distance.npz", "ppe_config.json"):
        assert (tmp_path / "run" / name).exists()


def test_the_evidence_grid_s_slices_are_the_estimators_it_defines(
        tmp_path, cube, ages, lats, lons, valid, obs_long):
    data = _data(cube, ages, lats, lons, valid)
    points = [(ex.ESTIMATOR_HGAOENKF if kappa == 0 else ex.ESTIMATOR_EVIDENCE,
               ex.AnalogPoint(k=k, hybrid_w=0.5, kappa=kappa))
              for kappa in (0.0, 1.0) for k in (3, 4)]
    rows = ex.run_analog_grid(*data, obs_long, tmp_path, taper=TAPER, points=points,
                              b_scales=B_SCALES)
    hg = ex.select(rows[rows["method"] == ex.ESTIMATOR_HGAOENKF])
    ev = ex.select(rows[rows["method"] == ex.ESTIMATOR_EVIDENCE])
    assert hg["kappa"] == 0.0 and ev["kappa"] == 1.0
    assert ex.select(rows)["value"] == min(hg["value"], ev["value"])
    assert ex.point_from_row(hg).kappa == 0.0
    recorded = json.load(open(tmp_path / ex.GRID_CONFIG))["points"]
    assert [p["kappa"] for p in recorded] == [0.0, 0.0, 1.0, 1.0]


def test_a_run_records_the_point_the_next_stage_reads_back(
        tmp_path, cube, ages, lats, lons, valid, obs_long, varying):
    data = _data(cube, ages, lats, lons, valid)
    point = ex.AnalogPoint(k=4, hybrid_w=0.75, kappa=1.0, tendency_share=0.5, stack=STACK)
    cfg = ex.run_snapshot(*data, obs_long, tmp_path / "ppe", estimator=ex.ESTIMATOR_MTA,
                          taper=TAPER, point=point, b_scales=B_SCALES)
    assert ex.point_from_record(cfg["point"]) == point
    on_disk = json.load(open(tmp_path / "ppe" / "ppe_config.json"))
    assert ex.point_from_record(on_disk["point"]) == point and on_disk["taper"] == TAPER
    M = ex.read_metrics(tmp_path / "ppe")
    own = M[M["method"] == ex.ESTIMATOR_MTA]
    assert (own["tendency_share"] == 0.5).all() and (own["kappa"] == 1.0).all()

    wh = ex.run_withholding(*data, varying, tmp_path / "wh", estimator=ex.ESTIMATOR_MTA,
                            taper=TAPER, point=point, b_scales=B_SCALES)
    assert ex.point_from_record(wh["point"]) == point


def test_the_correlation_check_runs_at_a_fixed_point(tmp_path, cube, ages, lats, lons, valid,
                                                     obs_long):
    data = _data(cube, ages, lats, lons, valid)
    points = [(tag, ex.AnalogPoint(k=4, hybrid_w=0.5, selection=rule))
              for rule, tag in ex.CORRELATION_RULES.items()]
    rows = ex.run_analog_grid(*data, obs_long, tmp_path, taper=TAPER, points=points,
                              b_scales=B_SCALES)
    assert set(rows["method"]) == set(ex.CORRELATION_RULES.values())
    assert (rows["analog_k"] == 4).all() and (rows["kappa"] == 0.0).all()
    assert ex.point_from_row(ex.select(rows), selection=ANALOG_CORRELATION).k == 4


def test_selection_is_the_argmin_of_the_pooled_selection_rrmse():
    rows = pd.DataFrame(
        [("selection", "pooled", 1.0, 0.6), ("selection", "pooled", 5.0, 0.5),
         ("test", "pooled", 2.0, 0.1), ("selection", "mtco", 2.0, 0.1)],
        columns=["split", "channel", "b_scale", "value"]).assign(metric="rrmse")
    assert ex.select(rows)["b_scale"] == 5.0
    with pytest.raises(ValueError, match="no finite"):
        ex.select(rows[rows["split"] == "test"])
