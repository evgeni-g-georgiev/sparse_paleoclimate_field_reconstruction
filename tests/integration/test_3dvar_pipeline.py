"""End-to-end smoke test of the withholding 3DVar runner (paleoreco.assim.experiments).

Runs the real orchestration on tiny synthetic inputs into a tmp dir: it must produce
a tidy metrics.csv with the expected schema (including the selection/test split) and
finite skill. Guards the whole assim wiring.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paleoreco.assim import experiments as ex
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.observations import representativeness_variance
from paleoreco.data import VARS

SKILL_METRICS = {"ce", "corr", "rmse", "rrmse", "amplitude", "ssim"}
CALIBRATION_METRICS = {"crps", "crpss", "rcrv_bias", "rcrv_dispersion", "coverage90"}
POINT_METRICS = (SKILL_METRICS - {"ssim"}) | CALIBRATION_METRICS


def _paired_obs(ages, lats, lons) -> pd.DataFrame:
    """Six sites in three shared cells (two per cell), so co-cell pairs exist.

    Cells carry deliberately different scatter amplitudes so the per-fold rep_var
    estimate varies with which sites are dropped, exercising the leakage-clean path.
    """
    nodes = [(lats[1], lons[1]), (lats[5], lons[6]), (lats[9], lons[9])]
    amp_by_cell = [0.5, 2.0, 8.0]
    rows = []
    for sid in range(1, 7):
        ci, pos = (sid - 1) // 2, (sid - 1) % 2
        la, lo = nodes[ci]
        phase = 0.0 if pos == 0 else 1.2
        for t, a in enumerate(ages):
            for chan in ("mtco", "mtwa"):
                amp = amp_by_cell[ci] * (1.7 if chan == "mtwa" else 1.0)
                rows.append({
                    "site": sid, "sample": sid, "channel": chan,
                    "age": int(a), "age_mean": int(a),
                    "lat": float(la), "lon": float(lo),
                    "y": float(amp * np.sin(0.7 * t + phase)), "sse": 0.5, "my": 0.0,
                })
    return pd.DataFrame(rows)


def test_run_withholding_nested_cv_structure(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    out_dir = tmp_path / "wh"
    df = ex.run_withholding(cube, ages, lats, lons, valid, obs_long, str(out_dir),
                            k_folds=3, fold_kind="random",
                            b_scales=(0.5, 1.0), seed=0)
    tdv = df[df["method"] == "3dvar"]
    # Held-out selection split (pooled only) plus a reported test split (per fold + pooled).
    assert set(tdv["split"].unique()) == {"selection", "test"}
    assert set(tdv.loc[tdv["split"] == "selection", "fold"]) == {-1}
    assert set(tdv.loc[tdv["split"] == "test", "fold"]) == {0, 1, 2, -1}
    assert set(df["metric"]).issubset(POINT_METRICS)     # no SSIM without a field truth
    assert CALIBRATION_METRICS.issubset(set(df["metric"]))
    # The prior-free references are what put the 3DVar CE in context, so they must ship.
    assert set(df.loc[df["method"] != "3dvar", "method"]) == {"nearest", "idw"}

    assert (out_dir / "withholding_random_predictions.npz").exists()
    with np.load(out_dir / "withholding_random_predictions.npz") as z:
        assert {"post_var_pred", "prior_var_pred", "sse", "distance_km",
                "naive_nearest", "naive_idw"}.issubset(set(z.files))
        # Calibration needs a spread per (b_scale, target) aligned with the predictions.
        assert z["post_var_pred"].shape == z["climatological_pred"].shape
        assert np.isfinite(z["distance_km"]).all()


def test_withholding_rep_var_is_leakage_clean(tmp_path, cube, ages, lats, lons, valid):
    """rep_var is applied per fold from the assimilated sites, never the full network.

    Uses a co-cell network so the estimate is non-zero and fold-dependent, then checks
    each applied value against the estimator over that fold's assim set.
    """
    long = _paired_obs(ages, lats, lons)
    out_dir = tmp_path / "wh_rep"
    ex.run_withholding(cube, ages, lats, lons, valid, long, str(out_dir),
                       k_folds=3, fold_kind="random", b_scales=(0.5, 1.0), seed=0)

    cfg = json.loads((out_dir / "withholding_random_config.json").read_text())
    assert set(cfg["rep_var_full"]) == set(VARS)

    cell = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                          long["channel"].to_numpy(), lats, lons)
    folds = [set(f.tolist()) for f in ex._site_folds(long, 3, "random", 0)]
    all_sites = set(long["site"].unique().tolist())
    per_fold = [representativeness_variance(long, cell, sites=all_sites - f) for f in folds]

    with np.load(out_dir / "withholding_random_predictions.npz") as z:
        assert z["rep_var"].shape == z["actual"].shape
        assert z["rep_var"].max() > 0            # co-cell fixture yields a real estimate
        for c, name in enumerate(VARS):
            applied = np.unique(np.round(z["rep_var"][z["channel"] == c], 9))
            expected = {round(pf.get(name, 0.0), 9) for pf in per_fold}
            assert set(applied).issubset(expected)     # each value is a per-fold estimate
    # The per-fold estimate genuinely varies (so the subset check is not trivially the
    # single global value), which is what leakage-cleanliness requires.
    assert len({round(pf["mtco"], 9) for pf in per_fold}) > 1
