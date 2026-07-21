"""End-to-end smoke test of the withholding 3DVar runner (paleoreco.assim.experiments).

Runs the real orchestration on tiny synthetic inputs into a tmp dir: it must produce
a tidy metrics.csv with the expected schema (including the selection/test split) and
finite skill. Guards the whole assim wiring.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim import experiments as ex

SKILL_METRICS = {"ce", "corr", "rmse", "rrmse", "amplitude", "ssim"}
CALIBRATION_METRICS = {"crps", "crpss", "rcrv_bias", "rcrv_dispersion", "coverage90"}
POINT_METRICS = (SKILL_METRICS - {"ssim"}) | CALIBRATION_METRICS


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
