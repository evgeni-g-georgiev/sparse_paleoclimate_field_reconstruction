"""3DVar: the covariance taper, and 3DVar on every lane.

Runs first. The taper it selects on the snapshot lane is inherited by every analog
estimator on every lane, so the comparison between estimators is a comparison of estimators
rather than of differently regularized covariances.

The observation-model ablation also lives here, carried by 3DVar so no analog parameter is
confounded with it: the withholding lane under the stated error alone and with each of the
spatial and temporal terms. The full model is the withholding run itself.
"""

from __future__ import annotations

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex

ESTIMATOR = ex.ESTIMATOR_3DVAR
STAGES = ["taper-grid", "snapshot", "time-series", "withholding", "obs-model"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("01_3dvar", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    data = (cube, ages, lats, lons, valid)
    b_scales = C.SMOKE_B if smoke else ex.B_SCALES
    d_grid = paths.grid_dir(ex.GRID_TAPER)

    if stages.run("taper-grid"):
        ex.run_taper_grid(*data, long_ppe, C.clear_dir(d_grid), b_scales=b_scales,
                          **(C.SMOKE_TAPER_GRID if smoke else ex.TAPER_GRID))

    if stages.run("snapshot"):
        taper = ex.taper_from_row(ex.select(ex.read_metrics(d_grid)))
        ex.run_snapshot(*data, long_ppe, C.clear_dir(paths.run_dir(ESTIMATOR, paths.LANE_PPE)),
                        estimator=ESTIMATOR, taper=taper, b_scales=b_scales)

    if stages.run("time-series"):
        taper, _ = C.operating_point(ESTIMATOR)
        ex.run_trajectory(*data, long_ppe,
                          C.clear_dir(paths.run_dir(ESTIMATOR, paths.LANE_TRAJECTORY)),
                          estimator=ESTIMATOR, taper=taper, b_scales=b_scales)

    if stages.run("withholding"):
        taper, _ = C.operating_point(ESTIMATOR)
        ex.run_withholding(*data, long_wh,
                           C.clear_dir(paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING)),
                           estimator=ESTIMATOR, taper=taper, b_scales=b_scales)

    if stages.run("obs-model"):
        taper, _ = C.operating_point(ESTIMATOR)
        for model in ex.OBS_MODEL_ARMS:
            print(f"  arm {model.name}", flush=True)
            ex.run_withholding(
                *data, long_wh,
                C.clear_dir(paths.ablation_dir(ex.ABLATION_OBS_MODEL) / model.name),
                estimator=ESTIMATOR, taper=taper, model=model, b_scales=b_scales)
    stages.done()


if __name__ == "__main__":
    main()
