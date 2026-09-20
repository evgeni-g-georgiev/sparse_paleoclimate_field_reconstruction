"""HGAOEnKF: the published hybrid gain analog offline EnKF (Sun et al. 2024).

The baseline the contribution is measured against. It also runs the two readings of Sun
et al.'s own selection rule, which is what licenses ranking analogs by R-weighted misfit
instead, and the sweep over the analog covariance's own lengthscale, which is what
licenses tapering it exactly as the static one is tapered.
"""

from __future__ import annotations

import _common as C
import _analog as A

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.analog import (
    ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN, ANALOG_MISFIT,
)

RULE = ANALOG_MISFIT
ESTIMATOR = ex.hgaoenkf_estimator(RULE)
CORRELATION_RULES = (ANALOG_CORRELATION, ANALOG_CORRELATION_PERCHAN)
SMOKE_B = (1.0, 5.0)
SMOKE_K = (20,)
SMOKE_W = (1.0,)

STAGES = ["ppe analog grid", "correlation-rule demonstration", "analog taper sweep",
          "trajectory", "withholding"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("02_hgaoenkf", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    b_scales = SMOKE_B if smoke else ex.B_SCALES
    k_grid = SMOKE_K if smoke else ex.K_GRID
    w_grid = SMOKE_W if smoke else ex.HYBRID_W_GRID
    lengthscales = (None, 10000.0) if smoke else ex.ANALOG_LOCALIZATION_GRID_PPE

    three = ex.ESTIMATOR_3DVAR
    taper_ppe = C.inherited_taper(paths.run_dir(three, paths.LANE_PPE) / "ppe_config.json")
    taper_wh = C.inherited_taper(
        paths.run_dir(three, paths.LANE_WITHHOLDING) / "withholding_random_config.json")

    d_ppe = paths.run_dir(ESTIMATOR, paths.LANE_PPE)
    if stages.run(STAGES[0]):
        C.clear_dir(d_ppe)
        ex.run_hgaoenkf_ppe_grid(cube, ages, lats, lons, valid, long_ppe, str(d_ppe),
                                 k_grid=k_grid, hybrid_w_grid=w_grid, selection=RULE,
                                 b_scales=b_scales, progress_every=5, **taper_ppe)

    # Read from disk rather than from the stage above, so a later stage can run
    # alone against the operating point already stored.
    point = A.ppe_point(d_ppe / "ppe_config.json")

    if stages.run(STAGES[1]):
        for rule in CORRELATION_RULES:
            d_rule = paths.run_dir(ex.hgaoenkf_estimator(rule), paths.LANE_PPE)
            C.clear_dir(d_rule)
            ex.run_hgaoenkf_ppe_grid(cube, ages, lats, lons, valid, long_ppe, str(d_rule),
                                     k_grid=k_grid, hybrid_w_grid=w_grid, selection=rule,
                                     b_scales=b_scales, progress_every=5, **taper_ppe)

    if stages.run(STAGES[2]):
        A.taper_sweep(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                      long_ppe=long_ppe, out_dir=paths.ablation_dir("analog_taper_misfit"),
                      taper=taper_ppe, selection=RULE, lengthscales=lengthscales,
                      b_scales=b_scales)

    if stages.run(STAGES[3]):
        A.trajectory(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                     long_ppe=long_ppe, out_dir=paths.run_dir(ESTIMATOR, paths.LANE_TRAJECTORY),
                     taper=taper_ppe, selection=RULE, estimator=ESTIMATOR, b_scales=b_scales)

    if stages.run(STAGES[4]):
        A.withholding(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                      long_wh=long_wh, out_dir=paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING),
                      taper=taper_wh, selection=RULE, b_scales=b_scales)
    stages.done()


if __name__ == "__main__":
    main()
