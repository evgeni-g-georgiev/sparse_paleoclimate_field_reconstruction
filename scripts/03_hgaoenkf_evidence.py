"""HGAOEnKF-evidence: the analog ensemble chosen by marginal likelihood.

The published rule scores a candidate's residual against R alone. This one scores it
against ``c H B H' + R``, so selection stops spending its budget on the directions the
update will fix and buys ensemble span with what it saves. It also carries the tendency
term, whose weight the grid sweeps down to zero, so the published estimator is the
grid's own corner rather than a separate run.
"""

from __future__ import annotations

import _common as C
import _analog as A

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.analog import ANALOG_EVIDENCE

RULE = ANALOG_EVIDENCE
ESTIMATOR = ex.hgaoenkf_estimator(RULE)
SMOKE_B = (1.0, 5.0)

STAGES = ["ppe analog grid", "analog taper sweep", "trajectory", "withholding"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("03_hgaoenkf_evidence", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    b_scales = SMOKE_B if smoke else ex.B_SCALES
    k_grid = (20,) if smoke else ex.K_GRID
    w_grid = (1.0,) if smoke else ex.HYBRID_W_GRID
    terms = dict(tendency_theta_grid=(0.0, 1.0) if smoke else ex.TENDENCY_THETA_GRID,
                 tendency_lag_yr_grid=(400.0,) if smoke else ex.TENDENCY_LAG_YR_GRID,
                 redundancy_theta_grid=ex.REDUNDANCY_THETA_GRID)
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
                                 b_scales=b_scales, progress_every=5, **terms, **taper_ppe)

    # Read from disk rather than from the stage above, so a later stage can run
    # alone against the operating point already stored.
    point = A.ppe_point(d_ppe / "ppe_config.json")

    if stages.run(STAGES[1]):
        A.taper_sweep(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                      long_ppe=long_ppe, out_dir=paths.ablation_dir("analog_taper_evidence"),
                      taper=taper_ppe, selection=RULE, lengthscales=lengthscales,
                      b_scales=b_scales)

    if stages.run(STAGES[2]):
        A.trajectory(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                     long_ppe=long_ppe, out_dir=paths.run_dir(ESTIMATOR, paths.LANE_TRAJECTORY),
                     taper=taper_ppe, selection=RULE, estimator=ESTIMATOR, b_scales=b_scales)

    if stages.run(STAGES[3]):
        A.withholding(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                      long_wh=long_wh, out_dir=paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING),
                      taper=taper_wh, selection=RULE, b_scales=b_scales)
    stages.done()


if __name__ == "__main__":
    main()
