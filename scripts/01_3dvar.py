"""Pixel 3DVar across the three lanes.

Runs first. The regularizer it selects on the same-model lane is inherited by every
analog estimator there, and the one it selects on the real-proxy lane is inherited by
their withholding runs, so the comparison between estimators is a comparison of
estimators rather than of differently regularized covariances.
"""

from __future__ import annotations

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex

# The pixel regularizer: distance taper, shrinkage toward the diagonal, channel coupling.
# The lengthscales are chordal, so they span what the taper does across a 12742 km diameter
# rather than what its nominal support is: 5000 reaches zero inside the sphere, 12500 leaves
# 0.19 between antipodes, and 35000 leaves 0.82, which is nearly the untapered corner.
GRID = dict(
    localization_grid=(None, 5000.0, 8000.0, 12500.0, 20000.0, 25000.0, 35000.0),
    shrinkage_grid=(0.0, 0.25, 0.5, 0.75),
    alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
)
SMOKE_GRID = dict(localization_grid=(None, 20000.0), shrinkage_grid=(0.0,),
                  alpha_grid=(1.0,))
SMOKE_B = (1.0, 5.0)

ESTIMATOR = ex.ESTIMATOR_3DVAR
STAGES = ["ppe taper grid", "withholding taper grid", "trajectory"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("01_3dvar", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    grid = SMOKE_GRID if smoke else GRID
    b_scales = SMOKE_B if smoke else ex.B_SCALES

    d_ppe = paths.run_dir(ESTIMATOR, paths.LANE_PPE)
    if stages.run(STAGES[0]):
        C.clear_dir(d_ppe)
        ex.run_ppe_pixel_grid(cube, ages, lats, lons, valid, long_ppe, str(d_ppe),
                              b_scales=b_scales, progress_every=25, **grid)

    if stages.run(STAGES[1]):
        d_wh = paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING)
        C.clear_dir(d_wh)
        ex.run_withholding_pixel_grid(cube, ages, lats, lons, valid, long_wh, str(d_wh),
                                      b_scales=b_scales, fold_kind="random",
                                      progress_every=1, **grid)

    if stages.run(STAGES[2]):
        d_traj = paths.run_dir(ESTIMATOR, paths.LANE_TRAJECTORY)
        C.clear_dir(d_traj)
        ex.run_trajectory(cube, ages, lats, lons, valid, long_ppe, str(d_traj),
                          b_scales=b_scales, progress_every=25,
                          **C.inherited_taper(d_ppe / "ppe_config.json"))
    stages.done()


if __name__ == "__main__":
    main()
