"""MTA-HGAOEnKF: the evidence-rule ensemble with multiscale tendency augmentation.

Selection makes the k analogs agree in the directions the observations constrain, so the
analog covariance is flattest where the gain relies on it most. MTA gives a share ``w`` of
the covariance's whitened observation-space variance to the archive's own first and second
differences at fixed lags. The grid scans ``w`` with the ensemble size, the hybrid weight
and the amplitude, at 3DVar's taper and evidence-only's kappa.

The ladder re-scores the operating point with fewer lags. Each smaller stack is scanned
over ``w`` with ``k``, ``alpha`` and ``kappa`` held, so a row can be read both at the chosen
``w`` and at its own best one; the full stack's scan is the grid itself.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex

ESTIMATOR = ex.ESTIMATOR_MTA
STAGES = ["mta-grid", "snapshot", "time-series", "withholding", "ladder"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("03_mta", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    data = (cube, ages, lats, lons, valid)
    b_scales = C.SMOKE_B if smoke else ex.B_SCALES
    shares = C.SMOKE_SHARE if smoke else ex.TENDENCY_SHARE_GRID
    d_grid = paths.grid_dir(ex.GRID_MTA)

    if stages.run("mta-grid"):
        taper, _ = C.operating_point(ex.ESTIMATOR_3DVAR)
        _, evidence = C.operating_point(ex.ESTIMATOR_EVIDENCE)
        axes = (shares, C.SMOKE_K, C.SMOKE_W) if smoke else (shares, ex.MTA_K_GRID,
                                                             ex.HYBRID_W_GRID)
        points = [(ESTIMATOR, ex.AnalogPoint(k=k, hybrid_w=w, kappa=evidence.kappa,
                                              tendency_share=share, stack=ex.MTA_STACK))
                  for share, k, w in itertools.product(*axes)]
        ex.run_analog_grid(*data, long_ppe, C.clear_dir(d_grid), taper=taper, points=points,
                           b_scales=b_scales)

    if stages.run("snapshot"):
        taper, _ = C.operating_point(ex.ESTIMATOR_3DVAR)
        best = ex.select(ex.read_metrics(d_grid))
        ex.run_snapshot(*data, long_ppe, C.clear_dir(paths.run_dir(ESTIMATOR, paths.LANE_PPE)),
                        estimator=ESTIMATOR, taper=taper,
                        point=ex.point_from_row(best, stack=ex.MTA_STACK), b_scales=b_scales)

    if stages.run("time-series"):
        taper, point = C.operating_point(ESTIMATOR)
        ex.run_trajectory(*data, long_ppe,
                          C.clear_dir(paths.run_dir(ESTIMATOR, paths.LANE_TRAJECTORY)),
                          estimator=ESTIMATOR, taper=taper, point=point, b_scales=b_scales)

    if stages.run("withholding"):
        taper, point = C.operating_point(ESTIMATOR)
        ex.run_withholding(*data, long_wh,
                           C.clear_dir(paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING)),
                           estimator=ESTIMATOR, taper=taper, point=point, b_scales=b_scales)

    if stages.run("ladder"):
        taper, point = C.operating_point(ESTIMATOR)
        points = [(ex.ladder_estimator(name),
                   replace(point, stack=ex.MTA_STACKS[name], tendency_share=share))
                  for name in ex.LADDER_SCANNED for share in shares]
        ex.run_analog_grid(*data, long_ppe, C.clear_dir(paths.ablation_dir(ex.ABLATION_LADDER)),
                           taper=taper, points=points, b_scales=b_scales)
    stages.done()


if __name__ == "__main__":
    main()
