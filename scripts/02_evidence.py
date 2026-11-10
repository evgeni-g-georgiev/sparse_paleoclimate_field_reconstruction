"""HGAOEnKF and the evidence rule: one analog grid, and the estimators it defines.

The grid scans the evidence scale kappa with the ensemble size, the hybrid weight and the
amplitude, at 3DVar's taper. Its ``kappa = 0`` slice is the published HGAOEnKF, whose
R-weighted misfit rule is the evidence rule without the background term; its best point
over ``kappa > 0`` is HGAOEnKF with the evidence rule, evidence-only for short. Both are
then run on the time-series and withholding lanes at their snapshot settings.

The correlation check runs Sun et al.'s correlation rule, pooled and per channel, at
HGAOEnKF's ensemble size and hybrid weight, each selecting its own amplitude.
"""

from __future__ import annotations

import itertools

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex

ESTIMATORS = (ex.ESTIMATOR_HGAOENKF, ex.ESTIMATOR_EVIDENCE)
STAGES = ["evidence-grid", "snapshot", "correlation-check", "time-series", "withholding"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("02_evidence", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    data = (cube, ages, lats, lons, valid)
    b_scales = C.SMOKE_B if smoke else ex.B_SCALES
    d_grid = paths.grid_dir(ex.GRID_EVIDENCE)

    if stages.run("evidence-grid"):
        taper, _ = C.operating_point(ex.ESTIMATOR_3DVAR)
        axes = ((C.SMOKE_KAPPA, C.SMOKE_K, C.SMOKE_W) if smoke
                else (ex.KAPPA_GRID, ex.K_GRID, ex.HYBRID_W_GRID))
        # A kappa = 0 point is HGAOEnKF itself, so its rows carry that estimator's tag.
        points = [(ex.ESTIMATOR_HGAOENKF if kappa == 0.0 else ex.ESTIMATOR_EVIDENCE,
                   ex.AnalogPoint(k=k, hybrid_w=w, kappa=kappa))
                  for kappa, k, w in itertools.product(*axes)]
        ex.run_analog_grid(*data, long_ppe, C.clear_dir(d_grid), taper=taper, points=points,
                           b_scales=b_scales)

    if stages.run("snapshot"):
        taper, _ = C.operating_point(ex.ESTIMATOR_3DVAR)
        grid = ex.read_metrics(d_grid)
        for estimator in ESTIMATORS:
            best = ex.select(grid[grid["method"] == estimator])
            print(f"  {estimator}", flush=True)
            ex.run_snapshot(*data, long_ppe,
                            C.clear_dir(paths.run_dir(estimator, paths.LANE_PPE)),
                            estimator=estimator, taper=taper, point=ex.point_from_row(best),
                            b_scales=b_scales)

    if stages.run("correlation-check"):
        taper, misfit = C.operating_point(ex.ESTIMATOR_HGAOENKF)
        points = [(tag, ex.AnalogPoint(k=misfit.k, hybrid_w=misfit.hybrid_w, selection=rule))
                  for rule, tag in ex.CORRELATION_RULES.items()]
        ex.run_analog_grid(*data, long_ppe,
                           C.clear_dir(paths.ablation_dir(ex.ABLATION_CORRELATION)),
                           taper=taper, points=points, b_scales=b_scales)

    if stages.run("time-series"):
        for estimator in ESTIMATORS:
            taper, point = C.operating_point(estimator)
            print(f"  {estimator}", flush=True)
            ex.run_trajectory(*data, long_ppe,
                              C.clear_dir(paths.run_dir(estimator, paths.LANE_TRAJECTORY)),
                              estimator=estimator, taper=taper, point=point,
                              b_scales=b_scales)

    if stages.run("withholding"):
        for estimator in ESTIMATORS:
            taper, point = C.operating_point(estimator)
            print(f"  {estimator}", flush=True)
            ex.run_withholding(*data, long_wh,
                               C.clear_dir(paths.run_dir(estimator, paths.LANE_WITHHOLDING)),
                               estimator=estimator, taper=taper, point=point,
                               b_scales=b_scales)
    stages.done()


if __name__ == "__main__":
    main()
