"""HGAOEnKF-MT: the analog ensemble augmented with a multiscale flow stack.

Selection makes the k analogs agree precisely in the directions the observations
constrain, so the covariance is flattest exactly where the gain leans hardest. The
archive's own rate of change never had to agree with anything and restores spread there,
and D-O variability has structure across timescales that a single difference cannot
sample. The stack defines the estimator and is swept separately rather than tuned; the
grid varies its weight, whose zero corner drops it.
"""

from __future__ import annotations

import _common as C
import _analog as A

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.analog import ANALOG_EVIDENCE
from paleoreco.assim.hgaoenkf import make_hgaoenkf

RULE = ANALOG_EVIDENCE
ESTIMATOR = ex.ESTIMATOR_HGAOENKF_MT
STACK = dict(tendency_extra_lags_yr=ex.MT_EXTRA_LAGS_YR,
             tendency_curvature_yr=ex.MT_CURVATURE_YR,
             tendency_normalise=True, preserve_obs_trace=True)
SWITCHES = dict(tendency_normalise=True, preserve_obs_trace=True)
SMOKE_B = (1.0, 5.0)

STAGES = ["ppe analog grid", "trajectory", "withholding", "flow stack sweep"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("04_hgaoenkf_mt", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    long_ppe, long_wh = C.load_networks(ages if smoke else None)
    b_scales = SMOKE_B if smoke else ex.B_SCALES
    k_grid = (20,) if smoke else ex.K_GRID
    w_grid = (1.0,) if smoke else ex.HYBRID_W_GRID
    terms = dict(tendency_theta_grid=(0.0, 2.0) if smoke else ex.MT_THETA_GRID,
                 tendency_lag_yr_grid=(ex.MT_REFERENCE_LAG_YR,),
                 redundancy_theta_grid=ex.REDUNDANCY_THETA_GRID)

    three = ex.ESTIMATOR_3DVAR
    taper_ppe = C.inherited_taper(paths.run_dir(three, paths.LANE_PPE) / "ppe_config.json")
    taper_wh = C.inherited_taper(
        paths.run_dir(three, paths.LANE_WITHHOLDING) / "withholding_random_config.json")

    d_ppe = paths.run_dir(ESTIMATOR, paths.LANE_PPE)
    if stages.run(STAGES[0]):
        C.clear_dir(d_ppe)
        ex.run_hgaoenkf_ppe_grid(cube, ages, lats, lons, valid, long_ppe, str(d_ppe),
                                 k_grid=k_grid, hybrid_w_grid=w_grid, selection=RULE,
                                 estimator=ESTIMATOR, b_scales=b_scales, progress_every=5,
                                 **terms, **STACK, **taper_ppe)

    # Read from disk rather than from the stage above, so a later stage can run alone
    # against the operating point already stored.
    point = A.ppe_point(d_ppe / "ppe_config.json")

    if stages.run(STAGES[1]):
        A.trajectory(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                     long_ppe=long_ppe,
                     out_dir=paths.run_dir(ESTIMATOR, paths.LANE_TRAJECTORY),
                     taper=taper_ppe, selection=RULE, estimator=ESTIMATOR,
                     b_scales=b_scales, stack=STACK, switches=SWITCHES)

    if stages.run(STAGES[2]):
        A.withholding(point, cube=cube, ages=ages, lats=lats, lons=lons, valid=valid,
                      long_wh=long_wh,
                      out_dir=paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING),
                      taper=taper_wh, selection=RULE, estimator=ESTIMATOR,
                      b_scales=b_scales, stack=STACK)

    if point["tendency_theta"] <= 0.0:
        stages.skip(STAGES[3], "the grid selected zero flow weight, so there is no stack")
    elif stages.run(STAGES[3]):
        d_stack = paths.ablation_dir("mt_stack")
        C.clear_dir(d_stack)
        stacks = {"1lag": ex.MT_STACKS["1lag"]} if smoke else ex.MT_STACKS
        for i, (name, (extra, curvature)) in enumerate(stacks.items(), 1):
            print(f"  stack {i}/{len(stacks)}: {name}", flush=True)
            ex.run_ppe(
                cube, ages, lats, lons, valid, long_ppe, str(d_stack),
                make_method=make_hgaoenkf(
                    cube, ages, lats, lons, k=point["k"], hybrid_w=point["hybrid_w"],
                    selection=RULE,
                    **{key: point[key] for key in ex.TERM_KEYS},
                    tendency_extra_lags_yr=extra, tendency_curvature_yr=curvature,
                    tendency_normalise=True, preserve_obs_trace=True),
                estimator=ex.mt_stack_estimator(name),
                method_cols=ex.analog_cols(point["k"], point["hybrid_w"],
                                           **{key: point[key] for key in ex.TERM_KEYS},
                                           **SWITCHES),
                b_scales=b_scales, progress_every=100, **taper_ppe)
    stages.done()


if __name__ == "__main__":
    main()
