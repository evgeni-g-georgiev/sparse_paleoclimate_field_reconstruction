"""The MIS3 reconstruction, and what it rests on.

The lanes exist to choose an estimator. That choice is made, so this stops measuring and
produces the product: one analysis per age of the whole archive, at the operating point
the same-model lane selected, with the real network at each age.

The variants rerun it with one thing changed each, which is the only uncertainty
statement available where there is no truth to score against.
"""

from __future__ import annotations

import json

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.analog import ANALOG_EVIDENCE
from paleoreco.assim.hgaoenkf import make_hgaoenkf
from paleoreco.assim.reconstruction import run_reconstruction

RULE = ANALOG_EVIDENCE
ESTIMATOR = ex.ESTIMATOR_HGAOENKF_MT
B_SCALES = (1.0, 5.0, 10.0)
STAGES = ["product", "sensitivity variants"]


def variants(wide_exclusion: float, withholding_k: int) -> dict:
    """Each variant is the product with one choice reversed.

    The wide exclusion band scales with the archive: excluding a window wider than the
    archive leaves too few candidates to draw an ensemble from, and the estimator says so
    rather than quietly drawing a smaller one.
    """
    return {
        "no_flow": (dict(tendency_theta=0.0), "the flow stack off"),
        "exclude_0": (dict(exclude_yr=0.0), "no exclusion band around the target age"),
        f"exclude_{int(wide_exclusion)}":
            (dict(exclude_yr=wide_exclusion), f"a {int(wide_exclusion)} yr exclusion band"),
        "withholding_point":
            (dict(k=withholding_k), "the real-proxy lane's ensemble size"),
    }


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("06_product", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    _, long = C.load_networks(ages if smoke else None)
    b_scales = (5.0,) if smoke else B_SCALES

    d_ppe = paths.run_dir(ESTIMATOR, paths.LANE_PPE)
    cfg = json.load(open(d_ppe / "ppe_config.json"))
    taper = {k: cfg["selected"].get(k, cfg[k]) for k in C.TAPER_KEYS}
    win = cfg["selected"]
    terms = {k: float(win[k]) for k in ex.TERM_KEYS}
    stack = dict(tendency_extra_lags_yr=tuple(cfg["tendency_extra_lags_yr"]),
                 tendency_curvature_yr=tuple(cfg["tendency_curvature_yr"]),
                 tendency_normalise=True, preserve_obs_trace=True)

    def method(**over):
        kw = dict(k=int(win["analog_k"]), hybrid_w=float(win["hybrid_w"]), selection=RULE,
                  exclude_yr=ex.EXCLUDE_YR, **terms, **stack)
        kw.update(over)
        # The estimator rejects a stack it cannot use, so switching the weight off drops
        # the stack rather than failing to build.
        if kw["tendency_theta"] <= 0.0:
            for key in ("tendency_extra_lags_yr", "tendency_curvature_yr",
                        "tendency_normalise", "preserve_obs_trace"):
                kw.pop(key, None)
            kw["tendency_lag_yr"] = 0.0
        return make_hgaoenkf(cube, ages, lats, lons, **kw)

    if stages.run(STAGES[0]):
        d_main = paths.product_dir()
        C.clear_dir(d_main)
        run_reconstruction(
            cube, ages, lats, lons, valid, long, str(d_main), make_method=method(),
            b_scales=b_scales, temporal_mode=ex.TEMPORAL_DEFLATE, estimator=ESTIMATOR,
            method_cols=ex.analog_cols(int(win["analog_k"]), float(win["hybrid_w"]), **terms,
                                       tendency_normalise=True, preserve_obs_trace=True),
            progress_every=100, **taper)

    if stages.run(STAGES[1]):
        wh = json.load(open(paths.run_dir(ESTIMATOR, paths.LANE_WITHHOLDING)
                            / "withholding_random_config.json"))
        taper_wh = {k: wh[k] for k in C.TAPER_KEYS}
        # The band the report quotes, narrowed only where the archive is too short to leave
        # an ensemble's worth of candidates outside it.
        span = float(ages[-1] - ages[0])
        wide = min(2000.0, span / 10.0)
        for name, (over, note) in variants(wide, int(wh["selected"]["analog_k"])).items():
            print(f"  variant {name}: {note}", flush=True)
            d_var = paths.product_dir(name)
            C.clear_dir(d_var)
            run_reconstruction(
                cube, ages, lats, lons, valid, long, str(d_var), make_method=method(**over),
                b_scales=b_scales, temporal_mode=ex.TEMPORAL_DEFLATE, estimator=ESTIMATOR,
                progress_every=200,
                **(taper_wh if name == "withholding_point" else taper))
    stages.done()


if __name__ == "__main__":
    main()
