"""Stages the three analog estimators share.

Each of them grids the analog parameters on the same-model lane, carries that operating
point to the consecutive run and to the real proxies, and answers one question about a
choice the grid holds fixed. Only the parameters differ, so the stages live here.
"""

from __future__ import annotations

import _common as C

from paleoreco.assim import experiments as ex
from paleoreco.assim.hgaoenkf import make_hgaoenkf

ANALOG_KEYS = ("analog_k", "hybrid_w") + ex.TERM_KEYS
K_FOLDS = 5


def ppe_point(ppe_config) -> dict:
    """The analog parameters the same-model grid selected."""
    win = C.read_selected(ppe_config, ANALOG_KEYS)
    return {"k": int(win["analog_k"]), "hybrid_w": float(win["hybrid_w"]),
            **{key: float(win[key]) for key in ex.TERM_KEYS}}


def _terms(point: dict) -> dict:
    return {key: point[key] for key in ex.TERM_KEYS}


def _stack_for(point: dict, stack: dict) -> dict:
    """The flow stack, or nothing where the weight that switches it on is zero.

    The estimator rejects a stack it cannot use, so the weight-zero corner drops it
    rather than failing to build. That corner is the ablation the grid is read for.
    """
    return stack if point["tendency_theta"] > 0.0 else {}


def trajectory(point, *, cube, ages, lats, lons, valid, long_ppe, out_dir, taper,
               selection, estimator, b_scales, stack=None, switches=None,
               progress_every=25):
    """The consecutive run at the operating point the same-model grid chose."""
    stack = stack or {}
    switches = switches or {}
    on = point["tendency_theta"] > 0.0
    C.clear_dir(out_dir)
    return ex.run_trajectory(
        cube, ages, lats, lons, valid, long_ppe, str(out_dir),
        make_method=make_hgaoenkf(cube, ages, lats, lons, k=point["k"],
                                  hybrid_w=point["hybrid_w"], selection=selection,
                                  **_terms(point), **_stack_for(point, stack)),
        estimator=estimator,
        method_cols=ex.analog_cols(point["k"], point["hybrid_w"], **_terms(point),
                                   **(switches if on else {})),
        temporal_modes=ex.TEMPORAL_MODES, b_scales=b_scales,
        progress_every=progress_every, **taper)


def withholding(point, *, cube, ages, lats, lons, valid, long_wh, out_dir, taper,
                selection, estimator=None, b_scales, stack=None, progress_every=1):
    """The real-proxy lane at the same-model operating point, not re-gridded.

    The grid this replaces moved the selection metric by under 0.4% and the reported CE
    by under 0.006, on a lane whose whole spread between estimators is 0.003. Inheriting
    costs about 0.005 CE, uniformly, and buys back the majority of the tuning budget.
    """
    C.clear_dir(out_dir)
    return ex.run_hgaoenkf_withholding_grid(
        cube, ages, lats, lons, valid, long_wh, str(out_dir),
        k_grid=(point["k"],), hybrid_w_grid=(point["hybrid_w"],),
        tendency_theta_grid=(point["tendency_theta"],),
        tendency_lag_yr_grid=(point["tendency_lag_yr"],),
        redundancy_theta_grid=(point["redundancy_theta"],),
        exclude_yr=ex.EXCLUDE_YR, selection=selection, estimator=estimator,
        report_temporal_modes=ex.TEMPORAL_MODES, k_folds=K_FOLDS, b_scales=b_scales,
        progress_every=progress_every, **(_stack_for(point, stack or {})), **taper)


def taper_sweep(point, *, cube, ages, lats, lons, valid, long_ppe, out_dir, taper,
                selection, lengthscales, b_scales, stack=None, switches=None,
                progress_every=100):
    """Score the analog covariance under its own localization lengthscale.

    Sun et al. (2024) Table 2 give the flow-dependent covariance a lengthscale separate
    from the static one, tighter as the ensemble shrinks. ``None`` is the static
    covariance's own, which is what the estimator ships with; the sweep is what says
    whether departing from it would buy anything. It selects nothing: the result is read
    as a number, not fed back.
    """
    stack = stack or {}
    switches = switches or {}
    on = point["tendency_theta"] > 0.0
    C.clear_dir(out_dir)
    for i, km in enumerate(lengthscales, 1):
        print(f"  lengthscale {i}/{len(lengthscales)}: "
              f"{'static' if km is None else f'{int(km)} km'}", flush=True)
        ex.run_ppe(
            cube, ages, lats, lons, valid, long_ppe, str(out_dir),
            make_method=make_hgaoenkf(cube, ages, lats, lons, k=point["k"],
                                      hybrid_w=point["hybrid_w"], selection=selection,
                                      analog_localization_km=km,
                                      **_terms(point), **_stack_for(point, stack)),
            estimator=ex.analog_localization_estimator(selection, km),
            method_cols=ex.analog_cols(point["k"], point["hybrid_w"], **_terms(point),
                                       **(switches if on else {})),
            b_scales=b_scales, progress_every=progress_every, **taper)
