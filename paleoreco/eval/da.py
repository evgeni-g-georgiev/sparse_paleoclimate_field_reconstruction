"""Skill metrics for data-assimilation reconstructions.

Metrics take numpy arrays of truth and reconstruction, so every estimator is scored the
same way. Fields are scored over the valid grid, unweighted by cell area, so the
high-latitude variance carrying the D-O signal keeps its weight.

Skill: coefficient of efficiency (CE, the headline, with a built-in climatology
baseline), RMSE normalised by the truth std (rRMSE), and the amplitude ratio, as pooled
scalars, as a per-cell CE map over a stack of truths, and as a function of distance to
the nearest observation. Field reconstructions also carry a masked structural similarity
(SSIM) over the valid grid.

Where the stack of truths is a consecutive time series, the per-cell CE map applied to
low-pass filtered inputs resolves skill by timescale.

Whether the stated uncertainty matches these errors is :mod:`paleoreco.eval.calibration`.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim.priors import great_circle_km_between


# ---------------------------------------------------------------------------
# Pooled scalar skill.
# ---------------------------------------------------------------------------
def coefficient_of_efficiency(truth: np.ndarray, recon: np.ndarray, ref: np.ndarray) -> float:
    """CE = 1 - SSE(recon) / SSE(ref). 1 perfect, 0 = no better than ``ref``."""
    sse = np.sum((truth - recon) ** 2)
    ss_ref = np.sum((truth - ref) ** 2)
    return float(1.0 - sse / ss_ref)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error over the flattened inputs."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def relative_rmse(truth: np.ndarray, recon: np.ndarray) -> float:
    """RMSE normalised by the truth std (std-NRMSE); nan if the truth is constant.

    Differs from CE off the zero-mean case: CE divides SSE by the zero-reference
    energy, this divides by the truth variance, so a constant slice that CE still
    scores has no defined rRMSE.
    """
    sd = float(np.std(truth))
    if sd == 0.0:
        return float("nan")
    return rmse(truth, recon) / sd


def amplitude_ratio(truth: np.ndarray, recon: np.ndarray) -> float:
    """``std(recon) / std(truth)``: is the reconstruction as variable as the truth?

    Separates the two ways a low CE arises. Below 1 the analysis is shrunk toward the
    background and errs by being too flat; near 1 the amplitude is right and the error is
    in the direction. Under a Gaussian update the CE-optimal ratio is the correlation
    itself, so a ratio far above it means the observations are being trusted past their
    skill.
    """
    sd = float(np.std(truth))
    if sd == 0.0:
        return float("nan")
    return float(np.std(recon)) / sd


def ce_map(truth_stack: np.ndarray, recon_stack: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-cell CE across the leading (truth) axis. ``ref`` broadcasts over it."""
    sse = np.sum((truth_stack - recon_stack) ** 2, axis=0)
    ss_ref = np.sum((truth_stack - ref) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - sse / ss_ref


# ---------------------------------------------------------------------------
# Temporal filtering, for skill resolved by timescale.
# ---------------------------------------------------------------------------
def lowpass_time(stack: np.ndarray, window_yr: float, step_yr: float) -> np.ndarray:
    """Centred moving average of ``window_yr`` along the leading (time) axis.

    Keeps variability slower than the window. A band is the difference of two of
    these, which is what separates skill at one timescale from skill inherited
    from a slower component that carries most of the variance.

    The kernel is capped at the length of the series. ``np.convolve`` in ``"same"`` mode
    returns the longer of its two arguments, so a window wider than the series would
    otherwise come back longer than it went in. Whether a window that wide can be
    measured at all is :func:`timescale_trim`'s question, not this one's.
    """
    x = np.asarray(stack, dtype=float)
    k = max(1, min(int(round(window_yr / step_yr)), len(x)))
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"), 0, x)


def timescale_trim(window_yr: float, step_yr: float) -> int:
    """Samples to drop from each end after :func:`lowpass_time`.

    The convolution zero-pads, so the first and last window of the filtered series
    is pulled towards zero and would score as spurious error.
    """
    k = max(1, int(round(window_yr / step_yr)))
    return 0 if k <= 1 else k


# ---------------------------------------------------------------------------
# Structural similarity over the field.
# ---------------------------------------------------------------------------
def masked_ssim(truth: np.ndarray, recon: np.ndarray, valid: np.ndarray,
                data_range: float) -> float:
    """Mean SSIM over ``valid`` cells of one 2-D field pair (Wang 2004 params).

    SSIM is computed densely over the whole grid, then the SSIM map is averaged
    over ``valid`` only. ``data_range`` is passed in rather than inferred so the
    stabilising constants stay on a fixed scale across a stack of fields. Edge
    cells inherit some leakage from masked neighbours, accepted because the masked
    region is the climatically inert poles and ice.
    """
    from skimage.metrics import structural_similarity

    _, smap = structural_similarity(
        truth, recon, data_range=data_range, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False, full=True,
    )
    return float(smap[valid].mean())


# ---------------------------------------------------------------------------
# Skill vs distance to the nearest observation.
# ---------------------------------------------------------------------------
def nearest_obs_distance(
    lats: np.ndarray, lons: np.ndarray, obs_lat: np.ndarray, obs_lon: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) from every grid cell to its nearest observation.

    Returned flat over the lat-major, lon-minor spatial axis.
    """
    lat_cell = np.repeat(lats, len(lons))
    lon_cell = np.tile(lons, len(lats))
    return great_circle_km_between(lat_cell, lon_cell,
                                   np.asarray(obs_lat), np.asarray(obs_lon)).min(axis=1)


def skill_vs_distance(
    truth: np.ndarray, recon: np.ndarray, ref: np.ndarray,
    distance: np.ndarray, edges: np.ndarray,
) -> dict:
    """CE in distance-to-obs bins, pooled over the flattened inputs.

    All inputs are 1-D and aligned (cell x truth already flattened). Returns bin
    centres, per-bin CE and counts; empty bins yield NaN.
    """
    idx = np.digitize(distance, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    ce = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    for b in range(len(centres)):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b]:
            ce[b] = coefficient_of_efficiency(truth[sel], recon[sel], ref[sel])
    return {"distance_km": centres, "ce": ce, "count": counts}
