"""Long-format metric rows, the schema every lane writes.

Each row is the configuration columns a caller passes as ``base``, then ``channel``
(``pooled``, ``mtco`` or ``mtwa``), ``metric`` and ``value``. Coefficients of efficiency are
taken against a zero anomaly, the climatology.
"""

from __future__ import annotations

import numpy as np

from paleoreco.data import VARS
from paleoreco.eval import calibration, da

# ``None`` pools both channels.
CHANNELS = (("pooled", None),) + tuple((name, c) for c, name in enumerate(VARS))


def _rows(base: dict, channel: str, values: dict) -> list[dict]:
    return [{**base, "channel": channel, "metric": metric, "value": value}
            for metric, value in values.items()]


def _skill(truth: np.ndarray, recon: np.ndarray) -> dict:
    return {"ce": da.coefficient_of_efficiency(truth, recon, np.zeros_like(truth)),
            "rrmse": da.relative_rmse(truth, recon),
            "amplitude": da.amplitude_ratio(truth, recon)}


def _channel_masks(channel: np.ndarray):
    """``(name, mask)`` per entry of :data:`CHANNELS` over a flat channel-index array."""
    for name, c in CHANNELS:
        yield name, (np.ones(len(channel), dtype=bool) if c is None else channel == c)


def field_skill_rows(truth: np.ndarray, recon: np.ndarray, safe_valid: np.ndarray,
                     base: dict) -> list[dict]:
    """CE, rRMSE and amplitude over the valid cells of a ``(T, 2, n_lat, n_lon)`` stack."""
    rows = []
    for name, c in CHANNELS:
        t = truth[:, :, safe_valid] if c is None else truth[:, c, safe_valid]
        r = recon[:, :, safe_valid] if c is None else recon[:, c, safe_valid]
        rows += _rows(base, name, _skill(t.ravel(), r.ravel()))
    return rows


def obs_skill_rows(actual: np.ndarray, pred: np.ndarray, channel: np.ndarray,
                   base: dict) -> list[dict]:
    """CE, rRMSE and amplitude in observation space, pooled and per channel."""
    rows = []
    for name, sel in _channel_masks(channel):
        if sel.sum() >= 2:
            rows += _rows(base, name, _skill(actual[sel], pred[sel]))
    return rows


def calibration_rows(truth: np.ndarray, mean: np.ndarray, var: np.ndarray,
                     channel: np.ndarray, base: dict) -> list[dict]:
    """CRPS and RCRV dispersion over flat, aligned arrays, pooled and per channel.

    ``var`` is the predictive variance: the posterior variance for a noise-free truth, plus
    the observation error terms when the truth is a measurement.
    """
    crps = calibration.crps_gaussian(truth, mean, var)
    rows = []
    for name, sel in _channel_masks(channel):
        if sel.sum() >= 2:
            rows += _rows(base, name, {
                "crps": float(np.mean(crps[sel])),
                "rcrv_dispersion": calibration.rcrv_dispersion(truth[sel], mean[sel], var[sel])})
    return rows


def field_calibration_rows(truth: np.ndarray, recon: np.ndarray, post_var: np.ndarray,
                           safe_valid: np.ndarray, base: dict) -> list[dict]:
    """Calibration rows over the valid cells of a field stack.

    The truth is a model state with no observation error, so the predictive variance is
    the posterior variance alone.
    """
    t = truth[:, :, safe_valid]
    chan = np.broadcast_to(np.arange(len(VARS))[None, :, None], t.shape)
    return calibration_rows(t.ravel(), recon[:, :, safe_valid].ravel(),
                            post_var[:, :, safe_valid].ravel(), chan.ravel(), base)


def ssim_rows(truth: np.ndarray, recon: np.ndarray, safe_valid: np.ndarray,
              base: dict) -> list[dict]:
    """Masked SSIM per channel, averaged over truths; ``pooled`` is the mean of the two.

    ``data_range`` is fixed per channel over the whole truth stack, so per-truth SSIMs are
    comparable.
    """
    per_chan = {}
    for c, name in enumerate(VARS):
        dr = float(truth[:, c, safe_valid].max() - truth[:, c, safe_valid].min())
        per_chan[name] = float(np.mean([da.masked_ssim(t[c], r[c], safe_valid, dr)
                                        for t, r in zip(truth, recon)]))
    rows = _rows(base, "pooled", {"ssim": float(np.mean(list(per_chan.values())))})
    for name, value in per_chan.items():
        rows += _rows(base, name, {"ssim": value})
    return rows


def _median_ce_rows(truth_f: np.ndarray, recon_f: np.ndarray, safe_valid: np.ndarray,
                    trim: int, metric: str, base: dict) -> list[dict]:
    """Median per-cell CE of one filtered truth-recon pair, or nothing if too short.

    A median over cells, since after a wide filter a pooled statistic has few independent
    points.
    """
    n = len(truth_f)
    if n - 2 * trim < 3:
        return []
    t, r = truth_f[trim:n - trim], recon_f[trim:n - trim]
    ce = da.ce_map(t, r, np.zeros_like(t[0]))
    rows = []
    for name, c in CHANNELS:
        cells = ce[:, safe_valid] if c is None else ce[c, safe_valid]
        rows += _rows(base, name, {metric: float(np.nanmedian(cells))})
    return rows


def timescale_rows(truth: np.ndarray, recon: np.ndarray, safe_valid: np.ndarray, base: dict,
                   *, step_yr: float, windows, bands) -> list[dict]:
    """Skill by timescale for a consecutive run of states.

    ``ce_lp{window}`` keeps everything slower than the window; ``ce_bp{a}_{b}`` is the band
    between two windows, the difference of their low-passes.
    """
    needed = set(windows) | {w for band in bands for w in band}
    lp_t = {w: da.lowpass_time(truth, w, step_yr) for w in needed}
    lp_r = {w: da.lowpass_time(recon, w, step_yr) for w in needed}
    rows = []
    for w in windows:
        rows += _median_ce_rows(lp_t[w], lp_r[w], safe_valid, da.timescale_trim(w, step_yr),
                                f"ce_lp{int(w)}", base)
    for lo, hi in bands:
        rows += _median_ce_rows(lp_t[lo] - lp_t[hi], lp_r[lo] - lp_r[hi], safe_valid,
                                da.timescale_trim(hi, step_yr), f"ce_bp{int(lo)}_{int(hi)}",
                                base)
    return rows
