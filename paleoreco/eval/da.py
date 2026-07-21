"""Skill metrics for data-assimilation reconstructions.

Metrics are method-agnostic: they take numpy arrays of truth, reconstruction, and
(for the uncertainty maps) posterior variance, so 3DVar, EnKF, and a generative
posterior score identically. Reconstructions are scored over the valid field,
unweighted by cell area, so the high-latitude variance carrying the D-O signal keeps
its weight.

Skill: coefficient of efficiency (CE, the headline, with a built-in climatology
baseline), Pearson correlation, degC RMSE, RMSE normalised by the truth std (RRMSE),
and the amplitude ratio, as pooled scalars, per-cell maps over a stack of truths, and
as a function of distance to the nearest observation. Field reconstructions also carry
a masked structural similarity (SSIM) over the valid grid, and a prior-to-posterior
uncertainty reduction from the variance diagonals.

Whether the stated uncertainty matches these errors is :mod:`paleoreco.eval.calibration`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
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


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the flattened inputs."""
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error over the flattened inputs."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def relative_rmse(truth: np.ndarray, recon: np.ndarray) -> float:
    """RMSE normalised by the truth std (std-NRMSE); nan if the truth is constant.

    Differs from CE off the zero-mean case: CE divides SSE by the zero-reference
    energy, this divides by the truth variance, so a constant slice that CE still
    scores has no defined RRMSE.
    """
    sd = float(np.std(truth))
    if sd == 0.0:
        return float("nan")
    return rmse(truth, recon) / sd


def amplitude_ratio(truth: np.ndarray, recon: np.ndarray) -> float:
    """``std(recon) / std(truth)``: is the reconstruction as variable as the truth?

    Separates the two ways a low CE arises. Below 1 the analysis is shrunk toward the
    background and errs by being too flat; near 1 the amplitude is right and the error is
    in the direction, which only the correlation can improve. Under a Gaussian update the
    CE-optimal ratio is the correlation itself, so a ratio far above it means the
    observations are being trusted past their skill.
    """
    sd = float(np.std(truth))
    if sd == 0.0:
        return float("nan")
    return float(np.std(recon)) / sd


# ---------------------------------------------------------------------------
# Per-cell skill maps over a stack of truths.
# ---------------------------------------------------------------------------
def ce_map(truth_stack: np.ndarray, recon_stack: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-cell CE across the leading (truth) axis. ``ref`` broadcasts over it."""
    sse = np.sum((truth_stack - recon_stack) ** 2, axis=0)
    ss_ref = np.sum((truth_stack - ref) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - sse / ss_ref


def rmse_map(truth_stack: np.ndarray, recon_stack: np.ndarray) -> np.ndarray:
    """Per-cell RMSE across the leading (truth) axis."""
    return np.sqrt(np.mean((truth_stack - recon_stack) ** 2, axis=0))


# ---------------------------------------------------------------------------
# Prior-to-posterior error reduction.
# ---------------------------------------------------------------------------
def uncertainty_reduction(prior_var: np.ndarray, post_var: np.ndarray) -> np.ndarray:
    """Per-cell drop in standard deviation from prior to posterior, 1 - sd_a/sd_b.

    The claimed reduction the analysis delivers: 0 where the observations add
    nothing, 1 where the cell is fully constrained. Reads the variances (diag B,
    diag A) so it needs no truth.
    """
    return 1.0 - np.sqrt(post_var) / np.sqrt(prior_var)


def error_reduction_map(
    truth_stack: np.ndarray, prior_stack: np.ndarray, post_stack: np.ndarray
) -> np.ndarray:
    """Per-cell realized drop in RMSE from background to analysis, 1 - rmse_a/rmse_b.

    The realized counterpart of :func:`uncertainty_reduction`: positive where
    assimilation moved the field closer to the truth, negative where it hurt.
    """
    rmse_prior = rmse_map(truth_stack, prior_stack)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - rmse_map(truth_stack, post_stack) / rmse_prior


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
    """CE and RMSE in distance-to-obs bins, pooled over the flattened inputs.

    All inputs are 1-D and aligned (cell x truth already flattened). Returns bin
    centres, per-bin CE/RMSE, and counts; empty bins yield NaN.
    """
    idx = np.digitize(distance, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    ce = np.full(len(centres), np.nan)
    rms = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    for b in range(len(centres)):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b]:
            ce[b] = coefficient_of_efficiency(truth[sel], recon[sel], ref[sel])
            rms[b] = rmse(truth[sel], recon[sel])
    return {"distance_km": centres, "ce": ce, "rmse": rms, "count": counts}


# ---------------------------------------------------------------------------
# Plotters.
# ---------------------------------------------------------------------------
def plot_skill_map(
    skill: np.ndarray, channels: tuple[str, ...], title: str | None = None,
    cmap: str = "RdYlBu_r", vmin: float | None = None, vmax: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """One map per channel of a per-cell skill field ``(C, n_lat, n_lon)``."""
    n = skill.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 3.2), squeeze=False)
    for c in range(n):
        ax = axes[0, c]
        im = ax.imshow(skill[c], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(channels[c], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_prior_posterior_uncertainty(
    prior_var: np.ndarray, post_var: np.ndarray, valid: np.ndarray,
    channels: tuple[str, ...], title: str | None = None, save_path: str | None = None,
) -> plt.Figure:
    """Per-channel prior std, posterior std, and their reduction, side by side.

    ``prior_var`` and ``post_var`` are ``(C, n_lat, n_lon)`` variance fields
    (diag B, diag A). The two std panels share a colour scale per channel so the
    shrink is read by eye; the reduction panel is :func:`uncertainty_reduction` on
    a fixed 0-to-1 scale. Cells off ``valid`` are blanked.
    """
    reduction = uncertainty_reduction(prior_var, post_var)
    prior_sd, post_sd = np.sqrt(prior_var), np.sqrt(post_var)
    mask = np.broadcast_to(valid, prior_var.shape)

    n = prior_var.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 3.0 * n), squeeze=False)
    for c in range(n):
        vmax = np.nanmax(np.where(mask[c], prior_sd[c], np.nan))
        panels = (
            ("prior std (degC)", prior_sd[c], "viridis", 0.0, vmax),
            ("posterior std (degC)", post_sd[c], "viridis", 0.0, vmax),
            ("reduction", reduction[c], "RdYlBu_r", 0.0, 1.0),
        )
        for col, (name, field, cmap, vmin, vhi) in enumerate(panels):
            ax = axes[c, col]
            im = ax.imshow(np.where(mask[c], field, np.nan), origin="lower",
                           cmap=cmap, vmin=vmin, vmax=vhi, aspect="auto")
            ax.set_title(f"{channels[c]}  {name}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig
