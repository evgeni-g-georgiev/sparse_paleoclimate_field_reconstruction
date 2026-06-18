"""Gaussianity plots for data-assimilation innovation diagnostics.

The marginal test compares pooled standardised innovations to N(0,1)
(histogram and normal QQ). The pairwise test compares whitened component
pairs to N(0,I) (scatter against the reference density contours, plus a
radial chi-squared(2) QQ); both are N(0,I)/N(0,1) under the 3DVar error
assumptions, so departures flag where those assumptions break.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

# Mass quantiles whose N(0,I) iso-density radii are drawn as reference circles,
# coloured along a sequential palette so the nested rings stay distinguishable
# without clashing.
_REF_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99)


def _summary(z: np.ndarray) -> str:
    """One-line moment summary for a panel title."""
    return (
        f"n={z.size}  mean={z.mean():.3f}  sd={z.std():.3f}  "
        f"skew={stats.skew(z):.3f}  exkurt={stats.kurtosis(z):.3f}"
    )


def plot_innovation_gaussianity(
    z_by_channel: dict[str, np.ndarray],
    title: str | None = None,
    bins: int = 80,
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram (vs N(0,1) pdf) and normal QQ plot for each pooled-z group.

    ``z_by_channel`` maps a label (e.g. ``"pooled"``, ``"mtco"``, ``"mtwa"``) to a
    1-D array of standardised innovations. One row per group, two columns.
    """
    labels = list(z_by_channel)
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    grid = np.linspace(-5, 5, 400)

    for row, label in enumerate(labels):
        z = np.asarray(z_by_channel[label], dtype=float)
        z = z[np.isfinite(z)]

        ax_hist = axes[row, 0]
        ax_hist.hist(z, bins=bins, density=True, color="steelblue", alpha=0.7)
        ax_hist.plot(grid, stats.norm.pdf(grid), color="black", lw=1.5, label="N(0,1)")
        ax_hist.set_xlim(-5, 5)
        ax_hist.set_title(f"{label}: {_summary(z)}", fontsize=9)
        ax_hist.set_xlabel("standardised innovation z")
        ax_hist.legend(loc="upper right", fontsize=8)

        ax_qq = axes[row, 1]
        stats.probplot(z, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"{label}: normal QQ", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def _scatter_panel(ax: plt.Axes, z: np.ndarray) -> float:
    """Whitened pair scatter with N(0,I) mass-quantile rings; returns the axis limit."""
    ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.35, color="black", edgecolor="none")

    ring_colors = plt.get_cmap("YlOrBr")(np.linspace(0.45, 0.85, len(_REF_QUANTILES)))
    for q, c in zip(_REF_QUANTILES, ring_colors):
        r = np.sqrt(-2.0 * np.log(1.0 - q))  # iso-density radius holding mass q
        ax.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor=c, lw=1.8))
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    handles = [
        Line2D([0], [0], color=c, lw=1.8, label=f"{int(round(q * 100))}% mass")
        for q, c in zip(_REF_QUANTILES, ring_colors)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7, title="N(0,I)")

    lim = max(4.0, float(np.nanpercentile(np.abs(z), 99.0)) + 0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return lim


def plot_pairwise_gaussianity(
    pairs: dict[str, np.ndarray],
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Whitened-pair scatter (vs N(0,I)) and radial QQ for each component pair.

    ``pairs`` maps a label to an ``(N, 2)`` array of whitened pair innovations.
    One row per pair, two columns: the scatter against the N(0,I) density
    contours, and a QQ of the squared radius against chi-squared(2) (its
    distribution under N(0,I)).
    """
    labels = list(pairs)
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.2 * n), squeeze=False)
    r50 = -2.0 * np.log(0.5)  # squared radius of the 50% mass ring under N(0,I)

    for row, label in enumerate(labels):
        z = np.asarray(pairs[label], dtype=float)
        z = z[np.isfinite(z).all(axis=1)]

        _scatter_panel(axes[row, 0], z)
        r2 = (z ** 2).sum(axis=1)
        frac = float(np.mean(r2 <= r50))
        axes[row, 0].set_title(
            f"{label}: n={len(z)}  inside 50% ring={frac:.2f} (exp 0.50)", fontsize=9
        )
        axes[row, 0].set_xlabel("whitened z1")
        axes[row, 0].set_ylabel("whitened z2")

        stats.probplot(r2, dist=stats.chi2, sparams=(2,), plot=axes[row, 1])
        axes[row, 1].set_title(f"{label}: chi-squared(2) QQ of z1^2+z2^2", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig
