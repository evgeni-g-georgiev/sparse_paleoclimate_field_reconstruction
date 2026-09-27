"""Drawings that read no data: the method map, the estimator schematics and the selection geometry.

The schematics share one coordinate frame, x in 0-100 and y in roughly 0-13, so a box of a given
height prints at the same physical size in every panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

from paleoreco.report.style import (
    ANALYSIS, CONNECT, DATA, FLOW, OURS, REPORT_WIDTH, STATIC,
)

ARROW = dict(arrowstyle="-|>,head_length=0.45,head_width=0.16", lw=0.85,
             color=CONNECT, shrinkA=0, shrinkB=0)
DASHED = (0, (4, 2.5))

ROW0, ROW1 = 5.4, 8.8           # the main left-to-right flow
MID = (ROW0 + ROW1) / 2
OBS0, OBS1 = 10.6, 12.9         # the observation node, above the flow
MEAN_Y = 0.7                    # the prior-mean path, below it


def stage(ax, x0, x1, y0, y1, text, style, fs=6.5):
    """One labelled box in a flow. Returns its extent for the arrows to attach to."""
    face, edge = style
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, mutation_aspect=0.5,
                                boxstyle="round,pad=0,rounding_size=1.0", facecolor=face,
                                edgecolor=edge, linewidth=0.85, zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
            linespacing=1.32, zorder=4)
    return (x0, x1, y0, y1)


def flow(ax, start, end, **kw):
    """A straight arrow from ``start`` to ``end``."""
    opts = dict(ARROW)
    opts.update(kw)
    ax.annotate("", xy=end, xytext=start, arrowprops=opts, zorder=2)


def route(ax, xs, ys, color=CONNECT, **kw):
    """An elbowed connector, drawn under every box so it can pass behind one."""
    ax.plot(xs, ys, color=color, lw=0.85, zorder=1, solid_capstyle="round", **kw)


def method_family_map():
    """Method families placed by how each forms its prior mean and its prior covariance."""
    # Both axes are ordinal, so a position carries membership rather than a distance.
    mean_axis = [
        "the archive\nclimatology",
        "archive states selected\nby the observations",
        "a forecast from\nthe previous age",
        "a draw from a\ntrained density",
    ]
    cov_axis = [
        "the whole archive",
        "one random draw,\nreused at every age",
        "the selected\nstates alone",
        "the selected states\nand the whole\narchive, blended",
        "a trained density",
    ]
    # HGAOEnKF and MTA-HGAOEnKF straddle their shared column so the arrow between them has room.
    methods = [
        (0.00, 0, "3DVar", "published"),
        (0.00, 1, "Offline EnSRF\npaleoDA", "published"),
        (1.00, 0, "AOEnKF-B", "published"),
        (1.00, 2, "Analog offline EnKF", "published"),
        (0.55, 3, "HGAOEnKF", "published"),
        (1.30, 3, "MTA-HGAOEnKF", "ours"),
        (2.05, 3, "Online paleoDA", "published"),
        (3.00, 4, "Generative DA", "published"),
    ]
    face = {"published": STATIC[0], "ours": OURS[0]}
    edge = {"published": STATIC[1], "ours": OURS[1]}

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.85), constrained_layout=True)
    for y in range(len(cov_axis)):
        ax.axhline(y, color="0.93", lw=0.6, zorder=0)
    for x in range(len(mean_axis)):
        ax.axvline(x, color="0.93", lw=0.6, zorder=0)

    boxes = {}
    for x, y, label, kind in methods:
        boxes[label.split("\n")[0]] = ax.annotate(
            label, (x, y), ha="center", va="center", zorder=3,
            fontsize=6.8, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=face[kind],
                      edgecolor=edge[kind], linewidth=0.8))

    ax.set_xlim(-0.52, 3.52)
    ax.set_ylim(-0.40, 4.40)
    ax.set_xticks(range(len(mean_axis)), mean_axis, fontsize=6.5)
    ax.set_yticks(range(len(cov_axis)), cov_axis, fontsize=6.5, linespacing=1.3)
    ax.set_xlabel("What the prior mean is formed from", fontsize=7.5, labelpad=4)
    ax.set_ylabel("What the prior covariance\nis estimated from", fontsize=7.5, labelpad=4)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
        ax.spines[side].set_linewidth(0.7)

    def box_edges(ann):
        # Measured from the drawn patch, so the arrow meets both borders whatever the font metrics.
        fig.canvas.draw()
        bb = ann.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1, _) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x0, x1

    # A small overlap rather than an exact meeting, so no antialiasing seam shows at either end.
    bite = 0.004
    _, hga_right = box_edges(boxes["HGAOEnKF"])
    mt_left, _ = box_edges(boxes["MTA-HGAOEnKF"])
    ax.annotate("", xy=(mt_left + bite, 3.0), xytext=(hga_right - bite, 3.0), zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.18",
                                lw=0.9, color=edge["ours"], shrinkA=0, shrinkB=0))
    return fig


def estimator_schematic():
    """3DVar above HGAOEnKF, drawn as flows from the archive to the analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 3.2), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.3, 13.2)
        ax.axis("off")

    ax = axes[0]
    ax.text(0, 13.1, "(a)  3DVar", fontsize=7.4, fontweight="bold", va="top")
    archive = stage(ax, 1, 17, ROW0, ROW1, "LOVECLIM\narchive", STATIC)
    cov = stage(ax, 25, 52, ROW0, ROW1,
                "one static covariance $\\mathbf{B}$,\nfrom every archive state", STATIC)
    gain = stage(ax, 59, 77, ROW0, ROW1, "one gain", STATIC)
    out = stage(ax, 84, 100, ROW0, ROW1, "analysis", ANALYSIS)
    stage(ax, 59, 77, OBS0, OBS1, "observations $\\mathbf{y}$, $\\mathbf{R}$", DATA)
    for left, right in ((archive, cov), (cov, gain), (gain, out)):
        flow(ax, (left[1], MID), (right[0], MID))
    flow(ax, (68, OBS0), (68, ROW1))
    route(ax, [9, 9, 92], [ROW0, MEAN_Y, MEAN_Y], ls=DASHED)
    flow(ax, (92, MEAN_Y), (92, ROW0), ls=DASHED)
    ax.text(50, MEAN_Y + 0.55, "prior mean: the archive climatology, zero in anomaly units",
            fontsize=6.3, color="#5c5c5c", ha="center", va="bottom")

    ax = axes[1]
    ax.text(0, 13.1, "(b)  HGAOEnKF", fontsize=7.4, fontweight="bold", va="top")
    archive = stage(ax, 1, 15, ROW0, ROW1, "LOVECLIM\narchive", STATIC)
    select = stage(ax, 19, 36, ROW0, ROW1,
                   "score every state\nagainst $\\mathbf{y}$;\nkeep the best $k$", DATA, fs=6.3)
    cov_flow = stage(ax, 40, 64, 7.5, 9.9,
                     "flow-dependent covariance,\nfrom the $k$ selected states", FLOW, fs=6.3)
    cov_static = stage(ax, 40, 64, 3.5, 5.9,
                       "static covariance $\\mathbf{B}$,\nfrom every archive state", STATIC, fs=6.3)
    gain = stage(ax, 68, 84, ROW0, ROW1, "two gains,\nsummed at $\\alpha$", STATIC)
    out = stage(ax, 87, 100, ROW0, ROW1, "analysis", ANALYSIS)
    stage(ax, 38, 66, OBS0, OBS1, "observations $\\mathbf{y}$, $\\mathbf{R}$", DATA)
    flow(ax, (archive[1], MID), (select[0], MID))
    flow(ax, (44, OBS0), (30, ROW1))
    flow(ax, (60, OBS0), (76, ROW1))
    flow(ax, (select[1], 7.6), (cov_flow[0], 8.7), color=FLOW[1])
    # The static covariance never sees the selection, so its connector leaves the archive
    # before the scoring box rather than passing through it.
    route(ax, [8, 8, 38, 38], [ROW0, 2.0, 2.0, 4.7])
    flow(ax, (38, 4.7), (cov_static[0], 4.7))
    flow(ax, (cov_flow[1], 8.7), (gain[0], 7.9), color=FLOW[1])
    flow(ax, (cov_static[1], 4.7), (gain[0], 6.3))
    flow(ax, (gain[1], MID), (out[0], MID))
    route(ax, [27, 27, 93.5], [ROW0, MEAN_Y, MEAN_Y], color=FLOW[1], ls=DASHED)
    flow(ax, (93.5, MEAN_Y), (93.5, ROW0), color=FLOW[1], ls=DASHED)
    ax.text(60, MEAN_Y + 0.55, "prior mean: the mean of the $k$ selected states",
            fontsize=6.3, color=FLOW[1], ha="center", va="bottom")
    return fig


def mta_schematic():
    """MTA-HGAOEnKF in the frame of :func:`estimator_schematic`, its additions in orange."""
    # A lane for the time-ordering connector sits between the flow and the observations, so
    # the y-range grows and the height grows with it to keep the same scale.
    y_top = 14.3
    obs_lo, obs_hi = 11.7, 14.0
    obs_mid = (obs_lo + obs_hi) / 2
    lane = 10.8
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 1.62 * (y_top + 0.3) / 13.5),
                           constrained_layout=True)
    ax.set_xlim(0, 100); ax.set_ylim(-0.3, y_top); ax.axis("off")
    ax.text(0, y_top - 0.1, "(c)  MTA-HGAOEnKF", fontsize=7.4, fontweight="bold", va="top")

    archive = stage(ax, 1, 12, ROW0, ROW1, "LOVECLIM\narchive", STATIC)
    # Scoring every state and keeping k is inherited, so only the rule's name is orange.
    select = stage(ax, 15, 31, ROW0, ROW1, "", DATA)
    ax.text(23, 8.2, "score every state by", fontsize=6.3, ha="center", va="center", zorder=4)
    ax.text(23, 7.1, "the evidence rule", fontsize=6.3, ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.5", facecolor=OURS[0],
                      edgecolor=OURS[1], linewidth=0.7))
    ax.text(23, 6.0, "keep the best $k$", fontsize=6.3, ha="center", va="center", zorder=4)
    augment = stage(ax, 34, 54, 7.5, 9.9,
                    "add the archive's flow\nat nine timescales", OURS, fs=6.3)
    cov_flow = stage(ax, 57, 71, 7.5, 9.9, "flow-dependent\ncovariance", FLOW, fs=6.3)
    cov_static = stage(ax, 57, 71, 3.5, 5.9, "static covariance $\\mathbf{B}$", STATIC, fs=6.3)
    gain = stage(ax, 74, 88, ROW0, ROW1, "two gains,\nsummed at $\\alpha$", STATIC)
    out = stage(ax, 90, 100, ROW0, ROW1, "analysis", ANALYSIS)
    obs = stage(ax, 44, 74, obs_lo, obs_hi, "observations $\\mathbf{y}$, $\\mathbf{R}$", DATA)

    flow(ax, (archive[1], MID), (select[0], MID))
    # Both observation connectors leave sideways and drop straight down, clear of every box.
    route(ax, [obs[0], 23], [obs_mid, obs_mid])
    flow(ax, (23, obs_mid), (23, ROW1))
    route(ax, [obs[1], 81], [obs_mid, obs_mid])
    flow(ax, (81, obs_mid), (81, ROW1))
    flow(ax, (select[1], 7.6), (augment[0], 8.7), color=FLOW[1])
    flow(ax, (augment[1], 8.7), (cov_flow[0], 8.7), color=OURS[1])
    # The augmentation reads the archive's time ordering, not the scores. Its connector has to
    # cross the selection arrow, and a white gap under that arrow draws the crossing as a bridge.
    route(ax, [6.5, 6.5, 44], [ROW1, lane, lane], color=OURS[1], ls=DASHED)
    ax.plot([21.9, 24.1], [lane, lane], color="white", lw=2.5, zorder=1.5,
            solid_capstyle="butt")
    flow(ax, (44, lane), (44, augment[3]), color=OURS[1], ls=DASHED)
    ax.text(33.5, lane + 0.15, "the archive's time ordering", fontsize=6.0, color=OURS[1],
            ha="center", va="bottom")
    # Low enough to stay clear of the static covariance, high enough to clear the label below.
    route(ax, [6.5, 6.5, 55, 55], [ROW0, 2.9, 2.9, 4.7])
    flow(ax, (55, 4.7), (cov_static[0], 4.7))
    flow(ax, (cov_flow[1], 8.7), (gain[0], 7.9), color=FLOW[1])
    flow(ax, (cov_static[1], 4.7), (gain[0], 6.3))
    flow(ax, (gain[1], MID), (out[0], MID))
    route(ax, [23, 23, 95], [ROW0, MEAN_Y, MEAN_Y], color=FLOW[1], ls=DASHED)
    flow(ax, (95, MEAN_Y), (95, ROW0), color=FLOW[1], ls=DASHED)
    ax.text(59, MEAN_Y + 0.45, "prior mean: the mean of the $k$ selected states, unchanged",
            fontsize=6.3, color=FLOW[1], ha="center", va="bottom")
    return fig


def whitened_geometry():
    """The misfit and evidence rules as regions around the observation, in whitened coordinates.

    With ``q = U^T R^-1/2 (y - H x)`` the pool has variance ``lambda`` along each axis, so the
    cloud's elongation is the eigenvalue ratio. The misfit rule keeps a ball, the evidence rule
    an ellipse with semi-axes ``sqrt(s (1 + kappa lambda))``. The ratio drawn is far milder than
    the spectrum the rules meet, so the figure understates the anisotropy.
    """
    lam = np.array([6.0, 0.30])
    kappa, n_drawn, k_drawn = 1.0, 400, 100
    innovation = np.array([0.60, 0.18])       # the prior mean's own whitened misfit
    rng = np.random.default_rng(7)
    q = innovation + rng.normal(size=(n_drawn, 2)) * np.sqrt(lam)

    misfit = (q ** 2).sum(axis=1)
    evidence = (q ** 2 / (kappa * lam + 1.0)).sum(axis=1)
    take_m = misfit <= np.sort(misfit)[k_drawn - 1]
    take_e = evidence <= np.sort(evidence)[k_drawn - 1]
    radius = np.sqrt(np.sort(misfit)[k_drawn - 1])
    semi = np.sqrt(np.sort(evidence)[k_drawn - 1] * (kappa * lam + 1.0))

    fig, ax = plt.subplots(figsize=(4.75, 2.85), constrained_layout=True)
    ax.axhline(0.0, color="0.88", lw=0.5, zorder=0)
    ax.axvline(0.0, color="0.88", lw=0.5, zorder=0)
    # The cloud's own 1- and 2-sigma contours, unlabelled so no leader crosses the regions.
    for n_sd, alpha in ((1.0, 0.6), (2.0, 0.32)):
        ax.add_patch(Ellipse(innovation, *(2 * n_sd * np.sqrt(lam)), fill=False,
                             color="0.55", lw=0.55, ls=(0, (1, 1.6)), alpha=alpha, zorder=1))
    # Colour marks where the rules disagree, grey where they agree.
    classes = ((~take_m & ~take_e, "0.80", 6.5, "kept by neither"),
               (take_m & take_e, "0.38", 8.0, "kept by both"),
               (take_m & ~take_e, "#3c5f8f", 11.0, "misfit only"),
               (take_e & ~take_m, "#c4692a", 11.0, "evidence only"))
    for mask, colour, size, _ in classes:
        ax.scatter(q[mask, 0], q[mask, 1], s=size, color=colour, lw=0, zorder=3)
    ax.add_patch(Circle((0, 0), radius, fill=False, color="#3c5f8f", lw=1.1, zorder=5))
    ax.add_patch(Ellipse((0, 0), *(2 * semi), fill=False, color="#c4692a", lw=1.1, zorder=5))
    ax.plot(0, 0, marker="+", ms=5.5, mew=1.1, color="0.15", zorder=6)

    ax.annotate(r"$\|\mathbf{q}\|^{2}\leq r^{2}$", xy=(0.12, radius), xytext=(-24, 15),
                textcoords="offset points", fontsize=6.4, color="#3c5f8f", ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#3c5f8f", shrinkB=1.5))
    ax.annotate(r"$\sum_p q_p^{2}/(1+\kappa\lambda_p)\leq s$",
                xy=(semi[0] * 0.80, -semi[1] * 0.62), xytext=(14, -24),
                textcoords="offset points", fontsize=6.4, color="#c4692a", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#c4692a", shrinkB=1.5))

    ax.set_aspect("equal")                  # a ball drawn as an oval would misstate the rule
    ax.set_xlim(-2.85, 2.85); ax.set_ylim(-1.68, 1.68)
    ax.set_xlabel(r"$q_1$   (large $\lambda_1$: the prior spreads here)")
    ax.set_ylabel(r"$q_2$   (small $\lambda_2$)")
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=2.8, lw=0, label=lab)
                       for _, c, _, lab in classes],
              loc="lower left", ncol=2, frameon=False, handlelength=1.0, borderpad=0.1,
              labelspacing=0.2, columnspacing=1.0, handletextpad=0.35, fontsize=6.2)
    return fig
