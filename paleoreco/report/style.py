"""House style for the report figures: page width, rcParams, palette and saving."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from paleoreco import paths
from paleoreco.assim import experiments as ex

# Figures are drawn at their printed size, the text width of A4 at 2.5 cm margins in inches,
# so labels print at the point sizes below rather than being scaled down.
REPORT_WIDTH = 6.3

RC = {
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
}

# Schematic roles as (face, edge): the archive-wide static path, the data-selected
# flow-dependent path, the observations, the analysis, and what this project adds.
STATIC = ("#dce6f2", "#3c5f8f")
FLOW = ("#d9e8d3", "#4a7a3f")
DATA = ("#ededed", "#6b6b6b")
ANALYSIS = ("#ffffff", "#333333")
OURS = ("#f7d9c4", "#c4692a")
CONNECT = "#7f7f7f"

# One colour and marker per estimator. Orange is the project's own method throughout, and the
# product and the simulation it was built on reuse the orange and the archive-wide blue.
ESTIMATORS = (
    (ex.ESTIMATOR_3DVAR, "3DVar", "#6b6b6b", "o"),
    (ex.ESTIMATOR_HGAOENKF, "HGAOEnKF", "#3c5f8f", "s"),
    (ex.ESTIMATOR_HGAOENKF_MT, "MTA-HGAOEnKF", "#c4692a", "D"),
)
PRODUCT_C, SIM_C = "#c4692a", "#3c5f8f"

# Differences in skill: blue where a comparator is better, orange where the contribution is.
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "mt_diff", ["#3c5f8f", "#eef1f4", "#f7f2ee", "#c4692a"])


def use_report_style() -> None:
    """Apply the report rcParams to the running process."""
    plt.rcParams.update(RC)


def save(fig, name: str) -> None:
    """Write ``fig`` as ``<name>.png`` among the report figures.

    The figure is left open so a notebook still displays it.
    """
    paths.REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.REPORT_FIGURES / f"{name}.png")


def count_axis(ax, edges, counts):
    """Bin populations as pale bars behind a curve, on their own unlabelled right-hand axis."""
    twin = ax.twinx()
    twin.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
             color="0.90", edgecolor="white", linewidth=0.4, zorder=0)
    twin.set_ylim(0, counts.max() * 3.1)
    twin.set_yticks([])
    twin.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    return twin
