"""Where the inputs live and where generated artefacts go.

The one module that knows the layout, so moving a directory is a single edit rather than
a search across every driver. Inputs resolve against the repository root, which is this
file's parent's parent, so a script runs the same from any working directory.

The outputs root defaults to ``outputs/`` and can be pointed elsewhere with the
``PALEORECO_OUTPUTS`` environment variable, read when this module is imported. A relative
value is taken against the repository root, so ``PALEORECO_OUTPUTS=outputs/_smoke`` means
the same thing from a script and from the notebook.

A run directory is named for what it holds: one per (estimator, lane), one per tuning grid,
one per ablation, and one for the product.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
PRIOR_CSV = DATA / "Prior.csv"
OBSERVATION_CSV = DATA / "Observation.csv"
PRIOR_CACHE = DATA / "cache" / "prior_cube.npz"

OUTPUTS_ENV = "PALEORECO_OUTPUTS"

# Lane directory names.
LANE_PPE = "ppe"
LANE_TRAJECTORY = "trajectory"
LANE_WITHHOLDING = "withholding"


def set_outputs_root(root: str | Path) -> None:
    """Point every output path at ``root``, relative paths taken against the repository.

    A reduced run writes somewhere else entirely rather than into the directories a full
    run owns, so it cannot leave a toy artefact behind that later reads as a real one.
    """
    global OUTPUTS, RUNS, GRIDS, ABLATIONS, PRODUCT, FIGURES, REPORT_FIGURES
    root = Path(root)
    OUTPUTS = root if root.is_absolute() else ROOT / root
    RUNS = OUTPUTS / "runs"
    GRIDS = OUTPUTS / "grids"
    ABLATIONS = OUTPUTS / "ablations"
    PRODUCT = OUTPUTS / "product"
    FIGURES = OUTPUTS / "figures"
    REPORT_FIGURES = FIGURES / "report"


set_outputs_root(os.environ.get(OUTPUTS_ENV, "outputs"))


def run_dir(estimator: str, lane: str) -> Path:
    """Directory for one estimator's artefacts on one lane."""
    return RUNS / estimator / lane


def grid_dir(name: str) -> Path:
    """Directory for one tuning grid's scan."""
    return GRIDS / name


def ablation_dir(name: str) -> Path:
    """Directory for one sensitivity pass, named for the question it answers."""
    return ABLATIONS / name
