"""Where the inputs live and where generated artefacts go.

The one module that knows the layout, so moving a directory is a single edit rather than
a search across every driver. Everything resolves against the repository root, which is
this file's parent's parent, so a script runs the same from any working directory.

A run directory is named for what it holds rather than for what produced it: one
directory per (estimator, lane), one per ablation, one per product variant.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
PRIOR_CSV = DATA / "Prior.csv"
OBSERVATION_CSV = DATA / "Observation.csv"
PRIOR_CACHE = DATA / "cache" / "prior_cube.npz"

OUTPUTS = ROOT / "outputs"
RUNS = OUTPUTS / "runs"
ABLATIONS = OUTPUTS / "ablations"
PRODUCT = OUTPUTS / "product"
FIGURES = OUTPUTS / "figures"
REPORT_FIGURES = FIGURES / "report"
TABLES = OUTPUTS / "tables"

# Lane directory names. The ``lane`` column inside metrics.csv is written by the runners
# and is not this: the withholding files carry the fold kind, the directory does not.
LANE_PPE = "ppe"
LANE_TRAJECTORY = "trajectory"
LANE_WITHHOLDING = "withholding"


def set_outputs_root(root: str | Path) -> None:
    """Point every output path at ``root``.

    A reduced run writes somewhere else entirely rather than into the directories a full
    run owns, so it cannot leave a toy artefact behind that later reads as a real one.
    """
    global OUTPUTS, RUNS, ABLATIONS, PRODUCT, FIGURES, REPORT_FIGURES, TABLES
    OUTPUTS = Path(root)
    RUNS = OUTPUTS / "runs"
    ABLATIONS = OUTPUTS / "ablations"
    PRODUCT = OUTPUTS / "product"
    FIGURES = OUTPUTS / "figures"
    REPORT_FIGURES = FIGURES / "report"
    TABLES = OUTPUTS / "tables"


def run_dir(estimator: str, lane: str) -> Path:
    """Directory for one estimator's artefacts on one lane."""
    return RUNS / estimator / lane


def ablation_dir(name: str) -> Path:
    """Directory for one sensitivity pass, named for the question it answers."""
    return ABLATIONS / name


def product_dir(variant: str = "main") -> Path:
    """Directory for the reconstruction, or for one of its sensitivity variants."""
    return PRODUCT / variant
