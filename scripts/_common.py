"""Setup every pipeline script shares: threading, inputs, inherited configs, stages.

Import this first. It caps BLAS threading before numpy is loaded, which nothing can undo
once numpy is in the process, and puts the repository root on the path so a script runs
from any working directory.

``--smoke`` shrinks the archive and every grid, and redirects the outputs to
``outputs/_smoke``, so the whole pipeline can be exercised in minutes without touching a
real artefact.
"""

from __future__ import annotations

import os

# The assimilation runs thousands of small eigendecompositions, a size where splitting
# each one across cores costs more than it saves. Must precede every numpy import.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json          # noqa: E402
import shutil        # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from pathlib import Path   # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paleoreco import paths                          # noqa: E402
from paleoreco.assim import experiments as ex        # noqa: E402
from paleoreco.data import build_prior_cube          # noqa: E402
from paleoreco.assim.observations import (           # noqa: E402
    attach_site_stats, collapse_to_samples, load_observations, observation_site_stats,
)

# Ages a reduced run keeps. Long enough for the time-series lane to leave a run of covered
# ages after its pollen-free young end, short enough to finish in seconds.
SMOKE_AGES = 160
# Every grid at a reduced run. The amplitudes are the product's, so a reduced run's lanes
# carry every amplitude the product is published at.
SMOKE_B = ex.PRODUCT_B_SCALES
SMOKE_TAPER_GRID = dict(localization_grid=(None, 20000.0), shrinkage_grid=(0.0,),
                        alpha_grid=(1.0,))
SMOKE_K = (20,)
SMOKE_W = (0.5, 1.0)
SMOKE_KAPPA = (0.0, 1.0)
SMOKE_SHARE = (0.5, 1.0)


def smoke() -> bool:
    """Whether this process was asked for a reduced run."""
    if "--smoke" not in sys.argv:
        return False
    paths.set_outputs_root(paths.OUTPUTS / "_smoke")
    return True


def load_prior(n_ages: int | None = None):
    """``(cube, ages, lats, lons, valid)`` for the prior archive, optionally truncated."""
    p = build_prior_cube(prior_csv=str(paths.PRIOR_CSV), cache_path=str(paths.PRIOR_CACHE))
    cube, ages = p["cube"], p["ages"]
    if n_ages is not None:
        cube, ages = cube[:n_ages], ages[:n_ages]
    return cube, ages, p["lats"], p["lons"], p["valid"]


def load_networks(ages=None):
    """``(long_ppe, long_wh)``: the proxy table, and the same with site climatology.

    The real-proxy lanes score in anomaly space, so they need ``my`` per site. That
    climatology is taken over one row per sample rather than over the replicated table,
    which would weight a sample by the width of the block it is dated to.
    """
    long_ppe = load_observations(str(paths.OBSERVATION_CSV))
    if ages is not None:
        long_ppe = long_ppe[long_ppe["age"].isin(set(int(a) for a in ages))]
    long_wh = attach_site_stats(long_ppe,
                                observation_site_stats(collapse_to_samples(long_ppe)))
    return long_ppe, long_wh


def operating_point(estimator: str) -> tuple[dict, ex.AnalogPoint | None]:
    """``(taper, analog point)`` an estimator ran at on the snapshot lane.

    Fails loudly rather than defaulting: an inherited operating point that silently fell
    back would run a whole lane at the wrong configuration.
    """
    path = paths.run_dir(estimator, paths.LANE_PPE) / f"{paths.LANE_PPE}_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the stage that writes it first")
    cfg = json.load(open(path))
    return cfg["taper"], ex.point_from_record(cfg["point"])


def clear_dir(directory) -> Path:
    """Empty a run directory before writing it, so nothing from an earlier run survives."""
    shutil.rmtree(Path(directory), ignore_errors=True)
    return Path(directory)


def _requested_stages() -> set[str] | None:
    """Stage names ``--only`` restricted this process to, or ``None`` for all of them."""
    if "--only" not in sys.argv:
        return None
    i = sys.argv.index("--only")
    if i + 1 >= len(sys.argv):
        raise SystemExit("--only needs a stage name, or several separated by commas")
    return {name.strip() for name in sys.argv[i + 1].split(",") if name.strip()}


class Stages:
    """Numbered stage banners with the wall-clock spent so far.

    The lane runners print their own linear-rate ETA inside a loop; this says which loop
    is running and how far into the script it is.

    ``--only`` runs a subset, which is what makes a single lane re-runnable against the
    operating points already on disk. Names are checked against the script's own stages:
    a typo would otherwise run nothing at all and look like a success.
    """

    def __init__(self, script: str, names):
        self.script, self.names = script, list(names)
        self.wanted = _requested_stages()
        unknown = (self.wanted or set()) - set(self.names)
        if unknown:
            raise SystemExit(f"{script} has no stage {sorted(unknown)}; "
                             f"it has {self.names}")
        self.t0 = time.time()
        n = len(self.names)
        print(f"{script}: {n} stage{'' if n == 1 else 's'}", flush=True)
        for i, name in enumerate(self.names, 1):
            mark = "" if self.wanted is None or name in self.wanted else "   (not selected)"
            print(f"    {i}. {name}{mark}", flush=True)

    def run(self, name: str) -> bool:
        """Announce a stage and say whether this process should run it."""
        if self.wanted is not None and name not in self.wanted:
            return False
        i = self.names.index(name) + 1
        print(f"\n[{i}/{len(self.names)}] {name}   (elapsed {self.elapsed()})", flush=True)
        return True

    def done(self) -> None:
        print(f"\n{self.script} finished in {self.elapsed()}", flush=True)

    def elapsed(self) -> str:
        s = int(time.time() - self.t0)
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
