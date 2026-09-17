"""Check the raw inputs are present and shaped as the pipeline expects.

Neither file can be fetched automatically: the proxy reconstructions are Liu et al.
(2026) supplementary data and the prior is a LOVECLIM transient run supplied by the
project. This reports what is missing and where it goes.
"""

from __future__ import annotations

import sys

import _common as C

from paleoreco import paths

EXPECTED_GRID = (32, 64)
EXPECTED_CHANNELS = 2


def main() -> int:
    missing = [p for p in (paths.PRIOR_CSV, paths.OBSERVATION_CSV) if not p.exists()]
    if missing:
        print("missing input files:")
        for p in missing:
            print(f"    {p}")
        print("\nPrior.csv is the LOVECLIM transient run in long format "
              "(lon, lat, age, mtco, mtwa).")
        print("Observation.csv is the Liu et al. (2026) fxTWAPLS pollen reconstruction "
              "table.")
        print("Place both under the data directory above, then run this again.")
        return 1

    cube, ages, lats, lons, valid = C.load_prior()
    long_ppe, _ = C.load_networks()
    print(f"prior cube {cube.shape}  ages {ages.min()}..{ages.max()} step "
          f"{ages[1] - ages[0]}  valid cells {int(valid.sum())}")
    print(f"proxy table {len(long_ppe)} rows  {long_ppe['site'].nunique()} sites  "
          f"{long_ppe['sample'].nunique()} samples  "
          f"ages {long_ppe['age'].min()}..{long_ppe['age'].max()}")

    if (len(lats), len(lons)) != EXPECTED_GRID or cube.shape[1] != EXPECTED_CHANNELS:
        print(f"\nunexpected grid: got {(len(lats), len(lons))} and {cube.shape[1]} "
              f"channels, expected {EXPECTED_GRID} and {EXPECTED_CHANNELS}")
        return 1
    print("\ninputs look right")
    return 0


if __name__ == "__main__":
    sys.exit(main())
