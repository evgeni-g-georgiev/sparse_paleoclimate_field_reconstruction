"""Pull the 2 m air-temperature field (TREFHT) from one Vettoretti et al. 2022
CCSM4 run on ERDA into a compact .npz under the repo data directory.

Each remote file is ~9.6 GB of full CAM4 output; only the single 2-D variable is
read, via HTTP range requests (~14.5 MB each), threaded across months because the
read is latency-bound. Output ``<run>_surfacetemp.npz`` holds TREFHT of shape
(12, n_decades, n_lat, n_lon); :mod:`paleoreco.data.equilibrium` is the loader.

    python -m paleoreco.data.download_ccsm4 <run_dir> [co2_ppmv]
    e.g. python -m paleoreco.data.download_ccsm4 cesmi6gat31rblc170i 170
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

SHARE = "https://sid.erda.dk/share_redirect/Fo2F7YWBmv"
# Time-code suffix per month; constant across runs (month 12 carries the offset code).
CODES = {
    1: "01_211001_998901", 2: "02_211002_998902", 3: "03_211003_998903",
    4: "04_211004_998904", 5: "05_211005_998905", 6: "06_211006_998906",
    7: "07_211007_998907", 8: "08_211008_998908", 9: "09_211009_998909",
    10: "10_211010_998910", 11: "11_211011_998911", 12: "12_210912_998812",
}
VAR = "TREFHT"
# Repo root is three levels up: download_ccsm4.py -> data -> paleoreco -> root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def read_month(run: str, m: int, retries: int = 3):
    url = f"{SHARE}/{run}/atm/hist/ts/{run}_{CODES[m]}_cam2_decclimots.nc"
    for attempt in range(1, retries + 1):
        try:
            fs = fsspec.filesystem("http", block_size=2**20)
            ds = xr.open_dataset(fs.open(url), engine="h5netcdf", decode_times=False)
            field = ds[VAR].values.astype(np.float32)
            coords = (ds["lat"].values.astype(np.float32),
                      ds["lon"].values.astype(np.float32),
                      ds["time"].values.astype(np.float64))
            ds.close()
            print(f"  month {m:02d}: ok {field.shape}", flush=True)
            return m, field, coords
        except Exception as e:
            print(f"  month {m:02d}: attempt {attempt} failed: {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"month {m} failed after {retries} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one CCSM4 run's TREFHT field into the data directory."
    )
    parser.add_argument("run_dir", help="ERDA run directory, e.g. cesmi6gat31rblc170i")
    parser.add_argument("co2_ppmv", nargs="?", type=int, default=-1,
                        help="run CO2 in ppmv, stored as metadata (default: unspecified)")
    args = parser.parse_args()
    run, co2 = args.run_dir, args.co2_ppmv

    t0 = time.time()
    results, lat, lon, time_dec = {}, None, None, None
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(read_month, run, m): m for m in range(1, 13)}
        for f in as_completed(futs):
            m, field, (la, lo, td) = f.result()
            results[m] = field
            lat, lon, time_dec = la, lo, td

    trefht = np.stack([results[m] for m in range(1, 13)], axis=0)  # (12, n_dec, lat, lon)
    out = DATA_DIR / f"{run}_surfacetemp.npz"
    np.savez_compressed(
        out, TREFHT=trefht, lat=lat, lon=lon, time_dec=time_dec,
        month=np.arange(1, 13), run=run, co2_ppmv=co2,
        source="Vettoretti et al. 2022 Nat. Geosci.; ERDA share Fo2F7YWBmv",
        note="monthly decadal-mean climatologies; axes (month, decade, lat, lon); units K; grid T31 (~3.75deg)",
    )
    print(f"\nsaved {out}: TREFHT {trefht.shape} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
