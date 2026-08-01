"""Pull WeatherBench 2's ERA5 2 m temperature at 64x32 into a compact .npz.

The pre-training corpus for the generative prior: 92,044 six-hourly snapshots of
``2m_temperature`` from 1959-01-01 to 2021-12-31 on a 5.625 degree grid, the same
32 x 64 field shape the LOVECLIM prior uses. The store is public, so the whole
transfer is anonymous HTTPS against ``storage.googleapis.com``; no cloud SDK and
no credentials are involved.

Zarr keeps one compressed object per chunk, and this variable is chunked
``(100, 64, 32)`` into 921 of them, so the read is latency-bound and threads
across chunks. Output ``wb2_era5_64x32_t2m_6h.npz`` (~0.6 GB) holds ``t2m`` of
shape ``(92044, n_lat, n_lon)`` in Kelvin, plus the time axis, the grid, and the
two static surface fields.

Two things are deliberately *not* done here, because both are modelling choices
that belong downstream of a faithful copy:

* **Units.** Values stay in Kelvin, as stored.
* **Grid alignment.** Coordinates stay on WeatherBench's own axes (latitude
  -87.1875 to 87.1875, longitude 0 to 354.375). Those are not the LOVECLIM axes:
  the longitudes are the same set rotated by 31 cells (LOVECLIM's -174.375 is
  185.625 = 33 x 5.625), while the latitudes sit half a cell (2.8125 degrees) off
  LOVECLIM's -84.375 to 90.0. Whether that latitude offset is a real half-cell
  shift or only a difference in whether a coordinate labels a cell's centre or
  its northern edge decides between a relabel and a regrid, so the transform is
  left to the caller rather than guessed at here.

The array axes *are* reordered, from WeatherBench's ``(time, longitude,
latitude)`` to the ``(time, lat, lon)`` this package uses everywhere else. That
is a memory-layout convention, not a statement about the grid.

    python -m paleoreco.data.download_wb2
    python -m paleoreco.data.download_wb2 --workers 16 --force
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numcodecs
import numpy as np

BUCKET = "https://storage.googleapis.com/weatherbench2"
STORE = "datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
VAR = "2m_temperature"
# Free (a few KB each) and needed by anything that wants a land mask or orography
# as a conditioning channel, so they come along rather than costing a second trip.
STATIC = ("land_sea_mask", "geopotential_at_surface")
# Repo root is three levels up: download_wb2.py -> data -> paleoreco -> root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_NAME = "wb2_era5_64x32_t2m_6h.npz"
# Physical bracket for a 2 m temperature field in K. A decode landing outside it is
# a corrupt or misinterpreted chunk, not weather; ERA5 spans roughly 180 to 330 K.
T2M_RANGE_K = (150.0, 350.0)
SOURCE = ("WeatherBench 2 (Rasp et al. 2024), ERA5 regridded to 64x32 equiangular "
          "conservative; gs://weatherbench2/" + STORE)


# ---------------------------------------------------------------------------
# Zarr v2 over plain HTTPS.
# ---------------------------------------------------------------------------
def _fetch(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    """One object's bytes, retrying with a linear backoff on transient failures.

    A 404 is not retried: at this layer it means a key that does not exist, which
    no amount of waiting fixes.
    """
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:                      # noqa: BLE001 - network, any of them
            last = e
        if attempt < retries:
            time.sleep(5 * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url} ({last})")


def _metadata(base: str) -> dict:
    """The store's consolidated ``.zmetadata``, one request for every array's schema."""
    return json.loads(_fetch(f"{base}/.zmetadata"))["metadata"]


def _schema(meta: dict, name: str) -> dict:
    """One array's ``.zarray``, checked against the assumptions this reader makes."""
    za = meta[f"{name}/.zarray"]
    if za.get("zarr_format") != 2:
        raise ValueError(f"{name}: expected zarr v2, got format {za.get('zarr_format')}")
    if za.get("order") != "C":
        raise ValueError(f"{name}: expected C order, got {za.get('order')!r}")
    if za.get("filters"):
        raise ValueError(f"{name}: filters are not supported, got {za['filters']!r}")
    return za


def _chunks(shape, chunk_shape):
    """Yield ``(key, destination slices, valid extent)`` over an array's chunk grid.

    Zarr v2 stores edge chunks at full size, padded with the fill value, so a chunk
    decodes to ``chunk_shape`` and only ``valid`` of it belongs in the array.
    """
    counts = [-(-s // c) for s, c in zip(shape, chunk_shape)]
    for idx in itertools.product(*(range(n) for n in counts)):
        dest, valid = [], []
        for i, c, s in zip(idx, chunk_shape, shape):
            stop = min(i * c + c, s)
            dest.append(slice(i * c, stop))
            valid.append(slice(0, stop - i * c))
        yield ".".join(str(i) for i in idx), tuple(dest), tuple(valid)


def _decode(raw: bytes, za: dict) -> np.ndarray:
    """One chunk's bytes to its full ``chunks``-shaped array."""
    buf = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(buf, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def read_array(base: str, meta: dict, name: str) -> np.ndarray:
    """A small array (coordinate or static field), read serially in its native order."""
    za = _schema(meta, name)
    out = np.empty(za["shape"], dtype=np.dtype(za["dtype"]))
    for key, dest, valid in _chunks(za["shape"], za["chunks"]):
        out[dest] = _decode(_fetch(f"{base}/{name}/{key}"), za)[valid]
    return out


def read_field(base: str, meta: dict, name: str, *, workers: int,
               progress_every: int = 50) -> np.ndarray:
    """The big ``(time, longitude, latitude)`` variable, threaded, as ``(time, lat, lon)``.

    Each chunk is transposed as it lands rather than the whole cube at the end, which
    keeps the peak footprint at one copy of the ~0.7 GB array instead of two.
    """
    za = _schema(meta, name)
    dims = meta[f"{name}/.zattrs"]["_ARRAY_DIMENSIONS"]
    if dims != ["time", "longitude", "latitude"]:
        raise ValueError(f"{name}: expected (time, longitude, latitude) dims, got {dims}")
    n_time, n_lon, n_lat = za["shape"]
    if tuple(za["chunks"][1:]) != (n_lon, n_lat):
        # Only the time axis is chunked, so a chunk is a run of whole fields and the
        # destination slice below can ignore the spatial axes. Anything else would
        # write a partial field to a full-field slot.
        raise ValueError(f"{name}: expected whole spatial fields per chunk, got chunks "
                         f"{za['chunks']} for shape {za['shape']}")
    out = np.empty((n_time, n_lat, n_lon), dtype=np.dtype(za["dtype"]))

    plan = list(_chunks(za["shape"], za["chunks"]))
    total = len(plan)
    done, lock, t0 = 0, threading.Lock(), time.time()

    def one(item):
        key, dest, valid = item
        block = _decode(_fetch(f"{base}/{name}/{key}"), za)[valid]
        out[dest[0]] = block.transpose(0, 2, 1)     # (t, lon, lat) -> (t, lat, lon)
        nonlocal done
        with lock:
            done += 1
            if progress_every and (done % progress_every == 0 or done == total):
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                mb = out.nbytes / 2**20 * done / total
                print(f"  {name} {done}/{total} chunks ({100 * done / total:.0f}%, "
                      f"~{mb:.0f} MiB, elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, item) for item in plan]):
            f.result()                              # re-raise the first chunk that failed
    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def _check(t2m: np.ndarray, expected_shape: tuple[int, int, int]) -> None:
    """Fail loud on a cube that is the wrong shape, holey, or not a temperature field.

    A missing chunk would otherwise reach disk as whatever the buffer happened to
    hold, and a silently wrong pre-training corpus is far more expensive than a
    failed download.
    """
    if t2m.shape != expected_shape:
        raise ValueError(f"t2m shape {t2m.shape} != expected {expected_shape}")
    n_nan = int(np.isnan(t2m).sum())
    if n_nan:
        raise ValueError(f"t2m holds {n_nan} NaN values; ERA5 2 m temperature has none, "
                         "so a chunk did not decode")
    lo, hi = float(t2m.min()), float(t2m.max())
    if not (T2M_RANGE_K[0] < lo and hi < T2M_RANGE_K[1]):
        raise ValueError(f"t2m range {lo:.1f}..{hi:.1f} K falls outside "
                         f"{T2M_RANGE_K[0]}..{T2M_RANGE_K[1]} K")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WeatherBench 2 ERA5 64x32 2 m temperature into the data directory."
    )
    parser.add_argument("out", nargs="?", default=str(DATA_DIR / OUT_NAME),
                        help=f"output .npz path (default: data/{OUT_NAME})")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent chunk downloads (default: 8)")
    parser.add_argument("--store", default=STORE, help="zarr store path within the bucket")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the output already exists")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"exists: {out_path} (pass --force to re-download)")
        return

    base = f"{BUCKET}/{args.store}"
    t0 = time.time()
    print(f"store  {base}", flush=True)

    meta = _metadata(base)
    za = _schema(meta, VAR)
    n_time, n_lon, n_lat = za["shape"]
    lats = read_array(base, meta, "latitude")
    lons = read_array(base, meta, "longitude")
    times = read_array(base, meta, "time")
    time_attrs = meta["time/.zattrs"]
    print(f"grid   {n_lat} lat x {n_lon} lon | lat {lats[0]:.4f}..{lats[-1]:.4f} "
          f"| lon {lons[0]:.4f}..{lons[-1]:.4f}", flush=True)
    print(f"time   {n_time} steps, {time_attrs['units']}, "
          f"{times[0]} to {times[-1]} ({(times[-1] - times[0]) / 24 / 365.25:.1f} yr)", flush=True)
    print(f"fetch  {VAR} with {args.workers} workers "
          f"(~{n_time * n_lat * n_lon * 4 / 2**30:.2f} GiB decoded)", flush=True)

    t2m = read_field(base, meta, VAR, workers=args.workers)
    _check(t2m, (n_time, n_lat, n_lon))

    statics = {}
    for name in STATIC:
        # Static fields carry the same (longitude, latitude) order as the variable.
        statics[name] = read_array(base, meta, name).T.copy()

    payload = {
        "t2m": t2m, "time": times, "lats": lats, "lons": lons,
        "time_units": time_attrs["units"], "calendar": time_attrs["calendar"],
        "units": meta[f"{VAR}/.zattrs"]["units"],
        "store": args.store, "source": SOURCE,
        "note": ("axes (time, lat, lon); units K; grid 5.625 deg on WeatherBench's own "
                 "axes (lat -87.1875..87.1875, lon 0..354.375), NOT rotated or shifted "
                 "onto the LOVECLIM axes"),
        **statics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so an interrupted save cannot leave a
    # truncated file that a later run would mistake for a finished download.
    # savez_compressed appends ".npz" to a path that lacks it, which would rename a
    # file that is not the one just written; an open handle is taken verbatim.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out_path)

    size = out_path.stat().st_size / 2**30
    print(f"\nsaved {out_path}: t2m {t2m.shape} {t2m.dtype} "
          f"[{t2m.min():.1f}..{t2m.max():.1f} K] | {size:.2f} GiB | "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
