"""Shared synthetic fixtures for the paleoreco test suite.

Everything here is tiny and built in-memory: no test touches the real
``data/Prior.csv`` / ``Observation.csv`` / npz runs, so the whole suite runs in
seconds and is portable to CI. Fixtures favour deterministic seeded arrays so
numeric assertions are stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# Grid small enough to be fast, but >= 11 cells per axis so the Gaussian-windowed
# SSIM (win_size 11) in the DA integration test has a valid window.
N_AGES = 12
N_LAT = 12
N_LON = 12


@pytest.fixture
def lats() -> np.ndarray:
    return np.linspace(-70.0, 70.0, N_LAT).astype(np.float32)


@pytest.fixture
def lons() -> np.ndarray:
    # 8 cells spanning the seam so longitude wrap is exercised.
    return np.linspace(-180.0, 135.0, N_LON).astype(np.float32)


@pytest.fixture
def ages() -> np.ndarray:
    # 500-yr spacing so per-age neighbour offsets have room to move.
    return (30_000 + 500 * np.arange(N_AGES)).astype(np.int64)


@pytest.fixture
def cube(ages) -> np.ndarray:
    """(N_AGES, 2, N_LAT, N_LON) float32 with genuine per-cell time variance.

    A smooth spatial pattern plus seeded per-age noise, with mtwa offset above
    mtco so the two channels differ.
    """
    rng = np.random.default_rng(0)
    base = np.linspace(-30.0, 10.0, N_LAT * N_LON).reshape(N_LAT, N_LON)
    cube = np.empty((N_AGES, 2, N_LAT, N_LON), dtype=np.float32)
    for k in range(N_AGES):
        noise = rng.normal(0.0, 2.0, size=(N_LAT, N_LON))
        cube[k, 0] = base + noise               # mtco
        cube[k, 1] = base + 12.0 + noise * 0.5  # mtwa, warmer
    return cube


@pytest.fixture
def valid() -> np.ndarray:
    return np.ones((N_LAT, N_LON), dtype=bool)


@pytest.fixture
def obs_long(ages, lats, lons) -> pd.DataFrame:
    """Long pseudo-proxy table for the DA pipeline: one row per (site, channel, age).

    Six sites placed on real grid coordinates, both temperature channels, present
    at every age so every held-out age has a usable network. Columns match what
    ``observations_at_age`` / ``experiments.run_ppe`` read: age, lat, lon, channel,
    sse, plus site/sample/my for the withholding lane.
    """
    site_lat = [lats[1], lats[3], lats[5], lats[6], lats[2], lats[4]]
    site_lon = [lons[1], lons[2], lons[6], lons[4], lons[5], lons[0]]
    rows = []
    for sid, (la, lo) in enumerate(zip(site_lat, site_lon), start=1):
        for a in ages:
            for chan in ("mtco", "mtwa"):
                rows.append({
                    "site": sid, "sample": sid, "channel": chan,
                    "age": int(a), "age_mean": int(a),
                    "lat": float(la), "lon": float(lo),
                    "y": 0.0, "sse": 1.0, "my": 0.0,
                })
    return pd.DataFrame(rows)
