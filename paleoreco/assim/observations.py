"""Proxy observation network: loading, per-site statistics, per-age slices.

The observation table (Liu et al. 2026 pollen reconstructions) carries one
row per ``(site, age)`` with both temperature channels. This module melts it
to long format (one row per ``(site, channel, age)``) and exposes the pieces
data assimilation needs: the per-site climatology used for anomaly/normalised
scoring, and the set of observations active at a single age.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Channel -> (value column, error-variance column) in the raw observation CSV.
# ``sse_*`` is already a variance (squared standard error), so it enters R directly.
_OBS_COLS: dict[str, tuple[str, str]] = {
    "mtco": ("mtco", "sse_mtco"),
    "mtwa": ("mtwa", "sse_mtwa"),
}


def load_observations(obs_csv: str) -> pd.DataFrame:
    """Long-format proxy table, one row per ``(site, channel, age)``.

    Columns: ``site``, ``sample``, ``channel``, ``age``, ``age_mean``, ``y``,
    ``sse``, ``lat``, ``lon``. A physical sample appears at every age in its
    dating window with identical ``y``/``sse``; ``sample`` and ``age_mean`` are
    carried so :func:`collapse_to_samples` can undo that replication.
    """
    df = pd.read_csv(obs_csv)
    parts = []
    for channel, (val_col, sse_col) in _OBS_COLS.items():
        parts.append(pd.DataFrame({
            "site": df["siteID"].to_numpy(),
            "sample": df["sampleID"].to_numpy(),
            "channel": channel,
            "age": df["age"].to_numpy(),
            "age_mean": df["age_mean"].to_numpy(),
            "y": df[val_col].to_numpy(),
            "sse": df[sse_col].to_numpy(),
            "lat": df["latitude"].to_numpy(),
            "lon": df["longitude"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def collapse_to_samples(long: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(site, channel, sample)``: the age nearest ``age_mean``.

    Each sample is replicated across its dating window with identical values, so
    pooling the raw rows pseudo-replicates wide-dated samples. Keeping the single
    age closest to the sample's central date restores one weight per sample.
    """
    d = (long["age"] - long["age_mean"]).abs()
    return (
        long.assign(_d=d)
        .sort_values("_d")
        .drop_duplicates(["site", "channel", "sample"], keep="first")
        .drop(columns="_d")
        .reset_index(drop=True)
    )


def observation_site_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Per ``(site, channel)`` mean ``my`` and std ``sy`` of ``y`` across time.

    These are the per-site shift and scale for anomaly and normalised scoring.
    ``sy`` uses ``ddof=0`` so a single-observation site gives ``sy = 0`` rather
    than ``NaN``; such sites are unusable under normalised scoring and the caller
    is expected to drop them.
    """
    g = long.groupby(["site", "channel"])["y"]
    return pd.DataFrame({"my": g.mean(), "sy": g.std(ddof=0)}).reset_index()


def attach_site_stats(long: pd.DataFrame, site_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge ``my``/``sy`` onto the long table, keyed by ``(site, channel)``."""
    return long.merge(site_stats, on=["site", "channel"], how="left")


def observations_at_age(long: pd.DataFrame, age: int) -> dict[str, np.ndarray]:
    """All observation rows at one age, as a column -> array dict.

    Carries through whatever columns ``long`` holds (including ``my``/``sy`` if
    :func:`attach_site_stats` was applied), so downstream scoring has the per-site
    statistics aligned row-for-row.
    """
    rows = long[long["age"] == age]
    return {col: rows[col].to_numpy() for col in rows.columns}
