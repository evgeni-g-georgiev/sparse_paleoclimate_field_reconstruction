"""Proxy observation network: loading, per-site statistics, per-age slices.

The observation table (Liu et al. 2026 pollen reconstructions) carries one
row per ``(site, age)`` with both temperature channels. This module melts it
to long format (one row per ``(site, channel, age)``) and exposes the pieces
data assimilation needs: the per-site climatology used for anomaly/normalised
scoring, the representativeness variance of the network from co-cell proxy
pairs, and the set of observations active at a single age.
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


def representativeness_variance(
    long: pd.DataFrame, cell: np.ndarray, *, sites=None
) -> dict[str, float]:
    """Per-channel representativeness variance from co-cell, co-age proxy pairs.

    A point proxy does not measure the mean of its grid cell; ``rep_var`` is the
    residual scatter between two proxies in the same cell at the same age, once each
    proxy's stated error variance is removed:

        rep_var = mean_pairs[ (a_i - a_j)^2 / 2 - (sse_i + sse_j) / 2 ],   a = y - my.

    ``cell`` is the flattened grid-cell index of each ``long`` row (aligned row for
    row). ``sites`` restricts the pairs to a site set, which is what keeps the estimate
    free of a held-out site in the withholding cross-validation. Returns
    ``{channel: rep_var}``, clamped at zero (a channel with no usable pair maps to 0.0).
    """
    anom = long["y"].to_numpy(dtype=np.float64) - long["my"].to_numpy(dtype=np.float64)
    df = pd.DataFrame({
        "cell": np.asarray(cell),
        "age": long["age"].to_numpy(),
        "channel": long["channel"].to_numpy(),
        "site": long["site"].to_numpy(),
        "a": anom,
        "a2": anom ** 2,
        "sse": long["sse"].to_numpy(dtype=np.float64),
    })
    if sites is not None:
        df = df[df["site"].isin(sites)]

    out: dict[str, float] = {}
    for channel, sub in df.groupby("channel", sort=False):
        g = sub.groupby(["cell", "age"], sort=False)
        k = g.size().to_numpy(dtype=np.float64)
        # Per group the pairwise identities sum_{i<j}(a_i - a_j)^2 = k*sum(a^2) - (sum a)^2
        # and sum_{i<j}(sse_i + sse_j) = (k-1)*sum(sse) turn the pooled pair average into
        # four groupby sums, so no O(pairs) loop is needed.
        sum_d2 = k * g["a2"].sum().to_numpy() - g["a"].sum().to_numpy() ** 2
        sum_sse_pair = (k - 1.0) * g["sse"].sum().to_numpy()
        n_pairs = k * (k - 1.0) / 2.0
        keep = k >= 2
        denom = float(n_pairs[keep].sum())
        if denom == 0.0:
            out[channel] = 0.0
            continue
        rep = (0.5 * sum_d2[keep].sum() - 0.5 * sum_sse_pair[keep].sum()) / denom
        out[channel] = max(float(rep), 0.0)
    return out


def observations_at_age(long: pd.DataFrame, age: int) -> dict[str, np.ndarray]:
    """All observation rows at one age, as a column -> array dict.

    Carries through whatever columns ``long`` holds (including ``my``/``sy`` if
    :func:`attach_site_stats` was applied), so downstream scoring has the per-site
    statistics aligned row-for-row.
    """
    rows = long[long["age"] == age]
    return {col: rows[col].to_numpy() for col in rows.columns}
