"""Proxy observation network: loading, per-site statistics, per-age slices, error model.

The observation table (Liu et al. 2026 pollen reconstructions) carries one
row per ``(site, age)`` with both temperature channels. This module melts it
to long format (one row per ``(site, channel, age)``) and exposes the pieces
data assimilation needs: the per-site climatology observations are anomalised
about, the representativeness variance of the network from co-cell proxy
pairs, each sample's own position on the age axis, the error that position
implies when the sample is used at some other age, and the set of observations
active at a single age.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Channel -> (value column, error column) in the raw observation CSV.
# ``sse_*`` is Liu et al.'s bootstrapped standard error in degC (fxTWAPLS::sse.sample
# returns the square root of the bootstrap variance), so it is squared on load and the
# ``sse`` column below carries a variance, which is what every consumer treats it as.
_OBS_COLS: dict[str, tuple[str, str]] = {
    "mtco": ("mtco", "sse_mtco"),
    "mtwa": ("mtwa", "sse_mtwa"),
}


@dataclass(frozen=True)
class ObservationModel:
    """Which error terms R carries beyond each proxy's stated error ``sse``.

    ``spatial`` adds the representativeness variance of a point proxy standing in for its
    grid cell. ``temporal`` treats a sample used away from its own block centre with
    :func:`deflate`.
    """

    spatial: bool
    temporal: bool

    @property
    def name(self) -> str:
        """``sse`` followed by the terms switched on, e.g. ``sse_spatial_temporal``."""
        terms = [t for t, on in (("spatial", self.spatial), ("temporal", self.temporal)) if on]
        return "_".join(["sse", *terms])


FULL_MODEL = ObservationModel(spatial=True, temporal=True)
# The lowest lag correlation a sample is credited with; see :func:`temporal_terms`.
RHO_FLOOR = 0.05


def load_observations(obs_csv: str) -> pd.DataFrame:
    """Long-format proxy table, one row per ``(site, channel, age)``.

    Columns: ``site``, ``sample``, ``channel``, ``age``, ``age_mean``, ``y``,
    ``sse``, ``lat``, ``lon``. A physical sample appears at every age in the
    contiguous block the ``age`` chronology assigns it, with identical
    ``y``/``sse``; ``sample`` and ``age_mean`` are carried so
    :func:`collapse_to_samples` can undo that replication.

    ``age`` and ``age_mean`` are different chronologies for the same samples:
    ``age`` is the scale Liu et al. adjusted by dynamic time warping and is what
    the assimilation runs on, ``age_mean`` the original age-model estimate it
    superseded.
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
            "sse": df[sse_col].to_numpy() ** 2, # sse = sample specific error
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


def sample_block_centres(long: pd.DataFrame) -> pd.DataFrame:
    """``long`` with a ``centre`` column: the midpoint of each sample's age block.

    A sample carries one measurement but is replicated across every age in its
    block, so at most of those ages it describes a climate state some way off in
    time. The midpoint is the sample's own position on the age axis, which is what
    an observation operator needs in order to fetch the state the sample actually
    reports on.
    """
    block = long.groupby(["site", "sample"])["age"].agg(["min", "max"])
    centre = ((block["min"] + block["max"]) / 2.0).rename("centre").reset_index()
    return long.merge(centre, on=["site", "sample"], how="left")


def observation_site_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Per ``(site, channel)`` mean ``my`` of ``y`` across time, the site's climatology."""
    return long.groupby(["site", "channel"])["y"].mean().rename("my").reset_index()


def attach_site_stats(long: pd.DataFrame, site_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge ``my`` onto the long table, keyed by ``(site, channel)``."""
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


def temporal_terms(
    S: np.ndarray, var: np.ndarray, gather: np.ndarray, lag_yr: np.ndarray,
    step_yr: float, *, floor: float = RHO_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Lag correlation and unexplained variance at each observation's cell and lag.

    ``rho = 1 - S / (2 var)`` is how much of the analysis-age state a sample sitting
    ``lag_yr`` away still carries. It is floored rather than allowed negative: past
    the floor the two moments share no signal, and a negative correlation would read
    the observation backwards instead of setting it aside. The residual variance uses
    the floored value, so a fully decorrelated sample carries the whole climatological
    variance as noise.

    A block centre is a midpoint, so it lands on a half step for about half the
    network. The lag index rounds half away from zero; ``np.rint`` rounds half to
    even, which would send a half-step lag of 0.5 to zero and leave those samples
    uncorrected.
    """
    lag = np.floor(np.abs(np.asarray(lag_yr, dtype=np.float64)) / step_yr + 0.5)
    lag = np.clip(lag.astype(np.int64), 0, len(S) - 1)
    g = np.asarray(gather)
    v = var[g]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.clip(1.0 - S[lag, g] / (2.0 * v), floor, 1.0)
    return rho, v * (1.0 - rho ** 2)


def deflate(
    y_anom: np.ndarray, r_diag: np.ndarray, rho: np.ndarray, resid_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Observation and error variance corrected for the sample's distance in time.

    A sample used away from its own moment reports a state that is both noisier and,
    since a lagged state regresses toward the mean, smaller in magnitude than the one
    being solved for. Dividing by ``rho`` undoes the attenuation so the observation
    states its claim at full size, and R carries the cost: it grows as ``1 / rho^2``
    while the observation grows as ``1 / rho``, so the gain collapses and a hopelessly
    stale sample switches itself off.
    """
    y = np.asarray(y_anom, dtype=np.float64)
    r = np.asarray(r_diag, dtype=np.float64)
    return y / rho, (r + resid_var) / rho ** 2


def max_block_lag(long: pd.DataFrame, step_yr: float) -> int:
    """Widest gap in age steps between an age and its own sample's block centre.

    Sizes the structure function so no observation needs a clipped lag. ``long`` must
    carry the ``centre`` column of :func:`sample_block_centres`.
    """
    lag = (long["age"] - long["centre"]).abs().to_numpy(dtype=np.float64)
    return int(np.floor(lag.max() / step_yr + 0.5)) if len(lag) else 0


def observations_at_age(long: pd.DataFrame, age: int) -> dict[str, np.ndarray]:
    """All observation rows at one age, as a column -> array dict.

    Carries through whatever columns ``long`` holds (including ``my`` if
    :func:`attach_site_stats` was applied), so the per-site climatology stays aligned
    row-for-row.
    """
    rows = long[long["age"] == age]
    return {col: rows[col].to_numpy() for col in rows.columns}
