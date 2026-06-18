"""Pairwise whitened innovations for the 2D joint Gaussianity diagnostic.

The marginal test standardises each innovation by its own variance; this is
the joint analogue for a component pair. For two components A, B the
innovation pair is whitened by its 2x2 predicted covariance
``Sigma_pair = H B H^T + R`` restricted to the pair, so the result is
``N(0, I)`` under the 3DVar joint-Gaussian assumption and a scatter over time
is a round isotropic blob iff that assumption holds.

Anomaly scoring only: ``H`` is plain nearest-cell selection (scale 1) and the
observation errors are independent, so the off-diagonal of ``Sigma_pair`` is
exactly the prior cross-covariance ``B[gA, gB]`` between the two cells.

``rank_pairs`` orders candidate pairs by the prior correlation ``rho`` so the
2D test probes genuine cross-structure: a near-zero-``rho`` pair whitens to
two independent ``N(0,1)`` and adds nothing over the marginal test. Ranking on
``|rho|`` carries no assumption about the shape of any departure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paleoreco.assim.innovation import nearest_age_index, obs_cell_index


def _component_table(
    long: pd.DataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    safe_flat: np.ndarray,
) -> pd.DataFrame:
    """One row per ``(site, channel)`` on a valid cell: gather index and age set.

    Drops components whose nearest cell is masked, since a masked background
    cell has no usable variance for whitening.
    """
    comp = (
        long.groupby(["site", "channel"], sort=False)
        .agg(lat=("lat", "first"), lon=("lon", "first"))
        .reset_index()
    )
    comp["g"] = obs_cell_index(
        comp["lat"].to_numpy(), comp["lon"].to_numpy(), comp["channel"].to_numpy(), lats, lons
    )
    comp = comp[safe_flat[comp["g"].to_numpy()]].reset_index(drop=True)
    return comp


def rank_pairs(
    long: pd.DataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    B: np.ndarray,
    sigma_x: np.ndarray,
    safe_flat: np.ndarray,
    *,
    min_shared_ages: int = 30,
) -> pd.DataFrame:
    """Candidate component pairs ranked by prior correlation ``|rho|``.

    Pairs are restricted to distinct cells (a same-cell pair has ``rho = 1`` by
    construction) and to at least ``min_shared_ages`` co-observed ages so the
    pooled scatter has enough points to read. Returns columns ``siteA, chanA,
    gA, siteB, chanB, gB, rho, n_shared`` sorted by ``|rho|`` descending.
    """
    comp = _component_table(long, lats, lons, safe_flat)
    g = comp["g"].to_numpy()
    n_comp = len(comp)

    # Presence matrix (component x age); the product counts co-observed ages.
    obs_ages = np.sort(long["age"].unique())
    age_pos = {int(a): i for i, a in enumerate(obs_ages)}
    presence = np.zeros((n_comp, len(obs_ages)), dtype=np.int32)
    age_index = long.groupby(["site", "channel"], sort=False)["age"].apply(
        lambda s: [age_pos[int(a)] for a in s]
    )
    for i, key in enumerate(zip(comp["site"], comp["channel"])):
        presence[i, age_index.loc[key]] = 1
    shared = presence @ presence.T

    # Prior correlation between every cell pair from B and its diagonal std.
    rho = B[np.ix_(g, g)] / np.outer(sigma_x[g], sigma_x[g])

    iu, ju = np.triu_indices(n_comp, k=1)
    keep = (g[iu] != g[ju]) & (shared[iu, ju] >= min_shared_ages)
    iu, ju = iu[keep], ju[keep]

    out = pd.DataFrame({
        "siteA": comp["site"].to_numpy()[iu], "chanA": comp["channel"].to_numpy()[iu],
        "gA": g[iu],
        "siteB": comp["site"].to_numpy()[ju], "chanB": comp["channel"].to_numpy()[ju],
        "gB": g[ju],
        "rho": rho[iu, ju], "n_shared": shared[iu, ju],
    })
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def whitened_pair(
    pairrow: pd.Series,
    *,
    long: pd.DataFrame,
    cube: np.ndarray,
    ages: np.ndarray,
    mean_flat: np.ndarray,
    diagB: np.ndarray,
    B: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Whitened anomaly innovations for one pair, pooled over co-observed ages.

    ``pairrow`` carries ``siteA, chanA, gA, siteB, chanB, gB`` (e.g. a row of
    :func:`rank_pairs`). Returns ``(z, shared_ages)`` where ``z`` is ``(N, 2)``,
    ``N(0, I)`` under the joint-Gaussian assumption. ``shared_ages`` is returned
    so callers can colour points by D-O event without this module depending on
    the split definitions.
    """
    gA, gB = int(pairrow["gA"]), int(pairrow["gB"])
    cols = ["age", "y", "sse", "my"]
    a = long[(long["site"] == pairrow["siteA"]) & (long["channel"] == pairrow["chanA"])][cols]
    b = long[(long["site"] == pairrow["siteB"]) & (long["channel"] == pairrow["chanB"])][cols]
    m = a.merge(b, on="age", suffixes=("A", "B"))
    shared_ages = m["age"].to_numpy()

    # Background anomaly H(x_b) at each cell: the Prior snapshot minus its time
    # mean for per_age, identically zero for the climatological background.
    if kind == "per_age":
        ai = nearest_age_index(shared_ages, ages)
        flat = cube.reshape(len(ages), -1)
        xbA = flat[ai, gA] - mean_flat[gA]
        xbB = flat[ai, gB] - mean_flat[gB]
    elif kind == "climatological":
        xbA = xbB = 0.0
    else:
        raise ValueError(f"unknown background kind {kind!r}; expected 'per_age' or 'climatological'")

    dA = (m["yA"].to_numpy() - m["myA"].to_numpy()) - xbA
    dB = (m["yB"].to_numpy() - m["myB"].to_numpy()) - xbB

    # Sigma_pair: diagonals sigma_x^2 + sse vary with age via sse; the
    # off-diagonal B[gA, gB] is fixed. Closed-form 2x2 Cholesky whitening.
    s11 = diagB[gA] + m["sseA"].to_numpy()
    s22 = diagB[gB] + m["sseB"].to_numpy()
    s21 = B[gA, gB]
    l11 = np.sqrt(s11)
    l21 = s21 / l11
    rem = s22 - l21 ** 2
    if np.any(rem <= 1e-12):
        raise ValueError("degenerate pair: Sigma_pair not positive definite (|rho| -> 1)")
    z1 = dA / l11
    z2 = (dB - l21 * z1) / np.sqrt(rem)
    return np.column_stack([z1, z2]), shared_ages
