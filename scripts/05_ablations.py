"""What the observation-error budget is made of.

Two questions about R on the real-proxy lane, both carried by 3DVar so no analog
parameter is confounded with them.

The ablation crosses the two terms that inflate R beyond the proxy's own stated error:
the representativeness variance, which says a point proxy does not resolve its grid cell,
and the temporal term, which says a sample dated to a block does not resolve the moment
it is assimilated at. Every arm sweeps ``b_scale``, so the arms can be read both at one
frozen amplitude, which isolates the term, and at each arm's own, which is what would
ship.

The second question is whether the representativeness variance could be resolved per
cell rather than per channel. It cannot, and the table says why: the network reaches a
small fraction of the grid, most of the cells it reaches hold one site, and on the few
that hold two the pair estimate is negative often enough to be noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.observations import TEMPORAL_DEFLATE, TEMPORAL_OFF
from paleoreco.data import VARS

# (tag, representativeness term, staleness treatment)
ARMS = (
    ("sse", False, TEMPORAL_OFF),
    ("sse_rep", True, TEMPORAL_OFF),
    ("sse_temporal", False, TEMPORAL_DEFLATE),
    ("sse_rep_temporal", True, TEMPORAL_DEFLATE),
)
# Covariates a per-cell estimate would have to be predicted from, since the network
# reaches too few cells to carry one directly.
COVARIATES = ("n_sites", "elev_range", "elev_std", "lat_range", "prior_gradient",
              "prior_temporal_std", "n_pairs")
STAGES = ["noise-budget ablation", "per-cell representativeness"]


def _per_cell_table(cube, lats, lons, long) -> pd.DataFrame:
    """Per-cell pair estimates of the representativeness variance, and their covariates.

    ``long`` carries the same per-site climatology the lane subtracts, so the residual
    scatter measured here is the one the pooled estimator pools.
    """
    cell = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                          long["channel"].to_numpy(), lats, lons)
    n_cells = len(lats) * len(lons)
    raw = pd.read_csv(str(paths.OBSERVATION_CSV),
                      usecols=["siteID", "latitude", "longitude", "elevation"])
    raw = raw.drop_duplicates("siteID")

    out = []
    for ci, channel in enumerate(VARS):
        sel = long["channel"].to_numpy() == channel
        sub = long[sel]
        c = cell[sel] % n_cells
        anom = sub["y"].to_numpy(float) - sub["my"].to_numpy(float)
        t = pd.DataFrame({"cell": c, "age": sub["age"].to_numpy(), "a": anom,
                          "a2": anom ** 2, "sse": sub["sse"].to_numpy(float)})
        g = t.groupby(["cell", "age"], sort=False)
        k = g.size().to_numpy(float)
        # Same pairwise identities the pooled estimator uses, kept per cell.
        sum_d2 = k * g["a2"].sum().to_numpy() - g["a"].sum().to_numpy() ** 2
        sum_sse = (k - 1.0) * g["sse"].sum().to_numpy()
        pairs = k * (k - 1.0) / 2.0
        idx = pd.MultiIndex.from_tuples(list(g.groups.keys()), names=["cell", "age"])
        per = pd.DataFrame({"cell": idx.get_level_values(0),
                            "num": 0.5 * sum_d2 - 0.5 * sum_sse, "n_pairs": pairs})[k >= 2]
        agg = per.groupby("cell").sum()
        rep = (agg["num"] / agg["n_pairs"]).rename("rep_var")

        sites = pd.DataFrame({"cell": c, "site": sub["site"].to_numpy()}).drop_duplicates()
        sites = sites.merge(raw, left_on="site", right_on="siteID")
        cov = sites.groupby("cell").agg(
            n_sites=("site", "nunique"),
            elev_range=("elevation", lambda s: s.max() - s.min()),
            elev_std=("elevation", "std"),
            lat_range=("latitude", lambda s: s.max() - s.min()))

        field = cube[:, ci].mean(axis=0)
        gy, gx = np.gradient(field)
        table = cov.join(rep).join(agg["n_pairs"]).dropna(subset=["rep_var"])
        table["prior_gradient"] = np.sqrt(gy ** 2 + gx ** 2).ravel()[table.index]
        table["prior_temporal_std"] = cube[:, ci].std(axis=0).ravel()[table.index]
        table["channel"] = channel
        table["occupied_cells"] = len(np.unique(c))
        out.append(table.reset_index())
    return pd.concat(out, ignore_index=True)


def _covariate_report(table: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation of the per-cell estimate against each candidate covariate."""
    rows = []
    for channel, sub in table.groupby("channel"):
        for name in COVARIATES:
            x, y = sub[name].astype(float), sub["rep_var"].astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            rho, p = stats.spearmanr(x[mask], y[mask]) if mask.sum() > 5 else (np.nan, np.nan)
            rows.append({"channel": channel, "covariate": name, "n_cells": int(mask.sum()),
                         "spearman_rho": rho, "p_value": p,
                         "frac_negative": float((sub["rep_var"] < 0).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("05_ablations", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    _, long_wh = C.load_networks(ages if smoke else None)
    b_scales = (1.0, 5.0) if smoke else ex.B_SCALES
    taper_wh = C.inherited_taper(
        paths.run_dir(ex.ESTIMATOR_3DVAR, paths.LANE_WITHHOLDING)
        / "withholding_random_config.json")

    if stages.run(STAGES[0]):
        d = paths.ablation_dir("rep_var")
        C.clear_dir(d)
        for tag, use_rep, mode in ARMS:
            print(f"  arm {tag}", flush=True)
            ex.run_withholding(cube, ages, lats, lons, valid, long_wh, str(d),
                               estimator=f"{ex.ESTIMATOR_3DVAR}_{tag}",
                               temporal_modes=(mode,), use_rep_var=use_rep,
                               b_scales=b_scales, progress_every=1, **taper_wh)

    if stages.run(STAGES[1]):
        paths.TABLES.mkdir(parents=True, exist_ok=True)
        table = _per_cell_table(cube, lats, lons, long_wh)
        table.to_csv(paths.TABLES / "rep_var_per_cell.csv", index=False)
        report = _covariate_report(table)
        report.to_csv(paths.TABLES / "rep_var_covariates.csv", index=False)
        print(report.to_string(index=False), flush=True)
    stages.done()


if __name__ == "__main__":
    main()
