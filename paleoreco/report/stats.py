"""Quantities derived from the archive, the proxies and the stored analyses.

Two resampling tests share one rule: whole units are redrawn with replacement, sites on the
real-proxy lane and truth states on the snapshot lane, because a unit contributes many
correlated rows and a row bootstrap would give an interval far tighter than the data support.
Coefficients of efficiency are taken against a zero anomaly, as the lanes score them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paleoreco.assim import experiments as ex
from paleoreco.assim.background import temporal_structure_function
from paleoreco.assim.ensrf import whitened_block
from paleoreco.assim.hgaoenkf import HGAOEnKF
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.observations import (
    observations_at_age,
    sample_block_centres,
    temporal_terms,
)
from paleoreco.assim.priors import build_prior
from paleoreco.data import VARS
from paleoreco.data.splits import chronological_half_split

# Liu et al. (2026) Sect. 2.4 take each event as 300 yr before its onset to 600 yr after.
# Ages run backwards, so the interstadial half is the younger one.
DO_POST_YR, DO_PRE_YR = 600, 300
# Northern extratropics, the band Liu et al. report their seasonal ratio over.
NET_LAT = 23.5
N_BOOT = 5000
RHO_FLOOR = 0.05                # the floor observations.temporal_terms applies


# ---------------------------------------------------------------------------
# The archive.
# ---------------------------------------------------------------------------

def lag1_autocorr(x):
    """Per-cell lag-1 autocorrelation along the age axis, each half demeaned separately."""
    a, b = x[:-1], x[1:]
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def decorrelation(cube, valid, n_lags=81):
    """``(domain-mean autocorrelation, state-vector correlation)`` against lag in steps.

    The state-vector curve is the mean cosine between whole anomaly states that many steps
    apart, the object a covariance is estimated from; the domain mean is one number per age.
    """
    A = (cube - cube.mean(axis=0, keepdims=True)).reshape(len(cube), -1).astype(np.float64)
    dm = (cube - cube.mean(axis=0, keepdims=True))[:, :, valid].mean(axis=(1, 2))
    dm = dm - dm.mean()
    acf = np.array([1.0 if k == 0 else float((dm[:-k] * dm[k:]).sum() / (dm ** 2).sum())
                    for k in range(n_lags)])
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    G = An @ An.T
    svc = np.array([float(np.mean(np.diagonal(G, k))) for k in range(len(cube))])
    return acf, svc


def leading_pcs(cube, lats, n_pc=4):
    """``(scores, variance share)`` of the archive's leading principal components.

    Cells are weighted by sqrt(cos(lat)) so polar cells do not count as heavily as equatorial
    ones; the cosine is clipped at zero because at exactly +90 degrees it rounds negative. Each
    component's sign is fixed by making its largest loading positive, which stays well defined
    for components that barely correlate with the domain mean.
    """
    anom = (cube - cube.mean(axis=0, keepdims=True)).reshape(len(cube), -1).astype(np.float64)
    lat_w = np.sqrt(np.clip(np.cos(np.radians(lats.astype(np.float64))), 0.0, None))
    cell_w = np.broadcast_to(lat_w[None, :, None],
                             (cube.shape[1], cube.shape[2], cube.shape[3])).ravel()
    U, S, Vt = np.linalg.svd(anom * cell_w, full_matrices=False)
    lead = Vt[:n_pc]
    orient = np.sign(lead[np.arange(n_pc), np.abs(lead).argmax(axis=1)])
    return (U[:, :n_pc] * S[:n_pc]) * orient, S ** 2 / (S ** 2).sum()


def covariance_rank(cube, idx):
    """``(numerical rank, eigenvalues holding 90% and 99% of the trace)`` of a sample B."""
    Ac = cube[idx].reshape(len(idx), -1).astype(np.float64)
    Ac = Ac - Ac.mean(0)
    ev = np.linalg.svd(Ac, compute_uv=False) ** 2 / (len(idx) - 1)
    cum = np.cumsum(ev) / ev.sum()
    return (int((ev > ev[0] * 1e-10).sum()), int(np.searchsorted(cum, 0.90)) + 1,
            int(np.searchsorted(cum, 0.99)) + 1)


def bin_centres(edges):
    """Arithmetic midpoint of each bin; the innermost bin starts at zero, so not geometric."""
    return 0.5 * (np.asarray(edges)[:-1] + np.asarray(edges)[1:])


def area_weights(lats, lons):
    """Cosine-latitude weights over the grid, so a mean does not over-count the poles."""
    return np.broadcast_to(np.cos(np.deg2rad(np.asarray(lats, dtype=float)))[:, None],
                           (len(lats), len(lons)))


def area_mean(field, weights):
    """Weighted mean over the trailing two axes."""
    return (field * weights).sum(axis=(-2, -1)) / weights.sum()


def lag_correlation(cube, ages, max_lag=58):
    """``(lag in yr, rho[lag, cell])`` on the older half, as the observation model reads it.

    Only the older half is used, so on a lane whose truth is a younger model state the truth
    cannot leak into the correction for the distance to it.
    """
    prior_idx, _ = chronological_half_split(ages, stride=1)
    S, cell_var = temporal_structure_function(cube, prior_idx, max_lag=max_lag)
    lag_yr = np.arange(S.shape[0]) * 25.0
    return lag_yr, np.clip(1.0 - S / (2.0 * cell_var[None, :]), RHO_FLOOR, 1.0)


# ---------------------------------------------------------------------------
# The proxy network.
# ---------------------------------------------------------------------------

def nearest_index(values, axis):
    """Index of the nearest entry of a sorted ascending axis, for each value."""
    i = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    return np.where(values - axis[i - 1] < axis[i] - values, i - 1, i)


def cells_reached(long, lats, lons):
    """``(n_lat, n_lon)`` mask of the grid cells any proxy row falls in."""
    n_cells = len(lats) * len(lons)
    gidx = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                          long["channel"].to_numpy(), lats, lons) % n_cells
    reached = np.zeros(n_cells, dtype=bool)
    reached[np.unique(gidx)] = True
    return reached.reshape(len(lats), len(lons))


def site_frame_offset(cube, ages, samples, lats, lons):
    """Per site-channel error of anomalising about the site's own record mean.

    Measured on the model as a stand-in: the site's cell averaged over the ages it reports,
    minus the cell's mean over the whole archive.
    """
    flat = cube.reshape(len(ages), -1).astype(np.float64)
    full_mean = flat.mean(axis=0)
    samples_g = samples.assign(g=obs_cell_index(samples["lat"].to_numpy(),
                                                samples["lon"].to_numpy(),
                                                samples["channel"].to_numpy(), lats, lons))
    return np.array([flat[np.clip(np.searchsorted(ages, s["age"].to_numpy()), 0, len(ages) - 1),
                          int(s["g"].iloc[0])].mean() - full_mean[int(s["g"].iloc[0])]
                     for _, s in samples_g.groupby(["site", "channel"])])


def representativeness_pairs(long, cell):
    """Per channel, the co-cell co-age proxy pairs the spatial error variance is estimated from.

    Groups rows exactly as :func:`paleoreco.assim.observations.representativeness_variance`
    does and returns ``{channel: (pairs, sites, cells)}`` over the groups holding two or more.
    """
    df = pd.DataFrame({"cell": np.asarray(cell), "age": long["age"].to_numpy(),
                       "channel": long["channel"].to_numpy(), "site": long["site"].to_numpy()})
    out = {}
    for channel, sub in df.groupby("channel", sort=False):
        size = sub.groupby(["cell", "age"], sort=False)["site"].transform("size")
        paired = sub[size >= 2]
        k = paired.groupby(["cell", "age"], sort=False).size().to_numpy(dtype=np.float64)
        out[channel] = (int((k * (k - 1.0) / 2.0).sum()), paired["site"].nunique(),
                        paired["cell"].nunique())
    return out


# ---------------------------------------------------------------------------
# The analog prior and the tendency stack.
# ---------------------------------------------------------------------------

def trajectory_pool(cube, ages, lats, lons, valid, taper):
    """``(prior, pool anomalies, pool indices)`` of the older half under one lane's taper."""
    prior_idx, _ = chronological_half_split(ages)
    prior = build_prior(cube, ages, lats, lons, prior_idx, valid, **taper)
    pool = (cube[prior_idx].reshape(len(prior_idx), -1).astype(np.float64)
            - prior.clim_mean.ravel())
    return prior, pool, prior_idx


def mta_stack(pool, pool_ages, lats, lons):
    """The shipped estimator's tendency stack, built over ``pool``; no analysis is run.

    Only the stack geometry is read from it, so the taper passed in is irrelevant.
    """
    return HGAOEnKF(pool, pool_ages, np.eye(1), (len(VARS), len(lats), len(lons)),
                    lats, lons, k=60, hybrid_w=1.0,
                    taper_meta={"localization_km": None, "shrinkage_lambda": 0.0, "alpha": 1.0},
                    tendency_theta=2.0, tendency_lag_yr=ex.MT_REFERENCE_LAG_YR,
                    tendency_extra_lags_yr=ex.MT_EXTRA_LAGS_YR,
                    tendency_curvature_yr=ex.MT_CURVATURE_YR,
                    tendency_normalise=True, preserve_obs_trace=True)


def stack_response(mt, periods):
    """``(stack, one lag)`` response to a cycle of each period, each scaled to its own peak.

    A centred first difference at lag l returns a component of period T scaled by
    |sin(2 pi l / T)| and a second difference by 2|1 - cos(2 pi l / T)|. The blocks enter the
    covariance as extra rows, so their contributions add in quadrature.
    """
    omega = 2 * np.pi / periods
    response = np.zeros_like(periods)
    for block in mt._tendency_blocks:
        lag, second = block
        h = (2 * np.abs(1 - np.cos(omega * lag)) if second else np.abs(np.sin(omega * lag)))
        response += (mt._tendency_scale[block] * h) ** 2
    response = np.sqrt(response); response /= response.max()
    one_lag = np.abs(np.sin(omega * ex.MT_REFERENCE_LAG_YR)); one_lag /= one_lag.max()
    return response, one_lag


def whitened_spectra(prior, long, ages, lats, lons, trajectory):
    """Eigenvalues of the whitened ``H B H^T`` for every network a trajectory lane assimilated.

    Each network is rebuilt from the age it was borrowed from, less the rows whose offset in
    time ran off the archive, so it is the network the analysis actually met.
    """
    safe_flat = np.broadcast_to(prior.safe_valid, (2,) + prior.safe_valid.shape).ravel()
    by_shape = sample_block_centres(long).groupby("age")
    spectra = []
    for age, shape_age in zip(trajectory["ages"], trajectory["shape_ages"]):
        sl = by_shape.get_group(int(shape_age))
        g = obs_cell_index(sl["lat"].to_numpy(), sl["lon"].to_numpy(),
                           sl["channel"].to_numpy(), lats, lons)
        r = sl["sse"].to_numpy()
        keep = safe_flat[g] & (r > 0)
        g, r = g[keep], r[keep].astype(np.float64)
        _, on_axis = ex.transplanted_source(sl["centre"].to_numpy()[keep],
                                            int(shape_age), int(age), ages)
        g, r = g[on_axis], r[on_axis]
        if len(g) < 10:
            continue
        wb = whitened_block(prior.B[:, g], prior.B[np.ix_(g, g)], r)
        spectra.append(wb.Lam[wb.Lam > 0])
    return spectra


# ---------------------------------------------------------------------------
# D-O composites.
# ---------------------------------------------------------------------------

def event_halves(ages, onset):
    """``(interstadial, stadial)`` age masks for one onset."""
    return ((ages >= onset - DO_POST_YR) & (ages <= onset),
            (ages >= onset) & (ages <= onset + DO_PRE_YR))


def do_composite(field, ages, onsets):
    """Interstadial-minus-stadial difference per event, stacked on a leading event axis.

    Each half is a mean rather than one 25-yr state, which would carry a single network's noise.
    """
    out = []
    for onset in onsets:
        warm, cold = event_halves(ages, onset)
        if warm.any() and cold.any():
            out.append(field[warm].mean(axis=0) - field[cold].mean(axis=0))
    return np.asarray(out)


def proxy_rows_on_grid(long, ages, lats, lons, structure, step_yr):
    """Every proxy row placed on the grid and the age axis, with its lag correlation ``rho``.

    Lets a gridded field be read through the network exactly as the pollen is.
    """
    n_cells = len(lats) * len(lons)
    dated = sample_block_centres(long)
    gather = obs_cell_index(dated["lat"].to_numpy(), dated["lon"].to_numpy(),
                            dated["channel"].to_numpy(), lats, lons)
    rho, _ = temporal_terms(*structure, gather,
                            np.abs(dated["age"] - dated["centre"]).to_numpy(), step_yr)
    at_grid = dated.assign(cell=gather % n_cells, chan=gather // n_cells, rho=rho)
    at_grid = at_grid.assign(ti=at_grid["age"].map(
        {int(a): i for i, a in enumerate(ages)})).dropna(subset=["ti"])
    at_grid["ti"] = at_grid["ti"].astype(int)
    return at_grid


def at_site(field, n_lon):
    """A reader of ``field[age, channel, lat, lon]`` at the rows of :func:`proxy_rows_on_grid`."""
    return lambda d: field[d["ti"].to_numpy(), d["chan"].to_numpy(),
                           d["cell"].to_numpy() // n_lon, d["cell"].to_numpy() % n_lon]


def site_composite(rows, onsets, value, min_lat: float | None = NET_LAT):
    """Interstadial-minus-stadial at the proxy sites, as ``(n_events, n_channels)``.

    One mean per site per half of an event window, differenced, then averaged over the sites
    north of ``min_lat`` that report both channels. The sites are a biased sample of any
    region, so a field is compared with the pollen through this estimator, not an area mean.
    """
    out = []
    for onset in onsets:
        halves = []
        for lo, hi in ((onset - DO_POST_YR, onset), (onset, onset + DO_PRE_YR)):
            d = rows[(rows["age"] >= lo) & (rows["age"] <= hi)].copy()
            d["v"] = value(d)
            halves.append(d.groupby(["site", "channel", "lat"])["v"].mean())
        d = (halves[0] - halves[1]).dropna().reset_index()
        if min_lat is not None:
            d = d[d["lat"] > min_lat]
        pair = d.pivot_table(index="site", columns="channel", values="v").dropna()
        out.append([pair[v].mean() if v in pair else np.nan for v in VARS])
    return np.asarray(out)


def box_mask(lats, lons, lat_range, lon_range):
    """``(n_lat, n_lon)`` mask of a latitude-longitude box, longitudes in -180..180."""
    la, lo = np.meshgrid(np.asarray(lats, float), np.asarray(lons, float), indexing="ij")
    lo = ((lo + 180.0) % 360.0) - 180.0
    return ((la > lat_range[0]) & (la < lat_range[1])
            & (lo > lon_range[0]) & (lo < lon_range[1]))


# ---------------------------------------------------------------------------
# Scoring against held-out data.
# ---------------------------------------------------------------------------

def ce_against_climatology(truth, pred):
    """Coefficient of efficiency against a zero anomaly."""
    return float(1.0 - np.sum((truth - pred) ** 2) / np.sum(truth ** 2))


def _paired_interval(parts, n_units, n_boot, seed):
    """2.5 and 97.5 percentiles of ``CE(better) - CE(worse)`` over redrawn units.

    ``parts`` holds per-unit sums: squared error of the better and the worse predictor, and
    squared truth. Every draw scores both predictors on the same units.
    """
    counts = np.random.default_rng(seed).multinomial(n_units, np.full(n_units, 1.0 / n_units),
                                                     size=n_boot)
    totals = counts @ parts.T
    draws = (totals[:, 1] - totals[:, 0]) / totals[:, 2]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def site_bootstrap_dce(truth, better, worse, site, n_boot=N_BOOT, seed=0):
    """``(dCE, lo, hi)`` for two predictors of the same rows, resampling whole sites."""
    codes, index = pd.factorize(site)
    n = len(index)
    parts = np.stack([np.bincount(codes, (truth - better) ** 2, n),
                      np.bincount(codes, (truth - worse) ** 2, n),
                      np.bincount(codes, truth ** 2, n)])
    lo, hi = _paired_interval(parts, n, n_boot, seed)
    return (ce_against_climatology(truth, better) - ce_against_climatology(truth, worse),
            lo, hi)


def truth_bootstrap_dce(truth, better, worse, mask, n_boot=N_BOOT, seed=0):
    """``(dCE, lo, hi)`` for two ``(truth, channel, lat, lon)`` analyses, resampling truths.

    Scored over the cells in ``mask`` and both channels.
    """
    t = truth[..., mask].astype(np.float64)
    b = better[..., mask].astype(np.float64)
    w = worse[..., mask].astype(np.float64)
    parts = np.stack([((t - b) ** 2).sum(axis=(1, 2)), ((t - w) ** 2).sum(axis=(1, 2)),
                      (t ** 2).sum(axis=(1, 2))])
    lo, hi = _paired_interval(parts, len(t), n_boot, seed)
    return ce_against_climatology(t, b) - ce_against_climatology(t, w), lo, hi


# Progressively stricter exclusions of withheld rows near an assimilated site. Overlapping
# rather than nested: every row, rows whose own cell carried no assimilated observation, then
# great-circle distance cuts.
VOID_TESTS = (("all rows", None), ("different cell", "cell"),
              ("> 250 km", 250.0), ("> 500 km", 500.0), ("> 1000 km", 1000.0))


def withheld_mask(kind, distance_km, cell_observed):
    """Row mask for one entry of :data:`VOID_TESTS`."""
    if kind is None:
        return np.ones(len(distance_km), dtype=bool)
    if kind == "cell":
        return ~cell_observed
    return np.nan_to_num(distance_km, nan=-1.0) > float(kind)


def withheld_identity(long, ages, lats, lons, safe_valid, cfg):
    """``(age, gather, centre, site, channel, actual, sse, cell_observed)`` of every pooled
    test row of the real-proxy lane, in the order the lane stored its predictions.

    The lane stores what each held-out prediction was, but not the age it sat at, so a second
    predictor cannot be scored on the same rows without this. It replays the lane's fold
    partition and age loop over the proxy table and reads nothing from an estimator; the
    caller checks the result against the stored arrays. ``cell_observed`` marks rows whose own
    cell also carried an assimilated observation at that age.
    """
    long = long[long["age"].isin({int(a) for a in ages})]
    n_cells = len(lats) * len(lons)
    safe_flat = np.broadcast_to(safe_valid, (len(VARS), len(lats), len(lons))).ravel()
    dated = sample_block_centres(long)
    obs_ages = np.intersect1d(long["age"].unique(), ages)
    folds = [set(f.tolist()) for f in ex._site_folds(
        long, int(cfg["k_folds"]), cfg["fold_kind"], int(cfg["seed"]))]
    every = set(long["site"].unique().tolist())
    rows = []
    for fold in folds:
        assimilated = every - fold
        for age in obs_ages:
            o = observations_at_age(dated, int(age))
            g = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
            usable = safe_flat[g] & (o["sse"] > 0) & np.isfinite(o["my"])
            given = usable & np.array([s in assimilated for s in o["site"]])
            held = usable & np.array([s in fold for s in o["site"]])
            if not given.any() or not held.any():
                continue
            given_cells = set(g[given].tolist())
            rows.append(pd.DataFrame({
                "age": int(age), "gather": g[held], "centre": o["centre"][held],
                "site": o["site"][held], "channel": g[held] // n_cells,
                "actual": o["y"][held] - o["my"][held], "sse": o["sse"][held],
                "cell_observed": [c in given_cells for c in g[held]]}))
    return pd.concat(rows, ignore_index=True)


def loveclim_at_sites(cube, ages, clim_mean, ident, structure, step_yr):
    """The simulation as a predictor of the withheld pollen.

    The simulation's anomaly at each row's cell and age is multiplied by the same lag
    correlation every estimator's prediction carries, which is what makes them comparable.
    """
    gather = ident["gather"].to_numpy()
    age, centre = ident["age"].to_numpy(), ident["centre"].to_numpy()
    rho, _ = temporal_terms(*structure, gather, np.abs(age - centre), step_yr)
    anom = (cube.astype(np.float64) - clim_mean).reshape(len(ages), -1)
    index = {int(a): i for i, a in enumerate(ages)}
    return rho * anom[np.array([index[a] for a in age]), gather]
