"""The three evaluation lanes: what each one observes, and how an estimator is scored on it.

* snapshot (``ppe``): the older half of the archive builds the prior; each younger-half
  state is a truth, observed through two real network geometries drawn from random proxy
  ages, one selecting the operating point and one reporting it. Pseudo-observations sit on
  the truth's own cell and age with ``N(0, sse)`` noise, so R is ``diag(sse)``.
* time series (``trajectory``): the same split, reconstructed in sequence. The test arm
  assimilates each age's own network and the selection arm one borrowed from another age.
  Every observation reads the state at its own sample's offset in time, so it carries the
  temporal term of :func:`paleoreco.assim.observations.deflate`; there is no spatial term.
* withholding: the real network. The sites fall into five folds, and each fold is
  predicted once from an analysis of the other four, so every withheld row is scored
  against an analysis that never assimilated its site.

On the two pseudo-proxy lanes the truths and pseudo-observations are anomalies about the
younger half's own mean, against a zero background; B and the analog pool come from the
older half.

A lane is built once, so every configuration a grid scans meets the same networks and the
same noise. ``scan_*`` returns the skill rows a selection reads, from mean-only analyses
over an amplitude sweep; ``score_*`` analyses in full and returns everything else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paleoreco.data import VARS, cell_climatology
from paleoreco.data.splits import age_step, chronological_half_split
from paleoreco.assim.background import temporal_structure_function
from paleoreco.assim.innovation import nearest_age_index, obs_cell_index
from paleoreco.assim.method import Method
from paleoreco.assim.observations import (
    ObservationModel,
    deflate,
    max_block_lag,
    observations_at_age,
    representativeness_variance,
    sample_block_centres,
    temporal_terms,
)
from paleoreco.assim.priors import great_circle_km_between
from paleoreco.eval import da
from paleoreco.eval.rows import (
    calibration_rows,
    field_calibration_rows,
    field_skill_rows,
    obs_skill_rows,
    ssim_rows,
    timescale_rows,
)

SPLITS = ("selection", "test")
# Minimum observations for a pseudo-proxy network to be drawn.
MIN_OBS = 10
# Attempts to find a borrowed network whose offsets stay on the archive at one age.
_MAX_SHAPE_DRAWS = 20
# Timescales the time-series lane resolves. A low-pass window keeps everything slower than
# it; a band is the difference of two low-passes and isolates one timescale.
LOWPASS_WINDOWS = (25, 100, 250, 500, 1000, 2000)
BANDS = ((25, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000))
DIST_EDGES_KM = np.array([0, 500, 1000, 2000, 3000, 5000, 8000, 20000], dtype=float)
K_FOLDS = 5


def report_progress(label: str, done: int, total: int, t0: float) -> None:
    """One-line progress update with a linear-rate ETA over the loop so far."""
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    print(f"  {label} {done}/{total} ({100 * done / total:.0f}%, "
          f"elapsed {elapsed:.0f}s, eta {eta:.0f}s)", flush=True)


class _Progress:
    """Reports a loop at ten even steps, or stays silent without a label."""

    def __init__(self, label: str | None, total: int):
        self.label, self.total, self.t0 = label, total, time.time()
        self.every = max(1, total // 10)

    def step(self, done: int) -> None:
        if self.label and (done % self.every == 0 or done == self.total):
            report_progress(self.label, done, self.total, self.t0)


def _naive_base(base: dict, kind: str) -> dict:
    """A prior-free reference's row columns: no configuration and no amplitude."""
    return {**{key: np.nan for key in base}, "method": kind, "b_scale": np.nan}


# ---------------------------------------------------------------------------
# Networks.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Network:
    """One assimilation's observations on a pseudo-proxy lane.

    ``y`` and ``r`` are what the update consumes; ``y_read`` is the observation as read,
    before any correction, which the prior-free references interpolate.
    """

    gather: np.ndarray
    y: np.ndarray
    r: np.ndarray
    y_read: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def _obs_geometry(o: dict, lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray) -> dict:
    """Gather indices, error variance, site coords and block centres for usable observations.

    ``centre`` is each sample's block midpoint, present only if the caller attached it.
    """
    gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
    keep = safe_flat[gather] & (o["sse"] > 0)
    geom = {"gather": gather[keep], "sse": o["sse"][keep].astype(np.float64),
            "lat": o["lat"][keep], "lon": o["lon"][keep]}
    if "centre" in o:
        geom["centre"] = o["centre"][keep].astype(np.float64)
    return geom


def _draw_shapes(long: pd.DataFrame, rng: np.random.Generator,
                 lats: np.ndarray, lons: np.ndarray, safe_flat: np.ndarray,
                 n: int, min_obs: int) -> list[tuple[int, dict]]:
    """``n`` distinct proxy ages and their networks, each holding at least ``min_obs`` rows.

    An age contributes only its network geometry. The source age is returned because a
    borrowed sample's time offset is relative to its own network's age.
    """
    picked = []
    for a in rng.permutation(long["age"].unique()):
        o = observations_at_age(long, int(a))
        if not len(o.get("age", [])):
            continue
        geom = _obs_geometry(o, lats, lons, safe_flat)
        if len(geom["gather"]) >= min_obs:
            picked.append((int(a), geom))
            if len(picked) == n:
                return picked
    raise ValueError(
        f"fewer than {n} proxy ages carry {min_obs} usable observations on this grid")


def _own_network(long: pd.DataFrame, age: int, lats: np.ndarray, lons: np.ndarray,
                 safe_flat: np.ndarray, min_obs: int) -> dict | None:
    """The proxy network an age actually carries, or ``None`` where it is too thin."""
    o = observations_at_age(long, int(age))
    if not len(o.get("age", [])):
        return None
    geom = _obs_geometry(o, lats, lons, safe_flat)
    return geom if len(geom["gather"]) >= min_obs else None


def transplanted_source(centre: np.ndarray, shape_age: int, age: int,
                        ages_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Archive index each observation reads, and which ones land on the axis.

    A network borrowed from ``shape_age`` reports on ``age + (centre - shape_age)``: the
    time offset is relative to the network's own age. Offsets that run off the archive are
    dropped rather than clamped to its ends.
    """
    src_age = float(age) + (np.asarray(centre, dtype=np.float64) - float(shape_age))
    on_axis = (src_age >= ages_i[0]) & (src_age <= ages_i[-1])
    return nearest_age_index(src_age[on_axis], ages_i), on_axis


# ---------------------------------------------------------------------------
# Prior-free references.
# ---------------------------------------------------------------------------
def _naive_fields(lats: np.ndarray, lons: np.ndarray, net: Network,
                  shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Nearest-site and inverse-square-distance fields interpolated from one network.

    A channel with no observations stays zero.
    """
    n_lat, n_lon = len(lats), len(lons)
    lat_cell, lon_cell = np.repeat(lats, n_lon), np.tile(lons, n_lat)
    chan = net.gather // (n_lat * n_lon)
    out = {kind: np.zeros(shape) for kind in ("nearest", "idw")}
    for c in range(shape[0]):
        sel = chan == c
        if not sel.any():
            continue
        d = great_circle_km_between(lat_cell, lon_cell, net.lat[sel], net.lon[sel])
        w = 1.0 / np.clip(d, 1.0, None) ** 2
        y = net.y_read[sel]
        out["nearest"][c] = y[np.argmin(d, axis=1)].reshape(n_lat, n_lon)
        out["idw"][c] = ((w / w.sum(axis=1, keepdims=True)) @ y).reshape(n_lat, n_lon)
    return out


def _naive_at_targets(assim: dict, target: dict, n_chan: int) -> tuple[dict, np.ndarray]:
    """Prior-free predictions at withheld rows, and each one's distance to the nearest
    assimilated site of its channel.

    A channel with no assimilated observation predicts zero.
    """
    n_t = len(target["lat"])
    out = {"nearest": np.zeros(n_t), "idw": np.zeros(n_t)}
    dist = np.full(n_t, np.nan)
    for c in range(n_chan):
        tsel, asel = target["chan"] == c, assim["chan"] == c
        if not tsel.any() or not asel.any():
            continue
        d = great_circle_km_between(target["lat"][tsel], target["lon"][tsel],
                                    assim["lat"][asel], assim["lon"][asel])
        y = assim["y"][asel]
        nearest = np.argmin(d, axis=1)
        dist[tsel] = d[np.arange(int(tsel.sum())), nearest]
        out["nearest"][tsel] = y[nearest]
        w = 1.0 / np.clip(d, 1.0, None) ** 2
        out["idw"][tsel] = (w / w.sum(axis=1, keepdims=True)) @ y
    return out, dist


# ---------------------------------------------------------------------------
# Snapshot and time-series lanes.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PseudoProxyLane:
    """A pseudo-proxy lane's truths and the networks that observe them.

    ``networks`` holds one ``{split: Network}`` per truth. ``meta`` records the split and
    the draw for the config.
    """

    lats: np.ndarray
    lons: np.ndarray
    safe_valid: np.ndarray
    prior_idx: np.ndarray
    truth_anom: np.ndarray
    truth_clim: np.ndarray
    networks: tuple[dict, ...]
    meta: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(VARS), len(self.lats), len(self.lons))


def _split_meta(ages_i, prior_idx, truth_idx, truth_stride) -> dict:
    """Which ages built the prior and which supplied the truths."""
    return {"split": "chronological_midpoint", "prior_half": "older",
            "n_prior_ages": int(len(prior_idx)), "truth_stride": int(truth_stride),
            "prior_ages": [int(ages_i[prior_idx].min()), int(ages_i[prior_idx].max())],
            "truth_ages": [int(ages_i[truth_idx].min()), int(ages_i[truth_idx].max())]}


def _younger_half(cube, ages, valid, stride):
    """``(ages_i, prior_idx, truth_idx, safe_valid, truth_clim, truth_anom)`` of the split."""
    ages_i = np.asarray(ages, dtype=np.int64)
    prior_idx, truth_idx = chronological_half_split(ages_i, stride=stride)
    safe_valid = cell_climatology(cube, prior_idx, valid)["safe_valid"]
    truth_cube = cube[truth_idx].astype(np.float64)
    truth_clim = truth_cube.mean(axis=0)
    return ages_i, prior_idx, truth_idx, safe_valid, truth_clim, truth_cube - truth_clim


def snapshot_lane(cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                  valid: np.ndarray, long: pd.DataFrame, *, truth_stride: int = 1,
                  seed: int = 0, min_obs: int = MIN_OBS) -> PseudoProxyLane:
    """Every ``truth_stride``-th younger-half state, each observed by two drawn networks.

    Each pseudo-observation is the truth at its nearest cell plus ``N(0, sse)`` noise, drawn
    independently for the two networks.
    """
    ages_i, prior_idx, truth_idx, safe_valid, truth_clim, truth_anom = _younger_half(
        cube, ages, valid, truth_stride)
    safe_flat = np.broadcast_to(safe_valid, (len(VARS), *safe_valid.shape)).ravel()
    rng = np.random.default_rng(seed)
    networks, drawn = [], []
    for truth in truth_anom:
        shapes = _draw_shapes(long, rng, lats, lons, safe_flat, len(SPLITS), min_obs)
        drawn.append([age for age, _ in shapes])
        nets = {}
        for split, (_, geom) in zip(SPLITS, shapes):
            g = geom["gather"]
            y = truth.ravel()[g] + rng.normal(0.0, np.sqrt(geom["sse"]))
            nets[split] = Network(gather=g, y=y, r=geom["sse"], y_read=y,
                                  lat=geom["lat"], lon=geom["lon"])
        networks.append(nets)
    meta = {**_split_meta(ages_i, prior_idx, truth_idx, truth_stride),
            "n_truths": int(len(truth_anom)), "seed": int(seed), "min_obs": int(min_obs),
            "networks": {split: "a proxy age drawn at random" for split in SPLITS},
            "drawn_ages": np.asarray(drawn, dtype=np.int64)}
    return PseudoProxyLane(lats=lats, lons=lons, safe_valid=safe_valid, prior_idx=prior_idx,
                           truth_anom=truth_anom, truth_clim=truth_clim,
                           networks=tuple(networks), meta=meta)


def trajectory_lane(cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                    valid: np.ndarray, long: pd.DataFrame, *, seed: int = 0,
                    min_obs: int = MIN_OBS) -> PseudoProxyLane:
    """The younger half as a consecutive run, each observation read at its own offset.

    The test arm assimilates the age's own network, which changes little between
    neighbouring ages. The selection arm borrows a network from another age, redrawn if too
    many of its offsets run off the archive. Ages with no usable network are recorded and
    skipped. The structure function comes from the prior ages only.
    """
    ages_i, prior_idx, truth_idx, safe_valid, truth_clim, truth_anom = _younger_half(
        cube, ages, valid, 1)
    step_yr = age_step(ages_i)
    safe_flat = np.broadcast_to(safe_valid, (len(VARS), *safe_valid.shape)).ravel()
    # Observations are read from the whole run, since a block centre can fall in the older
    # half; B still sees prior ages only.
    all_anom = cube.reshape(len(ages_i), -1).astype(np.float64) - truth_clim.ravel()
    long = sample_block_centres(long)
    max_lag = max_block_lag(long, step_yr)
    S, var = temporal_structure_function(cube, prior_idx, max_lag=max_lag)

    rng = np.random.default_rng(seed)
    networks, covered, skipped = [], [], []
    for ti, age in enumerate(ages_i[truth_idx]):
        own = _own_network(long, int(age), lats, lons, safe_flat, min_obs)
        if own is None:
            skipped.append(int(age))
            continue
        sources = {"test": (own, *transplanted_source(own["centre"], int(age), age, ages_i))}
        for _ in range(_MAX_SHAPE_DRAWS):
            (shape_age, geom), = _draw_shapes(long, rng, lats, lons, safe_flat, 1, min_obs)
            src, on_axis = transplanted_source(geom["centre"], shape_age, age, ages_i)
            if on_axis.sum() >= min_obs:
                sources["selection"] = (geom, src, on_axis)
                break
        else:
            raise ValueError(
                f"no borrowed network kept {min_obs} observations on the archive at age "
                f"{int(age)} in {_MAX_SHAPE_DRAWS} draws")
        nets = {}
        for split in SPLITS:
            geom, src, on_axis = sources[split]
            g, sse = geom["gather"][on_axis], geom["sse"][on_axis]
            stale = all_anom[src, g] + rng.normal(0.0, np.sqrt(sse))
            # The lag is taken from the age actually read, after rounding.
            rho, resid = temporal_terms(S, var, g, np.abs(ages_i[src] - age), step_yr)
            y, r = deflate(stale, sse, rho, resid)
            nets[split] = Network(gather=g, y=y, r=r, y_read=stale,
                                  lat=geom["lat"][on_axis], lon=geom["lon"][on_axis])
        networks.append(nets)
        covered.append(ti)

    if len(covered) < 4:
        raise ValueError(f"only {len(covered)} ages carried a usable network; nothing to score")
    covered = np.asarray(covered)
    # The timescale filters assume contiguous ages; skipped ages may only sit at the ends.
    if np.any(np.diff(covered) != 1):
        raise ValueError("the scored ages are not consecutive; the timescale filters "
                         "assume a uniform step")
    scored = ages_i[truth_idx][covered]
    meta = {**_split_meta(ages_i, prior_idx, truth_idx, 1),
            "n_truths": int(len(covered)), "seed": int(seed), "min_obs": int(min_obs),
            "networks": {"selection": "a proxy age drawn at random",
                         "test": "the age's own proxy network"},
            "step_yr": step_yr, "max_block_lag_steps": max_lag,
            "ages": scored, "skipped_ages": [int(a) for a in skipped],
            "lowpass_windows": [int(w) for w in LOWPASS_WINDOWS],
            "bands": [[int(a), int(b)] for a, b in BANDS]}
    # The anomaly frame is the whole younger half, not just the scored ages.
    return PseudoProxyLane(lats=lats, lons=lons, safe_valid=safe_valid, prior_idx=prior_idx,
                           truth_anom=truth_anom[covered], truth_clim=truth_clim,
                           networks=tuple(networks), meta=meta)


def scan_pseudo_proxy(lane: PseudoProxyLane, method: Method, b_scales, base: dict,
                      progress: str | None = None) -> list[dict]:
    """Skill rows on both splits at every amplitude, from mean-only analyses."""
    b_scales = tuple(float(b) for b in b_scales)
    zero = np.zeros(int(np.prod(lane.shape)))
    recon = {split: np.zeros((len(b_scales), len(lane.networks), *lane.shape))
             for split in SPLITS}
    tick = _Progress(progress, len(lane.networks))
    for ti, nets in enumerate(lane.networks):
        for split in SPLITS:
            net = nets[split]
            res = method.apply_sweep(method.prepare_sweep(net.gather, net.r, b_scales),
                                     net.y, zero, mean_only=True)
            for bj, r in enumerate(res):
                recon[split][bj, ti] = r.mean_anom
        tick.step(ti + 1)
    rows = []
    for bj, b in enumerate(b_scales):
        for split in SPLITS:
            rows += field_skill_rows(lane.truth_anom, recon[split][bj], lane.safe_valid,
                                     {**base, "b_scale": b, "split": split})
    return rows


def _analyse_test_arm(lane: PseudoProxyLane, method: Method, b_scale: float,
                      progress: str | None):
    """Full analyses of every test network at one amplitude, and the prior-free fields."""
    zero = np.zeros(int(np.prod(lane.shape)))
    T = len(lane.networks)
    recon, post = np.zeros((T, *lane.shape)), np.zeros((T, *lane.shape))
    naive = {kind: np.zeros((T, *lane.shape)) for kind in ("nearest", "idw")}
    analog_index = []
    tick = _Progress(progress, T)
    for ti, nets in enumerate(lane.networks):
        net = nets["test"]
        gain = method.prepare_sweep(net.gather, net.r, (b_scale,))
        res = method.apply_sweep(gain, net.y, zero)[0]
        recon[ti], post[ti] = res.mean_anom, res.posterior_var
        if hasattr(method, "select"):
            analog_index.append(method.select(gain, net.y))
        for kind, field in _naive_fields(lane.lats, lane.lons, net, lane.shape).items():
            naive[kind][ti] = field
        tick.step(ti + 1)
    return recon, post, naive, analog_index


def _base_fields(lane: PseudoProxyLane, recon, post, naive, analog_index, b_scale) -> dict:
    """Arrays every pseudo-proxy lane stores for its selected amplitude, fields in float32.

    ``analog_index`` holds the members each analysis selected, where the estimator selects
    any.
    """
    fields = {"truth_anom": lane.truth_anom.astype(np.float32), "truth_clim": lane.truth_clim,
              "safe_valid": lane.safe_valid, "lats": np.asarray(lane.lats),
              "lons": np.asarray(lane.lons), "b_scale": np.asarray(b_scale),
              "recon": recon.astype(np.float32), "post_var": post.astype(np.float32),
              "naive_nearest": naive["nearest"].astype(np.float32),
              "naive_idw": naive["idw"].astype(np.float32)}
    if analog_index:
        fields["analog_index"] = np.stack(analog_index)
    return fields


def score_snapshot(lane: PseudoProxyLane, method: Method, b_scale: float, base: dict,
                   progress: str | None = None) -> tuple[list[dict], dict, dict]:
    """``(rows, fields, skill_vs_distance)`` of the test split at one amplitude.

    The rows are SSIM and calibration, and the prior-free references' skill and SSIM; the
    skill rows themselves come from :func:`scan_pseudo_proxy`.
    """
    recon, post, naive, analog_index = _analyse_test_arm(lane, method, b_scale, progress)
    sv = lane.safe_valid
    at_c = {**base, "b_scale": float(b_scale), "split": "test"}
    rows = (ssim_rows(lane.truth_anom, recon, sv, at_c)
            + field_calibration_rows(lane.truth_anom, recon, post, sv, at_c))
    for kind, field in naive.items():
        ref = {**_naive_base(base, kind), "split": "test"}
        rows += field_skill_rows(lane.truth_anom, field, sv, ref)
        rows += ssim_rows(lane.truth_anom, field, sv, ref)

    T = len(lane.networks)
    dist = np.stack([da.nearest_obs_distance(lane.lats, lane.lons, nets["test"].lat,
                                             nets["test"].lon) for nets in lane.networks])
    t_pool = lane.truth_anom[:, :, sv].reshape(T, -1).ravel()
    curve = da.skill_vs_distance(t_pool, recon[:, :, sv].reshape(T, -1).ravel(),
                                 np.zeros_like(t_pool),
                                 np.tile(dist[:, sv.ravel()], (1, len(VARS))).ravel(),
                                 DIST_EDGES_KM)
    skill = {"edges": DIST_EDGES_KM, "count": curve["count"], "ce": curve["ce"]}
    fields = {**_base_fields(lane, recon, post, naive, analog_index, b_scale),
              "drawn_ages": lane.meta["drawn_ages"]}
    return rows, fields, skill


def score_trajectory(lane: PseudoProxyLane, method: Method, b_scale: float, base: dict,
                     progress: str | None = None) -> tuple[list[dict], dict]:
    """``(rows, fields)`` of the test arm at one amplitude.

    The rows are SSIM, skill by timescale and calibration, and the prior-free references'
    skill, SSIM and timescale rows.
    """
    recon, post, naive, analog_index = _analyse_test_arm(lane, method, b_scale, progress)
    sv, truth = lane.safe_valid, lane.truth_anom
    by_scale = dict(step_yr=lane.meta["step_yr"], windows=LOWPASS_WINDOWS, bands=BANDS)
    at_c = {**base, "b_scale": float(b_scale), "split": "test"}
    rows = (ssim_rows(truth, recon, sv, at_c)
            + timescale_rows(truth, recon, sv, at_c, **by_scale)
            + field_calibration_rows(truth, recon, post, sv, at_c))
    for kind, field in naive.items():
        ref = {**_naive_base(base, kind), "split": "test"}
        rows += (field_skill_rows(truth, field, sv, ref) + ssim_rows(truth, field, sv, ref)
                 + timescale_rows(truth, field, sv, ref, **by_scale))
    fields = {**_base_fields(lane, recon, post, naive, analog_index, b_scale),
              "ages": lane.meta["ages"], "step_yr": np.asarray(lane.meta["step_yr"]),
              # The test arm assimilates each age's own network.
              "shape_ages": lane.meta["ages"]}
    return rows, fields


# ---------------------------------------------------------------------------
# Withholding lane.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProxyRows:
    """Real-proxy rows at one age, with every error term an observation model may charge.

    ``y`` is the anomaly about the site's own climatology; ``rep`` is the fold's spatial
    error variance for each row's channel; ``rho`` and ``resid`` are the temporal terms.
    """

    gather: np.ndarray
    y: np.ndarray
    sse: np.ndarray
    rep: np.ndarray
    rho: np.ndarray
    resid: np.ndarray

    def assimilated(self, model: ObservationModel) -> tuple[np.ndarray, np.ndarray]:
        """``(y, R diagonal)`` as the update consumes them under ``model``."""
        r = self.sse + self.rep if model.spatial else self.sse
        return deflate(self.y, r, self.rho, self.resid) if model.temporal else (self.y, r)

    def predictive(self, model: ObservationModel, mean: np.ndarray,
                   var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction of these rows and its variance, from ``H x`` and ``H`` posterior var.

        With the temporal term a withheld sample reads the state at the analysis age
        attenuated by its own ``rho``, and carries the variance the lag leaves unexplained.
        """
        noise = self.sse + self.rep if model.spatial else self.sse
        if model.temporal:
            return self.rho * mean, self.rho ** 2 * var + noise + self.resid
        return mean, var + noise


@dataclass(frozen=True)
class WithheldBatch:
    """One fold at one age: the rows it assimilates and where the withheld rows sit."""

    fold: int
    age: int
    assim: ProxyRows
    target_gather: np.ndarray


@dataclass(frozen=True)
class WithholdingLane:
    """Every (fold, age) analysis, and the withheld rows they predict, pooled in order.

    ``targets`` concatenates the batches' withheld rows; ``info`` carries their site, fold,
    age, channel, distance to the nearest assimilated site, whether their own cell was
    assimilated, and the prior-free predictions.
    """

    shape: tuple[int, int, int]
    batches: tuple[WithheldBatch, ...]
    targets: ProxyRows
    info: dict
    meta: dict


def site_folds(long: pd.DataFrame, k: int, seed: int) -> list[np.ndarray]:
    """Partition the sites into ``k`` random folds."""
    ids = np.sort(long["site"].unique())
    return list(np.array_split(np.random.default_rng(seed).permutation(ids), k))


def _proxy_rows(o: dict, sel: np.ndarray, gather: np.ndarray, rep: np.ndarray,
                lag: np.ndarray, S: np.ndarray, var: np.ndarray, step_yr: float,
                n_cells: int) -> ProxyRows:
    g = gather[sel]
    rho, resid = temporal_terms(S, var, g, lag[sel], step_yr)
    return ProxyRows(gather=g, y=(o["y"][sel] - o["my"][sel]).astype(np.float64),
                     sse=o["sse"][sel].astype(np.float64), rep=rep[g // n_cells],
                     rho=rho, resid=resid)


def withholding_lane(cube: np.ndarray, ages: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                     valid: np.ndarray, long: pd.DataFrame, *, k_folds: int = K_FOLDS,
                     seed: int = 0) -> WithholdingLane:
    """Five-fold cross-validation over whole sites, one pass.

    ``long`` must carry the per-site climatology ``my``. The prior and the temporal
    structure function use every age, since what is held out is real proxies, not model
    states. The spatial error variance is re-estimated per fold from the assimilated sites
    alone, so a withheld site never informs the analysis that scores it. A withheld row is
    scored only at ages that also carry an assimilated one.
    """
    shape = (len(VARS), len(lats), len(lons))
    n_cells = len(lats) * len(lons)
    prior_idx = np.arange(len(ages))
    safe_valid = cell_climatology(cube, prior_idx, valid)["safe_valid"]
    safe_flat = np.broadcast_to(safe_valid, shape).ravel()
    long = sample_block_centres(long)
    step_yr = age_step(ages)
    max_lag = max_block_lag(long, step_yr)
    S, var = temporal_structure_function(cube, prior_idx, max_lag=max_lag)
    cell_all = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons)

    # The per-age slices do not depend on the fold, so they are read once.
    slices = []
    for age in np.intersect1d(long["age"].unique(), ages):
        o = observations_at_age(long, int(age))
        gather = obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)
        usable = safe_flat[gather] & (o["sse"] > 0) & np.isfinite(o["my"])
        slices.append((int(age), o, gather, usable, np.abs(o["age"] - o["centre"])))

    folds = site_folds(long, k_folds, seed)
    every = set(long["site"].unique().tolist())
    batches, targets, info = [], [], {key: [] for key in (
        "site", "fold", "age", "distance_km", "cell_observed", "naive_nearest", "naive_idw")}
    rep_by_fold = []
    for fi, fold in enumerate(folds):
        held = set(fold.tolist())
        given = every - held
        rv = representativeness_variance(long, cell_all, sites=given)
        rep = np.array([rv.get(v, 0.0) for v in VARS])
        rep_by_fold.append(rep)
        for age, o, gather, usable, lag in slices:
            a_sel = usable & np.isin(o["site"], list(given))
            t_sel = usable & np.isin(o["site"], list(held))
            if not a_sel.any() or not t_sel.any():
                continue
            assim = _proxy_rows(o, a_sel, gather, rep, lag, S, var, step_yr, n_cells)
            target = _proxy_rows(o, t_sel, gather, rep, lag, S, var, step_yr, n_cells)
            batches.append(WithheldBatch(fold=fi, age=age, assim=assim,
                                         target_gather=target.gather))
            targets.append(target)
            naive, dist = _naive_at_targets(
                {"lat": o["lat"][a_sel], "lon": o["lon"][a_sel], "y": assim.y,
                 "chan": assim.gather // n_cells},
                {"lat": o["lat"][t_sel], "lon": o["lon"][t_sel],
                 "chan": target.gather // n_cells}, len(VARS))
            given_cells = set(assim.gather.tolist())
            info["site"].append(o["site"][t_sel])
            info["fold"].append(np.full(len(target.gather), fi))
            info["age"].append(np.full(len(target.gather), age))
            info["distance_km"].append(dist)
            info["cell_observed"].append(np.array([c in given_cells for c in target.gather]))
            info["naive_nearest"].append(naive["nearest"])
            info["naive_idw"].append(naive["idw"])

    pooled = ProxyRows(*(np.concatenate([getattr(t, f) for t in targets])
                         for f in ("gather", "y", "sse", "rep", "rho", "resid")))
    info = {key: np.concatenate(v) for key, v in info.items()}
    info["channel"] = pooled.gather // n_cells
    rep_by_fold = np.asarray(rep_by_fold)
    rv_full = representativeness_variance(long, cell_all)
    meta = {"k_folds": int(k_folds), "seed": int(seed), "n_prior_ages": int(len(prior_idx)),
            "step_yr": step_yr, "max_block_lag_steps": max_lag,
            "rep_var_full": {v: float(rv_full.get(v, 0.0)) for v in VARS},
            "rep_var_by_fold": {v: [float(x) for x in rep_by_fold[:, c]]
                                for c, v in enumerate(VARS)}}
    return WithholdingLane(shape=shape, batches=tuple(batches), targets=pooled, info=info,
                           meta=meta)


def score_withholding(lane: WithholdingLane, method: Method, b_scales, base: dict,
                      model: ObservationModel,
                      progress: str | None = None) -> tuple[list[dict], dict]:
    """``(rows, predictions)``: skill and calibration of every withheld row at every amplitude.

    Each analysis assimilates under ``model`` and its predictions are scored under the same
    model (:meth:`ProxyRows.predictive`). The prior-free references are scored once.
    """
    b_scales = tuple(float(b) for b in b_scales)
    zero = np.zeros(int(np.prod(lane.shape)))
    mean, var = [], []
    tick = _Progress(progress, len(lane.batches))
    for i, batch in enumerate(lane.batches):
        y, r = batch.assim.assimilated(model)
        res = method.apply_sweep(method.prepare_sweep(batch.assim.gather, r, b_scales), y, zero)
        mean.append(np.stack([x.predict_obs(batch.target_gather) for x in res]))
        var.append(np.stack([x.predict_obs_var(batch.target_gather) for x in res]))
        tick.step(i + 1)
    pred, pred_var = lane.targets.predictive(model, np.concatenate(mean, axis=1),
                                             np.concatenate(var, axis=1))
    actual, channel = lane.targets.y, lane.info["channel"]
    rows = []
    for bj, b in enumerate(b_scales):
        at_c = {**base, "b_scale": b}
        rows += obs_skill_rows(actual, pred[bj], channel, at_c)
        rows += calibration_rows(actual, pred[bj], pred_var[bj], channel, at_c)
    for kind in ("nearest", "idw"):
        rows += obs_skill_rows(actual, lane.info[f"naive_{kind}"], channel,
                               _naive_base(base, kind))
    t = lane.targets
    predictions = {"b_scales": np.asarray(b_scales), "pred": pred, "pred_var": pred_var,
                   "actual": actual, "gather": t.gather, "sse": t.sse, "rep_var": t.rep,
                   "rho": t.rho, "resid_var": t.resid, **lane.info}
    return rows, predictions
