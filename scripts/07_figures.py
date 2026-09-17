"""Every figure and table the report prints, regenerated from stored artefacts.

Reads the lane and product directories and the raw inputs; runs no estimator, with two
exceptions that are cheap and deterministic: the lag correlation Chapter 5 plots, and the
analog ensembles Chapter 6 anatomises.

Figures are authored at their printed size. A 60-page A4 report with 2.5 cm margins has a
text width of about 16 cm, so a figure drawn wide and scaled down would render 9 pt labels
at roughly 4 pt. Everything below is sized for that width at 1:1.

Chapters 7 and 8 are not drafted, and the figures they owe need artefacts this script
already reads. They are recorded here so the list does not live only in a draft:

    7.1  test CE against b_scale per lane, each estimator at its own point
    7.2  paired contrasts between estimators
    7.3  PPE field gallery, HGAOEnKF-MT against pixel 3DVar
    7.4  per-cell CE
    7.5  realised error reduction beside claimed uncertainty reduction
    7.6  PPE skill against distance to the nearest observation
    7.7  withholding skill against distance to the nearest assimilated site
    7.8  skill by timescale
    7.9  one cell as a filtered time series
    7.10 rRMSE against the number of timescale blocks
    8.1  sites assimilated per age, prior-only ages marked
    8.2  domain mean and one Nordic Seas cell, raw and smoothed, across b_scale
    8.3  posterior uncertainty: map, and domain-mean spread against age
    8.4  D-O composite warming and the seasonal contrast
    8.5  Desroziers amplitude balance
"""

from __future__ import annotations

from dataclasses import dataclass

import _common as C

import json

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.patches import FancyBboxPatch
from scipy import stats

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.innovation import obs_cell_index
from paleoreco.assim.observations import (
    attach_site_stats,
    collapse_to_samples,
    load_observations,
    observation_site_stats,
    observations_at_age,
    representativeness_variance,
    sample_block_centres,
)
from paleoreco.data import VARS, build_prior_cube
from paleoreco.data.splits import DO_EVENT_WINDOWS, chronological_half_split

# One house style for the whole report, so no figure is visually out of family.
REPORT_WIDTH = 6.3      # inches; the text width at 2.5 cm margins on A4

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})

CELL_DEG = 5.625        # LOVECLIM grid spacing, both axes
PLATE = ccrs.PlateCarree()


def save(fig, name):
    """Write a report figure and echo where it went."""
    paths.REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    path = paths.REPORT_FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path

def cell_edges(centres: np.ndarray, clip: tuple[float, float] | None = None) -> np.ndarray:
    """Edges of a uniform cell-centred axis, one longer than the centres.

    The LOVECLIM axes give cell centres, so the outer edge of the field sits half a cell
    beyond the first and last centre. Passing the centres to an ``imshow`` extent instead
    places the field half a cell out over its whole width, which is 2.8125 degrees here.
    """
    c = np.asarray(centres, dtype=float)
    step = np.diff(c)
    if not np.allclose(step, step[0]):
        raise ValueError(f"axis is not uniform; steps run {step.min()} to {step.max()}")
    h = step[0] / 2.0
    edges = np.concatenate([c - h, [c[-1] + h]])
    return edges if clip is None else np.clip(edges, *clip)


def draw_field(ax, lats, lons, field, *, vmin, vmax, cmap="RdBu_r"):
    """Draw a (n_lat, n_lon) field on a PlateCarree axis over exact cell edges."""
    lat_e = cell_edges(lats, clip=(-90.0, 90.0))
    lon_e = cell_edges(lons)
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", transform=PLATE,
              rasterized=True)
    mesh = ax.pcolormesh(lon_e, lat_e, field, **kw)
    # The wrapped copy fills the sliver the dateline-straddling cell leaves at the left edge.
    ax.pcolormesh(lon_e - 360.0, lat_e, field, **kw)
    return mesh


# The grid's own southern edge, so no map carries a blank strip below the data.
LAT_S = -87.1875


def map_axes(ax, *, xlabel=False, ylabel=True, coastlines=True):
    """Label a PlateCarree axis so no reader has to infer what the axes are.

    ``zero_direction_label`` stays off: the prime meridian is 0 degrees, not "0 W".
    Coastlines are modern; see the note at the top of this notebook.
    """
    ax.set_extent([-180, 180, LAT_S, 90], crs=PLATE)
    if coastlines:
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3,
                       edgecolor="0.25")
    ax.set_xticks([-120, 0, 120], crs=PLATE)
    ax.set_yticks([-60, -30, 0, 30, 60], crs=PLATE)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    if xlabel:
        ax.set_xlabel("Longitude")
    if ylabel:
        ax.set_ylabel("Latitude")
    ax.tick_params(length=2, pad=1.5)

# A drawing, not an experiment. The four roles below are the visual language Chapter 6's
# panel (c) has to be drawn in, so they are named here rather than inlined: blue is the
# static archive-wide path, green the data-selected flow-dependent one, grey the observations
# and whatever consumes them directly, white the analysis. Orange is Figure 2.1's "this work"
# colour and is deliberately absent, because nothing in this chapter is ours.

STATIC = ("#dce6f2", "#3c5f8f")
FLOW = ("#d9e8d3", "#4a7a3f")
DATA = ("#ededed", "#6b6b6b")
ANALYSIS = ("#ffffff", "#333333")
# Orange is what this project adds: unused in Chapter 5, spent in Chapter 6.
OURS = ("#f7d9c4", "#c4692a")
CONNECT = "#7f7f7f"
ARROW = dict(arrowstyle="-|>,head_length=0.45,head_width=0.16", lw=0.85,
             color=CONNECT, shrinkA=0, shrinkB=0)
DASHED = (0, (4, 2.5))

# Both panels share one coordinate frame, so a box of a given height renders at the same
# physical size in each; a reader compares them by looking, and unequal scales would break
# that. The empty band under panel (a) is the price and is deliberate.
ROW0, ROW1 = 5.4, 8.8           # the main left-to-right flow
MID = (ROW0 + ROW1) / 2
OBS0, OBS1 = 10.6, 12.9         # the observation node, above the flow
MEAN_Y = 0.7                    # the prior-mean path, below it


def stage(ax, x0, x1, y0, y1, text, style, fs=6.5):
    """One labelled object in the flow. Returns its box for the arrows to attach to."""
    face, edge = style
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, mutation_aspect=0.5,
                                boxstyle="round,pad=0,rounding_size=1.0", facecolor=face,
                                edgecolor=edge, linewidth=0.85, zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
            linespacing=1.32, zorder=4)
    return (x0, x1, y0, y1)


def flow(ax, start, end, **kw):
    opts = dict(ARROW)
    opts.update(kw)
    ax.annotate("", xy=end, xytext=start, arrowprops=opts, zorder=2)


def route(ax, xs, ys, color=CONNECT, **kw):
    """An elbowed connector, drawn under every box so it can pass behind one."""
    ax.plot(xs, ys, color=color, lw=0.85, zorder=1, solid_capstyle="round", **kw)

@dataclass
class Inputs:
    """Everything the chapters read, loaded once."""

    cube: np.ndarray
    ages: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    valid: np.ndarray
    long: pd.DataFrame
    raw: pd.DataFrame
    recon: dict
    recon_cfg: dict


def load_inputs(n_ages: int | None = None) -> Inputs:
    """The prior cube, the proxy table with site climatology, and the stored product.

    ``n_ages`` truncates the archive to match a reduced run, so the figures read the same
    span the artefacts were built over rather than pairing a full archive with them.
    """
    p = build_prior_cube(prior_csv=str(paths.PRIOR_CSV), cache_path=str(paths.PRIOR_CACHE))
    cube, ages = p["cube"], p["ages"]
    if n_ages is not None:
        cube, ages = cube[:n_ages], ages[:n_ages]
    long = load_observations(str(paths.OBSERVATION_CSV))
    long = attach_site_stats(long, observation_site_stats(collapse_to_samples(long)))
    d = paths.product_dir()
    recon = np.load(d / "reconstruction_fields.npz")
    recon_cfg = json.load(open(d / "reconstruction_config.json"))
    inp = Inputs(cube=cube, ages=ages, lats=p["lats"], lons=p["lons"],
                 valid=p["valid"], long=long,
                 raw=pd.read_csv(str(paths.OBSERVATION_CSV)),
                 recon=recon, recon_cfg=recon_cfg)
    print(f"prior  : {inp.cube.shape} over {inp.ages.min()}-{inp.ages.max()} yr BP at "
          f"{int(np.unique(np.diff(inp.ages))[0])} yr spacing")
    print(f"proxies: {long['site'].nunique()} sites, {long['sample'].nunique()} samples, "
          f"{long['age'].nunique()} distinct ages")
    print(f"product: {recon['mean_anom'].shape} at b_scale {list(recon['b_scales'])}, "
          f"estimator {recon_cfg['estimator']}")
    return inp

def chapter1(inp: Inputs) -> None:
    """Figure 1.1: the reconstruction problem, and what this produces."""
    cube, ages, lats, lons = inp.cube, inp.ages, inp.lats, inp.lons
    long, recon = inp.long, inp.recon

    AGE_FIG11 = 38175          # onset of D-O event 8, on the archive's own age axis
    CHAN_FIG11 = 0             # MTCO
    B_FIG11 = 5.0              # the product's operating point

    ti = int(np.argmin(np.abs(recon["ages"] - AGE_FIG11)))
    bi = int(np.argmin(np.abs(recon["b_scales"] - B_FIG11)))
    age = int(recon["ages"][ti])
    safe_valid, clim = recon["safe_valid"], recon["clim_mean"]

    # The simulation's own state at this age, in the same anomaly frame as the posterior.
    sim_anom = cube[np.searchsorted(ages, age)].astype(np.float64) - clim
    post_anom = recon["mean_anom"][bi, ti]

    # The proxies at this age, in each site's own anomaly frame.
    o = observations_at_age(long, age)
    sel = (o["channel"] == VARS[CHAN_FIG11]) & np.isfinite(o["my"])
    obs_lat, obs_lon = o["lat"][sel], o["lon"][sel]
    obs_anom = (o["y"][sel] - o["my"][sel]).astype(np.float64)
    n_cells = len(lats) * len(lons)
    cells_hit = np.unique(obs_cell_index(o["lat"], o["lon"], o["channel"], lats, lons)[sel] % n_cells)

    print(f"age {age} yr BP | {sel.sum()} {VARS[CHAN_FIG11].upper()} sites reaching "
          f"{len(cells_hit)} of {n_cells} cells ({100 * len(cells_hit) / n_cells:.1f}%)")
    print(f"proxy anomaly rms {np.sqrt((obs_anom ** 2).mean()):.2f} degC | "
          f"LOVECLIM at those cells {np.sqrt((sim_anom[CHAN_FIG11].ravel()[cells_hit] ** 2).mean()):.2f} degC")

    V = 4.0
    masked = lambda f: np.where(safe_valid, f, np.nan)

    # Panel titles carry identity only; what each panel means belongs in the LaTeX caption,
    # where it can be read at full size rather than squeezed into a 2-inch column.
    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH, 1.45), subplot_kw={"projection": PLATE},
                             constrained_layout=True)

    mesh = draw_field(axes[0], lats, lons, masked(sim_anom[CHAN_FIG11]), vmin=-V, vmax=V)
    axes[0].set_title("(a) LOVECLIM simulation")

    axes[1].set_facecolor("0.94")
    axes[1].scatter(obs_lon, obs_lat, c=obs_anom, s=7, cmap="RdBu_r", vmin=-V, vmax=V,
                    edgecolor="k", linewidth=0.2, transform=PLATE, zorder=3)
    axes[1].set_title("(b) pollen reconstructions")

    draw_field(axes[2], lats, lons, masked(post_anom[CHAN_FIG11]), vmin=-V, vmax=V)
    axes[2].scatter(obs_lon, obs_lat, s=0.8, c="k", linewidth=0, transform=PLATE, zorder=3)
    axes[2].set_title("(c) HGAOEnKF-MT posterior")

    for i, ax in enumerate(axes):
        map_axes(ax, ylabel=(i == 0))
    fig.supxlabel("Longitude", fontsize=7.5, y=0.02)

    cb = fig.colorbar(mesh, ax=axes, shrink=0.95, pad=0.012, aspect=18, extend="both")
    cb.set_label(f"{VARS[CHAN_FIG11].upper()} anomaly (°C)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    save(fig, "fig01_01_reconstruction_problem")

def chapter2() -> None:
    """Figure 2.1: the method-family map. A drawing; it reads nothing."""
    # Both axes ask one question about two different objects, so they share their vocabulary where
    # they can: the selected states are the same states on both, which is exactly what an analog
    # method does with them. Each axis is ordinal and runs in the order the chapter introduces the
    # families, so a position carries membership, not a distance.
    MEAN_AXIS = [
        "the archive\nclimatology",
        "archive states selected\nby the observations",
        "a forecast from\nthe previous age",
        "a draw from a\ntrained density",
    ]
    COV_AXIS = [
        "the whole archive",
        "one random draw,\nreused at every age",
        "the selected\nstates alone",
        "the selected states\nand the whole\narchive, blended",
        "a trained density",
    ]

    # Works are cited in the LaTeX caption, not in the boxes: the report numbers its references, so
    # an author-year list here buys nothing, and those long labels were what forced every collision
    # the layout used to work around. HGAOEnKF and HGAOEnKF-MT straddle their shared column so the
    # connector between them has somewhere to go; everything else sits on its own tick.
    METHODS = [
        (0.00, 0, "3DVar", "published"),
        (0.00, 1, "Offline EnSRF\npaleoDA", "published"),
        (1.00, 0, "AOEnKF-B", "published"),
        (1.00, 2, "Analog offline EnKF", "published"),
        (0.55, 3, "HGAOEnKF", "published"),
        (1.30, 3, "HGAOEnKF-MT\n(this work)", "ours"),
        (2.05, 3, "Online paleoDA", "published"),
        (3.00, 4, "Generative DA", "published"),
    ]

    FACE = {"published": "#dce6f2", "ours": "#f7d9c4"}
    EDGE = {"published": "#3c5f8f", "ours": "#c4692a"}

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.85), constrained_layout=True)
    for y in range(len(COV_AXIS)):
        ax.axhline(y, color="0.93", lw=0.6, zorder=0)
    for x in range(len(MEAN_AXIS)):
        ax.axvline(x, color="0.93", lw=0.6, zorder=0)

    boxes = {}
    for x, y, label, kind in METHODS:
        boxes[label.split("\n")[0]] = ax.annotate(
            label, (x, y), ha="center", va="center", zorder=3,
            fontsize=6.8, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=FACE[kind],
                      edgecolor=EDGE[kind], linewidth=0.8))

    ax.set_xlim(-0.52, 3.52)
    ax.set_ylim(-0.40, 4.40)
    ax.set_xticks(range(len(MEAN_AXIS)), MEAN_AXIS, fontsize=6.5)
    ax.set_yticks(range(len(COV_AXIS)), COV_AXIS, fontsize=6.5, linespacing=1.3)
    ax.set_xlabel("What the prior mean is formed from", fontsize=7.5, labelpad=4)
    ax.set_ylabel("What the prior covariance\nis estimated from", fontsize=7.5, labelpad=4)
    ax.tick_params(length=0)
    for _s in ("left", "bottom"):
        ax.spines[_s].set_color("0.6")
        ax.spines[_s].set_linewidth(0.7)


    def box_edges(ann):
        """Left and right edge of a drawn annotation's box, in data coordinates.

        Hand-placing the connector meant guessing where the text ended, which left it stubbed
        against one border and floating off the other. Measuring the drawn patch lets it start and
        finish exactly on the two borders whatever the font metrics turn out to be.
        """
        fig.canvas.draw()
        bb = ann.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1, _) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x0, x1


    # A hair of overlap rather than an exact meeting, so no antialiasing seam shows at either end.
    # The boxes are drawn above the connector, so none of it is visible inside them.
    BITE = 0.004
    _, hga_right = box_edges(boxes["HGAOEnKF"])
    mt_left, _ = box_edges(boxes["HGAOEnKF-MT"])

    # The contribution keeps the analog mean and keeps the hybrid gain, so it belongs in the same
    # row and the same column band as the estimator it extends; only the content of the
    # flow-dependent covariance changes. Putting it anywhere else on this map would misstate it -
    # a box offset upward reads as belonging to the row above, which here is the learned prior the
    # report explicitly rejects.
    ax.annotate("", xy=(mt_left + BITE, 3.0), xytext=(hga_right - BITE, 3.0), zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.18",
                                lw=0.9, color=EDGE["ours"], shrinkA=0, shrinkB=0))

    save(fig, "fig02_01_method_family_map")

def chapter3(inp: Inputs) -> None:
    """Figures 3.1-3.3 and Table 3.1: the data and its character."""
    cube, ages, lats, lons, valid = inp.cube, inp.ages, inp.lats, inp.lons, inp.valid
    long, raw = inp.long, inp.raw

    # The raw observation table as well as the long form: the dating range and the per-age site
    # count are columns the long form drops, and both are Chapter 3 quantities.
    raw = pd.read_csv(str(paths.OBSERVATION_CSV))
    NCELL = len(lats) * len(lons)
    mask_valid = lambda f: np.where(valid, f, np.nan)


    def lag1_autocorr(x):
        """Per-cell lag-1 autocorrelation along the age axis.

        Pearson between the series and its own one-step shift, each half demeaned separately.
        Notebook 01 uses the biased form (one common mean, divided by the full sum of squares),
        which reads 0.003-0.005 lower per channel. The difference is immaterial to the chapter,
        which quotes the correlation itself and derives no other quantity from it.
        """
        a, b = x[:-1], x[1:]
        a = a - a.mean(0)
        b = b - b.mean(0)
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


    mean_all, std_all = cube.mean(axis=0), cube.std(axis=0)
    r1 = np.stack([lag1_autocorr(cube[:, 0].astype(np.float64)),
                   lag1_autocorr(cube[:, 1].astype(np.float64))])
    r1_med = float(np.median(r1[:, valid]))

    COLS = [("climatology", mean_all, dict(vmin=-45, vmax=30, cmap="RdBu_r"), "°C"),
            ("temporal s.d.", std_all, dict(vmin=0, vmax=3.0, cmap="viridis"), "°C"),
            ("lag-1 autocorrelation", r1, dict(vmin=0.7, vmax=1.0, cmap="magma"), r"$\rho_1$")]

    fig, axes = plt.subplots(2, 3, figsize=(REPORT_WIDTH, 2.60), subplot_kw={"projection": PLATE},
                             constrained_layout=True)
    for j, (label, field, kw, unit) in enumerate(COLS):
        for c, name in enumerate(VARS):
            mesh = draw_field(axes[c, j], lats, lons, mask_valid(field[c]), **kw)
            axes[c, j].set_title(f"({'adbecf'[2 * j + c]}) {name.upper()} {label}", fontsize=6.8)
            map_axes(axes[c, j], ylabel=(j == 0))
        cb = fig.colorbar(mesh, ax=axes[:, j], orientation="horizontal", shrink=0.92, pad=0.02,
                          aspect=22, extend="both" if j else "min")
        cb.set_label(unit, fontsize=6.8)
        cb.ax.tick_params(labelsize=6, length=2)
    save(fig, "fig03_01_prior_structure")

    print(f"per-cell s.d. median {np.median(std_all[0][valid]):.2f} / "
          f"{np.median(std_all[1][valid]):.2f} degC, 90th pct "
          f"{np.percentile(std_all[0][valid], 90):.2f} / {np.percentile(std_all[1][valid], 90):.2f}")
    print(f"per-cell rho1 median {r1_med:.4f} (mtco {np.median(r1[0][valid]):.3f}, "
          f"mtwa {np.median(r1[1][valid]):.3f}), 5-95 pct "
          f"[{np.percentile(r1[:, valid], 5):.3f}, {np.percentile(r1[:, valid], 95):.3f}]")

    gidx = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                          long["channel"].to_numpy(), lats, lons) % NCELL
    reached = np.zeros(NCELL, dtype=bool)
    reached[np.unique(gidx)] = True
    reached = reached.reshape(len(lats), len(lons))

    sites = raw.groupby("siteID").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                      n=("sampleID", "nunique"))
    per_age = raw.groupby("age")["siteID"].nunique()
    free_ages = np.setdiff1d(ages, per_age.index.to_numpy())
    date_range = (raw.drop_duplicates("sampleID").eval("age_oldest - age_youngest")).to_numpy()
    lag_yr = sample_block_centres(long).eval("age - centre").abs().to_numpy()

    fig = plt.figure(figsize=(REPORT_WIDTH, 4.95), layout="constrained")
    sub_top, sub_bot = fig.subfigures(2, 1, height_ratios=[2.35, 2.60])

    ax = sub_top.subplots(subplot_kw={"projection": PLATE})
    # Fix the extent and freeze it before anything is drawn. The wrapped mesh at lon-360 would
    # otherwise re-autoscale the axes and the layout solver would size the box to that wider
    # aspect rather than to the world map actually shown.
    ax.set_extent([-180, 180, -58, 80], crs=PLATE)
    ax.set_autoscale_on(False)
    draw_field(ax, lats, lons, np.where(reached, 1.0, np.nan), vmin=0, vmax=3.2, cmap="Greys")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, edgecolor="0.25")
    sc = ax.scatter(sites["lon"], sites["lat"], c=sites["n"], s=9, cmap="viridis",
                    norm=plt.matplotlib.colors.LogNorm(), edgecolor="k", linewidth=0.25,
                    transform=PLATE, zorder=3)
    ax.set_xticks([-120, 0, 120], crs=PLATE)
    ax.set_yticks([-30, 0, 30, 60], crs=PLATE)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_ylabel("Latitude")
    ax.tick_params(length=2, pad=1.5)
    ax.set_title(f"(a) the {len(sites)} sites, and the {int(reached.sum())} of {NCELL} "
                 "grid cells they reach (shaded)")
    cb = sub_top.colorbar(sc, ax=ax, shrink=0.94, pad=0.012, aspect=15)
    cb.set_label("samples per site", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)

    gs = sub_bot.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    axb = sub_bot.add_subplot(gs[0, :])
    for lo_, hi in DO_EVENT_WINDOWS.values():
        axb.axvspan(lo_ / 1000, hi / 1000, color="0.87", lw=0, zorder=0)
    axb.plot(per_age.index.to_numpy() / 1000, per_age.to_numpy(), lw=0.7, color="#2b6cb0")
    axb.axhline(per_age.median(), color="C1", lw=0.8, ls="--")
    axb.plot(free_ages / 1000, np.full_like(free_ages, 6.0, dtype=float), lw=0, marker="|",
             ms=5, color="C3")
    axb.annotate(f"{len(free_ages)} ages with no pollen", (free_ages.mean() / 1000, 6.0),
                 textcoords="offset points", xytext=(-6, 4), ha="right", va="bottom",
                 fontsize=6.3, color="C3")
    axb.set_xlim(ages.max() / 1000, ages.min() / 1000 - 0.4)
    axb.set_ylim(0, 138)
    axb.set_xlabel("Age (ka BP)")
    axb.set_ylabel("Sites reporting")
    axb.set_title("(b) sites reporting at each age; the eight D–O windows shaded, "
                  f"the median of {int(per_age.median())} dashed")

    axc = sub_bot.add_subplot(gs[1, 0])
    axc.hist(date_range, bins=np.arange(0, 8001, 250), color="#4a7fb5")
    axc.axvline(np.median(date_range), color="C1", lw=1.0, ls="--",
                label=f"median {np.median(date_range):.0f} yr")
    axc.set_xlim(0, 8000)
    axc.set_ylim(0, 1550)
    axc.set_xlabel("Dating range of a sample (yr)")
    axc.set_ylabel("Samples")
    axc.legend(loc="upper right")
    axc.set_title("(c) chronological uncertainty")

    axd = sub_bot.add_subplot(gs[1, 1])
    axd.hist(lag_yr, bins=np.arange(0, 801, 25), color="#4a7fb5")
    axd.axvline(np.median(lag_yr), color="C1", lw=1.0, ls="--",
                label=f"median {np.median(lag_yr):.0f} yr")
    axd.axvline(np.percentile(lag_yr, 90), color="C3", lw=1.0, ls=":",
                label=f"90th pct. {np.percentile(lag_yr, 90):.0f} yr")
    axd.set_xlim(0, 800)
    axd.set_ylim(0, 17500)
    axd.set_xlabel("|age − block centre| (yr)")
    axd.set_ylabel("Assimilated rows")
    axd.legend(loc="upper right")
    axd.set_title("(d) distance in time at the point of use")
    save(fig, "fig03_02_proxy_network")

    nh = int((sites["lat"] > 0).sum())
    print(f"{len(sites)} sites ({nh} NH, {100 * nh / len(sites):.1f}%), "
          f"lat {sites['lat'].min():.2f} to {sites['lat'].max():.2f}, "
          f"lon {sites['lon'].min():.2f} to {sites['lon'].max():.2f}")
    print(f"{int(reached.sum())}/{NCELL} cells ({100 * reached.sum() / NCELL:.1f}%); "
          f"sites per age median {int(per_age.median())} (range {per_age.min()}-{per_age.max()}); "
          f"{len(free_ages)} empty ages, {free_ages.min()}-{free_ages.max()} yr BP")
    print(f"dating range: median {np.median(date_range):.0f} yr, "
          f"IQR {np.percentile(date_range, 25):.0f}-{np.percentile(date_range, 75):.0f}, "
          f"max {date_range.max():.0f}")
    print(f"lag at use: median {np.median(lag_yr):.0f} yr, p90 {np.percentile(lag_yr, 90):.0f}, "
          f"max {lag_yr.max():.0f}, over {lag_yr.size} rows")

    def snap(values, axis):
        """Index of the nearest entry of a sorted ascending axis, for each value."""
        i = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
        return np.where(values - axis[i - 1] < axis[i] - values, i - 1, i)


    ai = snap(raw["age"].to_numpy(), ages)
    la, lo_ = snap(raw["latitude"].to_numpy(), lats), snap(raw["longitude"].to_numpy(), lons)
    resid = {"mtco": cube[ai, 0, la, lo_] - raw["mtco"].to_numpy(),
             "mtwa": cube[ai, 1, la, lo_] - raw["mtwa"].to_numpy()}

    # Two decorrelation curves, both of the anomaly cube with the archive mean removed.
    #
    # The domain-mean series is one number per age, so its autocorrelation says how fast the
    # archive's overall level changes. The state-vector curve is the mean cosine between whole
    # 4096-dimensional anomaly states that many years apart, which is the object the covariance
    # is estimated from and so the one Section 3.3 argues about. They are reported together
    # because the second is the stronger statement and the first is the more familiar one.
    A = (cube - cube.mean(axis=0, keepdims=True)).reshape(len(ages), -1).astype(np.float64)
    dm = (cube - cube.mean(axis=0, keepdims=True))[:, :, valid].mean(axis=(1, 2))
    dm = dm - dm.mean()
    lags = np.arange(0, 81)
    acf = np.array([1.0 if k == 0 else float((dm[:-k] * dm[k:]).sum() / (dm ** 2).sum())
                    for k in lags])

    # Mean correlation between whole state vectors at each separation: the mean of each
    # off-diagonal of the Gram matrix of row-normalised anomaly states.
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    G = An @ An.T
    svc = np.array([float(np.mean(np.diagonal(G, k))) for k in range(len(ages))])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 1.75), constrained_layout=True)
    for ch, col in zip(VARS, ("#2b6cb0", "#c4692a")):
        axa.hist(resid[ch], bins=np.arange(-15, 35.1, 1.0), histtype="step", lw=1.0, color=col,
                 label=f"{ch.upper()}, median {np.median(resid[ch]):+.2f} °C")
    axa.axvline(0, color="k", lw=0.7)
    axa.set_xlim(-15, 35)
    axa.set_ylim(0, 8600)
    axa.set_xlabel("LOVECLIM − pollen (°C)")
    axa.set_ylabel("Co-located rows")
    axa.legend(loc="upper right")
    axa.set_title("(a) the model–proxy offset")

    axb.plot(lags * 25, svc[:81], lw=1.2, color="#2b6cb0", label="whole state vector")
    axb.plot(lags * 25, acf, lw=1.0, ls="--", color="#c4692a", label="domain-mean anomaly")
    axb.axhline(0, color="k", lw=0.6)
    for yr in (500, 1000):
        k = yr // 25
        axb.plot([yr], [svc[k]], marker="o", ms=3, color="C3")
        axb.annotate(f"{svc[k]:.2f}", (yr, svc[k]), textcoords="offset points", xytext=(4, 5),
                     fontsize=6.3, color="C3")
    axb.set_xlim(0, 2000)
    axb.set_ylim(-0.05, 1.18)
    axb.set_xlabel("Separation in time (yr)")
    axb.set_ylabel("Correlation")
    axb.legend(loc="upper right")
    axb.set_title(f"(b) how fast the archive renews itself; "
                  f"per-cell median $\\rho_1$ = {r1_med:.3f}")
    save(fig, "fig03_03_offset_and_scarcity")

    for ch in VARS:
        r = resid[ch]
        print(f"{ch}: median {np.median(r):+.2f} degC, "
              f"MAD {np.median(np.abs(r - np.median(r))):.2f}, "
              f"5-95% [{np.percentile(r, 5):+.1f}, {np.percentile(r, 95):+.1f}], n={r.size}")
    print(f"domain-mean autocorrelation at 25 / 100 / 1000 yr: "
          f"{acf[1]:.4f} / {acf[4]:.4f} / {acf[40]:.4f}")
    print(f"state-vector correlation at 25 / 500 / 1000 / 2000 yr: "
          f"{svc[1]:.3f} / {svc[20]:.3f} / {svc[40]:.3f} / {svc[80]:.3f}")
    _efold = int(np.argmax(svc < np.exp(-1))) * 25
    _zero = int(np.argmax(svc <= 0)) * 25
    print(f"state-vector correlation falls below 1/e at {_efold} yr and reaches zero at {_zero} yr")

    D = len(VARS) * len(lats) * len(lons)
    samples = collapse_to_samples(long)          # one row per (site, channel, sample)

    print("--- Table 3.1: the prior ---")
    print(f"grid {len(lats)}x{len(lons)} at {CELL_DEG} deg; channels {VARS}; D = {D}")
    print(f"{len(ages)} states, {ages.min()}-{ages.max()} yr BP at "
          f"{int(np.unique(np.diff(ages))[0])} yr spacing; "
          f"valid cells {int(valid.sum())}/{valid.size}")
    print(f"covariance entries {D * D:,}; free parameters D(D+1)/2 = {D * (D + 1) // 2:,}")

    # What a sample covariance from N states can possibly be. The rank bound N-1 is algebraic and
    # needs no assumption about the archive; it is what Section 3.3 leads with, in place of the
    # effective-sample-size figure an earlier draft carried. The numerical rank is printed beside
    # it to confirm the bound is attained rather than merely bounding.
    for label, idx in (("all states", np.arange(len(ages))),
                       ("prior half", chronological_half_split(ages, stride=1)[0])):
        Ac = cube[idx].reshape(len(idx), -1).astype(np.float64)
        Ac = Ac - Ac.mean(0)
        ev = np.linalg.svd(Ac, compute_uv=False) ** 2 / (len(idx) - 1)
        cum = np.cumsum(ev) / ev.sum()
        print(f"  {label}: N = {len(idx)}, rank(B_hat) <= N-1 = {len(idx) - 1} of D = {D} "
              f"({(len(idx) - 1) / D:.1%} of directions); numerical rank "
              f"{int((ev > ev[0] * 1e-10).sum())}; "
              f"eigenvalues holding 90/99% of the trace: "
              f"{int(np.searchsorted(cum, 0.90)) + 1}/{int(np.searchsorted(cum, 0.99)) + 1}")
    print("--- Table 3.1: the observations ---")
    print(f"{long['site'].nunique()} sites, {long['sample'].nunique()} samples, "
          f"{len(long):,} long-format rows, {long['age'].nunique()} of {len(ages)} ages covered")
    print(f"median sse: " + ", ".join(
        f"{ch} {samples[samples.channel == ch]['sse'].median():.2f} degC^2" for ch in VARS))

    # Representativeness variance: co-cell, co-age proxy pairs with each proxy's own stated error
    # removed. Estimated over the assimilated rows, which is how run_withholding estimates it.
    rep_cell = obs_cell_index(long["lat"].to_numpy(), long["lon"].to_numpy(),
                              long["channel"].to_numpy(), lats, lons) % NCELL
    rep_var = representativeness_variance(long, rep_cell)
    print("representativeness variance: " + ", ".join(f"{k} {v:.2f} degC^2" for k, v in rep_var.items()))

    # The per-site anomaly frame against a full-record climatology, with the model as stand-in.
    flat = cube.reshape(len(ages), -1).astype(np.float64)
    full_mean = flat.mean(axis=0)
    samples_g = samples.assign(g=obs_cell_index(samples["lat"].to_numpy(), samples["lon"].to_numpy(),
                                                samples["channel"].to_numpy(), lats, lons))
    offset = np.array([flat[np.clip(np.searchsorted(ages, s["age"].to_numpy()), 0, len(ages) - 1),
                            int(s["g"].iloc[0])].mean() - full_mean[int(s["g"].iloc[0])]
                       for _, s in samples_g.groupby(["site", "channel"])])
    span = samples.groupby(["site", "channel"])["age"].agg(lambda a: (a.max() - a.min()))
    print(f"site-frame offset vs full climatology: rms {np.sqrt((offset ** 2).mean()):.3f} degC, "
          f"median |.| {np.median(np.abs(offset)):.3f}, over {offset.size} site-channel records; "
          f"{(span / (ages.max() - ages.min()) < 0.5).mean():.2f} of records span under half the interval")

def chapter4() -> None:
    """Lane sizes and the prior-free reference points, printed for the tables.

    One reading rule governs every withholding number the report quotes. That lane writes
    a row per fold and a row pooled over folds at ``fold = -1``. The pooled row is the one
    to quote; averaging it together with the per-fold rows shifts every withholding figure
    by enough to change a reported CE.
    """
    pooled = -1
    three = ex.ESTIMATOR_3DVAR
    d_ppe = paths.run_dir(three, paths.LANE_PPE)
    d_traj = paths.run_dir(three, paths.LANE_TRAJECTORY)
    d_wh = paths.run_dir(three, paths.LANE_WITHHOLDING)

    ppe = json.load(open(d_ppe / "ppe_config.json"))
    traj = json.load(open(d_traj / "trajectory_config.json"))
    wh = json.load(open(d_wh / "withholding_random_config.json"))

    print("--- Table 4.1: lane sizes ---")
    # Runs written before the field recorded a count instead; both arms drew then.
    both_drawn = {"selection": ex.DRAWN_NETWORK, "test": ex.DRAWN_NETWORK}
    ppe_net = ppe.get("networks", both_drawn)
    traj_net = traj.get("networks", both_drawn)

    print(f"pseudo-proxy : {ppe['n_truths']} truths x 2 observation realisations x "
          f"{ppe['n_noise']} noise draw; selection network {ppe_net['selection']}, "
          f"test {ppe_net['test']}; prior = "
          f"{ppe['prior_meta']['n_prior_ages']} states of the {ppe['prior_half']} half, "
          f"truths from the other; R = diag(sse)")
    print(f"trajectory   : {traj['n_covered_ages']} consecutive ages reconstructed over "
          f"{traj['scored_ages']} yr BP ({traj['n_skipped_ages']} skipped for carrying no "
          f"usable network); selection network {traj_net['selection']}, test "
          f"{traj_net['test']}; each observation read at its own offset in time; "
          f"prior = {traj['prior_meta']['n_prior_ages']} states")
    print(f"withholding  : {wh['k_folds']} {wh['fold_kind']} site folds; prior = "
          f"{wh['prior_meta']['n_prior_ages']} states spanning every age; "
          f"R = diag(sse + rep_var), rep_var {wh['rep_var_full']}")

    # `nearest` copies each cell's nearest observation, `idw` weights by inverse square
    # distance; neither uses the model. They are what makes a withholding CE readable: on
    # that lane CE is small by construction, so it is read against these, not against 1.
    rows = []
    for label, directory, lane in (("pseudo-proxy", d_ppe, "ppe"),
                                   ("trajectory", d_traj, "trajectory"),
                                   ("withholding", d_wh, "withholding_random")):
        m = pd.read_csv(directory / "metrics.csv")
        m = m[(m.channel == "pooled") & (m.do_event == "all") & (m.fold == pooled)
              & (m.lane == lane) & (m.split == "test")
              & (m.metric.isin(["rrmse", "ce", "amplitude"]))]
        for method in ("nearest", "idw"):
            sub = m[m.method == method]
            if len(sub):
                rows.append({"lane": label, "method": method,
                             **sub.groupby("metric")["value"].mean().to_dict()})
    print("\n--- Section 4.2: prior-free reference points, test split ---")
    print(pd.DataFrame(rows).set_index(["lane", "method"]).round(4).to_string())

def chapter5(inp: Inputs) -> None:
    """Figures 5.1-5.2: the two published estimators, and observation staleness."""
    cube, ages, lats, lons = inp.cube, inp.ages, inp.lats, inp.lons
    long = inp.long

    fig, axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 3.2), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.3, 13.2)
        ax.axis("off")

    # --- (a) pixel 3DVar: one covariance, one gain, a background that never moves --------------
    ax = axes[0]
    ax.text(0, 13.1, "(a)  3DVar", fontsize=7.4, fontweight="bold", va="top")
    archive = stage(ax, 1, 17, ROW0, ROW1, "LOVECLIM\narchive", STATIC)
    cov = stage(ax, 25, 52, ROW0, ROW1,
                "one static covariance $\\mathbf{B}$,\nfrom every archive state", STATIC)
    gain = stage(ax, 59, 77, ROW0, ROW1, "one gain", STATIC)
    out = stage(ax, 84, 100, ROW0, ROW1, "analysis", ANALYSIS)
    obs = stage(ax, 59, 77, OBS0, OBS1, "observations $\\mathbf{y}$, $\\mathbf{R}$", DATA)
    for left, right in ((archive, cov), (cov, gain), (gain, out)):
        flow(ax, (left[1], MID), (right[0], MID))
    flow(ax, (68, OBS0), (68, ROW1))
    route(ax, [9, 9, 92], [ROW0, MEAN_Y, MEAN_Y], ls=DASHED)
    flow(ax, (92, MEAN_Y), (92, ROW0), ls=DASHED)
    ax.text(50, MEAN_Y + 0.55, "prior mean: the archive climatology, zero in anomaly units",
            fontsize=6.3, color="#5c5c5c", ha="center", va="bottom")

    # --- (b) HGAOEnKF: the observations choose the ensemble, and there are two of everything ---
    ax = axes[1]
    ax.text(0, 13.1, "(b)  HGAOEnKF", fontsize=7.4, fontweight="bold", va="top")
    archive = stage(ax, 1, 15, ROW0, ROW1, "LOVECLIM\narchive", STATIC)
    select = stage(ax, 19, 36, ROW0, ROW1,
                   "score every state\nagainst $\\mathbf{y}$;\nkeep the best $k$", DATA, fs=6.3)
    cov_flow = stage(ax, 40, 64, 7.5, 9.9,
                     "flow-dependent covariance,\nfrom the $k$ selected states", FLOW, fs=6.3)
    cov_static = stage(ax, 40, 64, 3.5, 5.9,
                       "static covariance $\\mathbf{B}$,\nfrom every archive state", STATIC, fs=6.3)
    gain = stage(ax, 68, 84, ROW0, ROW1, "two gains,\nsummed at $\\alpha$", STATIC)
    out = stage(ax, 87, 100, ROW0, ROW1, "analysis", ANALYSIS)
    obs = stage(ax, 38, 66, OBS0, OBS1, "observations $\\mathbf{y}$, $\\mathbf{R}$", DATA)
    flow(ax, (archive[1], MID), (select[0], MID))
    flow(ax, (44, OBS0), (30, ROW1))
    flow(ax, (60, OBS0), (76, ROW1))
    flow(ax, (select[1], 7.6), (cov_flow[0], 8.7), color=FLOW[1])
    # The static covariance is built from the whole archive and never sees the selection, so its
    # connector has to leave the archive before the scoring box rather than pass through it.
    route(ax, [8, 8, 38, 38], [ROW0, 2.0, 2.0, 4.7])
    flow(ax, (38, 4.7), (cov_static[0], 4.7))
    flow(ax, (cov_flow[1], 8.7), (gain[0], 7.9), color=FLOW[1])
    flow(ax, (cov_static[1], 4.7), (gain[0], 6.3))
    flow(ax, (gain[1], MID), (out[0], MID))
    route(ax, [27, 27, 93.5], [ROW0, MEAN_Y, MEAN_Y], color=FLOW[1], ls=DASHED)
    flow(ax, (93.5, MEAN_Y), (93.5, ROW0), color=FLOW[1], ls=DASHED)
    ax.text(60, MEAN_Y + 0.55, "prior mean: the mean of the $k$ selected states",
            fontsize=6.3, color=FLOW[1], ha="center", va="bottom")

    save(fig, "fig05_01_estimator_schematic")

    # Imported here rather than at the top: this is the only figure in the report that reads the
    # archive along its age axis rather than across the grid, and keeping the import beside it
    # says so.
    from paleoreco.assim.background import temporal_structure_function

    RHO_FLOOR = 0.05                # observations.temporal_terms floors rho here
    MEDIAN_LAG, P90_LAG = 175.0, 537.5      # Section 3.2.2, over all 127,646 assimilated rows
    MARK = "#c4692a"

    # The prior half alone, which is the leakage guard the trajectory lane runs under: on a lane
    # whose truth is a model state, later ages would leak that truth into the operator that
    # corrects for the distance to it.
    prior_idx, _ = chronological_half_split(ages, stride=1)
    S, cell_var = temporal_structure_function(cube, prior_idx, max_lag=58)
    lag_yr = np.arange(S.shape[0]) * 25.0
    rho = np.clip(1.0 - S / (2.0 * cell_var[None, :]), RHO_FLOOR, 1.0)

    n_cells = len(lats) * len(lons)


    def flat_index(lat, lon, channel):
        """Row of the flattened state vector nearest one (lat, lon) in one channel."""
        i = int(np.argmin(np.abs(np.asarray(lats) - lat)))
        j = int(np.argmin(np.abs(np.asarray(lons) - lon)))
        return channel * n_cells + i * len(lons) + j


    # Three cells spanning the range the shaded band summarises: the North Atlantic, where the
    # D-O signal and most of the archive's variance live and where a state renews itself fastest;
    # a quiet tropical cell; and a mid-latitude Southern Hemisphere cell in the other channel.
    SHOWN = [("North Atlantic, MTCO", flat_index(62, -20, 0), "#3c5f8f"),
             ("Tropical Africa, MTCO", flat_index(2, 20, 0), "#4a7a3f"),
             ("Tasman Sea, MTWA", flat_index(-40, 145, 1), "#8b5aa8")]

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.4), constrained_layout=True)

    q10, q50, q90 = np.percentile(rho, [10, 50, 90], axis=1)
    ax.fill_between(lag_yr, q10, q90, color="0.89", lw=0, label="all cells, 10th–90th pct.")
    ax.plot(lag_yr, q50, color="0.30", lw=1.5, label="all cells, median")
    for name, idx, colour in SHOWN:
        ax.plot(lag_yr, rho[:, idx], color=colour, lw=1.0, label=name)
    ax.axhline(RHO_FLOOR, color="0.55", lw=0.7, ls=(0, (4, 3)))
    ax.text(1435, RHO_FLOOR + 0.015, "floor", color="0.45", fontsize=6.2, ha="right", va="bottom")
    for x, label in ((MEDIAN_LAG, "median lag"), (P90_LAG, "90th pct.")):
        ax.plot([x, x], [0, 1.0], color=MARK, lw=0.8, ls=":")
        ax.text(x + 18, 1.045, label, color=MARK, fontsize=6.3, va="center", ha="left")
    ax.set_xlim(0, 1450)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("lag between a sample and the analysis age (yr)")
    ax.set_ylabel(r"lag correlation $\rho$")
    # One panel, so no panel label: what it shows belongs in the LaTeX caption.
    ax.legend(loc="lower left", frameon=False, handlelength=1.4, borderpad=0.1,
              labelspacing=0.22, fontsize=6.3, bbox_to_anchor=(-0.005, 0.06))

    save(fig, "fig05_02_lag_correlation")

    # The three numbers Section 5.3 quotes, so the prose is read off the same object as the figure.
    for name, idx, _ in SHOWN:
        print(f"{name:24s} rho at {MEDIAN_LAG:.0f} yr {np.interp(MEDIAN_LAG, lag_yr, rho[:, idx]):.2f}, "
              f"at {P90_LAG:.0f} yr {np.interp(P90_LAG, lag_yr, rho[:, idx]):.2f}")
    print(f"{'all cells, median':24s} rho at {MEDIAN_LAG:.0f} yr {np.interp(MEDIAN_LAG, lag_yr, q50):.2f}, "
          f"at {P90_LAG:.0f} yr {np.interp(P90_LAG, lag_yr, q50):.2f}, "
          f"at 1000 yr {np.interp(1000, lag_yr, q50):.2f}")

def chapter6(inp: Inputs) -> None:
    """Figures 6.1-6.3: what the flow stack does, and why."""
    cube, ages, lats, lons, valid = inp.cube, inp.ages, inp.lats, inp.lons, inp.valid
    long, raw = inp.long, inp.raw

    # Shared inputs for Figures 6.1 and 6.2: the analog pool, the whitened spectrum of every
    # network the trajectory lane assimilates, and the spread each selection rule retains in it.
    #
    # The pool and the taper are the trajectory lane's own: the older half of the archive under
    # the taper notebook 07 selected on the pseudo-proxy lane, which is what the analog
    # estimators inherit. Nothing below runs an estimator - the members are read from the stored
    # `analog_index` of each lane, so this measures the ensembles that were actually placed.
    from types import SimpleNamespace

    from paleoreco.assim import experiments as ex
    from paleoreco.assim.ensrf import whitened_block
    from paleoreco.assim.hgaoenkf import HGAOEnKF
    from paleoreco.assim.priors import build_prior

    PRIOR_IDX, _ = chronological_half_split(ages)
    TAPER6 = dict(localization_km=15000.0, shrinkage_lambda=0.0, alpha=0.25)
    prior6 = build_prior(cube, ages, lats, lons, PRIOR_IDX, valid, **TAPER6)
    pool6 = (cube[PRIOR_IDX].reshape(len(PRIOR_IDX), -1).astype(np.float64)
             - prior6.clim_mean.ravel())
    safe_flat6 = np.broadcast_to(prior6.safe_valid, (2,) + prior6.safe_valid.shape).ravel()

    # The shipped estimator, built once. Only its stack geometry is used here - no analysis is
    # run - so the taper passed in is irrelevant and the pool is the one the lane drew from.
    mt6 = HGAOEnKF(pool6, ages[PRIOR_IDX], np.eye(1), (len(VARS), len(lats), len(lons)),
                   lats, lons, k=60, hybrid_w=1.0,
                   taper_meta={"localization_km": None, "shrinkage_lambda": 0.0, "alpha": 1.0},
                   tendency_theta=2.0, tendency_lag_yr=ex.MT_REFERENCE_LAG_YR,
                   tendency_extra_lags_yr=ex.MT_EXTRA_LAGS_YR,
                   tendency_curvature_yr=ex.MT_CURVATURE_YR,
                   tendency_normalise=True, preserve_obs_trace=True)

    traj6 = np.load(paths.run_dir(ex.ESTIMATOR_HGAOENKF_MT, paths.LANE_TRAJECTORY) / "trajectory_analysis.npz")
    IDX_EVIDENCE = traj6["analog_index"]
    IDX_MISFIT = np.load(paths.run_dir(ex.ESTIMATOR_HGAOENKF, paths.LANE_TRAJECTORY)
                         / "trajectory_analysis.npz")["analog_index"]
    # The lane borrows a network from one age to reconstruct another, so the assimilated
    # network is rebuilt from the age it came from. Reading the age being reconstructed
    # would describe a network the analysis never saw, and would fail outright at the ages
    # that carry no proxies of their own.
    by_shape6 = sample_block_centres(long).groupby("age")

    # Eigenvalue bands. The top edge is 1e3 rather than open so the plotted midpoint of the last
    # band sits inside the range the networks actually reach.
    LAM_EDGES = np.array([1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3])
    LAM_MID = np.sqrt(LAM_EDGES[:-1] * LAM_EDGES[1:])

    # The regime axis of section 6.6, fixed here because section 6.1 also needs to know where it
    # sits in the whitened spectrum. It is the pool's leading principal component, defined from
    # the archive alone and before any estimator is involved.
    _, S6, Vt6 = np.linalg.svd(pool6, full_matrices=False)
    REGIME_AXIS = Vt6[0]

    spectrum6, lam_max6, n_big6, m6, axis_lead6, beta6 = [], [], [], [], [], []
    retained6 = {name: [[] for _ in LAM_EDGES[:-1]] for name in ("misfit", "evidence")}
    for a, (age, shape_age) in enumerate(zip(traj6["ages"], traj6["shape_ages"])):
        sl = by_shape6.get_group(int(shape_age))
        g = obs_cell_index(sl["lat"].to_numpy(), sl["lon"].to_numpy(),
                           sl["channel"].to_numpy(), lats, lons)
        r = sl["sse"].to_numpy()
        keep = safe_flat6[g] & (r > 0)
        g, r = g[keep], r[keep].astype(np.float64)
        # An observation whose offset in time runs off the archive was dropped by the
        # lane, so it is dropped here too or the spectrum describes a wider network.
        _, on_axis = ex.transplanted_source(sl["centre"].to_numpy()[keep],
                                            int(shape_age), int(age), ages)
        g, r = g[on_axis], r[on_axis]
        if len(g) < 10:
            continue
        wb = whitened_block(prior6.B[:, g], prior6.B[np.ix_(g, g)], r)
        W = wb.rinv_sqrt[:, None] * wb.U            # into the basis of eq:baseline-whitened
        spectrum6.append(wb.Lam[wb.Lam > 0])
        order = np.argsort(wb.Lam)[::-1]
        m6.append(len(g)); lam_max6.append(wb.Lam[order[0]]); n_big6.append(int((wb.Lam > 1).sum()))
        # How much of the regime axis this network sees, and in which directions: the axis is a
        # vector in state space, read at the observed cells, whitened and resolved on the basis.
        q_axis = (REGIME_AXIS[g] * wb.rinv_sqrt) @ wb.U[:, order]
        axis_lead6.append((q_axis ** 2 / (q_axis ** 2).sum())[0])
        # The pool's own variance along each direction is the reference the ensembles are read
        # against, computed here rather than taken as Lam because B carries the taper.
        ref = ((pool6[:, g] - pool6[:, g].mean(0)) @ W).var(axis=0)
        band = np.digitize(wb.Lam, LAM_EDGES) - 1
        for name, sel in (("misfit", IDX_MISFIT[a]), ("evidence", IDX_EVIDENCE[a])):
            mem = pool6[sel]
            got = (((mem[:, g] - mem[:, g].mean(0)) @ W) ** 2).sum(0) / (len(sel) - 1.0)
            ratio = np.divide(got, ref, out=np.zeros_like(got), where=ref > 0)
            for b in range(len(LAM_EDGES) - 1):
                m = (ref > 0) & (band == b)
                if m.any():
                    retained6[name][b].append(np.median(ratio[m]))
        # The scalar eq:mt-trace solves for on this network: how far the augmented ensemble has
        # to be pulled back to carry the observation-space variance the k members carried alone.
        mem = pool6[IDX_EVIDENCE[a]]
        base = mem - mem.mean(axis=0)
        aug = np.vstack([base, mt6._tendency_rows(IDX_EVIDENCE[a], None)])
        beta6.append(mt6._trace_rescale(base, aug, SimpleNamespace(gather=g, r_diag=r)))
    spectrum6 = np.concatenate(spectrum6)
    PROFILE6 = {n: np.array([np.median(v) if v else np.nan for v in retained6[n]])
                for n in retained6}

    print(f"{len(m6)} networks, median {np.median(m6):.0f} assimilated rows")
    print(f"  lambda: median largest {np.median(lam_max6):.0f}, median eigenvalue "
          f"{np.median(spectrum6):.1e}, median count above 1: {np.median(n_big6):.0f}")
    print(f"  evidence weight on the largest direction at kappa = 1: "
          f"{1.0 / (np.median(lam_max6) + 1.0):.4f}")
    print(f"  regime axis: median {100 * np.median(axis_lead6):.0f}% of it lies in the single "
          f"largest-lambda direction")
    for n, v in PROFILE6.items():
        print(f"  {n:9s} retained spread by lambda band: {np.round(v, 3)}")
    print(f"  trace-preservation factor: median {np.median(beta6):.3f}, so the raw augmented "
          f"ensemble carries {1 / np.median(beta6) ** 2:.0f}x the observable variance")

    # The nine blocks, as section 6.4's normalisation paragraph describes them.
    print(f"\n{'lag':>6}{'order':>8}{'rms over the pool':>20}{'normalisation':>15}")
    for block in mt6._tendency_blocks:
        lag, second = block
        print(f"{lag:6.0f}{'2nd' if second else '1st':>8}"
              f"{mt6._block_amplitude(block):20.2f}{mt6._tendency_scale[block]:15.2f}")

    # Figure 6.1 - what each selection rule keeps, direction by direction.
    #
    # (a) is analytic: the weight 1/(kappa*lam + 1) of eq:mt-weight, drawn against the gain's own
    # filter factor lam/(lam + 1) so the complementarity is visible, over the spectrum the rule
    # actually meets. (b) is the measurement from the cell above.
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 2.15), constrained_layout=True)
    LAM = np.logspace(-6, 4, 400)
    LOG_TICKS = ([-6, -4, -2, 0, 2, 4],
                 [r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$1$", r"$10^{2}$", r"$10^{4}$"])

    ax = axes[0]
    hist = ax.twinx()                       # the spectrum sits behind the curves, unlabelled:
    hist.hist(np.log10(spectrum6), bins=60, color="#e8e8e8", edgecolor="none", zorder=0)
    hist.set_yticks([]); hist.spines["right"].set_visible(False); hist.set_xlim(-6, 4)
    ax.set_zorder(hist.get_zorder() + 1); ax.patch.set_visible(False)
    for kappa, style, colour in ((0, "-", "#6b6b6b"), (1, "-", "#c4692a"), (2, "--", "#e0a077")):
        ax.plot(np.log10(LAM), 1.0 / (kappa * LAM + 1.0), style, lw=1.2, color=colour,
                label=rf"$\kappa={kappa}$")
    ax.plot(np.log10(LAM), LAM / (LAM + 1.0), ":", lw=1.0, color="#3c5f8f", label="gain, $c=1$")
    ax.set_xlim(-6, 4); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(LOG_TICKS[0]); ax.set_xticklabels(LOG_TICKS[1])
    ax.set_xlabel(r"whitened eigenvalue $\lambda$"); ax.set_ylabel("selection weight")
    ax.legend(loc="center left", frameon=False, handlelength=1.6)
    ax.text(0.02, 1.06, "(a)", transform=ax.transAxes, fontsize=7.4, fontweight="bold",
            va="bottom")

    ax = axes[1]
    for name, colour, marker, label in (
            ("misfit", "#3c5f8f", "o", r"misfit ($\kappa=0$)"),
            ("evidence", "#c4692a", "s", r"evidence ($\kappa=1$)")):
        ax.plot(np.log10(LAM_MID), PROFILE6[name], marker + "-", ms=3.2, lw=1.2, color=colour,
                label=label)
    ax.set_xlim(-6, 4); ax.set_ylim(0, 1.05)
    ax.set_xticks(LOG_TICKS[0]); ax.set_xticklabels(LOG_TICKS[1])
    ax.set_xlabel(r"whitened eigenvalue $\lambda$")
    ax.set_ylabel("spread retained\n(fraction of the pool's)")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    ax.text(0.02, 1.06, "(b)", transform=ax.transAxes, fontsize=7.4, fontweight="bold",
            va="bottom")
    save(fig, "fig06_01_selection_geometry")

    # Figure 6.2 - the timescales one lag can and cannot see.
    #
    # Pure algebra apart from the per-block normalisation factors, which are read off the shipped
    # estimator rather than restated, so the curve is the stack the report actually runs. A
    # centred first difference at lag l returns a component of period T times |sin(2 pi l / T)|
    # and a second difference times 2|1 - cos(2 pi l / T)|; the blocks enter the covariance as
    # extra rows, so their contributions add in quadrature.
    T6 = np.linspace(100.0, 4000.0, 4000)
    omega = 2 * np.pi / T6
    response = np.zeros_like(T6)
    for block in mt6._tendency_blocks:
        lag, second = block
        h = (2 * np.abs(1 - np.cos(omega * lag)) if second else np.abs(np.sin(omega * lag)))
        response += (mt6._tendency_scale[block] * h) ** 2
    response = np.sqrt(response); response /= response.max()
    one_lag = np.abs(np.sin(omega * ex.MT_REFERENCE_LAG_YR)); one_lag /= one_lag.max()

    band = (T6 >= 100) & (T6 <= 4000)
    print(f"below a quarter of peak over: one lag {100 * (one_lag[band] < 0.25).mean():.1f}%, "
          f"stack {100 * (response[band] < 0.25).mean():.1f}% of the 100-4000 yr band")
    at800 = np.argmin(np.abs(T6 - 800))
    print(f"at 800 yr: one lag {one_lag[at800]:.3f}, stack {response[at800]:.3f}")

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 1.85), constrained_layout=True)
    ax.axvspan(150, 200, color="#f2f2f2", zorder=0)
    ax.axvline(1500, color="#c4c4c4", lw=0.8, ls=(0, (4, 2.5)), zorder=0)
    ax.plot(T6, one_lag, lw=1.0, color="#3c5f8f", label=r"one difference at $\ell = 400$ yr")
    ax.plot(T6, response, lw=1.2, color="#c4692a", label="the nine-block stack")
    ax.set_xscale("log"); ax.set_xlim(100, 4000); ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([100, 200, 400, 800, 1600, 3200])
    ax.set_xticklabels(["100", "200", "400", "800", "1600", "3200"])
    ax.set_xlabel("period of the archive's variability (yr)")
    ax.set_ylabel("response\n(fraction of peak)")
    ax.text(174, 1.10, "D--O transition", fontsize=6.2, color="#8a8a8a", ha="center", va="bottom")
    ax.text(1560, 1.10, "D--O cycle", fontsize=6.2, color="#8a8a8a", ha="left", va="bottom")
    ax.legend(loc="upper left", frameon=False, handlelength=1.8, bbox_to_anchor=(0.30, 1.03))
    save(fig, "fig06_02_stack_response")

    # Figure 6.3 - HGAOEnKF-MT as a third row of Figure 5.1.
    #
    # A drawing, not an experiment, and the visual language is Figure 5.1's commitment rather
    # than a fresh choice: blue is the static archive-wide path, green the data-selected
    # flow-dependent one, grey the observations and whatever consumes them directly, white the
    # analysis. **Orange is what this project adds** - reserved and deliberately unused in
    # Figure 5.1, and spent here. The coordinate frame, box geometry and arrow styles are copied
    # from that cell so a box of a given height renders at the same physical size in all three
    # panels.

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 1.62), constrained_layout=True)
    ax.set_xlim(0, 100); ax.set_ylim(-0.3, 13.2); ax.axis("off")
    ax.text(0, 13.1, "(c)  HGAOEnKF-MT", fontsize=7.4, fontweight="bold", va="top")

    archive = stage(ax, 1, 12, ROW0, ROW1, "LOVECLIM\narchive", STATIC)
    select = stage(ax, 15, 31, ROW0, ROW1,
                    "score every state by\nits evidence;\nkeep the best $k$", DATA, fs=6.3)
    augment = stage(ax, 34, 54, 7.5, 9.9,
                     "add the archive's flow\nat nine timescales", OURS, fs=6.3)
    cov_flow = stage(ax, 57, 71, 7.5, 9.9, "flow-dependent\ncovariance", FLOW, fs=6.3)
    cov_static = stage(ax, 57, 71, 3.5, 5.9, "static covariance $\\mathbf{B}$", STATIC, fs=6.3)
    gain = stage(ax, 74, 88, ROW0, ROW1, "two gains,\nsummed at $\\alpha$", STATIC)
    out = stage(ax, 90, 100, ROW0, ROW1, "analysis", ANALYSIS)
    obs = stage(ax, 32, 62, OBS0, OBS1, "observations $\\mathbf{y}$, $\\mathbf{R}$", DATA)

    flow(ax, (archive[1], MID), (select[0], MID))
    flow(ax, (38, OBS0), (25, ROW1))                  # the observations drive selection
    flow(ax, (56, OBS0), (81, ROW1))                  # and the innovation
    flow(ax, (select[1], 7.6), (augment[0], 8.7), color=FLOW[1])
    flow(ax, (augment[1], 8.7), (cov_flow[0], 8.7), color=OURS[1])
    # The augmentation reads the archive's time ordering and never passes through the scoring
    # box, so its connector leaves the archive before it and stays clear of the observations.
    route(ax, [6.5, 6.5, 41], [ROW1, 10.15, 10.15], color=OURS[1], ls=DASHED)
    flow(ax, (41, 10.15), (41, 9.9), color=OURS[1], ls=DASHED)
    ax.text(21, 10.3, "the archive's time ordering", fontsize=6.0, color=OURS[1],
            ha="center", va="bottom")
    # The static covariance never sees the selection either.
    route(ax, [6.5, 6.5, 55, 55], [ROW0, 2.0, 2.0, 4.7])
    flow(ax, (55, 4.7), (cov_static[0], 4.7))
    flow(ax, (cov_flow[1], 8.7), (gain[0], 7.9), color=FLOW[1])
    flow(ax, (cov_static[1], 4.7), (gain[0], 6.3))
    flow(ax, (gain[1], MID), (out[0], MID))
    route(ax, [23, 23, 95], [ROW0, MEAN_Y, MEAN_Y], color=FLOW[1], ls=DASHED)
    flow(ax, (95, MEAN_Y), (95, ROW0), color=FLOW[1], ls=DASHED)
    ax.text(59, MEAN_Y + 0.55, "prior mean: the mean of the $k$ selected states, unchanged",
            fontsize=6.3, color=FLOW[1], ha="center", va="bottom")

    save(fig, "fig06_03_estimator_schematic")

    # Section 6.6's numbers: what each prior carries along the archive's regime axis.
    #
    # No figure; the section prints three medians. The axis is the pool's leading principal
    # component, fixed from the archive before any estimator is involved, and the quantity is the
    # SHARE of each ensemble's variance lying along it - a share rather than an absolute, because
    # eq:mt-trace rescales the augmented ensemble by one scalar and a share is immune to it.
    from sklearn.mixture import GaussianMixture

    scores6 = pool6 @ REGIME_AXIS
    z6 = (scores6 - scores6.mean()) / scores6.std()
    g1 = GaussianMixture(1, random_state=0).fit(z6[:, None])
    g2 = GaussianMixture(2, random_state=0).fit(z6[:, None])
    mu6, sd6 = g2.means_.ravel(), np.sqrt(g2.covariances_.ravel())
    w6 = g2.weights_.ravel()
    print(f"regime axis: {S6[0] ** 2 / (S6 ** 2).sum():.3f} of the pool's variance; "
          f"dBIC {g1.bic(z6[:, None]) - g2.bic(z6[:, None]):.0f}, weights {w6.round(3)}, "
          f"separation {abs(mu6[0] - mu6[1]) / np.sqrt((sd6 ** 2 * w6).sum()):.2f} component sd")

    pool_share = scores6.var() / ((pool6 ** 2).sum() / (len(pool6) - 1))
    # DO_EVENT_WINDOWS runs 300 yr before each onset to 600 after, so the onset is its
    # lower edge plus 300; recovering it that way keeps this on the public API.
    onsets6 = np.array(sorted(lo + 300 for lo, _ in DO_EVENT_WINDOWS.values()))
    near6 = np.abs(traj6["ages"][:, None] - onsets6[None, :]).min(axis=1) <= 300
    shares6 = {}
    for name, idx in (("misfit members", IDX_MISFIT), ("evidence members", IDX_EVIDENCE),
                      ("MT augmented", IDX_EVIDENCE)):
        rows = []
        for a in range(len(traj6["ages"])):
            mem = pool6[idx[a]]
            dev = mem - mem.mean(axis=0)
            if name == "MT augmented":
                dev = np.vstack([dev, mt6._tendency_rows(idx[a], None)])
            rows.append(((dev @ REGIME_AXIS) ** 2).mean()
                        / ((dev ** 2).sum() / len(dev)))
        shares6[name] = np.array(rows) / pool_share

    print(f"\nshare of each prior's variance along the regime axis, relative to the pool's "
          f"({pool_share:.3f}):")
    print(f"{'ensemble':18s}{'median':>9s}{'near onset':>12s}{'elsewhere':>11s}")
    for name, v in shares6.items():
        print(f"{name:18s}{np.median(v):9.2f}{np.median(v[near6]):12.2f}"
              f"{np.median(v[~near6]):11.2f}")
    print(f"\n{near6.sum()} of {len(near6)} analyses lie within 300 yr of an interstadial onset")


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("07_figures", [f"chapter {n}" for n in (1, 2, 3, 4, 5, 6)])
    inp = load_inputs(C.SMOKE_AGES if smoke else None)
    for n, fn in ((1, chapter1), (2, chapter2), (3, chapter3),
                  (4, chapter4), (5, chapter5), (6, chapter6)):
        if stages.run(f"chapter {n}"):
            fn() if fn in (chapter2, chapter4) else fn(inp)
    stages.done()


if __name__ == "__main__":
    main()
