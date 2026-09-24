"""Every figure and table the report prints, regenerated from stored artefacts.

Reads the lane and product directories and the raw inputs; runs no estimator, with two
exceptions that are cheap and deterministic: the lag correlation Chapter 5 plots, and the
analog ensembles Chapter 6 anatomises.

Figures are authored at their printed size. A 60-page A4 report with 2.5 cm margins has a
text width of about 16 cm, so a figure drawn wide and scaled down would render 9 pt labels
at roughly 4 pt. Everything below is sized for that width at 1:1.
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
    temporal_terms,
)
from paleoreco.data import VARS, build_prior_cube
from paleoreco.eval import calibration, da
from paleoreco.data.splits import (
    DO_EVENT_WINDOWS,
    DO_ONSET_BP,
    chronological_half_split,
)

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
    axes[2].set_title("(c) MTA-HGAOEnKF posterior")

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
    # the layout used to work around. HGAOEnKF and MTA-HGAOEnKF straddle their shared column so the
    # connector between them has somewhere to go; everything else sits on its own tick.
    METHODS = [
        (0.00, 0, "3DVar", "published"),
        (0.00, 1, "Offline EnSRF\npaleoDA", "published"),
        (1.00, 0, "AOEnKF-B", "published"),
        (1.00, 2, "Analog offline EnKF", "published"),
        (0.55, 3, "HGAOEnKF", "published"),
        (1.30, 3, "MTA-HGAOEnKF\n(this work)", "ours"),
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
    mt_left, _ = box_edges(boxes["MTA-HGAOEnKF"])

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
    # the taper that lane ran, read from its stored config so the whitened basis below is the
    # one the estimators actually met rather than a restatement that can drift from it. Nothing
    # below runs an estimator - the members are read from the stored `analog_index` of each
    # lane, so this measures the ensembles that were actually placed.
    from types import SimpleNamespace

    from paleoreco.assim import experiments as ex
    from paleoreco.assim.ensrf import whitened_block
    from paleoreco.assim.hgaoenkf import HGAOEnKF
    from paleoreco.assim.priors import build_prior

    PRIOR_IDX, _ = chronological_half_split(ages)
    TAPER6 = C.inherited_taper(
        paths.run_dir(ex.ESTIMATOR_HGAOENKF_MT, paths.LANE_TRAJECTORY) / "trajectory_config.json")
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

    # Figure 6.3 - MTA-HGAOEnKF as a third row of Figure 5.1.
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
    ax.text(0, 13.1, "(c)  MTA-HGAOEnKF", fontsize=7.4, fontweight="bold", va="top")

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


# ---------------------------------------------------------------------------
# Chapter 7: three estimators, three lanes.
# ---------------------------------------------------------------------------
# One colour and one marker per estimator, fixed here and used by every figure below, so the
# three read the same way throughout the chapter. Grey is the static analysis, blue is the
# archive-wide path the schematics of Figures 2.1 and 5.1 already draw in blue, and orange is
# Figure 2.1's "this work" colour.
ESTIMATORS = (
    (ex.ESTIMATOR_3DVAR, "3DVar", "#6b6b6b", "o"),
    (ex.ESTIMATOR_HGAOENKF, "HGAOEnKF", "#3c5f8f", "s"),
    (ex.ESTIMATOR_HGAOENKF_MT, "MTA-HGAOEnKF", "#c4692a", "D"),
)
# Blue where a comparator is better, orange where the contribution is, in the same two colours
# the estimators carry everywhere else, so no reader has to consult a key to read the sign.
DIFF_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "mt_diff", ["#3c5f8f", "#eef1f4", "#f7f2ee", "#c4692a"])


TAPER_COLS = ("localization_km", "shrinkage_lambda", "alpha")


def _run(estimator, lane, name):
    """One stored artefact of one (estimator, lane), by file name."""
    path = paths.run_dir(estimator, lane) / name
    return json.load(open(path)) if name.endswith(".json") else np.load(path)


def _lane_rows(estimator, lane, file_lane, *, method, split="test", **fixed):
    """Pooled, all-event metric rows of one lane, filtered to one method and split."""
    M = pd.read_csv(paths.run_dir(estimator, lane) / "metrics.csv")
    m = ((M.method == method) & (M.split == split) & (M.channel == "pooled")
         & (M.do_event == "all") & (M.lane == file_lane))
    for key, value in fixed.items():
        m &= np.isclose(M[key], value)
    return M[m]


def _at_selected(M, cfg, b_scale):
    """Mask of the rows a lane's config.json recorded as its winning configuration.

    The taper is read from ``selected`` where the grid searched it and from the config's own
    columns where it was inherited, so one mask serves a lane that gridded the regularizer and
    a lane that was handed it. A key absent from ``selected`` is an axis the run never varied.
    """
    sel = cfg["selected"]
    m = np.isclose(M["b_scale"], b_scale)
    for key in TAPER_COLS:
        value = sel.get(key, cfg.get(key))
        m &= M[key].isna() if value is None else np.isclose(M[key], value)
    for key in ("analog_k", "hybrid_w") + ex.TERM_KEYS:
        if key in sel:
            m &= np.isclose(M[key], sel[key])
    return m


def _bin_centres(edges):
    """The midpoint of each distance bin, which is what the stored curve calls its centre.

    Arithmetic rather than geometric: the innermost bin starts at zero, whose geometric
    centre is zero and has no place on a log axis.
    """
    return 0.5 * (np.asarray(edges)[:-1] + np.asarray(edges)[1:])


def _count_axis(ax, edges, counts):
    """Bin populations as pale bars behind a skill curve, on their own right-hand axis.

    A curve over unequal distance bins says nothing about how much of the domain each bin
    holds, and on both lanes below that is exactly what decides how far the curve can be
    trusted.
    """
    twin = ax.twinx()
    twin.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
             color="0.90", edgecolor="white", linewidth=0.4, zorder=0)
    twin.set_ylim(0, counts.max() * 3.1)
    twin.set_yticks([])
    twin.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    return twin


def chapter7() -> None:
    """Figures 7.1-7.5: three estimators across the three evaluation lanes."""
    LANE_W = "withholding_random"
    deflate = {e: ex.method_label(e, ex.TEMPORAL_DEFLATE) for e, *_ in ESTIMATORS}

    # --- Figure 7.1: what the hybrid weight says the analog covariance is worth --------------
    # The prior mean is the analog mean at every weight and only the covariance entering the
    # gain moves, so the sweep separates the two things an analog prior supplies. At each
    # weight the ensemble size and amplitude are the pair that weight's own selection split
    # prefers, so no point on either curve is chosen on the split it is plotted from.
    def alpha_profile(estimator, theta):
        M = pd.read_csv(paths.run_dir(estimator, paths.LANE_PPE) / "metrics.csv")
        base = ((M.method == ex.method_label(estimator)) & (M.lane == ex.LANE_PPE)
                & (M.channel == "pooled") & (M.do_event == "all") & (M.metric == "rrmse")
                & np.isclose(M.tendency_theta, theta))
        sel, tst = M[base & (M.split == "selection")], M[base & (M.split == "test")]
        rows = []
        for w, g in sel.groupby("hybrid_w"):
            win = g.loc[g.value.idxmin()]
            hit = tst[np.isclose(tst.hybrid_w, w) & np.isclose(tst.analog_k, win.analog_k)
                      & np.isclose(tst.b_scale, win.b_scale)]
            rows.append((float(w), float(hit.value.iloc[0]), int(win.analog_k),
                         float(win.b_scale)))
        return np.array(sorted(rows))

    three_cfg = _run(ex.ESTIMATOR_3DVAR, paths.LANE_PPE, "ppe_config.json")
    three_M = pd.read_csv(paths.run_dir(ex.ESTIMATOR_3DVAR, paths.LANE_PPE) / "metrics.csv")
    three_rrmse = float(three_M[(three_M.method == "3dvar") & (three_M.split == "test")
                                & (three_M.channel == "pooled") & (three_M.do_event == "all")
                                & (three_M.metric == "rrmse")
                                & _at_selected(three_M, three_cfg,
                                               three_cfg["selected"]["b_scale"])].value.iloc[0])

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.7), constrained_layout=True)
    ax.axhline(three_rrmse, color="#6b6b6b", lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.text(0.985, three_rrmse - 0.0012, "3DVar", color="#6b6b6b", fontsize=6.4,
            ha="right", va="top")
    for estimator, label, colour, marker in ESTIMATORS[1:]:
        cfg = _run(estimator, paths.LANE_PPE, "ppe_config.json")
        theta = cfg["selected"]["tendency_theta"]
        prof = alpha_profile(estimator, theta)
        ax.plot(prof[:, 0], prof[:, 1], marker + "-", ms=3.6, lw=1.3, color=colour, label=label,
                zorder=3)
        chosen = float(cfg["selected"]["hybrid_w"])
        y = float(prof[np.isclose(prof[:, 0], chosen), 1][0])
        ax.plot([chosen], [y], marker="o", ms=9.5, mfc="none", mec=colour, mew=1.2, zorder=4)
        print(f"  7.1 {label:12s} theta={theta:g} alpha profile "
              f"{dict(zip(prof[:, 0], np.round(prof[:, 1], 4)))}, selection chose alpha={chosen:g}")
    ax.annotate("prior mean alone\n(static gain)", xy=(0.0, 0.5924), xytext=(0.095, 0.6065),
                fontsize=6.3, color="0.35", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.55",
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("analog covariance alone", xy=(1.0, 0.552), xytext=(0.80, 0.5645),
                fontsize=6.3, color="0.35", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.55",
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(0.549, 0.619)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"hybrid weight $\alpha$ on the analog covariance")
    ax.set_ylabel("test relative RMSE\n(snapshot lane)")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.15,
              labelspacing=0.25)
    save(fig, "fig07_01_hybrid_weight")

    # --- Figure 7.2: where in space the estimators separate ----------------------------------
    # The snapshot lane alone. The withholding lane carries the same statistic and it is
    # computed below, but it is reported as numbers rather than drawn: its curves lie on top of
    # one another at every distance, which is a null the prose states in one sentence and a
    # panel spends a third of a figure failing to show.
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.6), constrained_layout=True)
    for estimator, label, colour, marker in ESTIMATORS:
        cfg = _run(estimator, paths.LANE_PPE, "ppe_config.json")
        z = _run(estimator, paths.LANE_PPE, "ppe_skill_vs_distance.npz")
        bj = int(np.argmin(np.abs(z["b_scales"] - cfg["selected"]["b_scale"])))
        edges, counts = z["edges"], z["count"]
        # The widest bin holds a few hundred points against hundreds of thousands elsewhere,
        # so it is a handful of cells in the Pacific and Southern Ocean voids rather than a
        # measurement; it is dropped rather than drawn with an interval nothing supports.
        keep = counts > 1000 if (counts > 1000).any() else counts > 0
        centres = _bin_centres(edges)
        ax.plot(centres[keep], z["ce"][bj][keep], marker + "-", ms=3.8, lw=1.3, color=colour,
                label=label, zorder=3)
        if estimator == ex.ESTIMATOR_3DVAR:
            _count_axis(ax, edges[:keep.sum() + 1], counts[keep] / 1e3)
        print(f"  7.2 {label:12s} CE by distance {np.round(z['ce'][bj][keep], 4)}")
    mt = _run(ex.ESTIMATOR_HGAOENKF_MT, paths.LANE_PPE, "ppe_skill_vs_distance.npz")
    base = _run(ex.ESTIMATOR_HGAOENKF, paths.LANE_PPE, "ppe_skill_vs_distance.npz")
    for tag, other in (("3DVar", ex.ESTIMATOR_3DVAR), ("HGAOEnKF", ex.ESTIMATOR_HGAOENKF)):
        o = _run(other, paths.LANE_PPE, "ppe_skill_vs_distance.npz")
        oc = _run(other, paths.LANE_PPE, "ppe_config.json")["selected"]["b_scale"]
        mc = _run(ex.ESTIMATOR_HGAOENKF_MT, paths.LANE_PPE, "ppe_config.json")["selected"]["b_scale"]
        gap = (mt["ce"][int(np.argmin(np.abs(mt["b_scales"] - mc)))]
               - o["ce"][int(np.argmin(np.abs(o["b_scales"] - oc)))])
        print(f"  7.2 MT - {tag:9s} gain by distance {np.round(gap[keep], 4)}")
    ax.set_xscale("log")
    ax.set_xlim(200, 8200)
    ax.set_xticks([250, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["250", "500", "1000", "2000", "5000"])
    ax.set_xlabel("distance to the nearest observation (km)")
    ax.set_ylabel("coefficient of efficiency\n(snapshot lane)")
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.15,
              labelspacing=0.25)
    save(fig, "fig07_02_skill_vs_distance")

    # The same statistic on the real proxies, printed rather than plotted. Section 7.6 quotes
    # these, and they are what says the lane sits in the interpolation regime: the withheld
    # samples sit close to the data, and the three estimators are indistinguishable at every
    # distance they do reach.
    W_EDGES = np.array([0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    for estimator, label, _, _ in ESTIMATORS:
        cfg = _run(estimator, paths.LANE_WITHHOLDING, "withholding_random_config.json")
        z = _run(estimator, paths.LANE_WITHHOLDING, "withholding_random_predictions.npz")
        bj = int(np.argmin(np.abs(z["b_scales"] - cfg["selected"]["b_scale"])))
        actual = z["actual"]
        curve = da.skill_vs_distance(actual, z[f"climatological_pred_{deflate[estimator]}"][bj],
                                     np.zeros_like(actual), z["distance_km"], W_EDGES)
        if estimator == ex.ESTIMATOR_3DVAR:
            print(f"  7.6 withheld-sample distance: median "
                  f"{np.median(z['distance_km']):.0f} km, "
                  f"{100 * float((z['distance_km'] <= 2000).mean()):.1f}% within 2000 km; "
                  f"bins {W_EDGES.astype(int)} hold {curve['count']}")
        print(f"  7.6 {label:12s} withholding CE by distance {np.round(curve['ce'], 4)}")

    # --- Figure 7.3: the same comparison cell by cell ----------------------------------------
    # The snapshot lane's stored fields are the analyses the chapter reports: that lane writes
    # one amplitude and it is the selected one, so no map here is drawn at a configuration the
    # tables do not quote.
    ppe = {e: _run(e, paths.LANE_PPE, "ppe_analysis.npz") for e, *_ in ESTIMATORS}
    ref = ppe[ex.ESTIMATOR_3DVAR]
    lats7, lons7 = ref["lats"], ref["lons"]
    truth7 = ref["truth_anom"].astype(np.float64)
    zero7 = np.zeros_like(truth7[0])
    ce = {e: da.ce_map(truth7, ppe[e]["recon_climatological"].astype(np.float64), zero7)
          for e in ppe}
    mt_ce = ce[ex.ESTIMATOR_HGAOENKF_MT]
    # Symmetric and clipped at a round value just past the 98th percentile of the absolute
    # difference: a handful of cells in which the truth barely varies reach several CE units
    # and would otherwise set a scale on which nothing else is visible.
    DIFF_LIM = 0.25
    COLS = [("MTA-HGAOEnKF", mt_ce, dict(vmin=0.0, vmax=1.0, cmap="viridis"), "min", "CE"),
            ("minus 3DVar", mt_ce - ce[ex.ESTIMATOR_3DVAR],
             dict(vmin=-DIFF_LIM, vmax=DIFF_LIM, cmap=DIFF_CMAP), "both", r"$\Delta$CE"),
            ("minus HGAOEnKF", mt_ce - ce[ex.ESTIMATOR_HGAOENKF],
             dict(vmin=-DIFF_LIM, vmax=DIFF_LIM, cmap=DIFF_CMAP), "both", r"$\Delta$CE")]

    fig, axes = plt.subplots(2, 3, figsize=(REPORT_WIDTH, 2.60),
                             subplot_kw={"projection": PLATE}, constrained_layout=True)
    for j, (label, field, kw, extend, unit) in enumerate(COLS):
        for c, name in enumerate(VARS):
            mesh = draw_field(axes[c, j], lats7, lons7, field[c], **kw)
            axes[c, j].set_title(f"({'adbecf'[2 * j + c]}) {name.upper()} {label}", fontsize=6.8)
            map_axes(axes[c, j], ylabel=(j == 0))
        cb = fig.colorbar(mesh, ax=axes[:, j], orientation="horizontal", shrink=0.92, pad=0.02,
                          aspect=22, extend=extend)
        cb.set_label(unit, fontsize=6.8)
        cb.ax.tick_params(labelsize=6, length=2)
    save(fig, "fig07_03_percell_ce")

    for a, b, tag in ((ex.ESTIMATOR_HGAOENKF_MT, ex.ESTIMATOR_3DVAR, "MT - 3DVar"),
                      (ex.ESTIMATOR_HGAOENKF_MT, ex.ESTIMATOR_HGAOENKF, "MT - HGAOEnKF"),
                      (ex.ESTIMATOR_HGAOENKF, ex.ESTIMATOR_3DVAR, "HGAOEnKF - 3DVar")):
        d = ce[a] - ce[b]
        by_lat = [(lo, hi, float(d[:, (lats7 >= lo) & (lats7 < hi)].mean()))
                  for lo, hi in ((-90, -45), (-45, 0), (0, 45), (45, 90))]
        print(f"  7.3 {tag:16s} median {np.median(d):+.4f}, better in {100 * (d > 0).mean():.1f}% "
              f"of cells | MTCO {np.median(d[0]):+.4f} ({100 * (d[0] > 0).mean():.1f}%) "
              f"MTWA {np.median(d[1]):+.4f} ({100 * (d[1] > 0).mean():.1f}%) | "
              + " ".join(f"{lo}..{hi}:{v:+.4f}" for lo, hi, v in by_lat)
              + f" | clipped {100 * (np.abs(d) > DIFF_LIM).mean():.1f}%")

    # --- Figure 7.4: where in time the analog estimators separate ----------------------------
    # Every point comes from the metrics CSV, which carries the timescale rows at every
    # amplitude, so the lane can be read at whichever one it is reported at. That is the one its
    # config selected, which is also what the operating-point table quotes and what the stored
    # fields carry, so this figure, that table and any map of this lane describe one analysis.
    # Reading each treatment's own selected amplitude instead would report the corrected
    # analysis at a point three different treatments each chose differently.
    #
    # The figure draws the two analog estimators alone: what it is read for is what the flow
    # stack does to the scheme it extends, and a third line answering a different question
    # crowds two panels that are already dense. 3DVar is still scored and still printed below,
    # because Section 7.5 quotes its band values in the prose, where the comparison against a
    # static analysis belongs.
    band_x = np.arange(len(ex.BANDS), dtype=float)
    windows = np.array(ex.LOWPASS_WINDOWS, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 2.5), constrained_layout=True)
    for estimator, label, colour, marker in ESTIMATORS:
        cfg = _run(estimator, paths.LANE_TRAJECTORY, "trajectory_config.json")
        method = deflate[estimator]
        b = cfg["selected"]["b_scale"]
        rows = _lane_rows(estimator, paths.LANE_TRAJECTORY, ex.LANE_TRAJECTORY,
                          method=method, b_scale=b).set_index("metric")["value"]

        def read(*names):
            return np.array([float(rows.get(n, np.nan)) for n in names])

        band_ce = read(*(f"ce_bp{lo}_{hi}" for lo, hi in ex.BANDS))
        band_amp = read(*(f"amp_bp{lo}_{hi}" for lo, hi in ex.BANDS))
        band_r = read(*(f"corr_bp{lo}_{hi}" for lo, hi in ex.BANDS))
        lp_ce = read(*(f"ce_lp{int(w)}" for w in windows))
        if estimator != ex.ESTIMATOR_3DVAR:
            kw = dict(ms=3.4, lw=1.2, color=colour)
            axes[0].plot(band_x, band_ce, marker + "-", label=label, **kw)
            axes[1].plot(windows, lp_ce, marker + "-", label=label, **kw)
        print(f"  7.4 {label:12s} c={b:g} band CE {np.round(band_ce, 4)} | "
              f"lowpass CE {np.round(lp_ce, 4)} | band amp {np.round(band_amp, 4)} | "
              f"band r {np.round(band_r, 4)}")
    axes[0].axhline(0.0, color="0.45", lw=0.7)
    axes[0].text(4.35, 0.02, "climatology", fontsize=6.2, color="0.45", ha="right", va="bottom")
    axes[0].set_ylim(-0.66, 0.88)
    axes[0].set_ylabel("coefficient of efficiency")
    axes[0].set_xlabel("band (yr)")
    axes[0].text(0.02, 1.02, "(a) by band", transform=axes[0].transAxes, fontsize=7.0,
                 fontweight="bold", va="bottom")
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.5, borderpad=0.1,
                   labelspacing=0.22, fontsize=6.5, bbox_to_anchor=(-0.01, 0.97))
    axes[1].set_ylim(0.46, 0.82)
    axes[1].set_ylabel("coefficient of efficiency")
    axes[1].set_xlabel("everything slower than (yr)")
    axes[1].text(0.02, 1.02, "(b) cumulative", transform=axes[1].transAxes, fontsize=7.0,
                 fontweight="bold", va="bottom")
    # A band is named by the band it is, since a tick at its geometric centre reads as a
    # timescale the metric never isolated; the cumulative panel keeps the window itself.
    BAND_TICKS = ["25-100", "100-250", "250-500", "500-1k", "1k-2k"]
    axes[0].set_xlim(-0.35, len(ex.BANDS) - 0.65)
    axes[0].set_xticks(band_x)
    axes[0].set_xticklabels(BAND_TICKS, fontsize=6.0, rotation=38, ha="right",
                            rotation_mode="anchor")
    axes[1].set_xscale("log")
    axes[1].set_xlim(19, 2700)
    axes[1].set_xticks([25, 100, 250, 500, 1000, 2000])
    axes[1].set_xticklabels(["25", "100", "250", "500", "1k", "2k"])
    save(fig, "fig07_04_timescale")

    # --- Figure 7.5: whether the stated uncertainty matches the errors -----------------------
    # The networks are identical across the three runs, so the distance to the nearest
    # observation is a property of the lane rather than of an estimator and is built once.
    obs_n = ref["obs_n"]
    dist = np.stack([da.nearest_obs_distance(lats7, lons7, ref["obs_lat"][ti, :n],
                                             ref["obs_lon"][ti, :n])
                     for ti, n in enumerate(obs_n)])
    shape7 = truth7.shape
    dist_full = np.broadcast_to(dist.reshape(shape7[0], 1, shape7[2], shape7[3]),
                                shape7).ravel()
    edges7 = _run(ex.ESTIMATOR_3DVAR, paths.LANE_PPE, "ppe_skill_vs_distance.npz")["edges"]
    counts7 = _run(ex.ESTIMATOR_3DVAR, paths.LANE_PPE,
                   "ppe_skill_vs_distance.npz")["count"]
    keep7 = counts7 > 1000 if (counts7 > 1000).any() else counts7 > 0
    which = np.digitize(dist_full, edges7) - 1

    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 2.45), constrained_layout=True)
    axes[0].axhline(1.0, color="0.45", lw=0.8, ls=(0, (4, 3)))
    axes[0].text(7800, 1.02, "honest", fontsize=6.2, color="0.45", ha="right", va="bottom")
    for estimator, label, colour, marker in ESTIMATORS:
        z = ppe[estimator]
        resid = ((truth7 - z["recon_climatological"].astype(np.float64))
                 / np.sqrt(z["post_var"].astype(np.float64))).ravel()
        disp = np.array([calibration.rcrv(resid[which == b], np.zeros(int((which == b).sum())),
                                          np.ones(int((which == b).sum())))[1]
                         if (which == b).sum() > 1 else np.nan
                         for b in range(len(edges7) - 1)])
        axes[0].plot(_bin_centres(edges7)[keep7], disp[keep7], marker + "-", ms=3.4, lw=1.2,
                     color=colour, label=label, zorder=3)
        # Residuals past the window are dropped rather than clipped: piling them on the end
        # bins draws a spike that is an artefact of the axis, not of the posterior. At most
        # 2.2% of any estimator's residuals fall outside it.
        axes[1].hist(resid, bins=np.linspace(-5, 5, 121), density=True, histtype="step",
                     lw=1.1, color=colour, label=label)
        print(f"  7.5 {label:12s} dispersion by distance {np.round(disp[keep7], 3)} | "
              f"pooled {np.std(resid):.3f}")
    axes[0].set_xscale("log")
    axes[0].set_xlim(200, 8200)
    axes[0].set_xticks([250, 500, 1000, 2000, 5000])
    axes[0].set_xticklabels(["250", "500", "1000", "2000", "5000"])
    axes[0].set_ylim(0.75, 2.62)
    axes[0].set_xlabel("distance to the nearest observation (km)")
    axes[0].set_ylabel("dispersion $d$")
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.15,
                   fontsize=6.6)
    axes[0].text(0.02, 1.02, "(a) by distance", transform=axes[0].transAxes, fontsize=7.0,
                 fontweight="bold", va="bottom")
    grid7 = np.linspace(-5, 5, 400)
    axes[1].plot(grid7, stats.norm.pdf(grid7), color="0.25", lw=0.9, ls=(0, (4, 3)),
                 label="honest posterior")
    axes[1].set_xlim(-5, 5)
    axes[1].set_xlabel("standardised residual $z$")
    axes[1].set_ylabel("density")
    axes[1].legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.15,
                   fontsize=6.6)
    axes[1].text(0.02, 1.02, "(b) pooled", transform=axes[1].transAxes, fontsize=7.0,
                 fontweight="bold", va="bottom")
    save(fig, "fig07_05_calibration")


    # --- Figure 7.6: how far each prior can be trusted -----------------------------------------
    # Every lane scores its estimators at all ten amplitudes, so the sweep is a read rather than
    # a re-run. Only c varies along each curve: the taper, the ensemble size and the hybrid
    # weight are held at the values Table 7.1 records, so the curve is one estimator being
    # trusted further and further rather than a different estimator at each point.
    SWEEP = ((("a"), "snapshot", paths.LANE_PPE, ex.LANE_PPE, "ppe_config.json",
              ex.TEMPORAL_OFF, False),
             (("b"), "time series", paths.LANE_TRAJECTORY, ex.LANE_TRAJECTORY,
              "trajectory_config.json", ex.TEMPORAL_DEFLATE, False),
             (("c"), "withholding", paths.LANE_WITHHOLDING, LANE_W,
              "withholding_random_config.json", ex.TEMPORAL_DEFLATE, True))

    fig, axes = plt.subplots(1, 3, figsize=(REPORT_WIDTH, 2.35), constrained_layout=True)
    for ax, (tag, title, dirname, lanecol, cfgname, treat, by_fold) in zip(axes, SWEEP):
        for estimator, label, colour, marker in ESTIMATORS:
            cfg = _run(estimator, dirname, cfgname)
            M = pd.read_csv(paths.run_dir(estimator, dirname) / "metrics.csv")
            m = ((M.method == ex.method_label(estimator, treat)) & (M.lane == lanecol)
                 & (M.split == "test") & (M.channel == "pooled") & (M.do_event == "all")
                 & (M.metric == "ce"))
            if by_fold:
                m &= (M.fold == -1)
            # The run directories hold a whole grid, so the sweep is pinned to the configuration
            # the lane selected and only the amplitude is left free.
            sel = cfg["selected"]
            for key in TAPER_COLS:
                value = sel.get(key, cfg.get(key))
                m &= M[key].isna() if value is None else np.isclose(M[key], value)
            for key in ("analog_k", "hybrid_w") + ex.TERM_KEYS:
                if key in sel and M[key].notna().any():
                    m &= np.isclose(M[key], sel[key])
            curve = M[m].set_index("b_scale")["value"].sort_index()
            static = estimator == ex.ESTIMATOR_3DVAR
            ax.plot(curve.index, curve.values, marker=marker, ms=3.2, lw=1.3, color=colour,
                    ls=(0, (3.2, 1.8)) if static else "-", label=label,
                    zorder=5 if static else 3)
            b = sel["b_scale"]
            ax.plot([b], [curve.loc[b]], marker="o", ms=8.5, mfc="none", mec=colour, mew=1.1,
                    zorder=6 if static else 4)
            print(f"  7.6 {title:12s} {label:12s} chosen c={b:<5g} CE "
                  + " ".join(f"{k:g}:{v:.3f}" for k, v in curve.items()))
        ax.axhline(0.0, color="0.45", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlim(0.07, 150)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_xlabel(r"background amplitude $c$")
        ax.text(0.02, 1.02, f"({tag}) {title} lane", transform=ax.transAxes, fontsize=7.0,
                fontweight="bold", va="bottom")
    axes[0].set_ylabel("test coefficient of efficiency")
    axes[0].text(0.082, 0.025, "climatology", fontsize=6.2, color="0.45", ha="left",
                 va="bottom")
    axes[0].legend(loc="lower left", frameon=False, handlelength=1.5, borderpad=0.1,
                   labelspacing=0.22, fontsize=6.4)
    save(fig, "fig07_06_amplitude_sweep")

    # The headline the chapter's tables print, read off the same artefacts the figures are.
    for estimator, label, _, _ in ESTIMATORS:
        cfg = _run(estimator, paths.LANE_PPE, "ppe_config.json")
        M = pd.read_csv(paths.run_dir(estimator, paths.LANE_PPE) / "metrics.csv")
        b = cfg["selected"]["b_scale"]
        row = M[(M.method == ex.method_label(estimator)) & (M.split == "test")
                & (M.channel == "pooled") & (M.do_event == "all")
                & _at_selected(M, cfg, b)].set_index("metric")["value"]
        wcfg = _run(estimator, paths.LANE_WITHHOLDING, "withholding_random_config.json")
        wrow = _lane_rows(estimator, paths.LANE_WITHHOLDING, LANE_W, method=deflate[estimator],
                          b_scale=wcfg["selected"]["b_scale"], fold=-1).set_index("metric")["value"]
        tcfg = _run(estimator, paths.LANE_TRAJECTORY, "trajectory_config.json")
        trow = _lane_rows(estimator, paths.LANE_TRAJECTORY, ex.LANE_TRAJECTORY,
                          method=deflate[estimator],
                          b_scale=tcfg["selected"]["b_scale"]).set_index("metric")["value"]
        print(f"  T7.2 {label:12s} snapshot CE {row['ce']:.4f} amp {row['amplitude']:.3f} "
              f"ssim {row['ssim']:.3f} | time series CE {trow['ce']:.4f} amp "
              f"{trow['amplitude']:.3f} | withholding CE {wrow['ce']:.4f} amp "
              f"{wrow['amplitude']:.3f}")
        print(f"  T7.3 {label:12s} d {row['rcrv_dispersion']:.2f}/{trow['rcrv_dispersion']:.2f}/"
              f"{wrow['rcrv_dispersion']:.2f}  crps {row['crps']:.3f}/{trow['crps']:.3f}/"
              f"{wrow['crps']:.3f}  cov {row['coverage90']:.3f}/{trow['coverage90']:.3f}/"
              f"{wrow['coverage90']:.3f}")


# ---------------------------------------------------------------------------
# Chapter 8: the product.
# ---------------------------------------------------------------------------
# The reconstruction is orange and the simulation it was built on blue, in the same two
# colours Figure 2.1 gives "this work" and the archive-wide path.
PRODUCT_C, SIM_C = "#c4692a", "#3c5f8f"
# Where a panel separates the two channels rather than the two fields, it needs colours that
# carry neither of the meanings above.
CHANNEL_C = ("#1b7837", "#762a83")
# The amplitude the maps and tables are quoted at. Nothing stored selects one, so the product
# is published as a band across them all and this is the member Figure 1.1 also draws.
B_REFERENCE = 5.0
# Liu et al. (2026) Sect. 2.4 take each event as the interval from 300 yr before its onset to
# 600 yr after. Ages run backwards, so the interstadial half is the younger one.
DO_POST_YR, DO_PRE_YR = 600, 300
# The timescale the product is read at, which is the fastest band Chapter 7 finds it beats a
# climatology over.
SMOOTH_YR = 250.0
# Northern extratropics: the band Liu et al. report their MTCO-to-MTWA ratio over, and their
# Table 3 maximum-likelihood slope with its 95% interval.
NET_LAT = 23.5
LIU_NET_SLOPE, LIU_NET_CI = 2.9, (2.0, 3.8)
# Resampling the sites, not the rows: a site contributes thousands of correlated rows, so a
# row bootstrap would report an interval far tighter than the network supports. The count
# matches the redraws Section 7.2 reports.
N_BOOT = 5000


def _area_weights(lats, lons):
    """Cosine-latitude weights over the grid, so a mean does not over-count the poles."""
    return np.broadcast_to(np.cos(np.deg2rad(np.asarray(lats, dtype=float)))[:, None],
                           (len(lats), len(lons)))


def _area_mean(field, weights):
    """Weighted mean over the trailing two axes."""
    return (field * weights).sum(axis=(-2, -1)) / weights.sum()


def _event_halves(ages, onset):
    """``(interstadial, stadial)`` age masks for one onset, the younger half first."""
    return ((ages >= onset - DO_POST_YR) & (ages <= onset),
            (ages >= onset) & (ages <= onset + DO_PRE_YR))


def _do_composite(field, ages, onsets):
    """Interstadial-minus-stadial difference per event, stacked on a leading event axis.

    Both halves are means rather than endpoint reads: a single 25-yr state either side of an
    onset carries one network's analysis noise, and the product is not claimed at that
    resolution anyway.
    """
    out = []
    for onset in onsets:
        warm, cold = _event_halves(ages, onset)
        if warm.any() and cold.any():
            out.append(field[warm].mean(axis=0) - field[cold].mean(axis=0))
    return np.asarray(out)


def _site_composite(long, onsets, value):
    """Interstadial-minus-stadial at the proxy sites, per event, as ``(n_events, n_channels)``.

    The estimator every source in Figure 8.4(f) is put through: one mean per site per half of
    an event window, differenced, then averaged over the site-event pairs that report both
    channels. Reading a gridded field through it, rather than area-averaging the field, is
    what makes the field and the pollen comparable at all: the sites are a biased sample of
    the band, so the two estimators answer different questions over the same region.
    """
    rows = []
    for onset in onsets:
        halves = []
        for lo_, hi_ in ((onset - DO_POST_YR, onset), (onset, onset + DO_PRE_YR)):
            d = long[(long["age"] >= lo_) & (long["age"] <= hi_)].copy()
            d["v"] = value(d)
            halves.append(d.groupby(["site", "channel", "lat"])["v"].mean())
        d = (halves[0] - halves[1]).dropna().reset_index()
        d = d[d["lat"] > NET_LAT]
        pair = d.pivot_table(index="site", columns="channel", values="v").dropna()
        rows.append([pair[v].mean() if v in pair else np.nan for v in VARS])
    return np.asarray(rows)


def _product_variants():
    """``{name: npz}`` for every stored product directory, the shipped one first."""
    found = {}
    for d in sorted(paths.PRODUCT.iterdir() if paths.PRODUCT.is_dir() else []):
        f = d / "reconstruction_fields.npz"
        if f.is_file():
            found[d.name] = np.load(f)
    return {"main": found.pop("main"), **found} if "main" in found else found


def _spread(sd):
    """One number for how wide a posterior is over a field: the rms of its per-cell s.d."""
    return np.sqrt((np.asarray(sd) ** 2).mean(axis=(-2, -1)))


def _contrast_sd(post_var, cross_var):
    """Stated s.d. of ``MTCO - MTWA``, which needs the posterior covariance between them.

    Adding the two channel variances instead would overstate it wherever the analysis leaves
    the two channels positively correlated, which over this grid it mostly does.
    """
    v = np.asarray(post_var, dtype=np.float64)
    return np.sqrt(np.maximum(v[..., 0, :, :] + v[..., 1, :, :]
                              - 2.0 * np.asarray(cross_var, dtype=np.float64), 0.0))


def _structure_function(inp):
    """The lag structure function and per-cell variance of the whole archive, built once.

    Deterministic and cheap, and the same call the observation model of Section 5.3 makes,
    so reading it here keeps the figure script free of any estimator.
    """
    from paleoreco.assim.background import temporal_structure_function
    return temporal_structure_function(
        inp.cube, np.arange(len(inp.ages)),
        max_lag=int(inp.recon_cfg["max_block_lag_steps"]))


def _withheld_identity(inp, safe_valid):
    """``(age, gather index, block centre)`` of every pooled test row of the withholding lane.

    The lane stores what each held-out prediction was and what it should have been, but not
    which age it sat at, so a second predictor cannot be scored on the same rows without
    rebuilding that. The rebuild is a permutation, not an analysis: it replays the lane's own
    fold partition and age loop over the proxy table and reads nothing from an estimator. The
    caller checks it against the stored arrays and drops the comparison rather than trusting
    an order that no longer matches.

    ``cell_observed`` marks the rows whose own grid cell also carried an assimilated
    observation at that age. The lane's stored ``distance_km`` measures proximity in
    kilometres; this measures it in the only unit the analysis can actually resolve, since a
    withheld sample sharing a cell with an assimilated one is predicted from a cell the
    update has already been told the value of.
    """
    cfg = _run(ex.ESTIMATOR_HGAOENKF_MT, paths.LANE_WITHHOLDING,
               "withholding_random_config.json")
    lats, lons = inp.lats, inp.lons
    # The lane saw only the ages its own archive spanned, and the fold partition is drawn
    # from the sites that survive that filter, so a reduced run has to be narrowed the same
    # way before the order can match. On the full archive this drops nothing.
    long = inp.long[inp.long["age"].isin({int(a) for a in inp.ages})]
    n_cells = len(lats) * len(lons)
    safe_flat = np.broadcast_to(safe_valid, (len(VARS), len(lats), len(lons))).ravel()
    dated = sample_block_centres(long)
    obs_ages = np.intersect1d(long["age"].unique(), inp.ages)
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


def _loveclim_at_sites(inp, ident, structure):
    """The simulation read as a predictor of the withheld pollen, two ways.

    ``raw`` is the simulation's own anomaly at the site's cell and the age being predicted.
    ``deflated`` multiplies it by the same lag correlation the estimators' predictions carry,
    which is what makes the two comparable: a held-out sample reports a state some way off in
    time, so every predictor of it is attenuated by the same factor. ``at_centre`` instead
    reads the simulation at the sample's own block midpoint, which is the most the simulation
    could know if its chronology were exactly right.
    """
    cube, ages = inp.cube, inp.ages
    step_yr = float(inp.recon_cfg["step_yr"])
    gather = ident["gather"].to_numpy()
    age, centre = ident["age"].to_numpy(), ident["centre"].to_numpy()
    rho, _ = temporal_terms(*structure, gather, np.abs(age - centre), step_yr)
    anom = (cube.astype(np.float64) - inp.recon["clim_mean"]).reshape(len(ages), -1)
    index = {int(a): i for i, a in enumerate(ages)}
    at_age = anom[np.array([index[a] for a in age]), gather]
    near = np.clip(np.searchsorted(ages, centre), 0, len(ages) - 1)
    near = np.where((near > 0) & (np.abs(ages[near] - centre)
                                  > np.abs(ages[near - 1] - centre)), near - 1, near)
    return {"raw": at_age, "deflated": rho * at_age, "at_centre": anom[near, gather]}


def _ce_against_climatology(truth, pred):
    """The withholding lane's own CE: the reference is a zero anomaly, not a fitted mean."""
    return float(1.0 - np.sum((truth - pred) ** 2) / np.sum(truth ** 2))


def _site_bootstrap_dce(truth, better, worse, site, n_boot=N_BOOT, seed=0):
    """``(dCE, lo, hi)`` for two predictors, resampling whole sites with replacement.

    Every draw scores both predictors on the same redrawn rows, so the interval is on the
    paired difference rather than on either score.
    """
    codes, index = pd.factorize(site)
    n = len(index)
    parts = np.stack([np.bincount(codes, (truth - better) ** 2, n),
                      np.bincount(codes, (truth - worse) ** 2, n),
                      np.bincount(codes, truth ** 2, n)])
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    totals = counts @ parts.T
    draws = (totals[:, 1] - totals[:, 0]) / totals[:, 2]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return (_ce_against_climatology(truth, better) - _ce_against_climatology(truth, worse),
            float(lo), float(hi))


# Progressively stricter exclusions of withheld rows that sit close to an assimilated site.
# Overlapping rather than nested: the first is every row, the second drops rows whose own grid
# cell also carried an assimilated observation, and the rest cut on great-circle distance. They
# are read for how the interval behaves as the test is made harder, not as a monotone series.
VOID_TESTS = (("all rows", None), ("different cell", "cell"),
              ("> 250 km", 250.0), ("> 500 km", 500.0), ("> 1000 km", 1000.0))


def _withheld_mask(kind, distance_km, cell_observed):
    """Row mask for one entry of :data:`VOID_TESTS`."""
    if kind is None:
        return np.ones(len(distance_km), dtype=bool)
    if kind == "cell":
        return ~cell_observed
    return np.nan_to_num(distance_km, nan=-1.0) > float(kind)


def chapter8(inp: Inputs) -> None:
    """Figures 8.1-8.4: the MIS3 reconstruction, what it rests on, and what it shows."""
    cube, ages, lats, lons = inp.cube, inp.ages, inp.lats, inp.lons
    recon, cfg = inp.recon, inp.recon_cfg

    b_scales = recon["b_scales"]
    bi = int(np.argmin(np.abs(b_scales - B_REFERENCE)))
    b_ref = float(b_scales[bi])
    prior_only = recon["prior_only"]
    scored = ~prior_only
    step_yr = float(cfg["step_yr"])
    # The simulation in the product's own anomaly frame, so the two are directly comparable.
    sim = cube.astype(np.float64) - recon["clim_mean"]
    post = recon["mean_anom"].astype(np.float64)
    back = recon["prior_mean_anom"].astype(np.float64)
    w = _area_weights(lats, lons)
    net = np.broadcast_to(np.asarray(lats, float)[:, None] > NET_LAT, (len(lats), len(lons)))
    net_w = np.where(net, w, 0.0)
    onsets = [a for a in sorted(DO_ONSET_BP.values()) if ages.min() <= a <= ages.max()]

    n_cells = len(lats) * len(lons)
    site_cell = obs_cell_index(inp.long["lat"].to_numpy(), inp.long["lon"].to_numpy(),
                               inp.long["channel"].to_numpy(), lats, lons) % n_cells
    reached = np.zeros(n_cells, dtype=bool)
    reached[np.unique(site_cell)] = True
    reached = reached.reshape(len(lats), len(lons))
    sites = inp.long.groupby("site")[["lat", "lon"]].first()
    # Every proxy row placed on the grid and on the age axis, so a field can be read through
    # the network the same way the pollen is. Rows off this archive's age span are dropped,
    # which is what a reduced run leaves behind.
    structure = _structure_function(inp)
    dated = sample_block_centres(inp.long)
    gather_all = obs_cell_index(dated["lat"].to_numpy(), dated["lon"].to_numpy(),
                                dated["channel"].to_numpy(), lats, lons)
    rho_all, _ = temporal_terms(*structure, gather_all,
                                np.abs(dated["age"] - dated["centre"]).to_numpy(), step_yr)
    at_grid = dated.assign(cell=gather_all % n_cells, chan=gather_all // n_cells,
                           rho=rho_all)
    at_grid = at_grid.assign(ti=at_grid["age"].map(
        {int(a): i for i, a in enumerate(ages)})).dropna(subset=["ti"])
    at_grid["ti"] = at_grid["ti"].astype(int)
    n_lon = len(lons)
    at_site = lambda field: (lambda d: field[d["ti"].to_numpy(), d["chan"].to_numpy(),
                                             d["cell"].to_numpy() // n_lon,
                                             d["cell"].to_numpy() % n_lon])

    print(f"  8.0 {cfg['n_ages']} ages, {cfg['n_prior_only_ages']} prior-only, "
          f"{int(recon['n_obs'].sum())} rows; amplitudes {list(b_scales)}, quoted at c={b_ref:g}")
    print(f"  8.0 sites per age median {int(np.median(recon['n_sites'][scored]))} "
          f"(range {int(recon['n_sites'][scored].min())}-{int(recon['n_sites'].max())}); "
          f"{len(onsets)} D-O onsets in range; network reaches {int(reached.sum())} of "
          f"{n_cells} cells")
    # Every published amplitude's distance from the simulation it was built on, which is the
    # measure Section 8.1 quotes when it calls the product a derived analysis.
    print("  8.0 corr(product, simulation state) over scored ages: "
          + ", ".join(f"c={float(bb):g}: "
                      f"{np.corrcoef(post[j][scored].ravel(), sim[scored].ravel())[0, 1]:.4f}"
                      for j, bb in enumerate(b_scales)))

    # --- Figure 8.1: one state of the product, with what it publishes beside it ------------
    # A reader who never sees a field cannot judge the object. The window is the interstadial
    # half of the strongest event, read as a mean over its 600 yr rather than at one 25-yr
    # state, because Section 8.3 finds nothing is resolved below 250 yr.
    events = [e for e in sorted(DO_ONSET_BP) if DO_ONSET_BP[e] in onsets]
    if not events:
        print("  8.1 skipped: no D-O onset falls inside this archive")
        return
    comp_p, comp_s = _do_composite(post[bi], ages, onsets), _do_composite(sim, ages, onsets)
    net_p = np.array([[_area_mean(e[r], net_w) for r in range(len(VARS))] for e in comp_p])
    net_s = np.array([[_area_mean(e[r], net_w) for r in range(len(VARS))] for e in comp_s])
    lead = int(np.argmax(net_p[:, 0]))
    onset = DO_ONSET_BP[events[lead]]
    warm, _ = _event_halves(ages, onset)
    state = post[bi][warm].mean(axis=0)
    # The lower row is a property of the whole product rather than of this window, and it is
    # read as a fraction of the archive's own per-cell spread rather than in degrees: the
    # spread itself ranges over an order of magnitude across the grid, so degrees would put
    # the whole map in the bottom fifth of a colour bar and say nothing about what was learnt.
    kept = (np.sqrt(recon["post_var"].astype(np.float64)[bi][scored]).mean(axis=0)
            / np.sqrt(recon["prior_var"].astype(np.float64)))
    print(f"  8.1 GI-{events[lead]} onset {onset} yr BP; interstadial window "
          f"{int(ages[warm].min())}-{int(ages[warm].max())} yr BP, {int(warm.sum())} states")

    fig, axes = plt.subplots(2, 2, figsize=(REPORT_WIDTH, 3.55), constrained_layout=True,
                             subplot_kw={"projection": PLATE})
    # One scale across both channels here, unlike Figure 8.4: this panel pair is read for how
    # much larger the winter field is than the summer one at the same moment.
    lim = float(np.ceil(np.percentile(np.abs(state), 99.5)))
    for r, channel in enumerate(VARS):
        ax = axes[0, r]
        mesh = draw_field(ax, lats, lons, state[r], vmin=-lim, vmax=lim, cmap="RdBu_r")
        ax.scatter(sites["lon"], sites["lat"], s=1.0, c="k", linewidth=0, transform=PLATE,
                   zorder=3)
        map_axes(ax, ylabel=(r == 0))
        ax.set_title(f"({'ab'[r]}) {channel.upper()} anomaly", fontsize=6.8)
        print(f"  8.1 {channel} state {state[r].min():+.2f} to {state[r].max():+.2f} degC, "
              f"scale +-{lim:g}, {100 * float((np.abs(state[r]) > lim).mean()):.1f}% clipped")
    cb = fig.colorbar(mesh, ax=axes[0, :], shrink=0.86, pad=0.012, aspect=13, extend="both")
    cb.set_label("anomaly (°C)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2)

    for r, channel in enumerate(VARS):
        ax = axes[1, r]
        mesh = draw_field(ax, lats, lons, kept[r], vmin=0.0, vmax=2.0, cmap="PuOr_r")
        ax.scatter(sites["lon"], sites["lat"], s=1.0, c="#1b7837", linewidth=0,
                   transform=PLATE, zorder=3)
        map_axes(ax, xlabel=True, ylabel=(r == 0))
        ax.set_title(f"({'cd'[r]}) {channel.upper()} s.d. / archive s.d.", fontsize=6.8)
    cb = fig.colorbar(mesh, ax=axes[1, :], shrink=0.86, pad=0.012, aspect=13, extend="max",
                      ticks=[0.0, 0.5, 1.0, 1.5, 2.0])
    cb.set_label("posterior / climatological spread", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2)
    # The stated spread on the seasonal contrast needs the posterior covariance between the
    # channels, which the product stores and which no other figure reads.
    pv = recon["post_var"].astype(np.float64)[bi][scored]
    print(f"  8.1 contrast s.d. median {np.median(_contrast_sd(pv, recon['post_cross_var'][bi][scored])):.4f} "
          f"degC against {np.median(np.sqrt(pv[:, 0] + pv[:, 1])):.4f} ignoring the "
          f"cross-covariance; median posterior channel correlation "
          f"{np.median(recon['post_cross_var'].astype(np.float64)[bi][scored] / np.sqrt(pv[:, 0] * pv[:, 1])):.4f}")
    save(fig, "fig08_01_reconstruction_field")

    # --- Figure 8.2: the reconstruction against the simulation, through time ----------------
    # Two places, because a domain mean over a grid that is 94.5% unobserved and antiphased
    # between the hemispheres hides most of what the product does; the archive's most active
    # cell is where the D-O signal it was built to carry actually lives.
    hot = np.unravel_index(int(np.argmax(cube[:, 0].std(axis=0))), (len(lats), len(lons)))
    hot_lon = ((float(lons[hot[1]]) + 180.0) % 360.0) - 180.0
    hot_has_site = bool(reached[hot])
    places = ((f"{NET_LAT:g}°N–90°N", lambda f: _area_mean(f, net_w)),
              (f"{float(lats[hot[0]]):.0f}°N {abs(hot_lon):.0f}°"
               f"{'E' if hot_lon >= 0 else 'W'}"
               f"{'' if hot_has_site else ', no site'}", lambda f: f[..., hot[0], hot[1]]))
    smooth = lambda s: da.lowpass_time(np.asarray(s, float)[:, None], SMOOTH_YR, step_yr)[:, 0]
    # A low-pass zero-pads, so its first and last window is pulled towards zero; those ages
    # are dropped rather than drawn, as the timescale metrics of Chapter 7 drop them.
    trim = da.timescale_trim(SMOOTH_YR, step_yr)
    edge = np.ones(len(ages), dtype=bool)
    if trim:
        edge[:trim] = edge[-trim:] = False
    shown = scored & edge
    gap = np.where(shown, 1.0, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(REPORT_WIDTH, 3.45), sharex=True,
                             constrained_layout=True)
    for r, channel in enumerate(VARS):
        for c, (where, take) in enumerate(places):
            ax = axes[r, c]
            for o in onsets:
                ax.axvspan((o - DO_POST_YR) / 1000, (o + DO_PRE_YR) / 1000,
                           color="0.92", lw=0, zorder=0)
            band = np.array([smooth(take(post[j, :, r])) for j in range(len(b_scales))])
            sim_s, x = smooth(take(sim[:, r])), ages / 1000
            ax.axhline(0.0, color="0.65", lw=0.5, zorder=1)
            ax.fill_between(x, band.min(axis=0) * gap, band.max(axis=0) * gap, color=PRODUCT_C,
                            alpha=0.22, lw=0, zorder=2)
            ax.plot(x, take(post[bi, :, r]) * np.where(scored, 1.0, np.nan), color=PRODUCT_C,
                    lw=0.3, alpha=0.45, zorder=3)
            ax.plot(x, sim_s * np.where(edge, 1.0, np.nan), color=SIM_C, lw=1.0,
                    ls=(0, (4, 2.5)), zorder=4, label="LOVECLIM")
            ax.plot(x, band[bi] * gap, color=PRODUCT_C, lw=1.2, zorder=5,
                    label=f"reconstruction ($c = {b_ref:g}$)")
            ax.set_title(f"({'acbd'[2 * c + r]}) {channel.upper()}, {where}", fontsize=7.0)
            if c == 0:
                ax.set_ylabel(f"{channel.upper()} anomaly (°C)")
            if r == len(VARS) - 1:
                ax.set_xlabel("Age (ka BP)")
            raw, ref = take(post[bi, :, r])[scored], take(sim[:, r])[scored]
            # How much of the plotted spread the amplitude band accounts for, which is what
            # the caption claims and Section 8.3 qualifies.
            half = (band.max(axis=0) - band.min(axis=0))[shown] / 2.0
            print(f"  8.2 {channel} {where:18s}: product s.d. {raw.std():.3f} degC, LOVECLIM "
                  f"{ref.std():.3f}, r {np.corrcoef(raw, ref)[0, 1]:+.3f}; at {SMOOTH_YR:.0f} yr "
                  f"product {band[bi][shown].std():.3f}, LOVECLIM {sim_s[shown].std():.3f}, "
                  f"r {np.corrcoef(band[bi][shown], sim_s[shown])[0, 1]:+.3f}; band half-width "
                  f"median {np.median(half):.3f} degC ({100 * np.median(half) / band[bi][shown].std():.0f}% "
                  f"of the series s.d.), max {half.max():.3f}, extremes correlate "
                  f"{np.corrcoef(band[0][shown], band[-1][shown])[0, 1]:.4f}")
    axes[0, 0].legend(loc="upper left", frameon=False, handlelength=1.8, borderpad=0.1,
                      labelspacing=0.2, fontsize=6.4)
    # Older on the left, as Figure 3.2(b) draws the same axis.
    axes[0, 0].set_xlim(ages.max() / 1000, ages.min() / 1000)
    if prior_only.any():
        for ax in axes.ravel():
            ax.axvspan(ages[prior_only].min() / 1000, ages[prior_only].max() / 1000,
                       color="C3", alpha=0.14, lw=0, zorder=0)
        axes[0, 0].annotate(f"{int(prior_only.sum())} ages without pollen",
                            xy=(ages[prior_only].max() / 1000, 0.97),
                            xycoords=("data", "axes fraction"), textcoords="offset points",
                            xytext=(-3, 0), fontsize=6.0, color="C3", ha="right", va="top")
    save(fig, "fig08_02_reconstruction_series")

    # --- Figure 8.3: why not just read the simulation ---------------------------------------
    # The only measurement in the project that answers that question without using the pollen
    # twice, its honest qualification, and what the improvement consists of where it survives.
    method = ex.method_label(ex.ESTIMATOR_HGAOENKF_MT, ex.TEMPORAL_DEFLATE)
    zw = _run(ex.ESTIMATOR_HGAOENKF_MT, paths.LANE_WITHHOLDING,
              "withholding_random_predictions.npz")
    truth = zw["actual"]
    ident = _withheld_identity(inp, recon["safe_valid"])
    aligned = (len(ident) == len(truth)
               and np.allclose(ident["actual"].to_numpy(), truth)
               and np.array_equal(ident["site"].to_numpy(), zw["site"])
               and np.array_equal(ident["channel"].to_numpy(), zw["channel"]))
    print(f"  8.3 withheld rows {len(truth)} from {len(np.unique(zw['site']))} sites; "
          f"rebuilt identity aligned: {aligned}")
    wb = list(zw["b_scales"])
    wj = int(np.argmin(np.abs(np.asarray(wb) - b_ref)))
    product = zw[f"climatological_pred_{method}"][wj]
    lc = _loveclim_at_sites(inp, ident, structure) if aligned else None
    # The bars carry only the two predictors the comparison turns on. The prior-free
    # references and the unattenuated simulation are still scored, because the prose quotes
    # them, but a five-bar chart buried the one contrast it exists to show.
    for name, pred in (("the climatology", np.zeros_like(truth)),
                       ("nearest", zw["naive_nearest"]), ("idw", zw["naive_idw"])):
        print(f"  8.3 held-out CE {_ce_against_climatology(truth, pred):+.4f}  {name}")
    if aligned:
        for name, pred in (("LOVECLIM, at the analysis age", lc["raw"]),
                           ("LOVECLIM, at the block centre", lc["at_centre"])):
            print(f"  8.3 held-out CE {_ce_against_climatology(truth, pred):+.4f}  {name}")
    entries = [("LOVECLIM", lc["deflated"], SIM_C)] if aligned else []
    entries += [(f"the product, $c = {float(wb[wj]):g}$", product, PRODUCT_C)]
    values = [_ce_against_climatology(truth, p) for _, p, _ in entries]
    # What any predictor could reach: the withheld value is itself a measurement, and the
    # observation model says how much of its variance is that measurement's own error.
    noise = float((zw["sse"] + zw["rep_var"] + zw["resid_var"]).mean())
    ceiling = 1.0 - noise / float((truth ** 2).mean())

    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 1.85), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.06]})
    ypos = np.arange(len(entries), dtype=float)
    axes[0].barh(ypos, values, height=0.34, color=[c for _, _, c in entries], zorder=3)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([n for n, _, _ in entries], fontsize=6.0)
    axes[0].set_ylim(len(entries) - 0.5, -0.5)
    axes[0].set_xlabel("coefficient of efficiency")
    axes[0].set_xlim(min(-0.01, 1.15 * min(values)), 1.30 * max(values))
    axes[0].axvline(0.0, color="0.45", lw=0.7, zorder=2)
    axes[0].text(0.0, 1.03, "(a) predicting withheld pollen", transform=axes[0].transAxes,
                 fontsize=7.0, fontweight="bold", va="bottom")
    for name, value in zip([n for n, _, _ in entries], values):
        print(f"  8.3 held-out CE {value:+.4f}  {name}")
    print(f"  8.3 attainable CE {ceiling:.4f} (mean assumed noise variance {noise:.3f} of "
          f"{float((truth ** 2).mean()):.3f} degC^2; sse {float(zw['sse'].mean()):.3f}, rep "
          f"{float(zw['rep_var'].mean()):.3f}, temporal {float(zw['resid_var'].mean()):.3f})")

    # Panel (b): the same difference under progressively stricter exclusions of withheld rows
    # that sit close to an assimilated one. Folds are random over sites rather than
    # geographic, so this is where the lane's residual dependence is measured rather than
    # asserted.
    if aligned:
        cell_obs = ident["cell_observed"].to_numpy().astype(bool)
        print(f"  8.3 withheld rows whose own cell also carried an assimilated observation: "
              f"{int(cell_obs.sum())} of {len(cell_obs)} ({100 * cell_obs.mean():.1f}%), "
              f"{len(np.unique(zw['site'][cell_obs]))} sites; median distance to the nearest "
              f"assimilated site {np.nanmedian(zw['distance_km']):.0f} km")
        # Drawn horizontally, like panel (a), so the category names read straight and the
        # interval that matters is the one the eye already scans left to right.
        ys, ds, los, his, labels = [], [], [], [], []
        for label, kind in VOID_TESTS:
            m = _withheld_mask(kind, zw["distance_km"], cell_obs)
            if m.sum() < 50 or len(np.unique(zw["site"][m])) < 5:
                continue
            d, lo, hi = _site_bootstrap_dce(truth[m], product[m], lc["deflated"][m],
                                            zw["site"][m])
            ys.append(float(len(ys)))
            ds.append(d); los.append(lo); his.append(hi); labels.append(label)
            print(f"  8.3 {label:15s} rows {int(m.sum()):6d} sites "
                  f"{len(np.unique(zw['site'][m])):3d} product "
                  f"{_ce_against_climatology(truth[m], product[m]):+.4f} LOVECLIM "
                  f"{_ce_against_climatology(truth[m], lc['deflated'][m]):+.4f} dCE {d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]")
        ys = np.asarray(ys)
        axes[1].axvline(0.0, color="0.45", lw=0.7, zorder=1)
        axes[1].errorbar(ds, ys, xerr=[np.array(ds) - np.array(los),
                                       np.array(his) - np.array(ds)],
                         fmt="o", ms=3.2, lw=1.0, capsize=2.0, color=PRODUCT_C, zorder=3)
        axes[1].set_yticks(ys)
        axes[1].set_yticklabels(labels, fontsize=6.0)
        axes[1].set_ylim(len(ys) - 0.5, -0.5)
        axes[1].set_xlabel("$\\Delta$CE, product $-$ LOVECLIM")
    else:
        # The rebuilt row identity is what lets a second predictor be scored on these rows, so
        # without it there is no comparison to draw. A reduced run lands here; an empty frame
        # with default ticks would read as a failed plot rather than an absent one.
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "row identity could not be\nrebuilt for this run",
                     transform=axes[1].transAxes, fontsize=6.2, color="0.45",
                     ha="center", va="center")
    axes[1].text(0.0, 1.03, "(b) how much is proximity?", transform=axes[1].transAxes,
                 fontsize=7.0, fontweight="bold", va="bottom")

    # The seasonal statistic itself is reported in the text and in the chain below rather
    # than drawn: with the composite maps of Figure 8.4 beside it, a third panel of eight
    # points earned less than the space it cost.
    for tag, value in (("pollen", lambda d: (d["y"] - d["my"]).to_numpy()),
                       ("LOVECLIM", at_site(sim)), ("reconstruction", at_site(post[bi]))):
        m = np.nanmean(_site_composite(at_grid, onsets, value), axis=0)
        print(f"  8.3 site composite, {tag:14s}: dMTCO {m[0]:+.4f} dMTWA {m[1]:+.4f} "
              f"ratio {m[0] / m[1]:.4f}")
    save(fig, "fig08_03_evidence")

    # --- Figure 8.4: the D-O composite ------------------------------------------------------
    mean_p, mean_s = comp_p.mean(axis=0), comp_s.mean(axis=0)

    fig, maps = plt.subplots(2, 2, figsize=(REPORT_WIDTH, 3.55), constrained_layout=True,
                             subplot_kw={"projection": PLATE})
    # One scale per channel, not one for the figure: the MTCO composite reaches 16 degC and
    # the MTWA one 5, so a shared scale would saturate the first and flatten the second. The
    # limit is the wider of the two fields' 99.5th percentiles, which keeps the clipped
    # fraction under half a per cent in every panel; taking the 99th instead saturated the
    # whole Arctic of the simulation's MTWA, where a solid block reads as uniform warming
    # rather than as off-scale.
    for r, channel in enumerate(VARS):
        lim = float(np.ceil(max(np.percentile(np.abs(mean_p[r]), 99.5),
                                np.percentile(np.abs(mean_s[r]), 99.5))))
        for c, (field, label) in enumerate(((mean_p[r], "reconstruction"),
                                            (mean_s[r], "LOVECLIM"))):
            ax = maps[r, c]
            mesh = draw_field(ax, lats, lons, field, vmin=-lim, vmax=lim, cmap="RdBu_r")
            if c == 0:
                ax.scatter(sites["lon"], sites["lat"], s=1.1, c="k", linewidth=0,
                           transform=PLATE, zorder=3)
            map_axes(ax, xlabel=(r == len(VARS) - 1), ylabel=(c == 0))
            ax.set_title(f"({'abcd'[2 * r + c]}) {channel.upper()} {label}", fontsize=6.8)
        cb = fig.colorbar(mesh, ax=maps[r, :], shrink=0.88, pad=0.015, aspect=14,
                          extend="both")
        cb.set_label("interstadial − stadial (°C)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6, length=2)
        slope, icpt = np.polyfit(mean_s[r].ravel(), mean_p[r].ravel(), 1)
        resid = mean_p[r] - (slope * mean_s[r] + icpt)
        print(f"  8.4 {channel} map scale ±{lim:g} degC; product range "
              f"{mean_p[r].min():+.2f} to {mean_p[r].max():+.2f}, LOVECLIM "
              f"{mean_s[r].min():+.2f} to {mean_s[r].max():+.2f}; clipped "
              f"{100 * float((np.abs(mean_p[r]) > lim).mean()):.2f}% of the product, "
              f"{100 * float((np.abs(mean_s[r]) > lim).mean()):.2f}% of LOVECLIM")
        print(f"  8.4 {channel}: corr(product, LOVECLIM) "
              f"{np.corrcoef(mean_p[r].ravel(), mean_s[r].ravel())[0, 1]:.3f}, slope {slope:.2f}, "
              f"intercept {icpt:+.3f}, residual rms {resid.std():.3f} degC "
              f"({100 * resid.std() / mean_p[r].std():.0f}% of the product's own composite "
              f"s.d. {mean_p[r].std():.3f})")
    save(fig, "fig08_04_do_composite")

    # --- The chain from the background the analysis starts at to the observations it fits ---
    # Five sources, one estimator, the same sites and windows. The deflated row is what the
    # analysis is actually fitting: Section 5.3 divides each observation by its own lag
    # correlation before assimilating it, and that correction is larger in winter.
    chain = (("the analog background", at_site(back)),
             ("LOVECLIM", at_site(sim)),
             ("the pollen", lambda d: (d["y"] - d["my"]).to_numpy()),
             ("the pollen / rho", lambda d: ((d["y"] - d["my"]) / d["rho"]).to_numpy()),
             (f"the product, c={b_ref:g}", at_site(post[bi])))
    for tag, value in chain:
        m = np.nanmean(_site_composite(at_grid, onsets, value), axis=0)
        print(f"  T8.3 {tag:24s} dMTCO {m[0]:+.4f} dMTWA {m[1]:+.4f} ratio {m[0] / m[1]:.4f}")
    for channel in VARS:
        print(f"  T8.3 median rho, {channel}: "
              f"{float(np.median(at_grid.loc[at_grid['channel'] == channel, 'rho'])):.4f}")
    for j, bb in enumerate(b_scales):
        s = np.nanmean(_site_composite(at_grid, onsets, at_site(post[j])), axis=0)
        print(f"  8.3 c={float(bb):<5g} site ratio {s[0] / s[1]:.4f}")

    # Two regions the prose names: the one Liu et al. report as a model-data disagreement,
    # and the one where the network reports a signal the reconstruction does not carry. Read
    # through the site estimator of Figure 8.3(c), restricted to the sites inside each box.
    lon180 = ((at_grid["lon"] + 180.0) % 360.0) - 180.0
    boxes = (("western North America", (at_grid["lat"] > 30) & (at_grid["lat"] < 62)
              & (lon180 > -170) & (lon180 < -100)),
             ("the tropics", at_grid["lat"].abs() < NET_LAT))
    for name, inside in boxes:
        sub = at_grid[inside]
        if not len(sub):
            continue
        # The site estimator keeps the northern extratropics; inside a box the latitude cut
        # would empty it, so the box itself is the selection.
        region = sub.assign(lat=NET_LAT + 1.0)
        for tag, value in (("pollen", lambda d: (d["y"] - d["my"]).to_numpy()),
                           ("LOVECLIM", at_site(sim)),
                           ("product", at_site(post[bi]))):
            m = np.nanmean(_site_composite(region, onsets, value), axis=0)
            print(f"  8.3 {name:22s} {sub['site'].nunique():3d} sites, {tag:9s} "
                  f"dMTCO {m[0]:+.4f} dMTWA {m[1]:+.4f}")

    # Where the composite is consistent across events, and where it reaches without data.
    for r, channel in enumerate(VARS):
        t = comp_p[:, r].mean(0) / (comp_p[:, r].std(0, ddof=1) / np.sqrt(len(comp_p)))
        print(f"  8.4 {channel} across-event |t|: median {np.median(np.abs(t)):.3f}")
    for i, e in enumerate(events):
        print(f"  T8.x GI-{e:<3d} {DO_ONSET_BP[e]:6d} | product {net_p[i, 0]:+6.2f} "
              f"{net_p[i, 1]:+6.2f} | LOVECLIM {net_s[i, 0]:+6.2f} {net_s[i, 1]:+6.2f}")
    for tag, pts in (("product", net_p), ("LOVECLIM", net_s)):
        m, d_ = pts.mean(axis=0), pts[:, 0] - pts[:, 1]
        se = d_.std(ddof=1) / np.sqrt(len(d_)) if len(d_) > 1 else np.nan
        print(f"  8.3 NET {tag:9s} mean {m[0]:+.3f}/{m[1]:+.3f}, contrast {d_.mean():+.3f} "
              f"+-{se:.3f} (s.e. over {len(d_)} events), ratio of means {m[0] / m[1]:.4f}")
    for tag, mask in (("observed", net & reached), ("unobserved", net & ~reached)):
        if not mask.any():
            continue
        mw = np.where(mask, w, 0.0)
        ratio = lambda f: _area_mean(f[0], mw) / _area_mean(f[1], mw)
        print(f"  8.3 NET {tag:11s} ({int(mask.sum()):4d} cells): dMTCO "
              f"{_area_mean(mean_p[0], mw):+.3f} vs LOVECLIM {_area_mean(mean_s[0], mw):+.3f}; "
              f"ratio {ratio(mean_p):.4f} vs {ratio(mean_s):.4f}")

    # --- What the product claims to know, and where it should not be believed ---------------
    post_sd = np.sqrt(recon["post_var"].astype(np.float64))
    arch_sd = np.sqrt(recon["prior_var"].astype(np.float64))
    kept = post_sd[bi][scored].mean(axis=0) / arch_sd
    for r, channel in enumerate(VARS):
        print(f"  8.5 {channel}: posterior s.d. median "
              f"{np.median(post_sd[bi][scored][:, r]):.4f} degC; archive rms "
              f"{_spread(arch_sd[r]):.4f}; keeps {np.median(kept[r]):.4f} of the "
              f"climatological spread, {np.median(kept[r][reached]):.4f} where the network "
              f"reaches and {np.median(kept[r][~reached]):.4f} elsewhere; wider than the "
              f"archive at {100 * float((post_sd[bi][scored][:, r] > arch_sd[r]).mean()):.1f}% "
              f"of cells")
    for j, bb in enumerate(b_scales):
        print(f"  T8.2 c={float(bb):<5g} median posterior s.d. over both channels "
              f"{np.median(post_sd[j][scored]):.4f} degC")
    if prior_only.any():
        print(f"  8.5 pollen-free ages: MTCO posterior rms s.d. "
              f"{_spread(post_sd[bi][prior_only][:, 0]).mean():.4f} degC against the "
              f"archive's {_spread(arch_sd[0]):.4f}; anomaly max "
              f"{np.abs(post[bi][prior_only]).max():.3g}")
    spread_rms = np.sqrt((((post[:, scored].max(axis=0)
                            - post[:, scored].min(axis=0)) / 2) ** 2).mean())
    print(f"  8.5 across the band: rms half-range {spread_rms:.4f} degC, rms(c_min - c_max) "
          f"{np.sqrt(((post[0, scored] - post[-1, scored]) ** 2).mean()):.4f}, extremes "
          f"correlate {np.corrcoef(post[0, scored].ravel(), post[-1, scored].ravel())[0, 1]:.4f}; "
          f"stated posterior rms s.d. at c={b_ref:g} "
          f"{np.sqrt((post_sd[bi][scored] ** 2).mean()):.4f} degC; field s.d. "
          f"{post[bi][scored].std():.4f}")
    dist = np.abs(ages[recon["analog_index"][scored]] - ages[scored][:, None])
    print(f"  8.5 analogs: median distance {np.median(dist):.0f} yr, closest {dist.min():.0f}, "
          f"median within-analysis spread {np.median(ages[recon['analog_index'][scored]].std(axis=1)):.0f} yr, "
          f"{len(np.unique(recon['analog_index'][scored]))} of {len(ages)} states drawn")
    south = np.broadcast_to(np.asarray(lats, float)[:, None] < sites["lat"].min(),
                            (len(lats), len(lons)))
    if south.any():
        sw = np.where(south, w, 0.0)
        print(f"  8.5 south of {sites['lat'].min():.1f}N ({int(south.sum())} cells): product "
              f"dMTCO {_area_mean(mean_p[0], sw):+.3f} vs LOVECLIM "
              f"{_area_mean(mean_s[0], sw):+.3f}")
    hottest = np.unravel_index(int(np.argmax(mean_p[0])), mean_p[0].shape)
    over = mean_p[0] > 12.0
    if over.any():
        la = np.broadcast_to(np.asarray(lats, float)[:, None], mean_p[0].shape)[over]
        print(f"  8.5 {int(over.sum())} cells above +12 degC dMTCO, latitudes {la.min():.1f} "
              f"to {la.max():.1f} N, any with a site: {bool((reached & over).any())}")
    print(f"  8.5 peak product dMTCO {mean_p[0][hottest]:+.3f} at "
          f"{lats[hottest[0]]:.1f}N {lons[hottest[1]]:.1f}E; LOVECLIM there "
          f"{mean_s[0][hottest]:+.3f}")

    # --- Appendix table: how far the field moves when a design choice is reversed ------------
    for name, z in _product_variants().items():
        if name == "main":
            continue
        k = int(np.argmin(np.abs(z["b_scales"] - b_ref)))
        other = z["mean_anom"].astype(np.float64)[k]
        d = (post[bi] - other)[scored]
        rms = float(np.sqrt((d ** 2).mean()))
        cv = _do_composite(other, ages, onsets).mean(axis=0)
        site = np.nanmean(_site_composite(at_grid, onsets, at_site(other)), axis=0)
        print(f"  T8.4 {name:18s} (c={float(z['b_scales'][k]):g}) rms diff {rms:.4f} degC "
              f"({100 * rms / post[bi][scored].std():.0f}% of field s.d.), field s.d. "
              f"{other[scored].std():.4f}, NET ratio "
              f"{_area_mean(cv[0], net_w) / _area_mean(cv[1], net_w):.4f}, site ratio "
              f"{site[0] / site[1]:.4f}")


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("07_figures", [f"chapter {n}" for n in (1, 2, 3, 4, 5, 6, 7, 8)])
    inp = load_inputs(C.SMOKE_AGES if smoke else None)
    for n, fn in ((1, chapter1), (2, chapter2), (3, chapter3), (4, chapter4),
                  (5, chapter5), (6, chapter6), (7, chapter7), (8, chapter8)):
        if stages.run(f"chapter {n}"):
            fn() if fn in (chapter2, chapter4, chapter7) else fn(inp)
    stages.done()


if __name__ == "__main__":
    main()
