"""Drawing gridded fields on PlateCarree axes."""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

CELL_DEG = 5.625        # LOVECLIM grid spacing, both axes
PLATE = ccrs.PlateCarree()

# The grid's own southern edge, so no map carries a blank strip below the data.
LAT_S = -87.1875


def cell_edges(centres: np.ndarray, clip: tuple[float, float] | None = None) -> np.ndarray:
    """Edges of a uniform cell-centred axis, one longer than the centres.

    Using the centres as an image extent would shift the field half a cell, 2.8125 degrees here.
    """
    c = np.asarray(centres, dtype=float)
    step = np.diff(c)
    if not np.allclose(step, step[0]):
        raise ValueError(f"axis is not uniform; steps run {step.min()} to {step.max()}")
    h = step[0] / 2.0
    edges = np.concatenate([c - h, [c[-1] + h]])
    return edges if clip is None else np.clip(edges, *clip)


def draw_field(ax, lats, lons, field, *, vmin, vmax, cmap="RdBu_r"):
    """Draw a ``(n_lat, n_lon)`` field over exact cell edges and return the mesh."""
    lat_e = cell_edges(lats, clip=(-90.0, 90.0))
    lon_e = cell_edges(lons)
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", transform=PLATE,
              rasterized=True)
    mesh = ax.pcolormesh(lon_e, lat_e, field, **kw)
    # The wrapped copy fills the sliver the dateline-straddling cell leaves at the left edge.
    ax.pcolormesh(lon_e - 360.0, lat_e, field, **kw)
    return mesh


def map_axes(ax, *, xlabel=False, ylabel=True, coastlines=True):
    """Global extent, modern coastlines and degree-labelled ticks.

    ``zero_direction_label`` stays off: the prime meridian is 0 degrees, not "0 W".
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
