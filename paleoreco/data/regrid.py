"""Conservative regridding of rectilinear lat-lon fields onto a target grid.

Wraps xarray-regrid's area-weighted conservative remap. It resolves longitude
convention and periodicity from the coordinates themselves, so the source and
target may use different longitude ranges (0..360 vs -180..180) and different
latitude spacings (Gaussian vs regular) without any pre-alignment.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401  registers the .regrid accessor on xarray objects


def conservative_regrid(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """Area-weighted remap of ``field`` onto ``(tgt_lat, tgt_lon)``.

    The trailing two axes of ``field`` are latitude then longitude; any leading
    axes (months, decades, ...) are carried through unchanged.
    """
    field = np.asarray(field)
    *lead, n_lat, n_lon = field.shape
    flat = field.reshape(-1, n_lat, n_lon).astype(np.float64)
    da = xr.DataArray(
        flat, dims=("sample", "lat", "lon"),
        coords={"lat": np.asarray(src_lat, np.float64), "lon": np.asarray(src_lon, np.float64)},
    )
    target = xr.Dataset(coords={"lat": np.asarray(tgt_lat, np.float64),
                                "lon": np.asarray(tgt_lon, np.float64)})
    out = da.regrid.conservative(target, latitude_coord="lat", time_dim=None).values
    return out.reshape(*lead, len(tgt_lat), len(tgt_lon)).astype(field.dtype)
