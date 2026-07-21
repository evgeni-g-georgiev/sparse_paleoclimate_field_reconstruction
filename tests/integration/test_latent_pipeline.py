"""A latent method drives the real withholding lane and tags its rows.

Confirms the factory wiring: a PCA :func:`latent_var` plugged into ``run_withholding``
through ``make_method`` runs the full lane (prepare/apply sweep, decode, predict at
withheld sites) and writes finite, ``space``-tagged metric rows, with the
climatological-only background honoured.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim import experiments as ex
from paleoreco.assim.compressors import PCACompressor, latent_prior
from paleoreco.assim.latent import latent_var
from paleoreco.data.cube import apply_anomaly, compute_zscore_stats
from paleoreco.eval.shared import pod_fit


def test_run_withholding_with_pca_latent(tmp_path, cube, ages, lats, lons, valid, obs_long):
    stats = compute_zscore_stats(cube, np.arange(len(cube)), valid)
    cube_anom = apply_anomaly(cube, stats)
    safe = stats["safe_valid"]
    pca = PCACompressor.from_pod_fit(
        pod_fit(cube_anom, np.arange(len(cube)), safe, max_k=6), k=6)
    B_z, z_clim = latent_prior(pca, cube_anom, np.arange(len(cube)))

    def make_method(prior, shape):
        return latent_var(pca, B_z, z_clim, shape, prior.safe_valid)

    df = ex.run_withholding(
        cube, ages, lats, lons, valid, obs_long, str(tmp_path / "pca"),
        make_method=make_method, space="pca",
        k_folds=3, b_scales=(1.0,), seed=0,
    )

    threedvar = df[df["method"] == "3dvar"]
    assert len(threedvar) > 0
    assert (threedvar["space"] == "pca").all()
    assert set(threedvar["background"].unique()) == {"climatological"}
    # Nested CV writes both the held-out selection split and the reported test split.
    assert set(threedvar["split"].unique()) == {"selection", "test"}
    # RMSE is always defined; CE divides by the withheld-actual variance, which the
    # tiny synthetic fixture can drive to zero, so it is not a wiring signal here.
    rmse = threedvar[(threedvar["metric"] == "rmse") & (threedvar["channel"] == "pooled")]
    assert len(rmse) > 0 and np.isfinite(rmse["value"].to_numpy()).all()
    assert (tmp_path / "pca" / "metrics.csv").exists()
