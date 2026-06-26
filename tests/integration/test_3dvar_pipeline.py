"""End-to-end smoke test of the PPE 3DVar runner (paleoreco.assim.experiments).

Runs the real orchestration on tiny synthetic inputs into a tmp dir: it must
produce a tidy metrics.csv with the expected schema and finite skill, and write
the analysis/config artifacts. Guards the whole assim wiring through the refactor.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paleoreco.assim import experiments as ex


def test_run_ppe_writes_metrics_and_artifacts(tmp_path, cube, ages, lats, lons, valid, obs_long):
    out_dir = tmp_path / "ppe_raw"
    df = ex.run_ppe(
        cube, ages, lats, lons, valid, obs_long, str(out_dir),
        b_reg="raw", stride=3, n_noise=2,
        rep_levels=(0.0,), b_scales=(1.0,), seed=0,
    )

    # Returned rows and the persisted CSV agree on schema.
    assert isinstance(df, pd.DataFrame)
    expected_cols = {"method", "B_reg", "lane", "fold", "rep_K", "b_scale",
                     "background", "do_event", "channel", "metric", "value"}
    assert expected_cols.issubset(df.columns)

    assert os.path.exists(out_dir / "metrics.csv")
    assert os.path.exists(out_dir / "ppe_analysis.npz")
    assert os.path.exists(out_dir / "ppe_skill_vs_distance.npz")
    assert os.path.exists(out_dir / "config.json")

    # The 3DVar pooled CE rows exist and are finite.
    ce = df[(df["method"] == "3dvar") & (df["metric"] == "ce")
            & (df["channel"] == "pooled") & (df["do_event"] == "all")]
    assert len(ce) > 0
    assert np.isfinite(ce["value"].to_numpy()).all()


def test_run_ppe_posterior_var_within_prior(tmp_path, cube, ages, lats, lons, valid, obs_long):
    out_dir = tmp_path / "ppe_pv"
    ex.run_ppe(cube, ages, lats, lons, valid, obs_long, str(out_dir),
               b_reg="raw", stride=3, n_noise=1, rep_levels=(0.0,), b_scales=(1.0,), seed=0)
    with np.load(out_dir / "ppe_analysis.npz") as z:
        post_var = z["post_var"]       # (n_b, T, 2, n_lat, n_lon)
        prior_var = z["prior_var"]     # (2, n_lat, n_lon)
    # Assimilation never inflates variance above the background.
    assert np.all(post_var <= prior_var[None, None] + 1e-6)
