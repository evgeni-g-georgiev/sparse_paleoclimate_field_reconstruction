"""The method-factory interface must not move the pixel-3DVar numbers.

``run_ppe`` takes a ``make_method`` factory so latent methods (and a future generative
prior) plug in behind one call. With the factory left at its default (None) the runner
builds a pixel ``ThreeDVar``; passing that same construction explicitly must reproduce
the default run row for row, and the ``space`` tag must read ``pixel``. This guards the
method-agnostic contract the comparison notebook relies on.
"""

from __future__ import annotations

import pandas as pd

from paleoreco.assim import experiments as ex
from paleoreco.assim.threedvar import ThreeDVar


def test_make_method_default_matches_explicit_threedvar(
    tmp_path, cube, ages, lats, lons, valid, obs_long
):
    common = dict(n_shapes=3, n_select=2, n_noise=2, b_scales=(0.5, 1.0),
                  truth_stride=1, seed=0)
    df_default = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                            str(tmp_path / "default"), **common)
    df_factory = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                            str(tmp_path / "factory"),
                            make_method=lambda prior, shape: ThreeDVar(prior.B, shape),
                            **common)

    pd.testing.assert_frame_equal(df_default, df_factory)
    assert (df_default["space"] == "pixel").all()


def test_space_label_only_relabels(tmp_path, cube, ages, lats, lons, valid, obs_long):
    common = dict(n_shapes=3, n_select=2, n_noise=2, b_scales=(1.0,),
                  truth_stride=1, seed=0)
    base = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                      str(tmp_path / "base"), **common)
    relabel = ex.run_ppe(cube, ages, lats, lons, valid, obs_long,
                         str(tmp_path / "relabel"), space="pca", **common)

    # The space tag changes; the assimilated values do not.
    threedvar = base["method"] == "3dvar"
    pd.testing.assert_series_equal(
        base.loc[threedvar, "value"].reset_index(drop=True),
        relabel.loc[relabel["method"] == "3dvar", "value"].reset_index(drop=True),
    )
    assert (relabel.loc[relabel["method"] == "3dvar", "space"] == "pca").all()
