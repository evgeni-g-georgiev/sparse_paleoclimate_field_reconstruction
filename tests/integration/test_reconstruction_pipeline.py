"""End-to-end checks on the reconstruction driver.

The lane drivers are guarded by their metrics; this one writes fields nothing scores, so the
guards have to be structural: every age is present, an age with no network says so rather
than passing off a climatology as a reconstruction, and the analog step still sees the age
it is reconstructing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from paleoreco.assim import experiments as ex
from paleoreco.assim import reconstruction as rc
from paleoreco.assim.analog import ANALOG_MISFIT
from paleoreco.assim.hgaoenkf import make_hgaoenkf
from paleoreco.assim.observations import TEMPORAL_DEFLATE, TEMPORAL_OFF

K = 4
B_SCALES = (0.5, 2.0)
THIN, EMPTY = 5, 3          # ages given a below-floor network and no network at all


@pytest.fixture
def obs(obs_long, ages):
    """The shared network with two ages made unusable, and a climatology per site.

    Constant ``y`` would leave every innovation zero, so the values vary by site and age.
    """
    df = obs_long.copy()
    chan = df["channel"].map({"mtco": 0, "mtwa": 1}).to_numpy()
    df["y"] = df["site"].to_numpy() * 2.0 + chan + np.sin(df["age"].to_numpy() / 700.0)
    df["my"] = df.groupby(["site", "channel"])["y"].transform("mean")
    df = df[df["age"] != ages[EMPTY]]
    thin = (df["age"] == ages[THIN]) & (df["site"] != df["site"].iloc[0])
    return df[~thin].reset_index(drop=True)


def _run(tmp_path, cube, ages, lats, lons, valid, obs, **kw):
    kw.setdefault("b_scales", B_SCALES)
    kw.setdefault("min_obs", 3)
    return rc.run_reconstruction(cube, ages, lats, lons, valid, obs, str(tmp_path), **kw)


def _analog(cube, ages, lats, lons, **kw):
    return make_hgaoenkf(cube, ages, lats, lons, k=K, hybrid_w=1.0,
                         selection=ANALOG_MISFIT, **kw)


def _fields(tmp_path):
    return np.load(tmp_path / "reconstruction_fields.npz")


def test_the_product_covers_every_age(tmp_path, cube, ages, lats, lons, valid, obs):
    cfg = _run(tmp_path, cube, ages, lats, lons, valid, obs)
    z = _fields(tmp_path)

    assert os.path.exists(tmp_path / "reconstruction_config.json")
    assert not os.path.exists(tmp_path / "metrics.csv")   # nothing here is scored
    assert np.array_equal(z["ages"], np.asarray(ages, dtype=np.int64))
    assert z["mean_anom"].shape == (len(B_SCALES), len(ages), 2, len(lats), len(lons))
    assert z["post_var"].shape == z["mean_anom"].shape
    assert z["prior_mean_anom"].shape == z["mean_anom"].shape[1:]
    assert cfg["n_ages"] == len(ages)
    assert np.isfinite(z["mean_anom"]).all() and (z["post_var"] >= 0).all()


def test_an_age_with_no_network_carries_the_prior_and_says_so(
        tmp_path, cube, ages, lats, lons, valid, obs):
    """A climatology reported as a reconstruction is the failure this driver invites.

    An empty network factorizes without complaint, so an unflagged age would come back as a
    plausible field built on whichever candidates the ranking happened to leave first.
    """
    _run(tmp_path, cube, ages, lats, lons, valid, obs,
         make_method=_analog(cube, ages, lats, lons))
    z = _fields(tmp_path)

    assert set(np.flatnonzero(z["prior_only"])) == {EMPTY, THIN}
    assert z["n_obs"][EMPTY] == 0
    assert (z["mean_anom"][:, EMPTY] == 0).all()
    assert (z["analog_index"][[EMPTY, THIN]] == rc.NO_ANALOG).all()
    assert (z["analog_index"][0] >= 0).all()
    for bj, b in enumerate(z["b_scales"]):
        assert np.allclose(z["post_var"][bj, EMPTY], b * z["prior_var"])
    assert np.isnan(z["desroziers_r"][:, EMPTY]).all()
    assert np.isfinite(z["desroziers_r"][:, 0]).all()


def test_the_exclusion_band_reaches_the_estimator(
        tmp_path, cube, ages, lats, lons, valid, obs):
    """Dropping the age from the gain leaves the run finite and looking better, not worse.

    Without it the analog step selects the archive's own state at the target age, which
    degenerates the method into a per-age background; no field or metric would show it.
    """
    exclude = 1500.0
    _run(tmp_path, cube, ages, lats, lons, valid, obs,
         make_method=_analog(cube, ages, lats, lons, exclude_yr=exclude))
    z = _fields(tmp_path)

    drawn = z["analog_index"]
    for i in np.flatnonzero(~z["prior_only"]):
        assert (np.abs(z["ages"][drawn[i]] - z["ages"][i]) >= exclude).all()


def test_the_prior_field_is_the_analysis_with_the_gain_switched_off(
        tmp_path, cube, ages, lats, lons, valid, obs):
    """The prior column rides in the same sweep, so it must be that ensemble's own mean."""
    _run(tmp_path, cube, ages, lats, lons, valid, obs,
         make_method=_analog(cube, ages, lats, lons))
    z = _fields(tmp_path)

    clim = cube.reshape(len(ages), -1).astype(np.float64).mean(axis=0)
    pool = cube.reshape(len(ages), -1).astype(np.float64) - clim
    for i in np.flatnonzero(~z["prior_only"]):
        want = pool[z["analog_index"][i]].mean(axis=0).reshape(z["prior_mean_anom"].shape[1:])
        assert np.allclose(z["prior_mean_anom"][i], want, atol=1e-5)
    assert not np.allclose(z["prior_mean_anom"], z["mean_anom"][-1])


def test_a_larger_amplitude_pulls_the_analysis_towards_the_observations(
        tmp_path, cube, ages, lats, lons, valid, obs):
    """Guards that the diagnostics read the analysis and the prior, not one field twice.

    The whitened innovation is the norm the gain shrinks, one eigendirection at a time, so
    it falls with the amplitude at every age. The unweighted residual does not once R varies
    between observations, which is why neither column carries it.
    """
    _run(tmp_path, cube, ages, lats, lons, valid, obs,
         make_method=_analog(cube, ages, lats, lons))
    z = _fields(tmp_path)

    live = ~z["prior_only"]
    assert (z["chi2_an"][1, live] < z["chi2_an"][0, live]).all()
    assert (z["chi2_an"][:, live] <= z["chi2_bg"][live] + 1e-9).all()


def test_the_cross_variance_rides_along_only_where_it_is_produced(
        tmp_path, cube, ages, lats, lons, valid, obs):
    analog = tmp_path / "analog"
    static = tmp_path / "static"
    _run(analog, cube, ages, lats, lons, valid, obs,
         make_method=_analog(cube, ages, lats, lons))
    _run(static, cube, ages, lats, lons, valid, obs)

    assert "post_cross_var" in _fields(analog).files
    assert "analog_index" in _fields(analog).files
    assert "post_cross_var" not in _fields(static).files
    assert "analog_index" not in _fields(static).files


def test_the_staleness_treatment_reaches_the_observation_pair(
        tmp_path, cube, ages, lats, lons, valid, obs):
    """Deflating rescales both the observation and R, so neither field may sit unchanged."""
    off, deflate = tmp_path / "off", tmp_path / "deflate"
    common = dict(make_method=_analog(cube, ages, lats, lons))
    _run(off, cube, ages, lats, lons, valid, obs, temporal_mode=TEMPORAL_OFF, **common)
    _run(deflate, cube, ages, lats, lons, valid, obs, temporal_mode=TEMPORAL_DEFLATE,
         **common)

    assert not np.allclose(_fields(off)["mean_anom"], _fields(deflate)["mean_anom"])


def test_the_run_is_reproducible(tmp_path, cube, ages, lats, lons, valid, obs):
    """No noise is drawn anywhere, so two runs must agree exactly rather than closely."""
    a, b = tmp_path / "a", tmp_path / "b"
    common = dict(make_method=_analog(cube, ages, lats, lons))
    _run(a, cube, ages, lats, lons, valid, obs, **common)
    _run(b, cube, ages, lats, lons, valid, obs, **common)

    first, second = _fields(a), _fields(b)
    for key in first.files:
        assert np.array_equal(first[key], second[key], equal_nan=True), key


def test_the_config_records_what_the_fields_cannot(
        tmp_path, cube, ages, lats, lons, valid, obs):
    """Which ages fell back on the prior, and how the observations were treated."""
    cfg = _run(tmp_path, cube, ages, lats, lons, valid, obs,
               make_method=_analog(cube, ages, lats, lons),
               estimator=ex.ESTIMATOR_HGAOENKF,
               method_cols=ex.analog_cols(K, 1.0))
    on_disk = json.load(open(tmp_path / "reconstruction_config.json"))

    assert on_disk == cfg                                  # every value is JSON native
    assert on_disk["n_prior_only_ages"] == 2
    assert on_disk["prior_only_ages"] == [int(ages[EMPTY]), int(ages[THIN])]
    assert on_disk["background"] == "climatological"
    assert on_disk["temporal_mode"] == TEMPORAL_DEFLATE
    assert on_disk["estimator"] == ex.ESTIMATOR_HGAOENKF
    assert on_disk["b_scales"] == list(B_SCALES)
    assert set(on_disk["rep_var_full"]) == {"mtco", "mtwa"}
    assert on_disk["analog_k"] == K


def test_an_impossible_minimum_is_rejected(tmp_path, cube, ages, lats, lons, valid, obs):
    with pytest.raises(ValueError):
        _run(tmp_path, cube, ages, lats, lons, valid, obs, min_obs=0)
