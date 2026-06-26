"""Tests for the gain-form 3DVar analysis (paleoreco.assim.threedvar).

Pins the closed-form algebra against an explicit construction of
x_a = x_b + B H^T (H B H^T + R)^-1 (y - H x_b) and A = (I - K H) B.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.threedvar import ThreeDVar


@pytest.fixture
def setup():
    rng = np.random.default_rng(7)
    shape = (2, 2, 2)            # D = 8
    D = int(np.prod(shape))
    M = rng.normal(size=(D, D))
    B = M @ M.T + np.eye(D)      # SPD
    gather = np.array([0, 3, 5])
    sse = np.array([0.5, 1.0, 2.0])
    x_b = rng.normal(size=D)
    y = rng.normal(size=len(gather))
    return shape, D, B, gather, sse, x_b, y


def _explicit(B, gather, sse, x_b, y):
    B_obs = B[:, gather]
    S = B[np.ix_(gather, gather)] + np.diag(sse)
    Sinv = np.linalg.inv(S)
    d = y - x_b[gather]
    x_a = x_b + B_obs @ Sinv @ d
    A = B - B_obs @ Sinv @ B_obs.T
    return x_a, np.diag(A)


def test_analysis_matches_explicit_gain(setup):
    shape, D, B, gather, sse, x_b, y = setup
    tv = ThreeDVar(B, shape)
    from paleoreco.assim.method import Observations

    res = tv.analyze(Observations(gather=gather, y_anom=y, sse=sse), x_b)
    x_a_exp, pv_exp = _explicit(B, gather, sse, x_b, y)

    assert np.allclose(res.mean_anom.ravel(), x_a_exp)
    assert np.allclose(res.posterior_var.ravel(), pv_exp)


def test_posterior_var_does_not_exceed_prior(setup):
    shape, D, B, gather, sse, x_b, y = setup
    tv = ThreeDVar(B, shape)
    gain = tv.prepare(gather, sse)
    # Assimilation can only reduce variance: diag A <= diag B everywhere.
    assert np.all(gain.post_var <= np.diag(B) + 1e-9)
    # Observed cells are strictly reduced.
    assert np.all(gain.post_var[gather] < np.diag(B)[gather])


def test_analyze_many_equals_repeated_analyze(setup):
    shape, D, B, gather, sse, x_b, y = setup
    tv = ThreeDVar(B, shape)
    from paleoreco.assim.method import Observations

    obs = Observations(gather=gather, y_anom=y, sse=sse)
    rng = np.random.default_rng(11)
    bgs = [rng.normal(size=D) for _ in range(3)]
    many = tv.analyze_many(obs, bgs)
    for bg, res in zip(bgs, many):
        single = tv.analyze(obs, bg)
        assert np.allclose(res.mean_anom, single.mean_anom)


def test_sweep_at_unit_scale_matches_plain_analysis(setup):
    shape, D, B, gather, sse, x_b, y = setup
    tv = ThreeDVar(B, shape)
    from paleoreco.assim.method import Observations

    b_scales = np.array([0.5, 1.0, 2.0])
    sweep_gain = tv.prepare_sweep(gather, sse, b_scales)
    sweep = tv.apply_sweep(sweep_gain, y, x_b)
    post_var = tv.post_var_sweep(sweep_gain)

    plain = tv.analyze(Observations(gather=gather, y_anom=y, sse=sse), x_b)
    unit = list(b_scales).index(1.0)
    assert np.allclose(sweep[unit].mean_anom, plain.mean_anom)
    assert np.allclose(post_var[unit], plain.posterior_var.ravel())
    assert np.allclose(sweep[unit].posterior_var.ravel(), plain.posterior_var.ravel())
