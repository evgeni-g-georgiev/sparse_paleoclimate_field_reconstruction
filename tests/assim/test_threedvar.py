"""Tests for the gain-form 3DVar analysis (paleoreco.assim.threedvar).

Pins the amplitude sweep against an explicit construction of
x_a = x_b + k B H^T (k H B H^T + R)^-1 (y - H x_b) and A = (I - K H) k B.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.threedvar import ThreeDVar

B_SCALES = np.array([0.25, 1.0, 5.0])


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
    return shape, B, gather, sse, x_b, y


def _explicit(B, gather, sse, x_b, y, k):
    B_obs = k * B[:, gather]
    Sinv = np.linalg.inv(k * B[np.ix_(gather, gather)] + np.diag(sse))
    x_a = x_b + B_obs @ Sinv @ (y - x_b[gather])
    return x_a, np.diag(k * B - B_obs @ Sinv @ B_obs.T)


def test_every_amplitude_matches_the_explicit_gain(setup):
    shape, B, gather, sse, x_b, y = setup
    tv = ThreeDVar(B, shape)
    sweep = tv.apply_sweep(tv.prepare_sweep(gather, sse, B_SCALES), y, x_b)
    for k, res in zip(B_SCALES, sweep):
        x_a, post_var = _explicit(B, gather, sse, x_b, y, k)
        assert np.allclose(res.mean_anom.ravel(), x_a)
        assert np.allclose(res.posterior_var.ravel(), post_var)


def test_posterior_var_does_not_exceed_prior(setup):
    shape, B, gather, sse, _, _ = setup
    tv = ThreeDVar(B, shape)
    post_var = tv.post_var_sweep(tv.prepare_sweep(gather, sse, B_SCALES))
    for k, pv in zip(B_SCALES, post_var):
        # Assimilation can only reduce variance, and strictly so at the observed cells.
        assert np.all(pv <= k * np.diag(B) + 1e-9)
        assert np.all(pv[gather] < k * np.diag(B)[gather])


def test_the_mean_only_analysis_gives_the_full_analysis_mean(setup):
    """A grid scan reads the mean alone, so skipping the spread must not move it at all."""
    shape, B, gather, sse, x_b, y = setup
    tv = ThreeDVar(B, shape)
    gain = tv.prepare_sweep(gather, sse, B_SCALES)
    full = tv.apply_sweep(gain, y, x_b)
    lean = tv.apply_sweep(gain, y, x_b, mean_only=True)
    for a, b in zip(full, lean):
        assert np.array_equal(a.mean_anom, b.mean_anom)
        assert b.posterior_var is None
