"""Tests for the ensemble square-root gains (paleoreco.assim.ensrf).

Pins both gains against explicit constructions: the mean gain against
``k P H^T (k H P H^T + R)^-1``, and the reduced gain against the defining property that
the updated deviations carry the posterior covariance ``(I - K H) P``.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.ensrf import mean_gain_apply, sqrt_gain_apply, whitened_block

B_SCALES = (0.25, 1.0, 4.0)


@pytest.fixture
def setup():
    """A full-rank ensemble, so its sample covariance is the P both gains are checked on."""
    rng = np.random.default_rng(3)
    D, m, n_mem = 12, 5, 400
    X = rng.normal(size=(n_mem, D))
    X -= X.mean(axis=0, keepdims=True)
    P = (X.T @ X) / (n_mem - 1)
    gather = np.array([0, 2, 5, 7, 11])
    r = np.array([0.3, 1.0, 2.0, 0.7, 1.5])
    return X, P, gather, r, rng.normal(size=m)


def _explicit_gain(P, gather, r, b_scale):
    S = b_scale * P[np.ix_(gather, gather)] + np.diag(r)
    return b_scale * P[:, gather] @ np.linalg.inv(S)


@pytest.mark.parametrize("b_scale", B_SCALES)
def test_mean_gain_matches_explicit_solve(setup, b_scale):
    X, P, gather, r, d = setup
    wb = whitened_block(P[:, gather], P[np.ix_(gather, gather)], r)
    assert np.allclose(mean_gain_apply(wb, b_scale, d), _explicit_gain(P, gather, r, b_scale) @ d)


@pytest.mark.parametrize("b_scale", B_SCALES)
def test_sqrt_gain_gives_the_posterior_covariance(setup, b_scale):
    """The updated deviations must have sample covariance ``(I - K H) k P``.

    This is what the reduced gain exists for: applying the plain gain K to the deviations
    instead would shrink them past the posterior, the classic ensemble spread collapse.
    """
    X, P, gather, r, _ = setup
    n_mem = len(X)
    wb = whitened_block(P[:, gather], P[np.ix_(gather, gather)], r)

    dev = np.sqrt(b_scale) * X.T                                   # (D, n_mem)
    post = dev - sqrt_gain_apply(wb, b_scale, dev[gather])
    sample = (post @ post.T) / (n_mem - 1)

    K = _explicit_gain(P, gather, r, b_scale)
    expected = b_scale * P - K @ (b_scale * P[gather])
    assert np.allclose(sample, expected, atol=1e-8)


def test_sqrt_gain_shrinks_less_than_the_plain_gain(setup):
    """Sanity on the direction: the reduced gain must remove less spread than K would."""
    X, P, gather, r, _ = setup
    wb = whitened_block(P[:, gather], P[np.ix_(gather, gather)], r)
    dev = X.T
    reduced = dev - sqrt_gain_apply(wb, 1.0, dev[gather])
    plain = dev - _explicit_gain(P, gather, r, 1.0) @ dev[gather]
    assert reduced.var(axis=1, ddof=1).sum() > plain.var(axis=1, ddof=1).sum()


def test_one_factorization_serves_every_amplitude(setup):
    """The whole sweep comes off one eigendecomposition, so each scale must still be exact."""
    X, P, gather, r, d = setup
    wb = whitened_block(P[:, gather], P[np.ix_(gather, gather)], r)
    for b_scale in (0.1, 0.5, 2.0, 10.0, 50.0):
        assert np.allclose(mean_gain_apply(wb, b_scale, d),
                           _explicit_gain(P, gather, r, b_scale) @ d)


def test_rank_deficient_covariance_is_handled(setup):
    """A k-member covariance is rank k-1 in D dimensions, which is the normal case here."""
    _, _, gather, r, d = setup
    rng = np.random.default_rng(11)
    D, k = 12, 4
    Xp = rng.normal(size=(k, D))
    Xp -= Xp.mean(axis=0, keepdims=True)
    Yp = Xp[:, gather]
    wb = whitened_block((Xp.T @ Yp) / (k - 1), (Yp.T @ Yp) / (k - 1), r)
    inc = mean_gain_apply(wb, 1.0, d)
    assert inc.shape == (D,) and np.isfinite(inc).all()
