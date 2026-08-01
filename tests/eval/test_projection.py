"""Tests for the PC projections and their sign convention (paleoreco.eval.projection)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.eval.projection import orient_pcs, pc_pattern_correlation, pca_scores


def _stack(seed: int = 0, n: int = 40):
    """A (n, 2, 5, 6) field stack with structure the leading components can find."""
    rng = np.random.default_rng(seed)
    pattern = np.linspace(-1.0, 1.0, 2 * 5 * 6).reshape(2, 5, 6)
    amplitude = rng.normal(size=n)
    return amplitude[:, None, None, None] * pattern + 0.1 * rng.normal(size=(n, 2, 5, 6))


def test_pca_scores_returns_components_in_the_field_shape():
    res = pca_scores(_stack(), 3)
    assert res["scores"].shape == (40, 3)
    assert res["components"].shape == (3, 2, 5, 6)
    assert res["mean_field"].shape == (2, 5, 6)


def test_orient_pcs_gives_components_a_positive_spatial_mean():
    res = orient_pcs(pca_scores(_stack(), 3))
    means = res["components"].reshape(3, -1).mean(axis=1)
    assert (means > 0).all()
    assert res["flipped"].shape == (3,)


def test_orient_pcs_flips_scores_with_their_components():
    raw = pca_scores(_stack(), 3)
    oriented = orient_pcs(raw)
    # A sign convention must not change the rank-k reconstruction it describes.
    for k in range(3):
        assert np.allclose(raw["scores"][:, k, None, None, None] * raw["components"][k],
                           oriented["scores"][:, k, None, None, None] * oriented["components"][k])
    assert np.allclose(raw["explained_variance_ratio"], oriented["explained_variance_ratio"])


def test_orient_pcs_is_idempotent():
    once = orient_pcs(pca_scores(_stack(), 3))
    twice = orient_pcs(once)
    assert np.allclose(once["scores"], twice["scores"])
    assert not twice["flipped"].any()


def test_pc_pattern_correlation_is_one_against_itself():
    res = orient_pcs(pca_scores(_stack(), 3))
    assert np.allclose(pc_pattern_correlation(res, res, 3), 1.0)


def test_pc_pattern_correlation_rejects_mismatched_grids():
    a = pca_scores(_stack(), 2)
    b = pca_scores(_stack()[:, :, :4], 2)
    with pytest.raises(ValueError, match="different grids"):
        pc_pattern_correlation(a, b, 2)
