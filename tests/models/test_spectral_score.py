"""Tests for the subspace score network (paleoreco.models.spectral_score).

The load-bearing property is the Gaussian anchor: a zero output layer must leave
the EDM-preconditioned denoiser exactly equal to the denoiser of
``N(0, sigma_data^2 I)``, so an untrained model reproduces the Gaussian prior the
subspace is whitened against and training can only add departures from it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paleoreco.models.diffusion import SIGMA_DATA
from paleoreco.models.spectral_score import (
    ScoreMLP,
    build_spectral_denoiser,
    load_spectral_denoiser,
)


def test_zero_init_denoiser_is_the_exact_gaussian_denoiser():
    """``D(x; sigma) = c_skip x`` is the MMSE denoiser of ``N(0, sigma_data^2 I)``.

    This is the anchor the whole method depends on, so it is pinned exactly rather
    than to a tolerance that could hide a non-zero initialisation.
    """
    net = build_spectral_denoiser(dim=5, width=32, depth=2)
    x = torch.randn(7, 5, 1, 1)
    for sigma in (0.01, 0.5, 3.0, 80.0):
        c_skip = SIGMA_DATA ** 2 / (sigma ** 2 + SIGMA_DATA ** 2)
        assert torch.allclose(net(x, sigma), c_skip * x, atol=1e-6)


def test_zero_init_score_matches_the_gaussian_score():
    """Tweedie on the anchored denoiser must give ``-x / (sigma^2 + sigma_data^2)``."""
    net = build_spectral_denoiser(dim=4, width=16, depth=1)
    x = torch.randn(3, 4, 1, 1)
    for sigma in (0.05, 1.0, 10.0):
        want = -x / (sigma ** 2 + SIGMA_DATA ** 2)
        assert torch.allclose(net.score(x, torch.tensor(sigma)), want, atol=1e-6)


def test_forward_preserves_shape_and_accepts_flat_input():
    net = ScoreMLP(dim=6, width=16, depth=2, emb_dim=8)
    c_noise = torch.zeros(2)
    assert net(torch.randn(2, 6, 1, 1), c_noise).shape == (2, 6, 1, 1)
    assert net(torch.randn(2, 6), c_noise).shape == (2, 6)


def test_forward_rejects_a_mismatched_dimension():
    net = ScoreMLP(dim=6, width=16, depth=2, emb_dim=8)
    with pytest.raises(ValueError, match="expected 6 values"):
        net(torch.randn(2, 5, 1, 1), torch.zeros(2))


def test_trained_weights_move_the_output_off_the_anchor():
    """A non-zero output layer must actually change the denoiser, or the anchor
    test above would pass for a network that can never learn anything."""
    net = build_spectral_denoiser(dim=4, width=16, depth=1)
    torch.nn.init.normal_(net.net.out.weight, std=0.5)
    x = torch.randn(3, 4, 1, 1)
    c_skip = SIGMA_DATA ** 2 / (1.0 + SIGMA_DATA ** 2)
    assert not torch.allclose(net(x, 1.0), c_skip * x, atol=1e-4)


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"dim": 3, "depth": 0},
                                    {"dim": 3, "emb_dim": 7}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ScoreMLP(**kwargs)


def test_checkpoint_round_trip(tmp_path):
    net = build_spectral_denoiser(dim=5, width=32, depth=2, emb_dim=16)
    torch.nn.init.normal_(net.net.out.weight, std=0.1)
    path = tmp_path / "spectral.pt"
    torch.save({"state_dict": net.state_dict(), "config": net.config,
                "sigma_data": net.sigma_data, "spectral_k": 5}, path)

    back, ckpt = load_spectral_denoiser(str(path))
    assert ckpt["spectral_k"] == 5
    assert back.config == net.config
    x = torch.randn(4, 5, 1, 1)
    assert np.allclose(back(x, 0.7).detach().numpy(), net(x, 0.7).detach().numpy(),
                       atol=1e-6)
