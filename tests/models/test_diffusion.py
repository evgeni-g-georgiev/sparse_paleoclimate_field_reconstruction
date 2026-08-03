"""Denoiser architecture and EDM preconditioning contracts."""

from __future__ import annotations

import pytest
import torch

from paleoreco.models.diffusion import CircularUNet, EDMDenoiser, load_denoiser


def _denoiser(grid=(32, 64)):
    return EDMDenoiser(CircularUNet(base_channels=16, depth=2, grid_shape=grid))


def test_unet_forward_shape_default_and_small_grid():
    for grid in ((32, 64), (12, 12)):
        net = CircularUNet(base_channels=16, depth=2, grid_shape=grid)
        x = torch.randn(3, 2, *grid)
        out = net(x, torch.zeros(3))
        assert out.shape == x.shape


def test_preconditioning_recovers_input_at_zero_noise():
    # c_skip -> 1 and c_out -> 0 as sigma -> 0, so D(x; sigma) -> x.
    d = _denoiser()
    x = torch.randn(4, 2, 32, 64)
    out = d(x, torch.full((4,), 1e-4))
    assert torch.allclose(out, x, atol=1e-2)


def test_score_is_tweedie_of_denoiser():
    d = _denoiser()
    x = torch.randn(2, 2, 32, 64)
    sigma = torch.tensor([0.5, 1.3])
    expected = (d(x, sigma) - x) / sigma[:, None, None, None] ** 2
    assert torch.allclose(d.score(x, sigma), expected, atol=1e-5)


def test_sigma_accepts_scalar_and_per_sample():
    d = _denoiser()
    x = torch.randn(4, 2, 32, 64)
    assert d(x, torch.tensor(0.7)).shape == x.shape
    assert d(x, torch.rand(4) + 0.1).shape == x.shape


def test_load_denoiser_rejects_a_checkpoint_without_scales(tmp_path):
    """Weights alone do not define the frame they were trained in, so a checkpoint
    that predates the per-cell field must fail loudly rather than load."""
    d = _denoiser(grid=(12, 12))
    path = tmp_path / "diffusion.pt"
    torch.save({"state_dict": d.state_dict(), "config": d.config,
                "sigma_data": d.sigma_data}, path)
    with pytest.raises(KeyError, match="per-cell"):
        load_denoiser(str(path))


def test_load_denoiser_returns_the_stored_scale_field(tmp_path):
    d = _denoiser(grid=(12, 12))
    scales = [[[0.5] * 12] * 12] * 2
    path = tmp_path / "diffusion.pt"
    torch.save({"state_dict": d.state_dict(), "config": d.config,
                "sigma_data": d.sigma_data, "scales": scales}, path)
    net, loaded = load_denoiser(str(path))
    assert loaded == scales
    assert net.config == d.config
