"""Tests for masked reconstruction losses (paleoreco.losses)."""

from __future__ import annotations

import pytest
import torch

from paleoreco.training.losses import masked_mse, masked_rmse, vae_elbo_loss


def test_masked_mse_mean_counts_only_valid_terms():
    pred = torch.tensor([[[[1.0, 2.0]]]])      # (1,1,1,2)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[1.0, 0.0]])          # only the first cell counts
    # Valid squared error = 1.0 over 1 term.
    assert masked_mse(pred, target, mask) == pytest.approx(1.0)


def test_masked_mse_sum_and_none_reductions():
    pred = torch.tensor([[[[3.0, 4.0]]]])
    target = torch.zeros_like(pred)
    mask = torch.ones((1, 2))
    assert masked_mse(pred, target, mask, reduction="sum") == pytest.approx(25.0)
    none = masked_mse(pred, target, mask, reduction="none")
    assert none.shape == pred.shape
    assert none.flatten().tolist() == [9.0, 16.0]


def test_masked_mse_shape_and_reduction_guards():
    pred = torch.zeros((1, 1, 1, 2))
    with pytest.raises(ValueError):
        masked_mse(pred, torch.zeros((1, 1, 1, 3)), torch.ones((1, 2)))
    with pytest.raises(ValueError):
        masked_mse(pred, torch.zeros_like(pred), torch.ones((1, 2)), reduction="bogus")


def test_masked_rmse_is_sqrt_of_mse():
    pred = torch.tensor([[[[3.0, 4.0]]]])
    target = torch.zeros_like(pred)
    mask = torch.ones((1, 2))
    mse = masked_mse(pred, target, mask)
    assert masked_rmse(pred, target, mask) == pytest.approx(float(mse.sqrt()))


def test_vae_elbo_kl_is_zero_at_standard_normal_posterior():
    x_hat = torch.zeros((2, 2, 4, 4))
    target = torch.zeros_like(x_hat)
    mask = torch.ones((4, 4))
    mu = torch.zeros((2, 3))
    logvar = torch.zeros((2, 3))     # sigma^2 = 1 -> KL = 0
    loss, recon, kl = vae_elbo_loss(x_hat, target, mu, logvar, mask, beta=1.0)
    assert kl == pytest.approx(0.0)
    assert recon == pytest.approx(0.0)
    assert loss == pytest.approx(0.0)


def test_vae_elbo_kl_matches_closed_form():
    x_hat = torch.zeros((1, 2, 4, 4))
    target = torch.zeros_like(x_hat)
    mask = torch.ones((4, 4))
    mu = torch.ones((1, 3))          # mu^2 = 1
    logvar = torch.zeros((1, 3))     # sigma^2 = 1, log = 0
    # KL per dim = 0.5*(1 + 1 - 0 - 1) = 0.5; summed over 3 dims = 1.5.
    _, _, kl = vae_elbo_loss(x_hat, target, mu, logvar, mask, beta=2.0)
    assert kl == pytest.approx(1.5)
