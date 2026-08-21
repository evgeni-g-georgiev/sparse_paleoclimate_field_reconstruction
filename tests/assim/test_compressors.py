"""Compressors round-trip and expose a usable latent prior.

PCA must invert exactly on its own basis and agree between the numpy and torch
decoders; the neural wrappers must produce codes and a differentiable decode. The
latent prior must be a symmetric PSD covariance with the climatological mean at the
centre of the code cloud (zero for PCA).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paleoreco.assim.compressors import (
    AECompressor, PCACompressor, VAECompressor, latent_prior,
)
from paleoreco.data.cube import apply_anomaly, compute_zscore_stats
from paleoreco.eval.shared import pod_fit
from paleoreco.models.autoencoder import ConvAE, ConvBetaVAE


def _anom(cube, valid):
    stats = compute_zscore_stats(cube, np.arange(len(cube)), valid)
    return apply_anomaly(cube, stats), stats["safe_valid"]


def test_pca_roundtrips_and_torch_matches_numpy(cube, valid):
    cube_anom, safe = _anom(cube, valid)
    pca = PCACompressor.from_pod_fit(pod_fit(cube_anom, np.arange(len(cube)), safe, max_k=6), k=6)
    z = pca.encode(cube_anom)
    assert z.shape == (len(cube), 6)
    rec = pca.decode(z)
    # Encoding a decoded code returns it (orthonormal basis is a projection;
    # tolerance reflects the randomised SVD's orthonormality, not the algebra).
    assert np.allclose(pca.encode(rec), z, atol=1e-5)
    # The torch decoder agrees with the numpy decoder.
    rec_t = pca.decode_torch(torch.as_tensor(z)).numpy()
    assert np.allclose(rec, rec_t, atol=1e-9)


def test_latent_prior_is_psd_with_zero_centroid_for_pca(cube, valid):
    cube_anom, safe = _anom(cube, valid)
    pca = PCACompressor.from_pod_fit(pod_fit(cube_anom, np.arange(len(cube)), safe, max_k=6), k=6)
    B_z, z_clim = latent_prior(pca, cube_anom, np.arange(len(cube)))
    assert B_z.shape == (6, 6)
    assert np.allclose(B_z, B_z.T)
    assert np.linalg.eigvalsh(B_z).min() > -1e-9
    assert np.allclose(z_clim, 0.0, atol=1e-5)


def test_latent_prior_highpass_moves_bz_but_not_z_clim(cube, valid, ages):
    """The latent band-pass filters the codes, so the tangent point stays put."""
    cube_anom, safe = _anom(cube, valid)
    idx = np.arange(len(cube))
    pca = PCACompressor.from_pod_fit(pod_fit(cube_anom, idx, safe, max_k=6), k=6)
    raw_B, raw_z = latent_prior(pca, cube_anom, idx)
    hp_B, hp_z = latent_prior(pca, cube_anom, idx, ages=ages, highpass_window=1500.0)

    assert not np.allclose(raw_B, hp_B)
    assert np.trace(hp_B) < np.trace(raw_B)
    assert np.allclose(raw_z, hp_z)
    assert np.linalg.eigvalsh(hp_B).min() > -1e-9


def test_latent_prior_highpass_requires_ages(cube, valid):
    cube_anom, safe = _anom(cube, valid)
    idx = np.arange(len(cube))
    pca = PCACompressor.from_pod_fit(pod_fit(cube_anom, idx, safe, max_k=6), k=6)
    with pytest.raises(ValueError, match="ages"):
        latent_prior(pca, cube_anom, idx, highpass_window=1500.0)


def test_neural_compressors_encode_decode_and_grad(cube, valid):
    cube_anom, safe = _anom(cube, valid)
    H, W = cube.shape[2:]
    for cls, net in (
        (AECompressor, ConvAE(latent_dim=4, base_channels=8, depth=2, grid_shape=(H, W))),
        (VAECompressor, ConvBetaVAE(latent_dim=4, base_channels=8, depth=2, grid_shape=(H, W))),
    ):
        comp = cls(net, safe, (2, H, W))
        z = comp.encode(cube_anom)
        assert z.shape == (len(cube), 4)
        assert comp.decode(z).shape == (len(cube), 2, H, W)
        zt = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        comp.decode_torch(zt).sum().backward()
        assert zt.grad is not None and torch.isfinite(zt.grad).all()
