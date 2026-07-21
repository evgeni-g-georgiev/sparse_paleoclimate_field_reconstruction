"""Latent 3DVar is correct against the pixel closed form.

A PCA decoder is affine, so :class:`LinearLatentVar` must reproduce a pixel
:class:`ThreeDVar` whose covariance is the decoded latent covariance ``P B_z P^T``
(the EOF-equivalence in miniature). :class:`TangentLinearLatentVar`, whose pushforward is
the autograd Jacobian, must match the exact affine gain on that same decoder. Sample-based
posterior variance must match the linearised diagonal where the decoder is affine, and the
re-centring must map the climatological background to the exact zero field.
"""

from __future__ import annotations

import numpy as np

from paleoreco.assim.compressors import PCACompressor, latent_prior
from paleoreco.assim.latent import LinearLatentVar, TangentLinearLatentVar
from paleoreco.assim.method import Observations
from paleoreco.assim.threedvar import ThreeDVar
from paleoreco.data.cube import apply_anomaly, compute_zscore_stats
from paleoreco.eval.shared import pod_fit


def _setup(cube, valid, k=6):
    stats = compute_zscore_stats(cube, np.arange(len(cube)), valid)
    cube_anom = apply_anomaly(cube, stats)
    safe = stats["safe_valid"]
    shape = (2, cube.shape[2], cube.shape[3])
    pca = PCACompressor.from_pod_fit(pod_fit(cube_anom, np.arange(len(cube)), safe, max_k=k), k=k)
    B_z, z_clim = latent_prior(pca, cube_anom, np.arange(len(cube)))
    return pca, B_z, z_clim, shape, safe


def _network(safe, shape, rng, m=8):
    safe_flat = np.broadcast_to(safe, shape).ravel()
    g = rng.choice(np.flatnonzero(safe_flat), size=m, replace=False)
    return Observations(gather=g, y_anom=rng.standard_normal(m),
                        sse=rng.uniform(0.3, 1.0, m)), np.zeros(int(np.prod(shape)))


def test_linear_latent_equals_pixel_threedvar(cube, valid):
    pca, B_z, z_clim, shape, safe = _setup(cube, valid)
    lv = LinearLatentVar(pca, B_z, z_clim, shape, safe)
    tv = ThreeDVar(lv.P @ B_z @ lv.P.T, shape)                       # rank-k EOF pixel B
    obs, zero = _network(safe, shape, np.random.default_rng(0))

    # The two are algebraically equal; the gap is the eigh vs Cholesky route on a
    # rank-deficient (rank-k in D) B, so the tolerance is numerical, not structural.
    r_lat, r_pix = lv.analyze(obs, zero), tv.analyze(obs, zero)
    assert np.allclose(r_lat.mean_anom[..., safe], r_pix.mean_anom[..., safe], atol=1e-4)
    assert np.allclose(r_lat.posterior_var[..., safe], r_pix.posterior_var[..., safe], atol=1e-5)


def test_tangent_linear_equals_linear_on_affine_decoder(cube, valid):
    pca, B_z, z_clim, shape, safe = _setup(cube, valid)
    lv = LinearLatentVar(pca, B_z, z_clim, shape, safe)
    tl = TangentLinearLatentVar(pca, B_z, z_clim, shape, safe, device="cpu")
    # The autograd Jacobian about z_clim is the exact affine pushforward, so the one-shot
    # gain matches the closed form (no iteration, unlike a variational solve).
    assert np.allclose(lv.P, tl.P, atol=1e-4)
    obs, zero = _network(safe, shape, np.random.default_rng(1))
    r_lin, r_tl = lv.analyze(obs, zero), tl.analyze(obs, zero)
    assert np.allclose(r_lin.mean_anom[..., safe], r_tl.mean_anom[..., safe], atol=1e-4)
    assert np.allclose(r_lin.posterior_var[..., safe], r_tl.posterior_var[..., safe], atol=1e-4)


def test_recentring_zeroes_climatological_background(cube, valid):
    pca, _B_z, _z_clim, shape, safe = _setup(cube, valid)
    d = pca.latent_dim
    mu = np.full_like(pca.mu, 0.5)                                   # nonzero decoded climatology
    pca_off = PCACompressor(pca.V_k, mu, pca.keep, shape)
    lv = LinearLatentVar(pca_off, np.eye(d), np.zeros(d), shape, safe)

    clim_field = lv._decode_field(np.zeros(d))
    assert np.abs(clim_field[..., safe]).max() > 0.1                 # the offset is real
    # A zero-innovation climatological analysis (obs equal the decoded background) must
    # return the exact zero field once decode(z_clim) is subtracted.
    safe_flat = np.broadcast_to(safe, shape).ravel()
    g = np.flatnonzero(safe_flat)[:8]
    obs = Observations(gather=g, y_anom=clim_field.ravel()[g], sse=np.full(8, 0.5))
    res = lv.analyze(obs, np.zeros(int(np.prod(shape))))
    assert np.abs(res.mean_anom[..., safe]).max() < 1e-6


def test_posterior_var_within_prior_and_sweep_matches_analyze(cube, valid):
    pca, B_z, z_clim, shape, safe = _setup(cube, valid)
    lv = LinearLatentVar(pca, B_z, z_clim, shape, safe)
    obs, zero = _network(safe, shape, np.random.default_rng(3))

    res = lv.analyze(obs, zero)
    assert res.mean_anom.shape == shape
    # predict_obs reads the decoded field at the observed cells.
    assert np.allclose(res.predict_obs(obs.gather), res.mean_anom.ravel()[obs.gather])
    # Assimilation never inflates variance above the prior.
    assert np.all(res.posterior_var[..., safe] <= lv.diagB.reshape(shape)[..., safe] + 1e-9)

    gain = lv.prepare_sweep(obs.gather, obs.sse, np.array([0.5, 1.0, 2.0]))
    sweep = lv.apply_sweep(gain, obs.y_anom, zero)
    assert np.allclose(sweep[1].mean_anom, res.mean_anom, atol=1e-9)
