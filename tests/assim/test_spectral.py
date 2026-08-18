"""Tests for the subspace score prior (paleoreco.assim.spectral).

The decisive property is **exact nesting**: because the leading block is whitened
against B's own eigenvalues and the complement is kept Gaussian and marginalised
analytically, a network output of zero must reproduce the 3DVar analysis at the
same background amplitude. If that holds, every Gaussian part of the method is
right and anything a trained network changes is attributable to the non-Gaussian
structure it learned.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paleoreco.assim.method import Observations
from paleoreco.assim.spectral import (
    SpectralSampler,
    build_spectral_basis,
    match_subspace_covariance,
)
from paleoreco.assim.threedvar import ThreeDVar
from paleoreco.models.diffusion import SIGMA_DATA
from paleoreco.models.spectral_score import build_spectral_denoiser

SHAPE = (2, 2, 2)
D = 8


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T / d + 0.3 * np.eye(d)


def _basis(k, B=None, safe_valid=None, seed=0):
    B = _spd(D, seed) if B is None else B
    safe = np.ones(SHAPE[1:], bool) if safe_valid is None else safe_valid
    return build_spectral_basis(B, SHAPE, safe, k)


def _sampler(k, *, b_scale=1.0, n_steps=128, B=None, safe_valid=None, **kw):
    basis = _basis(k, B=B, safe_valid=safe_valid)
    net = build_spectral_denoiser(dim=k, width=16, depth=1)
    return SpectralSampler(net, basis, b_scale=b_scale, n_steps=n_steps,
                           device="cpu", **kw)


# ---------------------------------------------------------------------------
# The basis.
# ---------------------------------------------------------------------------
def test_split_reconstructs_the_covariance():
    """``U_k diag(lam_k) U_k' + C_perp`` is B exactly, because U_k are B's own
    eigenvectors; nothing is discarded by truncating."""
    B = _spd(D, 3)
    b = _basis(3, B=B)
    rebuilt = (b.U_K * b.lam_K) @ b.U_K.T + b.half_perp @ b.half_perp.T
    assert np.allclose(rebuilt, B, atol=1e-8)


def test_eigenvalues_are_descending_and_positive():
    b = _basis(4, seed=1)
    assert np.all(np.diff(b.lam_K) <= 0)
    assert np.all(b.lam_K > 0)
    assert b.n_dim == 4
    assert np.allclose(b.U_K.T @ b.U_K, np.eye(4), atol=1e-10)


def test_whitening_maps_eigen_directions_to_sigma_data():
    """A direction carrying exactly its own prior standard deviation whitens to
    ``sigma_data`` on that axis and nothing elsewhere."""
    b = _basis(3, seed=2)
    for j in range(b.n_dim):
        x = b.U_K[:, j] * np.sqrt(b.lam_K[j])
        z = b.whiten(x[None, :])[0]
        want = np.zeros(b.n_dim)
        want[j] = SIGMA_DATA
        assert np.allclose(z, want, atol=1e-10)
        assert np.allclose(b.unwhiten(z[None, :])[0], x, atol=1e-10)


def test_whitened_states_have_unit_frame_variance():
    """Draws from ``N(0, B)`` whiten to covariance ``sigma_data^2 I``."""
    B = _spd(D, 5)
    b = _basis(4, B=B)
    rng = np.random.default_rng(0)
    lam, U = np.linalg.eigh(B)
    x = rng.standard_normal((20000, D)) @ (U * np.sqrt(np.clip(lam, 0, None))).T
    cov = np.cov(b.whiten(x), rowvar=False)
    assert np.allclose(cov, SIGMA_DATA ** 2 * np.eye(4), atol=0.02)


def test_negative_eigenvalues_are_clipped():
    """The tuned pixel B is a Schur product of great-circle Gaspari-Cohn with a
    coupling factor and is not positive semi-definite; whitening takes a square
    root, so the projection has to happen here."""
    B = _spd(D, 7)
    lam, U = np.linalg.eigh(B)
    lam[0] = -0.5                                    # force an indefinite matrix
    B_bad = (U * lam) @ U.T
    b = build_spectral_basis(B_bad, SHAPE, np.ones(SHAPE[1:], bool), 2)
    assert b.meta["n_negative_clipped"] == 1
    rebuilt = (b.U_K * b.lam_K) @ b.U_K.T + b.half_perp @ b.half_perp.T
    assert np.allclose(rebuilt, (U * np.clip(lam, 0, None)) @ U.T, atol=1e-8)


def test_basis_meta_records_provenance():
    b = build_spectral_basis(_spd(D, 0), SHAPE, np.ones(SHAPE[1:], bool), 3,
                             meta={"prior": "chunkA", "alpha": 0.25})
    assert b.meta["k"] == 3
    assert b.meta["prior"] == "chunkA" and b.meta["alpha"] == 0.25
    assert 0.0 < b.meta["leading_variance_share"] <= 1.0
    assert b.meta["n_complement_modes"] == D - 3


@pytest.mark.parametrize("k", [0, D, D + 1])
def test_out_of_range_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must satisfy"):
        _basis(k)


def test_k_beyond_the_rank_is_rejected():
    """A rank-2 covariance cannot support a 3-dim whitened subspace."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((D, 2))
    with pytest.raises(ValueError, match="too small to whiten"):
        build_spectral_basis(A @ A.T, SHAPE, np.ones(SHAPE[1:], bool), 3)


def test_shape_and_covariance_size_must_agree():
    with pytest.raises(ValueError, match="flattens to"):
        build_spectral_basis(_spd(D, 0), (2, 2, 3), np.ones((2, 3), bool), 2)


# ---------------------------------------------------------------------------
# Covariance matching of the training states.
# ---------------------------------------------------------------------------
def _states(basis, n=800, seed=0, bimodal=False):
    """States with a controlled subspace part and a Gaussian complement.

    With ``bimodal`` the first retained direction carries two well-separated modes,
    which is the structure the matching must not destroy.
    """
    rng = np.random.default_rng(seed)
    k, R = basis.n_dim, basis.half_perp.shape[1]
    a = rng.standard_normal((n, k)) * np.linspace(4.0, 1.0, k)
    a = a + 0.4 * a[:, [0]]                       # correlate the directions
    if bimodal:
        a[:, 0] = rng.choice([-4.0, 4.0], size=n) + rng.standard_normal(n) * 0.6
    return a @ basis.U_K.T + 0.3 * (rng.standard_normal((n, R)) @ basis.half_perp.T)


def _bic_margin(v):
    """BIC of one Gaussian minus that of a two-component mixture; positive = two modes."""
    from sklearn.mixture import GaussianMixture

    x = np.asarray(v).reshape(-1, 1)
    return float(GaussianMixture(1, random_state=0, n_init=5).fit(x).bic(x)
                 - GaussianMixture(2, random_state=0, n_init=10).fit(x).bic(x))


def test_matching_pins_the_subspace_covariance_to_B():
    """The whole point: after matching, the retained block carries B's eigenvalues,
    which is the covariance the lane's 3DVar baseline was tuned with."""
    b = _basis(3, seed=1)
    x = _states(b)
    matched, info = match_subspace_covariance(b, x)
    cov = np.cov(matched @ b.U_K, rowvar=False, ddof=1)
    assert np.allclose(cov, np.diag(b.lam_K), rtol=1e-8, atol=1e-8)
    assert np.allclose(info["sd_after"] ** 2, b.lam_K, rtol=1e-2)
    assert info["condition"] > 1.0


def test_matched_states_whiten_to_the_anchor():
    """Whitening matched states must give exactly the frame the denoiser is anchored
    to, so an untrained model and the training target agree on the second moment."""
    b = _basis(4, seed=2)
    matched, _ = match_subspace_covariance(b, _states(b))
    z = b.whiten(matched)
    assert np.allclose(np.cov(z, rowvar=False, ddof=1),
                       SIGMA_DATA ** 2 * np.eye(4), rtol=1e-8, atol=1e-8)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-8)


def test_matching_leaves_the_complement_untouched():
    """Only the learned block may move; the complement already carries B's covariance
    because the sampler draws it from B."""
    b = _basis(3, seed=3)
    x = _states(b)
    matched, _ = match_subspace_covariance(b, x)
    delta = matched - x
    in_subspace = (delta @ b.U_K) @ b.U_K.T
    assert np.allclose(delta, in_subspace, atol=1e-9)


def test_matching_is_an_invertible_linear_map():
    """Invertibility is what guarantees no distributional information is lost."""
    b = _basis(3, seed=4)
    x = _states(b)
    matched, _ = match_subspace_covariance(b, x)
    a, a_m = x @ b.U_K, matched @ b.U_K
    T, *_ = np.linalg.lstsq(a - a.mean(0), a_m, rcond=None)
    assert np.linalg.matrix_rank(T) == b.n_dim
    assert np.allclose((a - a.mean(0)) @ T, a_m, atol=1e-8)


def test_matching_preserves_the_joint_distribution():
    """The fix restores the second moment without touching the higher ones.

    It does rotate, and that is unavoidable: pinning a covariance whose off-diagonals
    are non-zero must mix the directions, so the per-coordinate marginals afterwards
    are blends and individually look more Gaussian than the originals did. Nothing is
    lost, because the map is invertible - every projection of the original reappears
    along the corresponding transformed direction. That invariance is what is pinned
    here, rather than the marginals of the new axes.
    """
    b = _basis(3, seed=5)
    x = _states(b, bimodal=True)
    matched, info = match_subspace_covariance(b, x)
    a = x @ b.U_K
    a = a - a.mean(axis=0)

    back = (matched @ b.U_K) @ np.linalg.inv(info["transform"])
    assert np.allclose(back, a, atol=1e-8)
    assert _bic_margin(a[:, 0]) > 0, "the fixture is not bimodal, so the test is empty"
    assert _bic_margin(back[:, 0]) == pytest.approx(_bic_margin(a[:, 0]), rel=1e-2)


def test_matching_rejects_too_few_states():
    b = _basis(3, seed=6)
    with pytest.raises(ValueError, match="need more than k=3"):
        match_subspace_covariance(b, _states(b, n=3))


def test_matching_rejects_a_rank_deficient_subspace():
    """States confined to one retained direction cannot define the block."""
    b = _basis(3, seed=7)
    rng = np.random.default_rng(0)
    x = np.outer(rng.standard_normal(200), b.U_K[:, 0])
    with pytest.raises(ValueError, match="singular|ill-conditioned"):
        match_subspace_covariance(b, x)


def test_matching_preserves_the_input_shape():
    b = _basis(2, seed=8)
    x = _states(b, n=50).reshape(50, *SHAPE)
    matched, _ = match_subspace_covariance(b, x)
    assert matched.shape == x.shape


# ---------------------------------------------------------------------------
# The sampler.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cond_cov", ["anchor", "jacobian"])
@pytest.mark.parametrize("b_scale", [1.0, 2.0])
def test_zero_network_posterior_mean_matches_3dvar(b_scale, cond_cov):
    """Exact nesting, under either conditional covariance. With the network output
    zero the prior is ``N(0, b_scale B)`` and both branches compute its exact
    annealed conditional, so the sampler must reproduce the gain-form analysis
    including the complement completion."""
    B = _spd(D, 0)
    samp = _sampler(2, b_scale=b_scale, B=B, cond_cov=cond_cov)
    gather = np.array([0, 5])
    y = np.array([1.2, -0.8])
    r = np.full(2, 0.05)

    ens = samp.sample_posterior(gather, y, r, b_scale, 4000, seed=2)
    got = ens.reshape(4000, -1).mean(axis=0)

    want = ThreeDVar(b_scale * B, SHAPE).analyze(
        Observations(gather=gather, y_anom=y, sse=r), np.zeros(D)).mean_anom.ravel()
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 0.1


def test_zero_network_posterior_variance_matches_3dvar():
    """The ensemble spread must match ``diag((I - KH) B)`` too, not just the mean."""
    B = _spd(D, 0)
    samp = _sampler(3, b_scale=1.0, B=B)
    gather = np.array([1, 6])
    y = np.array([0.4, -0.9])
    r = np.full(2, 0.2)

    ens = samp.sample_posterior(gather, y, r, 1.0, 6000, seed=5)
    got = ens.reshape(6000, -1).var(axis=0, ddof=1)
    want = ThreeDVar(B, SHAPE).analyze(
        Observations(gather=gather, y_anom=y, sse=r), np.zeros(D)).posterior_var.ravel()
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 0.15


def test_prior_sampling_recovers_the_scaled_covariance():
    B = _spd(D, 0)
    samp = _sampler(3, b_scale=2.0, B=B, n_steps=128)
    x = samp.sample_prior(4000, seed=1).reshape(4000, -1)
    rel = np.linalg.norm(np.cov(x, rowvar=False) - 2.0 * B) / np.linalg.norm(2.0 * B)
    assert np.abs(x.mean(0)).max() < 0.25
    assert rel < 0.4


def test_sample_prior_honours_an_explicit_amplitude():
    samp = _sampler(2, b_scale=1.0, n_steps=32)
    lo = samp.sample_prior(600, seed=0, b_scale=0.5).var()
    hi = samp.sample_prior(600, seed=0, b_scale=4.0).var()
    assert hi > 3.0 * lo


def test_masked_cells_stay_zero():
    safe = np.ones(SHAPE[1:], bool)
    safe[0] = False
    samp = _sampler(2, n_steps=8, safe_valid=safe)
    x = samp.sample_prior(3, seed=0)
    assert np.allclose(x[:, :, ~safe], 0.0)


def test_chunked_draws_produce_an_independent_ensemble():
    """A draw larger than ``max_batch`` is taken in chunks; the chunks must not
    share a noise stream, or the ensemble would carry duplicates."""
    samp = _sampler(2, n_steps=8, max_batch=4)
    x = samp.sample_prior(12, seed=3).reshape(12, -1)
    assert x.shape[0] == 12
    pair = np.abs(x[:4] - x[4:8]).max()
    assert pair > 1e-6


def test_analyze_returns_mean_and_variance():
    samp = _sampler(2, b_scale=2.0, n_steps=8, n_samples=6)
    obs = Observations(gather=np.array([0, 5]), y_anom=np.array([0.5, -0.5]),
                       sse=np.array([0.1, 0.1]))
    res = samp.analyze(obs, np.zeros(D))
    assert res.mean_anom.shape == SHAPE
    assert np.isfinite(res.mean_anom).all()
    assert res.posterior_var is not None and (res.posterior_var >= 0).all()


def test_imap_posterior_preserves_order():
    from paleoreco.assim.generative import PosteriorJob

    samp = _sampler(2, n_steps=8)
    jobs = [PosteriorJob(np.array([0, 5]), np.array([1.0, -1.0]), np.full(2, 0.1),
                         1.0, 3, seed) for seed in (11, 12)]
    ens = list(samp.imap_posterior(jobs))
    assert len(ens) == 2 and all(e.shape == (3, *SHAPE) for e in ens)
    direct = samp.sample_posterior(jobs[1].gather, jobs[1].y_anom, jobs[1].r_diag,
                                   1.0, 3, seed=12)
    assert np.allclose(ens[1], direct)


# ---------------------------------------------------------------------------
# Exact conditional covariance from the denoiser Jacobian.
# ---------------------------------------------------------------------------
def _anchor_cov(sigma):
    """``Cov(z | z_sigma)`` for the Gaussian anchor: a scalar times the identity."""
    return SIGMA_DATA ** 2 * sigma ** 2 / (SIGMA_DATA ** 2 + sigma ** 2)


@pytest.mark.parametrize("sigma", [0.05, 0.5, 3.0, 80.0])
def test_jacobian_conditional_cov_reduces_to_the_anchor(sigma):
    """With a zero-initialised network the model *is* ``N(0, sigma_data^2 I)``, whose
    conditional covariance is known in closed form. This pins the Jacobian
    convention and the ``sigma^2`` factor together: get either wrong and it fails.
    """
    samp = _sampler(4, n_steps=4, cond_cov="jacobian")
    xg = torch.randn(6, 4, 1, 1, requires_grad=True)
    gam = samp._conditional_cov(samp.denoiser(xg, sigma), xg, sigma)
    want = _anchor_cov(sigma) * torch.eye(4)
    assert gam.shape == (6, 4, 4)
    assert torch.allclose(gam, want.expand(6, 4, 4), atol=1e-5)


def test_jacobian_conditional_cov_is_symmetric_and_psd():
    """A trained network's Jacobian is not constrained to be either, so the
    projection has to make it so before it is used as a covariance."""
    samp = _sampler(4, n_steps=4, cond_cov="jacobian")
    torch.nn.init.normal_(samp.denoiser.net.out.weight, std=0.6)
    xg = torch.randn(8, 4, 1, 1, requires_grad=True)
    gam = samp._conditional_cov(samp.denoiser(xg, 0.7), xg, 0.7)
    assert torch.allclose(gam, gam.transpose(1, 2), atol=1e-6)
    assert torch.linalg.eigvalsh(gam).min() >= -1e-6
    assert 0.0 <= samp.clip_rate <= 1.0


def test_woodbury_quadratic_form_matches_the_direct_inverse():
    """The identity the jacobian branch rests on, checked against building and
    inverting ``S = C_y + A Gamma A'`` outright."""
    rng = np.random.default_rng(3)
    m, k = 7, 3
    A = rng.standard_normal((m, k))
    C_y = _spd(m, 5)
    r = rng.standard_normal(m)
    Q = rng.standard_normal((k, k))
    Gam = Q @ Q.T                                     # PSD, possibly ill-conditioned

    Cy_inv = np.linalg.inv(C_y)
    M, G = Cy_inv @ A, A.T @ np.linalg.inv(C_y) @ A
    W = np.linalg.solve(np.eye(k) + Gam @ G, Gam)
    q = M.T @ r
    got = r @ Cy_inv @ r - q @ W @ q
    want = r @ np.linalg.inv(C_y + A @ Gam @ A.T) @ r
    assert got == pytest.approx(want, rel=1e-9)

    # A zeroed Gamma is what the PSD projection can produce, and must be safe.
    W0 = np.linalg.solve(np.eye(k), np.zeros((k, k)))
    assert (r @ Cy_inv @ r - q @ W0 @ q) == pytest.approx(r @ Cy_inv @ r, rel=1e-12)


def test_the_two_conditional_covariances_agree_for_a_gaussian_model():
    """The strongest available statement that the jacobian branch generalises rather
    than replaces: with a zero-initialised network the two modes must produce the
    same ensemble from the same seed, not merely similar skill."""
    B = _spd(D, 0)
    gather, y, r = np.array([0, 5]), np.array([1.2, -0.8]), np.full(2, 0.05)
    ens = [_sampler(3, b_scale=2.0, n_steps=16, B=B, cond_cov=c)
           .sample_posterior(gather, y, r, 2.0, 24, seed=4)
           for c in ("anchor", "jacobian")]
    assert np.allclose(ens[0], ens[1], atol=1e-4)


def test_cond_cov_mode_is_validated():
    basis = _basis(2)
    net = build_spectral_denoiser(dim=2, width=8, depth=1)
    with pytest.raises(ValueError, match="cond_cov must be one of"):
        SpectralSampler(net, basis, cond_cov="exact", device="cpu")


def test_denoiser_and_basis_dimension_must_agree():
    basis = _basis(3)
    net = build_spectral_denoiser(dim=2, width=8, depth=1)
    with pytest.raises(ValueError, match="does not match|is 2-dim"):
        SpectralSampler(net, basis, device="cpu")
