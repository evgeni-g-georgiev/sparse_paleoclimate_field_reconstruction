"""Pooled sampling returns what serial sampling returns.

The lanes hand their whole job list to whichever sampler they were given, so the
two execution paths must agree draw for draw.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paleoreco.assim.generative import GuidedSampler, PosteriorJob
from paleoreco.assim.sampler_pool import SamplerPool
from paleoreco.models.diffusion import CircularUNet, EDMDenoiser
from paleoreco.training._common import set_seed

GRID = (12, 12)
SCALES = np.array([1.5, 1.0])
KW = dict(n_steps=4, n_correct=1, device="cpu")


def _net():
    set_seed(0)
    return EDMDenoiser(CircularUNet(base_channels=16, depth=2, grid_shape=GRID))


def _jobs(n=3):
    rng = np.random.default_rng(0)
    gather = np.array([5, 20, 150, 200])
    return [PosteriorJob(gather, rng.normal(0.0, 1.0, len(gather)),
                         np.full(len(gather), 0.1), gamma=0.01, n=2, seed=k)
            for k in range(n)]


def _checkpoint(tmp_path):
    """A checkpoint carrying the payload keys the trainer writes."""
    net = _net()
    path = tmp_path / "diffusion.pt"
    torch.save({"state_dict": net.state_dict(), "config": net.config,
                "sigma_data": net.sigma_data,
                "channel_scales": SCALES.tolist()}, path)
    return str(path)


def test_imap_posterior_matches_sample_posterior():
    samp = GuidedSampler(_net(), SCALES, np.ones(GRID, bool), **KW)
    jobs = _jobs()
    for job, ens in zip(jobs, samp.imap_posterior(jobs)):
        direct = samp.sample_posterior(job.gather, job.y_anom, job.r_diag,
                                       job.gamma, job.n, job.seed)
        assert np.array_equal(ens, direct)


def test_pool_matches_serial(tmp_path):
    jobs = _jobs()
    serial = list(GuidedSampler(_net(), SCALES, np.ones(GRID, bool),
                                **KW).imap_posterior(jobs))
    with SamplerPool(_checkpoint(tmp_path), np.ones(GRID, bool),
                     n_workers=2, **KW) as pool:
        pooled = list(pool.imap_posterior(jobs))
        single = pool.sample_posterior(jobs[0].gather, jobs[0].y_anom, jobs[0].r_diag,
                                       jobs[0].gamma, jobs[0].n, jobs[0].seed)
        prior = pool.sample_prior(2, seed=0)

    assert len(pooled) == len(serial)
    for a, b in zip(serial, pooled):
        assert np.allclose(a, b)
    assert np.allclose(single, serial[0])
    assert prior.shape == (2, 2, *GRID)


def test_pool_closes_its_workers(tmp_path):
    pool = SamplerPool(_checkpoint(tmp_path), np.ones(GRID, bool), n_workers=2, **KW)
    pool.close()
    with pytest.raises(ValueError):
        pool.sample_prior(1)
