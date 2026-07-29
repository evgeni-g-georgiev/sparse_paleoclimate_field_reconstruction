"""Worker-process pool over the guided sampler.

One posterior draw occupies a small fraction of a GPU, so submitting draws one at a
time leaves most of the device idle. Each worker rebuilds its own
:class:`~paleoreco.assim.generative.GuidedSampler` from a checkpoint and every job
carries its own seed, so a pooled run returns what a serial one returns.

The sampling surface mirrors :class:`~paleoreco.assim.generative.GuidedSampler`, so
callers that only draw samples take either object.
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import torch

from paleoreco.assim.generative import GuidedSampler, PosteriorJob
from paleoreco.models.diffusion import load_denoiser

_SAMPLER: GuidedSampler | None = None


def _init_worker(checkpoint_path: str, safe_valid: np.ndarray, kwargs: dict) -> None:
    """Build this worker's sampler once, so the model and CUDA context are reused."""
    global _SAMPLER
    net, scales = load_denoiser(checkpoint_path)
    _SAMPLER = GuidedSampler(net, np.asarray(scales), safe_valid, **kwargs)


def _run_posterior(job) -> np.ndarray:
    return _SAMPLER.sample_posterior(job.gather, job.y_anom, job.r_diag,
                                     job.gamma, job.n, job.seed)


def _run_prior(n: int, seed: int) -> np.ndarray:
    return _SAMPLER.sample_prior(n, seed)


class SamplerPool:
    """A :class:`GuidedSampler` replicated across worker processes.

    ``n_steps``, ``n_correct``, ``tau`` and ``sd`` are mirrored from the sampler the
    workers build, so a run's provenance record does not depend on how it executed.
    """

    def __init__(self, checkpoint_path: str, safe_valid: np.ndarray, *,
                 n_workers: int = 4, gamma: float = 1e-2, n_samples: int = 16,
                 n_steps: int = 64, n_correct: int = 2, corrector_tau: float = 0.3,
                 device: str | None = None):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1; got {n_workers}")
        self.gamma = gamma
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_correct = n_correct
        self.tau = corrector_tau
        # Read sigma_data on the CPU rather than building a sampler here, so the
        # parent process never takes a CUDA context of its own.
        self.sd = float(torch.load(checkpoint_path, map_location="cpu")["sigma_data"])

        kwargs = {"gamma": gamma, "n_samples": n_samples, "n_steps": n_steps,
                  "n_correct": n_correct, "corrector_tau": corrector_tau,
                  "device": device}
        # spawn, not fork: a forked process cannot initialise CUDA.
        ctx = multiprocessing.get_context("spawn")
        self._pool = ctx.Pool(n_workers, initializer=_init_worker,
                              initargs=(str(checkpoint_path),
                                        np.asarray(safe_valid, dtype=bool), kwargs))

    def imap_posterior(self, jobs):
        """Posterior ensembles for a sequence of :class:`PosteriorJob`, in order."""
        return self._pool.imap(_run_posterior, jobs, chunksize=1)

    def sample_posterior(self, gather: np.ndarray, y_anom: np.ndarray,
                         r_diag: np.ndarray, gamma: float, n: int,
                         seed: int = 0) -> np.ndarray:
        """``n`` posterior fields in anomaly units, ``(n, 2, H, W)``."""
        return self._pool.apply(
            _run_posterior, (PosteriorJob(gather, y_anom, r_diag, gamma, n, seed),))

    def sample_prior(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` unconditional prior fields in anomaly units, ``(n, 2, H, W)``."""
        return self._pool.apply(_run_prior, (n, seed))

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self) -> "SamplerPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
