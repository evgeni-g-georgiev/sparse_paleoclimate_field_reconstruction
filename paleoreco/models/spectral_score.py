"""Score network for a whitened low-dimensional subspace of the state.

The pixel denoiser of :mod:`paleoreco.models.diffusion` has to learn a 4096-dim
distribution from a few hundred autocorrelated states, and its leading modes sit
at noise levels the EDM training distribution never reaches. This network instead
models only the leading ``dim`` directions of the background covariance, whitened
so that their Gaussian model is exactly ``N(0, sigma_data^2 I)``; the orthogonal
complement stays Gaussian and is handled analytically by
:mod:`paleoreco.assim.spectral`.

Two consequences follow from the whitening, and both are the point of the design:

* Every retained direction has the same variance, so one noise band resolves all
  of them. In the per-cell frame the pixel model trains in, the leading modes sit
  at ``sigma`` of order 10 to 20 while the training lognormal has its 99th
  percentile near 5.
* EDM's preconditioning writes the denoiser as ``c_skip x + c_out F(...)``, and
  ``c_skip x`` is the exact denoiser of ``N(0, sigma_data^2 I)``. Zero-initialising
  the output layer therefore makes an untrained model *equal* to the Gaussian
  prior that 3DVar assumes, so training can only add departures from it.

The state is carried as ``(N, dim, 1, 1)`` - a field over a one-cell grid with
``dim`` channels - rather than ``(N, dim)``. The trailing axes are degenerate and
nothing here is spatial; they exist so that
:class:`~paleoreco.models.diffusion.EDMDenoiser` and the training loop of
:mod:`paleoreco.training.trainer_diffusion`, which both assume a 4-D field, apply
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from paleoreco.models.diffusion import (
    EDMDenoiser,
    SIGMA_DATA,
    _sinusoidal_embedding,
)


class ScoreMLP(nn.Module):
    """Noise-conditioned MLP over ``R^dim``, presented as a 1x1 field.

    The noise embedding is concatenated to the input rather than applied as FiLM:
    at these widths the network is small enough that the extra conditioning
    machinery buys nothing, and a single concatenation keeps the zero-initialised
    output layer the only thing standing between the model and the exact Gaussian
    denoiser.
    """

    def __init__(self, *, dim: int, width: int = 256, depth: int = 3,
                 emb_dim: int = 64):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        if emb_dim < 2 or emb_dim % 2:
            raise ValueError(f"emb_dim must be a positive even number for the "
                             f"sinusoidal features; got {emb_dim}")
        self.config = {"dim": dim, "width": width, "depth": depth, "emb_dim": emb_dim}
        self.dim = dim
        self.emb_dim = emb_dim

        self.map_noise = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                       nn.Linear(emb_dim, emb_dim))
        layers: list[nn.Module] = [nn.Linear(dim + emb_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(width, dim)
        # Load-bearing: a zero output makes the wrapped denoiser exactly the
        # denoiser of N(0, sigma_data^2 I), which is the Gaussian prior the
        # subspace is whitened against.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = x.reshape(shape[0], -1)
        if h.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} values per sample, got {h.shape[1]} "
                             f"from shape {tuple(shape)}")
        emb = self.map_noise(_sinusoidal_embedding(c_noise.reshape(-1), self.emb_dim))
        return self.out(self.body(torch.cat([h, emb], dim=1))).reshape(shape)


def build_spectral_denoiser(*, dim: int, sigma_data: float = SIGMA_DATA,
                            **cfg) -> EDMDenoiser:
    """A :class:`ScoreMLP` under the EDM preconditioning, ready to train or sample."""
    return EDMDenoiser(ScoreMLP(dim=dim, **cfg), sigma_data)


def load_spectral_denoiser(path: str, map_location="cpu") -> tuple[EDMDenoiser, dict]:
    """Rebuild a subspace denoiser from a checkpoint; return it and the full payload.

    The payload is returned so a caller can check the provenance scalars the
    notebook writes (``spectral_k`` and the taper the basis was built from) before
    trusting the weights: unlike the pixel checkpoint, the frame this model lives
    in is not stored here but rebuilt from the background covariance, so a
    checkpoint from a different basis would load silently.
    """
    ckpt = torch.load(path, map_location=map_location)
    net = EDMDenoiser(ScoreMLP(**ckpt["config"]), ckpt["sigma_data"])
    net.load_state_dict(ckpt["state_dict"])
    return net, ckpt
