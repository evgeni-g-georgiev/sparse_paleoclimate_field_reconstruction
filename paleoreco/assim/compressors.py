"""Compressors that present one encode/decode contract to latent assimilation.

A compressor maps the anomaly field ``(N, 2, n_lat, n_lon)`` to a low-dimensional
code and back. Latent 3DVar treats the decoder as part of the observation operator,
so the contract exposes a differentiable ``decode_torch`` (for the gradient-based
solve and the decoder Jacobian) alongside numpy ``encode``/``decode``. ``linear``
flags whether the decoder is affine: PCA admits the closed-form gain, the neural
compressors require the iterative solve.

Everything is in anomaly units (see :func:`paleoreco.data.apply_anomaly`), the same
frame the pixel assimilation and the trained networks use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class Compressor(ABC):
    """Encode/decode between the anomaly field and a ``latent_dim`` code."""

    linear: bool
    latent_dim: int
    shape: tuple[int, int, int]            # (2, n_lat, n_lon)

    @abstractmethod
    def encode(self, fields: np.ndarray) -> np.ndarray:
        """``(N, 2, H, W)`` anomaly fields to ``(N, latent_dim)`` codes."""

    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """``(N, latent_dim)`` codes to ``(N, 2, H, W)`` anomaly fields."""

    @abstractmethod
    def decode_torch(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable decode of ``(N, latent_dim)`` to ``(N, 2, H, W)`` on ``z.device``."""

    def to(self, device) -> "Compressor":
        return self


# ---------------------------------------------------------------------------
# Linear: principal components.
# ---------------------------------------------------------------------------
class PCACompressor(Compressor):
    """Affine compressor over a POD basis; the decoder is ``z V_k + mu``.

    Built from a :func:`paleoreco.eval.shared.pod_fit` basis sliced to rank ``k``.
    ``V_k`` rows are orthonormal modes over the valid cells; ``mu`` is the fit-set
    mean (near zero in anomaly space). The estimator reads :meth:`linear_decoder`
    to assemble the exact observation operator ``H = select . V``.
    """

    linear = True

    def __init__(self, V_k: np.ndarray, mu: np.ndarray, keep_mask: np.ndarray,
                 shape: tuple[int, int, int]):
        self.V_k = np.asarray(V_k, dtype=np.float64)          # (k, D_valid)
        self.mu = np.asarray(mu, dtype=np.float64).reshape(1, -1)
        self.keep = np.asarray(keep_mask, dtype=bool)         # (2*H*W,)
        self.shape = shape
        self.latent_dim = self.V_k.shape[0]
        self._keep_idx_np = np.flatnonzero(self.keep)

    @classmethod
    def from_pod_fit(cls, pod_basis: dict, k: int) -> "PCACompressor":
        if not (1 <= k <= pod_basis["V_max"].shape[0]):
            raise ValueError(f"k={k} out of range for basis max_k={pod_basis['V_max'].shape[0]}")
        return cls(pod_basis["V_max"][:k], pod_basis["mu"], pod_basis["keep_mask"],
                   pod_basis["shape"])

    def linear_decoder(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(V_k, mu, keep_mask)`` so a caller can build ``H`` and decode in closed form."""
        return self.V_k, self.mu, self.keep

    def encode(self, fields: np.ndarray) -> np.ndarray:
        X = np.asarray(fields, dtype=np.float64).reshape(len(fields), -1)[:, self.keep]
        return ((X - self.mu) @ self.V_k.T).astype(np.float64)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        xv = np.asarray(codes, dtype=np.float64) @ self.V_k + self.mu     # (N, D_valid)
        full = np.zeros((len(xv), self.keep.size), dtype=np.float64)
        full[:, self.keep] = xv
        return full.reshape(len(xv), *self.shape)

    def decode_torch(self, z: torch.Tensor) -> torch.Tensor:
        V_k = torch.as_tensor(self.V_k, device=z.device, dtype=z.dtype)
        mu = torch.as_tensor(self.mu, device=z.device, dtype=z.dtype)
        idx = torch.as_tensor(self._keep_idx_np, device=z.device)
        xv = z @ V_k + mu                                                  # (N, D_valid)
        full = z.new_zeros((z.shape[0], self.keep.size))
        full[:, idx] = xv
        return full.reshape(z.shape[0], *self.shape)


# ---------------------------------------------------------------------------
# Nonlinear: trained autoencoder / variational autoencoder.
# ---------------------------------------------------------------------------
class _NeuralCompressor(Compressor):
    """Wraps a trained net whose decoder is the nonlinear part of ``H``.

    The networks take a 3-channel input ``[mtco_anom, mtwa_anom, valid_mask]`` and
    decode to 2 channels, so :meth:`encode` appends the constant mask channel
    (mirroring :class:`paleoreco.data.PaleoFieldDataset`). The net is held in eval
    mode; the latent code is the encoder output (the posterior mean for the VAE).
    """

    linear = False

    def __init__(self, net, mask: np.ndarray, shape: tuple[int, int, int], device="cpu"):
        self.net = net.eval()
        self.mask = np.asarray(mask, dtype=bool)
        self.shape = shape
        self.latent_dim = net.latent_dim
        self.device = device
        self.net.to(device)
        self._mask_t = torch.from_numpy(self.mask.astype(np.float32)).to(device)

    def to(self, device) -> "_NeuralCompressor":
        self.device = device
        self.net.to(device)
        self._mask_t = self._mask_t.to(device)
        return self

    def _input(self, fields_t: torch.Tensor) -> torch.Tensor:
        m = self._mask_t.expand(fields_t.shape[0], 1, *self.shape[1:])
        return torch.cat([fields_t, m], dim=1)

    @abstractmethod
    def _encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output reduced to the code used as the latent state."""

    def encode(self, fields: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(fields, dtype=np.float32), device=self.device)
        with torch.no_grad():
            codes = self._encode_codes(self._input(x))
        return codes.detach().cpu().numpy().astype(np.float64)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        z = torch.as_tensor(np.asarray(codes, dtype=np.float32), device=self.device)
        with torch.no_grad():
            out = self.net.decode(z)
        return out.detach().cpu().numpy().astype(np.float64)

    def decode_torch(self, z: torch.Tensor) -> torch.Tensor:
        return self.net.decode(z)


class AECompressor(_NeuralCompressor):
    """Deterministic autoencoder (:class:`paleoreco.models.ConvAE`)."""

    def _encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.encode(x)


class VAECompressor(_NeuralCompressor):
    """Variational autoencoder (:class:`paleoreco.models.ConvBetaVAE`); code is the mean."""

    def _encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.encode(x)[0]


# ---------------------------------------------------------------------------
# Latent prior from the encoded prior ages.
# ---------------------------------------------------------------------------
def latent_prior(compressor: Compressor, cube_anom: np.ndarray,
                 prior_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latent background covariance ``B_z`` and climatological mean ``z_clim``.

    Encodes the prior ages once; ``B_z`` is the sample covariance of the codes and
    ``z_clim`` their mean. ``z_clim`` is the centre of the latent prior the cost
    measures against, so it is the climatological first guess (0 for PCA/VAE).
    """
    codes = compressor.encode(cube_anom[np.asarray(prior_idx)])           # (N, d)
    B_z = np.atleast_2d(np.cov(codes, rowvar=False, ddof=1))
    z_clim = codes.mean(axis=0)
    return B_z, z_clim
