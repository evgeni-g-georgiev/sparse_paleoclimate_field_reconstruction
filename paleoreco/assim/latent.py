"""Latent-space 3DVar: assimilate a low-dimensional code, decode to the field.

The state and its covariance ``B_z`` live in the compressor's latent space; the
observations, ``R``, and the innovation stay in pixel/anomaly space; the observation
operator is ``H = decode . select``. The decoder is linearised to its Jacobian ``P``
about ``z_clim`` (exact for the affine PCA decoder, the tangent for a neural one), so the
analysis is a single closed-form latent BLUE. Fan et al. 2025 (AE-O2L) find this one-step
latent gain beats a variational solve through the decoder in the sparse-observation regime.

Posterior uncertainty is the linearised pixel diagonal ``diag(P A_z P^T)``.
Reported fields are re-centred to ``decode(z_a) - decode(z_b) + background`` so the
climatological background maps to the zero-anomaly field (pixel and PCA already do), which
removes a neural decoder's reconstruction error of the background. All estimators return
pixel-space :class:`AnalysisResult`, identical to pixel 3DVar for the drivers and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from paleoreco.assim.compressors import Compressor
from paleoreco.assim.method import AnalysisResult, Method, Observations

_UNIT = np.array([1.0])


@dataclass(frozen=True)
class _LatentSweepGain:
    """Per-network latent-3DVar operators, all value-free (no y, no background).

    ``H`` is the decoder linearised at the obs cells; ``Lam``/``U``/``G`` diagonalize
    the R^-1/2-whitened obs block for the b_scale sweep (as in pixel 3DVar).
    ``post_cov_z`` is the latent posterior covariance per b_scale, ``(n_b, d, d)``, and
    ``post_var_pixel`` its linearised pixel diagonal, ``(n_b, D)``.
    """

    gather: np.ndarray
    H: np.ndarray
    Lam: np.ndarray
    U: np.ndarray
    G: np.ndarray
    rinv_sqrt: np.ndarray
    b_scales: np.ndarray
    post_cov_z: np.ndarray
    post_var_pixel: np.ndarray


class _LatentVar(Method):
    """Shared latent-3DVar machinery; subclasses differ only in the pushforward ``P``.

    ``P`` (``D x d``) is the decoder Jacobian about ``z_clim``: it builds ``H`` (its rows
    at the observed cells) and maps a latent covariance to the pixel posterior variance.
    For an affine decoder it is exact and constant; for a neural decoder it is the tangent
    at the climatological mean.
    """

    def __init__(self, compressor: Compressor, B_z: np.ndarray, z_clim: np.ndarray,
                 shape: tuple[int, int, int], safe_valid: np.ndarray):
        self.compressor = compressor
        self.B_z = np.asarray(B_z, dtype=np.float64)
        self.z_clim = np.asarray(z_clim, dtype=np.float64).ravel()
        self.shape = shape
        self.latent_dim = self.z_clim.size
        self.safe_valid = np.asarray(safe_valid, dtype=bool)
        self.keep = np.broadcast_to(self.safe_valid, shape).ravel()       # (D,)
        self.P = self._pushforward()                                      # (D, d)
        self.diagB = self._pixel_diag(self.B_z)                           # (D,)

    # -- subclass hooks --------------------------------------------------
    def _pushforward(self) -> np.ndarray:
        raise NotImplementedError

    def _means(self, gain: _LatentSweepGain, y_anom: np.ndarray,
               z_b: np.ndarray) -> np.ndarray:
        """Latent analyses ``(n_b, d)``, one per b_scale."""
        raise NotImplementedError

    # -- shared ----------------------------------------------------------
    def _pixel_diag(self, A_z: np.ndarray) -> np.ndarray:
        """Diagonal of ``P A_z P^T``: a latent covariance pushed to pixel space, ``(D,)``."""
        return ((self.P @ A_z) * self.P).sum(axis=1)

    def _z_b(self, background_anom: np.ndarray) -> np.ndarray:
        """Latent background: the prior mean for the climatological (zero) field, else encode."""
        bg = np.asarray(background_anom, dtype=np.float64).ravel()
        if not np.any(bg):
            return self.z_clim
        return self.compressor.encode(bg.reshape(1, *self.shape))[0]

    def _decode_field(self, z: np.ndarray) -> np.ndarray:
        """Decode one latent code to a masked ``(2, H, W)`` anomaly field."""
        field = self.compressor.decode(z.reshape(1, -1))[0]
        return field * self.safe_valid

    def prepare_sweep(self, gather: np.ndarray, r_diag: np.ndarray,
                      b_scales: np.ndarray) -> _LatentSweepGain:
        g = np.asarray(gather)
        r = np.asarray(r_diag, dtype=np.float64)
        b_scales = np.asarray(b_scales, dtype=np.float64)
        H = self.P[g]                                                     # (m, d)
        rinv_sqrt = 1.0 / np.sqrt(r)
        HBzHt = H @ self.B_z @ H.T                                        # (m, m)
        A_t = (HBzHt * rinv_sqrt[:, None]) * rinv_sqrt[None, :]
        A_t = 0.5 * (A_t + A_t.T)
        Lam, U = np.linalg.eigh(A_t)
        G = (self.B_z @ H.T * rinv_sqrt[None, :]) @ U                     # (d, m)

        post_cov_z = np.stack([self._post_cov(G, Lam, k) for k in b_scales])   # (n_b, d, d)
        post_var_pixel = np.stack([self._pixel_diag(A) for A in post_cov_z])   # (n_b, D)
        return _LatentSweepGain(gather=g, H=H, Lam=Lam, U=U, G=G,
                                rinv_sqrt=rinv_sqrt, b_scales=b_scales,
                                post_cov_z=post_cov_z, post_var_pixel=post_var_pixel)

    def _post_cov(self, G: np.ndarray, Lam: np.ndarray, k: float) -> np.ndarray:
        """Latent posterior covariance at b_scale ``k``: ``k B_z - k^2 G (kLam+1)^-1 G^T``."""
        Gk = G / (k * Lam + 1.0)[None, :]
        return k * self.B_z - k ** 2 * (Gk @ G.T)

    def apply_sweep(self, gain: _LatentSweepGain, y_anom: np.ndarray,
                    background_anom: np.ndarray) -> list[AnalysisResult]:
        z_b = self._z_b(background_anom)
        z_a = self._means(gain, np.asarray(y_anom, dtype=np.float64), z_b)   # (n_b, d)
        bg_field = np.asarray(background_anom, dtype=np.float64).reshape(self.shape) * self.safe_valid
        c_field = self._decode_field(z_b) - bg_field                        # remove decode(z_b) bias
        out = []
        for ki in range(len(gain.b_scales)):
            mean_anom = self._decode_field(z_a[ki]) - c_field
            out.append(AnalysisResult(
                mean_anom=mean_anom,
                posterior_var=gain.post_var_pixel[ki].reshape(self.shape)))
        return out

    def post_var_sweep(self, gain: _LatentSweepGain) -> np.ndarray:
        return gain.post_var_pixel

    def analyze(self, obs: Observations, background_anom: np.ndarray) -> AnalysisResult:
        gain = self.prepare_sweep(obs.gather, obs.sse, _UNIT)
        return self.apply_sweep(gain, obs.y_anom, background_anom)[0]

    def analyze_many(self, obs: Observations,
                     backgrounds: list[np.ndarray]) -> list[AnalysisResult]:
        gain = self.prepare_sweep(obs.gather, obs.sse, _UNIT)
        return [self.apply_sweep(gain, obs.y_anom, bg)[0] for bg in backgrounds]


class LinearLatentVar(_LatentVar):
    """Closed-form latent 3DVar for an affine (PCA) decoder."""

    def _pushforward(self) -> np.ndarray:
        V_k, _mu, keep = self.compressor.linear_decoder()                # V_k (d, D_valid)
        P = np.zeros((keep.size, V_k.shape[0]), dtype=np.float64)
        P[keep] = V_k.T
        return P

    def _means(self, gain: _LatentSweepGain, y_anom: np.ndarray,
               z_b: np.ndarray) -> np.ndarray:
        pred_b = self.compressor.decode(z_b.reshape(1, -1))[0].ravel()[gain.gather]
        q = gain.U.T @ (gain.rinv_sqrt * (y_anom - pred_b))              # (m,)
        return np.stack([z_b + k * (gain.G @ (q / (k * gain.Lam + 1.0)))
                         for k in gain.b_scales])                        # (n_b, d)


class TangentLinearLatentVar(LinearLatentVar):
    """Latent 3DVar for a neural decoder, linearised to its tangent about ``z_clim``.

    The mean solve is the affine closed form of :class:`LinearLatentVar` applied to the
    autograd Jacobian ``P``; the innovation still uses the true nonlinear decode at the
    background, so this is a single Gauss-Newton step from the climatological mean.
    """

    def __init__(self, compressor, B_z, z_clim, shape, safe_valid, *,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        compressor.to(self.device)
        super().__init__(compressor, B_z, z_clim, shape, safe_valid)

    def _pushforward(self) -> np.ndarray:
        z0 = torch.as_tensor(self.z_clim, device=self.device, dtype=torch.float32)

        def flat_decode(z):
            return self.compressor.decode_torch(z.reshape(1, -1)).reshape(-1)

        J = torch.autograd.functional.jacobian(flat_decode, z0)         # (D, d)
        return J.detach().cpu().numpy().astype(np.float64)


def latent_var(compressor: Compressor, B_z: np.ndarray, z_clim: np.ndarray,
               shape: tuple[int, int, int], safe_valid: np.ndarray,
               *, device: str | None = None) -> Method:
    """Build the latent estimator matching the compressor (affine vs neural tangent)."""
    if compressor.linear:
        return LinearLatentVar(compressor, B_z, z_clim, shape, safe_valid)
    return TangentLinearLatentVar(compressor, B_z, z_clim, shape, safe_valid, device=device)
