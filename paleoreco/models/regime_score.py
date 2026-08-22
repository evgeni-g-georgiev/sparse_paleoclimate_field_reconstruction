"""The regime mixture as an EDM denoiser, and the learned residual that sits on it.

A diffusion model is defined by its denoiser ``D(x; sigma) = E[x_0 | x_sigma]``, from which
Tweedie gives the score. A Gaussian mixture convolved with Gaussian noise is still a
Gaussian mixture, so that expectation is closed form at every noise level:

    p_sigma(x) = sum_j w_j N(x; m_j, k C_j + sigma^2 I)
    D(x; sigma) = sum_j r_j(x, sigma) mu_j(x, sigma),
        mu_j = m_j + k C_j (k C_j + sigma^2 I)^-1 (x - m_j)

Each component's eigendecomposition is taken once when the mixture is built, so every noise
level is a diagonal reweighting in a fixed basis and nothing is inverted here. A closed-form
score also has no band where it is untrained, so the question of whether a noise schedule
covers the field's spectrum arises only for the residual below.

:class:`RegimeDenoiser` adds a network to the mixture under EDM preconditioning,

    D(x; sigma) = D_mixture(x; sigma) + t c_out(sigma) F(c_in(sigma) x; c_noise(sigma))

with ``F``'s output layer zero-initialised, so an untrained network or ``t = 0`` reproduces
the mixture denoiser identically. Training is ordinary denoising score matching against the
same clean states; since the mixture term is fixed and known, the network learns the
correction to it. That is the residual-from-the-mean decomposition applied to the score.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from paleoreco.models.autoencoder import CircularLonPad2d  # noqa: F401  (re-export path)
from paleoreco.models.diffusion import CircularUNet, SIGMA_DATA


class MixtureDenoiser(nn.Module):
    """Exact EDM denoiser of ``sum_j w_j N(m_j, b_scale C_j)``.

    Each component's eigenbasis is a buffer, so ``forward`` is two batched matrix products
    per component and a softmax. ``b_scale`` enters through the eigenvalues, so one built
    object serves any amplitude through :meth:`at_scale`. ``sigma_data`` sets the EDM
    preconditioning only; the mixture's own scale comes from its covariances.
    """

    def __init__(self, mixture, b_scale: float = 1.0, sigma_data: float = SIGMA_DATA):
        super().__init__()
        self.mixture = mixture                       # plain attribute: never in state_dict
        self.b_scale = float(b_scale)
        self.sigma_data = float(sigma_data)
        self.shape = tuple(mixture.shape)
        self.n_regimes = mixture.n_regimes
        self.register_buffer("w_log", torch.log(torch.as_tensor(mixture.w, dtype=torch.float32)))
        self.register_buffer("mu", torch.as_tensor(mixture.mu, dtype=torch.float32))
        self.register_buffer("U", torch.as_tensor(mixture.U, dtype=torch.float32))
        self.register_buffer("lam", torch.as_tensor(mixture.lam, dtype=torch.float32))
        self.register_buffer(
            "mask", torch.as_tensor(np.broadcast_to(mixture.safe_valid, mixture.shape)
                                    .astype(np.float32).reshape(-1)))

    def at_scale(self, b_scale: float) -> "MixtureDenoiser":
        """The same mixture at another background amplitude; buffers are shared.

        A shallow copy keeps the eigenbases, the largest tensors here, on the device they
        are already on rather than re-uploading them once per amplitude.
        """
        other = copy.copy(self)
        other.b_scale = float(b_scale)
        return other

    def _sigma_vec(self, sigma, n: int, like: torch.Tensor) -> torch.Tensor:
        """``sigma`` as an ``(n, 1)`` tensor, whether it arrived as a scalar or per sample."""
        s = torch.as_tensor(sigma, device=like.device, dtype=like.dtype).reshape(-1)
        if s.numel() == 1:
            s = s.expand(n)
        return s[:, None]

    def _parts(self, x: torch.Tensor, sigma):
        """Per-component log-density, conditional mean, and the shrinkage factors.

        ``sigma`` is a scalar or one value per sample: the reverse process passes a scalar,
        the EDM training loop a vector. Returns ``(logp, mu_j, f)`` shaped ``(n, J)``,
        ``(n, J, D)`` and ``(J, n, D)``, with ``f`` the ``k lam / (k lam + sigma^2)``
        shrinkage each component applies.

        The projections stay in the working dtype, but the log-density is accumulated in
        double: it sums ``D`` terms spanning the whole eigenvalue range, where single
        precision loses accuracy the well-conditioned matrix products do not.
        """
        s2 = self._sigma_vec(sigma, x.shape[0], x) ** 2               # (n, 1)
        kl = self.b_scale * self.lam                                  # (J, D)

        logp, mus, fs = [], [], []
        for j in range(self.n_regimes):
            denom = kl[j][None, :] + s2                               # (n, D)
            f = kl[j][None, :] / denom
            q = (x - self.mu[j][None, :]) @ self.U[j]                 # (n, D) in basis j
            acc = ((q.double() ** 2) / denom.double()).sum(dim=1) + torch.log(
                denom.double()).sum(dim=1)
            logp.append(self.w_log[j].double() - 0.5 * acc)
            mus.append(self.mu[j][None, :] + (q * f) @ self.U[j].T)
            fs.append(f)
        return torch.stack(logp, dim=1), torch.stack(mus, dim=1), torch.stack(fs)

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        logp, mus, _ = self._parts(flat, sigma)
        r = torch.softmax(logp, dim=1).to(mus.dtype)                  # (n, J)
        out = (r[:, :, None] * mus).sum(dim=1)                        # (n, D)
        return (out * self.mask[None, :]).reshape(x.shape)

    def within_cov_obs_blocks(self, sigma: float, gather: torch.Tensor) -> torch.Tensor:
        """``H Cov(x_0 | x_sigma, regime j) H^T`` for each component, ``(J, m, m)``.

        For a Gaussian component this is ``U_j diag(k lam sigma^2 / (k lam + sigma^2))
        U_j^T``, which does not depend on the state, so one ``(m, m)`` factorisation per
        component per noise level serves the whole batch. ``sigma`` must be scalar; every
        sample in the reverse process sits at the same noise level.
        """
        s = np.asarray(sigma.detach().cpu() if torch.is_tensor(sigma) else sigma).reshape(-1)
        if s.size != 1:
            raise ValueError("within_cov_obs_blocks needs one noise level for the batch")
        s2 = float(s[0]) ** 2
        kl = self.b_scale * self.lam                                  # (J, D)
        out = []
        for j in range(self.n_regimes):
            Ug = self.U[j][gather]                                    # (m, D)
            f = kl[j] * s2 / (kl[j] + s2)                             # (D,)
            out.append((Ug * f[None, :]) @ Ug.T)
        return torch.stack(out)

    def obs_predictive(self, x: torch.Tensor, sigma, gather: torch.Tensor):
        """Pieces of the observation predictive ``p(y | x_sigma)``.

        Since ``p(x_0 | x_sigma)`` is itself a Gaussian mixture, so is the predictive:

            p(y | x_sigma) = sum_j r_j(x) N(y; H mu_j(x), H Gamma_j H^T + R)

        Returns the state-dependent pieces, ``log r_j`` ``(n, J)`` and ``H mu_j``
        ``(n, J, m)``; the covariance blocks come from :meth:`within_cov_obs_blocks`.
        """
        flat = x.reshape(x.shape[0], -1)
        logp, mus, _ = self._parts(flat, sigma)
        return torch.log_softmax(logp, dim=1).to(mus.dtype), mus[:, :, gather]


class RegimeDenoiser(nn.Module):
    """The mixture denoiser plus a tempered, zero-initialised learned residual.

    ``temper`` decides how much of the network is used; ``0`` makes this exactly
    :class:`MixtureDenoiser`. The mixture is a plain attribute rather than a submodule, so
    ``state_dict`` carries only the network and the mixture is rebuilt from the prior
    states on load.
    """

    def __init__(self, mixture_denoiser: MixtureDenoiser, net: CircularUNet,
                 temper: float = 1.0):
        super().__init__()
        object.__setattr__(self, "_mix", mixture_denoiser)   # not a submodule
        self.net = net
        self.temper = float(temper)
        self.sigma_data = float(mixture_denoiser.sigma_data)
        self.config = dict(net.config)

    @property
    def mixture(self) -> MixtureDenoiser:
        return self._mix

    def at_temper(self, temper: float) -> "RegimeDenoiser":
        """The same trained network at another temper; weights are shared."""
        return RegimeDenoiser(self._mix, self.net, temper)

    def _apply(self, *args, **kwargs):
        """Carry the mixture along on every device and dtype move.

        It is not a submodule, so ``to``, ``double`` and ``cuda`` would leave it behind.
        ``_apply`` is the one hook they all route through.
        """
        self._mix._apply(*args, **kwargs)
        return super()._apply(*args, **kwargs)

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        base = self._mix(x, sigma)
        if self.temper == 0.0:
            return base
        sig = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1)
        if sig.numel() == 1:
            sig = sig.expand(x.shape[0])
        s = sig[:, None, None, None]
        sd = self.sigma_data
        c_out = s * sd / torch.sqrt(s ** 2 + sd ** 2)
        c_in = 1.0 / torch.sqrt(s ** 2 + sd ** 2)
        return base + self.temper * c_out * self.net(c_in * x, 0.25 * torch.log(sig))


def build_regime_denoiser(mixture, net_config: dict | None = None, *, b_scale: float = 1.0,
                          temper: float = 1.0, sigma_data: float = SIGMA_DATA,
                          grid_shape: tuple[int, int] | None = None) -> RegimeDenoiser:
    """A residual denoiser over ``mixture``, ready to train or sample.

    The network's output layer is zeroed, so before any training this object is the
    mixture denoiser exactly.
    """
    cfg = dict(net_config or {})
    cfg.setdefault("grid_shape", tuple(grid_shape or mixture.shape[1:]))
    cfg.setdefault("in_channels", mixture.shape[0])
    cfg.setdefault("out_channels", mixture.shape[0])
    net = CircularUNet(**cfg)
    nn.init.zeros_(net.out_conv.weight)
    nn.init.zeros_(net.out_conv.bias)
    return RegimeDenoiser(MixtureDenoiser(mixture, b_scale=b_scale, sigma_data=sigma_data),
                          net, temper=temper)


def load_regime_denoiser(path: str, mixture, *, b_scale: float = 1.0, temper: float = 1.0,
                         map_location="cpu") -> tuple[RegimeDenoiser, dict]:
    """Rebuild a residual denoiser against ``mixture``; return it and the full payload.

    The mixture is not in the checkpoint, so the caller rebuilds it and should check the
    provenance scalars in the payload first: a network trained against another partition,
    shrinkage or band-pass window would otherwise load silently against the wrong prior.
    """
    ckpt = torch.load(path, map_location=map_location)
    model = build_regime_denoiser(mixture, ckpt["config"], b_scale=b_scale, temper=temper,
                                  sigma_data=ckpt.get("sigma_data", SIGMA_DATA))
    model.net.load_state_dict({k[len("net."):]: v for k, v in ckpt["state_dict"].items()
                               if k.startswith("net.")})
    return model, ckpt
