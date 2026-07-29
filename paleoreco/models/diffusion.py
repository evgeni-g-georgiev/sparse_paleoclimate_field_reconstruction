"""Score-based denoiser for the Prior grid and its EDM preconditioning.

A U-Net that denoises the ``(2, n_lat, n_lon)`` anomaly field conditioned on the
noise level, wrapped by the EDM preconditioning of Karras et al. 2022 (Table 1,
"Ours") into a denoiser ``D(x; sigma)`` whose score is ``(D(x; sigma) - x) /
sigma^2`` (Tweedie). The convolutional blocks reuse the circular-longitude,
zero-latitude padding of :mod:`paleoreco.models.autoencoder` so the field wraps
at +-180 degrees while the poles do not.

The network operates on 2 channels with no mask input; masked cells are held at
zero by the training loss and the sampler, so the denoiser never learns to move
them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from paleoreco.models.autoencoder import CircularLonPad2d, _gn_groups

# EDM defaults (Karras 2022, Table 1, "Ours"), stated for sigma_data-scaled data.
SIGMA_DATA = 0.5
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
P_MEAN = -1.2
P_STD = 1.2


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of a per-sample scalar, ``(B,) -> (B, dim)``."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=values.device) / half)
    args = values[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class _ResBlock(nn.Module):
    """Two circular-padded convs with GroupNorm/SiLU and FiLM noise conditioning.

    The noise embedding enters as a per-channel scale and shift (FiLM) between the
    two convs, so the block's behaviour varies smoothly with the noise level.
    """

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.pad = CircularLonPad2d(padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.norm1 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=0)
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.film = nn.Linear(emb_dim, 2 * out_channels)
        self.act = nn.SiLU()
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(self.pad(x))))
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(self.norm2(self.conv2(self.pad(h))))
        return h + self.skip(x)


class _Down(nn.Module):
    """Conditioned ResBlock, then a stride-2 circular conv that halves both axes."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.block = _ResBlock(in_channels, out_channels, emb_dim)
        self.pad = CircularLonPad2d(padding=1)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x, emb)                 # skip taken before downsampling
        return self.down(self.pad(skip)), skip


class _Up(nn.Module):
    """Nearest-neighbour upsample, concatenate the matching skip, conditioned ResBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = _ResBlock(in_channels + skip_channels, out_channels, emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([self.up(x), skip], dim=1), emb)


class CircularUNet(nn.Module):
    """Noise-conditioned U-Net over the anomaly grid.

    ``depth`` stride-2 stages double the channels from ``base_channels``; the
    decoder mirrors them with skip connections. ``grid_shape`` axes must be
    divisible by ``2 ** depth``. The forward takes the (already ``c_in``-scaled)
    field and the EDM ``c_noise`` scalar per sample.
    """

    def __init__(self, *, in_channels: int = 2, out_channels: int = 2,
                 base_channels: int = 64, depth: int = 2, emb_dim: int = 128,
                 grid_shape: tuple[int, int] = (32, 64)):
        super().__init__()
        h, w = grid_shape
        factor = 2 ** depth
        if h % factor or w % factor:
            raise ValueError(f"grid_shape={grid_shape} must be divisible by 2**depth={factor}")
        self.config = {"in_channels": in_channels, "out_channels": out_channels,
                       "base_channels": base_channels, "depth": depth,
                       "emb_dim": emb_dim, "grid_shape": grid_shape}
        self.emb_dim = emb_dim

        self.map_noise = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                       nn.Linear(emb_dim, emb_dim))
        chs = [base_channels * (2 ** i) for i in range(depth)]

        self.in_pad = CircularLonPad2d(padding=1)
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=0)

        self.downs = nn.ModuleList()
        c = chs[0]
        skip_chs = []
        for out in chs:
            self.downs.append(_Down(c, out, emb_dim))
            skip_chs.append(out)
            c = out

        self.mid = _ResBlock(c, c, emb_dim)

        self.ups = nn.ModuleList()
        for out, skip_c in zip(reversed(chs), reversed(skip_chs)):
            self.ups.append(_Up(c, skip_c, out, emb_dim))
            c = out

        self.out_norm = nn.GroupNorm(_gn_groups(c), c)
        self.out_act = nn.SiLU()
        self.out_pad = CircularLonPad2d(padding=1)
        self.out_conv = nn.Conv2d(c, out_channels, 3, padding=0)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(_sinusoidal_embedding(c_noise, self.emb_dim))
        h = self.in_conv(self.in_pad(x))
        skips = []
        for down in self.downs:
            h, skip = down(h, emb)
            skips.append(skip)
        h = self.mid(h, emb)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, emb)
        h = self.out_act(self.out_norm(h))
        return self.out_conv(self.out_pad(h))


class EDMDenoiser(nn.Module):
    """EDM preconditioning wrapper turning a raw net into a denoiser ``D(x; sigma)``.

    ``D = c_skip x + c_out F(c_in x; c_noise)`` with the Karras 2022 coefficients,
    so the wrapped network always sees unit-variance inputs across noise levels.
    ``sigma`` is a scalar or a per-sample ``(B,)`` tensor.
    """

    def __init__(self, net: CircularUNet, sigma_data: float = SIGMA_DATA):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.config = dict(net.config)

    def _broadcast(self, sigma: torch.Tensor, batch: int, device, dtype) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=device, dtype=dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        return sigma

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = self._broadcast(sigma, x.shape[0], x.device, x.dtype)
        sd = self.sigma_data
        s = sigma[:, None, None, None]
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / torch.sqrt(s ** 2 + sd ** 2)
        c_in = 1.0 / torch.sqrt(s ** 2 + sd ** 2)
        c_noise = 0.25 * torch.log(sigma)
        return c_skip * x + c_out * self.net(c_in * x, c_noise)

    def score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Score ``grad log p(x; sigma) = (D(x; sigma) - x) / sigma^2`` (Tweedie)."""
        sigma = self._broadcast(sigma, x.shape[0], x.device, x.dtype)
        return (self.forward(x, sigma) - x) / sigma[:, None, None, None] ** 2


def load_denoiser(path: str, map_location="cpu") -> tuple[EDMDenoiser, list[float]]:
    """Rebuild a denoiser and its per-channel anomaly scales from a checkpoint.

    The scales come back as the stored list, which is what the sampler's normalised
    frame is defined against; a denoiser loaded without them cannot be used.
    """
    ckpt = torch.load(path, map_location=map_location)
    net = EDMDenoiser(CircularUNet(**ckpt["config"]), ckpt["sigma_data"])
    net.load_state_dict(ckpt["state_dict"])
    return net, ckpt["channel_scales"]
