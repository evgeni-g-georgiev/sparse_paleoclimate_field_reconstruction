"""Convolutional autoencoder for the Prior cube.

Architecture
------------
The AE compresses ``(in_channels, H, W)`` images to a ``latent_dim``-vector
and reconstructs ``(out_channels, H, W)``. Default channel order:

* input  channels = (mtco_z, mtwa_z, valid_mask)
* output channels = (mtco_z, mtwa_z), the mask is *conditioning input*,
  not part of what the AE is asked to reconstruct.

Encoder pipeline ``(B, 3, 32, 64) -> (B, latent_dim)``:

  Conv stride-2 stack:
      (3, 32, 64) -> (32, 16, 32) -> (64, 8, 16) -> (128, 4, 8)
  Flatten -> Linear(4096 -> latent_dim).

Decoder pipeline ``(B, latent_dim) -> (B, 2, 32, 64)``:

  Linear(latent_dim -> 4096) -> reshape to (128, 4, 8).
  Upsample(2x) + 3x3 conv stack:
      (128, 4, 8) -> (64, 8, 16) -> (32, 16, 32) -> (16, 32, 64)
  Final 3x3 conv -> (2, 32, 64). No activation, no norm.

All convolutions use **circular padding on longitude** (W axis) and zero
padding on latitude (H axis), via the :class:`CircularLonPad2d` module
below. This respects the fact that Earth wraps at ±180° but the poles
are not adjacent to each other.

Design choices flagged in code comments
---------------------------------------
* GroupNorm instead of BatchNorm: robust to batch-size variation, and
  the standard choice in modern diffusion-adjacent models (this encoder
  is intended for reuse).
* SiLU instead of ReLU: smoother gradients, modern default.
* Upsample+Conv instead of ConvTranspose: avoids checkerboard artefacts.
* No skip connections: bottleneck must be a true compression so the
  latent is interpretable for the Bousquet POD-vs-latent probe.
* No activation on the final layer: output is in z-score units
  (~N(0, 1)), so no range constraint is needed (or correct).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Custom padding: circular on longitude, zero on latitude.
# ---------------------------------------------------------------------------
class CircularLonPad2d(nn.Module):
    """Pad spatially: circular on width (longitude), zero on height (latitude).
    """

    def __init__(self, padding: int):
        super().__init__()
        if padding < 0:
            raise ValueError(f"padding must be non-negative; got {padding}")
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order is (W_left, W_right, H_top, H_bottom).
        p = self.padding
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, p, p), mode="constant", value=0.0)
        return x


# ---------------------------------------------------------------------------
# Building blocks shared by encoder and decoder.
# ---------------------------------------------------------------------------
def _gn_groups(num_channels: int) -> int:
    """Pick a GroupNorm group count: 8 if it divides cleanly, else 4, else 1.

    GroupNorm requires that ``num_channels % num_groups == 0``. For the
    channel counts we use (16, 32, 64, 128) every option divides cleanly
    into 8 groups; the fallbacks exist so ``base_channels`` can be tweaked
    without immediately breaking the model.
    """
    if num_channels % 8 == 0:
        return 8
    if num_channels % 4 == 0:
        return 4
    return 1


class ConvBlock(nn.Module):
    """3x3 conv with circular-lon padding, GroupNorm, SiLU activation.

    Used as the encoder's downsample step when ``stride=2``, or as a
    same-size refinement when ``stride=1``.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.pad = CircularLonPad2d(padding=1)
        # padding=0 on the conv; spatial pad is handled by `self.pad`.
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=0
        )
        self.norm = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(self.pad(x))))


class UpBlock(nn.Module):
    """2x nearest-neighbour upsample followed by 3x3 conv + GN + SiLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.pad = CircularLonPad2d(padding=1)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=0
        )
        self.norm = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(self.pad(self.up(x)))))


# ---------------------------------------------------------------------------
# Convolutional autoencoder.
# ---------------------------------------------------------------------------
class ConvAE(nn.Module):
    """Convolutional autoencoder with circular-longitude padding.

    Parameters
    ----------
    latent_dim : int
        Size of the bottleneck vector. Parametric to support the v1
        latent-dim sweep over ``{2, 4, 8, 16, 32, 64}``.
    in_channels : int, default 3
        Number of input channels. Default matches the dataset's
        ``(mtco_z, mtwa_z, valid_mask)`` layout.
    out_channels : int, default 2
        Number of channels reconstructed by the decoder. The mask is not
        reconstructed - it is conditioning input only - so the default
        is 2, matching the temperature channels.
    base_channels : int, default 32
        Channel multiplier. Encoder doubles per stage:
        ``base, 2*base, 4*base``. Decoder mirrors, then halves once more
        before the final conv.
    grid_shape : tuple[int, int], default (32, 64)
        ``(H, W)``. Currently locked to the LOVECLIM 5.625° grid; the
        architecture assumes three stride-2 downsamples are possible, so
        both axes must be divisible by 8.

    Attributes
    ----------
    bottleneck_shape : tuple[int, int, int]
        The pre-flatten encoder output shape ``(4*base, H/8, W/8)``.
        Exposed so the training loop / Bousquet probe / latent diffusion
        head can introspect.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
        grid_shape: tuple[int, int] = (32, 64),
    ):
        super().__init__()
        H, W = grid_shape
        if H % 8 or W % 8:
            raise ValueError(
                f"grid_shape={grid_shape} must be divisible by 8 on both "
                "axes (three stride-2 downsamples)."
            )
        if base_channels % 2:
            # We halve base_channels in the last UpBlock, so it needs to be
            # divisible by 2. (Default base_channels=32 obviously satisfies this.)
            raise ValueError(
                f"base_channels={base_channels} must be even (last UpBlock "
                "outputs base_channels // 2)."
            )

        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels
        H_low, W_low = H // 8, W // 8
        flat_dim = c3 * H_low * W_low

        # Public attributes .
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_shape = grid_shape
        self.bottleneck_shape: tuple[int, int, int] = (c3, H_low, W_low)

        # ---- Encoder ------------------------------------------------------
        self.encoder_conv = nn.Sequential(
            ConvBlock(in_channels, c1, stride=2),   # (H, W)     -> (H/2, W/2)
            ConvBlock(c1, c2, stride=2),            # (H/2, W/2) -> (H/4, W/4)
            ConvBlock(c2, c3, stride=2),            # (H/4, W/4) -> (H/8, W/8)
        )
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)

        # ---- Decoder ------------------------------------------------------
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            UpBlock(c3, c2),                        # (H/8, W/8) -> (H/4, W/4)
            UpBlock(c2, c1),                        # (H/4, W/4) -> (H/2, W/2)
            UpBlock(c1, c1 // 2),                   # (H/2, W/2) -> (H,   W)
        )
        # Final conv collapses to ``out_channels``. Bias enabled so the
        # output can have a non-zero mean. No activation and no norm: the
        # target lives in z-score units (~N(0, 1)), so any squashing
        # non-linearity would clip or bias the output.
        self.out_pad = CircularLonPad2d(padding=1)
        self.out_conv = nn.Conv2d(
            c1 // 2, out_channels, kernel_size=3, stride=1, padding=0
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, latent_dim)``.

        Exposed separately so the Bousquet probe and downstream latent
        diffusion can call the encoder without running the decoder.
        """
        h = self.encoder_conv(x)
        h = h.flatten(start_dim=1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``(B, latent_dim)`` to ``(B, out_channels, H, W)``.

        Exposed separately for latent traversals and for the downstream
        unconditional latent diffusion sampler.
        """
        h = self.decoder_fc(z)
        h = h.view(-1, *self.bottleneck_shape)
        h = self.decoder_conv(h)
        x_hat = self.out_conv(self.out_pad(h))
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, z)``.

        Returning both saves a redundant encode pass when the training
        loop wants to log latent statistics (norm, sparsity) alongside
        the reconstruction loss.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Convenience: parameter count, for the latent-dim sweep notebook.
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
