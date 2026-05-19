"""Convolutional building blocks and autoencoders for the Prior grid.

The blocks (``CircularLonPad2d``, ``ConvBlock``, ``UpBlock``) all use
**circular padding on longitude** (W axis) and zero padding on latitude
(H axis), respecting that Earth wraps at ±180° but the poles do not
neighbour each other. They can be composed into arbitrary convolutional
encoder/decoder stacks.

:class:`ConvAE` is one such assembly: see its docstring for shape
contracts and design choices.
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

    GroupNorm requires ``num_channels % num_groups == 0``. The default
    channel counts (16, 32, 64, 128) all divide cleanly into 8; the
    fallbacks keep the model alive if ``base_channels`` is tweaked.
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

    Architecture
    ------------
    The encoder is ``depth`` stride-2 ConvBlocks doubling channels per
    stage (``base, 2*base, ..., 2**(depth-1) * base``), followed by a
    flatten and a linear projection to ``latent_dim``. The decoder mirrors
    the encoder via UpBlocks; the last halves to ``base // 2`` before a
    final 3x3 conv to ``out_channels``. No activation or norm on the
    final layer; output is in z-score units (~N(0, 1)).

    Default IO channels: input = (mtco_z, mtwa_z, valid_mask); output =
    (mtco_z, mtwa_z). The mask is conditioning input, not reconstructed.

    Design choices
    --------------
    * GroupNorm, not BatchNorm: robust to batch-size variation.
    * SiLU, not ReLU: smoother gradients.
    * Upsample+Conv, not ConvTranspose: avoids checkerboard artefacts.
    * No skip connections: the bottleneck must be a true compression so
      the latent stays interpretable for downstream probes.

    Constraints
    -----------
    ``H`` and ``W`` must be divisible by ``2 ** depth``. ``base_channels``
    must be even (the last UpBlock halves it). ``depth >= 1``.

    Attributes
    ----------
    bottleneck_shape : ``(base * 2**(depth-1), H // 2**depth, W // 2**depth)``.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 3,
        grid_shape: tuple[int, int] = (32, 64),
    ):
        super().__init__()
        H, W = grid_shape
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        factor = 2 ** depth
        if H % factor or W % factor:
            raise ValueError(
                f"grid_shape={grid_shape} must be divisible by 2**depth="
                f"{factor} on both axes."
            )
        if base_channels % 2:
            raise ValueError(
                f"base_channels={base_channels} must be even (last UpBlock "
                "outputs base_channels // 2)."
            )

        enc_channels = [base_channels * (2 ** i) for i in range(depth)]
        c_top = enc_channels[-1]
        H_low, W_low = H // factor, W // factor
        flat_dim = c_top * H_low * W_low

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.grid_shape = grid_shape
        self.bottleneck_shape: tuple[int, int, int] = (c_top, H_low, W_low)

        # ---- Encoder ------------------------------------------------------
        enc_pairs = [(in_channels, enc_channels[0])] + [
            (enc_channels[i - 1], enc_channels[i]) for i in range(1, depth)
        ]
        self.encoder_conv = nn.Sequential(
            *[ConvBlock(c_in, c_out, stride=2) for c_in, c_out in enc_pairs]
        )
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)

        # ---- Decoder ------------------------------------------------------
        # Mirror the encoder; the last UpBlock halves base_channels.
        dec_pairs = [
            (enc_channels[i], enc_channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ] + [(base_channels, base_channels // 2)]
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            *[UpBlock(c_in, c_out) for c_in, c_out in dec_pairs]
        )
        # Final conv collapses to ``out_channels``. No activation and no
        # norm: the target lives in z-score units (~N(0, 1)), so any
        # squashing non-linearity would clip or bias the output.
        self.out_pad = CircularLonPad2d(padding=1)
        self.out_conv = nn.Conv2d(
            base_channels // 2, out_channels, kernel_size=3, stride=1, padding=0
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H, W)`` to ``(B, latent_dim)``.

        Exposed separately so callers can run the encoder without the decoder.
        """
        h = self.encoder_conv(x)
        h = h.flatten(start_dim=1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``(B, latent_dim)`` to ``(B, out_channels, H, W)``.

        Exposed separately for latent traversals and decoder-only sampling.
        """
        h = self.decoder_fc(z)
        h = h.view(-1, *self.bottleneck_shape)
        h = self.decoder_conv(h)
        x_hat = self.out_conv(self.out_pad(h))
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, z)``.

        Returning both saves a redundant encode pass when the caller wants
        to inspect the latent alongside the reconstruction.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Convenience: parameter count.
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
