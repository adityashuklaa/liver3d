"""Reference 3D U-Net: encoder, bottleneck, decoder, skip connections.

Two padded 3x3x3 convolutions per level, instance normalisation (small 3D
batches make batch statistics unreliable), leaky ReLU, and a final 1x1x1
convolution to the liver logit.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind: str, channels: int) -> nn.Module:
    kind = (kind or "instance").lower()
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


def _act(kind: str) -> nn.Module:
    kind = (kind or "leakyrelu").lower()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    raise ValueError(f"unknown activation: {kind}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str, act: str, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(norm, out_ch),
            _act(act),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(norm, out_ch),
            _act(act),
        ]
        if dropout and dropout > 0:
            layers.insert(3, nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net for binary liver segmentation (one logit channel)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        norm: str = "instance",
        activation: str = "leakyrelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [int(c) for c in channels]
        if len(channels) < 2:
            raise ValueError("channels needs at least two levels")
        self.channels = channels
        self.depth = len(channels) - 1

        self.encoders = nn.ModuleList()
        prev = in_channels
        for width in channels[:-1]:
            self.encoders.append(ConvBlock(prev, width, norm, activation, dropout))
            prev = width
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(prev, channels[-1], norm, activation, dropout)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for width in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose3d(prev, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width, norm, activation, dropout))
            prev = width
        self.head = nn.Conv3d(prev, out_channels, kernel_size=1)

    @property
    def size_divisor(self) -> int:
        return 2**self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)


def build_model(cfg) -> UNet3D:
    model_cfg = cfg["model"]
    architecture = model_cfg.get("architecture", "unet3d")
    if architecture != "unet3d":
        raise ValueError(f"unsupported architecture: {architecture}")
    return UNet3D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        channels=model_cfg.get("channels", [32, 64, 128, 256]),
        norm=model_cfg.get("norm", "instance"),
        activation=model_cfg.get("activation", "leakyrelu"),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
