#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    if channels >= 8:
        return 8
    if channels >= 4:
        return 4
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Return:
        feat: feature map for skip connection
        down: downsampled feature map
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        down = self.down(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class MinimalMelAutoencoder(nn.Module):
    """
    Minimal mel autoencoder.
    Input : [B, 1, n_mels, frames]
    Output: [B, 1, n_mels, frames]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        bottleneck_channels: int = 128,
    ) -> None:
        super().__init__()

        self.stem = ConvBlock(in_channels, base_channels)

        self.enc1 = DownBlock(base_channels, base_channels * 2)
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, bottleneck_channels),
            ConvBlock(bottleneck_channels, base_channels * 4),
        )

        self.dec1 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 2,
        )
        self.dec0 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels * 2,
            out_channels=base_channels,
        )

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, n_mels, frames]
        """
        x0 = self.stem(x)          # [B, C, H, W]
        skip1, x1 = self.enc1(x0)  # skip1: [B, 2C, H, W], x1: [B, 2C, H/2, W/2]
        skip2, x2 = self.enc2(x1)  # skip2: [B, 4C, H/2, W/2], x2: [B, 4C, H/4, W/4]

        z = self.bottleneck(x2)

        y1 = self.dec1(z, skip2)   # [B, 2C, H/2, W/2]
        y0 = self.dec0(y1, skip1)  # [B, C, H, W]

        out = self.head(y0)

        # Double safety: force exact shape match
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.1,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    loss = l1 + mse_weight * mse
    loss_dict = {
        "loss_total": loss.detach(),
        "loss_l1": l1.detach(),
        "loss_mse": mse.detach(),
    }
    return loss, loss_dict


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MinimalMelAutoencoder(in_channels=1, base_channels=32, bottleneck_channels=128)
    x = torch.randn(4, 1, 64, 313)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
    print("params:", count_parameters(model))