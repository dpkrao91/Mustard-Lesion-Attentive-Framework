"""Convolutional Block Attention Module (CBAM, Woo et al., ECCV 2018).

CBAM is the lesion-focusing component of PALM. Field-captured mustard images
contain soil, weeds and neighbouring leaves; CBAM's channel + spatial gating
lets the network suppress this background clutter and concentrate capacity on
the diseased lamina. It is lightweight (a few thousand parameters) so it does
not compromise the edge-deployability goal.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        attn = self.mlp(avg) + self.mlp(mx)
        attn = torch.sigmoid(attn).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x
