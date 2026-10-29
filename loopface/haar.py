#
# LoopFace Author: 
# Copyright (c) 2026 Yasin Arafat Licensed under the MIT License, see LICENSE
#

import torch 
import torch.nn as nn 
import torch.nn.functional as F


def _haar_filters() -> torch.Tensor:
    """The 4 orthonormal 2x2 Haar filters, shape (4, 1, 2, 2)."""
    ll = torch.tensor([[ 1.,  1.], [ 1.,  1.]])   # low-low  : average (blurred image)
    lh = torch.tensor([[ 1.,  1.], [-1., -1.]])   # low-high : top - bottom  -> horizontal edges
    hl = torch.tensor([[ 1., -1.], [ 1., -1.]])   # high-low : left - right  -> vertical edges
    hh = torch.tensor([[ 1., -1.], [-1.,  1.]])   # high-high: diagonal      -> diagonal edges
    return (torch.stack([ll, lh, hl, hh]) * 0.5).unsqueeze(1)   # 0.5 makes it orthonormal


class HaarTransform(nn.Module):
    """Forward Haar DWT: (B, C, H, W) -> (B, 4C, H/2, W/2)"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels * 4
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            #as haar is depthwise conv
            groups=in_channels,
            bias=False,
        )

        with torch.no_grad():
            # weight shape for depthwise conv: (out_channels, in_channels/groups = 1, 2, 2)
            # (4C, 1, 2, 2)
            weight = _haar_filters().repeat(in_channels, 1, 1, 1)
            self.conv.weight.copy_(weight)
        # fixed weights, the optimizer won't change them
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Odd H or W? Pad 1 pixel so stride-2 doesn't drop the last row/column
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        return self.conv(x)

    @staticmethod
    def split_bands(y: torch.Tensor):
        """(B, 4C, H, W) -> LL, LH, HL, HH, each (B, C, H, W)"""
        # conv out: 1,12,2,2:
        # return: 4 * [1,3,2,2]
        b, c4, h, w = y.shape
        y = y.view(b, c4 // 4, 4, h, w)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class InverseHaarTransform(nn.Module):
    """Inverse Haar DWT: (B, 4C, H/2, W/2) -> (B, C, H, W). Exact reconstruction."""
    def __init__(self, in_channels: int):   # in_channels = C of the ORIGINAL image
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels * 4,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
            groups=in_channels,
            bias=False,
        )
        with torch.no_grad():
            self.deconv.weight.copy_(_haar_filters().repeat(in_channels, 1, 1, 1))
        self.deconv.weight.requires_grad_(False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.deconv(y)




