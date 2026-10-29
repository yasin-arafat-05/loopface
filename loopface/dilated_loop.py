#
# LoopFace Author:
# Copyright (c) 2026 Yasin Arafat Licensed under the MIT License, see LICENSE
#
"""Dilated loop: the same 3x3 weights at every step, but the dilation changes with the step
(1, 2, 4, 1, 2, 4, ...), so the receptive field grows faster than with a plain 3x3 loop.
Layers and parameter count are identical to loop.LoopBlock / loop.LoopFace."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedLoopBlock(nn.Module):
    """3x3 conv (dilation d) -> GroupNorm -> PReLU. (B, C, H, W) -> (B, C, H, W) for any d."""

    def __init__(self, channels, act="prelu", gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(gn_groups, channels)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU(inplace=True)

    def forward(self, x, d=1):
        x = F.conv2d(x, self.conv.weight, padding=d, dilation=d)   # padding = d keeps H, W
        return self.act(self.norm(x))


class DilatedLoopFace(nn.Module):
    """Same interface as loop.LoopFace: feats = [P3, P4, P5] -> same list, same shapes.
    Step t uses dilation dilations[t % len(dilations)]."""

    def __init__(self, channels=64, train_L=(1, 6), test_L=4, dilations=(1, 2, 4), act="prelu"):
        super().__init__()
        self.block = DilatedLoopBlock(channels, act)
        self.train_L = tuple(train_L)
        self.test_L = test_L
        self.dilations = tuple(dilations)

    def _loop(self, p, L):
        h = torch.zeros_like(p)
        for t in range(L):
            d = self.dilations[t % len(self.dilations)]
            h = self.block(h + p, d)          # recall: p is injected at every step
        return h

    def forward(self, feats, L=None):
        if L is None:
            L = random.randint(*self.train_L) if self.training else self.test_L
        return [self._loop(p, L) for p in feats]