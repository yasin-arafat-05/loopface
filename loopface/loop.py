#
# LoopFace Author: 
# Copyright (c) 2026 Yasin Arafat Licensed under the MIT License, see LICENSE
#


import torch
import random
import torch.nn as nn 
import torch.nn.functional as F


class LoopBlock(nn.Module):
    """3x3 conv -> GroupNorm -> PReLU. Same shape in and out: (B, C, H, W) -> (B, C, H, W)."""
    def __init__(self, channels,act="prelu",gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,3,stride=1,padding=1,bias=False)
        self.norm = nn.GroupNorm(gn_groups,channels)
        self.act = nn.PReLU() if act=='prelu' else nn.ReLU(inplace=True)
    def forward(self,x):
        return self.act(self.norm(self.conv(x)))


class LoopFace(nn.Module):
    """One shared block, repeated L times on each pyramid level, with recall.
    Input : feats = [P3, P4, P5], e.g. [B,64,80,80], [B,64,40,40], [B,64,20,20]
    p = Praymid level feature map. (stride,2^3)
    Output: same list, same shapes.
    # see the model foward for this naming convention.
    """
    def __init__(self,channels=64, train_L=(1, 6), test_L=4, k_grad=None, act="prelu"):
        super().__init__()
        self.block = LoopBlock(channels,act)
        # looping for train and test:
        self.train_L = train_L
        self.test_L = test_L

    def _loop(self,p,L):
        # p from fpn
        h = torch.zeros_like(p)
        for _ in range(L):
            h = self.block(h+p)
        return h 

    def forward(self,feats,L=None):
        if L is None:
            L = random.randint(*self.train_L) if self.training else self.test_L
        return [self._loop(p,L) for p in feats]
