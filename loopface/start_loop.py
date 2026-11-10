# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""Pre-backbone loop after MobileNet0.25's first stride-2 conv: x = [B, 8, 320, 320] for a 640 input.
cell='plain' : h_t = B(h_{t-1} + x)
cell='gru'   : z, r = sigmoid(conv[x, h]);  h~ = tanh(conv[x, r*h]);  h_t = (1 - z) * h + z * h~
Output: x + gamma * h_L, gamma = 0 at start.
"""
import random

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, k, padding=k // 2)   # z and r
        self.cand = nn.Conv2d(2 * channels, channels, k, padding=k // 2)        # h~

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, dim=1)
        h_cand = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * h_cand


class PreLoop(nn.Module):
    def __init__(self, channels, train_L=(1, 6), test_L=4, cell='plain', gn_groups=4):
        super().__init__()
        self.cell_type = cell
        if cell == 'gru':
            self.cell = ConvGRUCell(channels)
        elif cell == 'plain':
            self.cell = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                      nn.GroupNorm(gn_groups, channels), nn.PReLU())
        else:
            raise ValueError(f'unknown pre_loop cell: {cell}')
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.train_L, self.test_L = tuple(train_L), test_L
        self.L = None

    def forward(self, x):
        L = self.L or (random.randint(*self.train_L) if self.training else self.test_L)
        h = torch.zeros_like(x)                                   # h0 = 0
        for _ in range(L):
            h = self.cell(x, h) if self.cell_type == 'gru' else self.cell(h + x)
        return x + self.gamma * h                                 # loop exit