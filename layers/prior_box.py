#
# Source: https://github.com/biubug6/Pytorch_Retinaface (layers/prior_box.py)
# Copyright (c) 2019 Pytorch_Retinaface authors. MIT License, see LICENSE.MIT
# Modified by loopface authors. Copyright (c) 2026 Yasin Arafat. MIT License, see LICENSE
#

import torch
import numpy as np
from math import ceil
from itertools import product as product


class PriorBox(object):
    """
    They made anchor boxes on image.
    Output: [P, 4], every row (cx, cy, w, h), for a image: (512×512)  P = 10752
    """
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        # imgae_size ÷ stride
        # 512/8 = 64,  512/16 = 32,  512/32 = 16
        # feature_maps = [[64, 64], [32, 32], [16, 16]]
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        # k = feature map (0, 1, 2),  f = [64, 64]
        for k, f in enumerate(self.feature_maps):
            # k=0, then []
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output