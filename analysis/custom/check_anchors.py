# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
Anchor coverage of the training set, measured BEFORE training (no model needed).
Uses the same dataset + augmentation as train.py, so it measures the faces the model really sees.
=============================x======================================
python -m analysis.custom.check_anchors --config cfg_mnet --every 5

i) --every 5: form a subset of total 5 images: we'll take one out of every 5 images.
ii) GT -> Ground Truth boxes, in xyxy format, normalized to 0-1.
iii) ignored -> No positive anchor can be assigned to these GT boxes. 
iv) forced -> Only one positive anchor can be assigned but forcefully.
v) natural -> One or more positive anchors can be assigned naturally.

"""

import random
import argparse

import torch
import numpy as np

from data import config as cfgs
from layers.prior_box import PriorBox
from data import WiderFaceDetection, preproc
from analysis.core.diagnostics import anchor_coverage, print_coverage

p = argparse.ArgumentParser(description='Anchor coverage per face-size bin')
p.add_argument('--config', default='cfg_mnet')
p.add_argument('--training_dataset', default='./dataset/train/label.txt')
p.add_argument('--every', type=int, default=5, help='use every k-th image (1 = all, slower)')
p.add_argument('--seed', type=int, default=0, help='random crops differ per run; fix the seed to repeat')
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
cfg = getattr(cfgs, args.config)
size = cfg['image_size']
priors = PriorBox(cfg, image_size=(size, size)).forward() # (P, 4) cx, cy, w, h in 0-1
ds = WiderFaceDetection(args.training_dataset, preproc(size, (104, 117, 123)))

gts = []
for i in range(0, len(ds), args.every):
    _, target = ds[i]   # target: (N, 15), boxes xyxy in 0-1
    gts.append(torch.as_tensor(target[:, :4], dtype=torch.float32))

print(f'{args.config}: {len(gts)} images, {sum(len(g) for g in gts)} faces, '
      f'{priors.shape[0]} anchors, min_sizes={cfg["min_sizes"]}, image_size={size}')
print_coverage(anchor_coverage(gts, priors, img_size=size))

