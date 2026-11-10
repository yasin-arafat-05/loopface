# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
What kind of false alarms does a model make? (WIDER val, approximate, see core.det_types)

Every detection above a score threshold gets one type:
  tp  : matches a face (IoU >= 0.5) that has no box yet           -> a hit
  dup : matches a face (IoU >= 0.5) that already has a better box -> duplicate box on the same face
  loc : best IoU with a face is 0.1 to 0.5                        -> a face is there, the box is badly placed
  bg  : best IoU with every face < 0.1                            -> background, no face nearby
(some bg boxes are real faces that WIDER did not label)

**Shows which kind of mistake a change removes, e.g. does a higher loop count L remove background
false alarms, duplicates, or badly placed boxes?**

usage:
python -m analysis.custom.false_alarm_types --tags cfg_mnet_loop_nossh_L1_seed0 cfg_mnet_loop_nossh_L4_seed0
python -m analysis.custom.false_alarm_types --tags cfg_mnet_L0_seed0 --score_thrs 0.3 0.5

Columns (per 100 images, so they are easy to compare):
i)   tp, dup, loc, bg : number of boxes of each type per 100 images
ii)  false            : dup + loc + bg per 100 images
iii) tp_count         : total hits (should match 'hits' of precision_by_score)
CSV columns: tag, score_thr, iou_thr, n_img, tp, dup, loc, bg (counts), and the same per 100 images.
"""

import os
import argparse

import numpy as np

from analysis.core.diagnostics import load_wider_val
from analysis.core.det_type import classify_detections, TYPE_NAMES

p = argparse.ArgumentParser(description='False alarm types on WIDER val')
p.add_argument('--tags', nargs='+', required=True, help='sub-folders of widerface_evaluate/widerface_txt/')
p.add_argument('--pred_root', default='widerface_evaluate/widerface_txt')
p.add_argument('--gt', default='widerface_evaluate/ground_truth/wider_face_val.mat')
p.add_argument('--score_thrs', type=float, nargs='+', default=[0.3, 0.5])
p.add_argument('--iou_thr', type=float, default=0.5)
p.add_argument('--loc_lo', type=float, default=0.1, help='IoU above this (and below iou_thr) counts as a badly placed box')
p.add_argument('--out', default='outputs/false_alarm_types.csv')
args = p.parse_args()

rows = []
for tag in args.tags:
    gts, dets = load_wider_val(args.gt, os.path.join(args.pred_root, tag))
    # greedy matching goes from the highest score down, so matching once at the lowest threshold
    # gives the same types for the boxes above every higher threshold
    scores, types, n_img = classify_detections(gts, dets, args.iou_thr, args.loc_lo, min(args.score_thrs))
    if len(scores) and (scores.min() < 0 or scores.max() > 1):
        print(f'WARNING {tag}: scores outside [0, 1] ({scores.min():.3f} .. {scores.max():.3f}); is column 5 the score?')
    print(f'\n{tag}  ({n_img} images)')
    print(f"{'thr':>6} {'tp/100':>8} {'dup/100':>8} {'loc/100':>8} {'bg/100':>8} {'false/100':>10} {'tp_count':>9}")
    for thr in sorted(args.score_thrs):
        keep = scores >= thr
        counts = np.bincount(types[keep], minlength=4)
        per100 = counts / n_img * 100
        false100 = per100[1:].sum()
        print(f'{thr:6.2f} {per100[0]:8.1f} {per100[1]:8.1f} {per100[2]:8.1f} {per100[3]:8.1f} {false100:10.1f} {counts[0]:9d}')
        rows.append((tag, thr, args.iou_thr, n_img, *counts, *per100))

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
new = not os.path.exists(args.out)
with open(args.out, 'a') as f:
    if new:
        f.write('tag,score_thr,iou_thr,n_img,' + ','.join(TYPE_NAMES) + ','
                + ','.join(f'{t}_per100' for t in TYPE_NAMES) + '\n')
    for r in rows:
        tag, thr, iou, n, *vals = r
        f.write(f'{tag},{thr},{iou},{n},' + ','.join(str(int(v)) for v in vals[:4]) + ','
                + ','.join(f'{v:.3f}' for v in vals[4:]) + '\n')
print(f'\nappended to {args.out}')
