# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
False alarms and precision per score threshold on WIDER val (approximate, see diagnostics.precision_by_score).

**No faces but model detected. It's hurt the AP.**
**Model give score to all detections boxes. We, use a threshold to filter out low score detetions. like theshold>0.5" 
**Two model AP is equal. But, by precision_by_score analysis we get the best models.**

usage:
python -m analysis.custom.precision_by_score --tags cfg_mnet_L0 cfg_mnet_loop_nossh_L1 cfg_mnet_loop_nossh_L4

i) dets: Total number of detections
ii) hits: Number of true positives(matches with ground truth)
iii) false: Number of false positives(don't match with ground truth)
iv) precision: True positives / (True positives + False positives)
v) false/img: Average false alarms per image.
"""

import os
import argparse

from analysis.core.diagnostics import load_wider_val, precision_by_score

p = argparse.ArgumentParser(description='Precision and false alarms per score threshold')
p.add_argument('--tags', nargs='+', required=True)
p.add_argument('--pred_root', default='widerface_evaluate/widerface_txt')
p.add_argument('--gt', default='widerface_evaluate/ground_truth/wider_face_val.mat')
p.add_argument('--thrs', nargs='+', type=float, default=[0.02, 0.3, 0.5])
p.add_argument('--iou_thr', type=float, default=0.5)
p.add_argument('--out', default='outputs/precision_by_score.csv')
args = p.parse_args()

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
new = not os.path.exists(args.out)
with open(args.out, 'a') as f:
    if new:
        f.write('tag,score_thr,n_det,hits,false_alarms,precision,false_alarms_per_image,iou_thr\n')
    for tag in args.tags:
        gts, dets = load_wider_val(args.gt, os.path.join(args.pred_root, tag))
        print(f'\n{tag}')
        print(f"{'thr':>6} {'dets':>8} {'hits':>8} {'false':>8} {'precision':>10} {'false/img':>10}")
        for thr, n, h, fa, prec, fpi in precision_by_score(gts, dets, args.thrs, args.iou_thr):
            print(f'{thr:6.2f} {n:8d} {h:8d} {fa:8d} {prec:10.3f} {fpi:10.2f}')
            f.write(f'{tag},{thr},{n},{h},{fa},{prec:.5f},{fpi:.4f},{args.iou_thr}\n')
print(f'\nappended to {args.out}')