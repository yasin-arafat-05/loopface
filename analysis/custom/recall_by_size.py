# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
Recall by face size on WIDER val (see diagnostics.recall_by_size).

Faces are grouped by size in pixels. For each group we count how many faces the model finds,
i.e. a detection with score >= score_thr overlaps the face with IoU >= iou_thr.

**Shows which face sizes a model misses (usually the tiny ones).**
**Together with precision_by_score: does a model gain AP by finding more faces, or by fewer false alarms?**

usage:
python -m analysis.custom.recall_by_size --tags cfg_mnet_L0_seed0 cfg_mnet_loop_nossh_L1_seed0 cfg_mnet_loop_nossh_L4_seed0
python -m analysis.custom.recall_by_size --tags cfg_mnet_L0_seed0 --score_thr 0.5

arguments:
--tags       one or more sub-folders of --pred_root (the prediction txt files of each model)
--score_thr  only detections with score >= this count (default 0.3)
--iou_thr    IoU needed to count a face as found (default 0.5)
--out        CSV file the results are appended to (default outputs/recall_by_size.csv)

printed table (one row per size bin, one column per tag):
i)   size(px): face size bin in pixels
ii)  GT:       number of ground-truth faces in the bin
iii) <tag>:    recall of that model in the bin = faces found / GT

CSV columns: tag, size_lo, size_hi, n_gt, recall, score_thr, iou_thr
Each run appends rows, so a tag run twice appears twice; drop duplicates before comparing.
"""

import os
import argparse

from analysis.core.diagnostics import load_wider_val, recall_by_size, DEFAULT_BINS, _edge

p = argparse.ArgumentParser(description='Recall by face size on WIDER val')
p.add_argument('--tags', nargs='+', required=True, help='sub-folders of widerface_evaluate/widerface_txt/')
p.add_argument('--pred_root', default='widerface_evaluate/widerface_txt')
p.add_argument('--gt', default='widerface_evaluate/ground_truth/wider_face_val.mat')
p.add_argument('--score_thr', type=float, default=0.3)
p.add_argument('--iou_thr', type=float, default=0.5)
p.add_argument('--out', default='outputs/recall_by_size.csv')
args = p.parse_args()

results = {}
for tag in args.tags:
    gts, dets = load_wider_val(args.gt, os.path.join(args.pred_root, tag))
    results[tag] = recall_by_size(gts, dets, DEFAULT_BINS, args.score_thr, args.iou_thr)

# side-by-side table: one row per size bin, one column per tag
bins = [(lo, hi, n) for lo, hi, n, _ in next(iter(results.values()))]
print(f"{'size(px)':>12} {'GT':>7} " + ' '.join(f'{t[-14:]:>15}' for t in args.tags))
for k, (lo, hi, n) in enumerate(bins):
    print(f'{lo:>5.0f}-{_edge(hi):<6} {n:7d} ' + ' '.join(f'{results[t][k][3]:15.3f}' for t in args.tags))

new = not os.path.exists(args.out)
with open(args.out, 'a') as f:
    if new:
        f.write('tag,size_lo,size_hi,n_gt,recall,score_thr,iou_thr\n')
    for t in args.tags:
        for lo, hi, n, r in results[t]:
            f.write(f'{t},{lo},{_edge(hi)},{n},{r:.5f},{args.score_thr},{args.iou_thr}\n')
print(f'appended to {args.out}')
