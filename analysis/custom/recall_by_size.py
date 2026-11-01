# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
Recall per face-size bin on WIDER val, from prediction folders written by test_widerface.py.
- **It's determine there are faces but our model give low scores.**
- **It's heart the AP. **
usage: 
python -m analysis.custom.recall_by_size --tags cfg_mnet_L0 cfg_mnet_loop_nossh_L1 cfg_mnet_loop_nossh_L4
python -m analysis.custom.recall_by_size --tags cfg_mnet_L0 cfg_mnet_loop_nossh_L1 cfg_mnet_loop_nossh_L4 --score_thr 0.02
python -m analysis.custom.recall_by_size --tags cfg_mnet_L0 cfg_mnet_loop_nossh_L1 cfg_mnet_loop_nossh_L4 --score_thr 0.5

i) --tags: sub-folders of widerface_evaluate/widerface_txt/ (e.g. cfg_mnet_L0)
ii) --score_thr: only consider detections with score >= score_thr (default 0.3)
iii) Determine which model weak in which size of faces. 
iv) Threshold 0.3/0.5 model is confident. Threshold 0.02 model is weak, but can detect more faces.
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
