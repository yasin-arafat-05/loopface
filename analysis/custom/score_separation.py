# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
How well do the scores separate hits from false alarms? (WIDER val, approximate, see core.det_types)

All detections with score >= min_score are split into hits (tp) and false alarms (dup + loc + bg).
A good detector gives hits high scores and false alarms low scores.

**AUROC = probability that a random hit has a higher score than a random false alarm
(1.0 = perfectly separated, 0.5 = scores carry no information). If a higher loop count L raises AUROC,
the loop works as a false-alarm filter: it pushes false alarms below the hits.**

usage:
python -m analysis.custom.score_separation --tags cfg_mnet_loop_nossh_L1_seed0 cfg_mnet_loop_nossh_L4_seed0

printed table:
i)   n_tp, n_false : number of hits and false alarms with score >= min_score
ii)  tp_mean       : mean score of the hits
iii) false_mean    : mean score of the false alarms
iv)  auroc         : separation of the two score distributions
outputs:
--out      one row per tag   : tag, min_score, iou_thr, n_tp, n_false, tp_mean, false_mean, auroc
--hist_out score histogram   : tag, bin_lo, bin_hi, n_tp, n_false, n_dup, n_loc, n_bg   (for the figure)
"""

import os
import argparse

import numpy as np
from analysis.core.diagnostics import load_wider_val
from analysis.core.det_type import classify_detections, score_auroc

p = argparse.ArgumentParser(description='Score separation of hits and false alarms on WIDER val')
p.add_argument('--tags', nargs='+', required=True, help='sub-folders of widerface_evaluate/widerface_txt/')
p.add_argument('--pred_root', default='widerface_evaluate/widerface_txt')
p.add_argument('--gt', default='widerface_evaluate/ground_truth/wider_face_val.mat')
p.add_argument('--min_score', type=float, default=0.02, help='the score threshold used for WIDER AP')
p.add_argument('--iou_thr', type=float, default=0.5)
p.add_argument('--loc_lo', type=float, default=0.1)
p.add_argument('--bins', type=int, default=20)
p.add_argument('--out', default='outputs/score_separation.csv')
p.add_argument('--hist_out', default='outputs/score_hist.csv')
args = p.parse_args()

edges = np.linspace(0.0, 1.0, args.bins + 1)
summary, hist = [], []
print(f"{'tag':<40} {'n_tp':>8} {'n_false':>9} {'tp_mean':>8} {'false_mean':>11} {'auroc':>7}")
for tag in args.tags:
    gts, dets = load_wider_val(args.gt, os.path.join(args.pred_root, tag))
    scores, types, n_img = classify_detections(gts, dets, args.iou_thr, args.loc_lo, args.min_score)
    is_tp = types == 0
    auc = score_auroc(scores, is_tp)
    tp_mean = scores[is_tp].mean() if is_tp.any() else float('nan')
    fa_mean = scores[~is_tp].mean() if (~is_tp).any() else float('nan')
    print(f'{tag:<40} {is_tp.sum():8d} {(~is_tp).sum():9d} {tp_mean:8.3f} {fa_mean:11.3f} {auc:7.4f}')
    summary.append((tag, args.min_score, args.iou_thr, int(is_tp.sum()), int((~is_tp).sum()), tp_mean, fa_mean, auc))
    per_type = [np.histogram(scores[types == t], edges)[0] for t in range(4)]
    for b in range(args.bins):
        tp, dup, loc, bg = (int(h[b]) for h in per_type)
        hist.append((tag, edges[b], edges[b + 1], tp, dup + loc + bg, dup, loc, bg))

for path, header, rows, fmt in [
        (args.out, 'tag,min_score,iou_thr,n_tp,n_false,tp_mean,false_mean,auroc',
         summary, lambda r: f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]:.4f},{r[7]:.5f}'),
        (args.hist_out, 'tag,bin_lo,bin_hi,n_tp,n_false,n_dup,n_loc,n_bg',
         hist, lambda r: f'{r[0]},{r[1]:.2f},{r[2]:.2f},' + ','.join(str(v) for v in r[3:]))]:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write(header + '\n')
        for r in rows:
            f.write(fmt(r) + '\n')
print(f'\nappended to {args.out} and {args.hist_out}')