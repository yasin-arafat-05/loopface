# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
Precision-recall curve on WIDER val (all faces, approximate, see core.det_types).

All detections with score >= min_score are sorted by score; walking down the list gives one
(recall, precision) point per detection. The curve is the standard picture behind AP:
a model whose curve lies higher has fewer false alarms at the same recall.

**Use it to show where L4 beats L1: same recall, higher precision.**
(WIDER's official AP ignores some faces per subset; this curve counts every face, so its area
is not the official AP, but the comparison between models is fair.)

usage:
python -m analysis.custom.pr_curve --tags cfg_mnet_loop_nossh_L1_seed0 cfg_mnet_loop_nossh_L4_seed0 cfg_mnet_nossh_L0_seed0

printed table: precision at recall 0.40, 0.50, 0.60, 0.65 and the maximum recall reached
CSV (--out): tag, recall, precision   (interpolated precision at recall 0.00, 0.01, ..., max)
"""

import os
import argparse

import numpy as np

from analysis.core.diagnostics import load_wider_val
from analysis.core.det_type import classify_detections, _pairs, _to_np

p = argparse.ArgumentParser(description='Precision-recall curve on WIDER val')
p.add_argument('--tags', nargs='+', required=True, help='sub-folders of widerface_evaluate/widerface_txt/')
p.add_argument('--pred_root', default='widerface_evaluate/widerface_txt')
p.add_argument('--gt', default='widerface_evaluate/ground_truth/wider_face_val.mat')
p.add_argument('--min_score', type=float, default=0.02)
p.add_argument('--iou_thr', type=float, default=0.5)
p.add_argument('--out', default='outputs/pr_curve.csv')
args = p.parse_args()

levels = np.round(np.arange(0.0, 1.0001, 0.01), 2)
show = [0.40, 0.50, 0.60, 0.65]
rows = []
print(f"{'tag':<40} " + ' '.join(f'P@R{r:.2f}' for r in show) + '   max_recall')
for tag in args.tags:
    gts, dets = load_wider_val(args.gt, os.path.join(args.pred_root, tag))
    n_gt = sum(len(_to_np(g, 4)) for g, _ in _pairs(gts, dets))
    scores, types, _ = classify_detections(gts, dets, args.iou_thr, 0.1, args.min_score)
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(types[order] == 0)
    recall = tp / max(n_gt, 1)
    precision = tp / np.arange(1, len(tp) + 1)
    # interpolated precision: best precision at this recall or higher (as in VOC / WIDER AP)
    interp = np.maximum.accumulate(precision[::-1])[::-1]
    max_r = recall[-1] if len(recall) else 0.0
    curve = []
    for lv in levels:
        if lv > max_r:
            break
        i = int(np.searchsorted(recall, lv, side='left'))
        curve.append((lv, float(interp[i])))
    pr = dict(curve)
    print(f'{tag:<40} ' + ' '.join(f"{pr.get(r, float('nan')):8.3f}" for r in show) + f'   {max_r:.3f}')
    rows += [(tag, r, pv) for r, pv in curve]

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
new = not os.path.exists(args.out)
with open(args.out, 'a') as f:
    if new:
        f.write('tag,recall,precision\n')
    for tag, r, pv in rows:
        f.write(f'{tag},{r:.2f},{pv:.5f}\n')
print(f'\nappended to {args.out}')

