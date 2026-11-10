# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""Detection types for diagnosis: every detection becomes a hit (tp), a duplicate (dup),
a badly placed box (loc) or background (bg). Depends only on numpy and scipy."""

import numpy as np
from scipy.stats import rankdata

TYPE_NAMES = ('tp', 'dup', 'loc', 'bg')   # 0 true positive, 1 duplicate, 2 badly placed box, 3 background


def _to_np(x, cols):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, cols))
    if a.ndim == 1:
        a = a[None]
    return a


def _box_iou_np(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _pairs(gts, dets):
    if isinstance(gts, dict):
        return [(gts[k], dets.get(k, [])) for k in gts]
    return list(zip(gts, dets))


def classify_detections(gts, dets, iou_thr=0.5, loc_lo=0.1, min_score=0.0):
    """Greedy matching per image, highest score first (as in WIDER evaluation), and a type for every
    detection with score >= min_score:
      tp  : IoU >= iou_thr with a face that has no box yet
      dup : IoU >= iou_thr with a face, but that face already has a higher-scored box
      loc : best IoU with any face in [loc_lo, iou_thr)  -> a face is there, the box is badly placed
      bg  : best IoU < loc_lo                             -> no face nearby
    Detections are [x1, y1, x2, y2, score]; faces are [x1, y1, x2, y2].
    Returns scores, types (int8, index into TYPE_NAMES), number of images."""
    pairs = _pairs(gts, dets)
    all_s, all_t = [], []
    for gt, det in pairs:
        gt = _to_np(gt, 4)[:, :4]
        det = _to_np(det, 5)
        det = det[det[:, 4] >= min_score]
        if len(det) == 0:
            continue
        det = det[np.argsort(-det[:, 4], kind='stable')]
        types = np.full(len(det), 3, dtype=np.int8)
        if len(gt):
            iou = _box_iou_np(det[:, :4], gt)
            best_any = iou.max(1)
            taken = np.zeros(len(gt), dtype=bool)
            for i in range(len(det)):
                row = np.where(taken, -1.0, iou[i])
                j = int(row.argmax())
                if row[j] >= iou_thr:
                    taken[j] = True
                    types[i] = 0
                elif best_any[i] >= iou_thr:
                    types[i] = 1
                elif best_any[i] >= loc_lo:
                    types[i] = 2
        all_s.append(det[:, 4])
        all_t.append(types)
    if not all_s:
        return np.zeros(0), np.zeros(0, dtype=np.int8), len(pairs)
    return np.concatenate(all_s), np.concatenate(all_t), len(pairs)


def score_auroc(scores, is_tp):
    """Probability that a random true positive has a higher score than a random false alarm (0.5 = no separation)."""
    n1, n0 = int(is_tp.sum()), int((~is_tp).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(scores)
    return float((r[is_tp].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))