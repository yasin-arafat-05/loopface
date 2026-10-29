# NumPy port of box_overlaps.pyx so no Cython build is needed.
# Original: Fast R-CNN, Copyright (c) 2015 Microsoft, MIT License, written by Sergey Karayev
# Ported by Yasin Arafat, 2026 (same +1 pixel rule, same output)
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """boxes (N, 4), query_boxes (K, 4) as x1, y1, x2, y2 -> IoU matrix (N, K)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    iw = (np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) -
          np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1)
    ih = (np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) -
          np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1)
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    area_b = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    area_q = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    ua = area_b[:, None] + area_q[None, :] - inter
    return np.divide(inter, ua, out=np.zeros_like(inter), where=inter > 0)