"""
utils/diagnostics.py : model-independent detection diagnostics.

Functions take plain tensors (no model, no cfg), so they work for any
detector in this repo. Boxes are xyxy. GT and detections must be in the
SAME pixel scale (write down which scale you use, e.g. 640 input).
Object size = sqrt(w*h) in pixels (COCO-style small/medium/large idea).
"""

import os
import torch
import collections
from scipy.io import loadmat
from utils.box_utils import iou_tensor, point_form

DEFAULT_BINS = (0, 8, 16, 32, 64, float("inf"))


def box_size(boxes):
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return (w * h).sqrt()


def _bin_index(size, bins):
    edges = torch.tensor(bins[1:-1], dtype=size.dtype)
    return torch.bucketize(size, edges, right=True)          # 0 .. len(bins)-2


def _edge(x):
    return "inf" if x == float("inf") else f"{x:.0f}"


def recall_by_size(gts, dets, bins=DEFAULT_BINS, score_thr=0.3, iou_thr=0.5):
    """gts: list of (N,4); dets: list of (M,5)=[x1,y1,x2,y2,score], one per image.
    Returns [(lo, hi, n_gt, recall)]. Approximate: ignores WIDER's ignore-regions,
    so it is NOT the official AP. Multi-class: call once per class."""
    nb = len(bins) - 1
    hit, tot = torch.zeros(nb), torch.zeros(nb)
    for g, d in zip(gts, dets):
        if len(g) == 0:
            continue
        d = d[d[:, 4] >= score_thr]
        if len(d):
            found = iou_tensor(g, d[:, :4]).max(dim=1).values >= iou_thr
        else:
            found = torch.zeros(len(g), dtype=torch.bool)
        idx = _bin_index(box_size(g), bins)
        for k in range(nb):
            m = idx == k
            tot[k] += m.sum()
            hit[k] += found[m].sum()
    return [(bins[k], bins[k + 1], int(tot[k]), float(hit[k] / tot[k].clamp(min=1)))
            for k in range(nb)]


def anchor_coverage(gts_norm, priors, img_size=640, bins=DEFAULT_BINS,
                    ignore_thr=0.2, match_thr=0.35):
    """
    How well do anchors cover GT? (thresholds follow match() in box_utils)
    gts_norm(Ground Truth): list of (N,4) xyxy in 0-1; priors(Anchors): (P,4) cx,cy,w,h in 0-1.
    Per size bin: % GT never matched (best IoU < ignore_thr), % matched only via
    the single forced anchor, % matched naturally (best IoU >= match_thr).
    """

    pri = point_form(priors)
    nb = len(bins) - 1
    cnt = torch.zeros(nb, 3)
    for g in gts_norm:
        if len(g) == 0:
            continue
        best = iou_tensor(g, pri).max(dim=1).values
        idx = _bin_index(box_size(g * img_size), bins)
        cat = torch.where(best < ignore_thr, 0, torch.where(best < match_thr, 1, 2))
        for k in range(nb):
            for c in range(3):
                cnt[k, c] += ((idx == k) & (cat == c)).sum()
    return [(bins[k], bins[k + 1], int(cnt[k].sum()),
             *(float(v) for v in cnt[k] / cnt[k].sum().clamp(min=1)))
            for k in range(nb)]


@torch.no_grad()
def recurrent_change(model, x, module_name="LoopBlock", **forward_kwargs):
    """Relative change between successive calls of a weight-shared module.
    Load TRAINED weights first. Values near 0 = converged (more iterations won't help).
    Pass the loop count, e.g. recurrent_change(net, x, L=16)."""
    outs = collections.defaultdict(list)
    hooks = [m.register_forward_hook(
                 lambda mod, i, o: outs[tuple(o.shape[-2:])].append(o.detach()))
             for m in model.modules() if m.__class__.__name__ == module_name]
    model.eval()(x, **forward_kwargs)
    for h in hooks:
        h.remove()
    return {size: [float((hs[t] - hs[t - 1]).norm() / (hs[t - 1].norm() + 1e-8))
                   for t in range(1, len(hs))]
            for size, hs in outs.items()}

def print_recall(rows):
    print(f"{'size(px)':>12} {'GT':>7} {'recall':>8}")
    for lo, hi, n, r in rows:
        print(f"{lo:>5.0f}-{_edge(hi):<6} {n:7d} {r:8.3f}")


def print_coverage(rows):
    print(f"{'size(px)':>12} {'GT':>7} {'ignored':>8} {'forced':>8} {'natural':>8}")
    for lo, hi, n, a, b, c in rows:
        print(f"{lo:>5.0f}-{_edge(hi):<6} {n:7d} {a:8.3f} {b:8.3f} {c:8.3f}")


def load_wider_val(gt_mat, pred_dir):
    """GT from widerface_evaluate/ground_truth/wider_face_val.mat and predictions from
    test_widerface.py txt files; both returned as xyxy in ORIGINAL image pixels."""
    m = loadmat(gt_mat)
    gts, dets = [], []
    for i, ev in enumerate(m['event_list']):
        event = ev[0][0]
        for j, f in enumerate(m['file_list'][i][0]):
            name = f[0][0]
            g = torch.tensor(m['face_bbx_list'][i][0][j][0], dtype=torch.float32).reshape(-1, 4)
            g[:, 2:] += g[:, :2]                                     # xywh -> xyxy
            lines = open(os.path.join(pred_dir, event, name + '.txt')).read().split('\n')[2:]
            d = torch.tensor([list(map(float, l.split())) for l in lines if l.strip()],
                             dtype=torch.float32).reshape(-1, 5)
            d[:, 2:4] += d[:, :2] # xywh -> xyxy, score stays
            gts.append(g); dets.append(d)
    return gts, dets


def precision_by_score(gts, dets, score_thrs=(0.02, 0.3, 0.5), iou_thr=0.5):
    """Per score threshold: how many detections overlap a labelled face (IoU >= iou_thr) and how many
    are false alarms. Returns [(thr, n_det, hits, false_alarms, precision, false_alarms_per_image)].
    Approximate: unlabelled or WIDER-ignored faces also count as false alarms; duplicates count as hits."""
    rows = []
    for thr in score_thrs:
        hit = fa = 0
        for g, d in zip(gts, dets):
            d = d[d[:, 4] >= thr]
            if len(d) == 0:
                continue
            if len(g) == 0:
                fa += len(d)
                continue
            ok = iou_tensor(d[:, :4], g).max(dim=1).values >= iou_thr
            hit += int(ok.sum())
            fa += int((~ok).sum())
        rows.append((thr, hit + fa, hit, fa, hit / max(hit + fa, 1), fa / max(len(dets), 1)))
    return rows