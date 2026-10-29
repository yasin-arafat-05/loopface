#
# Source: https://github.com/biubug6/Pytorch_Retinaface (test_widerface.py)
# Copyright (c) 2019 Pytorch_Retinaface authors. MIT License, see LICENSE.MIT
# Modified by Yasin Arafat, 2026: LoopFaceNet with --config and --loop, strict loading,
#   one output folder per (config, L), runs evaluation.py automatically, appends to results.csv
#
from __future__ import print_function
import os
import re
import sys
import time
import argparse
import subprocess

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import config as cfgs
from layers.prior_box import PriorBox
from models.loopface_net import LoopFaceNet
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description='WIDER FACE val: predict + evaluate')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth', type=str)
parser.add_argument('--config', default='', help='cfg name, e.g. cfg_mnet_haar_loop (overrides --network)')
parser.add_argument('--network', default='mobile0.25', help='mobile0.25 or resnet50 (when --config is empty)')
parser.add_argument('--loop', default=0, type=int, help='loop count L (0 = cfg loop_test_L)')
parser.add_argument('--resize', action='store_true', default=False,
                    help='MXNet-style test scale (short side 1600, long side <= 2150); default = original size')
parser.add_argument('--dataset_folder', default='./dataset/val/images/', type=str)
parser.add_argument('--val_list', default='./dataset/val/wider_val.txt', type=str)
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str)
parser.add_argument('--tag', default='', help='output sub-folder name (default: <config>_L<L>)')
parser.add_argument('--results', default='results.csv', help='AP table, one row per run')
parser.add_argument('--eval_only', action='store_true', help='skip inference, only evaluate existing txt files')
parser.add_argument('--no_eval', action='store_true', help='only write txt files, do not evaluate')
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--confidence_threshold', default=0.02, type=float)
parser.add_argument('--top_k', default=5000, type=int)
parser.add_argument('--nms_threshold', default=0.4, type=float)
parser.add_argument('--keep_top_k', default=750, type=int)
args = parser.parse_args()


def build_net(args):
    if args.config:
        cfg = dict(getattr(cfgs, args.config))
    else:
        cfg = dict(cfgs.cfg_mnet if args.network == 'mobile0.25' else cfgs.cfg_re50)
    cfg['pretrain'] = False
    net = LoopFaceNet(cfg=cfg, phase='test')
    sd = torch.load(args.trained_model, map_location='cpu', weights_only=False)
    if 'model' in sd:
        sd = sd['model']
    elif 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net, cfg


def predict(net, cfg, L, out_root, device):
    """Writes one txt per val image. Returns mean forward time in ms (warm-up excluded)."""
    names = [l.strip().lstrip('/') for l in open(args.val_list) if l.strip()]
    t_fwd, n_timed = 0.0, 0
    t0 = time.time()
    for i, name in enumerate(names):
        img_raw = cv2.imread(os.path.join(args.dataset_folder, name), cv2.IMREAD_COLOR)
        if img_raw is None:
            raise FileNotFoundError(os.path.join(args.dataset_folder, name))
        img = np.float32(img_raw)

        resize = 1.0
        if args.resize:
            im_min, im_max = min(img.shape[:2]), max(img.shape[:2])
            resize = 1600 / im_min
            if np.round(resize * im_max) > 2150:
                resize = 2150 / im_max
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        h, w = img.shape[:2]
        scale = torch.Tensor([w, h, w, h]).to(device)
        img -= (104, 117, 123)
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        tic = time.time()
        loc, conf, _ = net(x, L=L)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if i >= 10:                                   # skip warm-up images
            t_fwd += time.time() - tic
            n_timed += 1

        priors = PriorBox(cfg, image_size=(h, w)).forward().to(device)
        boxes = (decode(loc.squeeze(0), priors, cfg['variance']) * scale / resize).cpu().numpy()
        scores = conf.squeeze(0).cpu().numpy()[:, 1]

        inds = np.where(scores > args.confidence_threshold)[0]
        boxes, scores = boxes[inds], scores[inds]
        order = scores.argsort()[::-1][:args.top_k]
        boxes, scores = boxes[order], scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        dets = dets[py_cpu_nms(dets, args.nms_threshold)][:args.keep_top_k]

        save_name = os.path.join(out_root, name[:-4] + '.txt')
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        with open(save_name, 'w') as fd:
            fd.write(os.path.basename(save_name)[:-4] + '\n')
            fd.write(f'{len(dets)}\n')
            for b in dets:
                x1, y1 = int(b[0]), int(b[1])
                fd.write(f'{x1} {y1} {int(b[2]) - x1} {int(b[3]) - y1} {b[4]} \n')

        if (i + 1) % 200 == 0 or i + 1 == len(names):
            ms = 1000 * t_fwd / max(n_timed, 1)
            print(f'{i + 1}/{len(names)}  forward {ms:.1f} ms/img  elapsed {(time.time() - t0) / 60:.1f} min')
    return 1000 * t_fwd / max(n_timed, 1)


def evaluate(tag, out_root, L_used, ms, device):
    """Runs widerface_evaluate/evaluation.py and appends Easy/Medium/Hard AP to results.csv."""
    print(f'\nEvaluating {tag} ...')
    res = subprocess.run([sys.executable, 'evaluation.py', '-p', os.path.abspath(out_root), '-g', './ground_truth/'],
                         cwd='widerface_evaluate', stdout=subprocess.PIPE, text=True)
    print(res.stdout)
    ap = dict(re.findall(r'(Easy|Medium|Hard)\s+Val AP:\s*([0-9.]+)', res.stdout))
    if len(ap) != 3:
        raise RuntimeError('Could not read AP from evaluation.py output (see above).')
    row = [tag, args.trained_model, L_used, ap['Easy'], ap['Medium'], ap['Hard'],
           f'{ms:.1f}' if ms is not None else '', device.type]
    new = not os.path.exists(args.results)
    with open(args.results, 'a') as f:
        if new:
            f.write('tag,model,L,easy,medium,hard,ms_per_img,device\n')
        f.write(','.join(map(str, row)) + '\n')
    print(f'{tag}:  Easy {float(ap["Easy"]):.4f}  Medium {float(ap["Medium"]):.4f}  '
          f'Hard {float(ap["Hard"]):.4f}  ->  appended to {args.results}')


def main():
    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda')
    cudnn.benchmark = False             # every WIDER image has a different size; benchmark would re-tune each time

    net, cfg = build_net(args)
    net = net.to(device).eval()
    L = args.loop or None
    L_used = (L or cfg['loop_test_L']) if cfg.get('loop') else 0
    tag = args.tag or f'{args.config or args.network}_L{L_used}' + ('_resize' if args.resize else '')
    out_root = os.path.join(args.save_folder, tag)
    print(f'model={args.trained_model}  haar={cfg.get("haar")}  loop={cfg.get("loop")}  L={L_used}  -> {out_root}')

    ms = None
    if not args.eval_only:
        ms = predict(net, cfg, L, out_root, device)
        print(f'Done. Mean forward time: {ms:.1f} ms/img ({device.type})')
    if not args.no_eval:
        evaluate(tag, out_root, L_used, ms, device)


if __name__ == '__main__':
    main()