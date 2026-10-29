#
# Source: https://github.com/biubug6/Pytorch_Retinaface (detect.py)
# Copyright (c) 2019 Pytorch_Retinaface authors. MIT License, see LICENSE.MIT
# Modified by Yasin Arafat, 2026: --image/--out arguments, LoopFaceNet with --config
#   and --loop, strict checkpoint loading, accepts training checkpoints (last.pth)
#
from __future__ import print_function
import time
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import config as cfgs
from layers.prior_box import PriorBox
from models.loopface_net import LoopFaceNet
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description='RetinaFace / LoopFace single-image detection')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth', type=str)
parser.add_argument('--config', default='', help='cfg name, e.g. cfg_mnet_haar_loop (overrides --network)')
parser.add_argument('--network', default='mobile0.25', help='mobile0.25 or resnet50 (used when --config is empty)')
parser.add_argument('--loop', default=0, type=int, help='loop count L at test time (0 = cfg loop_test_L)')
parser.add_argument('--cpu', action='store_true', default=False, help='use CPU')
parser.add_argument('--confidence_threshold', default=0.02, type=float)
parser.add_argument('--top_k', default=5000, type=int)
parser.add_argument('--nms_threshold', default=0.4, type=float)
parser.add_argument('--keep_top_k', default=750, type=int)
parser.add_argument('--vis_thres', default=0.6, type=float, help='only draw boxes above this score')
parser.add_argument('--image', default='test.jpg', type=str, help='image to run on')
parser.add_argument('--out', default='result.jpg', type=str, help='where to save the drawn image')
args = parser.parse_args()


def build_net(args):
    if args.config:
        cfg = dict(getattr(cfgs, args.config))
    else:
        cfg = dict(cfgs.cfg_mnet if args.network == 'mobile0.25' else cfgs.cfg_re50)
    cfg['pretrain'] = False                                  # weights come from the checkpoint
    net = LoopFaceNet(cfg=cfg, phase='test')

    sd = torch.load(args.trained_model, map_location='cpu', weights_only=False)
    if 'model' in sd:                                        # our training checkpoint (last.pth)
        sd = sd['model']
    elif 'state_dict' in sd:                                 # old style checkpoint
        sd = sd['state_dict']
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)                     # wrong config -> loud error, not garbage
    print(f'Loaded {args.trained_model}  haar={cfg.get("haar")}  loop={cfg.get("loop")}')
    return net, cfg


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda')
    cudnn.benchmark = True

    img_raw = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_raw is None:
        raise FileNotFoundError(f'Could not read image: {args.image}')

    net, cfg = build_net(args)
    net = net.to(device).eval()
    L = args.loop or None                                    # None -> cfg loop_test_L

    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    img -= (104, 117, 123)
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    tic = time.time()
    loc, conf, landms = net(img, L=L)
    print(f'net forward time: {time.time() - tic:.4f}s  (L={L or cfg.get("loop_test_L") if cfg.get("loop") else "-"})')

    priors = PriorBox(cfg, image_size=(im_height, im_width)).forward().to(device)
    boxes = decode(loc.squeeze(0), priors, cfg['variance']) * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).cpu().numpy()[:, 1]
    landms = decode_landm(landms.squeeze(0), priors, cfg['variance'])
    scale1 = torch.Tensor([im_width, im_height] * 5).to(device)
    landms = (landms * scale1).cpu().numpy()

    # ignore low scores, keep top-K, NMS
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes, landms, scores = boxes[inds], landms[inds], scores[inds]
    order = scores.argsort()[::-1][:args.top_k]
    boxes, landms, scores = boxes[order], landms[order], scores[order]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets, landms = dets[keep][:args.keep_top_k], landms[keep][:args.keep_top_k]
    dets = np.concatenate((dets, landms), axis=1)

    # draw
    n = 0
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        n += 1
        text = f'{b[4]:.4f}'
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.putText(img_raw, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        for (x, y), c in zip(zip(b[5::2], b[6::2]),
                             [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.circle(img_raw, (x, y), 1, c, 4)
    cv2.imwrite(args.out, img_raw)
    print(f'{n} faces above {args.vis_thres}  ->  {args.out}')

# Examples (run from the project root):
# C:\Users\CSITOCT\Desktop\object_detection\
# python detect.py -m weights/mobilenet0.25_Final.pth --network mobile0.25 --cpu --image dataset\val\images\14--Traffic\14_Traffic_Traffic_14_253.jpg --out retina.jpg
# python detect.py -m runs/cfg_mnet_s0/epoch_20.pth --config cfg_mnet --cpu --image <img>
# python detect.py -m runs/cfg_mnet_haar_loop_s0/epoch_80.pth --config cfg_mnet_haar_loop --loop 8 --cpu --image dataset\val\images\2--Demonstration\2_Demonstration_Demonstration_Or_Protest_2_225.jpg --out loop8.jpg


