# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""
Parameters, multiply-accumulates (MACs) and latency per config and loop count L.

Random weights are enough: cost does not depend on the weight values. Latency = median of --runs
forward passes after --warmup passes, batch 1, one image of --size x --size, no file I/O
(unlike ms_per_img in results.csv, which also times writing the prediction files).

**Use it for Table 1 (params, GMACs) and for the AP-vs-compute figure: one loop model covers a
range of compute budgets by changing L at test time.**

usage:
python -m analysis.custom.benchmark --configs cfg_mnet cfg_mnet_nossh cfg_mnet_loop_nossh --loops 1 2 4 8 16 --device cpu --threads 1
python -m analysis.custom.benchmark --configs cfg_mnet cfg_mnet_loop_nossh --loops 1 4 16 --device cuda

CSV (--out): config, L, params, gmac, ms_median, ms_p90, device, threads, size
(L = 0 for configs without a test-time loop)
"""

import os
import time
import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

from data import config as C
from models.loopface_net import LoopFaceNet

p = argparse.ArgumentParser(description='Params, MACs and latency')
p.add_argument('--configs', nargs='+', required=True)
p.add_argument('--loops', type=int, nargs='+', default=[1, 2, 4, 8, 16])
p.add_argument('--size', type=int, default=640)
p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
p.add_argument('--threads', type=int, default=1, help='CPU threads (1 = single-core, like a small device)')
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--runs', type=int, default=50)
p.add_argument('--out', default='outputs/benchmark.csv')
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.backends.cudnn.benchmark = False
dev = torch.device(args.device)


def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


rows = []
print(f"{'config':<32} {'L':>3} {'params':>9} {'GMACs':>7} {'ms_med':>8} {'ms_p90':>8}")
for name in args.configs:
    cfg = dict(getattr(C, name), pretrain=False)          # no pretrained download; cost is the same
    net = LoopFaceNet(cfg).eval().to(dev)
    params = sum(t.numel() for t in net.parameters())
    has_loop = bool(cfg.get('loop') or cfg.get('pre_loop'))
    x = torch.randn(1, 3, args.size, args.size, device=dev)
    for L in (args.loops if has_loop else [0]):
        kw = {'L': L} if has_loop else {}
        with torch.inference_mode():
            with FlopCounterMode(display=False) as fc:
                net(x, **kw)
            gmac = fc.get_total_flops() / 2 / 1e9                # FLOPs = 2 x MACs
            for _ in range(args.warmup):
                net(x, **kw)
            sync()
            times = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                net(x, **kw)
                sync()
                times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        med, p90 = times[len(times) // 2], times[int(len(times) * 0.9)]
        print(f'{name:<32} {L:>3} {params:>9,} {gmac:7.3f} {med:8.2f} {p90:8.2f}')
        rows.append((name, L, params, gmac, med, p90))

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
new = not os.path.exists(args.out)
with open(args.out, 'a') as f:
    if new:
        f.write('config,L,params,gmac,ms_median,ms_p90,device,threads,size\n')
    for name, L, params, gmac, med, p90 in rows:
        f.write(f'{name},{L},{params},{gmac:.4f},{med:.3f},{p90:.3f},{args.device},{args.threads},{args.size}\n')
print(f'\nappended to {args.out}')
