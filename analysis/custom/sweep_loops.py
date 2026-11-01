# Copyright (c) 2026 Yasin Arafat
# Licensed under the MIT License, see LICENSE
"""Run test_widerface.py for several loop counts L (and optional image scales). Every run writes its
prediction folder widerface_evaluate/widerface_txt/<tag>/ and appends one row to results.csv.
Tags already in results.csv are skipped (use --redo to run them again).

Run from the project root:
  python -m analysis.custom.sweep_loops -m runs/cfg_mnet_loop_nossh_s1/final.pth --config cfg_mnet_loop_nossh --loops 1 4
  python -m analysis.custom.sweep_loops -m runs/cfg_mnet_loop_nossh_s0/final.pth --config cfg_mnet_loop_nossh --loops 1 4 --suffix _seed0
  python -m analysis.custom.sweep_loops -m runs/cfg_mnet_s0/final.pth --config cfg_mnet --scales 0.5 0.75 1.25 1.5
"""
import os
import re
import csv
import sys
import argparse
import subprocess

from data import config as cfgs

p = argparse.ArgumentParser(description='Loop / resolution sweep on WIDER val')
p.add_argument('-m', '--trained_model', required=True)
p.add_argument('--config', required=True)
p.add_argument('--loops', type=int, nargs='*', default=[1, 2, 4, 6, 8, 12, 16])
p.add_argument('--scales', type=float, nargs='*', default=[1.0],
               help='needs the --scale option in test_widerface.py for values other than 1.0')
p.add_argument('--suffix', default=None,
               help='added to every tag; default: _seedK taken from the run folder name (nothing for seed 0)')
p.add_argument('--results', default='results.csv')
p.add_argument('--cpu', action='store_true')
p.add_argument('--redo', action='store_true', help='run again even if the tag is already in the results file')
args = p.parse_args()

# seed suffix from the run folder name, e.g. runs/cfg_mnet_loop_nossh_s7/final.pth -> _seed7
if args.suffix is None:
    m = re.search(r'_s(\d+)$', os.path.basename(os.path.dirname(os.path.abspath(args.trained_model))))
    args.suffix = f'_seed{m.group(1)}' if m and m.group(1) != '0' else ''

has_loop = getattr(cfgs, args.config).get('loop', False)
loops = args.loops if has_loop else [0]          # L means nothing without a loop: run once


def done_tags():
    if not os.path.exists(args.results):
        return set()
    with open(args.results, newline='') as f:
        return {r['tag'] for r in csv.DictReader(f)}


n_run = 0
for s in args.scales:
    for L in loops:
        tag = f'{args.config}_L{L}' + (f'_s{s:g}' if s != 1.0 else '') + args.suffix
        if tag in done_tags() and not args.redo:
            print(f'skip {tag} (already in {args.results})')
            continue
        cmd = [sys.executable, '-u', 'test_widerface.py', '-m', args.trained_model,
               '--config', args.config, '--tag', tag, '--results', args.results]
        if has_loop:
            cmd += ['--loop', str(L)]
        if s != 1.0:
            cmd += ['--scale', str(s)]
        if args.cpu:
            cmd.append('--cpu')
        print('\n>>> ' + ' '.join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f'FAILED: {tag}  (fix the error above, then run the same command again; finished tags are skipped)')
        n_run += 1

print(f'\nDone: {n_run} new run(s). See {args.results}')