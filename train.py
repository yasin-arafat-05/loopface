#
# Source: https://github.com/biubug6/Pytorch_Retinaface (train.py)
# Copyright (c) 2019 Pytorch_Retinaface authors. MIT License, see LICENSE.MIT
# Modified by Yasin Arafat, 2026: LoopFaceNet, random L per batch, seeds,
#   per-epoch checkpoint + exact resume, CSV log, Windows-safe main(),
#   --single_batch overfit sanity check
#
from __future__ import print_function
import os
import json
import math
import time
import random
import argparse
import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data import config as cfgs
from data import WiderFaceDetection, detection_collate, preproc
from layers.multibox_loss import MultiBoxLoss
from layers.prior_box import PriorBox
from models.loopface_net import LoopFaceNet


def get_args():
    p = argparse.ArgumentParser(description='LoopFace training')
    p.add_argument('--config', default='cfg_mnet_haar_loop', help='name of a dict in data/config.py')
    p.add_argument('--training_dataset', default='./dataset/train/label.txt',
                   help='label.txt that sits next to the images/ folder')
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--gamma', default=0.1, type=float, help='LR multiplier at decay1 and decay2')
    p.add_argument('--epochs', default=0, type=int, help='override cfg epoch (0 = keep cfg)')
    p.add_argument('--decay1', default=0, type=int, help='override (0 = 76%% of epochs)')
    p.add_argument('--decay2', default=0, type=int, help='override (0 = 88%% of epochs)')
    p.add_argument('--batch_size', default=0, type=int, help='override cfg batch_size (0 = keep cfg)')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--out', default='./runs', help='every run gets its own folder here')
    p.add_argument('--resume', default='', help='checkpoint path; empty = auto-resume from run folder')
    p.add_argument('--print_every', default=50, type=int)
    p.add_argument('--max_iters', default=0, type=int, help='stop after this many iterations (smoke test)')
    p.add_argument('--single_batch', action='store_true',
                   help='overfit sanity check: train on ONE fixed batch repeatedly (no epochs, no resume, '
                        'constant LR). Loss should go close to 0 if the architecture/loss/targets are correct')
    p.add_argument('--sb_iters', default=500, type=int,
                   help='number of iterations in --single_batch mode')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32           # different for every worker, fixed by the generator
    np.random.seed(s)
    random.seed(s)


def lr_at(epoch, cfg, args):
    """Step decay computed from the epoch, so resume always gets the right LR."""
    n = int(epoch >= cfg['decay1']) + int(epoch >= cfg['decay2'])
    return args.lr * (args.gamma ** n)


def rng_state(gen):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader': gen.get_state()}


def set_rng_state(st, gen):
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'].cpu())
    if st['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda']])
    gen.set_state(st['loader'].cpu())


def run_single_batch(args, cfg, net, optimizer, criterion, priors, dataset, bs, device, run_dir):
    """Overfit test: grab ONE batch once (augmentation is frozen because we reuse the same tensors)
    and train on it for args.sb_iters iterations with a constant LR."""
    loader = DataLoader(dataset, bs, shuffle=False, num_workers=0, collate_fn=detection_collate)
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = [a.to(device) for a in targets]
    n_faces = sum(int(t.shape[0]) for t in targets)

    for g in optimizer.param_groups:
        g['lr'] = args.lr

    log_path = os.path.join(run_dir, 'single_batch_train_log.csv')
    log = open(log_path, 'w')                      # always a fresh log in this mode
    log.write('epoch,iter,L,loss_loc,loss_cls,loss_landm,loss_total,lr,sec_per_iter\n')

    print(f'[single_batch] images={images.shape[0]}  faces={n_faces}  iters={args.sb_iters}  '
          f'lr={args.lr:.0e} (constant)  run folder: {run_dir}')

    net.train()
    first_loss = None
    last_it = 0
    t_last = time.time()
    for it in range(args.sb_iters):
        L = random.randint(*cfg['loop_train_L']) if cfg.get('loop') else None
        out = net(images, L=L)

        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        if not torch.isfinite(loss):
            raise RuntimeError(f'Loss is {loss.item()} at single-batch iter {it}; stopping.')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()

        if it % args.print_every == 0 or it == args.sb_iters - 1:
            sec = (time.time() - t_last) / max(1, it - last_it)
            t_last, last_it = time.time(), it
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f'sb it {it}/{args.sb_iters}  L={L}  '
                  f'loc {loss_l.item():.3f}  cls {loss_c.item():.3f}  ldm {loss_landm.item():.3f}  '
                  f'total {loss.item():.3f}  {sec:.2f}s/it  mem {mem:.1f}GB')
            log.write(f'0,{it},{L},{loss_l.item():.5f},{loss_c.item():.5f},'
                      f'{loss_landm.item():.5f},{loss.item():.5f},{args.lr},{sec:.4f}\n')
            log.flush()

    log.close()
    final_loss = loss.item()
    torch.save(net.state_dict(), os.path.join(run_dir, 'single_batch.pth'))
    print(f'[single_batch] total loss: {first_loss:.4f} -> {final_loss:.4f}  '
          f'({final_loss / max(first_loss, 1e-12) * 100:.1f}% of the initial loss)')
    print(f'[single_batch] done. Log: {log_path}')


def main():
    args = get_args()
    cfg = dict(getattr(cfgs, args.config))

    # override cfg epoch, decay1, decay2, batch_size if requested
    if args.epochs:
        cfg['epoch'] = args.epochs
        cfg['decay1'] = args.decay1 or round(0.76 * args.epochs)
        cfg['decay2'] = args.decay2 or round(0.88 * args.epochs)

    if args.batch_size:
        cfg['batch_size'] = args.batch_size

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('No GPU found. MultiBoxLoss uses .cuda() internally, so training needs a GPU.')
    cudnn.benchmark = True

    prefix = 'single_batch_' if args.single_batch else ''
    suffix = '' if args.single_batch else ('_smoke' if args.max_iters else '')
    run_dir = os.path.join(args.out, f'{prefix}{args.config}_s{args.seed}{suffix}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'run.json'), 'w') as f:
        json.dump({'args': vars(args), 'cfg': cfg, 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                   'started': datetime.datetime.now().isoformat(timespec='seconds')},
                  f, indent=2, default=str)

    # model, optimizer, loss, anchors
    net = LoopFaceNet(cfg=cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'config={args.config}  haar={cfg.get("haar")}  loop={cfg.get("loop")}  '
          f'trainable params={n_params:,}  gpu={torch.cuda.get_device_name(0)}')

    optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)
    with torch.no_grad():
        priors = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size'])).forward().to(device)

    dataset = WiderFaceDetection(args.training_dataset, preproc(cfg['image_size'], (104, 117, 123)))
    bs = cfg['batch_size']

    # single-batch overfit mode: skip resume / epochs / checkpoints entirely
    if args.single_batch:
        run_single_batch(args, cfg, net, optimizer, criterion, priors, dataset, bs, device, run_dir)
        return

    epoch_size = math.ceil(len(dataset) / bs)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    # resume: explicit path, or last.pth in the run folder if it exists
    start_epoch = 0
    ckpt_path = args.resume or os.path.join(run_dir, 'last.pth')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        set_rng_state(ck['rng'], gen)
        start_epoch = ck['epoch']
        print(f'Resumed from {ckpt_path} at epoch {start_epoch}')

    log_path = os.path.join(run_dir, 'train_log.csv')
    new_log = not os.path.exists(log_path)
    log = open(log_path, 'a')
    if new_log:
        log.write('epoch,iter,L,loss_loc,loss_cls,loss_landm,loss_total,lr,sec_per_iter\n')

    print(f'images={len(dataset)}  batch={bs}  iters/epoch={epoch_size}  '
          f'epochs={cfg["epoch"]}  decay at {cfg["decay1"]}, {cfg["decay2"]}')

    global_it = start_epoch * epoch_size
    for epoch in range(start_epoch, cfg['epoch']):
        lr = lr_at(epoch, cfg, args)
        for g in optimizer.param_groups:
            g['lr'] = lr
        loader = DataLoader(dataset, bs, shuffle=True, num_workers=args.num_workers,
                            collate_fn=detection_collate, generator=gen,
                            worker_init_fn=seed_worker, pin_memory=True)
        net.train()
        t_epoch = time.time()
        t_last = time.time()
        for it, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = [a.to(device) for a in targets]

            L = random.randint(*cfg['loop_train_L']) if cfg.get('loop') else None
            out = net(images, L=L)

            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            if not torch.isfinite(loss):
                raise RuntimeError(f'Loss is {loss.item()} at epoch {epoch} iter {it}; stopping.')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_it += 1

            if it % args.print_every == 0:
                sec = (time.time() - t_last) / max(1, args.print_every if it else 1)
                t_last = time.time()
                eta = sec * ((cfg['epoch'] - epoch) * epoch_size - it)
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f'ep {epoch + 1}/{cfg["epoch"]}  it {it}/{epoch_size}  L={L}  '
                      f'loc {loss_l.item():.3f}  cls {loss_c.item():.3f}  ldm {loss_landm.item():.3f}  '
                      f'lr {lr:.0e}  {sec:.2f}s/it  mem {mem:.1f}GB  ETA {datetime.timedelta(seconds=int(eta))}')
                log.write(f'{epoch},{it},{L},{loss_l.item():.5f},{loss_c.item():.5f},'
                          f'{loss_landm.item():.5f},{loss.item():.5f},{lr},{sec:.4f}\n')
                log.flush()

            if args.max_iters and global_it >= args.max_iters:
                torch.save(net.state_dict(), os.path.join(run_dir, 'smoke.pth'))
                print(f'Smoke test done after {global_it} iterations. Run folder: {run_dir}')
                return

        # end of epoch: always save a resumable checkpoint
        msg = f'== epoch {epoch + 1} done in {(time.time() - t_epoch) / 60:.1f} min'
        if cfg.get('haar'):
            msg += f'  |haar gamma| = {net.body.haar.net[-1].weight.abs().mean().item():.4f}'
        print(msg)
        torch.save({'epoch': epoch + 1, 'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(), 'rng': rng_state(gen)},
                   os.path.join(run_dir, 'last.pth'))
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(run_dir, f'epoch_{epoch + 1}.pth'))

    torch.save(net.state_dict(), os.path.join(run_dir, 'final.pth'))
    log.close()
    print(f'Training finished. Weights: {os.path.join(run_dir, "final.pth")}')

""" 
# 1. single batch overfit sanity check (loss should go close to 0 if the architecture/loss/targets are correct):
python train.py --config cfg_mnet_haar_loop --single_batch --batch_size 8 --sb_iters 500 --print_every 10

# 2. smoke test: train for a few iterations and exit (no checkpoints, no resume):
python train.py --config cfg_mnet_haar_loop --max_iters 100 --print_every 10


"""

if __name__ == '__main__':         
    main()
