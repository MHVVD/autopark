"""Train the slot detector.

    ros2 run autopark train_slots --data ~/autopark_data/slots_v1 --epochs 12 --out ~/autopark_models/slotnet.pt

Also runs without ROS (e.g. Colab): `python -m autopark.train_slots ...` with the autopark and
autopark_sim Python packages on PYTHONPATH. Uses CUDA when available and supported.
Expects <data>/train and <data>/val as written by collect_slots. Writes <out> (best validation
loss) and <out>.csv (per-epoch losses).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from autopark.slot_net import SlotDataset, SlotNet, slot_loss


def pick_device(requested):
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        try:  # the laptop MX130 (sm_50) is detected but has no kernels in current PyTorch
            torch.zeros(1, device='cuda') + 1
            return 'cuda'
        except Exception:
            pass
    return 'cpu'


def run_epoch(net, loader, device, opt=None, sched=None, limit=None):
    train = opt is not None
    net.train(train)
    sums, n = {}, 0
    with torch.set_grad_enabled(train):
        for k, (x, t) in enumerate(loader):
            if limit and k >= limit:
                break
            x = x.to(device)
            t = {kk: v.to(device) for kk, v in t.items()}
            loss, parts = slot_loss(net(x), t)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            parts['total'] = loss.item()
            for kk, v in parts.items():
                sums[kk] = sums.get(kk, 0.0) + v * len(x)
            n += len(x)
    return {k: v / max(n, 1) for k, v in sums.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/autopark_data/slots_v1'))
    ap.add_argument('--out', default=os.path.expanduser('~/autopark_models/slotnet.pt'))
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--limit', type=int, default=0, help='max batches per epoch (smoke tests)')
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    train = SlotDataset(os.path.join(args.data, 'train'), augment=True)
    val = SlotDataset(os.path.join(args.data, 'val'))
    tl = torch.utils.data.DataLoader(train, args.batch, shuffle=True, num_workers=args.workers,
                                     drop_last=True, persistent_workers=args.workers > 0)
    vl = torch.utils.data.DataLoader(val, args.batch, num_workers=args.workers)
    net = SlotNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=1e-4)
    steps = args.epochs * (min(len(tl), args.limit) if args.limit else len(tl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    print(f'device {device} | train {len(train)} | val {len(val)} | {steps} steps', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best = np.inf
    with open(args.out + '.csv', 'w', newline='') as f:
        log = csv.writer(f)
        log.writerow(['epoch', 'split', 'total', 'heat', 'offset', 'direction', 'vacancy', 'seconds'])
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(net, tl, device, opt, sched, args.limit)
            va = run_epoch(net, vl, device, limit=args.limit)
            dt = time.time() - t0
            for split, r in (('train', tr), ('val', va)):
                log.writerow([ep, split] + [round(r[k], 5) for k in
                                            ('total', 'heat', 'offset', 'direction', 'vacancy')] + [round(dt)])
            f.flush()
            tag = ''
            if va['total'] < best:
                best = va['total']
                torch.save(dict(model=net.state_dict(), config={}, epoch=ep, val=va), args.out)
                tag = ' *saved*'
            print(f'epoch {ep:2d} {dt:5.0f}s | train {tr["total"]:.3f} | val {va["total"]:.3f} '
                  f'(heat {va["heat"]:.3f} off {va["offset"]:.3f} dir {va["direction"]:.3f} '
                  f'vac {va["vacancy"]:.3f}){tag}', flush=True)
    print(f'best val loss {best:.3f} -> {args.out}')


if __name__ == '__main__':
    main()
