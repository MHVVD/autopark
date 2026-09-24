"""Slot detector network, loss and dataset (PyTorch). Runs on CPU or GPU (e.g. Colab).

SlotNet: small residual encoder (stride 64) + FPN-style decoder back to stride 4, heads:
  heat (1, sigmoid)  offset (2, sigmoid)  direction (2, unit vector)  vacancy (1, sigmoid)
See slot_codec for the target definitions.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autopark import slot_codec as sc

MEAN = np.array([0.40, 0.40, 0.40], np.float32)
STD = np.array([0.20, 0.20, 0.20], np.float32)


def conv_bn(cin, cout, stride=1, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.a = conv_bn(cin, cout, stride)
        self.b = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        return F.relu(self.b(self.a(x)) + self.skip(x))


class SlotNet(nn.Module):
    def __init__(self, width=(24, 32, 64, 96, 128), head=32):
        super().__init__()
        c1, c2, c3, c4, c5 = width
        self.stem = nn.Sequential(conv_bn(3, c1, 2), conv_bn(c1, c2, 2))              # /4
        self.s3 = nn.Sequential(ResBlock(c2, c3, 2), ResBlock(c3, c3))               # /8
        self.s4 = nn.Sequential(ResBlock(c3, c4, 2), ResBlock(c4, c4))               # /16
        self.s5 = nn.Sequential(ResBlock(c4, c5, 2), ResBlock(c5, c5))               # /32
        self.s6 = nn.Sequential(ResBlock(c5, c5, 2), ResBlock(c5, c5))               # /64
        self.lat = nn.ModuleList([nn.Conv2d(c, head, 1) for c in (c2, c3, c4, c5, c5)])
        self.smooth = conv_bn(head, head)
        self.heat = nn.Conv2d(head, 1, 1)
        self.offset = nn.Conv2d(head, 2, 1)
        self.direction = nn.Conv2d(head, 2, 1)
        self.vacancy = nn.Conv2d(head, 1, 1)
        nn.init.constant_(self.heat.bias, -2.19)  # prior 0.1 (CenterNet)

    def forward(self, x):
        f2 = self.stem(x)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        f6 = self.s6(f5)
        y = self.lat[4](f6)
        for lat, f in zip(self.lat[3::-1], (f5, f4, f3, f2)):
            y = F.interpolate(y, size=f.shape[-2:], mode='bilinear', align_corners=False) + lat(f)
        y = self.smooth(y)
        d = self.direction(y)
        return dict(heat=self.heat(y), offset=self.offset(y),
                    direction=d / (d.norm(dim=1, keepdim=True) + 1e-6), vacancy=self.vacancy(y))


# ------------------------------------------------------------------ loss
def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet penalty-reduced focal loss, normalised by the number of peaks."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def slot_loss(out, t, w=(1.0, 1.0, 1.0, 1.0)):
    heat = focal_loss(out['heat'][:, 0], t['heat'])
    peak = t['reg_mask'].eq(1).float()
    n_peak = peak.sum().clamp(min=1)
    off = (torch.abs(torch.sigmoid(out['offset']) - t['offset']).sum(1) * peak).sum() / n_peak
    dmask = (t['reg_mask'] > 0).float()
    dirl = (torch.abs(out['direction'] - t['direction']).sum(1) * dmask).sum() / dmask.sum().clamp(min=1)
    vm = t['vac_mask']
    vac = (F.binary_cross_entropy_with_logits(out['vacancy'][:, 0], t['vacancy'], reduction='none')
           * vm).sum() / vm.sum().clamp(min=1)
    total = w[0] * heat + w[1] * off + w[2] * dirl + w[3] * vac
    return total, dict(heat=heat.item(), offset=off.item(), direction=dirl.item(), vacancy=vac.item())


# ------------------------------------------------------------------ data
def to_tensor(bev_bgr):
    """BEV (450x450 BGR uint8) -> normalised (3, INPUT, INPUT) float tensor (RGB)."""
    x = sc.crop(bev_bgr)[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


def flip_targets(t, axis):
    """Mirror image + targets. axis 1: columns (vehicle left <-> right), 0: rows (front <-> back)."""
    t = {k: np.flip(v, axis=v.ndim - 2 + axis).copy() for k, v in t.items()}
    comp = 0 if axis == 1 else 1          # offset/direction component along the flipped axis
    peak = t['reg_mask'] == 1
    o = t['offset'][comp]
    o[peak] = 1.0 - o[peak]              # p -> INPUT-1-p: cell j -> OUT-1-j, offset o -> 1-o
    t['direction'][comp] *= -1
    return t


class SlotDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=False, seed=0):
        self.root = root
        self.items = [json.loads(l) for l in open(os.path.join(root, 'labels.jsonl'))]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(os.path.join(self.root, 'images', it['image']))
        t = sc.encode(it['slots'], it['pose'])
        img = sc.crop(img)
        if self.augment:
            rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
            for axis in (0, 1):
                if rng.random() < 0.5:
                    img = np.flip(img, axis=axis)
                    t = flip_targets(t, axis)
            img = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
            img = img + rng.normal(0, rng.uniform(0, 6), img.shape)
            img = np.clip(img, 0, 255)
        x = img[..., ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())
        return x, {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in t.items()}


# ------------------------------------------------------------------ inference
def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = SlotNet(**ck.get('config', {}))
    net.load_state_dict(ck['model'])
    return net.eval().to(device)


@torch.no_grad()
def predict(net, bev_bgr, threshold=0.3, device='cpu'):
    """BEV image -> (slots, marking points, raw numpy outputs) in the ground frame."""
    out = net(to_tensor(bev_bgr)[None].to(device))
    heat = torch.sigmoid(out['heat'])[0, 0].cpu().numpy()
    offset = torch.sigmoid(out['offset'])[0].cpu().numpy()
    direction = out['direction'][0].cpu().numpy()
    vac = torch.sigmoid(out['vacancy'])[0, 0].cpu().numpy()
    pts = sc.decode_points(heat, offset, direction, threshold)
    slots = sc.pair_points(pts)
    for s in slots:
        s.vacancy = sc.slot_vacancy(s, vac)
    return slots, pts, dict(heat=heat, offset=offset, direction=direction, vacancy=vac)
