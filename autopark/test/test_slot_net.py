import math

import numpy as np
import pytest
import torch

from autopark import slot_codec as sc
from autopark.bev import BevGrid
from autopark.slot_net import SlotNet, flip_targets, slot_loss
from test_slot_codec import gt_slots

G = BevGrid()


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('pose', [(-3.0, 0.0, 0.0), (2.0, -0.8, 0.25)])
def test_flip_targets_equals_mirrored_geometry(axis, pose):
    t = sc.encode(gt_slots(), pose)
    f = flip_targets(t, axis)
    pts = sc.decode_points(t['heat'], t['offset'], t['direction'], 0.99)
    fpts = sc.decode_points(f['heat'], f['offset'], f['direction'], 0.99)
    assert len(pts) == len(fpts) > 0
    for p in pts:
        # columns mirror y -> -y; rows mirror x about the grid centre
        if axis == 1:
            q = (p.x, -p.y, p.dx, -p.dy)
        else:
            q = (G.x_min + G.x_max - p.x, p.y, -p.dx, p.dy)
        best = min(fpts, key=lambda r: math.hypot(r.x - q[0], r.y - q[1]))
        assert math.hypot(best.x - q[0], best.y - q[1]) < 1e-3
        assert best.dx * q[2] + best.dy * q[3] > 0.999
    assert f['vac_mask'].sum() == t['vac_mask'].sum()


def test_network_shapes_and_loss_backprop():
    torch.manual_seed(0)       # random input: a near-zero direction vector cannot be normalised
    net = SlotNet()
    x = torch.randn(2, 3, sc.INPUT, sc.INPUT)
    out = net(x)
    assert out['heat'].shape == (2, 1, sc.OUT, sc.OUT)
    assert torch.allclose(out['direction'].norm(dim=1), torch.ones(2, sc.OUT, sc.OUT), atol=1e-3)
    t = {k: torch.from_numpy(np.stack([v, v])) for k, v in sc.encode(gt_slots(), (0.0, 0.0, 0.0)).items()}
    loss, parts = slot_loss(out, t)
    loss.backward()
    assert math.isfinite(loss.item()) and set(parts) == {'heat', 'offset', 'direction', 'vacancy'}
