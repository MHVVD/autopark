import math

import numpy as np
import pytest

from autopark import slot_codec as sc
from autopark_sim.lot import slots as lot_slots
from autopark_sim.scenario import make_scenario


def gt_slots(seed=3):
    occ = {c.slot_id for c in make_scenario(seed).parked}
    return [dict(id=s.id, corners=[list(c) for c in s.corners()], heading=s.heading,
                 occupied=s.id in occ) for s in lot_slots()]


POSES = [(-3.0, 0.0, 0.0), (2.0, -0.8, 0.25), (5.5, 1.2, -0.4), (0.0, 0.0, math.pi)]


def test_frame_round_trips():
    col, row = sc.ground_to_input(3.2, -1.7)
    assert sc.input_to_ground(col, row) == pytest.approx((3.2, -1.7))
    dc, dr = sc.dir_ground_to_image(1.0, 0.0)          # forward = up in the image
    assert (dc, dr) == (-0.0, -1.0)
    assert sc.dir_image_to_ground(dc, dr) == (1.0, 0.0)


@pytest.mark.parametrize('pose', POSES)
def test_encode_decode_recovers_ground_truth(pose):
    slots_map = gt_slots()
    t = sc.encode(slots_map, pose)
    assert t['heat'].max() == 1.0 and t['vac_mask'].sum() > 0

    pts = sc.decode_points(t['heat'], t['offset'], t['direction'], threshold=0.99)
    truth = [p for p in sc.marking_points_from_slots(slots_map, pose)
             if 0 <= sc.ground_to_input(p.x, p.y)[0] < sc.INPUT and
             0 <= sc.ground_to_input(p.x, p.y)[1] < sc.INPUT]
    assert len(pts) == len(truth) > 0
    for q in truth:
        best = min(pts, key=lambda p: math.hypot(p.x - q.x, p.y - q.y))
        assert math.hypot(best.x - q.x, best.y - q.y) < 1e-3      # exact (sub-cell offsets)
        assert best.dx * q.dx + best.dy * q.dy > 0.999

    slots = sc.decode(t['heat'], t['offset'], t['direction'], t['vacancy'], threshold=0.99)
    # every slot whose two entrance corners are in view is recovered, with correct geometry
    visible = []
    for s in slots_map:
        g = sc.map_to_ground(np.asarray(s['corners'][:2]), pose)
        col, row = sc.ground_to_input(g[:, 0], g[:, 1])
        if np.all((col >= 0) & (col < sc.INPUT) & (row >= 0) & (row < sc.INPUT)):
            visible.append((s, g))
    assert len(slots) == len(visible) > 0
    for s, g in visible:
        m = g.mean(0)
        d = min(slots, key=lambda x: math.hypot(x.entrance[0] - m[0], x.entrance[1] - m[1]))
        assert math.hypot(d.entrance[0] - m[0], d.entrance[1] - m[1]) < 1e-3
        assert (d.left.x, d.left.y) == pytest.approx(tuple(g[0]), abs=1e-3)   # left/right order
        assert math.cos(d.heading - (s['heading'] - pose[2])) > 0.999
        if not math.isnan(d.vacancy):
            assert (d.vacancy > 0.5) == (not s['occupied'])


def test_pairing_rejects_non_slots():
    P = sc.MarkingPoint
    # parallel points 2.6 m apart -> one slot; 5.2 m apart -> none; opposite directions -> none
    assert len(sc.pair_points([P(0, 0, 0, 1), P(2.6, 0, 0, 1)])) == 1
    assert len(sc.pair_points([P(0, 0, 0, 1), P(5.2, 0, 0, 1)])) == 0
    assert len(sc.pair_points([P(0, 0, 0, 1), P(2.6, 0, 0, -1)])) == 0
    # a row of 3 points gives 2 slots, the middle point used once as left and once as right
    s = sc.pair_points([P(0, 0, 0, 1), P(2.6, 0, 0, 1), P(5.2, 0, 0, 1)])
    assert len(s) == 2
    # heading into +y: left corner is at smaller x
    one = sc.pair_points([P(2.6, 0, 0, 1), P(0, 0, 0, 1)])[0]
    assert (one.left.x, one.right.x) == (0, 2.6) and one.heading == pytest.approx(math.pi / 2)


def test_heading_blends_direction_and_entrance_normal():
    import math
    from autopark import slot_codec as sc
    # exact points: both estimates agree
    a = sc.MarkingPoint(2.0, 3.5, 0.0, 1.0)
    b = sc.MarkingPoint(4.6, 3.5, 0.0, 1.0)
    (s,) = sc.pair_points([a, b])
    assert abs(s.heading - math.pi / 2) < 1e-9
    # noisy line directions near the car: the entrance normal dominates
    a = sc.MarkingPoint(0.0, 2.0, math.cos(1.62), math.sin(1.62))
    b = sc.MarkingPoint(2.6, 2.0, math.cos(1.62), math.sin(1.62))
    (s,) = sc.pair_points([a, b])
    assert abs(s.heading - math.pi / 2) < 0.1 * (1.62 - math.pi / 2) + 1e-9
    # weights: 0.9 near, 0.25 far
    assert abs(sc.blend_heading(0.0, 0.1, 1.0) - 0.09) < 1e-3
    assert abs(sc.blend_heading(0.0, 0.1, 20.0) - 0.025) < 1e-3
