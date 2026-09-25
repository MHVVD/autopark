import math

import numpy as np

from autopark import collect_poses as cp
from autopark.eval_slots import aisle_angle
from autopark.geometry_check import point_in_convex
from autopark.hybrid_astar import CarGeometry
from autopark.plan_bench import true_car_rects
from autopark_sim.lot import slots as lot_slots
from autopark_sim.scenario import make_scenario

CAR = CarGeometry()


def test_sample_poses_mix_deterministic_and_collision_free():
    for seed in (10200, 31):
        a = cp.sample_poses(seed, 20)
        assert a == cp.sample_poses(seed, 20)
        assert len(a) == 20
        kinds = [k for k, _ in a]
        assert kinds.count('manoeuvre') == 10 and kinds.count('aisle') == 5 and kinds.count('slot') == 5
        cars = true_car_rects(make_scenario(seed))
        for _, p in a:
            assert cp.pose_ok(CAR, p, cars)


def test_pose_ok_rejects_contact_and_leaving_the_lot():
    sc = make_scenario(31)
    cars = true_car_rects(sc)
    occupied = next(s for s in lot_slots() if s.id not in sc.empty_slot_ids)
    cx, cy = occupied.center()
    assert not cp.pose_ok(CAR, (cx, cy, occupied.heading), cars)
    assert not cp.pose_ok(CAR, (0.0, 12.0, 0.0), cars)
    assert cp.pose_ok(CAR, (0.0, 0.0, 0.0), cars)


def test_slot_poses_overlap_an_empty_slot_facing_out_or_in():
    seed = 10201
    sc = make_scenario(seed)
    empty = [s for s in lot_slots() if s.id in sc.empty_slot_ids]
    for kind, p in cp.sample_poses(seed, 40):
        if kind != 'slot':
            continue
        centre = (p[0] + (CAR.front - CAR.rear) / 2 * math.cos(p[2]),
                  p[1] + (CAR.front - CAR.rear) / 2 * math.sin(p[2]))
        rear_ok = [s for s in empty if any(point_in_convex(c, s.corners()) for c in CAR.corners(p))]
        assert rear_ok, f'slot pose {p} touches no empty slot'
        s = min(rear_ok, key=lambda s: math.dist(centre, s.center()))
        d = abs(math.remainder(p[2] - s.heading, math.pi))    # along the slot axis, either way
        assert d <= math.radians(25) + 1e-9


def test_aisle_angle():
    for yaw, a in ((0.0, 0.0), (math.pi, 0.0), (math.pi / 2, 90.0), (-math.pi / 2, 90.0),
                   (math.radians(160), 20.0), (math.radians(-30), 30.0)):
        assert aisle_angle(yaw) == np.float64(a) or abs(aisle_angle(yaw) - a) < 1e-9
