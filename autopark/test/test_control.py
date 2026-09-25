import math
from types import SimpleNamespace

import numpy as np
import pytest

from autopark import parking_goal as pg
from autopark import parking_manager as pm
from autopark import reeds_shepp as rs
from autopark.hybrid_astar import HybridAStar, ObstacleMap
from autopark.kinematic_sim import KinematicCar
from autopark.stanley import Stanley, StanleyConfig


def rs_path(start, goal, radius=4.57):
    p = rs.shortest(start, goal, radius)
    x, y, th, d, k = rs.sample(start, p, radius, 0.1)
    return SimpleNamespace(x=x, y=y, yaw=th, direction=d, curvature=k)


def drive(path, start, delay=3, t_max=120.0, cfg=None):
    car = KinematicCar(start, delay_steps=delay)
    c = Stanley(cfg)
    c.set_path(path)
    t = 0.0
    while c.state not in ('done', 'aborted') and t < t_max:
        v, s = c.update(car.pose, car.v, car.steer, t)
        car.step(v, s, 0.02)
        t += 0.02
    return c, car


def lateral(pose, goal):
    dx, dy = pose[0] - goal[0], pose[1] - goal[1]
    return -math.sin(goal[2]) * dx + math.cos(goal[2]) * dy, math.cos(goal[2]) * dx + math.sin(goal[2]) * dy


# ---------------------------------------------------------------- Stanley
@pytest.mark.parametrize('goal', [(8.0, 0.0, 0.0), (-8.0, 0.0, 0.0), (6.0, 3.0, math.pi / 2),
                                  (-5.0, -4.0, -math.pi / 2), (3.0, 2.0, math.pi)])
def test_stanley_follows_reeds_shepp_paths(goal):
    """Forward, reverse, turns and cusps: ends on the goal and stays close to the path."""
    path = rs_path((0.0, 0.0, 0.0), goal)
    c, car = drive(path, (0.0, 0.0, 0.0))
    assert c.state == 'done'
    lat, lon = lateral(car.pose, goal)
    assert abs(lat) < 0.03 and abs(lon) < 0.03
    assert abs(math.remainder(car.yaw - goal[2], 2 * math.pi)) < math.radians(1.5)
    assert c.max_lateral < 0.08


@pytest.mark.parametrize('direction', [1, -1])
def test_stanley_converges_from_an_offset(direction):
    """Straight path, car starting 0.3 m to the side: the error decays."""
    xs = np.arange(0, 12.01, 0.1) * direction
    path = SimpleNamespace(x=xs, y=np.zeros_like(xs), yaw=np.zeros_like(xs),
                           direction=np.full(len(xs), direction), curvature=np.zeros_like(xs))
    c, car = drive(path, (0.0, 0.3, 0.0))
    assert c.state == 'done'
    assert abs(car.y) < 0.02 and abs(car.yaw) < math.radians(1.0)


def test_stanley_aborts_when_far_off():
    xs = np.arange(0, 10.01, 0.1)
    path = SimpleNamespace(x=xs, y=np.zeros_like(xs), yaw=np.zeros_like(xs),
                           direction=np.ones(len(xs)), curvature=np.zeros_like(xs))
    c, _ = drive(path, (0.0, 1.5, 0.0))
    assert c.state == 'aborted'


def test_stanley_parks_on_planned_paths():
    """Planner paths (full-lock arcs with steering jumps) in a parking row."""
    specs = []
    for i in range(8):
        specs.append(pg.SlotSpec(i, -9.1 + 2.6 * i, 3.5, math.pi / 2, vacant=i in (3, 5)))
        specs.append(pg.SlotSpec(8 + i, -9.1 + 2.6 * i, -3.5, -math.pi / 2, vacant=i == 2))
    pl = HybridAStar()
    for tid, dx in ((3, 3.0), (5, 5.0), (10, 4.0)):
        t = next(s for s in specs if s.id == tid)
        start = (t.x + dx, 0.2, 0.03)
        goal = pg.goal_pose(t)
        path = pl.plan(start, goal, ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, tid)))
        assert path is not None
        c, car = drive(path, start)
        assert c.state == 'done'
        lat, lon = lateral(car.pose, goal)
        assert abs(lat) < 0.03 and abs(lon) < 0.03
        assert c.max_lateral < 0.08


def test_stanley_keeps_progress_on_correction():
    path = rs_path((0.0, 0.0, 0.0), (-8.0, 0.0, 0.0))
    car = KinematicCar((0.0, 0.0, 0.0))
    c = Stanley(StanleyConfig())
    c.set_path(path)
    t = 0.0
    while t < 8.0:
        v, s = c.update(car.pose, car.v, car.steer, t)
        car.step(v, s, 0.02)
        t += 0.02
    idx = c.idx
    moved = pm.apply_correction(path, (-8.0, 0.0, 0.0), (0.0, 0.05, 0.0))
    c.set_path(moved, keep_progress=True)
    c.update(car.pose, car.v, car.steer, t)
    assert c.state == 'track' and abs(c.idx - idx) <= 2


# ---------------------------------------------------------------- manager logic
def _row(vacant=()):
    return ([pg.SlotSpec(i, -9.1 + 2.6 * i, 3.5, math.pi / 2, vacant=i in vacant) for i in range(8)]
            + [pg.SlotSpec(8 + i, -9.1 + 2.6 * i, -3.5, -math.pi / 2) for i in range(8)])


def test_aisle_line_from_both_rows_and_one_row():
    p, u = pm.aisle_line(_row(), (0.0, 1.0, 0.1))
    assert u == pytest.approx([1.0, 0.0], abs=1e-9)
    assert p[1] == pytest.approx(0.0, abs=1e-9)
    p, u = pm.aisle_line(_row()[:8], (0.0, 1.0, math.pi))       # one row, driving -x
    assert u == pytest.approx([-1.0, 0.0], abs=1e-9)
    assert p[1] == pytest.approx(0.0, abs=1e-9)
    assert pm.aisle_line([], (0, 0, 0)) is None


def test_pure_pursuit_steers_back_to_the_line():
    line = pm.aisle_line(_row(), (0.0, 0.0, 0.0))
    assert pm.pure_pursuit_to_line((0.0, 1.0, 0.0), line) < 0      # left of the line: steer right
    assert pm.pure_pursuit_to_line((0.0, -1.0, 0.0), line) > 0
    assert pm.pure_pursuit_to_line((0.0, 0.0, 0.0), line) == pytest.approx(0.0, abs=1e-9)


def test_choose_target_only_after_passing():
    slots = _row(vacant=(3, 5))            # entrances at x = -1.3 and 3.9
    u = np.array([1.0, 0.0])
    assert pm.choose_target(slots, (1.0, 0.0, 0.0), u, 3.0) is None
    assert pm.choose_target(slots, (2.0, 0.0, 0.0), u, 3.0).id == 3
    assert pm.choose_target(slots, (7.5, 0.0, 0.0), u, 3.0).id == 5     # the most recently passed
    assert pm.choose_target(slots, (7.5, 0.0, 0.0), u, 3.0, excluded={5}).id == 3


def test_first_segment_ends_at_the_cusp():
    path = rs_path((0.0, 0.0, 0.0), (3.0, 2.0, math.pi))
    d = list(path.direction)
    part, last = pm.first_segment(path)
    cusp = next(i for i in range(1, len(d)) if d[i] != d[i - 1]) - 1
    assert not last and len(part.x) == cusp + 1 and set(part.direction) == {d[0]}
    straight = rs_path((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    part, last = pm.first_segment(straight)
    assert last and len(part.x) == len(straight.x)


def test_rigid_correction_moves_the_goal_exactly():
    path = rs_path((0.0, 0.0, 0.0), (-6.0, 4.0, -math.pi / 2))
    g_old = (-6.0, 4.0, -math.pi / 2)
    g_new = (-5.95, 4.03, -math.pi / 2 + 0.01)
    moved = pm.apply_correction(path, g_old, pm.rigid_correction(g_old, g_new))
    assert (moved.x[-1], moved.y[-1]) == pytest.approx(g_new[:2], abs=1e-9)
    assert moved.yaw[-1] == pytest.approx(g_new[2], abs=1e-9)
    # rigid: distances between samples are unchanged
    d0 = np.hypot(np.diff(path.x), np.diff(path.y))
    d1 = np.hypot(np.diff(moved.x), np.diff(moved.y))
    assert d1 == pytest.approx(d0, abs=1e-9)


def test_remainder_continues_after_the_first_segment():
    path = rs_path((0.0, 0.0, 0.0), (3.0, 2.0, math.pi))
    part, last = pm.first_segment(path)
    rest = pm.remainder(path, len(part.x))
    assert not last
    assert len(part.x) + len(rest.x) == len(path.x)
    assert rest.x[-1] == path.x[-1] and rest.direction[0] != part.direction[-1]
