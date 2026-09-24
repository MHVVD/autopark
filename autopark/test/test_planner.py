import math
import random

import numpy as np
import pytest

from autopark import parking_goal as pg
from autopark import reeds_shepp as rs
from autopark.geometry_check import _pt_seg, path_clearance, point_in_convex, polygon_distance, rect
from autopark.hybrid_astar import CarGeometry, HybridAStar, ObstacleMap, PlannerConfig


def rand_pose(rng, r=5.0):
    return (rng.uniform(-r, r), rng.uniform(-r, r), rng.uniform(-math.pi, math.pi))


# ---------------------------------------------------------------- Reeds-Shepp
def test_rs_every_candidate_reaches_goal():
    rng = random.Random(0)
    for _ in range(500):
        s, g = rand_pose(rng), rand_pose(rng)
        cands = rs.paths(s, g, 1.7)
        assert cands
        for p in cands:
            xs, ys, th, _, _ = rs.sample(s, p, 1.7, 0.1)
            assert abs(xs[-1] - g[0]) < 1e-6 and abs(ys[-1] - g[1]) < 1e-6
            assert abs(rs.mod2pi(th[-1] - g[2])) < 1e-6


def test_rs_known_distances():
    assert rs.distance((0, 0, 0), (3, 0, 0), 1.0) == pytest.approx(3.0)
    assert rs.distance((0, 0, 0), (-3, 0, 0), 1.0) == pytest.approx(3.0)       # straight back
    # quarter turn on the circle: exactly one arc of length pi/2 * r
    assert rs.distance((0, 0, 0), (2, 2, math.pi / 2), 2.0) == pytest.approx(math.pi)


def test_rs_symmetric_and_triangle_inequality():
    """A missing path family would show up as a violation (a concatenation would be shorter)."""
    rng = random.Random(1)
    for _ in range(3000):
        a, b, c = rand_pose(rng), rand_pose(rng), rand_pose(rng)
        assert rs.distance(a, b, 1.0) == pytest.approx(rs.distance(b, a, 1.0), abs=1e-9)
        assert rs.distance(a, c, 1.0) <= rs.distance(a, b, 1.0) + rs.distance(b, c, 1.0) + 1e-9


def test_rs_sample_spacing_and_direction():
    p = rs.shortest((0, 0, 0), (-1.0, 3.0, math.pi / 2), 1.5)
    xs, ys, th, d, k = rs.sample((0, 0, 0), p, 1.5, 0.1)
    steps = np.hypot(np.diff(xs), np.diff(ys))
    assert steps.max() <= 0.1 + 1e-9
    assert len(d) == len(xs) == len(k)
    assert set(np.abs(np.round(np.array(k) * 1.5, 9))) <= {0.0, 1.0}


# ---------------------------------------------------------------- geometry
def test_polygon_distance():
    a = rect(0, 0, 0, 2, 2)
    assert polygon_distance(a, rect(3, 0, 0, 2, 2)) == pytest.approx(1.0)
    assert polygon_distance(a, rect(0.2, 0.1, 0.4, 1, 1)) == 0.0          # contained
    assert polygon_distance(a, rect(2, 0, 0, 2, 2)) == 0.0                # touching
    # diamond centred at (3, 3): its nearest edge is the line x + y = 6 - sqrt(2), corner (1, 1)
    assert polygon_distance(a, rect(3, 3, math.pi / 4, 2, 2)) == pytest.approx((6 - math.sqrt(2) - 2) / math.sqrt(2))
    assert point_in_convex((0.5, 0.5), a) and not point_in_convex((1.5, 0), a)


def test_obstacle_map_clearance_is_a_lower_bound():
    rng = np.random.default_rng(0)
    box = rect(0, 0, 0.3, 2.0, 1.0)
    om = ObstacleMap(-6, -6, 6, 6, [box], res=0.05)
    pts = rng.uniform(-4, 4, size=(3000, 2))
    lb = om.clearance(pts[:, 0], pts[:, 1])
    for (x, y), l in zip(pts, lb):
        true = 0.0 if point_in_convex((x, y), box) else min(_pt_seg((x, y), box[k], box[(k + 1) % 4]) for k in range(4))
        true = min(true, 6 - abs(x), 6 - abs(y))          # the map border is an obstacle too
        assert l <= true + 1e-6
        assert l >= true - 0.15            # and not too conservative
    # beyond the map border there is no clearance
    assert om.clearance(np.array([7.0]), np.array([0.0]))[0] == 0.0


def test_collision_check_matches_exact_geometry():
    """free() may only say 'free' when the true rectangle grown by the margin is free."""
    car, cfg = CarGeometry(), PlannerConfig()
    pl = HybridAStar(car, cfg)
    obst = [rect(0, 0, 0.2, 3.0, 2.0), rect(-3, 4, -0.5, 2.0, 5.0)]
    om = ObstacleMap(-12, -12, 12, 12, obst)
    rng = np.random.default_rng(1)
    n_free = 0
    for _ in range(2000):
        pose = (rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-math.pi, math.pi))
        if pl.free(om, [pose[0]], [pose[1]], [pose[2]])[0]:
            n_free += 1
            body = car.corners(pose)
            d = min(polygon_distance(body, o) for o in obst)
            assert d >= cfg.margin - 1e-6
    assert n_free > 300


# ---------------------------------------------------------------- parking geometry
def test_goal_pose_centres_car_in_slot_facing_out():
    car = CarGeometry()
    s = pg.SlotSpec(0, 1.0, 3.5, math.pi / 2)
    x, y, yaw = pg.goal_pose(s, car)
    assert yaw == pytest.approx(-math.pi / 2)
    body = car.corners((x, y, yaw))
    cx, cy = np.mean([p[0] for p in body]), np.mean([p[1] for p in body])
    assert cx == pytest.approx(1.0) and cy == pytest.approx(3.5 + 2.6)
    assert all(point_in_convex(p, pg.slot_rect(s)) for p in body)


def test_obstacles_skip_target_and_vacant_slots():
    specs = [pg.SlotSpec(i, 2.6 * i, 3.5, math.pi / 2, vacant=(i == 2)) for i in range(4)]
    polys = pg.obstacles(specs, target_id=1)
    assert len(polys) == 2 + 4          # slots 0 and 3 are occupied; 4 back bands


# ---------------------------------------------------------------- Hybrid A*
def _row_problem(target=3, vacant=(3,), start_dx=4.0):
    specs = []
    for i in range(8):
        specs.append(pg.SlotSpec(i, -9.1 + 2.6 * i, 3.5, math.pi / 2, vacant=i in vacant))
        specs.append(pg.SlotSpec(8 + i, -9.1 + 2.6 * i, -3.5, -math.pi / 2, vacant=False))
    t = specs[2 * target]
    start = (t.x + start_dx, 0.0, 0.0)
    goal = pg.goal_pose(t)
    om = ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, target))
    return start, goal, om, specs


def test_hybrid_astar_reverse_parks_between_cars():
    start, goal, om, specs = _row_problem()
    pl = HybridAStar()
    path = pl.plan(start, goal, om)
    assert path is not None, pl.last_info
    assert (path.x[-1], path.y[-1]) == pytest.approx(goal[:2], abs=1e-6)
    assert path.direction[-1] == -1                          # enters backwards
    assert pl.free(om, path.x, path.y, path.yaw).all()
    assert np.all(np.abs(path.curvature) <= 1 / pl.car.min_radius + 1e-9)
    assert np.hypot(np.diff(path.x), np.diff(path.y)).max() <= pl.cfg.sample + 1e-6
    # every piece between gear changes is long enough to drive
    for i0, i1 in path.segments():
        seg = np.sum(np.hypot(np.diff(path.x[i0:i1 + 1]), np.diff(path.y[i0:i1 + 1])))
        assert seg >= pl.cfg.min_segment - 1e-6


def test_hybrid_astar_reports_blocked_goal():
    start, goal, om, specs = _row_problem(vacant=())
    # target slot occupied by an obstacle -> goal in collision
    blocked = ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, target_id=-1))
    pl = HybridAStar()
    assert pl.plan(start, goal, blocked) is None
    assert pl.last_info['reason'] == 'goal in collision'


def test_path_clearance_exact():
    car = CarGeometry()
    obst = [rect(10, 0, 0, 2, 2)]
    xs = np.linspace(0, 2, 5)
    d, i = path_clearance(car, xs, np.zeros(5), np.zeros(5), obst)
    assert i == 4 and d == pytest.approx(10 - 1 - (2 + car.front))


# ---------------------------------------------------------------- explored region
def test_known_box_only_covers_slots_the_tracker_has_seen():
    """Every slot that reaches into KNOWN_BOX must have its entrance in the tracker's view, for
    car poses along the aisle (either direction, yaw within the tolerance, up to 1.5 m off the
    aisle centre). Otherwise an unseen occupied slot could be planned through."""
    from autopark.slot_tracker import in_view
    from autopark_sim.lot import slots
    lot = slots()
    tol = math.degrees(pg.KNOWN_YAW_TOL)
    bad = []
    for yaw in np.radians(np.linspace(-tol, tol, 5)):
        for flip in (0.0, math.pi):
            for y0 in (-1.5, 0.0, 1.5):
                for x0 in np.arange(-14.0, 14.0, 0.5):
                    pose = (x0, y0, yaw + flip)
                    box = pg.known_region([pose])[0]
                    for s in lot:
                        if not in_view(pose, s.entrance_x, s.entrance_y) and polygon_distance(box, s.corners()) == 0.0:
                            bad.append((pose, s.id))
    assert not bad, bad[:5]


def test_aisle_aligned():
    heads = [math.pi / 2, -math.pi / 2, math.pi / 2 + 0.02]
    assert pg.aisle_aligned(0.05, heads) and pg.aisle_aligned(math.pi - 0.05, heads)
    assert not pg.aisle_aligned(0.5, heads)
    assert not pg.aisle_aligned(0.0, [])


def test_unexplored_space_is_an_obstacle():
    om = ObstacleMap(-10, -10, 10, 10, [], known=[rect(0, 0, 0, 6, 4)])
    assert om.clearance(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(2.0, abs=0.08)
    assert om.clearance(np.array([5.0]), np.array([0.0]))[0] == 0.0


def test_planner_stays_inside_explored_space():
    start, goal, _, specs = _row_problem(start_dx=4.0)
    poses = [(x, 0.0, 0.0) for x in np.arange(start[0] - 12, start[0] + 0.1, 1.0)]
    known = pg.known_region(poses)
    om = ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, 3), known=known)
    pl = HybridAStar()
    path = pl.plan(start, goal, om)
    assert path is not None, pl.last_info
    for x, y, th in zip(path.x, path.y, path.yaw):
        body = pl.car.corners((x, y, th))
        assert all(any(point_in_convex(c, k) for k in known) for c in body)
    # the same start, but explored only far behind: the goal is unexplored -> refused
    far = pg.known_region([(start[0] - 30, 0.0, 0.0)])
    om2 = ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, 3), known=far)
    assert pl.plan(start, goal, om2) is None
