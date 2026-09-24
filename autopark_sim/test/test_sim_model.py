import math
import os
import xml.etree.ElementTree as ET

import pytest

from autopark_sim import lot
from autopark_sim.descriptions import supervisor_description, vehicle_description
from autopark_sim.generate_world import WORLDS, world_string
from autopark_sim.scenario import CAR_MODELS, EMPTY_LOT_SEED, make_scenario
from autopark_sim.vehicle_model import (WHEEL_BASE, ackermann_center_angle, encoder_speed,
                                        rate_limit, to_webots)

HERE = os.path.dirname(__file__)


# ---------- lot geometry ----------

def test_sixteen_slots_with_unique_ids():
    s = lot.slots()
    assert [x.id for x in s] == list(range(16))


def test_corners_order_and_size():
    for s in lot.slots():
        el, er, br, bl = s.corners()
        assert math.dist(el, er) == pytest.approx(lot.SLOT_WIDTH)
        assert math.dist(er, br) == pytest.approx(lot.SLOT_DEPTH)
        # heading points from the entrance to the back
        mid_e = ((el[0] + er[0]) / 2, (el[1] + er[1]) / 2)
        mid_b = ((bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2)
        assert math.atan2(mid_b[1] - mid_e[1], mid_b[0] - mid_e[0]) == pytest.approx(s.heading)
        # entrance_left is on the left when looking into the slot: cross(heading, el - e) > 0
        hx, hy = math.cos(s.heading), math.sin(s.heading)
        assert hx * (el[1] - mid_e[1]) - hy * (el[0] - mid_e[0]) > 0


def test_entrances_face_the_aisle():
    for s in lot.slots():
        assert abs(s.entrance_y) == pytest.approx(lot.AISLE_WIDTH / 2)
        cx, cy = s.center()
        assert abs(cy) > abs(s.entrance_y)  # slot extends away from the aisle


def test_corners_lie_on_painted_line_centres():
    lines = lot.painted_lines()

    def on_a_line(p):
        return any(abs(p[0] - x) <= sx / 2 + 1e-9 and abs(p[1] - y) <= sy / 2 + 1e-9
                   for x, y, sx, sy in lines)

    for s in lot.slots():
        for c in s.corners():
            assert on_a_line(c), (s.id, c)


# ---------- scenario ----------

def _footprint(car):
    m = CAR_MODELS[car.model]
    c, s = math.cos(car.yaw), math.sin(car.yaw)
    cx, cy = car.x + m['offset'] * c, car.y + m['offset'] * s
    hl, hw = m['length'] / 2, m['width'] / 2
    return [(cx + a * hl * c - b * hw * s, cy + a * hl * s + b * hw * c)
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def _overlap(p, q):
    """Separating axis test for two convex quads."""
    for poly in (p, q):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            a = [nx * x + ny * y for x, y in p]
            b = [nx * x + ny * y for x, y in q]
            if max(a) < min(b) or max(b) < min(a):
                return False
    return True


def test_scenario_is_deterministic():
    a, b = make_scenario(42), make_scenario(42)
    assert a.empty_slot_ids == b.empty_slot_ids
    assert [(c.model, c.x, c.y, c.yaw) for c in a.parked] == \
           [(c.model, c.x, c.y, c.yaw) for c in b.parked]
    assert a.ego == b.ego


@pytest.mark.parametrize('seed', range(50))
def test_scenario_valid(seed):
    sc = make_scenario(seed)
    assert 1 <= len(sc.empty_slot_ids) <= 3
    assert len(sc.parked) + len(sc.empty_slot_ids) == 16
    assert {c.slot_id for c in sc.parked}.isdisjoint(sc.empty_slot_ids)
    feet = [_footprint(c) for c in sc.parked]
    for i in range(len(feet)):
        for j in range(i + 1, len(feet)):
            assert not _overlap(feet[i], feet[j]), (seed, i, j)
    # parked cars stay out of the aisle (with 0.3 m margin)
    for f in feet:
        assert all(abs(y) > lot.AISLE_WIDTH / 2 - 0.3 for _, y in f)
    # ego starts in the aisle, clear of the rows
    x, y, _ = sc.ego
    assert x < lot.ROW_X0 - 2 and abs(y) < 1.0


def test_empty_lot_scenario():
    sc = make_scenario(EMPTY_LOT_SEED)
    assert sc.parked == [] and len(sc.empty_slot_ids) == 16


# ---------- vehicle model ----------

def test_rate_limit():
    assert rate_limit(0.0, 1.0, 1.0, 0.1) == pytest.approx(0.1)
    assert rate_limit(0.0, -1.0, 1.0, 0.1) == pytest.approx(-0.1)
    assert rate_limit(0.95, 1.0, 1.0, 0.1) == pytest.approx(1.0)


def test_to_webots_units_and_sign():
    assert to_webots(1.0, 0.2) == (pytest.approx(3.6), -0.2)


@pytest.mark.parametrize('delta', [-0.6, -0.3, -0.05, 0.05, 0.3, 0.6])
def test_ackermann_center_angle_inverts_ackermann_geometry(delta):
    # Wheel angles of an ideal Ackermann linkage for centre angle delta (left positive).
    L, T = WHEEL_BASE, 1.628
    R = L / math.tan(delta)                  # signed turn radius at the rear axle centre
    left, right = math.atan(L / (R - T / 2)), math.atan(L / (R + T / 2))
    assert ackermann_center_angle(left, right) == pytest.approx(delta, abs=1e-9)


def test_ackermann_center_angle_straight():
    assert ackermann_center_angle(0.0, 0.0) == 0.0


def test_encoder_speed():
    # 1 rad per 0.1 s on both wheels, r = 0.317 -> 3.17 m/s
    assert encoder_speed(1.0, 1.0, 0.1) == pytest.approx(3.17)
    assert encoder_speed(-1.0, -1.0, 0.1) == pytest.approx(-3.17)


# ---------- generated files ----------

@pytest.mark.parametrize('name', sorted(WORLDS))
def test_committed_worlds_match_generator(name):
    with open(os.path.join(HERE, '..', 'worlds', name)) as f:
        assert f.read() == world_string(WORLDS[name]), 'run: python3 -m autopark_sim.generate_world worlds'


def test_descriptions_are_valid_xml():
    v = ET.fromstring(vehicle_description({'gyroNoise': 0.01}))
    assert len(v.findall('webots/device')) == 4
    assert v.find('webots/plugin/gyroNoise').text == '0.01'
    s = ET.fromstring(supervisor_description(7))
    assert s.find('webots/plugin/seed').text == '7'
