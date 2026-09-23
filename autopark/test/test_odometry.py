import math

import pytest

from autopark.drive_test import SCRIPT, phase_at
from autopark.odometry import (AckermannOdometry, compose, inverse, wrap,
                               yaw_from_quaternion, yaw_rate_from_steering)
from autopark.odom_eval import dead_reckon, drift


def test_straight_forward_and_reverse():
    o = AckermannOdometry()
    for _ in range(100):
        o.update(1.0, 0.0, 0.1)
    assert (o.x, o.y, o.yaw) == pytest.approx((10.0, 0.0, 0.0))
    for _ in range(50):
        o.update(-1.0, 0.0, 0.1)
    assert o.x == pytest.approx(5.0)
    assert o.distance == pytest.approx(15.0)


@pytest.mark.parametrize('speed, radius', [(1.0, 5.0), (-0.8, 7.0), (1.5, -6.0)])
def test_constant_turn_matches_circle(speed, radius):
    """Arc integration of a constant-curvature motion lands on the exact circle."""
    o = AckermannOdometry()
    w = speed / radius
    t, dt = 0.0, 0.02
    while t < 10.0 - 1e-9:
        o.update(speed, w, dt)
        t += dt
    yaw = w * t
    exact = (radius * math.sin(yaw), radius * (1 - math.cos(yaw)))
    assert (o.x, o.y) == pytest.approx(exact, abs=1e-6)
    assert o.yaw == pytest.approx(wrap(yaw))


def test_yaw_rate_from_steering_signs():
    assert yaw_rate_from_steering(1.0, 0.3, 2.8) > 0     # forward, left -> ccw
    assert yaw_rate_from_steering(-1.0, 0.3, 2.8) < 0    # reverse, left -> cw
    assert yaw_rate_from_steering(1.0, 0.0, 2.8) == 0


def test_compose_and_wrap():
    assert compose((1.0, 2.0, math.pi / 2), (1.0, 0.0, 0.0)) == pytest.approx((1.0, 3.0, math.pi / 2))
    assert wrap(3 * math.pi) == pytest.approx(-math.pi)


def test_yaw_from_quaternion():
    a = 0.7
    assert yaw_from_quaternion(0, 0, math.sin(a / 2), math.cos(a / 2)) == pytest.approx(a)


def test_offline_dead_reckoning_and_drift_are_consistent():
    # Perfect data on a straight line: zero drift in both heading modes.
    states = [(i * 0.02, 1.0, 0.0) for i in range(101)]
    gyro = {round(t * 1000): 0.0 for t, _, _ in states}
    gt = {round(t * 1000): (5.0 + t, 1.0, 0.0) for t, _, _ in states}
    for mode in ('imu', 'steering'):
        rows = drift(dead_reckon(states, gyro, mode), gt, (5.0, 1.0, 0.0))
        assert max(r[2] for r in rows) == pytest.approx(0.0, abs=1e-9)


def test_drive_script_phases():
    assert phase_at(0.0) == 0
    assert phase_at(sum(d for d, _, _ in SCRIPT) + 0.1) is None
    # the script exercises forward/reverse with left steering
    assert any(v > 0 and s > 0 for _, v, s in SCRIPT)
    assert any(v < 0 and s > 0 for _, v, s in SCRIPT)


def test_inverse_and_frame_alignment():
    p = (1.5, -2.0, 0.7)
    assert compose(p, inverse(p)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    # odom-frame origin in map, from a GT pose and the odom pose at the same instant
    gt, odom = (10.0, 3.0, 1.2), (2.0, 0.5, 0.3)
    base = compose(gt, inverse(odom))
    assert compose(base, odom) == pytest.approx(gt)
