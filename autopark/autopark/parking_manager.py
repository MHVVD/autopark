"""Parking manager logic (no ROS): aisle estimate for searching, target selection, path
splitting at cusps and small path corrections. Used by parking_manager_node.

Frames: everything in odom; poses are rear-axle (x, y, yaw); slots are parking_goal.SlotSpec.
"""
import math
from types import SimpleNamespace

import numpy as np

from autopark import parking_goal as pg

HALF_AISLE = 3.5      # m, nominal distance from a row's entrance line to the aisle centre


def aisle_line(slots, pose, half_aisle=HALF_AISLE):
    """Aisle centre line from the tracked slots, as (point, unit direction), the direction
    pointing along the car's heading. None if there are no slots.

    Each slot gives a centre point half an aisle in front of its entrance; the line runs
    perpendicular to the slot headings (median, folded modulo pi) through their median lateral
    offset, so one mis-tracked slot cannot pull it."""
    if not slots:
        return None
    folded = sorted(((s.theta + math.pi / 2) % math.pi) for s in slots)
    a = folded[len(folded) // 2]
    u = np.array([math.cos(a), math.sin(a)])
    if u @ [math.cos(pose[2]), math.sin(pose[2])] < 0:
        u = -u
    n = np.array([-u[1], u[0]])
    centres = np.array([[s.x - half_aisle * math.cos(s.theta), s.y - half_aisle * math.sin(s.theta)]
                        for s in slots])
    p0 = np.array(pose[:2])
    offset = float(np.median((centres - p0) @ n))
    return p0 + offset * n, u


def pure_pursuit_to_line(pose, line, lookahead=5.0, wheel_base=2.8, max_steer=0.5):
    """Steering (rad, + left) that drives the rear axle onto the line, looking `lookahead` m
    ahead along it."""
    p, u = line
    x, y, yaw = pose
    along = (np.array([x, y]) - p) @ u
    target = p + (along + lookahead) * u
    dx, dy = target[0] - x, target[1] - y
    ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
    ld2 = dx * dx + dy * dy
    return max(-max_steer, min(max_steer, math.atan(2 * ly * wheel_base / ld2)))


def passed_by(slot, pose, u):
    """How far (m) the rear axle is past the slot entrance along the aisle direction u."""
    return float((np.array(pose[:2]) - [slot.x, slot.y]) @ u)


def choose_target(slots, pose, u, pass_distance, excluded=()):
    """The selectable-vacant slot the car passed most recently by at least `pass_distance`."""
    cands = [(passed_by(s, pose, u), s) for s in slots if s.vacant and s.id not in excluded]
    cands = [(d, s) for d, s in cands if d >= pass_distance]
    return min(cands, key=lambda c: c[0])[1] if cands else None


def first_segment(path):
    """The path up to and including its first cusp (the whole path if it has none), with the
    same fields as the ParkingPath-like input (x, y, yaw, direction, curvature lists)."""
    d = list(path.direction)
    end = len(d)
    for i in range(1, len(d)):
        if d[i] != d[i - 1]:
            end = i                  # samples 0..i-1: sample i-1 is the cusp
            break
    return SimpleNamespace(**{k: list(getattr(path, k))[:end] for k in ('x', 'y', 'yaw', 'direction', 'curvature')}), \
        end == len(d)


def remainder(path, start):
    """The path from sample `start` on (same fields as first_segment)."""
    return SimpleNamespace(**{k: list(getattr(path, k))[start:] for k in ('x', 'y', 'yaw', 'direction', 'curvature')})


def rigid_correction(goal_old, goal_new):
    """Transform (dx, dy, dyaw about the old goal) that moves goal_old onto goal_new."""
    return goal_new[0] - goal_old[0], goal_new[1] - goal_old[1], pg_wrap(goal_new[2] - goal_old[2])


def correction_size(goal_old, goal_new, lever=2.0):
    """Size (m) of the correction goal_old -> goal_new: translation + lever * |rotation|
    (1 deg ~ 3.5 cm at 2 m)."""
    dx, dy, dth = rigid_correction(goal_old, goal_new)
    return math.hypot(dx, dy) + lever * abs(dth)


def apply_correction(path, goal_old, corr):
    """Move path samples rigidly: rotate by dyaw about goal_old, then translate by (dx, dy).
    Curvature and direction are unchanged."""
    dx, dy, dth = corr
    c, s = math.cos(dth), math.sin(dth)
    gx, gy = goal_old[0], goal_old[1]
    xs = np.asarray(path.x) - gx
    ys = np.asarray(path.y) - gy
    return SimpleNamespace(x=(gx + c * xs - s * ys + dx).tolist(), y=(gy + s * xs + c * ys + dy).tolist(),
                           yaw=[pg_wrap(t + dth) for t in path.yaw], direction=list(path.direction),
                           curvature=list(path.curvature))


def pg_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def goal_for(slot, car=None):
    return pg.goal_pose(slot, car)
