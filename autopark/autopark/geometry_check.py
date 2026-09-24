"""Exact 2D checks for evaluating paths: rectangle-rectangle distance and in-slot tests. No ROS.

Independent of the planner's own (grid-based) collision model, so the evaluation does not
inherit its approximations.
"""
import math

import numpy as np


def _seg_dist(p, q, a, b):
    """Distance between segments pq and ab (2D)."""
    if _seg_intersect(p, q, a, b):
        return 0.0
    return min(_pt_seg(p, a, b), _pt_seg(q, a, b), _pt_seg(a, p, q), _pt_seg(b, p, q))


def _pt_seg(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    n2 = ax * ax + ay * ay
    t = 0.0 if n2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / n2))
    return math.hypot(p[0] - a[0] - t * ax, p[1] - a[1] - t * ay)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _seg_intersect(p, q, a, b):
    """Proper crossing only; touching and collinear cases give distance 0 via _pt_seg."""
    return (_cross(a, b, p) * _cross(a, b, q) < 0) and (_cross(p, q, a) * _cross(p, q, b) < 0)


def point_in_convex(pt, poly):
    sign = 0
    for k in range(len(poly)):
        c = _cross(poly[k], poly[(k + 1) % len(poly)], pt)
        if c != 0:
            if sign == 0:
                sign = 1 if c > 0 else -1
            elif (c > 0) != (sign > 0):
                return False
    return sign != 0          # a degenerate (zero-area) polygon contains nothing


def polygon_distance(a, b):
    """Distance between two convex polygons (0 if they overlap)."""
    if point_in_convex(a[0], b) or point_in_convex(b[0], a):
        return 0.0
    best = math.inf
    for i in range(len(a)):
        p, q = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            best = min(best, _seg_dist(p, q, b[j], b[(j + 1) % len(b)]))
            if best == 0.0:
                return 0.0
    return best


def rect(cx, cy, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((-length / 2, -width / 2), (length / 2, -width / 2),
                           (length / 2, width / 2), (-length / 2, width / 2))]


def path_clearance(car, xs, ys, yaws, obstacles, stride=1):
    """Smallest exact distance from the car rectangle, at every `stride`-th pose (and the last),
    to any obstacle polygon. Returns (distance, index of the pose)."""
    idx = list(range(0, len(xs), stride))
    if idx[-1] != len(xs) - 1:
        idx.append(len(xs) - 1)
    centres = [(np.mean([p[0] for p in o]), np.mean([p[1] for p in o]),
                max(math.hypot(p[0] - np.mean([q[0] for q in o]), p[1] - np.mean([q[1] for q in o])) for p in o))
               for o in obstacles]
    reach = car.front + car.rear + car.half_width
    best, where = math.inf, -1
    for i in idx:
        body = car.corners((xs[i], ys[i], yaws[i]))
        for o, (ox, oy, orad) in zip(obstacles, centres):
            if math.hypot(ox - xs[i], oy - ys[i]) - orad - reach > best:
                continue
            d = polygon_distance(body, o)
            if d < best:
                best, where = d, i
    return best, where
