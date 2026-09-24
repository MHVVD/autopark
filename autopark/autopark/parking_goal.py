"""From parking slots to a planning problem: goal pose and obstacles. No ROS imports.

A slot is given by its entrance centre (x, y), heading theta pointing INTO the slot, width and
depth (the tracker's output, in odom). Reverse perpendicular parking: at the goal the car is
centred in the slot and faces out of it, so it must enter backwards.

Obstacles come only from perception (the tracked slots), never from ground truth:
  * every slot that is not known to be vacant (occupied, or vacancy not yet confident) is a
    keep-out rectangle, grown to bound any car parked in it (`grow_entrance` towards the aisle,
    `grow_side` past the side lines). This bound is an assumption about how cars are parked;
    in the simulated lot the worst case over 3000 scenarios is 0.20 m and 0.03 m;
  * behind every slot there is a keep-out band (the rows' back edges; the planner must not
    drive through the end of an empty slot into unknown space).
  * space the cameras have not explored is keep-out (`known_region`): a slot that was never in
    the detector's view is not tracked, so without this rule an unseen occupied slot would
    look drivable.
Everything else (the explored aisle and vacant slots) is drivable.
"""
import math
from dataclasses import dataclass

from autopark.hybrid_astar import CarGeometry


# A tracked slot counts as vacant (a parking target, and free space for the planner) only when
# the fused vacancy is high AND it was seen from close range at least 3 times (milestone 4:
# 100 % precision with this rule at every noise level tested).
SELECT_VACANCY = 0.95
SELECT_CONFIDENCE = 0.3


def selectable(vacancy, confidence):
    return vacancy > SELECT_VACANCY and confidence >= SELECT_CONFIDENCE - 1e-6


@dataclass
class SlotSpec:
    id: int
    x: float            # entrance centre
    y: float
    theta: float        # into the slot
    width: float = 2.6
    depth: float = 5.2
    vacant: bool = False


def slot_rect(s, grow_entrance=0.0, grow_side=0.0, grow_back=0.0, start=0.0, end=None):
    """Rectangle of a slot as 4 (x, y) corners. Depth runs from `start` to `end` (default: the
    slot depth) measured from the entrance into the slot, then grown."""
    end = s.depth if end is None else end
    c, sn = math.cos(s.theta), math.sin(s.theta)
    fx, fy, lx, ly = c, sn, -sn, c
    d0, d1 = start - grow_entrance, end + grow_back
    hw = s.width / 2 + grow_side
    return [(s.x + d * fx + w * lx, s.y + d * fy + w * ly)
            for d, w in ((d0, hw), (d0, -hw), (d1, -hw), (d1, hw))]


def goal_pose(s, car=None, depth_offset=0.0):
    """Rear-axle pose that centres the car in the slot, facing out (reverse-in parking).
    depth_offset > 0 moves the car deeper into the slot."""
    car = car or CarGeometry()
    centre_to_axle = (car.front - car.rear) / 2           # car centre is ahead of the rear axle
    d = s.depth / 2 + centre_to_axle + depth_offset       # rear axle depth from the entrance
    c, sn = math.cos(s.theta), math.sin(s.theta)
    yaw = math.atan2(-sn, -c)
    return s.x + d * c, s.y + d * sn, yaw


def obstacles(slots, target_id, grow_entrance=0.25, grow_side=0.05, back_band=2.0):
    """Keep-out polygons for planning into slot `target_id`."""
    polys = []
    for s in slots:
        if s.id != target_id and not s.vacant:
            polys.append(slot_rect(s, grow_entrance, grow_side))
        polys.append(slot_rect(s, start=s.depth, end=s.depth + back_band, grow_side=0.05))
    return polys


# Explored area. The tracker treats a slot as in view (autopark.slot_tracker.in_view) when its
# entrance is inside the BEV with a 1.5 m margin and within 9 m of the car centre; such slots
# get tracked within 3 frames and are obstacles unless vacant. KNOWN_BOX is a car-frame box
# (rear axle; x_min, x_max, y_min, y_max) such that every slot reaching into it had its
# entrance in view, provided the car heads along the aisle within KNOWN_YAW_TOL (either way)
# and is within 1.5 m of the aisle centre. Checked exhaustively in test_planner.py; the front
# edge also leaves 2 m for the tracker's confirmation delay while driving.
KNOWN_BOX = (-3.5, 5.5, -9.0, 9.0)
KNOWN_YAW_TOL = math.radians(10.0)


def aisle_aligned(yaw, slot_headings, tol=KNOWN_YAW_TOL):
    """Whether a car heading is along the aisle (either direction). The aisle runs
    perpendicular to the slot headings (their median, folded modulo pi)."""
    if not slot_headings:
        return False
    folded = sorted(((h + math.pi / 2) % math.pi) for h in slot_headings)
    aisle = folded[len(folded) // 2]
    d = (yaw - aisle) % math.pi
    return min(d, math.pi - d) <= tol


def known_region(poses, box=KNOWN_BOX):
    """Explored area as convex polygons: the box at each pose the car has been at."""
    x0, x1, y0, y1 = box
    out = []
    for x, y, th in poses:
        c, s = math.cos(th), math.sin(th)
        out.append([(x + c * px - s * py, y + s * px + c * py)
                    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    return out


def window(points, margin=8.0):
    """Axis-aligned planning window (x_min, y_min, x_max, y_max) around points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin
