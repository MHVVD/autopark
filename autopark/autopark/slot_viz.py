"""Draw marking points and slots on a BEV image (for label checks, debugging and the demo)."""
import math

import cv2
import numpy as np

from autopark import slot_codec as sc
from autopark.bev import BevGrid

GRID = BevGrid()


def _px(x, y):
    col, row = GRID.to_pixel(x, y)
    return int(round(float(col))), int(round(float(row)))


def draw_points(img, pts, color=(0, 255, 255), arrow=0.8):
    for p in pts:
        a = _px(p.x, p.y)
        b = _px(p.x + arrow * p.dx, p.y + arrow * p.dy)
        cv2.circle(img, a, 4, color, 1, cv2.LINE_AA)
        cv2.arrowedLine(img, a, b, color, 1, cv2.LINE_AA, tipLength=0.3)
    return img


def draw_slots(img, slots, depth=sc.NOMINAL_DEPTH, label=True):
    """Green = vacant, red = occupied (vacancy > 0.5 means vacant)."""
    for s in slots:
        vac = s.vacancy if not math.isnan(s.vacancy) else 0.0
        color = (0, 220, 0) if vac > 0.5 else (0, 0, 230)
        poly = np.array([_px(x, y) for x, y in s.corners(depth)], np.int32)
        cv2.polylines(img, [poly], True, color, 1, cv2.LINE_AA)
        cv2.line(img, tuple(poly[0]), tuple(poly[1]), color, 2, cv2.LINE_AA)  # entrance
        if label:
            ex, ey = s.entrance
            cv2.putText(img, f'{vac:.2f}', _px(ex + 1.2 * math.cos(s.heading), ey + 1.2 * math.sin(s.heading)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def gt_slots_ground(slots_map, pose):
    """Ground-truth slots (map-frame dicts) -> codec Slot objects in the ground frame."""
    out = []
    for s in slots_map:
        g = sc.map_to_ground(np.asarray(s['corners'][:2]), pose)
        h = s['heading'] - pose[2]
        dx, dy = math.cos(h), math.sin(h)
        left = sc.MarkingPoint(g[0, 0], g[0, 1], dx, dy)
        right = sc.MarkingPoint(g[1, 0], g[1, 1], dx, dy)
        out.append(sc.Slot(left, right, math.atan2(dy, dx), vacancy=0.0 if s['occupied'] else 1.0))
    return out


def draw_path(img, xs, ys, dirs, footprints=()):
    """Path in the ground frame: forward blue, reverse orange; `footprints` are car corner lists
    (e.g. the goal pose) drawn as outlines."""
    pts = [_px(x, y) for x, y in zip(xs, ys)]
    for i in range(1, len(pts)):
        color = (255, 128, 0) if dirs[i] > 0 else (0, 140, 255)
        cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    for i in range(1, len(dirs) - 1):
        if dirs[i + 1] != dirs[i]:
            cv2.circle(img, pts[i], 5, (255, 255, 255), 1, cv2.LINE_AA)       # cusp
    for fp in footprints:
        poly = np.array([_px(x, y) for x, y in fp], np.int32)
        cv2.polylines(img, [poly], True, (255, 0, 255), 1, cv2.LINE_AA)
    return img
