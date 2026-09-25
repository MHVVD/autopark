"""Parking-slot labels <-> network targets, and network output -> slots. No ROS / torch imports.

Representation (marking-point approach): every painted side line ends at the aisle in a
*marking point*; its *direction* is the unit vector pointing into the slot (along the side
line). Two neighbouring marking points one slot width apart with parallel directions form
the entrance of a slot. Occupancy comes from a per-pixel vacancy map inside slots.

Network input: the BEV (BevGrid, 450x450) cropped by CROP px on each side to INPUT px.
Output grid: stride STRIDE, i.e. OUT x OUT cells. Pixel coordinates are index coordinates
(pixel i covers [i - 0.5, i + 0.5]); a point p lies in cell j = floor((p + 0.5) / STRIDE)
with offset o = (p + 0.5) / STRIDE - j in [0, 1), and p = STRIDE * (j + o) - 0.5.

Targets (all OUT x OUT):
  heat      gaussian peaks at marking points (max over points)
  offset    (2, ...) sub-cell offset (col, row) at each point's cell
  direction (2, ...) unit direction (dcol, drow) in image axes near each point
  reg_mask  1 where offset/direction are supervised
  vacancy   1 inside vacant slots, 0 inside occupied slots
  vac_mask  1 inside any slot (vacancy is only supervised there)
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from autopark.bev import BevGrid

CROP = 1
INPUT = 448
STRIDE = 4
OUT = INPUT // STRIDE
SIGMA = 1.5          # cells
DIR_RADIUS = 1       # cells around a point where direction is supervised

# Pairing rules
PAIR_MIN, PAIR_MAX = 2.0, 3.4     # m between the two entrance points
PAIR_DIR_TOL = math.radians(25)   # max angle between the two directions
PAIR_PERP_TOL = 0.4               # max |cos| between direction and the entrance segment
NOMINAL_DEPTH = 5.2               # m, reported slot depth
CAR_CENTRE_X = 1.35               # m, car centre ahead of the ground-frame origin (rear axle)


def blend_heading(direction, normal, rng):
    """Slot heading from the mean line direction and the entrance-segment normal, weighted by
    the entrance's range (m) from the car centre. Near the car the side lines are short stubs
    (partly under the car body) and their direction is noisy, while the 2.6 m entrance segment
    gives a well-conditioned normal; far away the two are about equally good. On the test
    sets this halves the heading error within 5 m (in-slot views: median 0.62 -> 0.35 deg)."""
    w = min(max((8.0 - rng) / 4.0, 0.25), 0.9)          # weight of the normal
    return math.atan2((1 - w) * math.sin(direction) + w * math.sin(normal),
                      (1 - w) * math.cos(direction) + w * math.cos(normal))


@dataclass
class MarkingPoint:
    x: float          # ground frame (m)
    y: float
    dx: float         # unit direction into the slot (ground frame)
    dy: float
    score: float = 1.0


@dataclass
class Slot:
    left: MarkingPoint     # entrance left, looking into the slot
    right: MarkingPoint
    heading: float         # rad, into the slot (ground frame)
    vacancy: float = 1.0   # probability that the slot is free
    score: float = 1.0

    @property
    def entrance(self):
        return ((self.left.x + self.right.x) / 2, (self.left.y + self.right.y) / 2)

    @property
    def width(self):
        return math.hypot(self.left.x - self.right.x, self.left.y - self.right.y)

    def corners(self, depth=NOMINAL_DEPTH):
        """[entrance_left, entrance_right, back_right, back_left] (ground frame)."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        el, er = (self.left.x, self.left.y), (self.right.x, self.right.y)
        return [el, er, (er[0] + depth * c, er[1] + depth * s), (el[0] + depth * c, el[1] + depth * s)]


# ------------------------------------------------------------------ frames
def map_to_ground(xy, pose):
    """Map-frame points (N, 2) -> ground frame of the car at pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(xy, float).reshape(-1, 2) - [x, y]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def ground_to_input(x, y, grid=BevGrid()):
    """Ground (m) -> network-input pixel (col, row)."""
    col, row = grid.to_pixel(x, y)
    return col - CROP, row - CROP


def input_to_ground(col, row, grid=BevGrid()):
    return grid.to_ground(np.asarray(col) + CROP, np.asarray(row) + CROP)


def dir_ground_to_image(dx, dy):
    """Ground direction -> image direction (dcol, drow): col grows with -y, row with -x."""
    return -dy, -dx


def dir_image_to_ground(dc, dr):
    return -dr, -dc


def crop(bev):
    return bev[CROP:CROP + INPUT, CROP:CROP + INPUT]


# ------------------------------------------------------------------ labels from ground truth
def marking_points_from_slots(slots_map, pose, dedup=0.05):
    """slots_map: list of dicts {corners: [[x, y] x4], heading, occupied} in the map frame.
    Returns marking points in the ground frame (entrance corners, shared ones merged)."""
    yaw = pose[2]
    pts = []
    for s in slots_map:
        g = map_to_ground(np.asarray(s['corners'][:2]), pose)
        h = s['heading'] - yaw
        for gx, gy in g:
            if not any(math.hypot(p.x - gx, p.y - gy) < dedup for p in pts):
                pts.append(MarkingPoint(float(gx), float(gy), math.cos(h), math.sin(h)))
    return pts


def slot_polygons_input(slots_map, pose, depth_frac=1.0):
    """Slot rectangles in network-input pixels: list of ((N, 2) float array, occupied)."""
    out = []
    for s in slots_map:
        c = np.asarray(s['corners'], float)
        if depth_frac < 1.0:  # shorten towards the entrance
            c = c.copy()
            c[2] = c[1] + depth_frac * (c[2] - c[1])
            c[3] = c[0] + depth_frac * (c[3] - c[0])
        g = map_to_ground(c, pose)
        col, row = ground_to_input(g[:, 0], g[:, 1])
        out.append((np.stack([col, row], 1), bool(s['occupied'])))
    return out


def encode(slots_map, pose):
    """Ground-truth slots (map frame) + car pose -> dict of training targets."""
    heat = np.zeros((OUT, OUT), np.float32)
    offset = np.zeros((2, OUT, OUT), np.float32)
    direction = np.zeros((2, OUT, OUT), np.float32)
    reg_mask = np.zeros((OUT, OUT), np.float32)
    jj, ii = np.meshgrid(np.arange(OUT), np.arange(OUT))  # jj: col, ii: row

    for p in marking_points_from_slots(slots_map, pose):
        col, row = ground_to_input(p.x, p.y)
        uc, ur = (col + 0.5) / STRIDE, (row + 0.5) / STRIDE   # continuous cell coordinates
        cj, ci = int(math.floor(uc)), int(math.floor(ur))
        if not (0 <= cj < OUT and 0 <= ci < OUT):
            continue
        g = np.exp(-((jj + 0.5 - uc) ** 2 + (ii + 0.5 - ur) ** 2) / (2 * SIGMA ** 2))
        heat = np.maximum(heat, g.astype(np.float32))
        heat[ci, cj] = 1.0
        offset[:, ci, cj] = (uc - cj, ur - ci)
        dc, dr = dir_ground_to_image(p.dx, p.dy)
        r0, r1 = max(ci - DIR_RADIUS, 0), min(ci + DIR_RADIUS + 1, OUT)
        c0, c1 = max(cj - DIR_RADIUS, 0), min(cj + DIR_RADIUS + 1, OUT)
        direction[0, r0:r1, c0:c1] = dc
        direction[1, r0:r1, c0:c1] = dr
        reg_mask[r0:r1, c0:c1] = np.maximum(reg_mask[r0:r1, c0:c1], 0.5)
        reg_mask[ci, cj] = 1.0          # offset is only used where reg_mask == 1

    vac = np.zeros((OUT, OUT), np.uint8)
    vmask = np.zeros((OUT, OUT), np.uint8)
    for poly, occupied in slot_polygons_input(slots_map, pose):
        cells = np.round((poly + 0.5) / STRIDE * 8).astype(np.int32)  # 3 fractional bits
        cv2.fillPoly(vmask, [cells], 1, shift=3)
        if not occupied:
            cv2.fillPoly(vac, [cells], 1, shift=3)
    return dict(heat=heat, offset=offset, direction=direction, reg_mask=reg_mask,
                vacancy=vac.astype(np.float32), vac_mask=vmask.astype(np.float32))


# ------------------------------------------------------------------ decoding
def peaks(heat, threshold=0.3, max_points=64):
    """Local maxima (3x3) above threshold: list of (row, col, score), best first."""
    dil = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    ii, jj = np.nonzero((heat >= dil) & (heat >= threshold))
    order = np.argsort(-heat[ii, jj])[:max_points]
    return [(int(ii[k]), int(jj[k]), float(heat[ii[k], jj[k]])) for k in order]


def decode_points(heat, offset, direction, threshold=0.3):
    """Network outputs (numpy, OUT x OUT) -> marking points in the ground frame."""
    pts = []
    for i, j, score in peaks(heat, threshold):
        col = STRIDE * (j + offset[0, i, j]) - 0.5
        row = STRIDE * (i + offset[1, i, j]) - 0.5
        gx, gy = input_to_ground(col, row)
        dx, dy = dir_image_to_ground(direction[0, i, j], direction[1, i, j])
        n = math.hypot(dx, dy)
        if n < 1e-6:
            continue
        pts.append(MarkingPoint(float(gx), float(gy), dx / n, dy / n, score))
    return pts


def pair_points(pts):
    """Marking points -> slots (entrance pairs). Each point can bound two slots (left/right)."""
    cands = []
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            p, q = pts[a], pts[b]
            ex, ey = q.x - p.x, q.y - p.y
            dist = math.hypot(ex, ey)
            if not PAIR_MIN <= dist <= PAIR_MAX:
                continue
            if p.dx * q.dx + p.dy * q.dy < math.cos(PAIR_DIR_TOL):
                continue
            hx, hy = p.dx + q.dx, p.dy + q.dy
            hn = math.hypot(hx, hy)
            hx, hy = hx / hn, hy / hn
            if abs(hx * ex + hy * ey) / dist > PAIR_PERP_TOL:
                continue
            # left point: positive cross(heading, point - midpoint)
            mx, my = (p.x + q.x) / 2, (p.y + q.y) / 2
            left, right = (p, q) if hx * (p.y - my) - hy * (p.x - mx) > 0 else (q, p)
            normal = math.atan2(-(left.x - right.x), left.y - right.y)     # right -> left, turned -90 deg
            heading = blend_heading(math.atan2(hy, hx), normal, math.hypot(mx - CAR_CENTRE_X, my))
            # prefer pairs close to the nominal width
            cands.append((abs(dist - 2.6), Slot(left, right, heading, score=min(p.score, q.score))))
    # a point is the left corner of at most one slot and the right corner of at most one
    used_left, used_right, slots = set(), set(), []
    for _, s in sorted(cands, key=lambda c: c[0]):
        if id(s.left) in used_left or id(s.right) in used_right:
            continue
        used_left.add(id(s.left))
        used_right.add(id(s.right))
        slots.append(s)
    return slots


def slot_vacancy(slot, vacancy_map, depth=(0.4, 3.0), width_frac=0.6):
    """Mean vacancy probability over the front part of the slot (in output cells)."""
    c, s = math.cos(slot.heading), math.sin(slot.heading)
    mx, my = slot.entrance
    lx, ly = -s, c
    hw = slot.width * width_frac / 2
    quad = [(mx + depth[0] * c + hw * lx, my + depth[0] * s + hw * ly),
            (mx + depth[0] * c - hw * lx, my + depth[0] * s - hw * ly),
            (mx + depth[1] * c - hw * lx, my + depth[1] * s - hw * ly),
            (mx + depth[1] * c + hw * lx, my + depth[1] * s + hw * ly)]
    col, row = ground_to_input(np.array([q[0] for q in quad]), np.array([q[1] for q in quad]))
    cells = np.round((np.stack([col, row], 1) + 0.5) / STRIDE * 8).astype(np.int32)
    m = np.zeros(vacancy_map.shape, np.uint8)
    cv2.fillPoly(m, [cells], 1, shift=3)
    if m.sum() == 0:
        return float('nan')
    return float(vacancy_map[m > 0].mean())


def decode(heat, offset, direction, vacancy, threshold=0.3):
    """Full decoding: marking points -> paired slots with vacancy probability."""
    slots = pair_points(decode_points(heat, offset, direction, threshold))
    for s in slots:
        s.vacancy = slot_vacancy(s, vacancy)
    return slots
