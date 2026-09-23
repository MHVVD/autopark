"""Parking-lot geometry: the single source of truth for the world file and the ground truth.

No ROS or Webots imports. Frame 'map': x along the aisle, y across it, z up, origin at the
aisle centre. Two rows of perpendicular slots face each other across the aisle:

    row A (ids 0..7):  entrances on y = +AISLE_WIDTH/2, slots extend towards +y
    row B (ids 8..15): entrances on y = -AISLE_WIDTH/2, slots extend towards -y

All slot dimensions are measured between painted-line centres.
"""
import math
from dataclasses import dataclass

SLOT_WIDTH = 2.6       # m
SLOT_DEPTH = 5.2       # m
LINE_WIDTH = 0.12      # m, painted line width
SLOTS_PER_ROW = 8
AISLE_WIDTH = 7.0      # m, between the two entrance lines
ROW_X0 = -SLOTS_PER_ROW * SLOT_WIDTH / 2  # x of the first side line in each row


@dataclass(frozen=True)
class Slot:
    id: int
    entrance_x: float      # centre of the entrance (open end)
    entrance_y: float
    heading: float         # rad, direction pointing into the slot
    width: float = SLOT_WIDTH
    depth: float = SLOT_DEPTH

    def _axes(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (c, s), (-s, c)  # into-slot unit vector, left unit vector (looking into the slot)

    def corners(self):
        """[entrance_left, entrance_right, back_right, back_left] as (x, y) tuples."""
        (fx, fy), (lx, ly) = self._axes()
        hw, d = self.width / 2, self.depth
        ex, ey = self.entrance_x, self.entrance_y
        el = (ex + hw * lx, ey + hw * ly)
        er = (ex - hw * lx, ey - hw * ly)
        return [el, er, (er[0] + d * fx, er[1] + d * fy), (el[0] + d * fx, el[1] + d * fy)]

    def center(self):
        (fx, fy), _ = self._axes()
        return (self.entrance_x + self.depth / 2 * fx, self.entrance_y + self.depth / 2 * fy)


def slots():
    """All 16 slots, ids 0..15."""
    out = []
    for row, (y0, heading) in enumerate([(AISLE_WIDTH / 2, math.pi / 2),
                                         (-AISLE_WIDTH / 2, -math.pi / 2)]):
        for i in range(SLOTS_PER_ROW):
            x = ROW_X0 + (i + 0.5) * SLOT_WIDTH
            out.append(Slot(row * SLOTS_PER_ROW + i, x, y0, heading))
    return out


def painted_lines():
    """Painted line boxes as (centre_x, centre_y, size_x, size_y). Axis-aligned.

    Per row: SLOTS_PER_ROW + 1 side lines running from the entrance to the back line,
    and one back line. The entrance side is open.
    """
    lines = []
    for sign in (+1, -1):
        y_entr = sign * AISLE_WIDTH / 2
        y_back = y_entr + sign * SLOT_DEPTH
        for k in range(SLOTS_PER_ROW + 1):
            x = ROW_X0 + k * SLOT_WIDTH
            lines.append((x, (y_entr + y_back) / 2, LINE_WIDTH, SLOT_DEPTH))
        row_len = SLOTS_PER_ROW * SLOT_WIDTH + LINE_WIDTH
        lines.append((0.0, y_back, row_len, LINE_WIDTH))
    return lines
