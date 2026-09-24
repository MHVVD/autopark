"""Deterministic scenario randomisation (which slots are empty, parked cars, ego start pose).

No ROS or Webots imports. The same seed always produces the same scenario, so experiments
are reproducible.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from autopark_sim.lot import slots

# Webots "Simple" car PROTOs (visual + bounding box, no physics).
# offset: distance from the PROTO origin (rear axle) to the car's geometric centre, m.
# length/width: approximate footprint, m.
CAR_MODELS = {
    'TeslaModel3Simple': dict(offset=1.37, length=4.84, width=1.85),
    'BmwX5Simple': dict(offset=1.43, length=4.85, width=1.94),
    'CitroenCZeroSimple': dict(offset=1.33, length=3.50, width=1.48),
    'LincolnMKZSimple': dict(offset=1.44, length=5.00, width=1.86),
    'RangeRoverSportSVRSimple': dict(offset=1.22, length=4.88, width=1.98),
    'ToyotaPriusSimple': dict(offset=1.39, length=4.49, width=1.76),
}
CAR_Z = 0.4  # m, height of the parked-car PROTO origin (no physics; verified on camera)
# The physical ToyotaPrius origin rests at z = 0.025 (measured); spawning higher makes it drop.
EGO_Z = 0.03

COLORS = [(0.7, 0.05, 0.05), (0.1, 0.2, 0.6), (0.85, 0.85, 0.85), (0.1, 0.1, 0.1),
          (0.5, 0.5, 0.52), (0.8, 0.7, 0.1), (0.15, 0.4, 0.2), (0.95, 0.95, 0.95)]

# Ego start pose ranges (map frame): in the aisle, before the rows, heading +x.
EGO_X = (-17.0, -14.0)
EGO_Y = (-0.8, 0.8)
EGO_YAW = (-0.08, 0.08)

# Parked-car jitter inside its slot.
JITTER_LATERAL = 0.2    # m
JITTER_DEPTH = 0.25     # m
JITTER_YAW = 0.06       # rad


@dataclass
class ParkedCar:
    slot_id: int
    model: str
    x: float          # PROTO origin (rear axle), map frame
    y: float
    yaw: float
    color: tuple

    def webots_string(self):
        r, g, b = self.color
        return (f'DEF PARKED_{self.slot_id} {self.model} {{ '
                f'translation {self.x:.4f} {self.y:.4f} {CAR_Z} '
                f'rotation 0 0 1 {self.yaw:.5f} '
                f'name "parked car {self.slot_id}" color {r} {g} {b} }}')


@dataclass
class Scenario:
    seed: int
    parked: list = field(default_factory=list)
    empty_slot_ids: list = field(default_factory=list)
    ego: tuple = (0.0, 0.0, 0.0)   # x, y, yaw of the ego rear axle


EMPTY_LOT_SEED = -1  # special seed: no parked cars (calibration / evaluation of line geometry)


def make_scenario(seed, n_empty=(1, 3)):
    """Random but reproducible scenario. n_empty: inclusive range of empty slots.
    seed == EMPTY_LOT_SEED gives an empty lot with the ego at its nominal start pose."""
    all_slots = slots()
    if seed == EMPTY_LOT_SEED:
        return Scenario(seed, [], [s.id for s in all_slots], (-15.5, 0.0, 0.0))
    rng = np.random.default_rng(seed)
    k = int(rng.integers(n_empty[0], n_empty[1] + 1))
    empty = sorted(int(i) for i in rng.choice(len(all_slots), size=k, replace=False))

    parked = []
    for s in all_slots:
        if s.id in empty:
            continue
        model = str(rng.choice(sorted(CAR_MODELS)))
        nose_in = bool(rng.random() < 0.5)
        lat = rng.uniform(-JITTER_LATERAL, JITTER_LATERAL)
        dep = rng.uniform(-JITTER_DEPTH, JITTER_DEPTH)
        dyaw = rng.uniform(-JITTER_YAW, JITTER_YAW)
        color = COLORS[int(rng.integers(len(COLORS)))]

        cx, cy = s.center()
        c, sn = math.cos(s.heading), math.sin(s.heading)
        cx += dep * c - lat * sn
        cy += dep * sn + lat * c
        yaw = s.heading + dyaw + (0.0 if nose_in else math.pi)
        off = CAR_MODELS[model]['offset']
        x = cx - off * math.cos(yaw)
        y = cy - off * math.sin(yaw)
        parked.append(ParkedCar(s.id, model, x, y, _wrap(yaw), color))

    ego = (rng.uniform(*EGO_X), rng.uniform(*EGO_Y), rng.uniform(*EGO_YAW))
    return Scenario(seed, parked, empty, ego)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
