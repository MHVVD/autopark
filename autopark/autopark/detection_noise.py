"""Detection noise injector (for experiments): /slots/detections -> /slots/detections_noisy.

Per detected slot, in the car frame, noise on the entrance position (pos_std, m) and heading
(yaw_std_deg), of one of two kinds (`mode`):
  white   independent Gaussian per detection and frame: the tracker can average it away
  field   systematic: a smooth random error field over the car frame, e(x, y), with that
          standard deviation and spatial wavelength `wavelength` m (like a slightly wrong
          camera calibration or BEV warp). A slot seen from the same viewpoint gets the same
          error in every frame, so averaging does not remove it; seen from elsewhere (e.g.
          from inside the slot instead of from the aisle) it gets a different error. Redrawn,
          reproducibly, for every scenario (/scenario/reset carries the scenario seed).
Detections are dropped with probability `dropout`, vacancy flipped with probability
`vacancy_flip`, and about `false_per_frame` false slots per frame (Poisson) are added at
random poses within 8 m. Seeded, so runs are reproducible.
"""
import math

import numpy as np
import rclpy
from autopark_msgs.msg import ParkingSlot, ParkingSlotArray
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Int64

from autopark import slot_codec as sc


class ErrorField:
    """Smooth zero-mean random fields (ex, ey, etheta) over the plane with standard deviations
    (pos_std, pos_std, yaw_std): random Fourier features, n waves per component with random
    directions and wavelengths within +-30 % of `wavelength`. Stationary: at any point each
    component is a sum of n cosines with random phases, variance std^2."""

    def __init__(self, rng, pos_std, yaw_std, wavelength=6.0, n=16):
        self.std = np.array([pos_std, pos_std, yaw_std])
        a = rng.uniform(0, 2 * np.pi, (3, n))
        k = 2 * np.pi / (wavelength * rng.uniform(0.7, 1.3, (3, n)))
        self.kx, self.ky = k * np.cos(a), k * np.sin(a)
        self.phase = rng.uniform(0, 2 * np.pi, (3, n))
        self.n = n

    def __call__(self, x, y):
        v = np.cos(self.kx * x + self.ky * y + self.phase).sum(1) * np.sqrt(2.0 / self.n)
        return self.std * v


def perturb(slots, rng, pos_std, yaw_std, dropout, vacancy_flip, false_per_frame, field=None):
    """slots: list of (x, y, theta, width, vacancy) in the car frame -> perturbed list.
    With `field` (ErrorField), position / heading errors come from the field at the slot's
    position instead of independent Gaussian noise."""
    out = []
    for x, y, th, w, vac in slots:
        if rng.random() < dropout:
            continue
        if field is not None:
            ex, ey, eth = field(x, y)
            x, y, th = x + ex, y + ey, th + eth
        else:
            x, y = x + rng.normal(0, pos_std), y + rng.normal(0, pos_std)
            th = th + rng.normal(0, yaw_std)
        if vac >= 0 and rng.random() < vacancy_flip:
            vac = 1.0 - vac
        out.append((x, y, th, w, vac))
    for _ in range(rng.poisson(false_per_frame)):
        r, a = rng.uniform(2.0, 8.0), rng.uniform(-math.pi, math.pi)
        out.append((1.35 + r * math.cos(a), r * math.sin(a), rng.uniform(-math.pi, math.pi),
                    2.6, float(rng.random())))
    return out


class DetectionNoise(Node):
    def __init__(self):
        super().__init__('detection_noise')
        self.declare_parameter('pos_std', 0.0)
        self.declare_parameter('yaw_std_deg', 0.0)
        self.declare_parameter('dropout', 0.0)
        self.declare_parameter('vacancy_flip', 0.0)
        self.declare_parameter('false_per_frame', 0.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('mode', 'white')
        self.declare_parameter('wavelength', 6.0)
        p = self.get_parameter
        self.cfg = dict(pos_std=p('pos_std').value, yaw_std=math.radians(p('yaw_std_deg').value),
                        dropout=p('dropout').value, vacancy_flip=p('vacancy_flip').value,
                        false_per_frame=p('false_per_frame').value)
        self.mode, self.wavelength, self.seed = p('mode').value, p('wavelength').value, p('seed').value
        if self.mode not in ('white', 'field'):
            raise ValueError("mode must be 'white' or 'field'")
        self.new_scenario(0)
        self.pub = self.create_publisher(ParkingSlotArray, '/slots/detections_noisy', 10)
        self.create_subscription(ParkingSlotArray, '/slots/detections', self.on_dets, 10)
        self.create_subscription(Int64, '/scenario/reset', lambda m: self.new_scenario(m.data), 10)
        self.get_logger().info(f'noise {self.mode} {self.cfg}')

    def new_scenario(self, scenario_seed):
        """Reproducible noise per scenario: the same scenario gets the same field (and white
        noise sequence) in every run, e.g. for plan once and closed loop."""
        self.rng = np.random.default_rng([self.seed, int(scenario_seed) & 0xFFFFFFFF])
        self.field = (ErrorField(np.random.default_rng([self.seed, int(scenario_seed) & 0xFFFFFFFF, 1]),
                                 self.cfg['pos_std'], self.cfg['yaw_std'], self.wavelength)
                      if self.mode == 'field' else None)

    def on_dets(self, msg):
        slots = [(s.entrance.x, s.entrance.y, s.entrance.theta, s.width, s.vacancy) for s in msg.slots]
        out = ParkingSlotArray()
        out.header = msg.header
        for x, y, th, w, vac in perturb(slots, self.rng, field=self.field, **self.cfg):
            c, s = math.cos(th), math.sin(th)
            lx, ly = -s * w / 2, c * w / 2
            slot = sc.Slot(sc.MarkingPoint(x + lx, y + ly, c, s), sc.MarkingPoint(x - lx, y - ly, c, s), th)
            m = ParkingSlot()
            m.id = -1
            m.corners = [Point(x=float(a), y=float(b), z=0.0) for a, b in slot.corners()]
            m.entrance.x, m.entrance.y, m.entrance.theta = float(x), float(y), float(th)
            m.width, m.depth = float(w), sc.NOMINAL_DEPTH
            m.vacancy = float(vac)
            m.occupied = bool(0 <= vac < 0.5)
            m.confidence = 1.0
            out.slots.append(m)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DetectionNoise()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
