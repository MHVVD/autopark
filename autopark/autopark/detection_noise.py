"""Detection noise injector (for experiments): /slots/detections -> /slots/detections_noisy.

Per detected slot, in the car frame: Gaussian noise on the entrance position (pos_std, m)
and heading (yaw_std_deg), dropped with probability `dropout`, vacancy flipped with
probability `vacancy_flip`. Additionally about `false_per_frame` false slots per frame
(Poisson) are added at random poses within 8 m. Seeded, so runs are reproducible.
"""
import math

import numpy as np
import rclpy
from autopark_msgs.msg import ParkingSlot, ParkingSlotArray
from geometry_msgs.msg import Point
from rclpy.node import Node

from autopark import slot_codec as sc


def perturb(slots, rng, pos_std, yaw_std, dropout, vacancy_flip, false_per_frame):
    """slots: list of (x, y, theta, width, vacancy) in the car frame -> perturbed list."""
    out = []
    for x, y, th, w, vac in slots:
        if rng.random() < dropout:
            continue
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
        p = self.get_parameter
        self.cfg = dict(pos_std=p('pos_std').value, yaw_std=math.radians(p('yaw_std_deg').value),
                        dropout=p('dropout').value, vacancy_flip=p('vacancy_flip').value,
                        false_per_frame=p('false_per_frame').value)
        self.rng = np.random.default_rng(p('seed').value)
        self.pub = self.create_publisher(ParkingSlotArray, '/slots/detections_noisy', 10)
        self.create_subscription(ParkingSlotArray, '/slots/detections', self.on_dets, 10)
        self.get_logger().info(f'noise {self.cfg}')

    def on_dets(self, msg):
        slots = [(s.entrance.x, s.entrance.y, s.entrance.theta, s.width, s.vacancy) for s in msg.slots]
        out = ParkingSlotArray()
        out.header = msg.header
        for x, y, th, w, vac in perturb(slots, self.rng, **self.cfg):
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
