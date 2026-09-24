"""Slot detector node: /bev/image -> /slots/detections (autopark_msgs/ParkingSlotArray).

Slots are in the BEV's frame (base_footprint, the ground below base_link) with the BEV's
stamp. corners = [entrance_left, entrance_right, back_right, back_left] (nominal depth),
entrance = centre of the entrance + heading into the slot, occupied = vacancy < 0.5,
confidence = the weaker of the two marking-point scores. Frames arriving while the network
is busy are dropped (queue depth 1). Publishes an overlay on /slots/debug_image when
`debug_image` is true.
"""
import math
import os
import time

import rclpy
import torch
from autopark_msgs.msg import ParkingSlot, ParkingSlotArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from autopark import slot_codec as sc
from autopark.slot_net import load_model, predict
from autopark.slot_viz import draw_points, draw_slots


def to_msg(slot):
    m = ParkingSlot()
    m.id = -1
    m.corners = [Point(x=float(x), y=float(y), z=0.0) for x, y in slot.corners()]
    m.entrance.x, m.entrance.y = (float(v) for v in slot.entrance)
    m.entrance.theta = float(slot.heading)
    m.width = float(slot.width)
    m.depth = sc.NOMINAL_DEPTH
    vac = 0.0 if math.isnan(slot.vacancy) else slot.vacancy
    m.occupied = bool(vac < 0.5)
    m.confidence = float(slot.score)
    return m


class SlotDetector(Node):
    def __init__(self):
        super().__init__('slot_detector')
        self.declare_parameter('model', os.path.expanduser('~/autopark_models/slotnet.pt'))
        self.declare_parameter('threshold', 0.3)
        self.declare_parameter('threads', 2)
        self.declare_parameter('debug_image', True)
        torch.set_num_threads(self.get_parameter('threads').value)
        path = self.get_parameter('model').value
        if not os.path.isfile(path):
            raise FileNotFoundError(f'slot detector model not found: {path} (train it with '
                                    'train_slots, or pass -p model:=<file>)')
        self.net = load_model(path)
        self.threshold = self.get_parameter('threshold').value
        self.debug = self.get_parameter('debug_image').value
        self.bridge = CvBridge()
        self.pub = self.create_publisher(ParkingSlotArray, '/slots/detections', 10)
        self.dbg = self.create_publisher(Image, '/slots/debug_image', 2)
        self.create_subscription(Image, '/bev/image', self.on_bev,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.n, self.t_sum = 0, 0.0
        self.get_logger().info(f'model {self.get_parameter("model").value}')

    def on_bev(self, msg):
        t0 = time.perf_counter()
        bev = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        slots, pts, _ = predict(self.net, bev, self.threshold)
        out = ParkingSlotArray()
        out.header = msg.header
        out.slots = [to_msg(s) for s in slots]
        self.pub.publish(out)
        if self.debug and self.dbg.get_subscription_count() > 0:
            img = draw_points(draw_slots(bev.copy(), slots), pts)
            d = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            d.header = msg.header
            self.dbg.publish(d)
        self.n += 1
        self.t_sum += time.perf_counter() - t0
        if self.n % 100 == 0:
            self.get_logger().info(f'{self.n} frames, mean {1000 * self.t_sum / self.n:.0f} ms/frame')


def main():
    rclpy.init()
    node = SlotDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
