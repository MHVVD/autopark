"""Slot tracker node: /slots/detections + /odom -> /slots/tracked (odom frame).

Each detection (base_footprint frame, BEV stamp) is moved into odom with the odometry pose at
its stamp and fused by autopark.slot_tracker. Published slots are the confirmed tracks:
  id = track id, entrance/corners in odom, covariance = 3x3 of (x, y, theta),
  vacancy = fused probability of being free, occupied = vacancy < 0.5,
  confidence = number of close-range vacancy observations / 10 (capped at 1).
/slots/tracked_image draws the tracks (in the current car frame) on the latest BEV.

Parameters: detections_topic, extra_pos_std / extra_yaw_std (added to the detector error
model, e.g. when noise is injected), view_margin, view_range, view_yaw_tol_deg (detections
of slots seen at more than this from perpendicular are ignored and do not count as misses).
"""
import math

import rclpy
from autopark_msgs.msg import ParkingSlot, ParkingSlotArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int64

from autopark import slot_codec as sc
from autopark.odometry import compose, inverse, yaw_from_quaternion
from autopark.slot_tracker import (CAR_CENTRE_X, VIEW_MARGIN, VIEW_RANGE, VIEW_YAW_TOL, Detection, SlotTracker,
                                   heading_valid, in_view)
from autopark.slot_viz import draw_slots

STEP_MS = 20


def _ms(stamp):
    return int(round((stamp.sec + stamp.nanosec * 1e-9) * 1000 / STEP_MS)) * STEP_MS


class SlotTrackerNode(Node):
    def __init__(self):
        super().__init__('slot_tracker')
        self.declare_parameter('detections_topic', '/slots/detections')
        self.declare_parameter('extra_pos_std', 0.0)
        self.declare_parameter('extra_yaw_std_deg', 0.0)
        self.declare_parameter('view_margin', VIEW_MARGIN)
        self.declare_parameter('view_range', VIEW_RANGE)
        self.declare_parameter('view_yaw_tol_deg', math.degrees(VIEW_YAW_TOL))
        p = self.get_parameter
        self.tracker = SlotTracker(extra_pos_std=p('extra_pos_std').value,
                                   extra_yaw_std=math.radians(p('extra_yaw_std_deg').value))
        self.view_margin = p('view_margin').value
        self.view_range = p('view_range').value
        self.yaw_tol = math.radians(p('view_yaw_tol_deg').value)

        self.odom = {}            # ms -> (x, y, yaw)
        self.last_pose = None     # odom pose at the last tracker update
        self.bev = None
        self.bridge = CvBridge()
        self.pub = self.create_publisher(ParkingSlotArray, '/slots/tracked', 10)
        self.img_pub = self.create_publisher(Image, '/slots/tracked_image', 2)
        self.create_subscription(Odometry, '/odom', self.on_odom, 200)
        self.create_subscription(ParkingSlotArray, p('detections_topic').value, self.on_dets, 10)
        self.create_subscription(Int64, '/scenario/reset', self.on_reset, 10)
        self.create_subscription(Image, '/bev/image', self.on_bev,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))

    def on_reset(self, _msg):
        self.tracker.reset()
        self.odom.clear()
        self.last_pose = None
        self.get_logger().info('tracks cleared (scenario reset)')

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.odom[_ms(m.header.stamp)] = (m.pose.pose.position.x, m.pose.pose.position.y,
                                          yaw_from_quaternion(q.x, q.y, q.z, q.w))
        if len(self.odom) > 600:
            for k in sorted(self.odom)[:-400]:
                del self.odom[k]

    def on_bev(self, m):
        self.bev = m

    def pose_at(self, stamp):
        t = _ms(stamp)
        for dt in (0, -STEP_MS, STEP_MS, -2 * STEP_MS, 2 * STEP_MS):
            if t + dt in self.odom:
                return self.odom[t + dt]
        return None

    def in_view_fn(self, pose):
        """Whether an odom-frame point is inside the detector's view from `pose`."""
        return lambda x, y, th=None: in_view(pose, x, y, self.view_margin, self.view_range, th, self.yaw_tol)

    def on_dets(self, msg):
        pose = self.pose_at(msg.header.stamp)
        if pose is None:
            self.get_logger().warn('no odometry for detection stamp; skipped', throttle_duration_sec=2.0)
            return
        if self.last_pose is not None:
            self.tracker.predict(math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1]))
        self.last_pose = pose

        dets = []
        for s in msg.slots:
            if not heading_valid(s.entrance.theta, self.yaw_tol):
                continue            # seen from an angle the detector was not trained for
            x, y, th = compose(pose, (s.entrance.x, s.entrance.y, s.entrance.theta))
            rng = math.hypot(s.entrance.x - CAR_CENTRE_X, s.entrance.y)
            dets.append(Detection(x, y, th, s.width, s.vacancy, rng))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        before = {t.id: t for t in self.tracker.tracks if t.confirmed}
        self.tracker.update(dets, stamp, self.in_view_fn(pose))
        after = {t.id for t in self.tracker.tracks}
        for tid, t in before.items():
            if tid not in after:
                gx, gy, _ = compose(inverse(pose), tuple(t.x))
                self.get_logger().info(f'track {tid} deleted after {t.hits} hits: {t.misses} misses in view, '
                                       f'last position in car frame ({gx:.1f}, {gy:.1f}) m')
        self.publish(msg.header.stamp, pose)

    def publish(self, stamp, pose):
        out = ParkingSlotArray()
        out.header.stamp = stamp
        out.header.frame_id = 'odom'
        slots_car = []
        inv = inverse(pose)
        for t in self.tracker.confirmed():
            s = self.to_slot(t)
            m = ParkingSlot()
            m.id = t.id
            m.corners = [Point(x=float(x), y=float(y), z=0.0) for x, y in s.corners()]
            m.entrance.x, m.entrance.y, m.entrance.theta = (float(v) for v in t.x)
            m.width = float(t.width)
            m.depth = sc.NOMINAL_DEPTH
            m.vacancy = float(t.vacancy)
            m.occupied = bool(t.vacancy < 0.5)
            m.confidence = float(min(t.n_vac_close / 10.0, 1.0))
            m.covariance = [float(v) for v in t.P.ravel()]
            out.slots.append(m)
            gx, gy, gth = compose(inv, tuple(t.x))
            slots_car.append(self.to_slot(t, (gx, gy, gth)))
        self.pub.publish(out)

        if self.bev is not None and self.img_pub.get_subscription_count() > 0:
            img = draw_slots(self.bridge.imgmsg_to_cv2(self.bev, 'bgr8'), slots_car)
            o = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            o.header = self.bev.header
            self.img_pub.publish(o)

    @staticmethod
    def to_slot(t, pose=None):
        x, y, th = pose if pose is not None else t.x
        c, s = math.cos(th), math.sin(th)
        lx, ly = -s * t.width / 2, c * t.width / 2
        left = sc.MarkingPoint(x + lx, y + ly, c, s)
        right = sc.MarkingPoint(x - lx, y - ly, c, s)
        return sc.Slot(left, right, th, vacancy=t.vacancy)


def main():
    rclpy.init()
    node = SlotTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
