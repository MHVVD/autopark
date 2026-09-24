"""Planner node: /parking/plan service -> Hybrid A* path into a tracked slot.

Inputs: /slots/tracked (odom frame) and /odom. On a PlanParking request the node takes the
current odometry pose as the start, the tracked slot's goal pose (centred, facing out) as the
goal, and the tracked slots as obstacles (see autopark.parking_goal: every slot that is not
selectable-vacant is a keep-out box). Only explored space is drivable: the node records the
odometry pose every metre while the car heads along the aisle and the explored area is the
union of parking_goal.KNOWN_BOX at those poses. The result is published on
  /parking/path       autopark_msgs/ParkingPath (transient local: late subscribers get it)
  /parking/path_viz   nav_msgs/Path (RViz)
  /parking/path_image the latest BEV with the tracked slots, the path and the goal footprint,
                      redrawn in the current car frame as the car moves.

slot_id = -1 picks the selectable vacant slot whose goal is nearest the car.
"""
import math
import time

import numpy as np
import rclpy
from autopark_msgs.msg import ParkingPath, ParkingSlotArray
from autopark_msgs.srv import PlanParking
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int64

from autopark import parking_goal as pg
from autopark import slot_codec as sc
from autopark.hybrid_astar import CarGeometry, HybridAStar, ObstacleMap, PlannerConfig
from autopark.odometry import compose, inverse, yaw_from_quaternion
from autopark.slot_viz import draw_path, draw_slots


def specs_from_tracks(msg):
    return [pg.SlotSpec(s.id, s.entrance.x, s.entrance.y, s.entrance.theta, s.width, s.depth,
                        vacant=pg.selectable(s.vacancy, s.confidence)) for s in msg.slots]


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner')
        self.declare_parameter('margin', PlannerConfig.margin)
        self.declare_parameter('max_steer', CarGeometry.max_steer)
        self.declare_parameter('timeout', PlannerConfig.timeout)
        p = self.get_parameter
        self.car = CarGeometry(max_steer=p('max_steer').value)
        self.cfg = PlannerConfig(margin=p('margin').value, timeout=p('timeout').value)
        self.planner = HybridAStar(self.car, self.cfg)

        self.tracks = None
        self.pose = None
        self.explored = []         # odom poses whose KNOWN_BOX is explored
        self.path = None           # (ParkingPath, odom xs, ys, dirs, goal footprint)
        self.bridge = CvBridge()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(ParkingPath, '/parking/path', latched)
        self.viz_pub = self.create_publisher(PathMsg, '/parking/path_viz', latched)
        self.img_pub = self.create_publisher(Image, '/parking/path_image', 2)
        self.create_subscription(ParkingSlotArray, '/slots/tracked', self.on_tracks, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 50)
        self.create_subscription(Image, '/bev/image', self.on_bev, 2)
        self.create_subscription(Int64, '/scenario/reset', self.on_reset, 10)
        # Planning runs in the service callback of a single-threaded executor, so other
        # callbacks wait until it is done (the car stands still while planning, and only the
        # latest odometry / tracks matter). A second thread would only contend for the GIL.
        self.create_service(PlanParking, '/parking/plan', self.on_plan)
        self.get_logger().info(f'planner ready: min turning radius {self.car.min_radius:.2f} m, '
                               f'margin {self.cfg.margin} m')

    def on_reset(self, _msg):
        self.tracks, self.path, self.pose = None, None, None
        self.explored = []

    def on_tracks(self, m):
        self.tracks = m

    def on_odom(self, m):
        q = m.pose.pose.orientation
        pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.pose = pose
        if self.tracks is None or not self.tracks.slots:
            return
        last = self.explored[-1] if self.explored else None
        if last is None or math.hypot(pose[0] - last[0], pose[1] - last[1]) >= 1.0:
            if pg.aisle_aligned(pose[2], [s.entrance.theta for s in self.tracks.slots]):
                self.explored.append(pose)
                del self.explored[:-300]

    def on_plan(self, req, res):
        tracks, start, explored = self.tracks, self.pose, list(self.explored)
        if tracks is None or start is None:
            res.success, res.message = False, 'no tracked slots or odometry yet'
            return res
        if not explored:
            res.success, res.message = False, 'nothing explored yet (drive along the aisle first)'
            return res
        specs = specs_from_tracks(tracks)
        target = self.choose(specs, req.slot_id, start)
        if target is None:
            res.success = False
            res.message = (f'slot {req.slot_id} is not a tracked selectable-vacant slot' if req.slot_id >= 0
                           else 'no selectable vacant slot')
            return res
        goal = pg.goal_pose(target, self.car)
        t0 = time.monotonic()
        omap = ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, target.id),
                           known=pg.known_region(explored))
        path = self.planner.plan(start, goal, omap)
        dt = time.monotonic() - t0
        info = self.planner.last_info
        if path is None:
            res.success = False
            res.message = f'slot {target.id}: no path ({info["reason"]}, {info["expansions"]} expansions, {dt:.2f} s)'
            self.get_logger().warn(res.message)
            return res
        msg = ParkingPath()
        msg.header.stamp = tracks.header.stamp
        msg.header.frame_id = 'odom'
        msg.slot_id = target.id
        msg.goal = Pose2D(x=goal[0], y=goal[1], theta=goal[2])
        msg.poses = [Pose2D(x=float(x), y=float(y), theta=float(t)) for x, y, t in zip(path.x, path.y, path.yaw)]
        msg.direction = [int(d) for d in path.direction]
        msg.curvature = [float(k) for k in path.curvature]
        msg.length = path.length
        msg.gear_changes = path.n_gear_changes
        msg.planning_time = dt
        self.path_pub.publish(msg)
        self.viz_pub.publish(self.to_nav_path(msg))
        self.path = (msg, path.x.copy(), path.y.copy(), path.direction.copy(), self.car.corners(goal))
        res.success, res.path = True, msg
        res.message = (f'slot {target.id}: {path.length:.1f} m, {path.n_gear_changes} gear changes, '
                       f'{dt:.2f} s ({info["expansions"]} expansions)')
        self.get_logger().info(res.message)
        return res

    @staticmethod
    def choose(specs, slot_id, start):
        vac = [s for s in specs if s.vacant]
        if slot_id >= 0:
            return next((s for s in vac if s.id == slot_id), None)
        if not vac:
            return None
        return min(vac, key=lambda s: math.hypot(*(np.subtract(pg.goal_pose(s)[:2], start[:2]))))

    def to_nav_path(self, msg):
        out = PathMsg()
        out.header = msg.header
        for p in msg.poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = p.x, p.y
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(p.theta / 2), math.cos(p.theta / 2)
            out.poses.append(ps)
        return out

    def on_bev(self, m):
        if self.img_pub.get_subscription_count() == 0 or self.pose is None:
            return
        img = self.bridge.imgmsg_to_cv2(m, 'bgr8')
        inv = inverse(self.pose)
        if self.tracks is not None:
            slots = []
            for s in specs_from_tracks(self.tracks):
                x, y, th = compose(inv, (s.x, s.y, s.theta))
                c, sn = math.cos(th), math.sin(th)
                lx, ly = -sn * s.width / 2, c * s.width / 2
                slots.append(sc.Slot(sc.MarkingPoint(x + lx, y + ly, c, sn), sc.MarkingPoint(x - lx, y - ly, c, sn),
                                     th, vacancy=1.0 if s.vacant else 0.0))
            draw_slots(img, slots, label=False)
        if self.path is not None:
            _, xs, ys, dirs, fp = self.path
            c, sn = math.cos(inv[2]), math.sin(inv[2])
            gx = inv[0] + c * xs - sn * ys
            gy = inv[1] + sn * xs + c * ys
            gfp = [compose(inv, (x, y, 0.0))[:2] for x, y in fp]
            draw_path(img, gx, gy, dirs, [gfp])
        o = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        o.header = m.header
        self.img_pub.publish(o)


def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
