"""webots_ros2_driver Python plugin for the gt_supervisor robot (Webots Supervisor).

Ground truth for labels and evaluation only -- the autonomy stack must never use it.

Publishes
  /ground_truth/pose   nav_msgs/Odometry           ego rear-axle pose + velocity, frame map,
                                                   every simulation step, Webots time stamps
  /ground_truth/slots  autopark_msgs/ParkingSlotArray  all 16 slots with occupancy, 2 Hz
  /scenario/reset      std_msgs/Int64              seed, after every reset
Services
  /ground_truth/reset  autopark_msgs/ResetScenario  re-randomise lot + ego pose (by seed)

<plugin> property: seed (scenario applied at start-up, default 0).
"""
import math

import rclpy
from autopark_msgs.msg import ParkingSlot, ParkingSlotArray
from autopark_msgs.srv import ResetScenario
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Int64

from autopark_sim.lot import SLOTS_PER_ROW, slots
from autopark_sim.scenario import EGO_Z, make_scenario

SLOTS_PERIOD = 0.5  # s


def _stamp(msg, t):
    msg.header.stamp.sec = int(t)
    msg.header.stamp.nanosec = int(round((t - int(t)) * 1e9))


def _quat_from_matrix(r):
    """Row-major 3x3 rotation matrix (list of 9) -> (x, y, z, w)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = r
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)


class GroundTruthSupervisor:
    def init(self, webots_node, properties):
        self.__sup = webots_node.robot
        self.__ego = self.__sup.getFromDef('EGO')
        self.__children = self.__sup.getRoot().getField('children')
        self.__slots = slots()
        self.__occupied = set()
        self.__last_slots_pub = -1e9

        if not rclpy.ok():
            rclpy.init(args=None)
        self.__node = rclpy.create_node('ground_truth')
        self.__pose_pub = self.__node.create_publisher(Odometry, '/ground_truth/pose', 10)
        self.__slots_pub = self.__node.create_publisher(ParkingSlotArray, '/ground_truth/slots', 10)
        self.__reset_pub = self.__node.create_publisher(Int64, '/scenario/reset', 10)
        self.__node.create_service(ResetScenario, '/ground_truth/reset', self.__on_reset)

        self.__apply(int(properties.get('seed', 0)))

    def __apply(self, seed):
        sc = make_scenario(seed)
        for i in range(2 * SLOTS_PER_ROW):
            node = self.__sup.getFromDef(f'PARKED_{i}')
            if node is not None:
                node.remove()
        for car in sc.parked:
            self.__children.importMFNodeFromString(-1, car.webots_string())
        x, y, yaw = sc.ego
        self.__ego.getField('translation').setSFVec3f([x, y, EGO_Z])
        self.__ego.getField('rotation').setSFRotation([0, 0, 1, yaw])
        self.__ego.resetPhysics()
        self.__occupied = {c.slot_id for c in sc.parked}
        self.__reset_pub.publish(Int64(data=seed))
        self.__last_slots_pub = -1e9  # publish slots right away
        self.__node.get_logger().info(f'scenario seed {seed}: empty slots {sc.empty_slot_ids}')
        return sc

    def __on_reset(self, request, response):
        sc = self.__apply(request.seed)
        response.success = True
        response.message = f'seed {request.seed}'
        response.empty_slot_ids = list(sc.empty_slot_ids)
        return response

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        t = self.__sup.getTime()

        pos = self.__ego.getPosition()
        rot = self.__ego.getOrientation()
        vel = self.__ego.getVelocity()  # world frame: vx vy vz wx wy wz
        odom = Odometry()
        _stamp(odom, t)
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        p = odom.pose.pose
        p.position.x, p.position.y, p.position.z = pos
        q = _quat_from_matrix(rot)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        # Twist in the body frame (R^T v)
        r = rot
        lin = [r[0] * vel[0] + r[3] * vel[1] + r[6] * vel[2],
               r[1] * vel[0] + r[4] * vel[1] + r[7] * vel[2],
               r[2] * vel[0] + r[5] * vel[1] + r[8] * vel[2]]
        ang = [r[0] * vel[3] + r[3] * vel[4] + r[6] * vel[5],
               r[1] * vel[3] + r[4] * vel[4] + r[7] * vel[5],
               r[2] * vel[3] + r[5] * vel[4] + r[8] * vel[5]]
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = lin
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = ang
        self.__pose_pub.publish(odom)

        if t - self.__last_slots_pub >= SLOTS_PERIOD:
            self.__last_slots_pub = t
            self.__slots_pub.publish(self.__slots_msg(t))

    def __slots_msg(self, t):
        msg = ParkingSlotArray()
        _stamp(msg, t)
        msg.header.frame_id = 'map'
        for s in self.__slots:
            m = ParkingSlot()
            m.id = s.id
            m.corners = [Point(x=float(x), y=float(y), z=0.0) for x, y in s.corners()]
            m.entrance.x, m.entrance.y, m.entrance.theta = s.entrance_x, s.entrance_y, s.heading
            m.width, m.depth = s.width, s.depth
            m.occupied = s.id in self.__occupied
            m.vacancy = 0.0 if m.occupied else 1.0
            m.confidence = 1.0
            msg.slots.append(m)
        return msg
