"""Controller node: follows /parking/path_exec with the Stanley controller (autopark.stanley).

Subscribes /parking/path_exec (ParkingPath, odom frame), /odom and /vehicle/state; runs at the
odometry rate (50 Hz) and publishes /cmd_ackermann and /parking/control_status while a path
is active. It publishes nothing while idle, so other nodes (the parking manager while
searching) can drive. A path with no poses cancels. A path with the same plan_id and a higher
revision is a correction of the current one: the controller keeps its progress.
After 'done' or 'aborted' it holds the car (zero speed) for `hold` s, then goes quiet.
"""
import math
from types import SimpleNamespace

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autopark_msgs.msg import ControlStatus, ParkingPath
from nav_msgs.msg import Odometry
from rclpy.node import Node

from autopark.odometry import yaw_from_quaternion
from autopark.stanley import Stanley, StanleyConfig


def _sec(st):
    return st.sec + st.nanosec * 1e-9


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')
        d = StanleyConfig()
        for name in ('k', 'k_he_rev', 'k_e_rev', 'v_forward', 'v_reverse', 'v_creep', 'decel'):
            self.declare_parameter(name, getattr(d, name))
        self.declare_parameter('hold', 1.0)
        cfg = StanleyConfig(**{n: self.get_parameter(n).value
                               for n in ('k', 'k_he_rev', 'k_e_rev', 'v_forward', 'v_reverse', 'v_creep', 'decel')})
        self.hold = self.get_parameter('hold').value
        self.ctl = Stanley(cfg)
        self.plan_id, self.revision = None, 0
        self.speed, self.steer = 0.0, 0.0
        self.t_end = None
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
        self.status_pub = self.create_publisher(ControlStatus, '/parking/control_status', 10)
        self.create_subscription(ParkingPath, '/parking/path_exec', self.on_path, 10)
        self.create_subscription(AckermannDriveStamped, '/vehicle/state', self.on_state, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

    def on_path(self, m):
        if not m.poses:
            self.ctl.cancel()
            self.plan_id, self.t_end = None, None
            self.get_logger().info('path cancelled')
            return
        path = SimpleNamespace(x=[p.x for p in m.poses], y=[p.y for p in m.poses], yaw=[p.theta for p in m.poses],
                               direction=list(m.direction), curvature=list(m.curvature))
        correction = m.plan_id == self.plan_id and m.revision > self.revision
        self.ctl.set_path(path, keep_progress=correction)
        if not correction:
            self.t_end = None
            self.get_logger().info(f'plan {m.plan_id}: {len(m.poses)} samples, {len(self.ctl.segs)} segments, '
                                   f'{m.length:.1f} m')
        self.plan_id, self.revision = m.plan_id, m.revision

    def on_state(self, m):
        self.speed, self.steer = m.drive.speed, m.drive.steering_angle

    def on_odom(self, m):
        if self.ctl.state == 'idle':
            return
        q = m.pose.pose.orientation
        pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        t = _sec(m.header.stamp)
        before = self.ctl.state
        v, delta = self.ctl.update(pose, self.speed, self.steer, t)
        if self.ctl.state != before:
            self.get_logger().info(f'plan {self.plan_id}: {before} -> {self.ctl.state}'
                                   + (f' (segment {self.ctl.seg_i + 1}/{len(self.ctl.segs)})'
                                      if self.ctl.state in ('align', 'track') else ''))
        if self.ctl.state in ('done', 'aborted'):
            if self.t_end is None:
                self.t_end = t
                self.get_logger().info(f'plan {self.plan_id} {self.ctl.state}: max lateral error '
                                       f'{100 * self.ctl.max_lateral:.1f} cm, max heading error '
                                       f'{math.degrees(self.ctl.max_heading):.1f} deg')
            if t - self.t_end > self.hold:
                self.publish_status(m.header.stamp)
                self.ctl.state = 'idle' if self.ctl.state == 'done' else self.ctl.state
                if self.ctl.state == 'idle':
                    return
        cmd = AckermannDriveStamped()
        cmd.header.stamp = m.header.stamp
        cmd.drive.speed = float(v)
        cmd.drive.steering_angle = float(delta)
        self.cmd_pub.publish(cmd)
        self.publish_status(m.header.stamp)

    def publish_status(self, stamp):
        s = ControlStatus()
        s.header.stamp = stamp
        s.plan_id = self.plan_id or 0
        s.state = self.ctl.state
        last = getattr(self.ctl, 'last', None)
        if last:
            s.segment, s.segments, s.direction = last['segment'], last['segments'], last['direction']
            s.lateral_error, s.heading_error, s.remaining = last['lateral'], last['heading'], last['remaining']
        s.max_lateral_error = getattr(self.ctl, 'max_lateral', 0.0)
        self.status_pub.publish(s)


def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
