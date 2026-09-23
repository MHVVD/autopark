"""Odometry node: /vehicle/state + /imu -> /odom and TF odom -> base_link.

Parameters
  heading_source  'imu' (gyro z, default) | 'steering' (bicycle model)
  wheel_base      m, used for heading_source=steering
Resets to the origin on /scenario/reset (published by the ground-truth supervisor) and on
the /odometry/reset service.
"""
import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int64
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from autopark.odometry import AckermannOdometry, yaw_rate_from_steering


def _t(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry')
        self.declare_parameter('heading_source', 'imu')
        self.declare_parameter('wheel_base', 2.8)
        self.heading_source = self.get_parameter('heading_source').value
        self.wheel_base = self.get_parameter('wheel_base').value
        if self.heading_source not in ('imu', 'steering'):
            raise ValueError(f"heading_source must be 'imu' or 'steering', got {self.heading_source}")

        self.odom = AckermannOdometry()
        self.last_t = None
        self.gyro_z = None

        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(AckermannDriveStamped, '/vehicle/state', self.on_state, 50)
        self.create_subscription(Imu, '/imu', self.on_imu, 50)
        self.create_subscription(Int64, '/scenario/reset', lambda _m: self.reset(), 10)
        self.create_service(Trigger, '/odometry/reset', self.on_reset_srv)

    def reset(self):
        self.odom.reset()
        self.last_t = None
        self.get_logger().info('odometry reset')

    def on_reset_srv(self, _req, res):
        self.reset()
        res.success = True
        return res

    def on_imu(self, msg):
        self.gyro_z = msg.angular_velocity.z

    def on_state(self, msg):
        t = _t(msg.header.stamp)
        v = msg.drive.speed
        if self.heading_source == 'imu':
            if self.gyro_z is None:
                return
            yaw_rate = self.gyro_z
        else:
            yaw_rate = yaw_rate_from_steering(v, msg.drive.steering_angle, self.wheel_base)

        if self.last_t is not None:
            dt = t - self.last_t
            if dt < 0:          # simulation reset without /scenario/reset
                self.reset()
            elif dt < 1.0:      # ignore gaps (e.g. paused simulation)
                self.odom.update(v, yaw_rate, dt)
        self.last_t = t
        self.publish(msg.header.stamp, v, yaw_rate)

    def publish(self, stamp, v, yaw_rate):
        o = self.odom
        qz, qw = math.sin(o.yaw / 2), math.cos(o.yaw / 2)
        m = Odometry()
        m.header.stamp = stamp
        m.header.frame_id = 'odom'
        m.child_frame_id = 'base_link'
        m.pose.pose.position.x, m.pose.pose.position.y = o.x, o.y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
        m.twist.twist.linear.x = v
        m.twist.twist.angular.z = yaw_rate
        self.pub.publish(m)

        tf = TransformStamped()
        tf.header = m.header
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x, tf.transform.translation.y = o.x, o.y
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf.sendTransform(tf)


def main():
    rclpy.init()
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
