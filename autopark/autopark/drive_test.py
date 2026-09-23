"""Scripted test drive on /cmd_ackermann (sim time). Exercises forward, reverse and both steering
directions so vehicle-interface conventions and odometry can be checked against ground truth.
Exits when the script is finished.
"""
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node

# (duration s, speed m/s, steering rad (+left))
SCRIPT = [
    (10.0, 1.5, 0.0),    # drive down the aisle past the row
    (4.0, 1.0, 0.3),     # forward, steer left  -> yaw increases
    (2.0, 0.0, 0.3),     # stop
    (6.0, -1.0, 0.3),    # reverse, steer left  -> yaw decreases
    (3.0, -0.8, 0.0),    # reverse straight
    (3.0, 0.0, 0.0),     # stop
]


def phase_at(t):
    """Index of the script phase active at time t since start, or None when finished."""
    for i, (d, _, _) in enumerate(SCRIPT):
        if t < d:
            return i
        t -= d
    return None


class DriveTest(Node):
    def __init__(self):
        super().__init__('drive_test', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.pub = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
        self.t0 = None
        self.phase = -1
        self.done = False
        self.create_timer(0.05, self.tick)

    def tick(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:       # /clock not received yet
            return
        if self.t0 is None:
            self.t0 = now
        i = phase_at((now - self.t0).nanoseconds * 1e-9)
        if i is None:
            self.pub.publish(AckermannDriveStamped())
            self.get_logger().info('script finished')
            self.done = True
            return
        if i != self.phase:
            self.phase = i
            self.get_logger().info(f'phase {i}: {SCRIPT[i]}')
        _, v, steer = SCRIPT[i]
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.drive.speed = v
        msg.drive.steering_angle = steer
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DriveTest()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
