"""Collect a slot-detection dataset: BEV images + ground-truth labels.

For each seed: reset the scenario, drive down the aisle on a randomised weaving path
(pure pursuit on the ground-truth pose -- this is a data tool, not part of the stack) and
save every `sample_period` s of simulated time
  <out_dir>/images/<seed>_<n>.png   the BEV (450x450)
  <out_dir>/labels.jsonl            one line per image: seed, t, pose [x, y, yaw] (map),
                                    speed, slots [{id, corners, heading, occupied}] (map)
Run with the stack up:  ros2 launch autopark bringup.launch.py gui:=false mode:=fast
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autopark_msgs.msg import ParkingSlotArray
from autopark_msgs.srv import ResetScenario
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from autopark.odometry import yaw_from_quaternion, wrap

WHEEL_BASE = 2.8
LOOKAHEAD = 3.5      # m
X_END = 13.0         # m, stop recording past the end of the rows
SETTLE = 1.0         # s simulated time after a reset before driving
TIMEOUT = 60.0       # s simulated time per seed


def reference_path(seed):
    """Weaving lateral offset y(x) in the aisle, parameters drawn from the seed."""
    rng = np.random.default_rng(seed + 7919)
    amp = rng.uniform(0.0, 1.3)
    wavelength = rng.uniform(8.0, 20.0)
    phase = rng.uniform(0, 2 * math.pi)
    offset = rng.uniform(-0.9, 0.9)
    speed = rng.uniform(1.0, 2.5)

    def y(x):
        return offset + amp * math.sin(2 * math.pi * x / wavelength + phase)
    return y, speed


def pure_pursuit(pose, y_ref, lookahead=LOOKAHEAD, wheel_base=WHEEL_BASE):
    """Steering angle (rad, +left) towards the point `lookahead` ahead on y_ref."""
    x, y, yaw = pose
    tx = x + lookahead
    dx, dy = tx - x, y_ref(tx) - y
    ly = -math.sin(yaw) * dx + math.cos(yaw) * dy     # lateral offset in the car frame
    ld2 = dx * dx + dy * dy
    return math.atan(2 * ly / ld2 * wheel_base)


class CollectSlots(Node):
    def __init__(self):
        super().__init__('collect_slots')
        self.declare_parameter('seed_start', 10000)
        self.declare_parameter('seed_count', 5)
        self.declare_parameter('sample_period', 0.5)
        self.declare_parameter('out_dir', os.path.expanduser('~/autopark_data/slots'))
        p = self.get_parameter
        self.seeds = list(range(p('seed_start').value, p('seed_start').value + p('seed_count').value))
        self.sample_period = p('sample_period').value
        self.out_dir = p('out_dir').value
        os.makedirs(os.path.join(self.out_dir, 'images'), exist_ok=True)
        self.labels = open(os.path.join(self.out_dir, 'labels.jsonl'), 'a')

        self.bridge = CvBridge()
        self.cmd = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
        self.reset_cli = self.create_client(ResetScenario, '/ground_truth/reset')
        self.create_subscription(Odometry, '/ground_truth/pose', self.on_pose, 100)
        self.create_subscription(ParkingSlotArray, '/ground_truth/slots', self.on_slots, 10)
        self.create_subscription(Image, '/bev/image', self.on_bev,
                                 QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE))

        self.poses = {}          # ms -> (x, y, yaw, speed)
        self.slots = None
        self.slots_t = -1.0
        self.t = 0.0
        self.state = 'idle'
        self.seed_i = -1
        self.saved = 0
        self.saved_seed = 0

    # ---------------------------------------------------------------- callbacks
    @staticmethod
    def _sec(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_pose(self, m):
        t = self._sec(m.header.stamp)
        q = m.pose.pose.orientation
        pose = (m.pose.pose.position.x, m.pose.pose.position.y,
                yaw_from_quaternion(q.x, q.y, q.z, q.w), m.twist.twist.linear.x)
        self.poses[int(round(t * 1000))] = pose
        if len(self.poses) > 500:
            for k in sorted(self.poses)[:-300]:
                del self.poses[k]
        self.t = t
        if self.state == 'drive':
            self.drive(pose)

    def on_slots(self, m):
        self.slots = [dict(id=s.id, corners=[[c.x, c.y] for c in s.corners],
                           heading=s.entrance.theta, occupied=s.occupied) for s in m.slots]
        self.slots_t = self._sec(m.header.stamp)

    def on_bev(self, m):
        if self.state != 'drive' or self.slots_t < self.reset_t:
            return
        t = self._sec(m.header.stamp)
        if t - self.last_sample < self.sample_period:
            return
        pose = self.poses.get(int(round(t * 1000)))
        if pose is None or pose[0] > X_END:
            return
        self.last_sample = t
        seed = self.seeds[self.seed_i]
        name = f'{seed}_{self.saved_seed:03d}.png'
        cv2.imwrite(os.path.join(self.out_dir, 'images', name), self.bridge.imgmsg_to_cv2(m, 'bgr8'))
        self.labels.write(json.dumps(dict(image=name, seed=seed, t=round(t, 3),
                                          pose=[round(v, 5) for v in pose[:3]],
                                          speed=round(pose[3], 3), slots=self.slots)) + '\n')
        self.saved += 1
        self.saved_seed += 1

    # ---------------------------------------------------------------- driving
    def drive(self, pose):
        x, y, yaw, _ = pose
        msg = AckermannDriveStamped()
        if self.t - self.reset_t < SETTLE:
            self.cmd.publish(msg)
            return
        if x > X_END or self.t - self.reset_t > TIMEOUT:
            self.cmd.publish(msg)
            self.state = 'next'
            return
        steer = pure_pursuit((x, y, yaw), self.y_ref)
        msg.drive.speed = self.speed
        msg.drive.steering_angle = max(-0.55, min(0.55, steer))
        msg.drive.steering_angle_velocity = 1.5
        self.cmd.publish(msg)

    def next_seed(self):
        self.labels.flush()
        self.seed_i += 1
        if self.seed_i >= len(self.seeds):
            self.state = 'done'
            return
        seed = self.seeds[self.seed_i]
        self.y_ref, self.speed = reference_path(seed)
        req = ResetScenario.Request()
        req.seed = seed
        self.state = 'resetting'
        self.future = self.reset_cli.call_async(req)

    def spin_step(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        if self.state in ('idle', 'next'):
            if self.state == 'next':
                self.get_logger().info(f'seed {self.seeds[self.seed_i]}: {self.saved_seed} images '
                                       f'(total {self.saved})')
            if not self.reset_cli.service_is_ready():
                return
            self.next_seed()
        elif self.state == 'resetting' and self.future.done():
            self.reset_t = self.t
            self.last_sample = -1e9
            self.saved_seed = 0
            self.state = 'drive'


def main():
    rclpy.init()
    node = CollectSlots()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and node.state != 'done':
            node.spin_step()
    except KeyboardInterrupt:
        pass
    node.labels.close()
    node.get_logger().info(f'done: {node.saved} images in {time.monotonic() - t0:.0f} s -> {node.out_dir}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
