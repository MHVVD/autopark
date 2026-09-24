"""Compute per-camera "own car body" masks from data.

The car body is fixed relative to each camera, while the ground, lines and other cars move past
when driving. So a pixel whose intensity hardly changes while the car moves sees the body.
Run in the calibration world (checkerboard floor, so every ground pixel changes strongly while
driving; on uniform asphalt mid-distance ground barely changes and would be mistaken for body)
together with `drive_test`. Only pixels whose ray hits the ground within MAX_RANGE are kept
(sky is irrelevant for the BEV). Accumulates only while |speed| > 0.5 m/s, then writes
<out_dir>/body_<cam>.png (255 = body) and body_<cam>_overlay.png for inspection.
"""
import os

import cv2
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from autopark.bev import CameraModel
from autopark_sim.descriptions import image_topic
from autopark_sim.rig import CAMERAS

STD_THRESHOLD = 4.0     # grey levels; body pixels only change with rendering noise
MIN_AREA = 300          # px, smaller static blobs are ignored
MARGIN = 4              # px dilation safety margin
MAX_RANGE = 25.0        # m, only pixels seeing the ground closer than this matter


def ground_pixels(spec, max_range=MAX_RANGE):
    """Bool image: True where the pixel's ray hits the ground within max_range."""
    cam = CameraModel(spec)
    u, v = np.meshgrid(np.arange(spec.width, dtype=float), np.arange(spec.height, dtype=float))
    gx, gy = cam.ray_to_ground(u, v)
    return np.hypot(gx - cam.t[0], gy - cam.t[1]) < max_range  # NaN compares False


def body_mask(std, threshold=STD_THRESHOLD, min_area=MIN_AREA, margin=MARGIN):
    """Temporal std image -> cleaned boolean body mask."""
    m = (std < threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    keep = cv2.dilate(keep, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return keep.astype(bool)


class Welford:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, np.float64)
        self.m2 = np.zeros(shape, np.float64)

    def add(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def std(self):
        return np.sqrt(self.m2 / max(self.n - 1, 1))


class CalibrateMasks(Node):
    def __init__(self):
        super().__init__('calibrate_masks')
        self.declare_parameter('frames', 150)
        self.declare_parameter('out_dir', os.path.expanduser('~/autopark_results/masks'))
        self.frames = self.get_parameter('frames').value
        self.out_dir = self.get_parameter('out_dir').value
        self.bridge = CvBridge()
        self.moving = False
        self.stats = {c.name: Welford((c.height, c.width)) for c in CAMERAS}
        self.last = {}
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        for c in CAMERAS:
            self.create_subscription(Image, image_topic(c.name),
                                     lambda m, n=c.name: self.on_image(n, m), qos)
        self.create_subscription(AckermannDriveStamped, '/vehicle/state',
                                 lambda m: setattr(self, 'moving', abs(m.drive.speed) > 0.5), 10)

    def on_image(self, name, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last[name] = img
        if self.moving and self.stats[name].n < self.frames:
            self.stats[name].add(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))

    def done(self):
        return all(s.n >= self.frames for s in self.stats.values())

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for name, st in self.stats.items():
            std = st.std()
            spec = next(c for c in CAMERAS if c.name == name)
            mask = body_mask(std) & ground_pixels(spec)
            cv2.imwrite(os.path.join(self.out_dir, f'body_{name}.png'), mask.astype(np.uint8) * 255)
            over = self.last[name].copy()
            over[mask] = (0.5 * over[mask] + (0, 0, 127)).astype(np.uint8)
            cv2.imwrite(os.path.join(self.out_dir, f'body_{name}_overlay.png'), over)
            print(f'{name}: {st.n} frames, body = {mask.mean() * 100:.1f} % of the image, '
                  f'std percentiles 5/50/95 = {np.percentile(std, [5, 50, 95]).round(1)}', flush=True)
        print(f'wrote masks to {self.out_dir}', flush=True)


def main():
    rclpy.init()
    node = CalibrateMasks()
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    if all(s.n > 10 for s in node.stats.values()):
        node.save()
    else:
        print('not enough moving frames:', {k: s.n for k, s in node.stats.items()}, flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
