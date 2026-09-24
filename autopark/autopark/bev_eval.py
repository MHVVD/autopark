"""Evaluate BEV geometry against ground truth: do the painted lines land where they should?

For sampled /bev/image frames, the painted-line centrelines of the lot (autopark_sim.lot) are
moved into the car frame with /ground_truth/pose and compared with the white pixels:
  * BEV: distance from each centreline sample to the nearest white BEV pixel (cm), by range
  * raw cameras: same in each fisheye image (px), which isolates per-camera calibration
Timing: camera stamps come from the ROS clock and may be off by a simulation step, so the
BEV metric is also computed with the ground-truth pose shifted by -2..+2 steps.
Use an empty lot (seed -1) so no lines are hidden by parked cars. Writes overlays to out_dir.
"""
import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from autopark.bev import BevGrid, CameraModel, load_body_masks, segment_hits_box
from autopark.odometry import yaw_from_quaternion
from autopark_sim.descriptions import image_topic
from autopark_sim.lot import painted_lines
from autopark_sim.rig import CAMERAS

STEP_MS = 20
WHITE = 140          # grey level threshold for painted lines
BANDS = [(0, 3), (3, 6), (6, 9)]   # m from the car centre
CAR_CENTRE_X = 1.35


def line_samples(step=0.05):
    """Painted-line centreline points (N, 2) in the map frame."""
    pts = []
    for x, y, sx, sy in painted_lines():
        if sx < sy:
            ys = np.arange(y - sy / 2 + step / 2, y + sy / 2, step)
            pts.append(np.stack([np.full_like(ys, x), ys], 1))
        else:
            xs = np.arange(x - sx / 2 + step / 2, x + sx / 2, step)
            pts.append(np.stack([xs, np.full_like(xs, y)], 1))
    return np.concatenate(pts)


def map_to_ground(pts, pose):
    """Map-frame points (N, 2) -> ground frame of a car at pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = pts - [x, y]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def white_distance(gray, thr=WHITE):
    """Distance (px) from every pixel to the nearest pixel brighter than thr."""
    return cv2.distanceTransform((gray <= thr).astype(np.uint8), cv2.DIST_L2, 3)


class BevEval(Node):
    def __init__(self):
        super().__init__('bev_eval')
        self.declare_parameter('frames', 40)
        self.declare_parameter('every', 3)
        self.declare_parameter('out_dir', os.path.expanduser('~/autopark_results/bev_eval'))
        self.n_frames = self.get_parameter('frames').value
        self.every = self.get_parameter('every').value
        self.out_dir = self.get_parameter('out_dir').value
        os.makedirs(self.out_dir, exist_ok=True)

        self.grid = BevGrid()
        self.cams = [CameraModel(c) for c in CAMERAS]
        from ament_index_python.packages import get_package_share_directory
        self.masks = load_body_masks(os.path.join(
            get_package_share_directory('autopark_sim'), 'config', 'masks'))
        self.samples = line_samples()
        self.bridge = CvBridge()
        self.gt = {}
        self.raw = {}            # name -> (stamp_ms, image)
        self.results = []        # per frame: dict
        self.seen = 0
        qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, '/ground_truth/pose', self.on_gt, 200)
        self.create_subscription(Image, '/bev/image', self.on_bev, qos)
        for c in CAMERAS:
            self.create_subscription(Image, image_topic(c.name),
                                     lambda m, n=c.name: self.on_raw(n, m), qos)

    @staticmethod
    def _ms(stamp):
        return int(round((stamp.sec + stamp.nanosec * 1e-9) * 1000 / STEP_MS)) * STEP_MS

    def on_gt(self, m):
        q = m.pose.pose.orientation
        self.gt[self._ms(m.header.stamp)] = (m.pose.pose.position.x, m.pose.pose.position.y,
                                             yaw_from_quaternion(q.x, q.y, q.z, q.w),
                                             m.twist.twist.linear.x)

    def on_raw(self, name, m):
        self.raw[name] = (self._ms(m.header.stamp), self.bridge.imgmsg_to_cv2(m, 'bgr8'))

    def on_bev(self, m):
        self.seen += 1
        if self.seen % self.every or len(self.results) >= self.n_frames:
            return
        t = self._ms(m.header.stamp)
        if t not in self.gt:
            return
        bev = self.bridge.imgmsg_to_cv2(m, 'bgr8')
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        dist = white_distance(gray)
        valid = gray > 0

        res = {'t': t, 'speed': self.gt[t][3], 'bev': {}, 'raw': {}}
        for off in (-2, -1, 0, 1, 2):
            pose = self.gt.get(t + off * STEP_MS)
            if pose is None:
                continue
            g = map_to_ground(self.samples, pose[:3])
            col, row = self.grid.to_pixel(g[:, 0], g[:, 1])
            ci, ri = np.round(col).astype(int), np.round(row).astype(int)
            h, w = gray.shape
            inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
            ci, ri, gi = ci[inside], ri[inside], g[inside]
            ok = valid[ri, ci]
            rng = np.hypot(gi[:, 0] - CAR_CENTRE_X, gi[:, 1])
            res['bev'][off] = [(dist[ri[ok], ci[ok]][(rng[ok] > a) & (rng[ok] <= b)] * self.grid.res)
                               for a, b in BANDS]
            if off == 0 and len(self.results) < 3:
                over = bev.copy()
                for c_, r_ in zip(ci, ri):
                    over[r_, c_] = (0, 0, 255)
                cv2.imwrite(os.path.join(self.out_dir, f'bev_overlay_{len(self.results)}.png'), over)

        # raw cameras: latest frame of each, against ground truth at that frame's own stamp;
        # only line points the camera can actually see (not hidden by the own car)
        for cam in self.cams:
            name = cam.spec.name
            if name not in self.raw or self.raw[name][0] not in self.gt:
                continue
            ts, img = self.raw[name]
            pose = self.gt[ts]
            g3 = np.concatenate([map_to_ground(self.samples, pose[:3]),
                                 np.zeros((len(self.samples), 1))], 1)
            u, v, ok = cam.project(g3)
            ok &= ~segment_hits_box(cam.t, g3)
            ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
            keep = ~self.masks[name][vi, ui] & (np.hypot(g3[ok, 0] - cam.t[0], g3[ok, 1] - cam.t[1]) < 8)
            d = white_distance(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            res['raw'][name] = d[vi[keep], ui[keep]]
            if len(self.results) < 1:
                over = img.copy()
                for a, b in zip(ui[keep], vi[keep]):
                    cv2.circle(over, (int(a), int(b)), 1, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(self.out_dir, f'raw_overlay_{name}.png'), over)
        self.results.append(res)

    def report(self):
        n = len(self.results)
        print(f'evaluated {n} BEV frames (speed range '
              f'{min(r["speed"] for r in self.results):+.2f}..{max(r["speed"] for r in self.results):+.2f} m/s)')
        print('\nBEV: distance from GT line centre to nearest white pixel [cm], by range from car centre')
        for off in (-2, -1, 0, 1, 2):
            rows = [r['bev'][off] for r in self.results if off in r['bev']]
            if not rows:
                continue
            txt = []
            for i, (a, b) in enumerate(BANDS):
                d = np.concatenate([x[i] for x in rows]) * 100
                if len(d):
                    txt.append(f'{a}-{b} m: median {np.median(d):4.1f} p90 {np.percentile(d, 90):5.1f} '
                               f'<=6cm {np.mean(d <= 6) * 100:5.1f}% (n={len(d)})')
            print(f'  GT shifted {off * STEP_MS:+4d} ms | ' + ' | '.join(txt))
        print('\nraw cameras: distance from projected GT line centre to nearest white pixel [px] (< 8 m, visible only)')
        for c in CAMERAS:
            d = [r['raw'][c.name] for r in self.results if c.name in r['raw']]
            if d:
                d = np.concatenate(d)
                print(f'  {c.name:6s}: median {np.median(d):.2f} px, p90 {np.percentile(d, 90):.2f} px, '
                      f'<=1.5px {np.mean(d <= 1.5) * 100:.1f}% (n={len(d)}, {len([x for x in self.results if c.name in x["raw"]])} frames)')
        print(f'\noverlays in {self.out_dir}', flush=True)


def main():
    rclpy.init()
    node = BevEval()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and len(node.results) < node.n_frames:
            rclpy.spin_once(node, timeout_sec=0.1)
            if not node.results and time.monotonic() - t0 > 90:
                print('no BEV frames with ground truth in 90 s', flush=True)
                break
    except KeyboardInterrupt:
        pass
    if node.results:
        node.report()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
