"""Visualizer: one live image of the whole stack, /viz/image (for the demo video and debugging).

Left, the bird's-eye view (car frame, x forward = up) with
  yellow arrows   raw detections of this frame (entrance + heading)
  green / red     tracked slots: selectable vacant / not (fused estimate, odom -> car frame)
  cyan, thick     the target slot
  blue / orange   the planned path, forward / reverse (the part being executed is bold)
  magenta         the goal footprint;  white: the car
Right, a map of the manoeuvre in the odom frame (tracked slots, the car's trail, the path and
the car) and the state of the parking manager and the controller.

With `record_dir` set, every image is also saved as <record_dir>/viz/<t_ms>.jpg with an index
(time, manager state) and the supervisor is asked to save the Webots 3D view to
<record_dir>/3d (needs gui:=true); make_demo_video combines them.
Only uses what the stack itself publishes (no ground truth).
"""
import math
import os
import textwrap

import cv2
import numpy as np

from autopark import parking_goal as pg
from autopark import slot_codec as sc
from autopark.hybrid_astar import CarGeometry
from autopark.odometry import compose, inverse
from autopark.slot_viz import _px

SCALE = 1.6                  # BEV 450 px -> 720 px
PANEL_W, H = 400, 720
MAP_PX_PER_M = 14.0
COL = dict(vacant=(0, 210, 0), occupied=(0, 0, 220), target=(255, 255, 0), det=(0, 230, 255),
           fwd=(255, 140, 0), rev=(0, 140, 255), goal=(255, 0, 255), car=(255, 255, 255),
           trail=(180, 180, 180), text=(235, 235, 235), dim=(140, 140, 140))
CAPTIONS = {
    'idle': 'Starting',
    'search': 'Searching: driving along the aisle, detecting and tracking slots',
    'stopping': 'Vacant slot passed: stopping',
    'planning': 'Planning a reverse-in path (Hybrid A* + Reeds-Shepp)',
    'executing': 'Driving the path (Stanley controller, forward and reverse)',
    'done': 'Parked',
    'failed': 'Failed',
}


def slot_corners(x, y, th, width=2.6, depth=sc.NOMINAL_DEPTH):
    c, s = math.cos(th), math.sin(th)
    lx, ly = -s * width / 2, c * width / 2
    return [(x + lx, y + ly), (x - lx, y - ly), (x - lx + depth * c, y - ly + depth * s),
            (x + lx + depth * c, y + ly + depth * s)]


def to_frame(pose, pts):
    """odom-frame points -> frame of `pose` (the car frame for the current odom pose)."""
    inv = inverse(pose)
    return [compose(inv, (x, y, 0.0))[:2] for x, y in pts]


def _poly(img, pts, color, thick=1, closed=True, px=None):
    px = px or (lambda x, y: tuple(int(round(v * SCALE)) for v in _px(x, y)))
    p = np.array([px(x, y) for x, y in pts], np.int32)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA)


def draw_bev(bev, pose, dets, tracks, target, path, exec_path, car=None):
    """bev: BEV image (car frame at `pose`); dets: [(x, y, th)] car frame; tracks: [(id, x, y,
    th, selectable)] odom frame; path / exec_path: (xs, ys, dirs) odom frame or None."""
    car = car or CarGeometry()
    img = cv2.resize(bev, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    img = (img * 0.85).astype(np.uint8)

    def px(x, y):
        c, r = _px(x, y)
        return int(round(c * SCALE)), int(round(r * SCALE))

    for tid, x, y, th, sel in tracks:
        color = COL['target'] if tid == target else COL['vacant'] if sel else COL['occupied']
        _poly(img, to_frame(pose, slot_corners(x, y, th)), color, 3 if tid == target else 1, px=px)
    for x, y, th in dets:
        a = px(x, y)
        b = px(x + 0.9 * math.cos(th), y + 0.9 * math.sin(th))
        cv2.arrowedLine(img, a, b, COL['det'], 2, cv2.LINE_AA, tipLength=0.3)
    for p, thick in ((path, 1), (exec_path, 3)):
        if p is None or len(p[0]) < 2:
            continue
        xs, ys, dirs = p[:3]
        q = to_frame(pose, list(zip(xs, ys)))
        for i in range(1, len(q)):
            cv2.line(img, px(*q[i - 1]), px(*q[i]), COL['fwd'] if dirs[i] > 0 else COL['rev'], thick, cv2.LINE_AA)
    if path is not None and len(path[0]):
        g = compose(inverse(pose), (path[0][-1], path[1][-1], path[3] if len(path) > 3 else 0.0))
        _poly(img, car.corners(g), COL['goal'], 1, px=px)
    _poly(img, car.corners((0.0, 0.0, 0.0)), COL['car'], 2, px=px)
    return img


def draw_panel(pose, trail, tracks, target, path, lines, caption):
    """Right panel: manoeuvre map (odom frame, centred on the car) and status text."""
    panel = np.full((H, PANEL_W, 3), 32, np.uint8)
    cx, cy, map_h = PANEL_W / 2, 250, 440

    def px(x, y):          # odom x -> right, odom y -> up, centred on the car
        return int(round(cx + (x - pose[0]) * MAP_PX_PER_M)), int(round(cy - (y - pose[1]) * MAP_PX_PER_M))

    cv2.rectangle(panel, (0, cy - map_h // 2), (PANEL_W - 1, cy + map_h // 2), (50, 50, 50), 1)
    mask = np.zeros_like(panel)
    for tid, x, y, th, sel in tracks:
        color = COL['target'] if tid == target else COL['vacant'] if sel else COL['occupied']
        _poly(mask, slot_corners(x, y, th), color, 2 if tid == target else 1, px=px)
    if len(trail) > 1:
        _poly(mask, trail, COL['trail'], 1, closed=False, px=px)
    if path is not None and len(path[0]) > 1:
        xs, ys, dirs = path[:3]
        for i in range(1, len(xs)):
            cv2.line(mask, px(xs[i - 1], ys[i - 1]), px(xs[i], ys[i]), COL['fwd'] if dirs[i] > 0 else COL['rev'],
                     1, cv2.LINE_AA)
    _poly(mask, CarGeometry().corners(pose), COL['car'], 2, px=px)
    band = slice(cy - map_h // 2 + 1, cy + map_h // 2)
    panel[band] = np.maximum(panel[band], mask[band])
    cv2.putText(panel, 'map (odometry frame)', (8, cy - map_h // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COL['dim'], 1, cv2.LINE_AA)
    y = cy + map_h // 2 + 26
    for part in textwrap.wrap(caption, 38):
        cv2.putText(panel, part, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL['target'], 1, cv2.LINE_AA)
        y += 21
    for k, line in enumerate(lines):
        cv2.putText(panel, line, (8, y + 6 + 21 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL['text'], 1, cv2.LINE_AA)
    return panel


def legend(img):
    items = (('detection', COL['det']), ('vacant', COL['vacant']), ('occupied', COL['occupied']),
             ('target', COL['target']), ('forward', COL['fwd']), ('reverse', COL['rev']), ('goal', COL['goal']))
    x = 8
    for name, color in items:
        cv2.rectangle(img, (x, H - 20), (x + 12, H - 8), color, -1)
        cv2.putText(img, name, (x + 16, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL['text'], 1, cv2.LINE_AA)
        x += 22 + 8 * len(name)
    return img


def status_lines(status, control, speed, steer, mode):
    lines = [f'mode: {mode}']
    if status is not None:
        lines.append(f'state: {status.state}   target slot: {status.slot_id if status.slot_id >= 0 else "-"}')
        lines.append(f'plans: {status.plans}   corrections: {status.corrections}')
    if control is not None and control.state in ('align', 'track', 'stop'):
        gear = 'forward' if control.direction > 0 else 'reverse'
        lines.append(f'segment {control.segment + 1}/{control.segments} ({gear}), {control.remaining:.1f} m left')
        lines.append(f'tracking error {100 * control.lateral_error:+.1f} cm, '
                     f'{math.degrees(control.heading_error):+.1f} deg')
    lines.append(f'speed {speed:+.2f} m/s   steering {math.degrees(steer):+.1f} deg')
    return lines


def compose_frame(bev, pose, dets, tracks, target, path, exec_path, trail, lines, caption):
    left = legend(draw_bev(bev, pose, dets, tracks, target, path, exec_path))
    return np.hstack([left, draw_panel(pose, trail, tracks, target, path, lines, caption)])


# ---------------------------------------------------------------------------------- ROS node
def main():
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from autopark_msgs.msg import ControlStatus, ParkingPath, ParkingSlotArray, ParkingStatus
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Int64, String

    from autopark.odometry import yaw_from_quaternion

    def ms(st):
        return st.sec * 1000 + int(round(st.nanosec / 1e6))

    class Visualizer(Node):
        def __init__(self):
            super().__init__('visualizer')
            self.declare_parameter('mode', 'closed')
            self.declare_parameter('record_dir', '')
            self.mode = self.get_parameter('mode').value
            self.record_dir = self.get_parameter('record_dir').value
            self.bridge = CvBridge()
            self.odom, self.pose, self.trail = {}, None, []
            self.dets, self.tracks, self.path, self.exec = [], [], None, None
            self.status = self.control = None
            self.speed = self.steer = 0.0
            self.pub = self.create_publisher(Image, '/viz/image', 2)
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Image, '/bev/image', self.on_bev, 2)
            self.create_subscription(ParkingSlotArray, '/slots/detections', self.on_dets, 5)
            self.create_subscription(ParkingSlotArray, '/slots/tracked', self.on_tracks, 5)
            self.create_subscription(ParkingPath, '/parking/path', self.on_path, latched)
            self.create_subscription(ParkingPath, '/parking/path_exec', self.on_exec, 10)
            self.create_subscription(ParkingStatus, '/parking/status', lambda m: setattr(self, 'status', m), 10)
            self.create_subscription(ControlStatus, '/parking/control_status',
                                     lambda m: setattr(self, 'control', m), 10)
            self.create_subscription(AckermannDriveStamped, '/vehicle/state', self.on_vehicle, 10)
            self.create_subscription(Odometry, '/odom', self.on_odom, 50)
            self.create_subscription(Int64, '/scenario/reset', self.on_reset, 10)
            self.index = None
            if self.record_dir:
                os.makedirs(os.path.join(self.record_dir, 'viz'), exist_ok=True)
                self.index = open(os.path.join(self.record_dir, 'viz', 'index.csv'), 'a')
                self.rec_pub = self.create_publisher(String, '/demo/record', 10)
                self.create_timer(1.0, self.start_3d)
                self.started_3d = False

        def start_3d(self):
            if not self.started_3d and self.rec_pub.get_subscription_count() > 0:
                self.rec_pub.publish(String(data=os.path.join(self.record_dir, '3d')))
                self.started_3d = True

        def on_reset(self, _m):
            self.trail, self.tracks, self.path, self.exec, self.dets = [], [], None, None, []

        def on_vehicle(self, m):
            self.speed, self.steer = m.drive.speed, m.drive.steering_angle

        def on_odom(self, m):
            q = m.pose.pose.orientation
            p = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
            self.odom[ms(m.header.stamp)] = p
            if len(self.odom) > 400:
                for k in sorted(self.odom)[:-200]:
                    del self.odom[k]
            if not self.trail or math.hypot(p[0] - self.trail[-1][0], p[1] - self.trail[-1][1]) > 0.1:
                self.trail.append(p[:2])

        def on_dets(self, m):
            self.dets = [(s.entrance.x, s.entrance.y, s.entrance.theta) for s in m.slots]

        def on_tracks(self, m):
            self.tracks = [(s.id, s.entrance.x, s.entrance.y, s.entrance.theta,
                            pg.selectable(s.vacancy, s.confidence)) for s in m.slots]

        @staticmethod
        def _path(m):
            if not m.poses:
                return None
            return ([p.x for p in m.poses], [p.y for p in m.poses], list(m.direction), m.poses[-1].theta)

        def on_path(self, m):
            self.path = self._path(m)

        def on_exec(self, m):
            self.exec = self._path(m)

        def on_bev(self, m):
            t = ms(m.header.stamp)
            pose = self.odom.get(t) or (self.odom[max(self.odom)] if self.odom else None)
            if pose is None:
                return
            bev = self.bridge.imgmsg_to_cv2(m, 'bgr8')
            target = self.status.slot_id if self.status is not None else -1
            state = self.status.state if self.status is not None else 'idle'
            img = compose_frame(bev, pose, self.dets, self.tracks, target, self.path, self.exec, self.trail,
                                status_lines(self.status, self.control, self.speed, self.steer, self.mode),
                                CAPTIONS.get(state, state))
            out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            out.header = m.header
            self.pub.publish(out)
            if self.index is not None:
                cv2.imwrite(os.path.join(self.record_dir, 'viz', f'{t:08d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.index.write(f'{t},{state}\n')
                self.index.flush()

    rclpy.init()
    node = Visualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.record_dir and node.started_3d:
            node.rec_pub.publish(String(data=''))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
