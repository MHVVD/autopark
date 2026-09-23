"""Record a drive and evaluate the vehicle interface + odometry against ground truth.

Records /ground_truth/pose, /vehicle/state, /imu, /odom and /cmd_ackermann for `duration`
seconds of simulated time, then prints:
  * convention checks: forward/reverse and left/right as seen by ground truth
  * sensor accuracy: encoder speed, steering angle, gyro vs ground truth
  * odometry drift of the running /odom node, and offline dead reckoning with
    heading from the IMU and from the steering angle (same recorded data)
and writes <out_dir>/odom_eval.csv and odom_eval.png.
"""
import math
import os
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

from autopark.odometry import (AckermannOdometry, compose, inverse, wrap, yaw_from_quaternion,
                               yaw_rate_from_steering)

WHEEL_BASE = 2.8


def _t(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _key(t):
    return int(round(t * 1000))  # ms: all simulation stamps are multiples of the 20 ms step


def dead_reckon(states, gyro, heading_source, wheel_base=WHEEL_BASE):
    """states: [(t, v, steer)], gyro: {ms: gz}. Returns [(t, x, y, yaw, distance)]."""
    odo = AckermannOdometry()
    out = []
    last = None
    for t, v, steer in states:
        if heading_source == 'imu':
            if _key(t) not in gyro:
                continue
            w = gyro[_key(t)]
        else:
            w = yaw_rate_from_steering(v, steer, wheel_base)
        if last is not None:
            odo.update(v, w, t - last)
        last = t
        out.append((t, odo.x, odo.y, odo.yaw, odo.distance))
    return out


def drift(track, gt, base):
    """Errors of an odom-frame track against ground truth. Returns list of
    (t, distance, pos_err, yaw_err, x_map, y_map)."""
    rows = []
    for t, x, y, yaw, dist in track:
        g = gt.get(_key(t))
        if g is None:
            continue
        mx, my, myaw = compose(base, (x, y, yaw))
        rows.append((t, dist, math.hypot(mx - g[0], my - g[1]), wrap(myaw - g[2]), mx, my))
    return rows


class OdomEval(Node):
    def __init__(self):
        super().__init__('odom_eval')
        self.declare_parameter('duration', 30.0)
        self.declare_parameter('out_dir', os.path.expanduser('~/autopark_results/odom_eval'))
        self.duration = self.get_parameter('duration').value
        self.out_dir = self.get_parameter('out_dir').value

        self.gt = {}        # ms -> (x, y, yaw, vx_body, wz)
        self.states = []    # (t, v, steer)
        self.gyro = {}      # ms -> gz
        self.odom = []      # (t, x, y, yaw)
        self.cmds = []      # (t, v, steer)
        self.t_first = None
        self.finished = False
        self.create_subscription(Odometry, '/ground_truth/pose', self.on_gt, 100)
        self.create_subscription(AckermannDriveStamped, '/vehicle/state', self.on_state, 100)
        self.create_subscription(Imu, '/imu', self.on_imu, 100)
        self.create_subscription(Odometry, '/odom', self.on_odom, 100)
        self.create_subscription(AckermannDriveStamped, '/cmd_ackermann', self.on_cmd, 100)
        self.get_logger().info(f'recording {self.duration} s of simulated time ...')

    def on_gt(self, m):
        p, tw = m.pose.pose, m.twist.twist
        q = p.orientation
        self.gt[_key(_t(m.header.stamp))] = (p.position.x, p.position.y,
                                            yaw_from_quaternion(q.x, q.y, q.z, q.w),
                                            tw.linear.x, tw.angular.z)

    def on_state(self, m):
        t = _t(m.header.stamp)
        if self.t_first is None:
            self.t_first = t
        self.states.append((t, m.drive.speed, m.drive.steering_angle))
        if t - self.t_first >= self.duration:
            self.finished = True

    def on_imu(self, m):
        self.gyro[_key(_t(m.header.stamp))] = m.angular_velocity.z

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.odom.append((_t(m.header.stamp), m.pose.pose.position.x, m.pose.pose.position.y,
                          yaw_from_quaternion(q.x, q.y, q.z, q.w)))

    def on_cmd(self, m):
        self.cmds.append((self.states[-1][0] if self.states else 0.0,
                          m.drive.speed, m.drive.steering_angle))

    # ------------------------------------------------------------------ analysis
    def analyse(self):
        lines = []
        pr = lines.append
        st = [s for s in self.states if _key(s[0]) in self.gt]
        pr(f'samples: {len(self.states)} states, {len(self.gt)} ground-truth poses, '
           f'{len(self.odom)} odom, {len(st)} matched')

        # Conventions (only while clearly moving)
        mov = [(v, steer, self.gt[_key(t)]) for t, v, steer in st if abs(self.gt[_key(t)][3]) > 0.3]
        fwd = [g for v, s, g in mov if v > 0.3]
        rev = [g for v, s, g in mov if v < -0.3]
        pr('\nconventions (ground truth while moving):')
        pr(f'  measured speed > 0 -> GT moves forward : {sum(g[3] > 0 for g in fwd)}/{len(fwd)}')
        pr(f'  measured speed < 0 -> GT moves backward: {sum(g[3] < 0 for g in rev)}/{len(rev)}')
        left_f = [g for v, s, g in mov if v > 0.3 and s > 0.1]
        left_r = [g for v, s, g in mov if v < -0.3 and s > 0.1]
        pr(f'  steer left + forward -> yaw increases  : {sum(g[4] > 0 for g in left_f)}/{len(left_f)}')
        pr(f'  steer left + reverse -> yaw decreases  : {sum(g[4] < 0 for g in left_r)}/{len(left_r)}')

        # Sensor accuracy
        if mov:
            v_err = np.array([v - g[3] for v, s, g in mov])
            implied = np.array([math.atan(WHEEL_BASE * g[4] / g[3]) for v, s, g in mov])
            s_err = np.array([s for v, s, g in mov]) - implied
            g_err = np.array([self.gyro[_key(t)] - self.gt[_key(t)][4] for t, v, s in st
                              if _key(t) in self.gyro])
            pr('\nsensors vs ground truth:')
            pr(f'  encoder speed   : mean {v_err.mean():+.4f} m/s, RMS {np.sqrt((v_err**2).mean()):.4f} m/s')
            pr(f'  steering angle  : mean {s_err.mean():+.4f} rad, RMS {np.sqrt((s_err**2).mean()):.4f} rad'
               '  (vs angle implied by GT yaw rate)')
            pr(f'  gyro yaw rate   : mean {g_err.mean():+.5f} rad/s, RMS {np.sqrt((g_err**2).mean()):.5f} rad/s')

        # Odometry drift
        if not self.odom:
            pr('\nno /odom received')
            return lines, {}
        # Start at the first odom sample that has ground truth (the subscriptions may connect
        # in any order), and express everything relative to the odom pose at that moment.
        first = next((o for o in self.odom if _key(o[0]) in self.gt), None)
        if first is None:
            pr('\nno /odom sample with matching ground truth')
            return lines, {}
        t0 = first[0]
        g0 = self.gt[_key(t0)]
        # base = map pose of the odom frame origin: GT pose composed with the inverse odom pose
        base = compose((g0[0], g0[1], g0[2]), inverse(first[1:4]))
        self.odom = [o for o in self.odom if o[0] >= t0]
        # The offline tracks start at t0 in their own frame, anchored at the GT pose at t0.
        base_offline = (g0[0], g0[1], g0[2])
        tracks = {
            'odom node': [(t, x, y, yaw, np.nan) for t, x, y, yaw in self.odom],
            'offline imu': dead_reckon([s for s in self.states if s[0] >= t0], self.gyro, 'imu'),
            'offline steering': dead_reckon([s for s in self.states if s[0] >= t0], self.gyro,
                                            'steering'),
        }
        bases = {'odom node': base, 'offline imu': base_offline,
                 'offline steering': base_offline}
        dist_ref = {r[0]: r[4] for r in tracks['offline imu']}
        pr('\nodometry drift vs ground truth:')
        results = {}
        for name, track in tracks.items():
            rows = drift(track, self.gt, bases[name])
            if not rows:
                continue
            rows = [(t, dist_ref.get(t, d), pe, ye, mx, my) for t, d, pe, ye, mx, my in rows]
            results[name] = rows
            t, d, pe, ye, _, _ = rows[-1]
            maxe = max(r[2] for r in rows)
            pct = 100 * pe / d if d > 0 else float('nan')
            pr(f'  {name:17s}: path {d:6.2f} m | final pos err {pe:.3f} m ({pct:.2f} % of path)'
               f' | max pos err {maxe:.3f} m | final yaw err {math.degrees(ye):+.2f} deg')
        return lines, results

    def save(self, results):
        os.makedirs(self.out_dir, exist_ok=True)
        import pandas as pd
        frames = [pd.DataFrame(rows, columns=['t', 'distance', 'pos_err', 'yaw_err', 'x', 'y'])
                  .assign(track=name) for name, rows in results.items()]
        gt = sorted(self.gt.items())
        frames.append(pd.DataFrame([(k / 1000, np.nan, 0, 0, v[0], v[1]) for k, v in gt],
                                   columns=['t', 'distance', 'pos_err', 'yaw_err', 'x', 'y'])
                      .assign(track='ground truth'))
        df = pd.concat(frames)
        csv = os.path.join(self.out_dir, 'odom_eval.csv')
        df.to_csv(csv, index=False)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
        t_lo = min(r[0] for rows in results.values() for r in rows)
        t_hi = max(r[0] for rows in results.values() for r in rows)
        g = df[(df.track == 'ground truth') & (df.t >= t_lo) & (df.t <= t_hi)]
        a.plot(g.x, g.y, 'k-', lw=2.5, label='ground truth')
        for name in results:
            d = df[df.track == name]
            a.plot(d.x, d.y, '--', lw=1.3, label=name)
            b.plot(d.t - t_lo, d.pos_err, lw=1.3, label=name)
        a.set_aspect('equal', adjustable='datalim')
        a.set_xlabel('x [m]')
        a.set_ylabel('y [m]')
        a.set_title('trajectory (map frame)')
        a.legend(fontsize=8, loc='lower right')
        b.set_xlabel('time [s]')
        b.set_ylabel('position error [m]')
        b.set_title('odometry position error')
        b.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(self.out_dir, 'odom_eval.png')
        fig.savefig(png, dpi=110)
        return csv, png


def main():
    rclpy.init()
    node = OdomEval()
    t_wall = time.monotonic()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
            if not node.states and time.monotonic() - t_wall > 60:
                node.get_logger().error('no /vehicle/state received in 60 s -- is the sim running?')
                break
    except KeyboardInterrupt:
        pass
    if node.states:
        lines, results = node.analyse()
        print('\n'.join(lines), flush=True)
        if results:
            csv, png = node.save(results)
            print(f'\nwrote {csv}\nwrote {png}', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
