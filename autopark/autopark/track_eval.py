"""Evaluate the slot tracker against ground truth while driving (evaluation tool, uses GT).

For each seed: reset, drive forward along the aisle (weaving) to X_TURN, then reverse back to
X_BACK so slots leave the view and come back. Records the tracker input (detections, car
frame) and output (/slots/tracked, odom frame).

Errors are measured *relative to the car*, as the planner uses them: a track is moved into the
car frame with the odometry pose at its stamp, the ground-truth slot with the ground-truth
pose at the same stamp. (Absolute errors in the frame fixed at the start of the run also
contain the odometry drift accumulated over the whole drive; reported as abs_*.)

Reports (one CSV row per run appended to <out>):
  det / track entrance error (median, p90) and heading error in the car frame, track NEES
  in the car frame (consistency; 3 = ideal), IDs per slot (1 = stable), false confirmed
  tracks, out-of-view memory error, selectable-vacant precision / recall at the end of the run
  (selectable = vacancy > 0.95 and confidence >= 0.3, i.e. >= 3 close observations).
"""
import csv
import math
import os
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autopark_msgs.msg import ParkingSlotArray
from autopark_msgs.srv import ResetScenario
from nav_msgs.msg import Odometry
from rclpy.node import Node

from autopark.collect_slots import pure_pursuit
from autopark.odometry import compose, inverse, wrap, yaw_from_quaternion

X_TURN, X_BACK = 12.0, -4.0
MATCH = 1.0            # m, association to ground truth for evaluation
SETTLE = 1.5


def _sec(st):
    return st.sec + st.nanosec * 1e-9


def _ms(t):
    return int(round(t * 50)) * 20


class TrackEval(Node):
    def __init__(self):
        super().__init__('track_eval', parameter_overrides=[])
        self.declare_parameter('seeds', [0, 1, 2])
        self.declare_parameter('label', 'default')
        self.declare_parameter('detections_topic', '/slots/detections_noisy')
        self.declare_parameter('speed', 1.2)
        self.declare_parameter('out', os.path.expanduser('~/autopark_results/track_eval/runs.csv'))
        p = self.get_parameter
        self.seeds = list(p('seeds').value)
        self.label = p('label').value
        self.speed = p('speed').value
        self.out = p('out').value
        self.cmd = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
        self.reset_cli = self.create_client(ResetScenario, '/ground_truth/reset')
        self.create_subscription(Odometry, '/ground_truth/pose', self.on_gt, 200)
        self.create_subscription(Odometry, '/odom', self.on_odom, 200)
        self.create_subscription(ParkingSlotArray, '/ground_truth/slots', self.on_gt_slots, 10)
        self.create_subscription(ParkingSlotArray, p('detections_topic').value, self.on_dets, 20)
        self.create_subscription(ParkingSlotArray, '/slots/tracked', self.on_tracks, 20)
        self.state, self.i = 'idle', -1
        self.t = 0.0
        self.results = []

    # --------------------------------------------------------------- recording
    def start_run(self):
        self.gt, self.odom, self.gt_slots = {}, {}, None
        self.base = None
        self.det_rows, self.trk_rows = [], []       # (dist, heading error)
        self.abs_err = []
        self.offset_hist = {k: 0 for k in (-2, -1, 0, 1, 2)}   # diagnostic: best stamp offset per frame
        self.ids_per_gt, self.false_tracks = {}, set()
        self.nees, self.memory_err = [], []
        self.final_tracks = []

    def on_gt(self, m):
        q = m.pose.pose.orientation
        pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.t = _sec(m.header.stamp)
        if self.state == 'drive':
            self.gt[_ms(self.t)] = pose
            self.drive(pose)

    def on_odom(self, m):
        if self.state != 'drive':
            return
        q = m.pose.pose.orientation
        t = _ms(_sec(m.header.stamp))
        self.odom[t] = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        if self.base is None and t in self.gt and self.t - self.reset_t > SETTLE:
            self.base = compose(self.gt[t], inverse(self.odom[t]))   # odom origin in map

    def on_gt_slots(self, m):
        if self.state == 'drive' and _sec(m.header.stamp) >= self.reset_t:
            self.gt_slots = [(s.entrance.x, s.entrance.y, s.entrance.theta, s.vacancy > 0.5) for s in m.slots]

    def match(self, x, y):
        d = [math.hypot(g[0] - x, g[1] - y) for g in self.gt_slots]
        k = int(np.argmin(d))
        return k, d[k]

    def on_dets(self, m):
        if self.state != 'drive' or self.base is None or self.gt_slots is None:
            return
        t = _ms(_sec(m.header.stamp))
        if t not in self.odom:
            return
        g = self.gt.get(t)
        if g is None:
            return
        # diagnostic: which ground-truth time (stamp + k steps) explains this frame best
        errs = {}
        for off in self.offset_hist:
            go = self.gt.get(t + 20 * off)
            if go is None or not m.slots:
                continue
            e = []
            for s in m.slots:
                x, y, _ = compose(go, (s.entrance.x, s.entrance.y, s.entrance.theta))
                e.append(self.match(x, y)[1])
            errs[off] = float(np.median(e))
        if len(errs) == 5:
            self.offset_hist[min(errs, key=errs.get)] += 1
        for s in m.slots:
            x, y, th = compose(g, (s.entrance.x, s.entrance.y, s.entrance.theta))   # car -> map via GT
            k, d = self.match(x, y)
            if d < MATCH:
                self.det_rows.append((d, abs(wrap(th - self.gt_slots[k][2]))))

    def on_tracks(self, m):
        if self.state != 'drive' or self.base is None or self.gt_slots is None:
            return
        t = _ms(_sec(m.header.stamp))
        if t not in self.odom or t not in self.gt:
            return
        odom, gtp = self.odom[t], self.gt[t]
        visible = self.visible_ids(t)
        snapshot = []
        for s in m.slots:
            # track -> car frame (odometry) -> map with the true car pose: error relative to the car
            car = compose(inverse(odom), (s.entrance.x, s.entrance.y, s.entrance.theta))
            x, y, th = compose(gtp, car)
            k, d = self.match(x, y)
            if d >= MATCH:
                self.false_tracks.add(s.id)
                continue
            g = self.gt_slots[k]
            self.ids_per_gt.setdefault(k, set()).add(s.id)
            eh = wrap(th - g[2])
            self.trk_rows.append((d, abs(eh)))
            ax, ay, _ = compose(self.base, (s.entrance.x, s.entrance.y, s.entrance.theta))
            self.abs_err.append(math.hypot(ax - g[0], ay - g[1]))
            # NEES: error rotated from map into odom (P is in odom): map->car (true yaw), car->odom
            rot = odom[2] - gtp[2]
            c, sn = math.cos(rot), math.sin(rot)
            ex, ey = x - g[0], y - g[1]
            e = np.array([c * ex - sn * ey, sn * ex + c * ey, eh])
            P = np.array(s.covariance).reshape(3, 3)
            self.nees.append(float(e @ np.linalg.solve(P, e)))
            if k not in visible:
                self.memory_err.append(d)
            snapshot.append((k, s.vacancy, s.confidence))
        self.final_tracks = snapshot

    def visible_ids(self, t):
        """GT slots whose entrance is within 8 m of the car centre and inside the BEV."""
        g = self.gt.get(t)
        if g is None:
            return set()
        out = set()
        inv = inverse(g)
        for k, s in enumerate(self.gt_slots):
            x, y, _ = compose(inv, (s[0], s[1], 0.0))
            if -6.1 < x < 8.8 and abs(y) < 7.5 and math.hypot(x - 1.35, y) < 8.0:
                out.add(k)
        return out

    # --------------------------------------------------------------- driving
    def drive(self, pose):
        msg = AckermannDriveStamped()
        if self.t - self.reset_t < SETTLE:
            self.cmd.publish(msg)
            return
        x, y, yaw = pose
        if self.phase == 'fwd':
            if x > X_TURN:
                self.phase = 'stop'
                self.stop_t = self.t
            else:
                msg.drive.speed = self.speed
                msg.drive.steering_angle = max(-0.5, min(0.5, pure_pursuit(pose, lambda v: 0.6 * math.sin(v / 3.0))))
        elif self.phase == 'stop':
            if self.t - self.stop_t > 1.5:
                self.phase = 'rev'
        elif self.phase == 'rev':
            if x < X_BACK:
                self.phase = 'done'
                self.done_t = self.t
            else:
                # reversing: yaw rate = v tan(steer) / L with v < 0, so steering has the opposite
                # effect. Aim for yaw_d = k*y (moving backwards with yaw > 0 reduces y).
                yaw_d = max(-0.3, min(0.3, 0.3 * y))
                msg.drive.speed = -self.speed
                msg.drive.steering_angle = max(-0.5, min(0.5, 1.5 * wrap(yaw - yaw_d)))
        elif self.phase == 'done' and self.t - self.done_t > 1.0:
            self.finish_run()
        self.cmd.publish(msg)

    def finish_run(self):
        seed = self.seeds[self.i]
        r = self.summarise(seed)
        self.results.append(r)
        self.get_logger().info(' | '.join(f'{k} {v}' for k, v in r.items()))
        self.state = 'next'

    def summarise(self, seed):
        d = np.array(self.det_rows) if self.det_rows else np.zeros((0, 2))
        tr = np.array(self.trk_rows) if self.trk_rows else np.zeros((0, 2))
        q = lambda a, p: round(float(np.percentile(a, p)) * 100, 1) if len(a) else float('nan')  # noqa: E731
        sel = [(k, v) for k, v, c in self.final_tracks if v > 0.95 and c >= 0.3]
        gt_vac = {k for k, s in enumerate(self.gt_slots or []) if s[3]}
        seen = {k for k in self.ids_per_gt}
        tp_slots = {k for k, _ in sel if k in gt_vac}             # distinct vacant slots found
        tp = sum(1 for k, _ in sel if k in gt_vac)
        return dict(label=self.label, seed=seed,
                    det_n=len(d), det_med_cm=q(d[:, 0], 50), det_p90_cm=q(d[:, 0], 90),
                    det_head_med_deg=round(math.degrees(np.median(d[:, 1])), 2) if len(d) else float('nan'),
                    trk_n=len(tr), trk_med_cm=q(tr[:, 0], 50), trk_p90_cm=q(tr[:, 0], 90),
                    trk_head_med_deg=round(math.degrees(np.median(tr[:, 1])), 2) if len(tr) else float('nan'),
                    abs_med_cm=q(self.abs_err, 50),
                    stamp_offset_hist='/'.join(str(v) for v in self.offset_hist.values()),
                    nees_mean=round(float(np.mean(self.nees)), 2) if self.nees else float('nan'),
                    memory_med_cm=q(self.memory_err, 50), memory_n=len(self.memory_err),
                    slots_tracked=len(seen), ids_per_slot=round(np.mean([len(v) for v in self.ids_per_gt.values()]), 3) if seen else float('nan'),
                    false_tracks=len(self.false_tracks),
                    vac_selectable=len(sel), vac_precision=round(tp / len(sel), 3) if sel else float('nan'),
                    vac_recall=round(len(tp_slots) / max(len(gt_vac & seen), 1), 3))

    def step(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        if self.state in ('idle', 'next'):
            if not self.reset_cli.service_is_ready():
                return
            self.i += 1
            if self.i >= len(self.seeds):
                self.state = 'done'
                return
            self.start_run()
            req = ResetScenario.Request()
            req.seed = self.seeds[self.i]
            self.future = self.reset_cli.call_async(req)
            self.state = 'resetting'
        elif self.state == 'resetting' and self.future.done():
            self.reset_t = self.t
            self.phase = 'fwd'
            self.state = 'drive'

    def save(self):
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        new = not os.path.exists(self.out)
        with open(self.out, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.results[0]))
            if new:
                w.writeheader()
            w.writerows(self.results)


def main():
    rclpy.init()
    node = TrackEval()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and node.state != 'done':
            node.step()
    except KeyboardInterrupt:
        pass
    if node.results:
        node.save()
        print(f'{len(node.results)} runs in {time.monotonic() - t0:.0f} s -> {node.out}', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
