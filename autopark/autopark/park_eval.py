"""Evaluate autonomous parking runs against ground truth (evaluation tool, uses GT).

Run the stack with the parking manager (bringup park:=true). For each seed: reset the
scenario, let the manager search, plan and park, and record the TRUE car pose (supervisor)
until the manager reports done or failed (or `timeout` s of simulation time). Scores:
  success         manager done AND the true final car rectangle is inside the lines of a
                  slot that is really empty AND no contact with a parked car along the
                  whole true trajectory (exact rectangle geometry, autopark.geometry_check)
  final error     true car centre / heading relative to that slot's centre line:
                  lateral (cm), depth (cm, + = deeper), heading (deg)
  min clearance   along the true trajectory to the true parked cars (m)
  time            from the start of the search to done; plans, corrections, max tracking
                  error (controller)
The true trajectory starts at the first ground-truth pose near the scenario's start pose:
poses still queued from before the reset (the car where the previous run left it) are
dropped and counted (stale_poses_dropped). Without this, a backlog under CPU load put the
previous parked pose into the new scenario, where another car may stand: a false contact.

    ros2 run autopark park_eval --ros-args -p use_sim_time:=true -p seeds:=[0,1,2] -p label:=closed
"""
import csv
import json
import math
import os
import time

import numpy as np
import rclpy
from autopark_msgs.msg import ControlStatus, ParkingStatus
from autopark_msgs.srv import ResetScenario
from nav_msgs.msg import Odometry
from rclpy.node import Node

from autopark.geometry_check import path_clearance, point_in_convex
from autopark.hybrid_astar import CarGeometry
from autopark.odometry import wrap, yaw_from_quaternion
from autopark.plan_bench import true_car_rects
from autopark_sim.lot import slots as lot_slots
from autopark_sim.scenario import make_scenario


def _sec(st):
    return st.sec + st.nanosec * 1e-9


class ParkEval(Node):
    def __init__(self):
        super().__init__('park_eval')
        self.declare_parameter('seeds', [0, 1, 2])
        self.declare_parameter('label', 'default')
        self.declare_parameter('timeout', 240.0)
        self.declare_parameter('out', os.path.expanduser('~/autopark_results/park_eval'))
        p = self.get_parameter
        self.seeds, self.label = list(p('seeds').value), p('label').value
        self.timeout, self.out = p('timeout').value, p('out').value
        self.reset_cli = self.create_client(ResetScenario, '/ground_truth/reset')
        self.create_subscription(Odometry, '/ground_truth/pose', self.on_gt, 100)
        self.create_subscription(ParkingStatus, '/parking/status', self.on_status, 10)
        self.create_subscription(ControlStatus, '/parking/control_status', self.on_control, 50)
        self.car = CarGeometry()
        self.lot = lot_slots()
        self.state, self.i, self.t = 'idle', -1, 0.0
        self.rows = []

    def on_gt(self, m):
        q = m.pose.pose.orientation
        self.t = _sec(m.header.stamp)
        if self.state == 'run':
            p = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
            if not self.traj and math.hypot(p[0] - self.scenario.ego[0], p[1] - self.scenario.ego[1]) > 0.5:
                self.stale += 1     # queued pose from before the reset (the previous scenario)
                return
            self.traj.append(p)

    def on_status(self, m):
        if self.state != 'run' or self.t < self.reset_t + 0.5:
            return
        self.status = m
        if m.state == 'search' and self.t_search is None:
            self.t_search = self.t
        if m.state in ('done', 'failed') and self.t_end is None:
            self.t_end = self.t

    def on_control(self, m):
        if self.state == 'run':
            self.max_track = max(self.max_track, m.max_lateral_error)

    def step(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        if self.state in ('idle', 'next'):
            if not self.reset_cli.service_is_ready():
                return
            self.i += 1
            if self.i >= len(self.seeds):
                self.state = 'done'
                return
            req = ResetScenario.Request()
            req.seed = self.seeds[self.i]
            self.scenario = make_scenario(req.seed)
            self.cars = true_car_rects(self.scenario)
            self.future = self.reset_cli.call_async(req)
            self.state = 'resetting'
        elif self.state == 'resetting' and self.future.done():
            self.reset_t = self.t
            self.traj, self.status, self.t_search, self.t_end, self.max_track = [], None, None, None, 0.0
            self.stale = 0
            self.state = 'run'
        elif self.state == 'run':
            finished = self.t_end is not None and self.t - self.t_end > 2.0     # let the car settle
            if finished or self.t - self.reset_t > self.timeout:
                self.record(timed_out=not finished)
                self.state = 'next'

    def record(self, timed_out):
        seed = self.seeds[self.i]
        st = self.status
        row = dict(label=self.label, seed=seed, manager_state=st.state if st else 'none',
                   message=st.message if st else '', timed_out=timed_out,
                   plans=st.plans if st else 0, corrections=st.corrections if st else 0,
                   time=(self.t_end - self.t_search) if (self.t_end and self.t_search) else float('nan'),
                   max_track_cm=100 * self.max_track, stale_poses_dropped=self.stale)
        traj = self.traj[25:] if len(self.traj) > 50 else self.traj      # skip the spawn settling
        os.makedirs(os.path.join(self.out, 'traj'), exist_ok=True)       # true trajectory, for plots / checks
        np.save(os.path.join(self.out, 'traj', f'{self.label}_{seed}.npy'), np.asarray(traj, np.float32))
        xs, ys, th = (np.array(v) for v in zip(*traj))
        clear, _ = path_clearance(self.car, xs, ys, th, self.cars, stride=5)
        row['min_clearance'] = clear
        final = traj[-1]
        body = self.car.corners(final)
        slot = next((s for s in self.lot if all(point_in_convex(c, s.corners()) for c in body)), None)
        row['slot'] = slot.id if slot else -1
        row['slot_empty'] = bool(slot and slot.id in self.scenario.empty_slot_ids)
        if slot:
            c, s = math.cos(slot.heading), math.sin(slot.heading)
            off = (self.car.front - self.car.rear) / 2
            cx, cy = final[0] + off * math.cos(final[2]), final[1] + off * math.sin(final[2])
            row.update(lateral_cm=100 * (-(cx - slot.entrance_x) * s + (cy - slot.entrance_y) * c),
                       depth_cm=100 * ((cx - slot.entrance_x) * c + (cy - slot.entrance_y) * s - slot.depth / 2),
                       yaw_deg=math.degrees(wrap(final[2] - slot.heading - math.pi)))
        row['success'] = bool(st and st.state == 'done' and row['slot_empty'] and clear > 0.0)
        self.rows.append(row)
        self.get_logger().info(' | '.join(f'{k} {v:.3g}' if isinstance(v, float) else f'{k} {v}'
                                          for k, v in row.items()))

    def save(self):
        os.makedirs(self.out, exist_ok=True)
        keys = sorted({k for r in self.rows for k in r})
        with open(os.path.join(self.out, f'{self.label}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)
        ok = [r for r in self.rows if r['success']]
        s = dict(runs=len(self.rows), success=len(ok),
                 outcomes={o: sum(r['manager_state'] == o for r in self.rows)
                           for o in sorted({r['manager_state'] for r in self.rows})},
                 contact=sum(r['min_clearance'] <= 0 for r in self.rows),
                 min_clearance_m=round(min(r['min_clearance'] for r in self.rows), 3))
        if ok:
            def stats(k, nd):
                a = np.abs([r[k] for r in ok])
                return dict(median=round(float(np.median(a)), nd), max=round(float(a.max()), nd))
            s.update(lateral_cm=stats('lateral_cm', 1), depth_cm=stats('depth_cm', 1), yaw_deg=stats('yaw_deg', 2),
                     time_s_median=round(float(np.median([r['time'] for r in ok])), 1),
                     plans_mean=round(float(np.mean([r['plans'] for r in ok])), 2),
                     max_track_cm=round(max(r['max_track_cm'] for r in ok), 1))
        with open(os.path.join(self.out, f'{self.label}_summary.json'), 'w') as f:
            json.dump(s, f, indent=1)
        print(json.dumps(s, indent=1), flush=True)


def main():
    rclpy.init()
    node = ParkEval()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and node.state != 'done':
            node.step()
    except KeyboardInterrupt:
        pass
    if node.rows:
        node.save()
        print(f'{len(node.rows)} runs in {time.monotonic() - t0:.0f} s -> {node.out}', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
