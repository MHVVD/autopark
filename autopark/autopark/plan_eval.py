"""Evaluate the planner on the live stack against ground truth (evaluation tool, uses GT).

For each seed: reset, drive along the aisle centre, and at each stop (map x in `stops`) stand
still for `settle` s, then call /parking/plan for every selectable-vacant tracked slot. The
planner sees only perception (tracked slots, odometry). Each returned path is moved into the
map frame with the true pose (T_map_odom = gt * odom^-1 while standing) and checked with
exact geometry against the TRUE parked cars (same deterministic scenario) and the true slot:
  success, planning time, length, gear changes, min clearance to the true cars (> 0: no
  collision), final car inside the true slot lines, final lateral / depth / heading error of
  the car in the true slot (this is the effect of perception + odometry errors on the goal).

    ros2 run autopark plan_eval --ros-args -p use_sim_time:=true -p seeds:=[0,1,2]
"""
import csv
import json
import math
import os
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autopark_msgs.msg import ParkingSlotArray
from autopark_msgs.srv import PlanParking, ResetScenario
from nav_msgs.msg import Odometry
from rclpy.node import Node

from autopark import parking_goal as pg
from autopark.collect_slots import pure_pursuit
from autopark.geometry_check import path_clearance, point_in_convex
from autopark.hybrid_astar import CarGeometry
from autopark.odometry import compose, inverse, wrap, yaw_from_quaternion
from autopark.plan_bench import true_car_rects
from autopark_sim.lot import slots as lot_slots
from autopark_sim.scenario import make_scenario

SETTLE_START = 1.5


def _sec(st):
    return st.sec + st.nanosec * 1e-9


class PlanEval(Node):
    def __init__(self):
        super().__init__('plan_eval')
        self.declare_parameter('seeds', [0, 1, 2])
        self.declare_parameter('stops', [-1.0, 11.0])
        self.declare_parameter('speed', 1.2)
        self.declare_parameter('settle', 1.5)
        self.declare_parameter('label', 'default')
        self.declare_parameter('out', os.path.expanduser('~/autopark_results/plan_eval'))
        p = self.get_parameter
        self.seeds = list(p('seeds').value)
        self.stops = list(p('stops').value)
        self.speed, self.settle = p('speed').value, p('settle').value
        self.label, self.out = p('label').value, p('out').value
        self.cmd = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
        self.reset_cli = self.create_client(ResetScenario, '/ground_truth/reset')
        self.plan_cli = self.create_client(PlanParking, '/parking/plan')
        self.create_subscription(Odometry, '/ground_truth/pose', self.on_gt, 200)
        self.create_subscription(Odometry, '/odom', self.on_odom, 200)
        self.create_subscription(ParkingSlotArray, '/slots/tracked', self.on_tracks, 10)
        self.car = CarGeometry()
        self.lot = lot_slots()
        self.state, self.i, self.t = 'idle', -1, 0.0
        self.gt_pose = self.odom_pose = None
        self.tracks = None
        self.rows = []

    def on_gt(self, m):
        q = m.pose.pose.orientation
        self.gt_pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.t = _sec(m.header.stamp)
        if self.state == 'run':
            self.drive()

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.odom_pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def on_tracks(self, m):
        self.tracks = m

    # --------------------------------------------------------------- driving + requests
    def drive(self):
        msg = AckermannDriveStamped()
        if self.t - self.reset_t < SETTLE_START:
            self.cmd.publish(msg)
            return
        if self.phase == 'drive':
            if self.gt_pose[0] >= self.stops[self.stop_i]:
                self.phase, self.stop_t = 'stopping', self.t
            else:
                msg.drive.speed = self.speed
                msg.drive.steering_angle = max(-0.5, min(0.5, pure_pursuit(self.gt_pose, lambda v: 0.0)))
        elif self.phase == 'stopping' and self.t - self.stop_t > self.settle:
            self.phase = 'plan'
            self.queue = None
        self.cmd.publish(msg)

    def plan_step(self):
        """While standing at a stop: request one plan at a time."""
        if self.queue is None:
            vac = [s for s in self.tracks.slots if pg.selectable(s.vacancy, s.confidence)] if self.tracks else []
            self.queue = [s.id for s in vac]
            self.T = compose(self.gt_pose, inverse(self.odom_pose))    # odom -> map, car standing
            self.track_snapshot = {s.id: s for s in self.tracks.slots} if self.tracks else {}
            self.future = None
            self.get_logger().info(f'seed {self.seeds[self.i]} stop x={self.stops[self.stop_i]:.0f}: '
                                   f'{len(self.queue)} selectable vacant slots {self.queue}')
        if self.future is None:
            if not self.queue:
                self.stop_i += 1
                self.phase = 'drive' if self.stop_i < len(self.stops) else 'done'
                if self.phase == 'done':
                    self.state = 'next'
                return
            if not self.plan_cli.service_is_ready():
                return
            self.req_id = self.queue.pop(0)
            req = PlanParking.Request()
            req.slot_id = self.req_id
            self.future = self.plan_cli.call_async(req)
        elif self.future.done():
            self.record(self.future.result())
            self.future = None

    def record(self, res):
        seed = self.seeds[self.i]
        trk = self.track_snapshot[self.req_id]
        ex, ey, _ = compose(self.T, (trk.entrance.x, trk.entrance.y, trk.entrance.theta))
        true_slot = min(self.lot, key=lambda s: math.hypot(s.entrance_x - ex, s.entrance_y - ey))
        row = dict(label=self.label, seed=seed, stop=self.stops[self.stop_i], track=self.req_id,
                   slot=true_slot.id, slot_is_empty=true_slot.id in self.scenario.empty_slot_ids,
                   car_past_slot=self.gt_pose[0] - true_slot.entrance_x,
                   ok=res.success, message=res.message)
        if res.success:
            p = res.path
            pts = [compose(self.T, (q.x, q.y, q.theta)) for q in p.poses]
            xs, ys, th = (np.array(v) for v in zip(*pts))
            clear, _ = path_clearance(self.car, xs, ys, th, self.cars)
            body = self.car.corners((xs[-1], ys[-1], th[-1]))
            inside = all(point_in_convex(c, true_slot.corners()) for c in body)
            c, s = math.cos(true_slot.heading), math.sin(true_slot.heading)
            cx = xs[-1] + (self.car.front - self.car.rear) / 2 * math.cos(th[-1])
            cy = ys[-1] + (self.car.front - self.car.rear) / 2 * math.sin(th[-1])
            row.update(time=p.planning_time, length=p.length, gears=p.gear_changes, clearance=clear,
                       inside=inside,
                       lateral_cm=100 * (-(cx - true_slot.entrance_x) * s + (cy - true_slot.entrance_y) * c),
                       depth_cm=100 * ((cx - true_slot.entrance_x) * c + (cy - true_slot.entrance_y) * s
                                       - true_slot.depth / 2),
                       yaw_deg=math.degrees(wrap(th[-1] - true_slot.heading - math.pi)))
            self.get_logger().info(f"  track {self.req_id} -> slot {true_slot.id}: {res.message}; "
                                   f"clearance {clear:.2f} m, inside {inside}, lateral {row['lateral_cm']:+.1f} cm, "
                                   f"depth {row['depth_cm']:+.1f} cm, yaw {row['yaw_deg']:+.2f} deg")
        else:
            self.get_logger().warn(f'  track {self.req_id} -> slot {true_slot.id}: {res.message}')
        self.rows.append(row)

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
            self.reset_t, self.tracks = self.t, None
            self.phase, self.stop_i = 'drive', 0
            self.state = 'run'
        elif self.state == 'run' and self.phase == 'plan':
            self.plan_step()

    def save(self):
        os.makedirs(self.out, exist_ok=True)
        keys = sorted({k for r in self.rows for k in r})
        with open(os.path.join(self.out, f'{self.label}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)
        ok = [r for r in self.rows if r['ok']]
        s = dict(requests=len(self.rows), solved=len(ok),
                 requested_slot_really_empty=sum(r['slot_is_empty'] for r in self.rows))
        if ok:
            s.update(collision_free=sum(r['clearance'] > 0 for r in ok),
                     min_clearance_m=round(min(r['clearance'] for r in ok), 3),
                     inside_slot=sum(r['inside'] for r in ok),
                     time_s_median=round(float(np.median([r['time'] for r in ok])), 3),
                     time_s_max=round(max(r['time'] for r in ok), 3),
                     length_m_median=round(float(np.median([r['length'] for r in ok])), 2),
                     gears_max=max(r['gears'] for r in ok),
                     lateral_cm_abs_max=round(max(abs(r['lateral_cm']) for r in ok), 1),
                     depth_cm_abs_max=round(max(abs(r['depth_cm']) for r in ok), 1),
                     yaw_deg_abs_max=round(max(abs(r['yaw_deg']) for r in ok), 2))
        with open(os.path.join(self.out, f'{self.label}_summary.json'), 'w') as f:
            json.dump(s, f, indent=1)
        print(json.dumps(s, indent=1), flush=True)


def main():
    rclpy.init()
    node = PlanEval()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and node.state != 'done':
            node.step()
    except KeyboardInterrupt:
        pass
    if node.rows:
        node.save()
        print(f'{len(node.rows)} plan requests in {time.monotonic() - t0:.0f} s -> {node.out}', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
