"""Collect a slot-detection dataset from arbitrary car poses: turned in the aisle, mid-manoeuvre
and inside slots (evaluation / data tool, uses ground truth).

collect_slots only drives along the aisle, so a detector trained on it has never seen the views
the car has while parking. For each seed this tool resets the scenario, then `poses_per_seed`
times teleports the ego car (/ground_truth/set_pose), waits `settle` s of simulated time for
the suspension to come to rest (also after the reset: the teleport keeps the car's height and
tilt), and saves the first BEV rendered after that. Output format as
collect_slots (<out_dir>/images/*.png + labels.jsonl), plus per image: kind (see below), the
requested pose and the car's height / roll / pitch at the capture (to check the settling).
A sample is dropped if the true pose moved more than 5 cm / 1 deg from the requested one.

Pose mix (sample_poses, pure function, reproducible from the seed):
  manoeuvre  points along a Hybrid A* reverse-in path into a random empty slot, planned on
             ground truth from a stop pose where the parking manager stops (plan_bench), with
             a small random offset (tracking error): the views while parking
  aisle      anywhere along the rows, anywhere across the aisle, any heading
  slot       in an empty slot, rear axle from the entrance to fully parked, facing out (75 %)
             or in, +-25 deg
Every pose keeps >= CLEARANCE m from the parked cars (exact rectangles) and stays in the lot.

Run with the stack up without the parking nodes (they would drive):
    ros2 launch autopark bringup.launch.py gui:=false mode:=fast park:=false planner:=false
    ros2 run autopark collect_poses --ros-args -p seed_start:=10200 -p seed_count:=200 \\
        -p out_dir:=$HOME/autopark_data/slots_v2/train
"""
import json
import math
import os
import time

import cv2
import numpy as np

from autopark import parking_goal as pg
from autopark.geometry_check import polygon_distance
from autopark.hybrid_astar import CarGeometry, HybridAStar, ObstacleMap, wrap
from autopark.plan_bench import sample_start, true_car_rects
from autopark_sim.lot import AISLE_WIDTH, ROW_X0, SLOT_DEPTH, slots as lot_slots
from autopark_sim.scenario import make_scenario

CLEARANCE = 0.15                              # m from the parked cars
LOT_X = (ROW_X0 - 4.0, -ROW_X0 + 4.0)         # car body limits (map frame)
LOT_Y = AISLE_WIDTH / 2 + SLOT_DEPTH
MIX = (('manoeuvre', 0.5), ('aisle', 0.25), ('slot', 0.25))
PATH_JITTER = (0.15, math.radians(3.0))       # m, rad: offset of manoeuvre poses from the path
POSE_TOL = (0.05, math.radians(1.0))          # max drift of the true pose from the request


def pose_ok(car, pose, cars, clearance=CLEARANCE):
    body = car.corners(pose)
    if not all(LOT_X[0] <= x <= LOT_X[1] and abs(y) <= LOT_Y for x, y in body):
        return False
    return all(polygon_distance(body, c) >= clearance for c in cars)


def _slot_specs(sc):
    return [pg.SlotSpec(s.id, s.entrance_x, s.entrance_y, s.heading, s.width, s.depth,
                        vacant=s.id in sc.empty_slot_ids) for s in lot_slots()]


def manoeuvre_poses(sc, rng, n, planner, car, cars):
    """n poses near ground-truth parking paths (a new path every <= 4 poses)."""
    lot = {s.id: s for s in lot_slots()}
    specs = _slot_specs(sc)
    out, attempts = [], 0
    while len(out) < n and attempts < 10:
        attempts += 1
        tid = int(rng.choice(sc.empty_slot_ids))
        start = sample_start(rng, lot[tid])
        goal = pg.goal_pose(specs[tid], car)
        if not pose_ok(car, start, cars):
            continue
        path = planner.plan(start, goal, ObstacleMap(*pg.window([start, goal]), pg.obstacles(specs, tid)))
        if path is None:
            continue
        for _ in range(min(4, n - len(out))):
            for _try in range(20):
                i = int(rng.integers(len(path.x)))
                a = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(0, PATH_JITTER[0])
                p = (path.x[i] + r * math.cos(a), path.y[i] + r * math.sin(a),
                     wrap(path.yaw[i] + rng.uniform(-PATH_JITTER[1], PATH_JITTER[1])))
                if pose_ok(car, p, cars):
                    out.append(p)
                    break
    return out


def aisle_pose(sc, rng, car, cars):
    for _ in range(200):
        p = (rng.uniform(ROW_X0 - 1.0, -ROW_X0 + 1.0), rng.uniform(-2.5, 2.5), rng.uniform(-math.pi, math.pi))
        if pose_ok(car, p, cars):
            return p
    return None


def slot_pose(sc, rng, car, cars):
    specs = _slot_specs(sc)
    for _ in range(200):
        s = specs[int(rng.choice(sc.empty_slot_ids))]
        gx, gy, gyaw = pg.goal_pose(s, car)
        back = rng.uniform(0.0, 4.5)             # m out of the parked pose, towards the aisle
        lat = rng.uniform(-0.3, 0.3)
        c, sn = math.cos(s.theta), math.sin(s.theta)
        yaw = gyaw + rng.uniform(-math.radians(25), math.radians(25))
        if rng.random() < 0.25:                  # nose in: same body position, rear axle at the
            yaw += math.pi                       # other end (front - rear nearer the entrance)
            back += car.front - car.rear
        p = (gx - back * c - lat * sn, gy - back * sn + lat * c, wrap(yaw))
        if pose_ok(car, p, cars):
            return p
    return None


def sample_poses(seed, n, car=None, planner=None):
    """[(kind, (x, y, yaw))] for scenario `seed`: the mix MIX, all collision-free."""
    car = car or CarGeometry()
    planner = planner or HybridAStar(car)
    sc = make_scenario(seed)
    cars = true_car_rects(sc)
    rng = np.random.default_rng(seed + 31337)
    counts = {k: int(round(f * n)) for k, f in MIX}
    counts['manoeuvre'] += n - sum(counts.values())
    out = [('manoeuvre', p) for p in manoeuvre_poses(sc, rng, counts['manoeuvre'], planner, car, cars)]
    for kind, fn in (('aisle', aisle_pose), ('slot', slot_pose)):
        for _ in range(counts[kind]):
            p = fn(sc, rng, car, cars)
            if p is not None:
                out.append((kind, p))
    while len(out) < n:                          # manoeuvre planning fell short: fill with aisle poses
        p = aisle_pose(sc, rng, car, cars)
        if p is None:
            break
        out.append(('aisle', p))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


# ---------------------------------------------------------------------------------- ROS node
def main():
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from autopark_msgs.msg import ParkingSlotArray
    from autopark_msgs.srv import ResetScenario, SetPose
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Pose2D
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    from autopark.odometry import yaw_from_quaternion

    def sec(st):
        return st.sec + st.nanosec * 1e-9

    class CollectPoses(Node):
        def __init__(self):
            super().__init__('collect_poses')
            self.declare_parameter('seed_start', 10200)
            self.declare_parameter('seed_count', 5)
            self.declare_parameter('poses_per_seed', 20)
            self.declare_parameter('settle', 2.0)
            self.declare_parameter('out_dir', os.path.expanduser('~/autopark_data/slots_v2/train'))
            p = self.get_parameter
            s0 = p('seed_start').value
            self.seeds = list(range(s0, s0 + p('seed_count').value))
            self.n_per_seed, self.settle = p('poses_per_seed').value, p('settle').value
            self.out_dir = p('out_dir').value
            os.makedirs(os.path.join(self.out_dir, 'images'), exist_ok=True)
            self.labels = open(os.path.join(self.out_dir, 'labels.jsonl'), 'a')
            self.bridge = CvBridge()
            self.car, self.planner = CarGeometry(), HybridAStar()
            self.cmd = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
            self.reset_cli = self.create_client(ResetScenario, '/ground_truth/reset')
            self.pose_cli = self.create_client(SetPose, '/ground_truth/set_pose')
            self.create_subscription(Odometry, '/ground_truth/pose', self.on_pose, 100)
            self.create_subscription(ParkingSlotArray, '/ground_truth/slots', self.on_slots, 10)
            self.create_subscription(Image, '/bev/image', self.on_bev,
                                     QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE))
            self.poses = {}            # ms -> (x, y, yaw, z, roll, pitch)
            self.t, self.slots, self.slots_t = 0.0, None, -1.0
            self.state, self.seed_i, self.saved, self.dropped = 'next_seed', -1, 0, 0
            self.SetPose, self.ResetScenario, self.Pose2D = SetPose, ResetScenario, Pose2D
            self.Ack = AckermannDriveStamped

        def on_pose(self, m):
            self.t = sec(m.header.stamp)
            q = m.pose.pose.orientation
            roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
            self.poses[int(round(self.t * 1000))] = (m.pose.pose.position.x, m.pose.pose.position.y,
                                                    yaw_from_quaternion(q.x, q.y, q.z, q.w),
                                                    m.pose.pose.position.z, roll, pitch)
            if len(self.poses) > 500:
                for k in sorted(self.poses)[:-300]:
                    del self.poses[k]

        def on_slots(self, m):
            self.slots = [dict(id=s.id, corners=[[c.x, c.y] for c in s.corners],
                               heading=s.entrance.theta, occupied=s.occupied) for s in m.slots]
            self.slots_t = sec(m.header.stamp)

        def on_bev(self, m):
            if self.state != 'wait_image':
                return
            t = sec(m.header.stamp)
            if t < self.t_pose + self.settle or self.slots_t < self.t_reset:
                return
            gt = self.poses.get(int(round(t * 1000)))
            if gt is None:
                return
            kind, req = self.queue[self.k]
            if (math.hypot(gt[0] - req[0], gt[1] - req[1]) > POSE_TOL[0]
                    or abs(wrap(gt[2] - req[2])) > POSE_TOL[1]):
                self.dropped += 1
                self.get_logger().warn(f'seed {self.seed} pose {self.k} ({kind}): the car moved to '
                                       f'({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.1f}) from '
                                       f'({req[0]:.2f}, {req[1]:.2f}, {math.degrees(req[2]):.1f}): dropped')
            else:
                name = f'{self.seed}_{self.k:03d}.png'
                cv2.imwrite(os.path.join(self.out_dir, 'images', name), self.bridge.imgmsg_to_cv2(m, 'bgr8'))
                self.labels.write(json.dumps(dict(
                    image=name, seed=self.seed, t=round(t, 3), kind=kind,
                    pose=[round(v, 5) for v in gt[:3]], speed=0.0,
                    requested=[round(v, 5) for v in req], z=round(gt[3], 4),
                    roll_deg=round(math.degrees(gt[4]), 3), pitch_deg=round(math.degrees(gt[5]), 3),
                    slots=self.slots)) + '\n')
                self.saved += 1
            self.k += 1
            self.state = 'next_pose'

        def stop_car(self):
            self.cmd.publish(self.Ack())

        def step(self):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.stop_car()
            if self.state == 'next_seed':
                if not (self.reset_cli.service_is_ready() and self.pose_cli.service_is_ready()):
                    return
                self.labels.flush()
                self.seed_i += 1
                if self.seed_i >= len(self.seeds):
                    self.state = 'done'
                    return
                self.seed = self.seeds[self.seed_i]
                self.queue = sample_poses(self.seed, self.n_per_seed, self.car, self.planner)
                self.k = 0
                req = self.ResetScenario.Request()
                req.seed = self.seed
                self.future, self.state = self.reset_cli.call_async(req), 'resetting'
            elif self.state == 'resetting' and self.future.done():
                self.t_reset = self.t
                self.state = 'reset_settle'
            elif self.state == 'reset_settle' and self.t > self.t_reset + self.settle:
                self.state = 'next_pose'   # the teleport keeps height and tilt: start from rest
            elif self.state == 'next_pose':
                if self.k >= len(self.queue):
                    self.get_logger().info(f'seed {self.seed}: {len(self.queue)} poses | total saved '
                                           f'{self.saved}, dropped {self.dropped}')
                    self.state = 'next_seed'
                    return
                x, y, yaw = self.queue[self.k][1]
                req = self.SetPose.Request()
                req.pose = self.Pose2D(x=float(x), y=float(y), theta=float(yaw))
                self.future, self.state = self.pose_cli.call_async(req), 'teleporting'
            elif self.state == 'teleporting' and self.future.done():
                self.t_pose = self.t
                self.state = 'wait_image'

    rclpy.init()
    node = CollectPoses()
    t0 = time.monotonic()
    try:
        while rclpy.ok() and node.state != 'done':
            node.step()
    except KeyboardInterrupt:
        pass
    node.labels.close()
    node.get_logger().info(f'done: {node.saved} images ({node.dropped} dropped) in '
                           f'{time.monotonic() - t0:.0f} s -> {node.out_dir}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
