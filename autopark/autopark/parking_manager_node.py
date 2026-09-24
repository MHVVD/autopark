"""Parking manager: the state machine that parks the car, using only perception and odometry.

  search     drive along the aisle (centre line estimated from the tracked slots, see
             autopark.parking_manager.aisle_line) until a selectable-vacant slot has been
             passed by `pass_distance` m (the planner only drives in explored space)
  stopping   brake to a standstill and let the tracks settle
  planning   /parking/plan into the target slot (failure: exclude it and search on)
  executing  the controller follows the path (/parking/path_exec); at its end:
               replan = 'open':   the whole plan is executed once (plan once)
               replan = 'closed': only the first segment (up to the first cusp) is sent; at
                                  the cusp the car is standing, so it replans from there with
                                  the latest slot estimate. The last segment (the reverse into
                                  the slot) is corrected while driving: when the tracked slot
                                  moves, the remaining path is moved rigidly with the goal
                                  (small corrections only; a large jump stops and replans).
  done       parked (controller reached the end of the path)
  failed     no slot within `max_search` m, planning failed `max_plans` times, or the
             controller aborted

Publishes /parking/status (ParkingStatus) and, while searching / stopping / planning,
/cmd_ackermann; while executing the controller drives. Restarts on /scenario/reset.
"""
import math
from types import SimpleNamespace

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autopark_msgs.msg import ControlStatus, ParkingPath, ParkingSlotArray, ParkingStatus
from autopark_msgs.srv import PlanParking
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Int64

from autopark import parking_manager as pm
from autopark.odometry import yaw_from_quaternion
from autopark.planner_node import specs_from_tracks


def _sec(st):
    return st.sec + st.nanosec * 1e-9


class ParkingManager(Node):
    def __init__(self):
        super().__init__('parking_manager')
        for name, default in (('replan', 'closed'), ('autostart', True), ('search_speed', 1.0),
                              ('pass_distance', 3.0), ('max_search', 45.0), ('settle', 1.0),
                              ('max_plans', 6), ('correction_min', 0.01), ('correction_max', 0.25),
                              ('start_delay', 2.0)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = p('replan')
        if self.mode not in ('open', 'closed'):
            raise ValueError("replan must be 'open' or 'closed'")
        self.autostart, self.search_speed = p('autostart'), p('search_speed')
        self.pass_distance, self.max_search, self.settle = p('pass_distance'), p('max_search'), p('settle')
        self.max_plans = p('max_plans')
        self.corr_min, self.corr_max = p('correction_min'), p('correction_max')
        self.start_delay = p('start_delay')

        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd_ackermann', 10)
        self.path_pub = self.create_publisher(ParkingPath, '/parking/path_exec', 10)
        self.status_pub = self.create_publisher(ParkingStatus, '/parking/status', 10)
        self.plan_cli = self.create_client(PlanParking, '/parking/plan')
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(AckermannDriveStamped, '/vehicle/state', self.on_vehicle, 10)
        self.create_subscription(ParkingSlotArray, '/slots/tracked', self.on_tracks, 10)
        self.create_subscription(ControlStatus, '/parking/control_status', self.on_control, 10)
        self.create_subscription(Int64, '/scenario/reset', self.on_reset, 10)
        self.create_timer(0.5, self.publish_status)
        self.next_plan_id = 1
        self.reset_state()
        self.get_logger().info(f'parking manager: replan={self.mode}')

    # ------------------------------------------------------------------ state
    def reset_state(self):
        self.state = 'idle'
        self.t = None
        self.t_state = None
        self.pose = None
        self.speed = 0.0
        self.tracks = None
        self.start_pose = None
        self.target = None
        self.excluded = set()
        self.plans = 0
        self.corrections = 0
        self.plan_id = 0
        self.exec = None           # dict: plan msg, part sent, goal, last segment?
        self.future = None
        self.message = ''

    def on_reset(self, _msg):
        self.cancel_path()
        self.reset_state()
        self.get_logger().info('scenario reset')

    def enter(self, state, message=''):
        self.state, self.t_state, self.message = state, self.t, message
        self.get_logger().info(f'-> {state}' + (f': {message}' if message else ''))
        self.publish_status()

    # ------------------------------------------------------------------ inputs
    def on_vehicle(self, m):
        self.speed = m.drive.speed

    def on_tracks(self, m):
        self.tracks = m
        if self.state == 'executing' and self.exec and self.exec['last'] and self.mode == 'closed':
            self.correct_final_approach()

    def on_control(self, m):
        if self.state != 'executing' or m.plan_id != self.plan_id:
            return
        if m.state == 'done':
            if self.exec['last']:
                self.enter('done', f'parked in slot {self.target} after {self.plans} plan(s), '
                                   f'{self.corrections} correction(s); '
                                   f'max tracking error {100 * m.max_lateral_error:.1f} cm')
            else:
                self.enter('planning', 'cusp reached: replanning with the latest slot estimate')
                self.request_plan()
        elif m.state == 'aborted':
            self.enter('failed', f'controller aborted (lateral {m.lateral_error:.2f} m, heading '
                                 f'{math.degrees(m.heading_error):.0f} deg)')

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.t = _sec(m.header.stamp)
        if self.t_state is None:
            self.t_state = self.t
        self.step()

    # ------------------------------------------------------------------ behaviour
    def step(self):
        st = self.state
        if st == 'idle':
            if self.autostart and self.t - self.t_state > self.start_delay:
                self.start_pose = self.pose
                self.enter('search')
            return
        if st == 'search':
            self.search()
        elif st == 'stopping':
            self.drive(0.0, 0.0)
            if abs(self.speed) < 0.02 and self.t - self.t_state > self.settle:
                self.enter('planning', f'target slot {self.target}')
                self.request_plan()
        elif st == 'planning':
            self.drive(0.0, None)
            if self.future is not None and self.future.done():
                self.on_plan_result(self.future.result())
        elif st in ('done', 'failed'):
            pass            # the controller holds the car after 'done'; nothing else drives

    def search(self):
        specs = specs_from_tracks(self.tracks) if self.tracks else []
        line = pm.aisle_line(specs, self.pose)
        if line is None:
            steer = 0.0      # no slot seen yet: hold the heading
            u = (math.cos(self.pose[2]), math.sin(self.pose[2]))
        else:
            steer = pm.pure_pursuit_to_line(self.pose, line)
            u = line[1]
        target = pm.choose_target(specs, self.pose, u, self.pass_distance, self.excluded) if line else None
        if target is not None:
            self.target = target.id
            self.enter('stopping', f'slot {target.id} passed by {pm.passed_by(target, self.pose, u):.1f} m')
            self.drive(0.0, 0.0)
            return
        if math.hypot(self.pose[0] - self.start_pose[0], self.pose[1] - self.start_pose[1]) > self.max_search:
            self.drive(0.0, 0.0)
            self.enter('failed', f'no vacant slot found within {self.max_search:.0f} m')
            return
        self.drive(self.search_speed, steer)

    def drive(self, speed, steer):
        m = AckermannDriveStamped()
        m.drive.speed = float(speed)
        m.drive.steering_angle = float(steer if steer is not None else 0.0)
        if steer is None:          # standing still: keep the wheels where they are
            m.drive.steering_angle_velocity = 1e-6
        self.cmd_pub.publish(m)

    def request_plan(self):
        if self.plans >= self.max_plans:
            self.enter('failed', f'{self.plans} plans without parking')
            return
        req = PlanParking.Request()
        req.slot_id = int(self.target)
        self.future = self.plan_cli.call_async(req)

    def on_plan_result(self, res):
        self.future = None
        if not res.success:
            if self.exec is None:
                self.excluded.add(self.target)
                self.cancel_path()
                self.enter('search', f'slot {self.target} rejected: {res.message}')
                self.target = None
            else:
                self.enter('failed', f'replanning failed: {res.message}')
            return
        self.plans += 1
        self.plan_id = self.next_plan_id
        self.next_plan_id += 1
        path = res.path
        goal = (path.goal.x, path.goal.y, path.goal.theta)
        if self.mode == 'closed':
            part, last = pm.first_segment(path_fields(path))
        else:
            part, last = path_fields(path), True
        self.exec = dict(full=path, part=part, last=last, goal=goal, revision=0)
        self.send(part, path, revision=0)
        self.enter('executing', f'plan {self.plan_id}: {res.message}; executing '
                                + ('the whole path' if last and self.mode == 'open' else
                                   'the final segment' if last else 'up to the first cusp'))

    def send(self, part, full, revision):
        m = ParkingPath()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.plan_id, m.revision, m.slot_id = self.plan_id, revision, full.slot_id
        m.goal = full.goal
        m.poses = [Pose2D(x=float(x), y=float(y), theta=float(t)) for x, y, t in zip(part.x, part.y, part.yaw)]
        m.direction = [int(d) for d in part.direction]
        m.curvature = [float(k) for k in part.curvature]
        m.length = float(sum(math.hypot(part.x[i + 1] - part.x[i], part.y[i + 1] - part.y[i])
                             for i in range(len(part.x) - 1)))
        m.gear_changes = sum(1 for i in range(1, len(part.direction)) if part.direction[i] != part.direction[i - 1])
        m.planning_time = full.planning_time
        self.path_pub.publish(m)

    def cancel_path(self):
        m = ParkingPath()
        m.header.frame_id = 'odom'
        self.path_pub.publish(m)
        self.exec = None

    def correct_final_approach(self):
        """Closed loop on the final segment: move the remaining path with the slot estimate."""
        track = next((s for s in specs_from_tracks(self.tracks) if s.id == self.target), None)
        if track is None:
            return
        new_goal = pm.goal_for(track)
        g = self.exec['goal']
        corr = pm.rigid_correction(g, new_goal)
        shift = math.hypot(corr[0], corr[1]) + 2.0 * abs(corr[2])     # 1 deg ~ 3.5 cm at 2 m
        if shift < self.corr_min:
            return
        if shift > self.corr_max:
            self.get_logger().warn(f'slot estimate jumped by {shift:.2f} m: keeping the path')
            return
        self.exec['part'] = pm.apply_correction(self.exec['part'], g, corr)
        self.exec['goal'] = new_goal
        self.exec['revision'] += 1
        self.corrections += 1
        self.send(self.exec['part'], self.exec['full'], self.exec['revision'])

    def publish_status(self):
        s = ParkingStatus()
        s.header.stamp = self.get_clock().now().to_msg()
        s.state = self.state
        s.slot_id = int(self.target) if self.target is not None else -1
        s.plan_id = self.plan_id
        s.plans, s.corrections = self.plans, self.corrections
        s.message = self.message
        self.status_pub.publish(s)


def path_fields(msg):
    return SimpleNamespace(x=[p.x for p in msg.poses], y=[p.y for p in msg.poses], yaw=[p.theta for p in msg.poses],
                           direction=list(msg.direction), curvature=list(msg.curvature))


def main():
    rclpy.init()
    node = ParkingManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
