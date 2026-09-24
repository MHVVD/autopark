"""Kinematic car for testing controllers without the simulator. No ROS imports.

Bicycle model at the rear axle with the vehicle plugin's actuator limits: speed and steering
move towards their commands at most MAX_ACCEL and MAX_STEER_RATE; optional command delay.
"""
import math
from collections import deque

from autopark_sim.vehicle_model import MAX_ACCEL, MAX_STEER, MAX_STEER_RATE, WHEEL_BASE


class KinematicCar:
    def __init__(self, pose, wheel_base=WHEEL_BASE, delay_steps=0):
        self.x, self.y, self.yaw = pose
        self.v = 0.0
        self.steer = 0.0
        self.L = wheel_base
        self.queue = deque([(0.0, 0.0)] * delay_steps)

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def step(self, v_cmd, steer_cmd, dt):
        self.queue.append((v_cmd, max(-MAX_STEER, min(MAX_STEER, steer_cmd))))
        v_cmd, steer_cmd = self.queue.popleft()
        dv = max(-MAX_ACCEL * dt, min(MAX_ACCEL * dt, v_cmd - self.v))
        ds = max(-MAX_STEER_RATE * dt, min(MAX_STEER_RATE * dt, steer_cmd - self.steer))
        self.v += dv
        self.steer += ds
        # exact arc for this step
        k = math.tan(self.steer) / self.L
        s = self.v * dt
        if abs(k) < 1e-9:
            self.x += s * math.cos(self.yaw)
            self.y += s * math.sin(self.yaw)
        else:
            th = self.yaw + k * s
            self.x += (math.sin(th) - math.sin(self.yaw)) / k
            self.y -= (math.cos(th) - math.cos(self.yaw)) / k
            self.yaw = (th + math.pi) % (2 * math.pi) - math.pi
