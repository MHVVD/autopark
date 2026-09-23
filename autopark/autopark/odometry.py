"""Ackermann dead reckoning. No ROS imports.

Reference point: rear-axle centre (base_link). Heading from the IMU yaw rate (default) or
from the bicycle model v * tan(steer) / L.
"""
import math


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_rate_from_steering(speed, steer, wheel_base):
    return speed * math.tan(steer) / wheel_base


class AckermannOdometry:
    def __init__(self):
        self.reset()

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.distance = 0.0   # total path length travelled, m

    def update(self, speed, yaw_rate, dt):
        """Integrate one step, exactly for constant speed and yaw rate over dt (circular arc);
        midpoint heading when the arc is (nearly) straight."""
        if dt <= 0:
            return
        dyaw = yaw_rate * dt
        if abs(dyaw) < 1e-6:
            yaw_mid = self.yaw + 0.5 * dyaw
            self.x += speed * math.cos(yaw_mid) * dt
            self.y += speed * math.sin(yaw_mid) * dt
        else:
            r = speed / yaw_rate
            self.x += r * (math.sin(self.yaw + dyaw) - math.sin(self.yaw))
            self.y += r * (math.cos(self.yaw) - math.cos(self.yaw + dyaw))
        self.yaw = wrap(self.yaw + dyaw)
        self.distance += abs(speed) * dt


def compose(base, rel):
    """SE(2) composition: pose rel (x, y, yaw) expressed in frame base -> world."""
    bx, by, byaw = base
    x, y, yaw = rel
    c, s = math.cos(byaw), math.sin(byaw)
    return bx + c * x - s * y, by + s * x + c * y, wrap(byaw + yaw)


def inverse(pose):
    """SE(2) inverse: compose(pose, inverse(pose)) == (0, 0, 0)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return -(c * x + s * y), -(-s * x + c * y), -yaw


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
