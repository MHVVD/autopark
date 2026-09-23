"""Vehicle constants and small pure functions used by the Webots vehicle plugin. No ROS/Webots.

Conventions (ROS): speed in m/s, positive forward; steering angle in rad, positive LEFT,
expressed as the equivalent single front wheel of a bicycle model (Ackermann centre angle).
Webots' Driver uses km/h and a steering angle that is positive RIGHT.
"""
import math

WHEEL_BASE = 2.8          # m, ToyotaPrius PROTO
TRACK = 1.628             # m
WHEEL_RADIUS = 0.317      # m, ToyotaPriusWheel.tireRadius

MAX_STEER = 0.6           # rad, commanded limit (Car PROTO default would allow 1.0)
MAX_SPEED = 3.0           # m/s, parking speeds only
MAX_ACCEL = 1.0           # m/s^2, default actuator limit
MAX_STEER_RATE = 0.8      # rad/s, default actuator limit
CMD_TIMEOUT = 0.5         # s without a command -> stop


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rate_limit(current, target, max_rate, dt):
    """Move current towards target by at most max_rate * dt."""
    step = max_rate * dt
    return current + clamp(target - current, -step, step)


def to_webots(speed, steering):
    """ROS (m/s, rad left+) -> Webots Driver (km/h, rad right+)."""
    return speed * 3.6, -steering


def ackermann_center_angle(left, right):
    """Bicycle-equivalent steering angle from the two front wheel angles (same sign convention).

    For Ackermann geometry cot(center) = (cot(left) + cot(right)) / 2.
    """
    if abs(left) < 1e-6 or abs(right) < 1e-6 or (left > 0) != (right > 0):
        return (left + right) / 2  # straight ahead, or inconsistent -> plain average
    cot = (1 / math.tan(left) + 1 / math.tan(right)) / 2
    return math.atan(1 / cot)


def encoder_speed(d_left, d_right, dt, radius=WHEEL_RADIUS):
    """Vehicle speed at the rear axle from rear-wheel encoder increments (rad) over dt (s)."""
    if dt <= 0:
        return 0.0
    return (d_left + d_right) / 2 * radius / dt
