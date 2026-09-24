"""Reeds-Shepp curves: shortest paths for a car that drives forward and backward with a minimum
turning radius. No ROS imports.

Formulas follow Reeds & Shepp (1990), in the numbering used by OMPL's ReedsSheppStateSpace
(sections 8.1-8.11), with the time-flip / reflection / backwards symmetries. Everything is in
units of the turning radius: a path is a list of (kind, length) segments, kind in 'LRS' and
length signed (positive = forward). For an L or R segment the length is also the heading change
in radians.
"""
import math

import numpy as np

PI = math.pi
TWO_PI = 2 * math.pi
ZERO = 10 * 2.220446049250313e-16


def mod2pi(x):
    v = math.fmod(x, TWO_PI)
    if v < -PI:
        v += TWO_PI
    elif v > PI:
        v -= TWO_PI
    return v


def _polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u, v, xi, eta, phi):
    delta = mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3
    tau = mod2pi(t1 + PI) if t2 < 0 else mod2pi(t1)
    omega = mod2pi(tau - u + v - phi)
    return tau, omega


# ---------------------------------------------------------------- base formulas (8.1 - 8.11)
def _LpSpLp(x, y, phi):
    u, t = _polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if t >= -ZERO:
        v = mod2pi(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def _LpSpRp(x, y, phi):
    u1, t1 = _polar(x + math.sin(phi), y - 1 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4:
        u = math.sqrt(u1 - 4)
        theta = math.atan2(2, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def _LpRmL(x, y, phi):
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4:
        u = -2 * math.asin(0.25 * u1)
        t = mod2pi(theta + 0.5 * u + PI)
        v = mod2pi(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def _LpRupLumRm(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    rho = 0.25 * (2 + math.hypot(xi, eta))
    if rho <= 1:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRumLumRp(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    rho = (20 - xi * xi - eta * eta) / 16
    if 0 <= rho <= 1:
        u = -math.acos(rho)
        if u >= -0.5 * PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _LpRmSmLm(x, y, phi):
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2:
        r = math.sqrt(rho * rho - 4)
        u = 2 - r
        t = mod2pi(theta + math.atan2(r, -2))
        v = mod2pi(phi - 0.5 * PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRmSmRm(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2:
        t = theta
        u = 2 - rho
        v = mod2pi(t + 0.5 * PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRmSLmRp(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2:
        u = 4 - math.sqrt(rho * rho - 4)
        if u <= ZERO:
            t = mod2pi(math.atan2((4 - u) * xi - 2 * eta, -2 * xi + (u - 4) * eta))
            v = mod2pi(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


# ---------------------------------------------------------------- families with symmetries
# Each base formula is applied to the goal as given, time-flipped (-x, y, -phi), reflected
# (x, -y, -phi) and both (-x, -y, phi). Time flip negates all lengths; reflection swaps L and R.
def _swap(word):
    return word.translate(str.maketrans('LR', 'RL'))


def _four(fn, x, y, phi, word, make):
    out = []
    for sx, sy, flip, refl in ((1, 1, False, False), (-1, 1, True, False),
                               (1, -1, False, True), (-1, -1, True, True)):
        r = fn(sx * x, sy * y, phi * (-1 if flip != refl else 1))
        if r is None:
            continue
        lengths = make(*r)
        if flip:
            lengths = [-v for v in lengths]
        out.append(list(zip(_swap(word) if refl else word, lengths)))
    return out


def _backwards(x, y, phi):
    return x * math.cos(phi) + y * math.sin(phi), x * math.sin(phi) - y * math.cos(phi)


def _csc(x, y, phi):
    return (_four(_LpSpLp, x, y, phi, 'LSL', lambda t, u, v: [t, u, v])
            + _four(_LpSpRp, x, y, phi, 'LSR', lambda t, u, v: [t, u, v]))


def _ccc(x, y, phi):
    xb, yb = _backwards(x, y, phi)
    return (_four(_LpRmL, x, y, phi, 'LRL', lambda t, u, v: [t, u, v])
            + _four(_LpRmL, xb, yb, phi, 'LRL', lambda t, u, v: [v, u, t]))


def _cccc(x, y, phi):
    return (_four(_LpRupLumRm, x, y, phi, 'LRLR', lambda t, u, v: [t, u, -u, v])
            + _four(_LpRumLumRp, x, y, phi, 'LRLR', lambda t, u, v: [t, u, u, v]))


def _ccsc(x, y, phi):
    h = 0.5 * PI
    xb, yb = _backwards(x, y, phi)
    return (_four(_LpRmSmLm, x, y, phi, 'LRSL', lambda t, u, v: [t, -h, u, v])
            + _four(_LpRmSmRm, x, y, phi, 'LRSR', lambda t, u, v: [t, -h, u, v])
            + _four(_LpRmSmLm, xb, yb, phi, 'LSRL', lambda t, u, v: [v, u, -h, t])
            + _four(_LpRmSmRm, xb, yb, phi, 'RSRL', lambda t, u, v: [v, u, -h, t]))


def _ccscc(x, y, phi):
    h = 0.5 * PI
    return _four(_LpRmSLmRp, x, y, phi, 'LRSLR', lambda t, u, v: [t, -h, u, -h, v])


def path_length(path):
    return sum(abs(v) for _, v in path)


def all_paths(x, y, phi):
    """All Reeds-Shepp candidate words from the origin (heading 0) to (x, y, phi), in units of the
    turning radius, as lists of (kind, signed length), zero-length segments removed."""
    out = []
    for p in _csc(x, y, phi) + _ccc(x, y, phi) + _cccc(x, y, phi) + _ccsc(x, y, phi) + _ccscc(x, y, phi):
        p = [(k, v) for k, v in p if abs(v) > 1e-10]
        if p:
            out.append(p)
    return out


def to_local(start, goal, radius):
    """Goal pose in the start frame, scaled by 1/radius."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return (c * dx + s * dy) / radius, (-s * dx + c * dy) / radius, mod2pi(goal[2] - start[2])


def paths(start, goal, radius):
    """All candidate paths between two poses (x, y, yaw) for a turning radius (m), sorted by
    length. Each is a list of (kind, signed length in m)."""
    x, y, phi = to_local(start, goal, radius)
    if abs(x) < 1e-12 and abs(y) < 1e-12 and abs(phi) < 1e-12:
        return [[]]
    cands = [[(k, v * radius) for k, v in p] for p in all_paths(x, y, phi)]
    return sorted(cands, key=path_length)


def shortest(start, goal, radius):
    c = paths(start, goal, radius)
    return c[0] if c else None


def distance(start, goal, radius):
    p = shortest(start, goal, radius)
    return path_length(p) if p is not None else math.inf


def sample(start, path, radius, step):
    """Sample a path (segments in m) from `start` every <= `step` m.

    Returns numpy arrays xs, ys, yaws, dirs (+1 fwd / -1 rev), curvatures (1/m, + = left), one
    entry per sample. dirs[i] / curvatures[i] describe the motion that reaches sample i (the
    start sample copies the first motion). Each segment's end point is sampled exactly."""
    x, y, th = start
    xs, ys, ths, dirs, ks = [np.array([x])], [np.array([y])], [np.array([th])], [], []
    for kind, length in path:
        k = {'L': 1.0 / radius, 'R': -1.0 / radius, 'S': 0.0}[kind]
        n = max(1, int(math.ceil(abs(length) / step - 1e-9)))
        s = length * np.arange(1, n + 1) / n
        if k == 0.0:
            sx, sy, sth = x + s * math.cos(th), y + s * math.sin(th), np.full(n, th)
        else:
            sth = th + k * s
            sx = x + (np.sin(sth) - math.sin(th)) / k
            sy = y - (np.cos(sth) - math.cos(th)) / k
        xs.append(sx)
        ys.append(sy)
        ths.append(sth)
        dirs.append(np.full(n, 1 if length >= 0 else -1, np.int8))
        ks.append(np.full(n, k))
        x, y, th = float(sx[-1]), float(sy[-1]), float(sth[-1])
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    ths = (np.concatenate(ths) + math.pi) % (2 * math.pi) - math.pi
    if dirs:
        dirs, ks = np.concatenate(dirs), np.concatenate(ks)
        dirs, ks = np.concatenate([dirs[:1], dirs]), np.concatenate([ks[:1], ks])
    else:
        dirs, ks = np.ones(1, np.int8), np.zeros(1)
    return xs, ys, ths, dirs, ks


def arc(x, y, th, s, k):
    """Pose after driving signed distance s along curvature k from (x, y, th)."""
    if abs(k) < 1e-12:
        return x + s * math.cos(th), y + s * math.sin(th), th
    th1 = th + k * s
    return (x + (math.sin(th1) - math.sin(th)) / k, y - (math.cos(th1) - math.cos(th)) / k,
            mod2pi(th1))
