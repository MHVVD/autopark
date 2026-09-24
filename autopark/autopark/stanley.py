"""Stanley path tracking, forward and reverse, with stops at cusps. No ROS imports.

Poses are rear-axle (base_link) poses; the path is a planner Path-like object (arrays x, y, yaw,
direction, curvature; direction[i] / curvature[i] belong to the motion that reaches sample i).

Per segment of constant direction d (+1 forward, -1 reverse) the controller works in a
"virtual forward" frame: heading phi = yaw (+ pi when reversing), speed u = |v| > 0; e is the
lateral offset of the reference point from its path (+ = left in that frame), he the heading
error (path minus car).
  forward: classic Stanley at the front axle (the steered wheels lead):
           delta = atan(L k_path) + he + atan2(-k e, k_soft + u).
  reverse: the rear axle leads and the steered wheels trail (in the virtual frame the car is a
           bicycle steered at its back, steering -delta). Stanley's heading + cross-track
           structure at the rear axle with gains per metre travelled:
           tan(delta') / L = k_virtual + k_he * he - k_e * e, delta = -delta'.
           For small errors e'' + k_he e' + k_e e = 0 per metre (k_he = 2, k_e = 1: critically
           damped, settles in about 2 m at any speed). The same structure with speed-domain
           gains (Stanley's atan2 term at the rear axle) gave e'' + (u/L) e' + (k u/L) e = 0,
           which at reversing speed is underdamped and needs about 5 m to settle.

Speed: v_max per direction, reduced to stop exactly at each segment end and to creep speed at
each steering jump in the path (Reeds-Shepp / Hybrid A* paths switch between full-lock arcs;
the steering actuator needs about 1.4 s for that), with a constant-deceleration profile, and
further, down to a stop, while the steering lags its command. At each segment end the car
stops; the next segment starts after the steering has turned to that segment's initial command
(at standstill), so gear changes do not cost tracking error.

States: 'align' (standstill, steering to the command), 'track', 'stop' (braking at a segment
end), 'done' (end of the path, stopped), 'aborted' (tracking error too large), 'idle'.
"""
import math
from dataclasses import dataclass

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class StanleyConfig:
    wheel_base: float = 2.8
    k: float = 1.2               # forward cross-track gain (1/s in the small-angle sense)
    k_soft: float = 0.3          # m/s, keeps the gain finite at low speed
    k_he_rev: float = 2.0        # reverse heading gain (1/m)
    k_e_rev: float = 1.0         # reverse cross-track gain (1/m^2)
    v_forward: float = 1.0       # m/s
    v_reverse: float = 0.6       # m/s
    v_creep: float = 0.12        # m/s, final approach speed
    decel: float = 0.35          # m/s^2 used for the stopping profile (actuator allows 1.0)
    stop_tol: float = 0.01       # m, segment end reached (along the path)
    standstill: float = 0.02     # m/s
    steer_tol: float = 0.05      # rad, steering aligned at standstill
    align_timeout: float = 3.0   # s
    steer_slow: float = 0.15     # rad of steering lag at which the car stops (speed ~ 1 - lag / this)
    steer_change: float = 0.1    # rad, a path steering jump this large is approached at creep speed
    max_steer: float = 0.6
    abort_lateral: float = 0.6   # m
    abort_heading: float = math.radians(35)


class Stanley:
    def __init__(self, config=None):
        self.cfg = config or StanleyConfig()
        self.state = 'idle'
        self.path = None

    # ------------------------------------------------------------------ path handling
    def set_path(self, path, keep_progress=False):
        """New path. keep_progress: the path is a small correction of the current one (same
        samples, slightly moved); stay in the current segment and state."""
        x, y, yaw = (np.asarray(a, float) for a in (path.x, path.y, path.yaw))
        d = np.asarray(path.direction, int)
        k = np.asarray(path.curvature, float)
        cut = (np.flatnonzero(d[1:] != d[:-1]) + 1).tolist()
        bounds = [0, *cut, len(x)]
        segs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            i0, i1 = max(a - 1, 0), b - 1                 # a segment starts at the cusp sample
            idx = np.arange(i0, i1 + 1)
            sd = int(d[a])                                # direction of the samples a..b-1
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x[idx]), np.diff(y[idx])))])
            # steering changes along the path (the actuator needs time for them): slow down
            # to creep speed before each one, as before a segment end
            dsteer = np.abs(np.diff(np.arctan(self.cfg.wheel_base * k[idx])))
            slow = s[1:][dsteer > self.cfg.steer_change]
            segs.append(dict(x=x[idx], y=y[idx], yaw=yaw[idx], k=k[idx], d=sd, s=s, slow=slow,
                             xf=x[idx] + self.cfg.wheel_base * np.cos(yaw[idx]),
                             yf=y[idx] + self.cfg.wheel_base * np.sin(yaw[idx]),
                             # the front axle moves along yaw + steering angle, not yaw
                             tf=yaw[idx] + np.arctan(self.cfg.wheel_base * k[idx])))
        if keep_progress and self.path is not None and len(segs) == len(self.segs) and self.state != 'idle':
            self.segs = segs
            self.path = path
            self.search_all = True        # samples moved: find the nearest one again
            return
        self.path, self.segs = path, segs
        self.seg_i, self.idx = 0, 0
        self.search_all = False
        self.state = 'align'
        self.t_state = None
        self.max_lateral = 0.0
        self.max_heading = 0.0

    def cancel(self):
        self.state = 'idle'
        self.path = None

    # ------------------------------------------------------------------ control
    def update(self, pose, speed, steer, t):
        """pose: rear-axle (x, y, yaw); speed (m/s, signed) and steer (rad) measured; t (s).
        Returns (speed command, steering command)."""
        cfg = self.cfg
        if self.state in ('idle', 'done', 'aborted'):
            return 0.0, steer if self.state != 'idle' else 0.0
        if self.t_state is None:
            self.t_state = t
        seg = self.segs[self.seg_i]
        d = seg['d']
        e, he, i, d_end = self._errors(seg, pose)
        u = abs(speed)
        k_path = seg['k'][min(i + 1, len(seg['k']) - 1)]
        if d > 0:
            delta = math.atan(cfg.wheel_base * k_path) + he + math.atan2(-cfg.k * e, cfg.k_soft + u)
        else:
            delta = -math.atan(cfg.wheel_base * (-k_path + cfg.k_he_rev * he - cfg.k_e_rev * e))
        delta = max(-cfg.max_steer, min(cfg.max_steer, delta))
        self.last = dict(lateral=e, heading=he, index=i, remaining=d_end, segment=self.seg_i,
                         segments=len(self.segs), direction=d)

        if self.state == 'align':
            if u < cfg.standstill and (abs(steer - delta) < cfg.steer_tol or t - self.t_state > cfg.align_timeout):
                self._enter('track', t)
            else:
                return 0.0, delta

        if self.state == 'track':
            self.max_lateral = max(self.max_lateral, abs(e))
            self.max_heading = max(self.max_heading, abs(he))
            if abs(e) > cfg.abort_lateral or abs(he) > cfg.abort_heading:
                self._enter('aborted', t)
                return 0.0, steer
            if d_end <= cfg.stop_tol:
                self._enter('stop', t)
            else:
                vmax = cfg.v_forward if d > 0 else cfg.v_reverse
                s_now = seg['s'][i]
                ahead = seg['slow'][seg['slow'] > s_now - 0.05]
                d_slow = float(ahead[0] - s_now) if len(ahead) else math.inf
                d_brake = min(d_end - cfg.stop_tol, max(d_slow, 0.0))
                v = min(vmax, math.sqrt(cfg.v_creep ** 2 + 2 * cfg.decel * max(d_brake, 0.0)))
                # the steering is rate limited: slow down (to a stop) while it lags its command,
                # so the car does not run off the path at curvature changes
                lag = abs(steer - delta)
                if lag < 0.5 * cfg.steer_slow:
                    v = max(v, cfg.v_creep)
                v *= max(0.0, 1.0 - lag / cfg.steer_slow)
                return d * v, delta

        if self.state == 'stop':
            if u < cfg.standstill:
                if self.seg_i + 1 < len(self.segs):
                    self.seg_i += 1
                    self.idx = 0
                    self._enter('align', t)
                else:
                    self._enter('done', t)
            return 0.0, delta
        return 0.0, delta

    def _enter(self, state, t):
        self.state, self.t_state = state, t

    def _errors(self, seg, pose):
        """Lateral / heading error of the reference point, nearest index, distance of the rear
        axle to the segment end along the direction of motion."""
        x, y, yaw = pose
        d = seg['d']
        if d > 0:
            rx, ry = x + self.cfg.wheel_base * math.cos(yaw), y + self.cfg.wheel_base * math.sin(yaw)
            px, py = seg['xf'], seg['yf']
        else:
            rx, ry = x, y
            px, py = seg['x'], seg['y']
        # nearest sample, searched forward from the last one (progress is monotone)
        lo = 0 if self.search_all else self.idx
        hi = len(px) if self.search_all else min(len(px), lo + 40)
        j = lo + int(np.argmin((px[lo:hi] - rx) ** 2 + (py[lo:hi] - ry) ** 2))
        self.idx, self.search_all = j, False
        phi_p = seg['yaw'][j] + (0.0 if d > 0 else math.pi)
        phi = yaw + (0.0 if d > 0 else math.pi)
        # lateral offset perpendicular to the reference point's direction of travel
        tan_p = seg['tf'][min(j + 1, len(px) - 1)] if d > 0 else phi_p
        e = -math.sin(tan_p) * (rx - px[j]) + math.cos(tan_p) * (ry - py[j])
        he = wrap(phi_p - phi)
        # remaining distance: along the path, and near the end exactly as the rear axle's
        # distance to the end point along the final direction of motion
        remaining = seg['s'][-1] - seg['s'][j]
        if remaining > 0.5:
            return e, he, j, remaining
        xe, ye, the = seg['x'][-1], seg['y'][-1], seg['yaw'][-1]
        d_end = d * (math.cos(the) * (xe - x) + math.sin(the) * (ye - y))
        return e, he, j, d_end
