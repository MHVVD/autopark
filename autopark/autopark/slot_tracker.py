"""Multi-slot tracker: one Kalman filter per parking slot in the odom frame. No ROS imports.

State per track: entrance centre and heading into the slot, x = [x, y, theta] (odom frame).
Slots are static, so prediction only inflates the covariance with odometry drift:
    Q = diag(q_pos * d, q_pos * d, q_yaw * d)^2-ish (variance grows linearly with the
    distance d the car travelled since the last step: random-walk drift).
Measurement: a detection's entrance pose transformed into odom with the odometry pose at the
detection stamp. Its covariance comes from the detector's measured error vs range
(meas_std) plus an optional extra std (e.g. when noise is injected for experiments).

Association: Mahalanobis distance d^2 = r^T S^-1 r (S = P + R, heading residual wrapped),
gated at chi2(3 dof, 99.9 %) and solved as a linear assignment.

Track life: a new detection starts a *tentative* track; it is *confirmed* after `confirm_hits`
hits. A track only accumulates misses while its entrance is predicted to be *in view* of the
detector (`in_view` callback, which includes the detector's valid range of view angles), so a
slot that leaves the field of view or is seen from an untrained angle during a manoeuvre is
remembered. Tentative tracks die after `max_misses_tentative` misses in view, confirmed ones
after `max_misses_confirmed`.

Physical constraint: two slot entrances cannot be closer than about one slot width, so an
unassociated detection within `min_separation` of an existing track is treated as an outlier
(no new track, and not a miss for that track either: the slot was seen), and tracks that
drift within `min_separation` of each other are merged (the one with more hits survives).

Occupancy: binary Bayes filter on "vacant" in log-odds. Each associated detection with a
vacancy probability p adds w(r) * logit(clip(p)), where the weight w falls with range because
far-range vacancy is unreliable (milestone 3: 28 of 30 false-vacant calls at 7-10 m).
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from autopark import slot_codec as sc
from autopark.bev import BevGrid

CHI2_3_999 = 16.27      # 99.9 % gate, 3 dof


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def meas_std(r):
    """Detector error model (1 sigma) vs entrance range r (m) from the car centre, fitted
    conservatively to the milestone 3 test set: (position m, heading rad)."""
    return 0.005 + 0.0025 * r, math.radians(0.3 + 0.04 * r)


_BEV_RES = BevGrid().res
CAR_CENTRE_X = 1.35     # m ahead of the rear axle (base_link)
VIEW_MARGIN = 1.5       # m inside the BEV border (both entrance corners visible)
VIEW_RANGE = 9.0        # m from the car centre
# The detector was trained on views from the aisle (car heading within ~20 deg of the aisle,
# i.e. slots nearly perpendicular to the car). Outside that it finds nothing or is biased, so
# such views neither count as observations nor as misses.
VIEW_YAW_TOL = math.radians(20.0)


def heading_valid(rel_heading, tol=VIEW_YAW_TOL):
    """Whether a slot heading relative to the car (rad) is in the detector's trained range:
    within `tol` of perpendicular (either side)."""
    return abs(abs((rel_heading + math.pi) % (2 * math.pi) - math.pi) - math.pi / 2) <= tol


def in_view(pose, x, y, margin=VIEW_MARGIN, view_range=VIEW_RANGE, theta=None, yaw_tol=VIEW_YAW_TOL):
    """Whether a slot entrance at (x, y) (same frame as `pose`, the rear-axle pose) is inside
    the detector's reliable view: within `view_range` of the car centre, at least `margin`
    inside the BEV input crop, and (if the slot heading `theta` is given) seen from within the
    detector's trained range of relative headings."""
    if theta is not None and not heading_valid(theta - pose[2], yaw_tol):
        return False
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = x - pose[0], y - pose[1]
    gx, gy = c * dx + s * dy, -s * dx + c * dy
    if math.hypot(gx - CAR_CENTRE_X, gy) > view_range:
        return False
    col, row = sc.ground_to_input(gx, gy)
    m = margin / _BEV_RES
    return m <= col < sc.INPUT - m and m <= row < sc.INPUT - m


def vacancy_weight(r):
    """Trust in a detection's vacancy vs range: 1 up to 5 m, falling linearly to 0.1 at 9 m."""
    return float(np.clip(1.0 - 0.9 * (r - 5.0) / 4.0, 0.1, 1.0))


@dataclass
class Detection:
    x: float           # odom frame
    y: float
    theta: float
    width: float
    vacancy: float     # -1 if unknown
    range: float       # m from the car centre at detection time
    R: np.ndarray = None


@dataclass
class Track:
    id: int
    x: np.ndarray                 # [x, y, theta]
    P: np.ndarray
    width: float
    hits: int = 1
    misses: int = 0               # consecutive misses while in view
    confirmed: bool = False
    vac_logodds: float = 0.0
    n_vac_close: int = 0          # vacancy observations within close_range
    last_seen: float = 0.0
    history: list = field(default_factory=list)

    @property
    def vacancy(self):
        return 1.0 / (1.0 + math.exp(-self.vac_logodds))


class SlotTracker:
    def __init__(self, extra_pos_std=0.0, extra_yaw_std=0.0, q_pos=0.01, q_yaw=math.radians(0.3),
                 confirm_hits=3, max_misses_tentative=3, max_misses_confirmed=10,
                 vac_clip=0.95, logodds_limit=8.0, close_range=7.0, init_inflate=2.0,
                 min_separation=1.2):
        self.extra_pos_std = extra_pos_std
        self.extra_yaw_std = extra_yaw_std
        self.q_pos = q_pos            # m of drift std per sqrt(m) travelled
        self.q_yaw = q_yaw            # rad per sqrt(m)
        self.confirm_hits = confirm_hits
        self.max_misses_tentative = max_misses_tentative
        self.max_misses_confirmed = max_misses_confirmed
        self.vac_clip = vac_clip
        self.logodds_limit = logodds_limit
        self.close_range = close_range
        self.init_inflate = init_inflate
        self.min_separation = min_separation
        self.tracks = []
        self.next_id = 0

    def reset(self):
        self.tracks = []
        self.next_id = 0

    # ---------------------------------------------------------------- model
    def R(self, rng):
        sp, sy = meas_std(rng)
        sp = math.hypot(sp, self.extra_pos_std)
        sy = math.hypot(sy, self.extra_yaw_std)
        return np.diag([sp * sp, sp * sp, sy * sy])

    def predict(self, distance):
        """Inflate all covariances for `distance` m travelled (odometry drift)."""
        if distance <= 0:
            return
        Q = np.diag([self.q_pos ** 2 * distance, self.q_pos ** 2 * distance, self.q_yaw ** 2 * distance])
        for t in self.tracks:
            t.P = t.P + Q

    @staticmethod
    def residual(det, track):
        return np.array([det.x - track.x[0], det.y - track.x[1], wrap(det.theta - track.x[2])])

    # ---------------------------------------------------------------- update
    def update(self, detections, stamp, in_view):
        """detections: list[Detection] (odom frame). in_view(x, y, theta) -> bool: whether a slot
        with entrance at odom (x, y) and heading theta is inside the detector's view right now."""
        for d in detections:
            if d.R is None:
                d.R = self.R(d.range)

        n, m = len(self.tracks), len(detections)
        cost = np.full((n, m), 1e6)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                r = self.residual(d, t)
                d2 = float(r @ np.linalg.solve(t.P + d.R, r))
                if d2 < CHI2_3_999:
                    cost[i, j] = d2
        pairs = []
        if n and m:
            rows, cols = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < CHI2_3_999]

        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}
        for i, j in pairs:
            self._correct(self.tracks[i], detections[j], stamp)

        unmatched = [d for j, d in enumerate(detections) if j not in matched_d]
        survivors = []
        for i, t in enumerate(self.tracks):
            seen_nearby = any(math.hypot(t.x[0] - d.x, t.x[1] - d.y) < self.min_separation for d in unmatched)
            if i not in matched_t and not seen_nearby and in_view(t.x[0], t.x[1], t.x[2]):
                t.misses += 1
            limit = self.max_misses_confirmed if t.confirmed else self.max_misses_tentative
            if t.misses < limit:
                survivors.append(t)
        self.tracks = survivors

        self._merge_close_tracks()
        for j, d in enumerate(detections):
            if j not in matched_d and not self._near_track(d.x, d.y):
                self._spawn(d, stamp)
        return pairs

    def _near_track(self, x, y):
        return any(math.hypot(t.x[0] - x, t.x[1] - y) < self.min_separation for t in self.tracks)

    def _merge_close_tracks(self):
        keep = []
        for t in sorted(self.tracks, key=lambda t: (t.confirmed, t.hits), reverse=True):
            if not any(math.hypot(t.x[0] - k.x[0], t.x[1] - k.x[1]) < self.min_separation for k in keep):
                keep.append(t)
        self.tracks = sorted(keep, key=lambda t: t.id)

    def _correct(self, t, d, stamp):
        r = self.residual(d, t)
        S = t.P + d.R
        K = t.P @ np.linalg.inv(S)
        t.x = t.x + K @ r
        t.x[2] = wrap(t.x[2])
        I_K = np.eye(3) - K
        t.P = I_K @ t.P @ I_K.T + K @ d.R @ K.T      # Joseph form (stays symmetric PSD)
        t.width += 0.2 * (d.width - t.width)
        t.hits += 1
        t.misses = 0
        t.last_seen = stamp
        if t.hits >= self.confirm_hits:
            t.confirmed = True
        self._update_vacancy(t, d)

    def _update_vacancy(self, t, d):
        if d.vacancy < 0:
            return
        p = min(max(d.vacancy, 1 - self.vac_clip), self.vac_clip)
        t.vac_logodds += vacancy_weight(d.range) * math.log(p / (1 - p))
        t.vac_logodds = max(-self.logodds_limit, min(self.logodds_limit, t.vac_logodds))
        if d.range <= self.close_range:
            t.n_vac_close += 1

    def _spawn(self, d, stamp):
        t = Track(self.next_id, np.array([d.x, d.y, d.theta], float), d.R * self.init_inflate,
                  d.width, last_seen=stamp)
        self.next_id += 1
        self._update_vacancy(t, d)
        if self.confirm_hits <= 1:
            t.confirmed = True
        self.tracks.append(t)

    def confirmed(self):
        return [t for t in self.tracks if t.confirmed]
