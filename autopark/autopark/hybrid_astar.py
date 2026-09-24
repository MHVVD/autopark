"""Hybrid A* with Reeds-Shepp analytic expansion for a car-like robot. No ROS imports.

Poses are (x, y, yaw) of the REAR AXLE (base_link) in a planar frame (the stack uses odom).

Search: nodes are continuous poses, pruned on a (x, y, yaw) grid. Each expansion drives short
arcs forward and backward at a few steering angles. Cost = travelled length (reverse weighted)
+ penalties for gear changes and steering. Heuristic: the larger of two lower bounds of the
path length: the obstacle-aware 2D distance from the goal (Dijkstra on a grid, rear axle kept
at least `axle_clearance` from obstacles) and the obstacle-free Reeds-Shepp distance. Near the
goal every node (further away every few nodes) tries an analytic Reeds-Shepp shot; the first
collision-free one ends the search.

Collision checking: points every `spacing` m on the car rectangle's outline must have
clearance >= margin + spacing/2 in a Euclidean distance field of the obstacles (conservative
for its grid resolution, see ObstacleMap), so the outline grown by `margin` touches no
obstacle. Points on the car's centre line must have positive clearance, which also rules out
an obstacle inside the car unless it is narrower than the car's half width and fits between
outline and axis; the obstacles here (slot rectangles, map border) are all wider than 2 m.
"""
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from autopark import reeds_shepp as rs


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class CarGeometry:
    """Rectangle around the rear axle. Defaults: ToyotaPrius PROTO (autopark_sim.rig)."""
    wheel_base: float = 2.8
    rear: float = 0.85          # rear bumper behind the rear axle, m
    front: float = 3.635        # front bumper ahead of the rear axle, m
    half_width: float = 0.88
    max_steer: float = 0.55     # rad used for planning (the actuator allows 0.6)

    @property
    def min_radius(self):
        return self.wheel_base / math.tan(self.max_steer)

    def outline_points(self, spacing):
        """Points (x, y) in the car frame every <= spacing m on the outline, and on the axis."""
        x0, x1, w = -self.rear, self.front, self.half_width
        nx = int(math.ceil((x1 - x0) / spacing))
        ny = int(math.ceil(2 * w / spacing))
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(-w, w, ny + 1)
        outline = np.concatenate([np.stack([xs, np.full_like(xs, -w)], 1), np.stack([xs, np.full_like(xs, w)], 1),
                                  np.stack([np.full_like(ys, x0), ys], 1), np.stack([np.full_like(ys, x1), ys], 1)])
        axis = np.stack([xs, np.zeros_like(xs)], 1)
        return outline, axis

    def corners(self, pose, grow=0.0):
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        pts = [(-self.rear - grow, -self.half_width - grow), (self.front + grow, -self.half_width - grow),
               (self.front + grow, self.half_width + grow), (-self.rear - grow, self.half_width + grow)]
        return [(x + c * px - s * py, y + s * px + c * py) for px, py in pts]


class ObstacleMap:
    """Distance-to-obstacle field over a rectangular region; outside it is an obstacle.

    Obstacles are convex polygons. A cell is marked when its centre lies within h = res/sqrt(2)
    of a polygon (half-plane test, a superset). For a query point q the returned clearance is
    edt(nearest cell to q) - 2h, a lower bound of the true distance from q to the polygons: the
    true nearest obstacle point p has a marked cell within h, and q is within h of its cell.

    `known` (optional): convex polygons whose union is the explored area. Everything outside it
    is an obstacle too; a cell counts as known only if its centre is at least h inside one of
    them, so the same lower-bound argument holds for the unknown area.
    """

    def __init__(self, x_min, y_min, x_max, y_max, polygons, res=0.025, known=None):
        self.x_min, self.y_min, self.res = x_min, y_min, res
        self.nx = int(math.ceil((x_max - x_min) / res))
        self.ny = int(math.ceil((y_max - y_min) / res))
        occ = np.zeros((self.ny, self.nx), bool)
        h = res / math.sqrt(2)
        cx = x_min + (np.arange(self.nx) + 0.5) * res
        cy = y_min + (np.arange(self.ny) + 0.5) * res
        for poly in polygons:
            _mark(occ, cx, cy, poly, h)
        if known is not None:
            seen = np.zeros_like(occ)
            for poly in known:
                _mark(seen, cx, cy, poly, -h)
            occ |= ~seen
        # the border of the map counts as an obstacle
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        self.occupied = occ
        self.edt = distance_transform_edt(~occ).astype(np.float32) * res
        self.slack = 2 * h

    def clearance(self, xs, ys):
        """Lower bound of the distance (m) from each point to the nearest obstacle (0 outside)."""
        i = np.floor((np.asarray(xs) - self.x_min) / self.res).astype(np.int64)
        j = np.floor((np.asarray(ys) - self.y_min) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        out = np.zeros(np.shape(i), np.float32)
        out[ok] = self.edt[j[ok], i[ok]]
        return np.maximum(out - self.slack, 0.0)


def _mark(grid, cx, cy, poly, offset):
    """Set cells whose centre lies inside the convex polygon grown by `offset` (m; negative
    shrinks). Growing uses half-planes, a superset of the true offset polygon."""
    poly = np.asarray(poly, float)
    if _signed_area(poly) < 0:
        poly = poly[::-1]
    lo, hi = poly.min(0) - max(offset, 0.0), poly.max(0) + max(offset, 0.0)
    i0, i1 = np.searchsorted(cx, [lo[0], hi[0]])
    j0, j1 = np.searchsorted(cy, [lo[1], hi[1]])
    if i1 <= i0 or j1 <= j0:
        return
    X, Y = np.meshgrid(cx[i0:i1], cy[j0:j1])
    inside = np.ones(X.shape, bool)
    for k in range(len(poly)):
        (ax, ay), (bx, by) = poly[k], poly[(k + 1) % len(poly)]
        ex, ey = bx - ax, by - ay
        n = math.hypot(ex, ey)
        if n < 1e-12:
            continue
        # outward normal of a counter-clockwise polygon is (ey, -ex)
        inside &= ((X - ax) * ey - (Y - ay) * ex) / n <= offset
    grid[j0:j1, i0:i1] |= inside


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass
class PlannerConfig:
    xy_res: float = 0.25                 # m, node pruning grid
    yaw_res: float = math.radians(5.0)
    step: float = 0.6                    # m, arc length of one expansion (> cell diagonal)
    sample: float = 0.1                  # m, collision-check spacing
    n_steer: int = 7                     # steering angles in [-max, max]
    margin: float = 0.10                 # m, safety margin around the car rectangle
    reverse_cost: float = 1.5            # multiplier on reverse length
    gear_cost: float = 3.0               # m-equivalent per change of direction
    steer_cost: float = 0.3              # per m, times |steer| / max_steer
    steer_change_cost: float = 0.5       # per change, times |delta steer| / max_steer
    heuristic_weight: float = 2.0        # > 1: weighted A* (trades path cost for speed)
    axle_clearance: float = 0.8          # m, for the 2D heuristic (< min axle-to-body distance)
    h_res: float = 0.25                  # m, 2D heuristic grid
    shot_range: float = 20.0             # m, try Reeds-Shepp shots below this RS distance to goal
    shot_always: float = 8.0             # m, below this every expansion tries a shot
    shot_every: int = 5                  # otherwise every n-th expansion does
    min_segment: float = 0.5             # m, shortest allowed piece between two gear changes
    max_expansions: int = 40000
    timeout: float = 10.0                # s


@dataclass
class Path:
    """Sampled path: rear-axle poses, driving direction (+1/-1) and curvature for each sample."""
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    direction: np.ndarray
    curvature: np.ndarray
    cost: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def length(self):
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    @property
    def n_gear_changes(self):
        return int(np.sum(self.direction[1:] != self.direction[:-1]))

    def segments(self):
        """(i0, i1) inclusive index ranges of constant driving direction. direction[i] is the
        direction of the motion that reaches sample i, so a segment starts at the previous
        segment's last sample (the cusp)."""
        cut = (np.flatnonzero(self.direction[1:] != self.direction[:-1]) + 1).tolist()
        bounds = [0, *cut, len(self.x)]
        return [(max(bounds[k] - 1, 0), bounds[k + 1] - 1) for k in range(len(bounds) - 1)]


class _Node:
    __slots__ = ('pose', 'g', 'parent', 'direction', 'steer', 'xs', 'ys', 'ths')

    def __init__(self, pose, g, parent, direction, steer, xs, ys, ths):
        self.pose, self.g, self.parent = pose, g, parent
        self.direction, self.steer = direction, steer
        self.xs, self.ys, self.ths = xs, ys, ths


class HybridAStar:
    def __init__(self, car=None, config=None):
        self.car = car or CarGeometry()
        self.cfg = config or PlannerConfig()
        self.spacing = 0.05
        # quick test: circles along the axis covering the whole rectangle; if they are all
        # clear the pose is free and the exact outline test is skipped
        n = 6
        hl = (self.car.front + self.car.rear) / (2 * n)
        self.circ_x = -self.car.rear + hl * (2 * np.arange(n) + 1)
        self.circ_need = np.float32(math.hypot(hl, self.car.half_width) + self.cfg.margin)
        outline, axis = self.car.outline_points(self.spacing)
        self.pts = np.concatenate([outline, axis])
        self.need = np.concatenate([np.full(len(outline), self.cfg.margin + self.spacing / 2),
                                    np.full(len(axis), 1e-6)]).astype(np.float32)

    # ---------------------------------------------------------------- collision
    def free(self, omap, xs, ys, ths):
        """True where the car at each pose (arrays) is collision free."""
        xs, ys, ths = (np.atleast_1d(np.asarray(a, float)) for a in (xs, ys, ths))
        c, s = np.cos(ths)[:, None], np.sin(ths)[:, None]
        cx = xs[:, None] + c * self.circ_x[None, :]
        cy = ys[:, None] + s * self.circ_x[None, :]
        ok = np.all(omap.clearance(cx, cy) >= self.circ_need, axis=1)
        rest = np.flatnonzero(~ok)
        if len(rest):
            c, s = c[rest], s[rest]
            px = xs[rest, None] + c * self.pts[None, :, 0] - s * self.pts[None, :, 1]
            py = ys[rest, None] + s * self.pts[None, :, 0] + c * self.pts[None, :, 1]
            ok[rest] = np.all(omap.clearance(px, py) >= self.need[None, :], axis=1)
        return ok

    def path_free(self, omap, xs, ys, ths, chunk=30):
        """All poses free? Checked in chunks, stopping at the first collision."""
        for i in range(0, len(xs), chunk):
            if not self.free(omap, xs[i:i + chunk], ys[i:i + chunk], ths[i:i + chunk]).all():
                return False
        return True

    def min_clearance(self, omap, xs, ys, ths):
        """Smallest clearance of the car outline (lower bound, m) over the poses."""
        xs, ys, ths = (np.atleast_1d(np.asarray(a, float)) for a in (xs, ys, ths))
        outline, _ = self.car.outline_points(self.spacing)
        c, s = np.cos(ths)[:, None], np.sin(ths)[:, None]
        px = xs[:, None] + c * outline[None, :, 0] - s * outline[None, :, 1]
        py = ys[:, None] + s * outline[None, :, 0] + c * outline[None, :, 1]
        return float(omap.clearance(px, py).min()) - self.spacing / 2

    # ---------------------------------------------------------------- heuristic
    def heuristic_grid(self, omap, goal):
        """2D Dijkstra distance (m) from the goal over cells where the rear axle fits."""
        cfg = self.cfg
        k = max(1, int(round(cfg.h_res / omap.res)))
        ny, nx = omap.ny // k, omap.nx // k
        # clearance at the coarse cell centres
        xs = omap.x_min + (np.arange(nx) + 0.5) * k * omap.res
        ys = omap.y_min + (np.arange(ny) + 0.5) * k * omap.res
        X, Y = np.meshgrid(xs, ys)
        free = omap.clearance(X, Y) >= cfg.axle_clearance
        res = k * omap.res
        gi = int((goal[0] - omap.x_min) / res)
        gj = int((goal[1] - omap.y_min) / res)
        dist = np.full((ny, nx), np.inf)
        if not (0 <= gi < nx and 0 <= gj < ny):
            return dist, res
        free[gj, gi] = True
        dist[gj, gi] = 0.0
        pq = [(0.0, gj, gi)]
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        while pq:
            d, j, i = heapq.heappop(pq)
            if d > dist[j, i]:
                continue
            for dj, di, w in moves:
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and free[jj, ii]:
                    nd = d + w * res
                    if nd < dist[jj, ii]:
                        dist[jj, ii] = nd
                        heapq.heappush(pq, (nd, jj, ii))
        # 8-connected grid paths are up to 8 % longer than straight lines; scale down so the
        # heuristic stays a lower bound of the true distance.
        return dist / 1.0824, res

    # ---------------------------------------------------------------- search
    def plan(self, start, goal, omap):
        """Plan from start to goal pose. Returns a Path or None; `self.last_info` has stats."""
        cfg, car = self.cfg, self.car
        t0 = time.monotonic()
        info = dict(expansions=0, shots=0, reason='')
        self.last_info = info
        if not self.free(omap, [start[0]], [start[1]], [start[2]])[0]:
            info['reason'] = 'start in collision'
            return None
        if not self.free(omap, [goal[0]], [goal[1]], [goal[2]])[0]:
            info['reason'] = 'goal in collision'
            return None

        hgrid, hres = self.heuristic_grid(omap, goal)
        info['t_heuristic'] = time.monotonic() - t0

        def h2d(x, y):
            i = int((x - omap.x_min) / hres)
            j = int((y - omap.y_min) / hres)
            if 0 <= i < hgrid.shape[1] and 0 <= j < hgrid.shape[0]:
                return hgrid[j, i]
            return math.inf

        def h(pose):
            # both are lower bounds of the path length, which is a lower bound of the cost
            return max(h2d(pose[0], pose[1]), rs.distance(pose, goal, radius))

        if not math.isfinite(h2d(start[0], start[1])):
            info['reason'] = 'goal not reachable in 2D'
            return None

        steers = np.linspace(-car.max_steer, car.max_steer, cfg.n_steer)
        n_sub = max(1, int(math.ceil(cfg.step / cfg.sample)))
        s_sub = np.arange(1, n_sub + 1) * (cfg.step / n_sub)
        # local samples of all primitives, shape (n_prim, n_sub)
        p_dir, p_steer, p_dx, p_dy, p_dth = [], [], [], [], []
        for d in (1, -1):
            for st in steers:
                k = math.tan(st) / car.wheel_base
                s = d * s_sub
                if abs(k) < 1e-9:
                    dx, dy, dth = s, np.zeros_like(s), np.zeros_like(s)
                else:
                    dth = k * s
                    dx, dy = np.sin(dth) / k, (1 - np.cos(dth)) / k
                p_dir.append(d)
                p_steer.append(float(st))
                p_dx.append(dx)
                p_dy.append(dy)
                p_dth.append(dth)
        p_dx, p_dy, p_dth = np.array(p_dx), np.array(p_dy), np.array(p_dth)
        n_prim = len(p_dir)

        def key(p):
            return (int(math.floor(p[0] / cfg.xy_res)), int(math.floor(p[1] / cfg.xy_res)),
                    int(math.floor(wrap(p[2]) / cfg.yaw_res)) % int(round(2 * math.pi / cfg.yaw_res)))

        radius = car.min_radius
        root = _Node(tuple(start), 0.0, None, 0, 0.0, [start[0]], [start[1]], [start[2]])
        closed = {}
        best_g = {key(start): 0.0}
        counter = 0
        open_ = [(cfg.heuristic_weight * h(start), counter, root)]

        while open_:
            if info['expansions'] >= cfg.max_expansions or time.monotonic() - t0 > cfg.timeout:
                info['reason'] = 'search limit'
                break
            _, _, node = heapq.heappop(open_)
            k = key(node.pose)
            if k in closed:
                continue
            closed[k] = node
            info['expansions'] += 1

            # analytic Reeds-Shepp shot: every expansion near the goal, every few further away
            d_rs = rs.distance(node.pose, goal, radius)
            if d_rs <= cfg.shot_range and (d_rs <= cfg.shot_always or info['expansions'] % cfg.shot_every == 1):
                shot = self._shot(node, goal, omap, radius)
                info['shots'] += 1
                if shot is not None:
                    path = self._extract(node, shot)
                    info['reason'] = 'ok'
                    info['time'] = time.monotonic() - t0
                    path.info = info
                    return path

            x, y, th = node.pose
            c, s = math.cos(th), math.sin(th)
            X = x + c * p_dx - s * p_dy
            Y = y + s * p_dx + c * p_dy
            TH = th + p_dth
            ok = self.free(omap, X.ravel(), Y.ravel(), TH.ravel()).reshape(n_prim, -1).all(axis=1)
            may_reverse = node.parent is None or self._run_length(node) >= cfg.min_segment - 1e-6
            for i in np.flatnonzero(ok):
                d, st = p_dir[i], p_steer[i]
                if node.direction == -d and not may_reverse:
                    continue
                end = (float(X[i, -1]), float(Y[i, -1]), wrap(float(TH[i, -1])))
                ke = key(end)
                if ke in closed:
                    continue
                g = node.g + self._cost(node, d, st, cfg.step)
                if g >= best_g.get(ke, math.inf):
                    continue
                he = h(end)
                if not math.isfinite(he):
                    continue
                best_g[ke] = g
                counter += 1
                child = _Node(end, g, node, d, st, X[i].tolist(), Y[i].tolist(), TH[i].tolist())
                heapq.heappush(open_, (g + cfg.heuristic_weight * he, counter, child))
        if not info['reason']:
            info['reason'] = 'no path'
        info['time'] = time.monotonic() - t0
        return None

    def _cost(self, node, d, st, length):
        cfg, car = self.cfg, self.car
        c = length * (cfg.reverse_cost if d < 0 else 1.0)
        c += cfg.steer_cost * length * abs(st) / car.max_steer
        if node.parent is not None:
            if node.direction != d:
                c += cfg.gear_cost
            c += cfg.steer_change_cost * abs(st - node.steer) / car.max_steer
        return c

    def _run_length(self, node):
        """Distance driven in the node's current direction since the last gear change."""
        d, n, total = node.direction, node, 0.0
        while n.parent is not None and n.direction == d:
            total += self.cfg.step
            n = n.parent
        return total

    def _path_cost(self, segs, prev_dir, prev_steer, first=True):
        cfg, car = self.cfg, self.car
        steer = math.atan(car.wheel_base / car.min_radius)
        c, d0, s0 = 0.0, prev_dir, prev_steer
        for kind, length in segs:
            d = 1 if length >= 0 else -1
            st = {'L': steer, 'R': -steer, 'S': 0.0}[kind]
            c += abs(length) * (cfg.reverse_cost if d < 0 else 1.0)
            c += cfg.steer_cost * abs(length) * abs(st) / car.max_steer
            if d0 != 0 and d != d0:
                c += cfg.gear_cost
            c += cfg.steer_change_cost * abs(st - s0) / car.max_steer
            d0, s0 = d, st
        return c

    def _shot(self, node, goal, omap, radius):
        cfg = self.cfg
        cands = rs.paths(node.pose, goal, radius)
        scored = []
        run = self._run_length(node) if node.parent is not None else math.inf
        for p in cands:
            if not self._segments_ok(p, node.direction, run):
                continue
            scored.append((self._path_cost(p, node.direction, node.steer), p))
        scored.sort(key=lambda t: t[0])
        for cost, p in scored[:6]:
            xs, ys, ths, dirs, ks = rs.sample(node.pose, p, radius, cfg.sample)
            if self.path_free(omap, xs, ys, ths):
                return cost, p, (xs, ys, ths, dirs, ks)
        return None

    def _segments_ok(self, p, prev_dir, prev_run):
        """Reject paths with a piece between two gear changes shorter than min_segment (also
        counting the node's current run when the first segment reverses it)."""
        m = self.cfg.min_segment - 1e-6
        runs, cur_d, cur_len = [], None, 0.0
        for _, length in p:
            d = 1 if length >= 0 else -1
            if d != cur_d and cur_d is not None:
                runs.append((cur_d, cur_len))
                cur_len = 0.0
            cur_d = d
            cur_len += abs(length)
        if cur_d is not None:
            runs.append((cur_d, cur_len))
        if not runs:
            return True
        if prev_dir != 0:
            if runs[0][0] == prev_dir:
                runs[0] = (prev_dir, runs[0][1] + prev_run)   # continues the node's run
            elif prev_run < m:
                return False
        return all(length >= m for _, length in runs)

    def _extract(self, node, shot):
        cost, p, (sx, sy, sth, sdir, sk) = shot
        car = self.car
        chain = []
        n = node
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()
        xs, ys, ths, dirs, ks = [chain[0].xs[0]], [chain[0].ys[0]], [chain[0].ths[0]], [], []
        for n in chain[1:]:
            k = math.tan(n.steer) / car.wheel_base
            xs += n.xs
            ys += n.ys
            ths += n.ths
            dirs += [n.direction] * len(n.xs)
            ks += [k] * len(n.xs)
        xs += sx[1:].tolist()
        ys += sy[1:].tolist()
        ths += sth[1:].tolist()
        dirs += sdir[1:].tolist()
        ks += sk[1:].tolist()
        # the start sample takes the direction and curvature of the first motion
        dirs.insert(0, dirs[0] if dirs else 1)
        ks.insert(0, ks[0] if ks else 0.0)
        return Path(np.array(xs), np.array(ys), np.array([wrap(t) for t in ths]),
                    np.array(dirs, np.int8), np.array(ks), cost=node.g + cost)
