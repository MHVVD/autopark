"""Offline planner benchmark on ground-truth scenarios (no simulator).

For each scenario seed and each empty slot, a start pose is sampled in the aisle where the car
would stop after passing the slot (heading along the aisle, up to +-10 deg, lateral within
+-1 m of the aisle centre, 3 m before to 8 m past the slot). The planner sees the slots the
way perception reports them (entrance, heading, vacant flag), with optional noise on the slot
poses. Every returned path is then checked against the TRUE parked-car rectangles with exact
polygon geometry (autopark.geometry_check), and its end pose against the true slot lines:
  collision-free (min clearance > 0), kinematically feasible (curvature, spacing, heading
  consistent with the direction of motion), final car inside the slot lines.

    ros2 run autopark plan_bench --seeds 0-99 --out ~/autopark_results/plan_bench
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

from autopark import parking_goal as pg
from autopark.geometry_check import path_clearance, point_in_convex, rect
from autopark.hybrid_astar import CarGeometry, HybridAStar, ObstacleMap, PlannerConfig, wrap
from autopark_sim.lot import slots as lot_slots
from autopark_sim.scenario import CAR_MODELS, make_scenario


def parse_seeds(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def true_car_rects(scenario):
    out = []
    for p in scenario.parked:
        m = CAR_MODELS[p.model]
        cx, cy = p.x + m['offset'] * math.cos(p.yaw), p.y + m['offset'] * math.sin(p.yaw)
        out.append(rect(cx, cy, p.yaw, m['length'], m['width']))
    return out


def sample_start(rng, slot):
    """Aisle pose after driving past `slot` along +x."""
    x = slot.entrance_x + rng.uniform(-3.0, 8.0)
    y = rng.uniform(-1.0, 1.0)
    return (x, y, rng.uniform(-math.radians(10), math.radians(10)))


def check_path(car, path, cars, slot, planning_radius):
    """Exact checks of a sampled path. Returns a dict of metrics."""
    xs, ys, th, dirs, ks = path.x, path.y, path.yaw, path.direction, path.curvature
    clear, _ = path_clearance(car, xs, ys, th, cars)
    step = np.hypot(np.diff(xs), np.diff(ys))
    # heading consistency: motion vector vs yaw * direction
    mv = np.stack([np.diff(xs), np.diff(ys)], 1)
    hv = np.stack([np.cos(th[1:]), np.sin(th[1:])], 1) * dirs[1:, None]
    cosang = np.sum(mv * hv, 1) / np.maximum(step, 1e-9)
    kin_ok = bool(np.all(np.abs(ks) <= 1 / planning_radius + 1e-6) and np.all(step <= 0.1 + 1e-6)
                  and np.all(cosang > 0.99))
    # final pose relative to the true slot
    c, s = math.cos(slot.heading), math.sin(slot.heading)
    body = car.corners((xs[-1], ys[-1], th[-1]))
    slot_poly = slot.corners()
    inside = all(point_in_convex(p, slot_poly) for p in body)
    cx = xs[-1] + (car.front - car.rear) / 2 * math.cos(th[-1])
    cy = ys[-1] + (car.front - car.rear) / 2 * math.sin(th[-1])
    lat = -(cx - slot.entrance_x) * s + (cy - slot.entrance_y) * c
    dep = (cx - slot.entrance_x) * c + (cy - slot.entrance_y) * s - slot.depth / 2
    yaw_err = wrap(th[-1] - (slot.heading + math.pi))
    return dict(clearance=clear, kinematic_ok=kin_ok, inside=inside, lateral=lat, depth=dep,
                yaw_err_deg=math.degrees(yaw_err), max_step=float(step.max()), min_cos=float(cosang.min()))


def perturb(specs, rng, pos_std, yaw_std):
    out = []
    for s in specs:
        out.append(pg.SlotSpec(s.id, s.x + rng.normal(0, pos_std), s.y + rng.normal(0, pos_std),
                               s.theta + rng.normal(0, yaw_std), s.width, s.depth, s.vacant))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', default='0-49')
    ap.add_argument('--starts', type=int, default=2, help='start poses per empty slot')
    ap.add_argument('--noise-pos', type=float, default=0.0, help='slot position noise std (m)')
    ap.add_argument('--noise-yaw-deg', type=float, default=0.0)
    ap.add_argument('--out', default=os.path.expanduser('~/autopark_results/plan_bench'))
    ap.add_argument('--label', default='gt')
    ap.add_argument('--plots', type=int, default=6, help='number of example plots to save')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    car, cfg = CarGeometry(), PlannerConfig()
    planner = HybridAStar(car, cfg)
    lot = {s.id: s for s in lot_slots()}
    rows, examples = [], []
    for seed in parse_seeds(args.seeds):
        sc = make_scenario(seed)
        cars = true_car_rects(sc)
        rng = np.random.default_rng(1000 + seed)
        specs = [pg.SlotSpec(s.id, s.entrance_x, s.entrance_y, s.heading, s.width, s.depth,
                             vacant=s.id in sc.empty_slot_ids) for s in lot.values()]
        for tid in sc.empty_slot_ids:
            for k in range(args.starts):
                start = sample_start(rng, lot[tid])
                seen = perturb(specs, rng, args.noise_pos, math.radians(args.noise_yaw_deg))
                target = seen[tid]
                goal = pg.goal_pose(target, car)
                polys = pg.obstacles(seen, tid)
                t0 = time.monotonic()
                omap = ObstacleMap(*pg.window([start, goal]), polys)
                path = planner.plan(start, goal, omap)
                dt = time.monotonic() - t0
                info = planner.last_info
                row = dict(seed=seed, slot=tid, k=k, car_past_slot=start[0] - lot[tid].entrance_x, start_y=start[1],
                           start_yaw_deg=math.degrees(start[2]), row='A' if tid < 8 else 'B',
                           ok=path is not None, reason=info['reason'], time=dt,
                           expansions=info['expansions'])
                if path is not None:
                    m = check_path(car, path, cars, lot[tid], car.min_radius)
                    row.update(length=path.length, gears=path.n_gear_changes, **m)
                    if len(examples) < args.plots and (k == 0):
                        examples.append((seed, tid, start, path, cars, sc))
                rows.append(row)
                print(f"seed {seed:3d} slot {tid:2d} past slot {row['car_past_slot']:+5.1f} y {start[1]:+4.1f}: "
                      + (f"ok {dt:5.2f}s len {row['length']:5.1f} m gears {row['gears']} "
                         f"clear {row['clearance']:.2f} m inside {row['inside']} kin {row['kinematic_ok']}"
                         if path is not None else f"FAIL ({info['reason']}, {dt:.2f}s)"), flush=True)

    keys = sorted({k for r in rows for k in r})
    csv_path = os.path.join(args.out, f'{args.label}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    with open(os.path.join(args.out, f'{args.label}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    if examples:
        plot_examples(examples, car, os.path.join(args.out, f'{args.label}_examples.png'))


def summarise(rows):
    ok = [r for r in rows if r['ok']]
    t = np.array([r['time'] for r in rows])
    s = dict(problems=len(rows), solved=len(ok), success_rate=len(ok) / max(len(rows), 1),
             failures={})
    for r in rows:
        if not r['ok']:
            s['failures'][r['reason']] = s['failures'].get(r['reason'], 0) + 1
    s['time_s'] = dict(median=float(np.median(t)), p90=float(np.percentile(t, 90)), max=float(t.max()))
    if ok:
        cl = np.array([r['clearance'] for r in ok])
        s.update(collision_free=int(np.sum(cl > 0)), min_clearance=float(cl.min()),
                 clearance_p5=float(np.percentile(cl, 5)),
                 kinematic_ok=int(sum(r['kinematic_ok'] for r in ok)),
                 inside_slot=int(sum(r['inside'] for r in ok)),
                 length_m=dict(median=float(np.median([r['length'] for r in ok])),
                               max=float(max(r['length'] for r in ok))),
                 gear_changes=dict(median=float(np.median([r['gears'] for r in ok])),
                                   max=int(max(r['gears'] for r in ok)),
                                   hist={str(g): int(sum(r['gears'] == g for r in ok))
                                         for g in sorted({r['gears'] for r in ok})}),
                 final_lateral_abs_max=float(max(abs(r['lateral']) for r in ok)),
                 final_depth_abs_max=float(max(abs(r['depth']) for r in ok)),
                 final_yaw_abs_max_deg=float(max(abs(r['yaw_err_deg']) for r in ok)))
    return s


def plot_examples(examples, car, path_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    n = len(examples)
    cols = min(3, n)
    rows_ = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(6 * cols, 4.2 * rows_), squeeze=False)
    lot = lot_slots()
    for ax, (seed, tid, start, path, cars, sc) in zip(axes.flat, examples):
        for s in lot:
            ax.add_patch(Polygon(s.corners(), fill=False, ec='0.6', lw=0.8))
        for c in cars:
            ax.add_patch(Polygon(c, fc='0.3', ec='k', alpha=0.6))
        ax.add_patch(Polygon(lot[tid].corners(), fc='g', alpha=0.15))
        for d, col in ((1, 'tab:blue'), (-1, 'tab:red')):
            m = np.where(path.direction == d, 1.0, np.nan)
            ax.plot(path.x * m, path.y * m, color=col, lw=1.5, label='forward' if d > 0 else 'reverse')
        for i in range(0, len(path.x), 12):
            ax.add_patch(Polygon(car.corners((path.x[i], path.y[i], path.yaw[i])), fill=False, ec='0.5', lw=0.4))
        ax.add_patch(Polygon(car.corners(start), fill=False, ec='tab:blue', lw=1.2))
        ax.add_patch(Polygon(car.corners((path.x[-1], path.y[-1], path.yaw[-1])), fill=False, ec='tab:green', lw=1.5))
        ax.set_title(f'seed {seed}, slot {tid}: {path.length:.1f} m, {path.n_gear_changes} gear changes', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlim(lot[tid].entrance_x - 9, lot[tid].entrance_x + 12)
        ax.set_ylim(-9.5, 9.5)
        ax.legend(fontsize=7, loc='lower right')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    print('saved', path_png)


if __name__ == '__main__':
    main()
