"""Analyse the detection-noise experiment (park_eval runs): plan once vs closed loop.

Reads <dir>/{open,closed}_{white,field}_{NN}.csv (NN = position noise in cm; heading noise is
NN/5 deg), one park_eval run per scenario seed, the same seeds in every condition. Writes
<dir>/summary.md, <dir>/summary.json and plots:
  corner_error.png   final corner error vs noise (median, quartiles, every run), per noise kind
  success.png        parked without contact / refused (no plan) / contact, per condition
  components.png     lateral, depth and heading error vs noise

Corner error: the largest distance between a corner of the parked car and where that corner
would be if the car were exactly centred and aligned in the slot (combines the lateral, depth
and heading errors into one number, in cm). Closed loop and plan once are compared per seed
(paired): median difference and the Wilcoxon signed-rank test over seeds where both parked.

    ros2 run autopark noise_report --dir ~/autopark_results/noise_sweep
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

from autopark.hybrid_astar import CarGeometry

CAR = CarGeometry()
HALF_L, HALF_W = (CAR.front + CAR.rear) / 2, CAR.half_width


def corner_error_cm(lat_cm, dep_cm, yaw_deg):
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    worst = 0.0
    for ax, ay in ((HALF_L, HALF_W), (HALF_L, -HALF_W), (-HALF_L, HALF_W), (-HALF_L, -HALF_W)):
        # corner of the rotated car minus the ideal corner, plus the translation (m)
        dx = (c * ax - s * ay - ax) + dep_cm / 100
        dy = (s * ax + c * ay - ay) + lat_cm / 100
        worst = max(worst, math.hypot(dx, dy))
    return 100 * worst


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
        m = re.match(r'(open|closed)_(white|field)_(\d+)\.csv$', os.path.basename(f))
        if not m:
            continue
        mode, kind, level = m.group(1), m.group(2), int(m.group(3))
        for r in csv.DictReader(open(f)):
            ok = r['success'] == 'True'
            row = dict(seed=int(r['seed']), success=ok, state=r['manager_state'], message=r['message'],
                       clearance=float(r['min_clearance']), plans=int(r['plans']), corrections=int(r['corrections']))
            if r.get('lateral_cm'):
                lat, dep, yaw = float(r['lateral_cm']), float(r['depth_cm']), float(r['yaw_deg'])
                row.update(lat=lat, dep=dep, yaw=yaw, corner=corner_error_cm(lat, dep, yaw))
            runs.setdefault((kind, level), {})[mode] = runs.get((kind, level), {}).get(mode, []) + [row]
    # level 0 is noise-free: the same for both kinds
    if ('white', 0) in runs:
        runs[('field', 0)] = runs[('white', 0)]
    return runs


def outcome(r):
    if r['success']:
        return 'parked'
    if r['clearance'] <= 0:
        return 'contact'
    if r['state'] == 'failed' and r['plans'] == 0:
        return 'refused'        # no slot passed the planner (safe), or none found
    return 'other'


def stats(vals):
    a = np.abs(np.asarray(vals, float))
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), median=round(float(np.median(a)), 2), q25=round(float(np.percentile(a, 25)), 2),
                q75=round(float(np.percentile(a, 75)), 2), max=round(float(a.max()), 2))


def paired(o, c):
    """Closed minus open corner error over seeds where both parked."""
    from scipy.stats import wilcoxon
    oo = {r['seed']: r['corner'] for r in o if r['success']}
    cc = {r['seed']: r['corner'] for r in c if r['success']}
    seeds = sorted(set(oo) & set(cc))
    if not seeds:
        return dict(n=0)
    d = np.array([cc[s] - oo[s] for s in seeds])
    p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else float('nan')
    return dict(n=len(d), median_diff=round(float(np.median(d)), 2), closed_better=int((d < 0).sum()),
                wilcoxon_p=round(p, 4))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expanduser('~/autopark_results/noise_sweep'))
    args = ap.parse_args(argv)
    runs = load(args.dir)
    kinds = ('white', 'field')
    levels = sorted({lv for _, lv in runs})
    summary = {}
    for (kind, lv), m in sorted(runs.items()):
        e = {}
        for mode in ('open', 'closed'):
            rs = m.get(mode, [])
            ok = [r for r in rs if r['success']]
            e[mode] = dict(runs=len(rs), **{k: sum(outcome(r) == k for r in rs) for k in ('parked', 'refused', 'contact', 'other')},
                           corner_cm=stats([r['corner'] for r in ok]), lateral_cm=stats([r['lat'] for r in ok]),
                           depth_cm=stats([r['dep'] for r in ok]), yaw_deg=stats([r['yaw'] for r in ok]),
                           min_clearance_m=round(min((r['clearance'] for r in rs), default=float('nan')), 3),
                           corrections_mean=round(float(np.mean([r['corrections'] for r in ok])), 1) if ok else None)
        e['paired'] = paired(m.get('open', []), m.get('closed', []))
        summary[f'{kind}_{lv:02d}'] = e
    with open(os.path.join(args.dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    lines = ['| noise | pos / heading std | mode | parked | refused | contact | corner error median (IQR) cm | '
             'lateral median / max cm | depth median / max cm | closed - open (paired) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for kind in kinds:
        for lv in levels:
            key = f'{kind}_{lv:02d}'
            if key not in summary or (kind == 'field' and lv == 0):
                continue
            e = summary[key]
            for mode in ('open', 'closed'):
                s = e[mode]
                ce, la, de = s['corner_cm'], s['lateral_cm'], s['depth_cm']
                pr = e['paired']
                pair = (f"{pr['median_diff']:+.1f} cm, closed better in {pr['closed_better']}/{pr['n']}, "
                        f"p = {pr['wilcoxon_p']:.3f}" if mode == 'closed' and pr.get('n') else '')
                noise = 'none' if lv == 0 else kind
                lines.append(f"| {noise} | {lv} cm / {lv / 5:g} deg | {mode} | {s['parked']}/{s['runs']} | {s['refused']} | "
                             f"{s['contact']} | " + (f"{ce['median']:.1f} ({ce['q25']:.1f}-{ce['q75']:.1f})" if ce['n'] else '-')
                             + ' | ' + (f"{la['median']:.1f} / {la['max']:.1f}" if la['n'] else '-')
                             + ' | ' + (f"{de['median']:.1f} / {de['max']:.1f}" if de['n'] else '-') + f' | {pair} |')
    with open(os.path.join(args.dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    plots(runs, levels, kinds, args.dir)
    print(f'-> {args.dir}/summary.md, summary.json, corner_error.png, success.png, components.png')


def plots(runs, levels, kinds, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(open='#d9822b', closed='#2b6cd9')
    names = dict(white='white noise (independent per frame)', field='systematic noise (viewpoint-dependent field)')
    rng = np.random.default_rng(0)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind in zip(axs, kinds):
        for k, mode in enumerate(('open', 'closed')):
            off = (k - 0.5) * 0.9
            meds, xs = [], []
            for lv in levels:
                rs = [r for r in runs.get((kind, lv), {}).get(mode, []) if r['success']]
                if not rs:
                    continue
                v = np.array([r['corner'] for r in rs])
                x = lv + off
                ax.scatter(x + rng.uniform(-0.25, 0.25, len(v)), v, s=10, color=colors[mode], alpha=0.35, lw=0)
                ax.errorbar(x, np.median(v), yerr=[[np.median(v) - np.percentile(v, 25)], [np.percentile(v, 75) - np.median(v)]],
                            fmt='o', color=colors[mode], capsize=3)
                meds.append(np.median(v))
                xs.append(x)
            ax.plot(xs, meds, color=colors[mode], label='plan once' if mode == 'open' else 'closed loop')
        ax.set_title(names[kind], fontsize=10)
        ax.set_xlabel('detection noise: position std (cm); heading std = position / 5 (deg)')
        ax.set_xticks(levels)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel('final corner error (cm)')
    axs[0].legend()
    fig.suptitle('Parking accuracy vs detection noise (successful runs; median, quartiles, every run)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'corner_error.png'), dpi=130)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    present = [oc for oc in ('parked', 'refused', 'contact', 'other')
               if any(outcome(r) == oc for m in runs.values() for rs in m.values() for r in rs)]
    for ax, kind in zip(axs, kinds):
        for k, mode in enumerate(('open', 'closed')):
            bottoms = np.zeros(len(levels))
            for oc, shade in ((o, sh) for o, sh in (('parked', 1.0), ('refused', 0.4), ('contact', 0.15), ('other', 0.25))
                              if o in present):
                vals = []
                for lv in levels:
                    rs = runs.get((kind, lv), {}).get(mode, [])
                    vals.append(100 * sum(outcome(r) == oc for r in rs) / max(len(rs), 1))
                x = np.array(levels) + (k - 0.5) * 1.6
                ax.bar(x, vals, width=1.5, bottom=bottoms, color=colors[mode], alpha=shade,
                       label=f"{'plan once' if mode == 'open' else 'closed loop'}: {oc}" if kind == 'white' else None,
                       edgecolor='k', lw=0.3)
                bottoms += vals
        ax.set_title(names[kind], fontsize=10)
        ax.set_xticks(levels)
        ax.set_xlabel('position noise std (cm)')
    axs[0].set_ylabel('% of runs')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=2 * len(present), fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, 'success.png'), dpi=130)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for i, kind in enumerate(kinds):
        for j, (key, lab) in enumerate((('lat', 'lateral (cm)'), ('dep', 'depth (cm)'), ('yaw', 'heading (deg)'))):
            ax = axs[i, j]
            for mode in ('open', 'closed'):
                med, q1, q3 = [], [], []
                for lv in levels:
                    v = np.abs([r[key] for r in runs.get((kind, lv), {}).get(mode, []) if r['success']])
                    med.append(np.median(v) if len(v) else np.nan)
                    q1.append(np.percentile(v, 25) if len(v) else np.nan)
                    q3.append(np.percentile(v, 75) if len(v) else np.nan)
                ax.plot(levels, med, 'o-', color=colors[mode], label='plan once' if mode == 'open' else 'closed loop')
                ax.fill_between(levels, q1, q3, color=colors[mode], alpha=0.15)
            ax.set_title(f'{kind}: |{lab}|', fontsize=9)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel('position noise std (cm)')
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'components.png'), dpi=130)


if __name__ == '__main__':
    main()
