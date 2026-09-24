"""Evaluate a trained slot detector on a labelled split (default: test, real 1-3-empty scenarios).

    ros2 run autopark eval_slots --model ~/autopark_models/slotnet.pt --split test

A ground-truth slot is *in view* if both entrance corners are inside the network input
(0.25 m margin) and on valid (non-black) BEV pixels. Matching:
  marking point: < 0.30 m          slot: entrance < 0.40 m and heading < 10 deg
A prediction counts as false positive only if it matches no ground-truth slot at all.
Vacant precision = predicted-vacant slots that are truly vacant (a wrong "vacant" means
driving into a parked car, so this is the safety-critical number).
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch

from autopark import slot_codec as sc
from autopark.slot_net import load_model, predict
from autopark.slot_viz import draw_points, draw_slots, gt_slots_ground

POINT_TOL = 0.30
SLOT_TOL = 0.40
HEADING_TOL = math.radians(10)
MARGIN = 0.25
BINS = [(0, 4), (4, 7), (7, 10)]     # m, entrance distance from the car centre
CAR_CENTRE_X = 1.35


def q(a, pct):
    """Percentile that is NaN (not an error) for an empty list."""
    return float(np.percentile(a, pct)) if len(a) else float('nan')


def in_view(x, y, bev_gray):
    col, row = sc.ground_to_input(x, y)
    m = MARGIN / 0.04
    if not (m <= col < sc.INPUT - m and m <= row < sc.INPUT - m):
        return False
    return bev_gray[int(round(row)) + sc.CROP, int(round(col)) + sc.CROP] > 0


def greedy_match(pred, gt, dist_fn, tol):
    """Greedy one-to-one matching by increasing distance. Returns list of (i_pred, i_gt, d)."""
    pairs = sorted((dist_fn(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt))
    used_p, used_g, out = set(), set(), []
    for d, i, j in pairs:
        if d < tol and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j, d))
    return out


def slot_dist(p, g):
    """Entrance distance, or inf if the headings differ by more than HEADING_TOL."""
    dh = abs(math.atan2(math.sin(p.heading - g.heading), math.cos(p.heading - g.heading)))
    if dh > HEADING_TOL:
        return math.inf
    return math.dist(p.entrance, g.entrance)


def bin_of(x, y):
    r = math.hypot(x - CAR_CENTRE_X, y)
    return next((k for k, (a, b) in enumerate(BINS) if a <= r < b), None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.expanduser('~/autopark_models/slotnet.pt'))
    ap.add_argument('--data', default=os.path.expanduser('~/autopark_data/slots_v1'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--out', default=os.path.expanduser('~/autopark_results/slot_eval'))
    ap.add_argument('--vis', type=int, default=8, help='number of overlay images to save')
    args = ap.parse_args(argv)

    torch.set_num_threads(4)
    net = load_model(args.model)
    root = os.path.join(args.data, args.split)
    items = [json.loads(l) for l in open(os.path.join(root, 'labels.jsonl'))]
    os.makedirs(args.out, exist_ok=True)
    vis_every = max(len(items) // max(args.vis, 1), 1)

    pt = dict(tp=0, fp=0, fn=0, err=[], dir_err=[])
    sl = {k: dict(tp=0, fn=0) for k in range(len(BINS))}
    sl_fp, ent_err, head_err = 0, [], []
    occ = dict(correct=0, total=0, pred_vac=0, pred_vac_true=0, gt_vac=0, gt_vac_found=0)
    t_inf = []

    for n, it in enumerate(items):
        bev = cv2.imread(os.path.join(root, 'images', it['image']))
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        slots, pts, _ = predict(net, bev, args.threshold)
        t_inf.append(time.perf_counter() - t0)

        # marking points
        gpts = sc.marking_points_from_slots(it['slots'], it['pose'])
        gvis = [g for g in gpts if in_view(g.x, g.y, gray)]
        m = greedy_match(pts, gvis, lambda p, g: math.hypot(p.x - g.x, p.y - g.y), POINT_TOL)
        pt['tp'] += len(m)
        pt['fn'] += len(gvis) - len(m)
        matched_any = greedy_match(pts, gpts, lambda p, g: math.hypot(p.x - g.x, p.y - g.y), POINT_TOL)
        pt['fp'] += len(pts) - len(matched_any)
        for i, j, d in m:
            pt['err'].append(d)
            pt['dir_err'].append(math.degrees(math.acos(max(-1, min(1, pts[i].dx * gvis[j].dx + pts[i].dy * gvis[j].dy)))))

        # slots
        gslots = gt_slots_ground(it['slots'], it['pose'])
        vis = [all(in_view(c.x, c.y, gray) for c in (g.left, g.right)) for g in gslots]
        m_all = greedy_match(slots, gslots, slot_dist, SLOT_TOL)
        sl_fp += len(slots) - len(m_all)
        matched_gt = {j: i for i, j, _ in m_all}
        for j, g in enumerate(gslots):
            if not vis[j]:
                continue
            b = bin_of(*g.entrance)
            if b is None:
                continue
            gv = g.vacancy > 0.5
            occ['gt_vac'] += gv
            if j in matched_gt:
                p = slots[matched_gt[j]]
                sl[b]['tp'] += 1
                ent_err.append(math.dist(p.entrance, g.entrance))
                head_err.append(abs(math.degrees(math.atan2(math.sin(p.heading - g.heading),
                                                            math.cos(p.heading - g.heading)))))
                if not math.isnan(p.vacancy):
                    pv = p.vacancy > 0.5
                    occ['total'] += 1
                    occ['correct'] += pv == gv
                    occ['gt_vac_found'] += pv and gv
            else:
                sl[b]['fn'] += 1
        for i, p in enumerate(slots):   # vacant precision over all predictions
            if not math.isnan(p.vacancy) and p.vacancy > 0.5:
                occ['pred_vac'] += 1
                js = [j for ii, j, _ in m_all if ii == i]
                occ['pred_vac_true'] += bool(js) and gslots[js[0]].vacancy > 0.5

        if n % vis_every == 0 and n // vis_every < args.vis:
            img = bev.copy()
            draw_slots(img, slots)
            draw_points(img, pts)
            ref = bev.copy()
            draw_slots(ref, gt_slots_ground(it['slots'], it['pose']), label=False)
            cv2.putText(ref, 'ground truth', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(img, 'prediction', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(args.out, f'pred_{n:04d}.png'), np.hstack([ref, img]))

    def pr(tp, fp, fn):
        return tp / max(tp + fp, 1) * 100, tp / max(tp + fn, 1) * 100

    lines = [f'{len(items)} images from {root}, threshold {args.threshold}',
             f'inference (CPU, network + decoding): median {np.median(t_inf) * 1000:.0f} ms', '']
    p, r = pr(pt['tp'], pt['fp'], pt['fn'])
    lines.append(f'marking points : precision {p:5.1f}%  recall {r:5.1f}%  | position error median '
                 f'{q(pt["err"], 50) * 100:.1f} cm, p90 {q(pt["err"], 90) * 100:.1f} cm | '
                 f'direction error median {q(pt["dir_err"], 50):.1f} deg')
    tp = sum(v['tp'] for v in sl.values())
    fn = sum(v['fn'] for v in sl.values())
    p, r = pr(tp, sl_fp, fn)
    lines.append(f'slots          : precision {p:5.1f}%  recall {r:5.1f}%  | entrance error median '
                 f'{q(ent_err, 50) * 100:.1f} cm, p90 {q(ent_err, 90) * 100:.1f} cm | '
                 f'heading error median {q(head_err, 50):.2f} deg, p90 {q(head_err, 90):.2f} deg')
    for k, (a, b) in enumerate(BINS):
        v = sl[k]
        lines.append(f'  recall {a}-{b} m from car centre: {v["tp"] / max(v["tp"] + v["fn"], 1) * 100:5.1f}% '
                     f'({v["tp"]}/{v["tp"] + v["fn"]})')
    lines.append(f'occupancy      : accuracy {occ["correct"] / max(occ["total"], 1) * 100:5.1f}% '
                 f'({occ["total"]} matched slots) | vacant precision '
                 f'{occ["pred_vac_true"] / max(occ["pred_vac"], 1) * 100:5.1f}% '
                 f'({occ["pred_vac_true"]}/{occ["pred_vac"]}) | vacant recall '
                 f'{occ["gt_vac_found"] / max(occ["gt_vac"], 1) * 100:5.1f}% ({occ["gt_vac_found"]}/{occ["gt_vac"]})')
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.out, 'metrics.txt'), 'w') as f:
        f.write(text + '\n')
    print(f'\noverlays + metrics.txt in {args.out}')


if __name__ == '__main__':
    main()
