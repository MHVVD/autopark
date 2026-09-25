"""Make the demo video from recorded runs (visualizer record_dir: 3d/<t_ms>.jpg from the Webots
supervisor's demo camera, viz/<t_ms>.jpg + viz/index.csv from the visualizer), plus title and
result cards.

Each output frame shows the Webots 3D view next to the visualizer image at the same simulated
time, with a caption for the parking manager's state. Frames are 1920x1080, `fps` per second of
video; `speed` s of simulated time per s of video (per segment).

    ros2 run autopark make_demo_video --out demo.mp4 \\
        --title "Vision-based autonomous parking" \\
        --segment ~/autopark_demo/run1 "Closed loop, scenario 5" 2 \\
        --image ~/autopark_results/noise_sweep/corner_error.png "Results" 6
"""
import argparse
import bisect
import glob
import os
import textwrap

import cv2
import numpy as np

from autopark.visualizer import CAPTIONS

W, H = 1920, 1080
BG = (24, 24, 24)
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, y, scale=1.0, color=(235, 235, 235), center=True, x=60, thick=1):
    for line in s.split('\n'):
        (tw, th), _ = cv2.getTextSize(line, FONT, scale, thick)
        if tw > W - 80:                       # shrink lines that would not fit
            scale *= (W - 80) / tw
            (tw, th), _ = cv2.getTextSize(line, FONT, scale, thick)
        cv2.putText(img, line, ((W - tw) // 2 if center else x, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += int(th * 1.8)
    return y


def card(title, body='', seconds=None, fps=10):
    """Title card, shown long enough to read (about 4 words per second, at least 3 s)."""
    if seconds is None:
        seconds = max(3.0, 1.5 + len((title + ' ' + body).split()) / 4.0)
    img = np.full((H, W, 3), BG, np.uint8)
    y = text(img, title, H // 2 - 80, 1.6, (255, 255, 255), thick=2)
    text(img, '\n'.join(textwrap.wrap(body, 70)), y + 30, 0.9, (200, 200, 200))
    return [img] * int(round(seconds * fps))


def image_card(path, caption, seconds, fps):
    img = np.full((H, W, 3), BG, np.uint8)
    pic = cv2.imread(path)
    s = min((W - 120) / pic.shape[1], (H - 200) / pic.shape[0])
    pic = cv2.resize(pic, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    y0, x0 = 140, (W - pic.shape[1]) // 2
    img[y0:y0 + pic.shape[0], x0:x0 + pic.shape[1]] = pic
    text(img, caption, 80, 1.2, (255, 255, 255))
    return [img] * int(round(seconds * fps))


def frames_by_time(d):
    files = sorted(glob.glob(os.path.join(d, '*.jpg')))
    return [int(os.path.basename(f)[:-4]) for f in files], files


def segment(run_dir, label, speed, fps, crop_left=0):
    t3, f3 = frames_by_time(os.path.join(run_dir, '3d'))
    tv, fv = frames_by_time(os.path.join(run_dir, 'viz'))
    states = {}
    idx = os.path.join(run_dir, 'viz', 'index.csv')
    if os.path.exists(idx):
        for line in open(idx):
            t, st = line.strip().split(',')
            states[int(t)] = st
    resets = [t for t, st in states.items() if st == 'reset']
    states = {t: st for t, st in states.items() if st != 'reset'}
    ts_states = sorted(states)
    if not tv:
        raise SystemExit(f'no visualizer frames in {run_dir}/viz')
    t_end = tv[-1]
    t_start = next((t for t in tv if t > max(resets)), tv[0]) if resets else tv[0]   # the evaluated run only
    out = []
    step = int(round(1000 * speed / fps))
    for t in range(t_start, t_end + 1, step):
        v = cv2.imread(fv[max(bisect.bisect_right(tv, t) - 1, 0)])
        panels = [v]
        if t3:
            panels.insert(0, cv2.imread(f3[max(bisect.bisect_right(t3, t) - 1, 0)])[:, crop_left:])
        h = 720
        panels = [cv2.resize(p, (int(round(p.shape[1] * h / p.shape[0])), h), interpolation=cv2.INTER_AREA)
                  for p in panels]
        row = np.hstack(panels)
        s = min(1.0, W / row.shape[1], (H - 220) / row.shape[0])
        row = cv2.resize(row, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = np.full((H, W, 3), BG, np.uint8)
        y0, x0 = max(130, (H - row.shape[0]) // 2 + 20), (W - row.shape[1]) // 2
        img[y0:y0 + row.shape[0], x0:x0 + row.shape[1]] = row
        st = states[ts_states[max(bisect.bisect_right(ts_states, t) - 1, 0)]] if ts_states else ''
        text(img, label, 55, 1.1, (255, 255, 255))
        text(img, CAPTIONS.get(st, st), 105, 0.9, (255, 220, 120))
        text(img, f'simulated time {(t - t_start) / 1000:5.1f} s   ({speed:g}x)', H - 30, 0.7, (170, 170, 170))
        out.append(img)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='demo.mp4')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--title', nargs='+', action='append', default=[], metavar=('TITLE', 'BODY'),
                    help='title card: title [body] (repeatable, in order with the other parts)')
    ap.add_argument('--segment', nargs=3, action='append', default=[], metavar=('DIR', 'LABEL', 'SPEED'))
    ap.add_argument('--image', nargs=3, action='append', default=[], metavar=('PNG', 'CAPTION', 'SECONDS'))
    ap.add_argument('--crop-left', type=int, default=0, help='px cut from the left of the 3D view')
    args, _ = ap.parse_known_args(argv)
    # keep the command-line order of the parts
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    order = [a for a in argv if a in ('--title', '--segment', '--image')]
    queues = dict(title=list(args.title), segment=list(args.segment), image=list(args.image))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W, H))
    n = 0
    for kind in order:
        part = queues[kind[2:]].pop(0)
        if kind == '--title':
            frames = card(part[0], ' '.join(part[1:]), fps=args.fps)
        elif kind == '--segment':
            frames = segment(os.path.expanduser(part[0]), part[1], float(part[2]), args.fps, args.crop_left)
        else:
            frames = image_card(os.path.expanduser(part[0]), part[1], float(part[2]), args.fps)
        for f in frames:
            writer.write(f)
        n += len(frames)
    writer.release()
    print(f'{n} frames ({n / args.fps:.0f} s) -> {args.out}')


if __name__ == '__main__':
    main()
