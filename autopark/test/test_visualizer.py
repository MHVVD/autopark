import math
from types import SimpleNamespace

import numpy as np

from autopark import visualizer as vz


def test_compose_frame_with_and_without_data():
    bev = np.full((450, 450, 3), 60, np.uint8)
    empty = vz.compose_frame(bev, (0.0, 0.0, 0.0), [], [], -1, None, None, [], vz.status_lines(None, None, 0, 0, 'open'),
                             vz.CAPTIONS['idle'])
    assert empty.shape == (vz.H, 720 + vz.PANEL_W, 3)
    tracks = [(1, 2.0, 3.5, math.pi / 2, True), (2, 4.6, 3.5, math.pi / 2, False)]
    xs = list(np.linspace(0, -3, 20))
    path = (xs, [0.0] * 20, [-1] * 20, 0.0)
    st = SimpleNamespace(state='executing', slot_id=1, plans=1, corrections=0)
    ct = SimpleNamespace(state='track', direction=-1, segment=0, segments=1, remaining=2.0, lateral_error=0.01,
                         heading_error=0.0)
    img = vz.compose_frame(bev, (1.0, 0.5, 0.2), [(3.0, 2.0, 1.5)], tracks, 1, path, path, [(0, 0), (1, 0.5)],
                           vz.status_lines(st, ct, -0.4, 0.2, 'closed'), vz.CAPTIONS['executing'])
    assert img.shape == empty.shape and not np.array_equal(img, empty)


def test_to_frame_inverts_the_pose():
    p = vz.to_frame((2.0, 1.0, math.pi / 2), [(2.0, 3.0)])[0]
    assert abs(p[0] - 2.0) < 1e-9 and abs(p[1]) < 1e-9
