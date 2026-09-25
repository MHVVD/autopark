import math

import numpy as np

from autopark import make_demo_video as mv
from autopark import noise_report as nr


def test_corner_error_translation_rotation_and_sign_symmetry():
    assert abs(nr.corner_error_cm(3.0, 4.0, 0.0) - 5.0) < 1e-9
    half_diag = math.hypot(nr.HALF_L, nr.HALF_W)
    rot = nr.corner_error_cm(0.0, 0.0, 1.0)
    assert abs(rot - 100 * 2 * half_diag * math.sin(math.radians(0.5))) < 1e-6
    # flipping the sign of both translations does not change the worst corner
    assert abs(nr.corner_error_cm(2.0, -1.0, 0.7) - nr.corner_error_cm(-2.0, 1.0, 0.7)) < 1e-9


def test_title_and_image_cards(tmp_path):
    frames = mv.card('Title', 'body text', seconds=1.0, fps=10)
    assert len(frames) == 10 and frames[0].shape == (mv.H, mv.W, 3)
    import cv2
    p = str(tmp_path / 'plot.png')
    cv2.imwrite(p, np.full((300, 500, 3), 200, np.uint8))
    frames = mv.image_card(p, 'Results', 0.5, 10)
    assert len(frames) == 5 and frames[0][400, mv.W // 2, 0] == 200
