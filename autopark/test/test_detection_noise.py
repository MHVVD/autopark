import math

import numpy as np

from autopark.detection_noise import ErrorField, perturb


def test_error_field_std_and_smoothness():
    """Over many draws the field has the requested std at any point; nearby points are almost
    the same (systematic), points a wavelength apart are nearly independent."""
    vals, near, far = [], [], []
    for k in range(3000):
        f = ErrorField(np.random.default_rng(k), 0.10, math.radians(2.0), wavelength=6.0)
        a = f(3.0, 2.0)
        vals.append(a)
        near.append(f(3.1, 2.0))
        far.append(f(9.0, 2.0))
    vals, near, far = np.array(vals), np.array(near), np.array(far)
    std = vals.std(0)
    assert abs(std[0] - 0.10) < 0.006 and abs(std[1] - 0.10) < 0.006
    assert abs(std[2] - math.radians(2.0)) < 0.0015
    assert abs(vals.mean(0)[0]) < 0.006
    assert np.corrcoef(vals[:, 0], near[:, 0])[0, 1] > 0.98
    assert abs(np.corrcoef(vals[:, 0], far[:, 0])[0, 1]) < 0.3


def test_field_noise_is_reproducible_and_the_same_in_every_frame():
    f = ErrorField(np.random.default_rng(5), 0.1, 0.03)
    g = ErrorField(np.random.default_rng(5), 0.1, 0.03)
    assert np.allclose(f(1.0, -2.0), g(1.0, -2.0))
    rng = np.random.default_rng(0)
    slot = [(4.0, 3.5, math.pi / 2, 2.6, 1.0)]
    a = perturb(slot, rng, 0.1, 0.03, 0.0, 0.0, 0.0, field=f)
    b = perturb(slot, rng, 0.1, 0.03, 0.0, 0.0, 0.0, field=f)
    assert a == b and a[0][:2] != (4.0, 3.5)
