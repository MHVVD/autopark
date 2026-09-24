import math

import numpy as np
import pytest

from autopark.bev import BevGrid, BevStitcher, CameraModel
from autopark_sim.rig import CAMERAS, FOOTPRINT, T_ground_base, camera_by_name


@pytest.mark.parametrize('spec', CAMERAS, ids=lambda c: c.name)
def test_pixel_ground_round_trip(spec):
    cam = CameraModel(spec)
    u, v = np.meshgrid(np.linspace(5, spec.width - 5, 25), np.linspace(5, spec.height - 5, 19))
    gx, gy = cam.ray_to_ground(u.ravel(), v.ravel())
    hit = ~np.isnan(gx)
    assert hit.sum() > 100  # most of the lower image sees the ground
    pts = np.stack([gx[hit], gy[hit], np.zeros(hit.sum())], axis=1)
    u2, v2, ok = cam.project(pts)
    assert ok.all()
    assert np.abs(u2 - u.ravel()[hit]).max() < 1e-6
    assert np.abs(v2 - v.ravel()[hit]).max() < 1e-6


@pytest.mark.parametrize('spec', CAMERAS, ids=lambda c: c.name)
def test_optical_axis_hits_image_centre(spec):
    cam = CameraModel(spec)
    axis = cam.R[:, 0]                       # camera x axis in the ground frame
    u, v, ok = cam.project((cam.t + 5 * axis)[None])
    assert ok[0]
    assert u[0] == pytest.approx(spec.cx)
    assert v[0] == pytest.approx(spec.cy)


def test_cameras_look_where_expected():
    # A ground point 4 m ahead / behind / left / right of the car is seen by that camera,
    # and a point to the left of the front camera appears in the left half of its image.
    expect = {'front': (7.5, 0.0), 'rear': (-4.5, 0.0), 'left': (1.9, 4.0), 'right': (1.9, -4.0)}
    for name, (x, y) in expect.items():
        _, _, ok = CameraModel(camera_by_name(name)).project(np.array([[x, y, 0.0]]))
        assert ok[0], name
    front = CameraModel(camera_by_name('front'))
    u, _, _ = front.project(np.array([[7.5, 1.0, 0.0]]))
    assert u[0] < front.spec.width / 2


def test_rest_pitch_is_applied():
    # base_link is pitched nose-up at rest: its x axis points slightly upwards in the ground frame
    T = T_ground_base()
    assert T[2, 0] > 0 and T[2, 3] == pytest.approx(0.0247)


def test_grid_pixel_conversions():
    g = BevGrid()
    h, w = g.shape
    assert (h, w) == (450, 450)
    gx, gy = g.pixel_centres()
    c, r = g.to_pixel(gx[10, 20], gy[10, 20])
    assert (c, r) == pytest.approx((20.0, 10.0))
    assert g.to_ground(20.0, 10.0) == pytest.approx((gx[10, 20], gy[10, 20]))
    assert gx[0, 0] > gx[-1, 0]      # top of the image is forward
    assert gy[0, 0] > gy[0, -1]      # left of the image is vehicle left


def _checker(x, y, size=0.5):
    return np.where((np.floor(x / size) + np.floor(y / size)) % 2 == 0, 230, 25)


def test_stitch_reconstructs_a_known_ground_pattern():
    """Render each camera's view of a checkerboard ground with the camera model, stitch, and
    compare the BEV against the checkerboard itself."""
    grid = BevGrid(res=0.05)
    st = BevStitcher(grid)
    images = {}
    for spec in CAMERAS:
        cam = CameraModel(spec)
        u, v = np.meshgrid(np.arange(spec.width) + 0.0, np.arange(spec.height) + 0.0)
        gx, gy = cam.ray_to_ground(u, v)
        val = np.where(np.isnan(gx), 0, _checker(np.nan_to_num(gx), np.nan_to_num(gy)))
        images[spec.name] = np.repeat(val[..., None], 3, axis=2).astype(np.uint8)
    bev = st.stitch(images)[..., 0].astype(float)

    bx, by = grid.pixel_centres()
    truth = _checker(bx, by)
    # compare inside the valid area away from checker edges (distance to an edge, m). Beyond 4 m
    # a fisheye pixel covers ~10 cm, so interpolation blurs a wider band around each edge.
    edge = 0.5 * np.minimum(np.abs(bx / 0.5 - np.round(bx / 0.5)), np.abs(by / 0.5 - np.round(by / 0.5)))
    rng = np.hypot(bx - 1.35, by)
    ok = np.abs(bev - truth) < 30
    near = st.valid & (rng < 4.0) & (edge > 0.05)
    far = st.valid & (rng >= 4.0) & (rng < 6.0) & (edge > 0.12)
    assert near.sum() > 8000 and far.sum() > 8000
    assert ok[near].mean() > 0.999
    assert ok[far].mean() > 0.99


def test_weights_normalised_and_car_masked():
    st = BevStitcher(BevGrid(res=0.1))
    total = np.sum([w[..., 0] for w in st.weights], axis=0)
    assert np.allclose(total[st.valid], 1.0, atol=1e-5)
    gx, gy = st.grid.pixel_centres()
    x0, x1, y0, y1 = FOOTPRINT
    on_car = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
    assert not st.valid[on_car].any()
    # the area around the car is covered all the way round (1 m band outside the footprint)
    ring = ((gx >= x0 - 1) & (gx <= x1 + 1) & (gy >= y0 - 1) & (gy <= y1 + 1)) & ~on_car
    assert st.valid[ring].mean() > 0.98


def test_eval_line_samples_and_frame_change():
    from autopark.bev_eval import line_samples, map_to_ground
    from autopark_sim.lot import painted_lines
    pts = line_samples(step=0.05)
    total = sum(max(sx, sy) for _, _, sx, sy in painted_lines())
    assert len(pts) == pytest.approx(total / 0.05, rel=0.01)
    # a car at (10, 2) facing +y sees the map point (10, 5) 3 m straight ahead
    g = map_to_ground(np.array([[10.0, 5.0]]), (10.0, 2.0, math.pi / 2))
    assert g[0] == pytest.approx((3.0, 0.0))


def test_segment_hits_box():
    from autopark.bev import segment_hits_box
    box = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    p0 = np.array([-1.0, 0.5, 0.5])
    p1 = np.array([[2.0, 0.5, 0.5],    # straight through
                   [-0.5, 0.5, 0.5],   # stops before the box
                   [2.0, 3.0, 0.5],    # passes beside it
                   [0.0, 0.5, 0.5]])   # ends on the surface
    assert segment_hits_box(p0, p1, box).tolist() == [True, False, False, False]


def test_own_car_occludes_ground_beside_it_for_the_rear_camera():
    from autopark.bev import segment_hits_box
    rear = CameraModel(camera_by_name('rear'))
    # beside the car's flank: hidden from the rear camera; well behind the car: visible
    hit = segment_hits_box(rear.t, np.array([[1.5, 1.3, 0.0], [-4.0, 0.0, 0.0]]))
    assert hit.tolist() == [True, False]
