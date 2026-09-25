import math

import numpy as np
import pytest

from autopark.slot_tracker import (CHI2_3_999, Detection, SlotTracker, meas_std, vacancy_weight,
                                   wrap)

ALWAYS = lambda x, y, th=None: True     # noqa: E731
NEVER = lambda x, y, th=None: False     # noqa: E731


def det(x, y, th=math.pi / 2, vac=-1.0, rng=5.0, width=2.6):
    return Detection(x, y, th, width, vac, rng)


def row(n=4, y=3.5):
    return [(-4 + 2.6 * i, y) for i in range(n)]


def test_error_model_grows_with_range():
    assert meas_std(2)[0] < meas_std(8)[0] and meas_std(2)[1] < meas_std(8)[1]
    assert vacancy_weight(3) == 1.0 and vacancy_weight(9) == pytest.approx(0.1)


def test_confirmation_and_one_track_per_slot():
    tr = SlotTracker()
    for k in range(5):
        tr.update([det(x, y) for x, y in row()], k * 0.1, ALWAYS)
        if k == 1:
            assert tr.confirmed() == []                  # 2 hits: still tentative
    assert len(tr.tracks) == len(tr.confirmed()) == 4    # no duplicates, ids stable
    assert sorted(t.id for t in tr.tracks) == [0, 1, 2, 3]


def test_noisy_detections_are_averaged_and_covariance_is_consistent():
    """Monte Carlo: 30 noisy detections of one slot; the estimate beats a single detection
    and the normalised estimation error squared (NEES) matches 3 dof on average."""
    rng = np.random.default_rng(0)
    truth = np.array([1.0, 3.5, math.pi / 2])
    sig_p, sig_y = 0.10, math.radians(2)
    nees, err_track, err_det = [], [], []
    for trial in range(300):
        tr = SlotTracker(extra_pos_std=sig_p, extra_yaw_std=sig_y)
        for k in range(30):
            sp = math.hypot(sig_p, meas_std(5)[0])
            sy = math.hypot(sig_y, meas_std(5)[1])
            n = rng.normal(0, 1, 3) * np.array([sp, sp, sy])
            d = det(*(truth[:2] + n[:2]), th=truth[2] + n[2])
            if k == 0:
                err_det.append(math.hypot(n[0], n[1]))
            tr.update([d], k * 0.1, ALWAYS)
        t = tr.tracks[0]
        e = np.array([t.x[0] - truth[0], t.x[1] - truth[1], wrap(t.x[2] - truth[2])])
        nees.append(e @ np.linalg.solve(t.P, e))
        err_track.append(math.hypot(e[0], e[1]))
    assert np.mean(nees) == pytest.approx(3.0, rel=0.2)          # consistent covariance
    assert np.median(err_track) < 0.3 * np.median(err_det)       # ~1/sqrt(30) of one detection


def test_out_of_view_track_is_remembered_but_in_view_misses_kill_it():
    tr = SlotTracker(max_misses_confirmed=5)
    for k in range(4):
        tr.update([det(0, 3.5)], k, ALWAYS)
    for k in range(50):                                   # not visible: kept
        tr.update([], 10 + k, NEVER)
    assert len(tr.confirmed()) == 1
    for k in range(5):                                    # visible but not detected: deleted
        tr.update([], 100 + k, ALWAYS)
    assert tr.tracks == []


def test_single_false_detection_never_confirms():
    tr = SlotTracker()
    tr.update([det(0, 3.5)], 0, ALWAYS)
    for k in range(3):
        tr.update([], 1 + k, ALWAYS)
    assert tr.tracks == []


def test_prediction_inflates_covariance_with_distance():
    tr = SlotTracker()
    tr.update([det(0, 3.5)], 0, ALWAYS)
    p0 = tr.tracks[0].P.copy()
    tr.predict(10.0)
    assert np.all(np.diag(tr.tracks[0].P) > np.diag(p0))


def test_association_gate_rejects_neighbouring_slot():
    tr = SlotTracker()
    for k in range(3):
        tr.update([det(0, 3.5)], k, ALWAYS)
    # a detection one slot width away must start a new track, not drag the existing one
    tr.update([det(2.6, 3.5)], 3, ALWAYS)
    assert len(tr.tracks) == 2 and abs(tr.tracks[0].x[0]) < 0.01


def test_heading_wraps_across_pi():
    tr = SlotTracker()
    for k, th in enumerate([math.pi - 0.002, -math.pi + 0.002, math.pi - 0.001]):
        tr.update([det(0, -3.5, th=th)], k, ALWAYS)
    assert len(tr.tracks) == 1
    assert abs(wrap(tr.tracks[0].x[2] - math.pi)) < 0.01


def test_vacancy_fusion_trusts_close_observations_more():
    tr = SlotTracker()
    # 6 far "vacant" votes (unreliable) then 3 close "occupied" votes: occupied wins
    for k in range(6):
        tr.update([det(0, 3.5, vac=0.9, rng=8.5)], k, ALWAYS)
    assert tr.tracks[0].vacancy > 0.5
    for k in range(3):
        tr.update([det(0, 3.5, vac=0.05, rng=3.0)], 10 + k, ALWAYS)
    t = tr.tracks[0]
    assert t.vacancy < 0.05 and t.n_vac_close == 3


def test_gate_value():
    assert CHI2_3_999 == pytest.approx(16.27, abs=0.01)


def test_noise_injector_statistics():
    from autopark.detection_noise import perturb
    rng = np.random.default_rng(1)
    slots = [(4.0, 3.5, math.pi / 2, 2.6, 1.0)] * 2000
    out = perturb(slots, rng, pos_std=0.1, yaw_std=0.05, dropout=0.25, vacancy_flip=0.1, false_per_frame=0)
    xs = np.array([o[0] for o in out])
    assert len(out) / len(slots) == pytest.approx(0.75, abs=0.03)
    assert xs.std() == pytest.approx(0.1, rel=0.1) and abs(xs.mean() - 4.0) < 0.01
    assert np.mean([o[4] == 0.0 for o in out]) == pytest.approx(0.1, abs=0.02)
    assert len(perturb([], rng, 0, 0, 0, 0, false_per_frame=50)) > 30


def test_outlier_detection_next_to_a_track_does_not_spawn_a_duplicate():
    tr = SlotTracker()
    for k in range(3):
        tr.update([det(0, 3.5)], k, ALWAYS)
    for k in range(15):  # a persistent 40 cm outlier: outside the tight gate, but physically the same slot
        tr.update([det(0.4, 3.5)], 3 + k, ALWAYS)
    assert len(tr.tracks) == 1 and tr.tracks[0].id == 0   # no duplicate, and not deleted as "missed"


def test_tracks_that_converge_are_merged():
    tr = SlotTracker()
    for k in range(3):
        tr.update([det(0, 3.5), det(2.6, 3.5)], k, ALWAYS)
    tr.tracks[1].x[:2] = [0.3, 3.5]          # drifted onto its neighbour
    tr.tracks[1].hits = 1
    tr.update([], 10, NEVER)
    assert len(tr.tracks) == 1 and tr.tracks[0].hits >= 3


def test_view_angle_validity():
    from autopark.slot_tracker import heading_valid, in_view
    tol = math.radians(20)
    assert heading_valid(math.pi / 2, tol) and heading_valid(-math.pi / 2 + 0.3, tol)
    assert not heading_valid(0.0, tol) and not heading_valid(math.pi / 4, tol)
    assert heading_valid(0.0) and heading_valid(math.pi)          # default: any heading
    # a slot 3.5 m to the left of the car: in view when perpendicular, not when the car has turned
    assert in_view((0.0, 0.0, 0.0), 1.35, 3.5, theta=math.pi / 2)
    assert not in_view((0.0, 0.0, math.radians(45)), 1.35, 3.5, theta=math.pi / 2, yaw_tol=tol)
    assert in_view((0.0, 0.0, math.radians(45)), 1.35, 3.5, theta=math.pi / 2)
    assert in_view((0.0, 0.0, 0.0), 1.35, 3.5)          # without a heading: position only
