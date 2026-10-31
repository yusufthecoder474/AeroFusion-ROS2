"""Tests for tracker_kalman.py (synthetic inputs only)."""

from tracker_kalman import IoUTracker, KalmanTracker, iou


def test_iou_identical_and_disjoint():
    box = [0.0, 0.0, 10.0, 10.0]
    assert iou(box, box) == 1.0
    assert iou(box, [20.0, 20.0, 30.0, 30.0]) == 0.0


def test_kalman_keeps_id_on_small_move():
    tracker = KalmanTracker()
    first = tracker.update([[0.0, 0.0, 20.0, 20.0]])
    second = tracker.update([[2.0, 0.0, 22.0, 20.0]])
    assert first[0][0] == second[0][0]


def test_kalman_reacquires_after_dropout():
    tracker = KalmanTracker()
    tracker.update([[0.0, 0.0, 20.0, 20.0]])
    tracker.update([[4.0, 0.0, 24.0, 20.0]])
    assert tracker.update([]) == []  # no output while coasted
    # Next box follows the same velocity; prediction must re-acquire ID 1,
    # where a stale box would still be close here but the ID must not change.
    reacquired = tracker.update([[8.0, 0.0, 28.0, 20.0]])
    assert reacquired[0][0] == 1


def test_new_far_detection_gets_new_id():
    tracker = KalmanTracker()
    tracker.update([[0.0, 0.0, 20.0, 20.0]])
    out = tracker.update([[0.0, 0.0, 20.0, 20.0], [200.0, 200.0, 220.0, 220.0]])
    assert sorted(t[0] for t in out) == [1, 2]


def test_track_dies_after_max_age_and_iou_baseline_keeps_interface():
    tracker = KalmanTracker(max_age=2)
    tracker.update([[0.0, 0.0, 20.0, 20.0]])
    tracker.update([])
    tracker.update([])
    tracker.update([])
    # Dead track: the same box now gets a new ID, not ID 1.
    assert tracker.update([[0.0, 0.0, 20.0, 20.0]])[0][0] == 2
    baseline = IoUTracker()
    out = baseline.update([[0.0, 0.0, 20.0, 20.0]])
    assert isinstance(out[0][0], int) and out[0][1][2] > out[0][1][0]
