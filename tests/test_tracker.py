import pytest

from conftest import det
from ctrack.predictor import KalmanPredictor
from ctrack.tracker import Tracker
from ctrack.types import TrackState as S

FPS = 30.0


@pytest.fixture
def tracker():
    return Tracker(KalmanPredictor(), coast_ms=300, reacquire_radius_px=80)


def _track_moving(tracker, t0=0.0, n=30, v=200.0):
    """Feed a target moving +x at v px/s for n frames; returns the last timestamp."""
    t = t0
    for i in range(n):
        t = t0 + i / FPS
        tracker.update(det(100 + v * t, 300), t)
    return t


def test_starts_searching_and_ignores_detections_without_lock(tracker):
    assert tracker.update(det(100, 100), 0.0) is S.SEARCHING
    assert tracker.estimate(0.0) is None


def test_lock_goes_tracking_on_next_frame_with_detection(tracker):
    tracker.request_lock()
    assert tracker.update(det(100, 100), 0.0) is S.TRACKING
    assert tracker.estimate(0.0)[0].x == pytest.approx(100)


def test_lock_request_waits_for_detection(tracker):
    tracker.request_lock()
    assert tracker.update(None, 0.0) is S.SEARCHING
    assert tracker.update(None, 0.033) is S.SEARCHING
    assert tracker.update(det(1, 1), 0.066) is S.TRACKING


def test_short_occlusion_coasts_and_resumes_without_relock(tracker):
    tracker.request_lock()
    t = _track_moving(tracker)
    assert tracker.update(None, t + 0.1) is S.COASTING
    est = tracker.estimate(t + 0.1)
    assert est is not None, "COASTING must still publish a predicted pose"
    assert est[0].x == pytest.approx(100 + 200 * (t + 0.1), abs=10)
    assert tracker.update(None, t + 0.3) is S.COASTING  # exactly coast_ms: still coasting
    assert tracker.update(det(100 + 200 * (t + 0.33), 300), t + 0.33) is S.TRACKING


def test_long_occlusion_goes_lost_without_pose(tracker):
    tracker.request_lock()
    t = _track_moving(tracker)
    tracker.update(None, t + 0.2)
    assert tracker.update(None, t + 0.35) is S.LOST
    assert tracker.estimate(t + 0.35) is None


def test_reacquire_near_predicted_position(tracker):
    tracker.request_lock()
    t = _track_moving(tracker)
    tracker.update(None, t + 0.35)
    assert tracker.state is S.LOST
    # Target reappears 0.6 s later where it would be at constant velocity. Reference extrapolation
    # is capped at coast_ms, so the gate centre lags by 0.3 s * 200 px/s = 60 px < 80 px radius.
    t2 = t + 0.6
    assert tracker.update(det(100 + 200 * t2, 300), t2) is S.TRACKING
    assert tracker.predictor.velocity().vx == 0  # re-initialized, not updated with stale velocity


def test_no_reacquire_far_away(tracker):
    tracker.request_lock()
    t = _track_moving(tracker)
    tracker.update(None, t + 0.35)
    assert tracker.update(det(1000, 600), t + 0.4) is S.LOST


def test_reset_from_any_state(tracker):
    tracker.request_lock()
    _track_moving(tracker)
    tracker.reset()
    assert tracker.state is S.SEARCHING
    assert not tracker.predictor.initialized
    assert tracker.update(det(1, 1), 5.0) is S.SEARCHING  # needs a new lock


def test_lock_ignored_while_tracking(tracker):
    tracker.request_lock()
    tracker.update(det(1, 1), 0.0)
    tracker.request_lock()
    assert not tracker.lock_requested
