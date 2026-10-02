import math
import time

import numpy as np
import pytest

from conftest import det
from ctrack.detector.async_detector import AsyncDetector
from ctrack.detector.base import Detector
from ctrack.metrics import MetricsLogger
from ctrack.pipeline import Pipeline
from ctrack.predictor import KalmanPredictor
from ctrack.publisher import NullPublisher
from ctrack.tracker import Tracker
from ctrack.transform import IdentityTransform
from ctrack.types import Detection, Frame, Pose, TrackState as S

FPS = 30.0
IMG = np.zeros((20, 40, 3), np.uint8)


class SlowDetector(Detector):
    """Returns the true pose of the frame after `delay_s` (a stand-in for a heavy neural detector)."""

    def __init__(self, delay_s: float = 0.07) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def detect(self, frame):
        time.sleep(self.delay_s)
        self.calls += 1
        gt = frame.ground_truth
        return Detection(gt.x, gt.y, gt.theta, 0.95) if gt is not None else None


class DummyCamera:
    is_live = True

    def read(self):
        return None

    def close(self):
        pass


def _frame(i, t0=1000.0, v=200.0):
    t = t0 + i / FPS
    return Frame(IMG, t, i, Pose(100 + v * (i / FPS), 200.0, 0.1))


def _pipeline(detector, async_detect=True):
    tracker = Tracker(KalmanPredictor(process_noise=2000, measurement_noise=2.0, horizon_ms=100))
    return Pipeline(DummyCamera(), detector, tracker, IdentityTransform(), NullPublisher(), async_detect=async_detect)


def _run(pipe, n=75, realtime=True):
    """Feed frames like a 30 fps camera; returns (steps, per-step processing time in ms)."""
    pipe.tracker.request_lock()
    steps, durations = [], []
    t_start = time.perf_counter()
    for i in range(n):
        if realtime:
            wait = t_start + i / FPS - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        f = _frame(i)
        t0 = time.perf_counter()
        steps.append(pipe.process(f))
        durations.append((time.perf_counter() - t0) * 1000.0)
    return steps, durations


def test_slow_detector_does_not_slow_down_the_frame_loop():
    pipe = _pipeline(SlowDetector(0.07))
    steps, ms = _run(pipe)
    pipe.close()
    assert np.percentile(ms, 95) < 15, f"process() blocked on the detector: p95 {np.percentile(ms, 95):.0f} ms"
    fresh = sum(s.detection_new for s in steps)
    assert 15 <= fresh <= 45, f"expected a fresh result every ~2-3 frames, got {fresh}/75"


def test_synchronous_mode_is_unchanged_and_blocks_on_the_detector():
    pipe = _pipeline(SlowDetector(0.05), async_detect=False)
    steps, ms = _run(pipe, n=10, realtime=False)
    assert np.mean(ms) >= 45
    assert all(s.detection_new for s in steps) and all(s.perception_ms == 0.0 for s in steps)


def test_delayed_measurements_are_compensated_by_the_prediction():
    pipe = _pipeline(SlowDetector(0.07))
    steps, _ = _run(pipe)
    pipe.close()
    assert steps[-1].state is S.TRACKING
    pose_err, stale_err = [], []
    for s in steps[20:]:
        gt = s.frame.ground_truth
        pose_err.append(math.hypot(s.pose.x - gt.x, s.pose.y - gt.y))
        stale_err.append(math.hypot(s.detection.x - gt.x, s.detection.y - gt.y))      # what you'd publish without compensation
    assert np.median(stale_err) > 12, "test setup: the raw detection should lag visibly (200 px/s * ~0.1 s)"
    assert np.percentile(pose_err, 90) < 8, f"published pose p90 {np.percentile(pose_err, 90):.1f} px"
    assert np.median(pose_err) < 0.4 * np.median(stale_err)


def test_perception_age_is_reported_and_logged(tmp_path):
    pipe = _pipeline(SlowDetector(0.07))
    steps, _ = _run(pipe, n=45)
    pipe.close()
    m = MetricsLogger(csv=str(tmp_path / "m.csv"))
    for s in steps:
        m.log(s)
    m.close()
    ages = [s.perception_ms for s in steps if s.detection_new]
    assert ages and 20 < np.median(ages) < 250                 # detection time + queueing, never ~0 in async mode
    assert m.stats()["perception_p95"] > 20
    rows = (tmp_path / "m.csv").read_text().splitlines()
    assert "perception_ms" in rows[0] and len(rows) == 46


def test_async_detector_processes_only_the_newest_pending_frame():
    ad = AsyncDetector(SlowDetector(0.05))
    for i in range(6):                                       # six frames arrive while the worker is busy
        ad.submit(_frame(i))
    time.sleep(0.35)
    results = ad.poll()
    ad.close()
    assert len(results) <= 3 and results[-1].frame_id == 5   # stale frames were dropped, the newest was processed
    assert ad.dropped >= 3
    assert [r.t_exposure for r in results] == sorted(r.t_exposure for r in results)   # capture order preserved


def test_worker_exception_surfaces_in_the_main_loop():
    class Broken(Detector):
        def detect(self, frame):
            raise RuntimeError("model exploded")

    ad = AsyncDetector(Broken())
    ad.submit(_frame(0))
    time.sleep(0.1)
    with pytest.raises(RuntimeError, match="model exploded"):
        ad.poll()
    ad.close()


def test_lost_object_is_reported_without_new_results():
    class Vanishing(Detector):
        def detect(self, frame):
            time.sleep(0.02)
            return Detection(100 + frame.frame_id, 200, 0.0, 0.9) if frame.frame_id < 20 else None

    pipe = _pipeline(Vanishing())
    steps, _ = _run(pipe, n=60)
    pipe.close()
    states = [s.state for s in steps]
    assert S.TRACKING in states[:20] and S.COASTING in states and states[-1] is S.LOST


# ---- Tracker.advance ------------------------------------------------------------------------------
@pytest.fixture
def tracker():
    return Tracker(KalmanPredictor(), coast_ms=300, stale_ms=150)


def test_advance_demotes_only_after_stale_and_coast_time(tracker):
    tracker.request_lock()
    tracker.update(det(100, 100), 10.0)
    assert tracker.advance(10.10) is S.TRACKING          # 100 ms without a result: still fine in async mode
    assert tracker.advance(10.20) is S.COASTING          # > stale_ms
    assert tracker.estimate(10.2) is not None            # coasting still publishes the prediction
    assert tracker.advance(10.35) is S.LOST              # > coast_ms
    assert tracker.estimate(10.35) is None


def test_advance_never_promotes_or_touches_searching_and_lost(tracker):
    assert tracker.advance(5.0) is S.SEARCHING
    tracker.request_lock()
    tracker.update(det(1, 1), 1.0)
    tracker.update(None, 1.1)                            # explicit miss reported by the detector
    assert tracker.state is S.COASTING
    assert tracker.advance(1.12) is S.COASTING           # a fresh-looking timestamp must not promote it
    tracker.advance(2.0)
    assert tracker.state is S.LOST
    assert tracker.advance(3.0) is S.LOST
