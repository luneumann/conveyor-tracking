"""End-to-end on the synthetic conveyor: real detection, tracking, prediction, publishing, metrics."""

import numpy as np
import pytest

from ctrack.camera import Recorder, SyntheticConveyorSource, VideoFileSource
from ctrack.detector import MarkerBlobDetector
from ctrack.metrics import MetricsLogger
from ctrack.pipeline import Pipeline
from ctrack.predictor import KalmanPredictor
from ctrack.publisher import NullPublisher
from ctrack.tracker import Tracker
from ctrack.transform import IdentityTransform
from ctrack.types import Frame, TrackState

W = 1280


def _pipeline(camera, horizon_ms=100, coast_ms=300):
    tracker = Tracker(KalmanPredictor(measurement_noise=1.0, horizon_ms=horizon_ms), coast_ms=coast_ms)
    return Pipeline(camera, MarkerBlobDetector(), tracker, IdentityTransform(), NullPublisher())


def _synthetic(**kw):
    args = dict(width=W, height=720, fps=30, velocity=(250.0, 15.0), omega=0.3, start=(100, 300, 0),
                realtime=False, max_frames=120, seed=3)
    args.update(kw)
    return SyntheticConveyorSource(**args)


def test_prediction_error_meets_prd_goal():
    """PRD 7: prediction error at 100 ms horizon, uniform motion, p95 < 2 % of image width."""
    pipe = _pipeline(_synthetic())
    metrics = MetricsLogger(csv=None)
    pipe.tracker.request_lock()
    while (step := pipe.step()) is not None:
        metrics.log(step)
    metrics.close()
    s = metrics.stats()
    assert len(metrics.all_pred_err_px) > 80
    assert s["pred_err_p95_pct"] < 2.0
    assert s["gt_err_p95"] < 2.0


def test_every_tracking_message_has_pose_and_seq_is_contiguous():
    pipe = _pipeline(_synthetic(max_frames=60))
    pipe.tracker.request_lock()
    while pipe.step() is not None:
        pass
    msgs = pipe.publisher.messages
    assert [m.seq for m in msgs] == list(range(len(msgs)))
    assert all(m.pose is not None for m in msgs if m.state is TrackState.TRACKING)
    assert msgs[-1].predicted is not None


class OccludedSource(SyntheticConveyorSource):
    """Blanks frames in [t_from, t_to) seconds to simulate an occlusion."""

    def __init__(self, t_from, t_to, **kw):
        super().__init__(**kw)
        self.occlusion = (t_from, t_to)

    def read(self):
        f = super().read()
        if f is None:
            return None
        t_rel = f.t_exposure - self._t0
        if self.occlusion[0] <= t_rel < self.occlusion[1]:
            return Frame(np.full_like(f.image, 35), f.t_exposure, f.frame_id, f.ground_truth)
        return f


def _states(pipe):
    pipe.tracker.request_lock()
    out = []
    while (step := pipe.step()) is not None:
        out.append((step.frame.t_exposure - pipe.camera._t0, step.state))
    return out


def test_short_occlusion_coasts_then_resumes():
    states = _states(_pipeline(OccludedSource(1.0, 1.2, **_kwargs())))
    during = {s for t, s in states if 1.0 <= t < 1.2}
    after = [s for t, s in states if t >= 1.2]
    assert during == {TrackState.COASTING}
    assert after[0] is TrackState.TRACKING  # no relock needed


def test_long_occlusion_lost_then_reacquired_within_1s():
    states = _states(_pipeline(OccludedSource(1.0, 1.5, **_kwargs())))
    assert TrackState.LOST in {s for t, s in states if 1.0 <= t < 1.5}
    t_back = next(t for t, s in states if t >= 1.5 and s is TrackState.TRACKING)
    assert t_back - 1.5 < 1.0  # PRD 7: re-acquire < 1 s


def _kwargs():
    return dict(width=W, height=720, fps=30, velocity=(150.0, 0.0), start=(100, 360, 0), realtime=False,
                max_frames=75, seed=4)


def test_record_and_replay_reproduce_timestamps(tmp_path):
    src = _synthetic(max_frames=20, image_noise=0.0)
    rec = Recorder(tmp_path / "clip", fps=30)
    original = []
    while (f := src.read()) is not None:
        rec.write(f)
        original.append((f.frame_id, f.t_exposure))
    rec.close()

    replay = VideoFileSource(str(tmp_path / "clip.mp4"))
    replayed = []
    while (f := replay.read()) is not None:
        replayed.append((f.frame_id, f.t_exposure))
    assert not replay.is_live
    assert [i for i, _ in replayed] == [i for i, _ in original]
    assert [t for _, t in replayed] == pytest.approx([t for _, t in original], abs=1e-6)


def test_looping_video_keeps_time_rising(tmp_path):
    import cv2
    import numpy as np
    from ctrack.camera.video_file import VideoFileSource
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for _ in range(3):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    cam = VideoFileSource(str(p), loop=True)
    frames = [cam.read() for _ in range(8)]
    ts = [f.t_exposure for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:])), ts
    assert [f.frame_id for f in frames] == list(range(8))
