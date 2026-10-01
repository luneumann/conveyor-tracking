import csv
import math

import numpy as np
import pytest

from ctrack.metrics import COLUMNS, MetricsLogger
from ctrack.pipeline import StepResult
from ctrack.types import Detection, Frame, Message, Pose, TrackState, Velocity

IMG = np.zeros((10, 1000, 3), np.uint8)


def _step(i, t, det_x=None, pred=None, horizon=0.1, state=TrackState.TRACKING):
    detection = Detection(det_x, 0.0, 0.0, 1.0) if det_x is not None else None
    pose = Pose(det_x if det_x is not None else 0.0, 0.0, 0.0)
    vel = Velocity(0, 0, 0)
    msg = Message(i, state, t, t + 0.01, pose, vel, "px", "image", 1.0)
    return StepResult(Frame(IMG, t, i), t + 0.005, detection, state, pose, vel,
                      Pose(pred, 0.0, 0.0) if pred is not None else None,
                      t + horizon if pred is not None else None, msg, True, 10.0)


def test_prediction_error_matched_to_nearest_frame(tmp_path):
    path = tmp_path / "m.csv"
    m = MetricsLogger(csv=str(path), match_tolerance_ms=25)
    # Frame 0 predicts x=110 at t=0.1. Nearest frame is t=0.09 (10 ms away, x=108) -> error 2 px.
    times = [0.0, 0.04, 0.09, 0.13, 0.17]
    xs = [100, 102, 108, 112, 116]
    for i, (t, x) in enumerate(zip(times, xs)):
        m.log(_step(i, t, det_x=x, pred=110.0 if i == 0 else None))
    m.close()
    rows = list(csv.DictReader(open(path)))
    assert [r["frame_id"] for r in rows] == ["0", "1", "2", "3", "4"]  # order preserved
    assert float(rows[0]["pred_err_px"]) == pytest.approx(2.0)
    assert rows[1]["pred_err_px"] == ""
    assert list(rows[0]) == COLUMNS


def test_rows_held_back_until_resolvable(tmp_path):
    m = MetricsLogger(csv=str(tmp_path / "m.csv"))
    m.log(_step(0, 0.0, det_x=0, pred=10.0))
    m.log(_step(1, 0.05, det_x=5))
    assert m.rows_written == 0  # prediction for t=0.1 not yet resolvable
    m.log(_step(2, 0.1, det_x=10))
    assert m.rows_written == 3
    assert m.all_pred_err_px == [pytest.approx(0.0)]


def test_no_match_outside_tolerance(tmp_path):
    m = MetricsLogger(csv=None, match_tolerance_ms=10)
    m.log(_step(0, 0.0, det_x=0, pred=10.0))
    m.log(_step(1, 0.05, det_x=5))
    m.log(_step(2, 0.2, det_x=20))  # nearest to 0.1 is 0.05 -> 50 ms away
    m.close()
    assert m.all_pred_err_px == []


def test_coasting_frames_are_not_ground_truth(tmp_path):
    m = MetricsLogger(csv=None, match_tolerance_ms=25)
    m.log(_step(0, 0.0, det_x=0, pred=10.0))
    m.log(_step(1, 0.1, det_x=99, state=TrackState.COASTING))  # detection ignored while not fused
    m.close()
    assert m.all_pred_err_px == []


def test_stats():
    m = MetricsLogger(csv=None, image_width=1000)
    for i in range(10):
        m.log(_step(i, i / 30, det_x=i, pred=i + 1.0, horizon=1 / 30))
    m.close()
    s = m.stats()
    assert s["fps"] == pytest.approx(30, rel=0.01)
    assert s["latency_p95"] == pytest.approx(10.0)
    assert s["pred_err_p95"] == pytest.approx(0.0)
    assert math.isnan(s["gt_err_p95"])
