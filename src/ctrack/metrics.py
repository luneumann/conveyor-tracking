"""Per-frame CSV logging, prediction-error matching and live percentiles (P0-8).

Each frame's prediction for t + horizon is resolved once a frame at or after the target time has
arrived: the frame with detection whose t_exposure is nearest to the target is the reference.
Rows are held back until resolved so each CSV row carries the error of its own prediction.
"""

from __future__ import annotations

import csv as csv_module
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import StepResult
from .types import Pose, TrackState

COLUMNS = [
    "frame_id", "t_exposure", "t_read", "t_sent", "latency_ms", "state",
    "x", "y", "theta", "vx", "vy", "omega",
    "det_x", "det_y", "det_theta", "confidence",
    "pred_t", "pred_x", "pred_y", "pred_theta",
    "pred_err_px", "pred_err_deg",
    "gt_err_px", "gt_err_deg",
]


def percentile(values: deque[float] | list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


class MetricsLogger:
    def __init__(self, csv: str | None = "logs/run.csv", match_tolerance_ms: float = 25.0, window: int = 300,
                 image_width: int | None = None) -> None:
        self.tolerance_s = match_tolerance_ms / 1000.0
        self.image_width = image_width
        self._file = None
        self._writer = None
        if csv:
            path = Path(csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w", newline="")
            self._writer = csv_module.DictWriter(self._file, fieldnames=COLUMNS)
            self._writer.writeheader()
        self._pending: deque[dict[str, Any]] = deque()
        self._measurements: deque[tuple[float, Pose]] = deque()
        self.latency = deque(maxlen=window)
        self.pred_err_px = deque(maxlen=window)
        self.pred_err_deg = deque(maxlen=window)
        self.gt_err_px = deque(maxlen=window)
        self._frame_times = deque(maxlen=60)
        self.rows_written = 0
        self.all_pred_err_px: list[float] = []  # full run, for summaries/tests

    def log(self, step: StepResult) -> None:
        t = step.frame.t_exposure
        if self.image_width is None:
            self.image_width = step.frame.image.shape[1]
        self._frame_times.append(step.t_read)
        self.latency.append(step.latency_ms)

        row: dict[str, Any] = {k: "" for k in COLUMNS}
        row.update(frame_id=step.frame.frame_id, t_exposure=f"{t:.6f}", t_read=f"{step.t_read:.6f}",
                   t_sent=f"{step.message.t_sent:.6f}", latency_ms=f"{step.latency_ms:.2f}",
                   state=step.state.value)
        if step.pose is not None and step.velocity is not None:
            row.update(x=_f(step.pose.x), y=_f(step.pose.y), theta=_f(step.pose.theta, 5),
                       vx=_f(step.velocity.vx), vy=_f(step.velocity.vy), omega=_f(step.velocity.omega, 5))
        if step.detection is not None:
            d = step.detection
            row.update(det_x=_f(d.x), det_y=_f(d.y), det_theta=_f(d.theta, 5), confidence=_f(d.confidence))
            if step.state is TrackState.TRACKING:
                self._measurements.append((t, d.pose))
        if step.predicted is not None and step.t_predicted is not None:
            p = step.predicted
            row.update(pred_t=f"{step.t_predicted:.6f}", pred_x=_f(p.x), pred_y=_f(p.y), pred_theta=_f(p.theta, 5))
            row["_pred"] = (step.t_predicted, p)
        gt = step.frame.ground_truth
        if gt is not None and step.pose is not None:
            err = gt.distance(step.pose)
            row.update(gt_err_px=_f(err), gt_err_deg=_f(math.degrees(gt.angle_diff(step.pose))))
            self.gt_err_px.append(err)

        self._pending.append(row)
        self._resolve(t)

    def _resolve(self, t_now: float, flush: bool = False) -> None:
        while self._pending:
            row = self._pending[0]
            pred = row.pop("_pred", None)
            if pred is not None:
                t_target, p = pred
                if t_target > t_now and not flush:
                    row["_pred"] = pred  # not resolvable yet; keep order
                    return
                ref = self._nearest_measurement(t_target)
                if ref is not None:
                    err_px = ref.distance(p)
                    err_deg = math.degrees(ref.angle_diff(p))
                    row.update(pred_err_px=_f(err_px), pred_err_deg=_f(err_deg))
                    self.pred_err_px.append(err_px)
                    self.pred_err_deg.append(err_deg)
                    self.all_pred_err_px.append(err_px)
            self._write(self._pending.popleft())
        # Measurements older than anything still pending can no longer be the nearest one.
        while len(self._measurements) > 2 and self._measurements[1][0] < t_now - 2.0:
            self._measurements.popleft()

    def _nearest_measurement(self, t_target: float) -> Pose | None:
        best = min(self._measurements, key=lambda m: abs(m[0] - t_target), default=None)
        if best is None or abs(best[0] - t_target) > self.tolerance_s:
            return None
        return best[1]

    def _write(self, row: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.writerow(row)
        self.rows_written += 1

    def stats(self) -> dict[str, float]:
        fps = math.nan
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else math.nan
        width = self.image_width or math.nan
        return {
            "fps": fps,
            "latency_p50": percentile(self.latency, 50),
            "latency_p95": percentile(self.latency, 95),
            "pred_err_p50": percentile(self.pred_err_px, 50),
            "pred_err_p95": percentile(self.pred_err_px, 95),
            "pred_err_p95_pct": percentile(self.pred_err_px, 95) / width * 100.0,
            "pred_err_deg_p95": percentile(self.pred_err_deg, 95),
            "gt_err_p95": percentile(self.gt_err_px, 95),
        }

    def close(self) -> None:
        self._resolve(math.inf, flush=True)
        if self._file is not None:
            self._file.close()


def _f(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"
