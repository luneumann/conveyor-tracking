"""Run a slow detector in a worker thread, decoupled from the camera / output rate.

The main loop submits every frame without waiting; the worker always processes the NEWEST submitted frame
(older pending frames are dropped, like a camera with a one-frame buffer) and publishes results with the
exposure time of the frame they belong to. Results therefore arrive in capture order, which lets the Kalman
filter fuse them as ordinary (merely late) measurements: it only ever advances to a measurement's own time and
predicts read-only beyond it, so no out-of-sequence handling (roll-back) is needed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from ..types import Detection, Frame
from .base import Detector


@dataclass(frozen=True)
class DetectionResult:
    t_exposure: float          # exposure time of the frame this result belongs to
    frame_id: int
    detection: Detection | None
    t_done: float              # wall time when the result became available
    compute_ms: float


class AsyncDetector:
    def __init__(self, inner: Detector) -> None:
        self.inner = inner
        self._cv = threading.Condition()
        self._pending: Frame | None = None
        self._results: deque[DetectionResult] = deque(maxlen=64)
        self._error: BaseException | None = None
        self._stop = False
        self.dropped = 0
        self.mean_compute_ms = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="ctrack-detector")
        self._thread.start()

    def submit(self, frame: Frame) -> None:
        """Hand over a frame (non-blocking). A frame still waiting is replaced by this newer one."""
        with self._cv:
            if self._pending is not None:
                self.dropped += 1
            self._pending = frame
            self._cv.notify()

    def poll(self) -> list[DetectionResult]:
        """All results finished since the last call, oldest first. Re-raises a worker exception."""
        with self._cv:
            if self._error is not None:
                err, self._error = self._error, None
                raise err
            out = list(self._results)
            self._results.clear()
            return out

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending is not None or self._stop)
                if self._stop:
                    return
                frame, self._pending = self._pending, None
            t0 = time.perf_counter()
            try:
                det = self.inner.detect(frame)
            except BaseException as e:  # surfaced to the main loop by poll()
                with self._cv:
                    self._error = e
                return
            ms = (time.perf_counter() - t0) * 1000.0
            self.mean_compute_ms = ms if self.mean_compute_ms == 0 else 0.9 * self.mean_compute_ms + 0.1 * ms
            with self._cv:
                self._results.append(DetectionResult(frame.t_exposure, frame.frame_id, det, time.time(), ms))

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)
        self.inner.close()
