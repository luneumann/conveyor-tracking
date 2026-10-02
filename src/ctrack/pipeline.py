"""UI-free pipeline core: one frame in, one StepResult out (P2-8)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .camera.base import CameraSource
from .detector.async_detector import AsyncDetector
from .detector.base import Detector
from .predictor import KalmanPredictor
from .publisher.base import Publisher
from .registry import CAMERAS, DETECTORS, PUBLISHERS, TRANSFORMS, load_builtins
from .tracker import Tracker
from .transform import Transform
from .types import Detection, Frame, Message, Pose, TrackState, Velocity


@dataclass(frozen=True)
class StepResult:
    frame: Frame
    t_read: float
    detection: Detection | None
    state: TrackState
    pose: Pose | None  # image frame, at t_exposure
    velocity: Velocity | None
    predicted: Pose | None  # image frame, at t_predicted
    t_predicted: float | None
    message: Message
    published: bool
    latency_ms: float
    # Asynchronous detection: `detection` may belong to an EARLIER frame than `frame`.
    detection_t: float | None = None      # exposure time of the frame the detection was made on
    detection_new: bool = False           # a fresh result arrived in this step (log it / feed metrics once)
    perception_ms: float | None = None    # age of the newest detection at this frame: t_exposure - detection_t


class Pipeline:
    def __init__(self, camera: CameraSource, detector: Detector, tracker: Tracker, transform: Transform,
                 publisher: Publisher, include_predicted: bool = True,
                 clock: Callable[[], float] = time.time, async_detect: bool = False,
                 detection_max_age_s: float = 0.25) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.transform = transform
        self.publisher = publisher
        self.include_predicted = include_predicted
        self.clock = clock
        self.horizon_s = tracker.predictor.horizon_s
        self._seq = 0
        # Asynchronous detection decouples camera/output rate from the detector (see detector/async_detector.py).
        self._async = AsyncDetector(detector) if async_detect else None
        self.detection_max_age_s = detection_max_age_s
        self._last_det: tuple[float, Detection] | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> Pipeline:
        load_builtins()
        parts: dict[str, Any] = {   # detector first: its start-up (model compilation) must not let camera frames pile up
            "detector": lambda: DETECTORS.create(cfg["detector"]),
            "camera": lambda: CAMERAS.create(cfg["camera"]),
            "tracker": lambda: Tracker(KalmanPredictor(**cfg["predictor"]), **cfg["tracker"]),
            "transform": lambda: TRANSFORMS.create(cfg["transform"]),
            "publisher": lambda: PUBLISHERS.create(
                {k: v for k, v in cfg["output"].items() if k != "include_predicted"}),
        }
        built = {name: overrides[name] if name in overrides else make() for name, make in parts.items()}
        return cls(**built, include_predicted=cfg["output"].get("include_predicted", True),
                   async_detect=bool(cfg.get("pipeline", {}).get("async_detect", False)))

    def step(self) -> StepResult | None:
        frame = self.camera.read()
        if frame is None:
            return None
        return self.process(frame)

    def process(self, frame: Frame) -> StepResult:
        t_read = self.clock()
        t = frame.t_exposure
        detection_t: float | None = None
        detection_new = False
        if self._async is None:
            detection = self.detector.detect(frame)
            state = self.tracker.update(detection, t)
            if detection is not None:
                detection_t, detection_new = t, True
        else:
            self._async.submit(frame)
            for r in self._async.poll():                   # late measurements, fused at their own capture time
                self.tracker.update(r.detection, r.t_exposure)
                if r.detection is not None:
                    self._last_det = (r.t_exposure, r.detection)
                    detection_new = True
            state = self.tracker.advance(t)
            detection = None
            if self._last_det is not None and t - self._last_det[0] <= self.detection_max_age_s:
                detection_t, detection = self._last_det    # newest detection (possibly a few frames old), for the overlay
        perception_ms = (t - detection_t) * 1000.0 if detection_t is not None else None

        pose = velocity = predicted = None
        t_predicted = None
        est = self.tracker.estimate(t)
        if est is not None:
            pose, velocity = est
            t_predicted = t + self.horizon_s
            predicted = self.tracker.predictor.predict(t_predicted)

        out_pose, out_vel = self.transform.to_target(pose, velocity) if pose is not None else (None, None)
        out_pred = self.transform.to_target(predicted)[0] if predicted is not None else None

        t_sent = self.clock()
        message = Message(
            seq=self._seq, state=state, t_exposure=t, t_sent=t_sent, pose=out_pose, velocity=out_vel,
            unit=self.transform.unit, frame=self.transform.frame,
            confidence=detection.confidence if detection else 0.0,
            predicted=out_pred if self.include_predicted else None,
            t_predicted=t_predicted if self.include_predicted else None,
        )
        published = self.publisher.publish(message)
        if published:
            self._seq += 1
        latency_ms = (t_sent - (t if self.camera.is_live else t_read)) * 1000.0
        return StepResult(frame, t_read, detection, state, pose, velocity, predicted, t_predicted,
                          message, published, latency_ms, detection_t, detection_new, perception_ms)

    def close(self) -> None:
        self.camera.close()
        if self._async is not None:
            self._async.close()          # also closes the wrapped detector
        else:
            self.detector.close()
        self.publisher.close()
