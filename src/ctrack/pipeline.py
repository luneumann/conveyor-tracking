"""UI-free pipeline core: one frame in, one StepResult out (P2-8)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .camera.base import CameraSource
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


class Pipeline:
    def __init__(self, camera: CameraSource, detector: Detector, tracker: Tracker, transform: Transform,
                 publisher: Publisher, include_predicted: bool = True,
                 clock: Callable[[], float] = time.time) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.transform = transform
        self.publisher = publisher
        self.include_predicted = include_predicted
        self.clock = clock
        self.horizon_s = tracker.predictor.horizon_s
        self._seq = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> Pipeline:
        load_builtins()
        parts: dict[str, Any] = {
            "camera": lambda: CAMERAS.create(cfg["camera"]),
            "detector": lambda: DETECTORS.create(cfg["detector"]),
            "tracker": lambda: Tracker(KalmanPredictor(**cfg["predictor"]), **cfg["tracker"]),
            "transform": lambda: TRANSFORMS.create(cfg["transform"]),
            "publisher": lambda: PUBLISHERS.create(
                {k: v for k, v in cfg["output"].items() if k != "include_predicted"}),
        }
        built = {name: overrides[name] if name in overrides else make() for name, make in parts.items()}
        return cls(**built, include_predicted=cfg["output"].get("include_predicted", True))

    def step(self) -> StepResult | None:
        frame = self.camera.read()
        if frame is None:
            return None
        return self.process(frame)

    def process(self, frame: Frame) -> StepResult:
        t_read = self.clock()
        detection = self.detector.detect(frame)
        t = frame.t_exposure
        state = self.tracker.update(detection, t)

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
                          message, published, latency_ms)

    def close(self) -> None:
        self.camera.close()
        self.detector.close()
        self.publisher.close()
