"""Hand detector based on MediaPipe Tasks HandLandmarker (ADR-003)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..registry import DETECTORS
from ..types import Detection, Frame, Pose, wrap_angle
from .base import Detector

WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17


def hand_pose_from_landmarks(points: np.ndarray) -> Pose:
    """Rigid hand frame from landmarks 0, 5, 17 in pixel coordinates (PRD 5.3).

    Origin: centroid of the three points. x-axis: wrist -> midpoint(5, 17).
    theta: angle of the x-axis to the image horizontal, in (-pi, pi]. Fingers are ignored.
    """
    wrist, index_mcp, pinky_mcp = points[WRIST, :2], points[INDEX_MCP, :2], points[PINKY_MCP, :2]
    origin = (wrist + index_mcp + pinky_mcp) / 3.0
    axis = (index_mcp + pinky_mcp) / 2.0 - wrist
    theta = wrap_angle(math.atan2(axis[1], axis[0]))
    return Pose(float(origin[0]), float(origin[1]), theta)


@DETECTORS.register("hand")
class HandDetector(Detector):
    def __init__(self, min_confidence: float = 0.6, model_path: str = "models/hand_landmarker.task") -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at '{model_path}'. Run: python tools/fetch_model.py"
            )
        self.min_confidence = min_confidence
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame: Frame) -> Detection | None:
        import mediapipe as mp

        h, w = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = max(int(frame.t_exposure * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not result.hand_landmarks:
            return None
        confidence = float(result.handedness[0][0].score) if result.handedness else 1.0
        if confidence < self.min_confidence:
            return None
        points = np.array([[lm.x * w, lm.y * h] for lm in result.hand_landmarks[0]], dtype=np.float64)
        pose = hand_pose_from_landmarks(points)
        return Detection(pose.x, pose.y, pose.theta, confidence, keypoints=points)

    def close(self) -> None:
        self._landmarker.close()
