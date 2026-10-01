"""Colour-threshold detector for the synthetic conveyor source (ADR-008). Not for real scenes."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..registry import DETECTORS
from ..types import Detection, Frame, wrap_angle
from .base import Detector


@DETECTORS.register("blob")
class MarkerBlobDetector(Detector):
    """Bright body + red direction dot. Pose = body centroid, theta = centroid -> dot."""

    def __init__(self, min_area: float = 500.0, min_confidence: float = 0.5, expected_ratio: float = 0.086) -> None:
        self.min_area = min_area
        # Dot/body area ratio of the synthetic part: pi * (0.25 w)^2 / (l * w) for the default 160x70 part.
        self.expected_ratio = expected_ratio
        self.min_confidence = min_confidence

    def detect(self, frame: Frame) -> Detection | None:
        img = frame.image
        b, g, r = (img[..., i].astype(np.int16) for i in range(3))
        red = ((r > 150) & (g < 110) & (b < 110)).astype(np.uint8)
        obj = ((img.max(axis=2) > 120)).astype(np.uint8) | red

        body = _largest_blob(obj, self.min_area)
        if body is None:
            return None
        cx, cy, area = body
        dot = _largest_blob(red, self.min_area * 0.02)
        if dot is None:
            return None
        dx, dy, dot_area = dot
        theta = wrap_angle(math.atan2(dy - cy, dx - cx))
        # Confidence: penalise implausible dot/body ratios (e.g. object half out of view).
        ratio = dot_area / area
        confidence = float(np.clip(1.0 - abs(ratio - self.expected_ratio) / self.expected_ratio, 0.0, 1.0))
        if confidence < self.min_confidence:
            return None
        return Detection(cx, cy, theta, confidence)


def _largest_blob(mask: np.ndarray, min_area: float) -> tuple[float, float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] < min_area:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"]
