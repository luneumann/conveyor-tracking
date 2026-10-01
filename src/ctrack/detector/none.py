from __future__ import annotations

from ..registry import DETECTORS
from ..types import Detection, Frame
from .base import Detector


@DETECTORS.register("none")
class NoDetector(Detector):
    """Finds nothing: camera preview only (e.g. while the first object is still being taught)."""

    def detect(self, frame: Frame) -> Detection | None:
        return None
