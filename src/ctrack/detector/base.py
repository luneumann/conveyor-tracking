from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Detection, Frame


class Detector(ABC):
    """Returns a generic planar 3-DoF pose, not tied to a specific object type (P2-2)."""

    # Where the tracker expects the object at the next frame's exposure time: (x, y, theta). Set by the pipeline
    # before each detect() when it knows (async mode). Detectors that search locally centre on it; others ignore it.
    hint: tuple[float, float, float] | None = None

    @abstractmethod
    def detect(self, frame: Frame) -> Detection | None:
        """Return the detection, or None if nothing found / below confidence threshold."""

    def close(self) -> None:
        pass
