from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Detection, Frame


class Detector(ABC):
    """Returns a generic planar 3-DoF pose, not tied to a specific object type (P2-2)."""

    @abstractmethod
    def detect(self, frame: Frame) -> Detection | None:
        """Return the detection, or None if nothing found / below confidence threshold."""

    def close(self) -> None:
        pass
