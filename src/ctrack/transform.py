"""Image -> target coordinate transforms. Stage 2 adds ConveyorPlaneTransform (mm, belt frame)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .registry import TRANSFORMS
from .types import Pose, Velocity


class Transform(ABC):
    unit: str
    frame: str

    @abstractmethod
    def to_target(self, pose: Pose, velocity: Velocity | None = None) -> tuple[Pose, Velocity | None]:
        """Map a pose (and its velocity) from image pixels into the target frame."""


@TRANSFORMS.register("identity")
class IdentityTransform(Transform):
    unit = "px"
    frame = "image"

    def to_target(self, pose: Pose, velocity: Velocity | None = None) -> tuple[Pose, Velocity | None]:
        return pose, velocity
