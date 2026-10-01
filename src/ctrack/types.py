"""Core data types shared by all pipeline stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

MESSAGE_VERSION = 1


def wrap_angle(a: float) -> float:
    """Normalize an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "theta": round(self.theta, 5)}

    def distance(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_diff(self, other: Pose) -> float:
        """Absolute angular difference in radians, wrap-aware."""
        return abs(wrap_angle(self.theta - other.theta))


@dataclass(frozen=True)
class Velocity:
    vx: float
    vy: float
    omega: float

    def to_dict(self) -> dict[str, float]:
        return {"vx": round(self.vx, 3), "vy": round(self.vy, 3), "omega": round(self.omega, 5)}


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    t_exposure: float
    frame_id: int
    ground_truth: Pose | None = None


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    theta: float
    confidence: float
    keypoints: np.ndarray | None = field(default=None, compare=False)
    contour: np.ndarray | None = field(default=None, compare=False)  # (n, 2) outline in frame pixels, for the overlay

    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.theta)


class TrackState(str, Enum):
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    COASTING = "COASTING"
    LOST = "LOST"


@dataclass(frozen=True)
class Message:
    """Stream message, see PRD 5.4."""

    seq: int
    state: TrackState
    t_exposure: float
    t_sent: float
    pose: Pose | None
    velocity: Velocity | None
    unit: str
    frame: str
    confidence: float
    predicted: Pose | None = None
    t_predicted: float | None = None

    def __post_init__(self) -> None:
        # P0-6: a TRACKING message must always carry a valid pose.
        if self.state is TrackState.TRACKING and (
            self.pose is None or not all(map(math.isfinite, (self.pose.x, self.pose.y, self.pose.theta)))
        ):
            raise ValueError("TRACKING message requires a finite pose")
        if (self.pose is None) != (self.velocity is None):
            raise ValueError("pose and velocity must be set together")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "v": MESSAGE_VERSION,
            "seq": self.seq,
            "state": self.state.value,
            "t_exposure": round(self.t_exposure, 6),
            "t_sent": round(self.t_sent, 6),
            "pose": self.pose.to_dict() if self.pose else None,
            "velocity": self.velocity.to_dict() if self.velocity else None,
            "unit": self.unit,
            "frame": self.frame,
            "confidence": round(self.confidence, 3),
        }
        if self.predicted is not None and self.t_predicted is not None:
            d["predicted"] = {"t": round(self.t_predicted, 6), **self.predicted.to_dict()}
        return d
