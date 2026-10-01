"""Synthetic conveyor source with ground truth (P1-2, ADR-008)."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from ..registry import CAMERAS
from ..types import Frame, Pose, wrap_angle
from .base import CameraSource

BODY_COLOR = (230, 230, 230)  # BGR, light grey body
MARKER_COLOR = (40, 40, 220)  # BGR, red direction dot


def render_object(image: np.ndarray, pose: Pose, length: float, width: float) -> None:
    """Draw the synthetic part: a rectangle body with a red dot towards +x of its pose."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    hl, hw = length / 2, width / 2
    corners = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    rot = np.array([[c, -s], [s, c]])
    pts = (corners @ rot.T + [pose.x, pose.y]).astype(np.float32)
    cv2.fillPoly(image, [np.round(pts * 16).astype(np.int32)], BODY_COLOR, lineType=cv2.LINE_AA, shift=4)
    dot = (pose.x + c * hl * 0.6, pose.y + s * hl * 0.6)
    cv2.circle(image, (round(dot[0] * 16), round(dot[1] * 16)), round(width * 0.25 * 16), MARKER_COLOR,
               -1, lineType=cv2.LINE_AA, shift=4)


@CAMERAS.register("synthetic")
class SyntheticConveyorSource(CameraSource):
    """Object moving at constant velocity across the image, wrapping around horizontally.

    realtime=True paces frames to `fps` on the wall clock (live demo, latency is meaningful).
    realtime=False produces frames as fast as possible on a simulated clock (tests, tuning).
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0,
                 velocity: tuple[float, float] = (200.0, 0.0), omega: float = 0.0,
                 start: tuple[float, float, float] = (100.0, 360.0, 0.0),
                 object_size: tuple[float, float] = (160.0, 70.0), image_noise: float = 6.0,
                 jitter_s: float = 0.0, max_frames: int | None = None, realtime: bool = True,
                 seed: int = 0) -> None:
        self.size = (width, height)
        self.fps = fps
        self.velocity = velocity
        self.omega = omega
        self.start = start
        self.object_size = object_size
        self.image_noise = image_noise
        self.jitter_s = jitter_s
        self.max_frames = max_frames
        self.realtime = realtime
        self.is_live = realtime
        self.rng = np.random.default_rng(seed)
        self._frame_id = 0
        self._t0 = time.time()
        self._t_sim = 0.0

    def pose_at(self, t_rel: float) -> Pose:
        w = self.size[0]
        x = self.start[0] + self.velocity[0] * t_rel
        # Wrap with margin so the object fully leaves before re-entering.
        margin = self.object_size[0]
        x = (x + margin) % (w + 2 * margin) - margin
        y = self.start[1] + self.velocity[1] * t_rel
        return Pose(x, y, wrap_angle(self.start[2] + self.omega * t_rel))

    def read(self) -> Frame | None:
        if self.max_frames is not None and self._frame_id >= self.max_frames:
            return None
        t_rel = self._frame_id / self.fps + (self.rng.normal(0, self.jitter_s) if self.jitter_s else 0.0)
        if self.realtime:
            delay = self._t0 + t_rel - time.time()
            if delay > 0:
                time.sleep(delay)
        pose = self.pose_at(t_rel)
        w, h = self.size
        image = np.full((h, w, 3), 35, dtype=np.uint8)
        render_object(image, pose, *self.object_size)
        if self.image_noise:
            noise = self.rng.normal(0, self.image_noise, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frame = Frame(image=image, t_exposure=self._t0 + t_rel, frame_id=self._frame_id, ground_truth=pose)
        self._frame_id += 1
        return frame
