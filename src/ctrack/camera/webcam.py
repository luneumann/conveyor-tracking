from __future__ import annotations

import sys
import time

import cv2

from ..registry import CAMERAS
from ..types import Frame
from .base import CameraSource


@CAMERAS.register("webcam")
class WebcamSource(CameraSource):
    """OpenCV webcam with software timestamps (ADR-007).

    t_exposure = wall time right after grab() minus exposure_offset_ms.
    """

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720,
                 exposure_offset_ms: float = 30.0, fps: float | None = None) -> None:
        # Windows: DirectShow opens in seconds, honours the short buffer and works with most USB cameras; MSMF is the fallback.
        self.cap = cv2.VideoCapture(device, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(device)
        if sys.platform == "win32" and not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam {device}. macOS: grant camera access to your terminal app "
                "(System Settings > Privacy & Security > Camera). Windows: Settings > Privacy > Camera must allow desktop apps, "
                "and no other program (Teams, Zoom, browser) may be using the camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Keep the driver queue short so we always process the newest frame.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.offset_s = exposure_offset_ms / 1000.0
        self._frame_id = 0

    def read(self) -> Frame | None:
        if not self.cap.grab():
            return None
        t_exposure = time.time() - self.offset_s
        ok, image = self.cap.retrieve()
        if not ok:
            return None
        frame = Frame(image=image, t_exposure=t_exposure, frame_id=self._frame_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self.cap.release()
