"""Video file playback and recording (P1-1)."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from ..registry import CAMERAS
from ..types import Frame
from .base import CameraSource


def timestamps_path(video_path: str | Path) -> Path:
    return Path(video_path).with_suffix(".csv")


@CAMERAS.register("video_file")
class VideoFileSource(CameraSource):
    """Replays a video. If a sibling `<name>.csv` from Recorder exists, its timestamps are used
    verbatim, so filter-parameter comparisons are reproducible. Otherwise timestamps are
    synthesized from the container fps.

    realtime: pace playback to the recorded timing (for the live demo); otherwise run as fast as
    possible (for batch evaluation).
    """

    is_live = False

    def __init__(self, path: str, realtime: bool = False, loop: bool = False) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file {self.path}")
        self.realtime = realtime
        self.loop = loop
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.timestamps: list[tuple[int, float]] | None = None
        ts_file = timestamps_path(self.path)
        if ts_file.exists():
            with open(ts_file) as f:
                self.timestamps = [(int(r["frame_id"]), float(r["t_exposure"])) for r in csv.DictReader(f)]
        self._index = 0
        self._wall_start: float | None = None

    def _timestamp(self, index: int) -> tuple[int, float]:
        if self.timestamps is not None and index < len(self.timestamps):
            return self.timestamps[index]
        return index, index / self.fps

    def read(self) -> Frame | None:
        ok, image = self.cap.read()
        if not ok:
            if not self.loop or self._index == 0:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._index = 0
            self._wall_start = None
            ok, image = self.cap.read()
            if not ok:
                return None
        frame_id, t = self._timestamp(self._index)
        if self.realtime:
            t0 = self._timestamp(0)[1]
            if self._wall_start is None:
                self._wall_start = time.time()
            delay = self._wall_start + (t - t0) - time.time()
            if delay > 0:
                time.sleep(delay)
        self._index += 1
        return Frame(image=image, t_exposure=t, frame_id=frame_id)

    def close(self) -> None:
        self.cap.release()


class Recorder:
    """Writes frames to `<path>.mp4` and their timestamps to `<path>.csv`."""

    def __init__(self, path: str | Path, fps: float = 30.0) -> None:
        self.video_path = Path(path).with_suffix(".mp4")
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._csv_file = open(timestamps_path(self.video_path), "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame_id", "t_exposure"])

    def write(self, frame: Frame) -> None:
        if self._writer is None:
            h, w = frame.image.shape[:2]
            self._writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self._writer.write(frame.image)
        self._csv.writerow([frame.frame_id, f"{frame.t_exposure:.6f}"])

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._csv_file.close()
