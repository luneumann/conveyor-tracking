"""Learned object detector (ADR-011): few-shot DINOv2 head + mask -> pose.

Per frame: if the object was seen recently, a square crop around it is classified (fast, ~65 ms); otherwise
(or if the crop result is implausible) the whole frame is classified at reduced resolution (~140 ms). The
probability map becomes a mask; its centroid and principal axis are the pose (theta modulo pi, unwrapped).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..objectmodel import ObjectModel, crop_view, global_view, mask_pose
from ..registry import DETECTORS
from ..types import Detection, Frame
from ..vision.onnx_models import DinoFeatures
from .base import Detector

MAX_MISSES = 5          # frames without result before the crop tracker gives up and searches globally


@DETECTORS.register("learned")
class LearnedObjectDetector(Detector):
    def __init__(self, model_path: str, models_dir: str = "models", min_score: float = 0.6,
                 threshold: float = 0.5, crop_factor: float = 1.8, global_width: int = 448,
                 global_interval: int = 3, view_size: int = 168) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Gelerntes Objekt nicht gefunden: {model_path}")
        self.model = ObjectModel.load(Path(model_path))
        self.dino = DinoFeatures(Path(models_dir))
        self.min_score = min_score
        self.threshold = threshold
        self.crop_factor = crop_factor
        self.global_width = global_width
        self.view_size = view_size          # crop edge fed to the backbone (multiple of 14): speed <-> accuracy
        self.global_interval = max(int(global_interval), 1)
        self._since_global = self.global_interval
        self._last: tuple[float, float, float, float] | None = None   # x, y, theta, long side (px)
        self._misses = 0

    # -- stages -------------------------------------------------------------------------------
    def _crop_search(self, img: np.ndarray) -> Detection | None:
        assert self._last is not None
        x, y, th, long_side = self._last
        side = max(self.crop_factor * long_side, 96.0)
        vs = self.view_size
        view, s = crop_view(img, x, y, side, vs)
        mp = mask_pose(self.model.prob_map(self.dino.extract(view)), (vs, vs), prev_theta=th, threshold=self.threshold)
        if mp is None or mp.score < self.min_score or not self.model.aspect_ok(mp.aspect):
            return None
        size = mp.bbox_long / s
        if not 0.5 < size / long_side < 2.0:      # implausible jump in apparent size -> distrust, search globally
            return None
        contour = np.column_stack([x + (mp.contour[:, 0] - vs / 2) / s, y + (mp.contour[:, 1] - vs / 2) / s])
        return Detection(x + (mp.x - vs / 2) / s, y + (mp.y - vs / 2) / s, mp.theta, mp.score, contour=contour)

    def _global_search(self, img: np.ndarray) -> Detection | None:
        h, w = img.shape[:2]
        view, _ = global_view(img, self.global_width)
        mp = mask_pose(self.model.prob_map(self.dino.extract(view)), (w, h), threshold=self.threshold)
        if mp is None or mp.score < self.min_score or not self.model.aspect_ok(mp.aspect):
            return None
        return Detection(mp.x, mp.y, mp.theta, mp.score, contour=mp.contour)

    # -- Detector -----------------------------------------------------------------------------
    def detect(self, frame: Frame) -> Detection | None:
        img = frame.image
        det = self._crop_search(img) if self._last is not None else None
        if det is None:
            self._since_global += 1
            if self._last is not None and self._misses < MAX_MISSES:
                self._misses += 1                   # brief dropout: keep the crop position, no global search yet
                if self._misses < MAX_MISSES:
                    return None
            if self._since_global >= self.global_interval:
                self._since_global = 0
                det = self._global_search(img)
        if det is None:
            if self._misses >= MAX_MISSES:
                self._last = None
            return None
        self._misses = 0
        xs, ys = det.contour[:, 0], det.contour[:, 1]
        long_side = float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))   # diagonal: rotation-safe size
        self._last = (det.x, det.y, det.theta, long_side / 1.2)
        return det
