"""Learned object detector (ADR-011): few-shot DINOv2 head + mask -> pose.

Per frame: if the object was seen recently, a square crop around it is classified (fast, ~65 ms); otherwise
(or if the crop result is implausible) the whole frame is classified at reduced resolution (~140 ms). The
probability map becomes a mask; its centroid and principal axis are the pose (theta modulo pi, unwrapped).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..objectmodel import ObjectModel, crop_view, global_view, mask_pose
from ..registry import DETECTORS
from ..types import Detection, Frame
from ..vision.onnx_models import DinoFeatures
from .base import Detector

FAST_GLOBAL_MS = 45.0   # below this a whole-frame search is cheap enough to run on every frame
REFINE_BUDGET_MS = 80.0   # refining costs more than this -> skip it (checked again every REFINE_RETRY_FRAMES)
REFINE_RETRY_FRAMES = 90
MAX_MISSES = 5          # frames without result before the crop tracker gives up and searches globally


@DETECTORS.register("learned")
class LearnedObjectDetector(Detector):
    def __init__(self, model_path: str, models_dir: str = "models", min_score: float = 0.6,
                 threshold: float = 0.5, crop_factor: float = 1.8, global_width: int = 448,
                 global_interval: int = 3, view_size: int = 168, refine_size: int = 0) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Gelerntes Objekt nicht gefunden: {model_path}")
        self.model = ObjectModel.load(Path(model_path))
        if not (Path(models_dir) / self.model.backbone).exists():
            raise FileNotFoundError(f"Das Objekt wurde mit {self.model.backbone} gelernt; die Datei fehlt in {models_dir}")
        self.dino = DinoFeatures(Path(models_dir), self.model.backbone)
        self.min_score = min_score
        self.threshold = threshold
        self.crop_factor = crop_factor
        self.global_width = global_width
        self.view_size = view_size          # crop edge fed to the backbone (multiple of 14): speed <-> accuracy
        # Two-stage accuracy mode: the object is FOUND and followed with the small view (robust, every frame),
        # then its mask is re-computed from a larger view of the same spot (finer outline). Skipped while too slow.
        self.refine_size = refine_size
        self.force_refine = False             # tests / benchmarks: refine even without the accelerator
        self._refine_ms = 0.0
        self._since_refine_try = 0
        self.global_interval = max(int(global_interval), 1)
        self._since_global = self.global_interval
        self._global_ms = 1e9                 # duration of the last whole-frame search (unknown at start: assume slow)
        self.warm_up()
        self._last: tuple[float, float, float, float] | None = None   # x, y, theta, long side (px)
        self._misses = 0

    def warm_up(self) -> None:
        """Prepare the accelerator for the shapes in use now, so no model loads (multi-second GIL stalls) happen mid-run."""
        shapes = [(self.view_size, self.view_size), (round(self.global_width * 9 / 16 / 14) * 14, self.global_width // 14 * 14)]
        if self.refine_size:
            shapes.append((self.refine_size, self.refine_size))
        self.dino.warm_up(shapes)

    def configure(self, view_size: int, refine_size: int, global_width: int) -> None:
        self.view_size, self.refine_size, self.global_width = view_size, refine_size, global_width
        self.warm_up()

    # -- stages -------------------------------------------------------------------------------
    def _crop_search(self, img: np.ndarray, at: tuple[float, float, float, float] | None = None,
                     vs: int | None = None) -> Detection | None:
        """Classify a square crop around `at` (x, y, theta, long side; default: the last detection)."""
        at = at or self._last
        assert at is not None
        x, y, th, long_side = at
        side = max(self.crop_factor * long_side, 96.0)
        vs = vs or self.view_size
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

    def _refine(self, img: np.ndarray, det: Detection) -> Detection:
        """Finer outline from a larger view at the found position; the coarse result stays if refining fails or is too slow."""
        if not self.refine_size or self.refine_size <= self.view_size:
            return det
        if not (self.dino.accelerated or self.force_refine):   # CPU only: the larger view is too slow to be worth a try
            return det
        if self._refine_ms > REFINE_BUDGET_MS:                # e.g. CPU only: a slow measurement is worse than a coarse one
            self._since_refine_try += 1
            if self._since_refine_try < REFINE_RETRY_FRAMES:   # re-check now and then (the accelerator may be ready by now)
                return det
        self._since_refine_try = 0
        xs, ys = det.contour[:, 0], det.contour[:, 1]
        long_side = float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min())) / 1.2
        t0 = time.perf_counter()
        fine = self._crop_search(img, (det.x, det.y, det.theta, long_side), self.refine_size)
        self._refine_ms = (time.perf_counter() - t0) * 1000.0
        if fine is None or self._refine_ms > REFINE_BUDGET_MS:
            return det
        return fine

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
            # Throttle only while the search is slower than a camera frame (CPU: ~150 ms); on the accelerator
            # (~27 ms) every frame may search, which cuts the time to re-find an object.
            interval = 1 if self._global_ms < FAST_GLOBAL_MS else self.global_interval
            if self._since_global >= interval:
                self._since_global = 0
                t0 = time.perf_counter()
                det = self._global_search(img)
                self._global_ms = (time.perf_counter() - t0) * 1000.0
        if det is None:
            if self._misses >= MAX_MISSES:
                self._last = None
            return None
        self._misses = 0
        det = self._refine(img, det)
        xs, ys = det.contour[:, 0], det.contour[:, 1]
        long_side = float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))   # diagonal: rotation-safe size
        self._last = (det.x, det.y, det.theta, long_side / 1.2)
        return det
