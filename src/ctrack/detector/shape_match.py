"""Template-matching pose detector (P2-6, ADR-009).

Teach-in: a reference image (crop of the part, optional mask). The taught orientation defines
theta = 0 and the template centre ((w-1)/2, (h-1)/2, pixel-centre coordinates) is the pose origin. Detection searches all rotations with
masked normalized cross-correlation, then refines position and angle to sub-pixel / sub-degree.

Pose convention: theta = rotation of the part relative to the taught image, same sense as the other
detectors (image coordinates, y down). Fine stage steps `fine_step_deg`; the final angle is a parabola
fit over neighbouring scores, so accuracy is well below one step.

Search strategy:
  * global: whole image at reduced resolution over all angles (slow, used for acquisition / after loss)
  * local:  full resolution around the last pose over a limited angle range (fast, used while tracking)
Both finish with a fine stage at full resolution around the coarse optimum.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..registry import DETECTORS
from ..types import Detection, Frame, wrap_angle
from .base import Detector


def _rotate_template(tpl: np.ndarray, mask: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by object angle theta (image convention: y down, positive = clockwise on screen).

    The result is cropped to the rotated bounding box, symmetric about the template centre, with odd
    side lengths, so the centre stays on a pixel: centre = (w - 1) / 2.
    """
    h, w = tpl.shape
    diag = int(math.ceil(math.hypot(w, h))) | 1
    canvas = np.zeros((diag, diag), np.float32)
    cmask = np.zeros((diag, diag), np.uint8)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    canvas[oy:oy + h, ox:ox + w] = tpl
    cmask[oy:oy + h, ox:ox + w] = mask
    c = (diag - 1) / 2.0
    # Rotate about the template's own centre ((w-1)/2, (h-1)/2 in pixel-centre coordinates), then shift
    # that centre onto the canvas centre. Without this the pose origin would move by half a pixel
    # depending on whether the template sides are odd or even.
    tcx, tcy = ox + (w - 1) / 2.0, oy + (h - 1) / 2.0
    # getRotationMatrix2D rotates counter-clockwise on screen, theta is clockwise-positive.
    M = cv2.getRotationMatrix2D((tcx, tcy), -math.degrees(theta), 1.0)
    M[0, 2] += c - tcx
    M[1, 2] += c - tcy
    rt = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR)
    rm = cv2.warpAffine(cmask, M, (diag, diag), flags=cv2.INTER_NEAREST)
    cs, sn = abs(math.cos(theta)), abs(math.sin(theta))
    bw = int(math.ceil(w * cs + h * sn)) | 1
    bh = int(math.ceil(w * sn + h * cs)) | 1
    x0, y0 = (diag - bw) // 2, (diag - bh) // 2
    return rt[y0:y0 + bh, x0:x0 + bw], rm[y0:y0 + bh, x0:x0 + bw]


def _parabola_peak(left: float, mid: float, right: float) -> float:
    """Sub-sample offset in [-0.5, 0.5] of the vertex through three equally spaced samples."""
    denom = left - 2.0 * mid + right
    return 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


@DETECTORS.register("shape_match")
class ShapeMatchDetector(Detector):
    def __init__(self, template: str, mask: str | None = None, min_score: float = 0.85,
                 downscale: int = 4, angle_step_deg: float = 6.0, fine_step_deg: float = 2.0,
                 local_margin_px: float = 100.0, local_angle_range_deg: float = 18.0,
                 local_downscale: int = 4, global_interval: int = 3) -> None:
        tpl = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            raise FileNotFoundError(f"Template image not found: {template}. Create one with tools/teach.py")
        self.template = tpl.astype(np.float32)
        if mask is not None:
            m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != tpl.shape:
                raise ValueError(f"Mask '{mask}' missing or not the same size as the template")
            self.mask = (m > 127).astype(np.uint8) * 255
        else:
            self.mask = np.full(tpl.shape, 255, np.uint8)
        self.min_score = min_score
        self.downscale = max(int(downscale), 1)
        self.angle_step = math.radians(angle_step_deg)
        self.fine_step = math.radians(fine_step_deg)
        self.local_margin = local_margin_px
        self._half_diag = math.hypot(*tpl.shape) / 2.0
        self.local_range = math.radians(local_angle_range_deg)
        self.local_downscale = max(int(local_downscale), 1)
        self.global_interval = max(int(global_interval), 1)
        self._since_global = self.global_interval  # run a global search on the very first frame
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._last: tuple[float, float, float] | None = None  # x, y, theta

    # -- template variants --------------------------------------------------------------------
    def _variant(self, scale: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
        key = (scale, round(math.degrees(theta) * 20))  # 0.05 deg buckets
        if key not in self._cache:
            tpl, mask = self.template, self.mask
            if scale > 1:
                size = (max(tpl.shape[1] // scale, 3), max(tpl.shape[0] // scale, 3))
                tpl = cv2.resize(tpl, size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            self._cache[key] = _rotate_template(tpl, mask, theta)
            if len(self._cache) > 4000:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    # -- matching -----------------------------------------------------------------------------
    @staticmethod
    def _match(img: np.ndarray, tpl: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if img.shape[0] < tpl.shape[0] or img.shape[1] < tpl.shape[1]:
            return None
        return cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED, mask=mask.astype(np.float32))

    def _best(self, img: np.ndarray, scale: int, thetas: list[float], ox: int = 0, oy: int = 0,
              subpixel: bool = False) -> tuple[float, float, float, float, list[float]] | None:
        """Best (score, cx, cy, theta) over thetas in `img` (a crop whose origin is (ox, oy) in the
        full-resolution frame). Coordinates are returned in full-resolution frame pixels. The last
        element is the per-theta peak scores."""
        best = None
        scores: list[float] = []
        for th in thetas:
            tpl, mask = self._variant(scale, th)
            r = self._match(img, tpl, mask)
            if r is None:
                scores.append(-1.0)
                continue
            r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, mx, _, loc = cv2.minMaxLoc(r)
            scores.append(float(mx))
            if best is None or mx > best[0]:
                dx = dy = 0.0
                if subpixel and 0 < loc[0] < r.shape[1] - 1 and 0 < loc[1] < r.shape[0] - 1:
                    dx = _parabola_peak(r[loc[1], loc[0] - 1], r[loc[1], loc[0]], r[loc[1], loc[0] + 1])
                    dy = _parabola_peak(r[loc[1] - 1, loc[0]], r[loc[1], loc[0]], r[loc[1] + 1, loc[0]])
                cx = (loc[0] + dx + (tpl.shape[1] - 1) / 2.0) * scale + ox
                cy = (loc[1] + dy + (tpl.shape[0] - 1) / 2.0) * scale + oy
                best = (float(mx), cx, cy, th)
        return None if best is None else (*best, scores)

    def _fine(self, gray: np.ndarray, cx: float, cy: float, theta: float, radius: float,
              half_range: float) -> tuple[float, float, float, float] | None:
        """Refine at full resolution around a coarse estimate."""
        h, w = gray.shape
        x0, y0 = max(int(cx - radius), 0), max(int(cy - radius), 0)
        x1, y1 = min(int(cx + radius), w), min(int(cy + radius), h)
        crop = gray[y0:y1, x0:x1]
        n = int(round(half_range / self.fine_step))
        thetas = [theta + i * self.fine_step for i in range(-n, n + 1)]
        res = self._best(crop, 1, thetas, x0, y0, subpixel=True)
        if res is None:
            return None
        score, bx, by, bt, scores = res
        i = thetas.index(bt)
        if 0 < i < len(scores) - 1:  # sub-degree angle by parabola over neighbouring angle scores
            bt += _parabola_peak(scores[i - 1], scores[i], scores[i + 1]) * self.fine_step
        return score, bx, by, wrap_angle(bt)

    def _global(self, gray: np.ndarray) -> tuple[float, float, float, float] | None:
        d = self.downscale
        small = cv2.resize(gray, (gray.shape[1] // d, gray.shape[0] // d), interpolation=cv2.INTER_AREA) if d > 1 else gray
        n = int(round(2 * math.pi / self.angle_step))
        thetas = [wrap_angle(i * self.angle_step) for i in range(n)]
        res = self._best(small, d, thetas)
        if res is None:
            return None
        _, cx, cy, th, _ = res
        # Crop must hold the rotated template plus the coarse-grid position uncertainty.
        return self._fine(gray, cx, cy, th, self._half_diag + 2 * d + 4, self.angle_step)

    def _local(self, gray: np.ndarray) -> tuple[float, float, float, float] | None:
        assert self._last is not None
        x, y, th = self._last
        d = self.local_downscale
        m = self.local_margin + self._half_diag
        h, w = gray.shape
        x0, y0 = max(int(x - m), 0), max(int(y - m), 0)
        crop = gray[y0:min(int(y + m), h), x0:min(int(x + m), w)]
        if d > 1:
            crop = cv2.resize(crop, (crop.shape[1] // d, crop.shape[0] // d), interpolation=cv2.INTER_AREA)
        n = int(round(self.local_range / self.angle_step))
        thetas = [th + i * self.angle_step for i in range(-n, n + 1)]
        res = self._best(crop, d, thetas, x0, y0)
        if res is None:
            return None
        _, cx, cy, bt, _ = res
        return self._fine(gray, cx, cy, bt, self._half_diag + 2 * d + 4, self.angle_step)

    # -- Detector -----------------------------------------------------------------------------
    def detect(self, frame: Frame) -> Detection | None:
        gray = frame.image if frame.image.ndim == 2 else cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        result = self._local(gray) if self._last is not None else None
        if result is None or result[0] < self.min_score:
            # The global search is slow (~250 ms at 1280x720); while nothing is found, run it only
            # every `global_interval` frames so the frame rate does not collapse.
            self._since_global += 1
            result = None
            if self._since_global >= self.global_interval:
                self._since_global = 0
                result = self._global(gray)
        if result is None or result[0] < self.min_score:
            self._last = None
            return None
        score, x, y, theta = result
        self._last = (x, y, theta)
        return Detection(x, y, theta, float(np.clip(score, 0.0, 1.0)))
