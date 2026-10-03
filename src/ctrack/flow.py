"""Frame-to-frame motion of the tracked object between (slow) detections: sparse optical flow + similarity fit.

The learned detector delivers a precise pose every 30-150 ms. In between this module follows the object's own
texture with pyramidal Lucas-Kanade on points inside its last outline, fits a rotation+translation (RANSAC) to
their motion and advances (x, y, theta) by it. It costs ~1-3 ms per frame, gives theta (a box tracker cannot) and
needs no model. It drifts, so every detection corrects it (`correct`), and it refuses to run longer than `max_age_s`
without a detection or when too few points agree (`step` then returns None).
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

WORK_WIDTH = 640           # frames are tracked at this width (positions are scaled back)
MAX_POINTS = 150
MIN_POINTS = 12


class FlowTracker:
    def __init__(self, max_age_s: float = 0.5, fb_tolerance_px: float = 1.0, max_step_px: float = 1e9) -> None:
        self.max_age_s = max_age_s
        self.gate_px, self.gate_rel = 8.0, 0.6     # allowed deviation from the expected shift: max(gate_px, gate_rel * |expected|)
        self.max_step_px = max_step_px            # motion blur makes flow lie on fast frames: beyond this per-frame shift, give up
        self.fb_tol = fb_tolerance_px
        self.active = False
        self._gray: np.ndarray | None = None
        self._pts: np.ndarray | None = None          # (n, 1, 2) float32, work-scale pixels
        self._poly: np.ndarray | None = None         # outline, work-scale pixels
        self._pose = np.zeros(3)                      # x, y (work scale), theta
        self._hist: deque[tuple[float, float, float, float]] = deque(maxlen=90)   # t, x, y, theta
        self._t_det = 0.0
        self._n_seed = 0
        self._scale = 1.0
        self.inliers = 0

    # -- helpers ----------------------------------------------------------------------------
    def _prepare(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        self._scale = WORK_WIDTH / w if w > WORK_WIDTH else 1.0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (round(w * self._scale), round(h * self._scale)), interpolation=cv2.INTER_AREA) if self._scale != 1.0 else g

    def _seed(self, gray: np.ndarray, poly: np.ndarray) -> bool:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))           # stay off the outline: background bleeds in there
        pts = cv2.goodFeaturesToTrack(gray, MAX_POINTS, 0.01, 5, mask=mask)
        if pts is None or len(pts) < MIN_POINTS:
            self._pts, self._n_seed = None, 0
            return False
        self._pts, self._n_seed = pts.astype(np.float32), len(pts)
        return True

    # -- API --------------------------------------------------------------------------------
    def reset(self, bgr: np.ndarray, x: float, y: float, theta: float, contour: np.ndarray, t: float) -> bool:
        """Start (or restart) on a fresh detection of THIS frame."""
        gray = self._prepare(bgr)
        s = self._scale
        poly = np.asarray(contour, np.float32) * s
        self._gray, self._poly = gray, poly
        self._pose = np.array([x * s, y * s, theta])
        self._hist.clear()
        self._hist.append((t, *self._pose))
        self._t_det = t
        self.active = self._seed(gray, poly)
        self.inliers = self._n_seed
        return self.active

    def step(self, bgr: np.ndarray, t: float, expected_step: tuple[float, float] | None = None) -> tuple[float, float, float] | None:
        """Advance to the next frame; the pose (x, y, theta) in full-resolution pixels, or None when unreliable.

        expected_step: the shift (px) the motion model predicts for this frame. On fast, blurred frames flow tends to agree
        on 'nothing moved' (static background points win the vote), which would tell the filter the object stopped; a result
        that disagrees strongly with the expectation is therefore rejected."""
        if not self.active or self._gray is None or self._pts is None:
            return None
        if t - self._t_det > self.max_age_s:
            self.active = False
            return None
        gray = self._prepare(bgr)
        p0 = self._pts
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._gray, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - p0).reshape(-1, 2), axis=1) < self.fb_tol)
        if ok.sum() < MIN_POINTS:
            self.active = False
            return None
        a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None or int(inl.sum()) < max(MIN_POINTS, 0.4 * len(a)):
            self.active = False
            return None
        inl = inl.ravel().astype(bool)
        self.inliers = int(inl.sum())
        c = self._pose[:2]
        shift = (M[:, :2] @ c + M[:, 2] - c) / self._scale
        if float(np.hypot(*shift)) > self.max_step_px:
            self.active = False
            return None
        if expected_step is not None:
            ex = np.asarray(expected_step, float)
            if float(np.hypot(*(shift - ex))) > max(self.gate_px, self.gate_rel * float(np.hypot(*ex))):
                self.active = False
                return None
        self._pose = np.array([*(M[:, :2] @ c + M[:, 2]), self._pose[2] + math.atan2(M[1, 0], M[0, 0])])
        self._poly = (self._poly @ M[:, :2].T + M[:, 2]).astype(np.float32)
        self._gray = gray
        self._pts = b[inl].reshape(-1, 1, 2).astype(np.float32)
        if len(self._pts) < 0.5 * self._n_seed:                       # lost many points: pick new ones inside the outline
            self._seed(gray, self._poly)
            if self._pts is None:
                self.active = False
                return None
        self._hist.append((t, *self._pose))
        s = self._scale
        return float(self._pose[0] / s), float(self._pose[1] / s), float(self._pose[2])

    def correct(self, bgr_now: np.ndarray, t_det: float, x: float, y: float, theta: float, contour: np.ndarray,
                t_now: float) -> None:
        """A detection made on the frame at t_det arrived now (frame at t_now): re-anchor and carry it forward
        by the motion the flow measured since t_det. Starts the flow if it was not active."""
        if not self.active or not self._hist:
            self.reset(bgr_now, x, y, theta, contour, t_now)
            return
        hist = np.array(self._hist)
        k = int(np.argmin(np.abs(hist[:, 0] - t_det)))
        if abs(hist[k, 0] - t_det) > 0.1:                                # no matching history: plain restart
            self.reset(bgr_now, x, y, theta, contour, t_now)
            return
        s = self._scale
        _, kx, ky, kth = hist[k]
        nx, ny, nth = self._pose
        d = nth - kth
        rot = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
        det_xy = np.array([x * s, y * s])
        new_xy = rot @ (det_xy - [kx, ky]) + [nx, ny]
        poly = (np.asarray(contour, np.float32) * s - [kx, ky]) @ rot.T + [nx, ny]
        gray = self._gray if self._gray is not None else self._prepare(bgr_now)
        new_pose = np.array([*new_xy, theta + d])
        offset = new_pose - self._pose
        self._hist = deque(((t, *(np.array([hx, hy, hth]) + offset)) for t, hx, hy, hth in self._hist), maxlen=90)
        self._pose = new_pose
        self._poly = poly.astype(np.float32)
        self._t_det = t_det
        self.active = self._seed(gray, self._poly)
        self.inliers = self._n_seed
