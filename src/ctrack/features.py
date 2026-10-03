"""Feature anchor for large objects (docs/OPTIMIERUNG.md, section 15).

The pose of a big object taken from its mask (centroid, principal axis) wobbles with every error along the mask's
long outline. Here the mask only says WHERE the object is; the pose comes from image features (SIFT) found inside it:
reference features taught from the photos are matched in the current frame, a rotation + translation (RANSAC) maps a
fixed ANCHOR point of the object into the frame, and the rotation gives theta. The anchor is the centre of an optional
region the user marked as the dedicated feature, otherwise the mask centroid of the first photo.

Photos show the object from different distances and angles, so the anchor is carried into every photo through the
best feature matches between the photos (the same physical point in each).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .types import Detection, wrap_angle

NORM_SIZE = 600.0          # features are computed with the object scaled to about this diagonal (px): comparable across distances
MIN_INLIERS = 10           # a match with fewer consistent features is not trusted
GOOD_INLIERS = 25          # the last used reference view is kept when it gives at least this many
LARGE_FRACTION = 0.20      # "auto" switches the feature anchor on for objects wider than this share of the image
RATIO = 0.75               # Lowe's ratio test


def _sift(n: int = 1000):
    return cv2.SIFT_create(nfeatures=n)


def _bbox_diag(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    return float(math.hypot(np.ptp(xs), np.ptp(ys))) if len(xs) else 0.0


def _axis(mask: np.ndarray) -> tuple[float, float, float]:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return m["m10"] / m["m00"], m["m01"] / m["m00"], 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])


def _features(bgr: np.ndarray, size: float, mask: np.ndarray | None = None, box: tuple[int, int, int, int] | None = None,
              n: int = 1000) -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT keypoints (full-resolution coordinates) and descriptors of the object scaled to NORM_SIZE.

    box = (x0, y0, x1, y1): only that part of the frame is looked at (coordinates are returned in frame pixels)."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
    s = min(1.0, NORM_SIZE / max(size, 1.0))
    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    m = None if mask is None else mask[y0:y1, x0:x1].astype(np.uint8) * 255
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if m is not None:
            m = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    kp, des = _sift(n).detectAndCompute(gray, m)
    if not kp or des is None:
        return np.zeros((0, 2), np.float32), None
    pts = np.float32([k.pt for k in kp]) / s + np.float32([x0, y0])
    return pts, des


_BF = cv2.BFMatcher(cv2.NORM_L2)


def _match(q_pts: np.ndarray, q_des: np.ndarray | None, r_pts: np.ndarray, r_des: np.ndarray | None):
    """Similarity (2x3) that maps reference points onto query points, and the number of consistent matches."""
    if q_des is None or r_des is None or len(q_des) < 8 or len(r_des) < 8:
        return None
    good = []
    for pair in _BF.knnMatch(q_des, r_des, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            good.append(pair[0])
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([r_pts[g.trainIdx] for g in good])
    dst = np.float32([q_pts[g.queryIdx] for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None or int(inl.sum()) < MIN_INLIERS:
        return None
    return M, int(inl.sum())


@dataclass
class FeatureView:
    pts: np.ndarray            # (n, 2) keypoints in the photo
    des: np.ndarray            # (n, 128)
    anchor: np.ndarray         # (2,) the object's anchor point in this photo
    theta: float               # principal axis of the object's mask in this photo (mod pi)


@dataclass
class FeatureModel:
    views: list[FeatureView]
    large: bool = False        # the taught object is big in the image: "auto" switches the feature anchor on
    region: bool = False       # a dedicated feature region was marked

    def to_arrays(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {"feat_n": np.array(len(self.views)), "feat_large": np.array(self.large),
                                    "feat_region": np.array(self.region)}
        for i, v in enumerate(self.views):
            d[f"feat_pts_{i}"], d[f"feat_des_{i}"] = v.pts, v.des
            d[f"feat_anchor_{i}"], d[f"feat_theta_{i}"] = v.anchor, np.array(v.theta)
        return d

    @classmethod
    def from_arrays(cls, d) -> FeatureModel | None:
        if "feat_n" not in d:
            return None
        views = [FeatureView(d[f"feat_pts_{i}"], d[f"feat_des_{i}"], d[f"feat_anchor_{i}"], float(d[f"feat_theta_{i}"]))
                 for i in range(int(d["feat_n"]))]
        return cls(views, bool(d["feat_large"]), bool(d["feat_region"])) if views else None


def build_feature_model(samples: list[tuple[np.ndarray, np.ndarray]], roi: tuple[float, float, float, float] | None = None
                        ) -> FeatureModel | None:
    """Reference features from the taught photos [(bgr, mask)].

    roi = (x, y, w, h) in the FIRST photo: the dedicated feature region (its centre is the anchor and only features
    around it are kept). None: the anchor is the mask centroid of the first photo and all object features are kept.
    Returns None when no photo besides possibly the first has enough features to be linked."""
    if not samples:
        return None
    raw = []
    for img, mask in samples:
        mask = np.asarray(mask, bool)
        if mask.sum() < 400:
            continue
        size = _bbox_diag(mask)
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        pts, des = _features(img, size, grown)
        cx, cy, th = _axis(mask)
        raw.append(dict(pts=pts, des=des, centroid=np.array([cx, cy]), theta=th, size=size, width=img.shape[1]))
    if not raw or raw[0]["des"] is None or len(raw[0]["pts"]) < MIN_INLIERS:
        return None
    if roi is not None:
        x, y, w, h = roi
        anchor0 = np.array([x + w / 2, y + h / 2], np.float32)
        radius0 = 0.5 * math.hypot(w, h)
    else:
        anchor0, radius0 = raw[0]["centroid"].astype(np.float32), None
    anchors: dict[int, np.ndarray] = {0: anchor0}
    scale: dict[int, float] = {0: 1.0}
    todo = set(range(1, len(raw)))
    while todo:                                        # link each photo to the best-matching already anchored one
        best = None
        for k in todo:
            for j in anchors:
                res = _match(raw[k]["pts"], raw[k]["des"], raw[j]["pts"], raw[j]["des"])
                if res and (best is None or res[1] > best[0]):
                    best = (res[1], k, j, res[0])
        if best is None:
            break
        _, k, j, M = best
        anchors[k] = (M[:, :2] @ anchors[j] + M[:, 2]).astype(np.float32)
        scale[k] = scale[j] * float(math.hypot(M[0, 0], M[1, 0]))
        todo.discard(k)
    views = []
    for k in sorted(anchors):
        r = raw[k]
        pts, des, a = r["pts"], r["des"], anchors[k]
        if radius0 is not None:                       # dedicated feature: only features around the anchor
            keep = np.linalg.norm(pts - a, axis=1) < 1.2 * radius0 * scale[k]
            pts, des = pts[keep], des[keep]
        if des is None or len(pts) < MIN_INLIERS:
            continue
        views.append(FeatureView(pts.astype(np.float32), des.astype(np.float32), a, r["theta"]))
    if not views:
        return None
    large = float(np.mean([r["size"] / r["width"] for r in raw])) > LARGE_FRACTION
    return FeatureModel(views, large, roi is not None)


class FeatureRefiner:
    """Run time: turns a mask-based detection into an anchor pose from matched features."""

    def __init__(self, model: FeatureModel, nfeatures: int = 1000) -> None:
        self.model = model
        self.nfeatures = nfeatures
        self._last_view = 0
        self._off: np.ndarray | None = None      # anchor - mask centroid in the object frame, in units of the object size
        self.last_offset_px = np.zeros(2)        # same, in image pixels (to centre searches on the mask centroid)
        self.last_inliers = 0

    def reset(self) -> None:
        self._off = None
        self.last_offset_px = np.zeros(2)
        self.last_inliers = 0

    @staticmethod
    def _size(det: Detection) -> float:
        c = det.contour
        return float(math.hypot(np.ptp(c[:, 0]), np.ptp(c[:, 1]))) if c is not None and len(c) > 3 else 0.0

    def refine(self, img: np.ndarray, det: Detection) -> Detection:
        """The detection with its pose moved to the feature anchor (same outline). When the features do not agree, the
        anchor is derived from the mask instead, in the same convention, so the pose never jumps between two reference points."""
        size = self._size(det)
        if size < 40:
            return det
        h, w = img.shape[:2]
        pad = 0.15 * size
        xs, ys = det.contour[:, 0], det.contour[:, 1]
        box = (int(max(xs.min() - pad, 0)), int(max(ys.min() - pad, 0)), int(min(xs.max() + pad, w)), int(min(ys.max() + pad, h)))
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            return det
        pts, des = _features(img, size, None, box, self.nfeatures)
        order = [self._last_view] + [i for i in range(len(self.model.views)) if i != self._last_view]
        best = None
        for i in order:
            v = self.model.views[i]
            res = _match(pts, des, v.pts, v.des)
            if res and (best is None or res[1] > best[1]):
                best = (res[0], res[1], i)
            if best and best[1] >= GOOD_INLIERS and i == self._last_view:
                break
        c, s = math.cos(det.theta), math.sin(det.theta)
        if best is None:
            self.last_inliers = 0
            if self._off is None:
                return det
            ax = det.x + size * (c * self._off[0] - s * self._off[1])
            ay = det.y + size * (s * self._off[0] + c * self._off[1])
            self.last_offset_px = np.array([ax - det.x, ay - det.y])
            return replace(det, x=float(ax), y=float(ay))
        M, n, i = best
        self._last_view, self.last_inliers = i, n
        v = self.model.views[i]
        a = M[:, :2] @ v.anchor + M[:, 2]
        th = v.theta + math.atan2(M[1, 0], M[0, 0])
        # theta of a mask is defined modulo pi; keep the branch closest to the detection's
        th = det.theta + (wrap_angle(2.0 * (th - det.theta)) / 2.0)
        d = np.array([a[0] - det.x, a[1] - det.y])
        self._off = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) / size
        self.last_offset_px = d
        return replace(det, x=float(a[0]), y=float(a[1]), theta=float(th))
