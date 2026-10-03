"""Feature anchor for large objects (docs/ARCHITECTURE.md ADR-014, docs/OPTIMIERUNG.md section 15).

The pose of a big object taken from its mask (centroid, principal axis) wobbles with every error along the mask's long
outline. Here the mask only says WHERE the object is (and bounds the search); the pose comes from image features (SIFT)
found on the object: reference features taught from the photos are matched in the current frame, a rotation + translation
(RANSAC) maps a fixed ANCHOR point of the object into the frame, and the rotation gives theta.

The anchor is the centre of the feature the user marked ("the door handle") on the taught photos - the position the robot
needs - or, without marks, the mask centroid of the first photo. Photos show the object from different distances and
angles, so the anchor (and the marked region's size) is carried into photos without a mark through the best feature
matches between the photos; marks on several photos are checked against each other.

(Matching only the small marked region was tried and measured: a smooth region holds too few keypoints - 0-8 for a door
handle area - and matches between photos fail; matching the whole object finds ~60 consistent features. Template matching
of the small patch was fast but noisy. So the marked region defines the anchor and is DRAWN, and the whole object is matched.)
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .types import Detection, wrap_angle

NORM_SIZE = {"sift": 300.0, "orb": 400.0}   # features are computed with the object scaled to about this diagonal (px), per descriptor type
KINDS = ("sift", "orb")    # "orb": about 1-2 px noise at +9 ms per frame; "sift": about 0.5 px at +24 ms (measured on the belt video)
MIN_INLIERS = 10           # fewer consistent features are not trusted
GOOD_INLIERS = 25          # the last used reference view is kept when it gives at least this many
LARGE_FRACTION = 0.20      # "auto" switches the feature anchor on for objects wider than this share of the image
MAX_ANCHOR_DEV = 0.15      # an anchor farther than this share of the object size from where the mask predicts it is a false match
SEARCH_PAD = 0.15          # search the mask's bounding box grown by this share of the object size
GUESS_PAD = 0.30           # the same for the early search that overlaps with the detector (previous box, grown further)


def _detector(kind: str, n: int):
    return cv2.ORB_create(nfeatures=n) if kind == "orb" else cv2.SIFT_create(nfeatures=n)


def _bbox_diag(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    return float(math.hypot(np.ptp(xs), np.ptp(ys))) if len(xs) else 0.0


def _axis(mask: np.ndarray) -> tuple[float, float, float]:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return m["m10"] / m["m00"], m["m01"] / m["m00"], 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])


def _features(bgr: np.ndarray, size: float, mask: np.ndarray | None = None, box: tuple[int, int, int, int] | None = None,
              n: int = 800, kind: str = "sift") -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT keypoints (frame coordinates) and descriptors of the object scaled to NORM_SIZE.

    box = (x0, y0, x1, y1): only that part of the frame is looked at."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
    s = min(1.0, NORM_SIZE[kind] / max(size, 1.0))
    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    m = None if mask is None else mask[y0:y1, x0:x1].astype(np.uint8) * 255
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if m is not None:
            m = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    kp, des = _detector(kind, n).detectAndCompute(gray, m)
    if not kp or des is None:
        return np.zeros((0, 2), np.float32), None
    pts = np.float32([k.pt for k in kp]) / s + np.float32([x0, y0])
    return pts, des


_BF = {"sift": cv2.BFMatcher(cv2.NORM_L2), "orb": cv2.BFMatcher(cv2.NORM_HAMMING)}
_RATIO = {"sift": 0.75, "orb": 0.80}


def _match(q_pts: np.ndarray, q_des: np.ndarray | None, r_pts: np.ndarray, r_des: np.ndarray | None, kind: str = "sift"):
    """Similarity (2x3) that maps reference points onto query points, and the number of consistent matches."""
    if q_des is None or r_des is None or len(q_des) < 8 or len(r_des) < 8:
        return None
    good = []
    for pair in _BF[kind].knnMatch(q_des, r_des, k=2):
        if len(pair) == 2 and pair[0].distance < _RATIO[kind] * pair[1].distance:
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
    anchor: np.ndarray         # (2,) the anchor point in this photo
    theta: float               # principal axis of the object's mask in this photo (mod pi)
    centroid: np.ndarray       # (2,) mask centroid in this photo
    size: float                # mask bounding-box diagonal in this photo
    roi_wh: np.ndarray | None = None   # (2,) width/height of the marked feature region in this photo (None: no region)

    @property
    def off(self) -> np.ndarray:
        """Anchor minus mask centroid in the object frame (rotated by -theta), in units of the object size."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        d = (self.anchor - self.centroid) / max(self.size, 1.0)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


@dataclass
class FeatureModel:
    views: list[FeatureView]   # SIFT references (precise)
    large: bool = False        # the taught object is big in the image: "auto" switches the feature anchor on
    region: bool = False       # the anchor is a feature region marked by the user (and is drawn)
    mark_dev: float = float("nan")   # median disagreement between marks on different photos (share of the object size)
    fast_views: list[FeatureView] | None = None   # the same photos with ORB features (fast); None in models taught before

    def views_for(self, kind: str) -> list[FeatureView]:
        return self.fast_views if kind == "orb" and self.fast_views else self.views

    def kind_for(self, kind: str) -> str:
        return "orb" if kind == "orb" and self.fast_views else "sift"

    def to_arrays(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {"feat_n": np.array(len(self.views)), "feat_large": np.array(self.large),
                                    "feat_region": np.array(self.region), "feat_markdev": np.array(self.mark_dev)}
        for prefix, views in (("feat", self.views), ("featf", self.fast_views or [])):
            d[f"{prefix}_count"] = np.array(len(views))
            for i, v in enumerate(views):
                d[f"{prefix}_pts_{i}"], d[f"{prefix}_des_{i}"] = v.pts, v.des
                d[f"{prefix}_anchor_{i}"], d[f"{prefix}_theta_{i}"] = v.anchor, np.array(v.theta)
                d[f"{prefix}_centroid_{i}"], d[f"{prefix}_size_{i}"] = v.centroid, np.array(v.size)
                if v.roi_wh is not None:
                    d[f"{prefix}_roiwh_{i}"] = v.roi_wh
        return d

    @classmethod
    def from_arrays(cls, d) -> FeatureModel | None:
        if "feat_n" not in d:
            return None

        def load(prefix: str, n: int) -> list[FeatureView]:
            out = []
            for i in range(n):
                anchor = d[f"{prefix}_anchor_{i}"]
                out.append(FeatureView(d[f"{prefix}_pts_{i}"], d[f"{prefix}_des_{i}"], anchor, float(d[f"{prefix}_theta_{i}"]),
                                       d[f"{prefix}_centroid_{i}"], float(d[f"{prefix}_size_{i}"]),
                                       d[f"{prefix}_roiwh_{i}"] if f"{prefix}_roiwh_{i}" in d else None))
            return out

        try:
            views = load("feat", int(d["feat_n"]))
            fast = load("featf", int(d["featf_count"])) if "featf_count" in d and int(d["featf_count"]) else None
        except KeyError:
            return None     # saved by an earlier prototype layout: behave like a model taught without features
        return cls(views, bool(d["feat_large"]), bool(d["feat_region"]), float(d["feat_markdev"]), fast) if views else None


def build_feature_model(samples: list[tuple[np.ndarray, np.ndarray]],
                        rois: list[tuple[float, float, float, float] | None] | None = None) -> FeatureModel | None:
    """Reference features from the taught photos [(bgr, mask)].

    rois: one optional (x, y, w, h) per photo - the same feature marked on the photos. A marked photo's anchor is its
    region's centre; unmarked photos get the anchor (and region size) carried over by feature matches. Without any mark
    the anchor is the mask centroid of the first photo. Returns None when no photo has enough features."""
    raw = []
    for k, (img, mask) in enumerate(samples):
        mask = np.asarray(mask, bool)
        if mask.sum() < 400:
            continue
        cx, cy, th = _axis(mask)
        size = _bbox_diag(mask)
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        pts, des = _features(img, size, grown, None, 1200, "sift")
        fpts, fdes = _features(img, size, grown, None, 1200, "orb")
        roi = None
        if rois is not None and k < len(rois) and rois[k] is not None:
            x, y, w, h = rois[k]
            H, W = img.shape[:2]
            x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, W), min(y + h, H)
            if x1 - x0 >= 24 and y1 - y0 >= 24:
                roi = (x0, y0, x1, y1)
        raw.append(dict(fpts=fpts, fdes=fdes, pts=pts, des=des, centroid=np.array([cx, cy], np.float32), theta=th, size=size, width=img.shape[1], roi=roi))
    raw = [r for r in raw if r["des"] is not None and len(r["pts"]) >= MIN_INLIERS]
    if not raw:
        return None
    anchors: dict[int, np.ndarray] = {}
    wh: dict[int, np.ndarray] = {}
    for k, r in enumerate(raw):
        if r["roi"] is not None:
            x0, y0, x1, y1 = r["roi"]
            anchors[k] = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
            wh[k] = np.array([x1 - x0, y1 - y0], np.float32)
    region = bool(anchors)
    if not anchors:
        anchors[0] = raw[0]["centroid"]
    marked = set(anchors)
    todo = set(range(len(raw))) - set(anchors)
    while todo:                                        # carry the anchor into every unmarked photo via the best-matching anchored one
        best = None
        for k in todo:
            for j in anchors:
                res = _match(raw[k]["pts"], raw[k]["des"], raw[j]["pts"], raw[j]["des"], "sift")
                if res and (best is None or res[1] > best[0]):
                    best = (res[1], k, j, res[0])
        if best is None:
            break
        _, k, j, M = best
        anchors[k] = (M[:, :2] @ anchors[j] + M[:, 2]).astype(np.float32)
        if j in wh:
            wh[k] = (wh[j] * float(math.hypot(M[0, 0], M[1, 0]))).astype(np.float32)
        todo.discard(k)
    devs = []                                          # do marks on different photos point at the same physical spot?
    for a in marked:
        for b in marked:
            if a < b:
                res = _match(raw[b]["pts"], raw[b]["des"], raw[a]["pts"], raw[a]["des"], "sift")
                if res:
                    pred = res[0][:, :2] @ anchors[a] + res[0][:, 2]
                    devs.append(float(np.linalg.norm(pred - anchors[b])) / max(raw[b]["size"], 1.0))
    def make(pk: str, dk: str, minimum: int) -> list[FeatureView]:
        return [FeatureView(raw[k][pk].astype(np.float32), raw[k][dk], anchors[k], raw[k]["theta"], raw[k]["centroid"], raw[k]["size"], wh.get(k))
                for k in sorted(anchors) if raw[k][dk] is not None and len(raw[k][pk]) >= minimum]

    views = make("pts", "des", MIN_INLIERS)
    fast = make("fpts", "fdes", MIN_INLIERS)
    large = float(np.mean([r["size"] / r["width"] for r in raw])) > LARGE_FRACTION
    return FeatureModel(views, large, region, float(np.median(devs)) if devs else float("nan"), fast or None)


class FeatureRefiner:
    """Run time: turns a mask-based detection into an anchor pose from matched features."""

    def __init__(self, model: FeatureModel, nfeatures: int = 800, quality: str = "fast") -> None:
        self.model = model
        self.nfeatures = nfeatures
        self.quality = quality                  # "fast" (ORB) or "precise" (SIFT)
        self._last_view = 0
        self._off: np.ndarray | None = None      # anchor - mask centroid in the object frame, in units of the object size
        self._rel_poly: np.ndarray | None = None  # feature region corners relative to the anchor, object frame, units of the object size
        self.last_offset_px = np.zeros(2)        # same offset in image pixels (to centre searches on the mask centroid)
        self.last_inliers = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctrack-features")

    def reset(self) -> None:
        self._off = None
        self._rel_poly = None
        self.last_offset_px = np.zeros(2)
        self.last_inliers = 0

    @staticmethod
    def _size(det: Detection) -> float:
        c = det.contour
        return float(math.hypot(np.ptp(c[:, 0]), np.ptp(c[:, 1]))) if c is not None and len(c) > 3 else 0.0

    # -- feature extraction (can run while the detector works) -----------------------------
    def _extract(self, img: np.ndarray, size: float, box: tuple[int, int, int, int]):
        pts, des = _features(img, size, None, box, self.nfeatures, self.model.kind_for(self.quality))
        return box, pts, des

    def prepare(self, img: np.ndarray, guess_contour: np.ndarray):
        """Start extracting features for the area where the object was last seen (grown), in a worker thread. The detector
        runs at the same time (its heavy part releases the interpreter lock), so this time is hidden. Returns a future."""
        h, w = img.shape[:2]
        c = np.asarray(guess_contour)
        size = float(math.hypot(np.ptp(c[:, 0]), np.ptp(c[:, 1])))
        if size < 40:
            return None
        pad = GUESS_PAD * size
        box = (int(max(c[:, 0].min() - pad, 0)), int(max(c[:, 1].min() - pad, 0)), int(min(c[:, 0].max() + pad, w)), int(min(c[:, 1].max() + pad, h)))
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            return None
        return self._pool.submit(self._extract, img, size, box)

    def _solve(self, img: np.ndarray, det: Detection, size: float, prepared=None):
        h, w = img.shape[:2]
        pad = SEARCH_PAD * size
        xs, ys = det.contour[:, 0], det.contour[:, 1]
        box = (int(max(xs.min() - pad, 0)), int(max(ys.min() - pad, 0)), int(min(xs.max() + pad, w)), int(min(ys.max() + pad, h)))
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            return None
        kind = self.model.kind_for(self.quality)
        pts = des = None
        if prepared is not None:
            try:
                pbox, ppts, pdes = prepared.result()
                if pbox[0] <= box[0] and pbox[1] <= box[1] and pbox[2] >= box[2] and pbox[3] >= box[3]:   # covers where the object is now
                    inside = (ppts[:, 0] >= box[0]) & (ppts[:, 0] < box[2]) & (ppts[:, 1] >= box[1]) & (ppts[:, 1] < box[3])
                    pts, des = ppts[inside], (pdes[inside] if pdes is not None else None)
            except Exception:
                pts = des = None
        if pts is None:
            pts, des = _features(img, size, None, box, self.nfeatures, kind)
        best = None
        views = self.model.views_for(self.quality)
        for i in [self._last_view] + [k for k in range(len(views)) if k != self._last_view]:
            v = views[i]
            res = _match(pts, des, v.pts, v.des, kind)
            if res and (best is None or res[1] > best[1]):
                best = (res[0], res[1], i)
            if best and best[1] >= GOOD_INLIERS and i == self._last_view:
                break
        return best

    def refine(self, img: np.ndarray, det: Detection, prepared=None) -> Detection:
        """The detection with its pose moved to the feature anchor (same outline) and the feature region attached. When the
        features do not agree, the anchor is derived from the mask instead, in the same convention, so the pose never jumps
        between two reference points."""
        size = self._size(det)
        if size < 40:
            return det
        c, s = math.cos(det.theta), math.sin(det.theta)
        poly = None
        best = self._solve(img, det, size, prepared)
        if best is not None:
            M, n, i = best
            v = self.model.views_for(self.quality)[min(i, len(self.model.views_for(self.quality)) - 1)]
            a = M[:, :2] @ v.anchor + M[:, 2]
            th = v.theta + math.atan2(M[1, 0], M[0, 0])
            off = v.off
            e = np.array([det.x + size * (c * off[0] - s * off[1]), det.y + size * (s * off[0] + c * off[1])])
            if math.hypot(a[0] - e[0], a[1] - e[1]) > MAX_ANCHOR_DEV * size:
                best = None                              # implausible: the mask says the feature is elsewhere
        if best is None:
            self.last_inliers = 0
            if self._off is None:
                return det
            ax = det.x + size * (c * self._off[0] - s * self._off[1])
            ay = det.y + size * (s * self._off[0] + c * self._off[1])
            self.last_offset_px = np.array([ax - det.x, ay - det.y])
            fb = None
            if self._rel_poly is not None:
                fb = np.array([ax, ay]) + size * (self._rel_poly @ np.array([[c, s], [-s, c]]))
            return replace(det, x=float(ax), y=float(ay), feature=fb, feature_matched=False)
        self._last_view, self.last_inliers = i, n
        if v.roi_wh is not None:                         # the marked region, mapped into this frame
            hw, hh = v.roi_wh / 2.0
            corners = v.anchor + np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], np.float32)
            poly = (corners @ M[:, :2].T + M[:, 2]).astype(np.float32)
        # theta of a mask is defined modulo pi; keep the branch closest to the detection's
        th = det.theta + (wrap_angle(2.0 * (th - det.theta)) / 2.0)
        d = np.array([a[0] - det.x, a[1] - det.y])
        self._off = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) / size
        self.last_offset_px = d
        if poly is not None:
            self._rel_poly = ((poly - a) / size) @ np.array([[c, -s], [s, c]])
        return replace(det, x=float(a[0]), y=float(a[1]), theta=float(th), feature=poly, feature_matched=True)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
