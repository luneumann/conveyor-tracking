"""Few-shot learned object model (ADR-011).

A frozen DINOv2 backbone gives every 14x14 px patch a 384-d feature. From a handful of images with a
mask of the object (one SAM click each), a per-object logistic-regression head learns which patches belong to
the object. To get scale / rotation / lighting robustness from ~5 images, each image is expanded into many
augmented views (the object zoomed to 15-80 % of the crop, rotated +-30 deg, shifted, brightness/contrast
jittered, optionally blurred) before features are extracted.

At run time the head turns patch features into a probability map; mask_pose() turns that into a pose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .features import FeatureModel
from .types import wrap_angle
from .vision.onnx_models import EMBED_DIM, PATCH, DinoFeatures

VIEW = 224                      # training / tracking crop size (16 x 16 patches)
GRID = VIEW // PATCH
BORDER = (114, 114, 114)        # constant border, used identically in training and at run time
POS_COVERAGE, NEG_COVERAGE = 0.7, 0.1
# Ridge strength of the logistic head: a trade-off, measured on 6 independent trainings (held-out scenes, see ADR-011).
#   l2 = 1      unstable: the same photos gave 0..52 of 60 failed scenes depending on the random augmentation (mean 31 %)
#   l2 = 300    mean failure 3 % (worst run 10/60); real object-free camera frames still teach it to reject a near-identical
#               distractor (false alarm 17 %, recall 75 % in the hard lookalike test)
#   l2 >= 1e4   0 failures, but the head degenerates to a class-mean prototype: empty scenes no longer help (false alarm 100 %)
# Prototype-anchored ridge and up-weighting real negatives were tried and gave "reject everything" or "accept everything".
DEFAULT_L2 = 300.0


def crop_view(bgr: np.ndarray, cx: float, cy: float, side: float, view: int = VIEW) -> tuple[np.ndarray, float]:
    """Square crop of `side` px centred on (cx, cy), resized to view x view (multiple of 14); returns (crop, scale)."""
    s = view / side
    M = np.array([[s, 0, view / 2 - s * cx], [0, s, view / 2 - s * cy]], np.float32)
    return cv2.warpAffine(bgr, M, (view, view), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER), s


def global_view(bgr: np.ndarray, width: int = 448) -> tuple[np.ndarray, float]:
    """Whole frame resized so that both sides are multiples of 14 (width ~ `width`); returns (view, scale)."""
    h, w = bgr.shape[:2]
    gw = max(width // PATCH, 4)
    gh = max(round(h * gw * PATCH / w / PATCH), 4)
    return cv2.resize(bgr, (gw * PATCH, gh * PATCH), interpolation=cv2.INTER_AREA), gw * PATCH / w


@dataclass
class ObjectModel:
    name: str
    w: np.ndarray            # (384,) head weights on standardized features
    b: float
    mu: np.ndarray           # (384,) feature mean
    sigma: np.ndarray        # (384,) feature std
    n_images: int = 0
    n_patches: int = 0
    created: str = ""
    aspect_lo: float = 0.0     # accepted long/short side ratio of the object's mask (0 = unchecked)
    aspect_hi: float = 0.0
    backbone: str = "dinov2_small.onnx"   # the head only fits the backbone it was trained on
    feat: "FeatureModel | None" = None   # optional feature anchor for big objects (features.py)

    def aspect_ok(self, aspect: float) -> bool:
        return self.aspect_hi <= 0 or self.aspect_lo <= aspect <= self.aspect_hi

    def prob_map(self, feats: np.ndarray) -> np.ndarray:
        """(gh, gw, 384) features -> (gh, gw) probability that a patch belongs to the object."""
        z = ((feats - self.mu) / self.sigma) @ self.w + self.b
        return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = self.feat.to_arrays() if self.feat is not None else {}
        np.savez(path, w=self.w, b=self.b, mu=self.mu, sigma=self.sigma, name=self.name,
                 n_images=self.n_images, n_patches=self.n_patches, created=self.created,
                 aspect_lo=self.aspect_lo, aspect_hi=self.aspect_hi, backbone=self.backbone, **extra)

    @classmethod
    def load(cls, path: Path) -> ObjectModel:
        d = np.load(path, allow_pickle=False)
        return cls(str(d["name"]), d["w"], float(d["b"]), d["mu"], d["sigma"], int(d["n_images"]),
                   int(d["n_patches"]), str(d["created"]),
                   float(d["aspect_lo"]) if "aspect_lo" in d else 0.0, float(d["aspect_hi"]) if "aspect_hi" in d else 0.0,
                   str(d["backbone"]) if "backbone" in d else "dinov2_small.onnx", FeatureModel.from_arrays(d))


# ---------------------------------------------------------------------------------------------------------
def _background_only(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The photo with the object painted over (small blurry fill): a plausible object-free background."""
    grow = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gsmall = cv2.resize(grow, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, gsmall, 5, cv2.INPAINT_TELEA)
    return cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def _random_background(bg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A VIEW x VIEW patch of a background photo at a random zoom, rotation and flip."""
    h, w = bg.shape[:2]
    side = rng.uniform(0.25, 1.0) * min(h, w)
    cx, cy = rng.uniform(side / 2, max(w - side / 2, side / 2 + 1)), rng.uniform(side / 2, max(h - side / 2, side / 2 + 1))
    ang = rng.uniform(-180, 180)
    s = VIEW / side
    M = cv2.getRotationMatrix2D((cx, cy), ang, s)
    M[:, 2] += (VIEW / 2 - cx, VIEW / 2 - cy)
    out = cv2.warpAffine(bg, M, (VIEW, VIEW), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return out[:, ::-1] if rng.random() < 0.5 else out


def mask_aspect(mask: np.ndarray) -> float:
    """Long / short side of the mask's min-area rectangle."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    return float(max(rw, rh) / max(min(rw, rh), 1.0))


def _distractor_view(bg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A background patch with 1-3 random flat/lightly textured shapes: 'things that are not my object'."""
    view = _random_background(bg, rng).copy()
    for _ in range(int(rng.integers(1, 4))):
        color = tuple(int(c) for c in rng.integers(20, 240, 3))
        cx, cy = rng.uniform(25, VIEW - 25, 2)
        size = rng.uniform(0.15, 0.6) * VIEW
        kind = rng.integers(3)
        layer = np.zeros((VIEW, VIEW), np.uint8)
        if kind == 0:
            cv2.ellipse(layer, (int(cx), int(cy)), (int(size / 2), int(size / 2 * rng.uniform(0.4, 1.0))), rng.uniform(0, 180), 0, 360, 255, -1)
        elif kind == 1:
            box = cv2.boxPoints(((cx, cy), (size, size * rng.uniform(0.25, 1.0)), rng.uniform(0, 180))).astype(np.int32)
            cv2.fillConvexPoly(layer, box, 255)
        else:
            ang = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 7))))
            pts = np.c_[cx + size / 2 * np.cos(ang) * rng.uniform(0.5, 1, len(ang)), cy + size / 2 * np.sin(ang) * rng.uniform(0.5, 1, len(ang))]
            cv2.fillPoly(layer, [pts.astype(np.int32)], 255)
        tex = np.clip(np.array(color, np.float32) + rng.normal(0, rng.uniform(0, 12), (VIEW, VIEW, 1)), 0, 255)
        a = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None]
        view = (view * (1 - a) + tex * a).astype(np.uint8)
    return view


def _photometric(view: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    view = np.clip(view.astype(np.float32) * rng.uniform(0.65, 1.35) + rng.uniform(-30, 30), 0, 255)
    if rng.random() < 0.4:
        view = cv2.GaussianBlur(view, (0, 0), rng.uniform(0.6, 2.2))
    if rng.random() < 0.5:
        view = view + rng.normal(0, rng.uniform(1, 8), view.shape)
    return np.clip(view, 0, 255).astype(np.uint8)


def _augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
             backgrounds: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One random view: object zoomed/rotated/shifted into a VIEW x VIEW crop, plus photometric jitter.

    With probability 0.6 the object is cut out and pasted onto a background taken from another taught photo,
    so the head cannot rely on the surroundings of the original photo.
    """
    m8 = mask.astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(m8)
    mom = cv2.moments(m8, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    long_side = max(bw, bh)
    frac = rng.uniform(0.15, 0.8)                       # object long side as a fraction of the crop
    scale = frac * VIEW / long_side
    ang = rng.uniform(-30, 30)
    room = max(VIEW * (1 - frac) / 2 - 4, 0)
    tx, ty = VIEW / 2 + rng.uniform(-1, 1) * room, VIEW / 2 + rng.uniform(-1, 1) * room
    M = cv2.getRotationMatrix2D((cx, cy), ang, scale)
    M[:, 2] += (tx - cx, ty - cy)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    view = cv2.warpAffine(img, M, (VIEW, VIEW), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER)
    vmask = cv2.warpAffine(m8 * 255, M, (VIEW, VIEW), flags=cv2.INTER_LINEAR)
    if backgrounds and rng.random() < 0.6:
        alpha = cv2.GaussianBlur(vmask.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None]
        view = (view * alpha + _random_background(backgrounds[rng.integers(len(backgrounds))], rng) * (1 - alpha)).astype(np.uint8)
    return _photometric(view, rng), vmask


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 25) -> tuple[np.ndarray, float]:
    """Class-balanced ridge logistic regression by Newton / IRLS. X: (n, d) standardized, y in {0, 1}."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    sw = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)) * n   # balance the classes
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    beta = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        grad = Xb.T @ (sw * (p - y)) + reg @ beta
        W = sw * p * (1 - p) + 1e-6
        H = (Xb * W[:, None]).T @ Xb + reg
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.abs(step).max() < 1e-5:
            break
    return beta[:-1], float(beta[-1])


def train_object_model(samples: list[tuple[np.ndarray, np.ndarray]], dino: DinoFeatures, name: str,
                       views_per_image: int = 12, seed: int = 0,
                       progress: Callable[[float], None] | None = None,
                       empty_scenes: list[np.ndarray] | None = None, l2: float = DEFAULT_L2,
                       hard_threshold: float = 0.2) -> ObjectModel:
    """Train the head from (BGR image, bool mask) pairs. ~0.8 s per image on an M3 (12 views each).

    empty_scenes: camera frames WITHOUT the object (the real surroundings). They are the most truthful negatives:
    used as backdrops for pasting the object and for distractors, as pure-background views, and for hard-negative mining.
    """
    if not samples:
        raise ValueError("Keine Beispielbilder")
    rng = np.random.default_rng(seed)
    for i, (img, mask) in enumerate(samples):
        if mask.sum() < 64:
            raise ValueError(f"Bild {i + 1}: Maske zu klein")
    empties = list(empty_scenes or [])
    painted = [_background_only(img, mask) for img, mask in samples]
    backgrounds = painted + empties
    feats, labels = [], []
    n_dis = max(views_per_image // 2, 4)
    n_empty_views = min(3 * len(empties), 36)
    n_hard = 6 * len(painted) + 3 * len(empties)
    total = len(samples) * (views_per_image + n_dis) + n_empty_views + n_hard
    done = 0
    for i, (img, mask) in enumerate(samples):
        for v in range(views_per_image):
            view, vmask = _augment(img, mask, rng, backgrounds)
            cov = cv2.resize(vmask.astype(np.float32) / 255.0, (GRID, GRID), interpolation=cv2.INTER_AREA)
            f = dino.extract(view).reshape(-1, EMBED_DIM)
            c = cov.reshape(-1)
            keep = (c >= POS_COVERAGE) | (c <= NEG_COVERAGE)     # skip ambiguous boundary patches
            feats.append(f[keep])
            labels.append((c[keep] >= POS_COVERAGE).astype(np.float64))
            done += 1
            if progress:
                progress(done / total)
        # Distractors: random shapes that are NOT the object, so the head learns more than "something salient".
        for _ in range(n_dis):
            view = _distractor_view(backgrounds[int(rng.integers(len(backgrounds)))], rng)
            feats.append(dino.extract(_photometric(view, rng)).reshape(-1, EMBED_DIM))
            labels.append(np.zeros(GRID * GRID))
            done += 1
            if progress:
                progress(min(done / total, 1.0))
    for _ in range(n_empty_views):                      # the real, object-free surroundings: pure negatives
        view = _photometric(_random_background(empties[int(rng.integers(len(empties)))], rng), rng)
        feats.append(dino.extract(view).reshape(-1, EMBED_DIM))
        labels.append(np.zeros(GRID * GRID))
        done += 1
        if progress:
            progress(min(done / total, 1.0))
    X, y = np.vstack(feats), np.concatenate(labels)
    if y.sum() < 10 or (1 - y).sum() < 10:
        raise ValueError("Zu wenige Objekt- bzw. Hintergrund-Bildstellen zum Lernen")
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    w, b = _fit_logistic((X - mu) / sigma, y, l2=l2)

    # Hard-negative mining: object-free views the head still calls "object" are added as negatives.
    hard = []
    for bi, bg in enumerate(backgrounds):
        for _ in range(6 if bi < len(painted) else 3):
            f = dino.extract(_photometric(_random_background(bg, rng), rng)).reshape(-1, EMBED_DIM)
            p = 1.0 / (1.0 + np.exp(-np.clip(((f - mu) / sigma) @ w + b, -30, 30)))
            hard.append(f[p > hard_threshold])
            done += 1
            if progress:
                progress(min(done / total, 1.0))
    H = np.vstack(hard) if hard else np.empty((0, EMBED_DIM))
    if len(H):
        X2, y2 = np.vstack([X, H]), np.concatenate([y, np.zeros(len(H))])
        mu, sigma = X2.mean(0), X2.std(0) + 1e-6
        w, b = _fit_logistic((X2 - mu) / sigma, y2, l2=l2)
        X, y = X2, y2
    aspects = [mask_aspect(m) for _, m in samples]
    return ObjectModel(name, w.astype(np.float32), b, mu.astype(np.float32), sigma.astype(np.float32),
                       n_images=len(samples), n_patches=len(y), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       aspect_lo=0.7 * min(aspects), aspect_hi=1.4 * max(aspects), backbone=dino.filename)


# ---------------------------------------------------------------------------------------------------------
@dataclass
class MaskPose:
    x: float
    y: float
    theta: float
    contour: np.ndarray       # (n, 2) in the coordinates of the probability map's pixel grid
    bbox_long: float          # long side of the mask's min-area rectangle, in the same units
    aspect: float             # long / short side of that rectangle
    score: float              # mean probability inside the mask


def mask_pose(prob: np.ndarray, out_size: tuple[int, int], prev_theta: float | None = None,
              threshold: float = 0.5, min_area: float = 0.004) -> MaskPose | None:
    """Probability map (gh, gw) -> pose of the largest blob, in coordinates of an out_size (w, h) image.

    theta is the orientation of the blob's principal axis. An elongated object only defines it modulo pi,
    so it is unwrapped against `prev_theta` to stay continuous (no 180 deg flips between frames).
    """
    w, h = out_size
    p = cv2.resize(prob, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    mask = (p > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    mass = np.bincount(lab.ravel(), weights=p.ravel(), minlength=n)    # choose the blob with most probability mass
    k = 1 + int(np.argmax(mass[1:]))
    if stats[k, cv2.CC_STAT_AREA] < min_area * w * h:
        return None
    blob = (lab == k).astype(np.uint8)
    mom = cv2.moments(blob, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    theta = 0.5 * math.atan2(2 * mom["mu11"], mom["mu20"] - mom["mu02"])   # principal axis, in (-pi/2, pi/2]
    if prev_theta is not None:
        while theta - prev_theta > math.pi / 2:
            theta -= math.pi
        while theta - prev_theta < -math.pi / 2:
            theta += math.pi
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    return MaskPose(cx, cy, wrap_angle(theta) if prev_theta is None else theta, c[:, 0, :].astype(np.float32),
                    float(max(rw, rh)), float(max(rw, rh) / max(min(rw, rh), 1.0)), float(p[blob > 0].mean()))
