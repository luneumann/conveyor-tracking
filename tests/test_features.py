import math

import cv2
import numpy as np
import pytest

from ctrack.features import FeatureModel, FeatureRefiner, build_feature_model
from ctrack.objectmodel import ObjectModel
from ctrack.types import Detection

W, H = 1200, 800
TEX = cv2.resize((np.random.default_rng(5).random((60, 90)) * 255).astype(np.uint8), (360, 240), interpolation=cv2.INTER_CUBIC)
TEX_BGR = cv2.cvtColor(TEX, cv2.COLOR_GRAY2BGR)
ANCHOR_TEX = np.array([120.0, 90.0])                     # a physical point of the object (in texture coordinates)


def _scene(cx, cy, scale=1.0, rot_deg=0.0, size_scale=1.0):
    """A textured rectangle on a plain background; returns (image, mask, 2x3 texture->frame transform)."""
    rng = np.random.default_rng(9)
    img = np.full((H, W, 3), 120, np.uint8)
    img += (rng.random((H, W, 1)) * 10).astype(np.uint8)
    M = cv2.getRotationMatrix2D((180, 120), rot_deg, scale * size_scale)
    M[:, 2] += (cx - 180, cy - 120)
    mask = cv2.warpAffine(np.full(TEX.shape, 255, np.uint8), M, (W, H)) > 0
    warped = cv2.warpAffine(TEX_BGR, M, (W, H), flags=cv2.INTER_LINEAR)
    img[mask] = warped[mask]
    return img, mask, M


def _det_from_mask(mask, jitter=(0.0, 0.0)):
    ys, xs = np.nonzero(mask)
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    th = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) + jitter
    return Detection(cx + jitter[0], cy + jitter[1], th, 0.9, contour=c)


@pytest.fixture(scope="module")
def model():
    samples = []
    for cx, cy, sc, rot in ((400, 300, 1.0, 0), (700, 450, 0.8, 12), (500, 500, 1.2, -10)):
        img, mask, _ = _scene(cx, cy, sc, rot)
        samples.append((img, mask))
    return build_feature_model(samples)


def test_model_links_the_views_and_flags_large_objects(model):
    assert model is not None and len(model.views) >= 2
    assert model.large                                    # object spans > 20 % of the image width
    small = build_feature_model([(_scene(400, 300, 0.4)[0], _scene(400, 300, 0.4)[1]), (_scene(600, 400, 0.45, 8)[0], _scene(600, 400, 0.45, 8)[1])])
    assert small is not None and not small.large


@pytest.mark.parametrize("cx,cy,scale,rot", [(450, 350, 1.0, 5), (650, 420, 0.9, -15), (520, 380, 1.1, 20)])
def test_anchor_pose_is_exact_even_when_the_mask_wobbles(model, cx, cy, scale, rot):
    img, mask, M = _scene(cx, cy, scale, rot)
    truth = M[:, :2] @ ANCHOR_TEX + M[:, 2]
    ref = FeatureRefiner(model)
    feat, mask_pos = [], []
    for jx, jy in ((14, -9), (-12, 11), (0, 0)):          # the mask outline wobbles, the texture does not
        det = _det_from_mask(mask, (jx, jy))
        out = ref.refine(img, det)
        feat.append((out.x, out.y))
        mask_pos.append((det.x, det.y))
    assert ref.last_inliers >= 10
    assert math.dist(feat[2], (cx, cy)) < 3.0            # absolute: photo 1's mask centroid is the object centre
    spread = lambda pts: max(math.dist(p, q) for p in pts for q in pts)
    # the anchor is the same physical point whatever the mask does; the mask centroid follows the wobble
    assert spread(feat) < 2.0, f"feature anchor moved with the mask jitter: {feat}"
    assert spread(mask_pos) > 15.0


def test_rotation_comes_from_the_features(model):
    img, mask, _ = _scene(500, 380, 1.0, 20)
    det = _det_from_mask(mask)
    out = FeatureRefiner(model).refine(img, det)
    # a rectangle's second-moment axis and the feature rotation agree to a few degrees (mod 180)
    assert abs(math.degrees(((out.theta - det.theta) + math.pi / 2) % math.pi - math.pi / 2)) < 4.0


def test_fallback_keeps_the_anchor_convention_when_features_fail(model):
    img, mask, _ = _scene(500, 380, 1.0, 0)
    det = _det_from_mask(mask)
    ref = FeatureRefiner(model)
    first = ref.refine(np.full((H, W, 3), 128, np.uint8), det)          # nothing to match, no history: detection unchanged
    assert (first.x, first.y) == (det.x, det.y)
    good = ref.refine(img, det)
    off = (good.x - det.x, good.y - det.y)
    img2, mask2, _ = _scene(560, 400, 1.0, 0)
    det2 = _det_from_mask(mask2)
    fb = ref.refine(np.full((H, W, 3), 128, np.uint8), det2)            # features fail now: same offset from the mask centroid
    assert math.hypot((fb.x - det2.x) - off[0], (fb.y - det2.y) - off[1]) < 1.0


def test_dedicated_region_sets_the_anchor_and_keeps_only_nearby_features():
    img, mask, M = _scene(450, 330, 1.0, 0)
    roi = (M[0, 0] * 80 + M[0, 2], M[1, 1] * 60 + M[1, 2], 80.0, 60.0)          # a box over the texture region around ANCHOR_TEX
    img2, mask2, M2 = _scene(650, 420, 0.9, 10)
    m = build_feature_model([(img, mask), (img2, mask2)], roi=roi)
    assert m is not None and m.region
    a0 = m.views[0].anchor
    assert np.allclose(a0, [roi[0] + roi[2] / 2, roi[1] + roi[3] / 2], atol=0.5)
    assert np.all(np.linalg.norm(m.views[0].pts - a0, axis=1) < 1.2 * 0.5 * math.hypot(roi[2], roi[3]) + 1)
    assert len(m.views[0].pts) < len(build_feature_model([(img, mask), (img2, mask2)]).views[0].pts)


def test_features_survive_saving_with_the_object(tmp_path, model):
    om = ObjectModel("x", np.zeros(384), 0.0, np.zeros(384), np.ones(384), feat=model)
    om.save(tmp_path / "x.npz")
    back = ObjectModel.load(tmp_path / "x.npz")
    assert back.feat is not None and len(back.feat.views) == len(model.views) and back.feat.large == model.large
    assert np.allclose(back.feat.views[0].des, model.views[0].des)
    plain = ObjectModel("y", np.zeros(384), 0.0, np.zeros(384), np.ones(384))
    plain.save(tmp_path / "y.npz")
    assert ObjectModel.load(tmp_path / "y.npz").feat is None                     # old models keep working
