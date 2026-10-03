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
    out = FeatureRefiner(model, quality="precise").refine(img, det)
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


def _roi(M, cx=120.0, cy=90.0, w=70.0, h=50.0):
    """The marked feature region around the texture point (cx, cy), as the user would draw it on this photo."""
    x, y = M[:, :2] @ np.array([cx, cy]) + M[:, 2]
    sc = math.hypot(M[0, 0], M[1, 0])
    return (x - w * sc / 2, y - h * sc / 2, w * sc, h * sc)


@pytest.fixture(scope="module")
def region_model():
    samples, rois = [], []
    for cx, cy, sc, rot in ((400, 300, 1.0, 0), (700, 450, 0.8, 12), (500, 500, 1.2, -10)):
        img, mask, M = _scene(cx, cy, sc, rot)
        samples.append((img, mask))
        rois.append(_roi(M))
    return build_feature_model(samples, rois)


def test_regions_marked_on_the_photos_define_the_anchor(region_model):
    m = region_model
    assert m is not None and m.region and len(m.views) == 3
    for v in m.views:
        assert v.roi_wh is not None and v.roi_wh[0] > 20
    assert m.mark_dev < 0.03, m.mark_dev                       # the marks agree (same physical spot on every photo)


def test_unmarked_photos_inherit_anchor_and_region_size_and_bad_marks_are_reported():
    scenes = [_scene(400, 300, 1.0, 0), _scene(700, 450, 0.8, 12), _scene(500, 500, 1.2, -10)]
    samples = [(a, b) for a, b, _ in scenes]
    m = build_feature_model(samples, [_roi(scenes[0][2]), None, None])
    assert m is not None and m.region and len(m.views) == 3
    for v, (_, _, M) in zip(m.views, scenes):
        truth = M[:, :2] @ ANCHOR_TEX + M[:, 2]
        assert math.dist(v.anchor, truth) < 3.0, "anchor carried into the unmarked photos"
        assert abs(v.roi_wh[0] - 70.0 * math.hypot(M[0, 0], M[1, 0])) < 6.0     # region size scaled with the object
    # marks that point at different places are detected
    bad = [_roi(scenes[0][2]), _roi(scenes[1][2], cx=300.0, cy=40.0), _roi(scenes[2][2])]
    assert build_feature_model(samples, bad).mark_dev > 0.06


@pytest.mark.parametrize("cx,cy,scale,rot", [(450, 350, 1.0, 5), (650, 420, 0.9, -15), (520, 380, 1.1, 20)])
def test_marked_feature_gives_the_exact_physical_point_even_with_a_wobbling_mask(region_model, cx, cy, scale, rot):
    img, mask, M = _scene(cx, cy, scale, rot)
    truth = M[:, :2] @ ANCHOR_TEX + M[:, 2]                    # the marked feature's centre in this frame
    ref = FeatureRefiner(region_model)
    for jx, jy in ((16, -10), (-14, 12), (0, 0)):
        det = _det_from_mask(mask, (jx, jy))
        out = ref.refine(img, det)
        assert out.feature_matched and ref.last_inliers >= 10, (jx, jy)
        assert math.dist((out.x, out.y), truth) < 2.5, (jx, jy, out.x, out.y, truth)
        assert out.feature is not None and out.feature.shape == (4, 2)
        assert math.dist(out.feature.mean(axis=0), truth) < 3.0         # the drawn feature box sits on the feature
        assert np.ptp(out.feature[:, 0]) < 0.4 * (mask.any(0).nonzero()[0].ptp())   # small compared with the object


def test_prepared_features_from_a_parallel_thread_give_the_same_result(region_model):
    img, mask, _ = _scene(500, 380, 1.0, 0)
    det = _det_from_mask(mask, (6, -4))
    plain = FeatureRefiner(region_model).refine(img, det)
    ref = FeatureRefiner(region_model)
    fut = ref.prepare(img, det.contour)                        # started before the detector would finish
    pre = ref.refine(img, det, fut)
    assert pre.feature_matched and math.dist((pre.x, pre.y), (plain.x, plain.y)) < 1.5
    ref.close()


def test_implausible_anchor_is_rejected_in_favour_of_the_mask_convention(region_model):
    img, mask, _ = _scene(500, 380, 1.0, 0)
    det = _det_from_mask(mask)
    ref = FeatureRefiner(region_model)
    good = ref.refine(img, det)
    far = Detection(det.x + 300, det.y + 200, det.theta, 0.9, contour=det.contour + (300, 200))   # the mask claims the object is elsewhere
    out = ref.refine(img, far)
    assert not out.feature_matched                             # the matched position disagrees with the mask by far: not trusted
    assert math.dist((out.x - far.x, out.y - far.y), (good.x - det.x, good.y - det.y)) < 1.0


def test_fallback_keeps_the_feature_box_and_convention(region_model):
    img, mask, _ = _scene(500, 380, 1.0, 0)
    det = _det_from_mask(mask)
    ref = FeatureRefiner(region_model)
    good = ref.refine(img, det)
    img2, mask2, _ = _scene(580, 400, 1.0, 0)
    det2 = _det_from_mask(mask2)
    fb = ref.refine(np.full((H, W, 3), 128, np.uint8), det2)
    assert not fb.feature_matched and fb.feature is not None
    assert math.dist((fb.x - det2.x, fb.y - det2.y), (good.x - det.x, good.y - det.y)) < 1.0
    assert math.dist(fb.feature.mean(axis=0), (fb.x, fb.y)) < 3.0


def test_features_survive_saving_with_the_object(tmp_path, model, region_model):
    om2 = ObjectModel("z", np.zeros(384), 0.0, np.zeros(384), np.ones(384), feat=region_model)
    om2.save(tmp_path / "z.npz")
    back2 = ObjectModel.load(tmp_path / "z.npz")
    assert back2.feat.region and np.allclose(back2.feat.views[0].roi_wh, region_model.views[0].roi_wh) and np.allclose(back2.feat.views[1].off, region_model.views[1].off)
    om = ObjectModel("x", np.zeros(384), 0.0, np.zeros(384), np.ones(384), feat=model)
    om.save(tmp_path / "x.npz")
    back = ObjectModel.load(tmp_path / "x.npz")
    assert back.feat is not None and len(back.feat.views) == len(model.views) and back.feat.large == model.large
    assert np.allclose(back.feat.views[0].des, model.views[0].des)
    plain = ObjectModel("y", np.zeros(384), 0.0, np.zeros(384), np.ones(384))
    plain.save(tmp_path / "y.npz")
    assert ObjectModel.load(tmp_path / "y.npz").feat is None                     # old models keep working


@pytest.mark.parametrize("quality,tol", [("precise", 2.5), ("fast", 4.0)])
def test_fast_and_precise_descriptors_both_find_the_marked_point(region_model, quality, tol):
    img, mask, M = _scene(520, 380, 1.0, 10)
    truth = M[:, :2] @ ANCHOR_TEX + M[:, 2]
    assert region_model.fast_views and len(region_model.fast_views) == len(region_model.views)
    out = FeatureRefiner(region_model, quality=quality).refine(img, _det_from_mask(mask, (9, -6)))
    assert out.feature_matched and math.dist((out.x, out.y), truth) < tol, (quality, out.x, out.y, truth)


def test_models_without_fast_features_use_sift_and_both_sets_are_saved(tmp_path, region_model):
    old = FeatureModel(region_model.views, region_model.large, region_model.region, region_model.mark_dev, None)
    assert old.kind_for("orb") == "sift" and old.views_for("orb") is old.views
    img, mask, M = _scene(520, 380, 1.0, 10)
    out = FeatureRefiner(old, quality="fast").refine(img, _det_from_mask(mask))
    assert out.feature_matched
    om = ObjectModel("w", np.zeros(384), 0.0, np.zeros(384), np.ones(384), feat=region_model)
    om.save(tmp_path / "w.npz")
    back = ObjectModel.load(tmp_path / "w.npz").feat
    assert back.fast_views and back.fast_views[0].des.dtype == np.uint8 and back.views[0].des.dtype == np.float32
