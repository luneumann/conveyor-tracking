import math
import time

import cv2
import numpy as np
import pytest

from conftest import ROOT
from ctrack.camera.synthetic import render_object
from ctrack.detector import LearnedObjectDetector, NoDetector
from ctrack.gui.engine import SPEEDS, Settings
from ctrack.objectmodel import (GRID, VIEW, ObjectModel, _fit_logistic, crop_view, global_view, mask_pose,
                                train_object_model)
from ctrack.types import Frame, Pose, wrap_angle
from ctrack.vision.onnx_models import PATCH, DinoFeatures, SamSegmenter, vision_models_present

MODELS = ROOT / "models"
needs_vision = pytest.mark.skipif(not vision_models_present(MODELS), reason="MobileSAM / DINOv2 ONNX models not downloaded")


def _blob_prob(cx, cy, length, width, angle, size=(224, 224), grid=16):
    """Probability map (grid x grid) of a rotated rectangle, rendered at `size` and averaged down."""
    img = np.zeros((size[1], size[0]), np.uint8)
    box = cv2.boxPoints(((cx, cy), (length, width), math.degrees(angle))).astype(np.int32)
    cv2.fillConvexPoly(img, box, 255)
    return cv2.resize(img.astype(np.float32) / 255.0, (grid, grid), interpolation=cv2.INTER_AREA)


def test_mask_pose_centroid_and_axis():
    prob = _blob_prob(130, 90, 120, 50, math.radians(25))
    mp = mask_pose(prob, (224, 224))
    assert mp is not None
    assert math.hypot(mp.x - 130, mp.y - 90) < 6
    assert abs(math.degrees(wrap_angle(2 * (mp.theta - math.radians(25))) / 2)) < 5
    assert 100 < mp.bbox_long < 150 and mp.score > 0.6
    assert mp.contour.shape[1] == 2 and len(mp.contour) > 8


def test_mask_pose_theta_is_unwrapped_against_previous():
    """An elongated blob defines theta only modulo pi; it must stay continuous with the previous estimate."""
    prob = _blob_prob(112, 112, 130, 40, math.radians(-10))
    free = mask_pose(prob, (224, 224)).theta
    assert abs(free - math.radians(-10)) < math.radians(6)
    for prev in (2.9, -2.9, 3.0):                       # previous estimate near +-pi: the 180 deg flipped branch
        t = mask_pose(prob, (224, 224), prev_theta=prev).theta
        assert abs(t - prev) <= math.pi / 2 + 1e-6
        assert abs(wrap_angle(t - math.radians(-10))) < math.radians(6) or abs(wrap_angle(t - math.radians(170))) < math.radians(6)


def test_mask_pose_empty_and_tiny_blobs():
    assert mask_pose(np.zeros((16, 16), np.float32), (224, 224)) is None
    tiny = np.zeros((16, 16), np.float32)
    tiny[8, 8] = 1.0                                   # one patch = 0.4 % of the area: below min_area
    assert mask_pose(tiny, (224, 224), min_area=0.01) is None


def test_mask_pose_picks_the_blob_with_most_probability_mass():
    prob = np.zeros((16, 16), np.float32)
    prob[2:5, 2:5] = 0.9                               # small, confident
    prob[8:15, 8:15] = 0.7                             # large
    mp = mask_pose(prob, (224, 224))
    assert mp.x > 112 and mp.y > 112


def test_logistic_regression_separates_and_balances_classes():
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, (60, 8))                # few positives
    neg = rng.normal(-1.0, 1.0, (2000, 8))             # many negatives
    X, y = np.vstack([pos, neg]), np.r_[np.ones(60), np.zeros(2000)]
    w, b = _fit_logistic(X, y)
    p = 1 / (1 + np.exp(-(X @ w + b)))
    assert (p[:60] > 0.5).mean() > 0.8                 # class weighting keeps the rare class alive
    assert (p[60:] < 0.5).mean() > 0.85


def test_crop_view_roundtrip_geometry():
    img = np.zeros((400, 640, 3), np.uint8)
    cv2.circle(img, (300, 180), 4, (255, 255, 255), -1)
    for view in (224, 168, 140):
        crop, s = crop_view(img, 310, 190, 100, view)
        assert crop.shape[:2] == (view, view) and s == pytest.approx(view / 100)
        ys, xs = np.nonzero(crop[..., 0] > 128)
        u, v = xs.mean(), ys.mean()
        assert 310 + (u - view / 2) / s == pytest.approx(300, abs=1.0)    # maps back to the frame position
        assert 190 + (v - view / 2) / s == pytest.approx(180, abs=1.0)


@pytest.mark.parametrize("w,h", [(1280, 720), (640, 400), (800, 600)])
def test_global_view_is_patch_aligned(w, h):
    view, s = global_view(np.zeros((h, w, 3), np.uint8), 448)
    assert view.shape[0] % PATCH == 0 and view.shape[1] % PATCH == 0
    assert s == pytest.approx(view.shape[1] / w)


def test_object_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    m = ObjectModel("x", rng.normal(size=384).astype(np.float32), 0.3, rng.normal(size=384).astype(np.float32),
                    np.ones(384, np.float32), n_images=5, n_patches=100, created="now")
    m.save(tmp_path / "o" / "x.npz")
    m2 = ObjectModel.load(tmp_path / "o" / "x.npz")
    feats = rng.normal(size=(4, 4, 384)).astype(np.float32)
    assert np.allclose(m.prob_map(feats), m2.prob_map(feats)) and (m2.name, m2.n_images) == ("x", 5)


def test_settings_speed_and_learned_validation():
    s = Settings()
    assert s.speed == "balanced" and s.update({"speed": "fast"}) == {"speed"}
    with pytest.raises(ValueError):
        s.update({"speed": "warp"})
    with pytest.raises(ValueError):
        s.update({"detector": "magic"})
    with pytest.raises(ValueError):
        s.update({"learned": "../x"})
    assert s.update({"obj_score": 5}) == {"obj_score"} and s.obj_score == 0.95
    assert all(v[0] % PATCH == 0 for v in SPEEDS.values())


def test_no_detector_finds_nothing():
    assert NoDetector().detect(Frame(np.zeros((10, 10, 3), np.uint8), 0.0, 0)) is None


# ----------------------------------------------------------------------------------------------------------
def _scene(scale, theta, cx, cy, seed, size=(640, 400)):
    """Textured background; returns (image, mask) of the synthetic part."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.normal(70, 30, (size[1], size[0], 1)).astype(np.float32) * np.ones((1, 1, 3), np.float32), (0, 0), 4)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    img = bg.copy()
    render_object(img, Pose(cx, cy, theta), 160 * scale, 70 * scale)
    mask = cv2.cvtColor(cv2.absdiff(img, bg), cv2.COLOR_BGR2GRAY) > 25
    return img, cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    dino = DinoFeatures(MODELS)
    samples = [_scene(sc, th, 320, 200, i) for i, (sc, th) in enumerate([(1.0, 0.0), (0.7, 0.4), (1.3, -0.3), (0.85, 0.15)])]
    model = train_object_model(samples, dino, "synth")
    path = tmp_path_factory.mktemp("obj") / "synth.npz"
    model.save(path)
    return path, model


@needs_vision
def test_training_reports_progress_and_validates_input():
    dino = DinoFeatures(MODELS)
    with pytest.raises(ValueError):
        train_object_model([], dino, "x")
    img, mask = _scene(1.0, 0.0, 320, 200, 0)
    with pytest.raises(ValueError, match="zu klein"):
        train_object_model([(img, np.zeros_like(mask))], dino, "x")
    seen = []
    train_object_model([(img, mask)], dino, "x", views_per_image=2, progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0) and all(a <= b + 1e-9 for a, b in zip(seen, seen[1:]))


@needs_vision
def test_trained_head_marks_the_object_and_not_the_background(trained):
    _, model = trained
    dino = DinoFeatures(MODELS)
    img, mask = _scene(0.9, 0.3, 300, 210, 77)
    view, s = global_view(img, 448)
    prob = model.prob_map(dino.extract(view))
    small = cv2.resize(mask.astype(np.float32), (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_AREA)
    assert prob[small > 0.9].mean() > 0.7              # patches fully on the object
    assert prob[small < 0.01].mean() < 0.2             # pure background


@needs_vision
@pytest.mark.parametrize("scale,theta,cx,cy", [(0.85, 0.5, 400, 250), (1.2, -0.4, 250, 180), (0.65, 0.2, 450, 200)])
def test_learned_detector_finds_pose_at_new_scale_and_rotation(trained, scale, theta, cx, cy):
    path, _ = trained
    det = LearnedObjectDetector(str(path), models_dir=str(MODELS))
    img, _ = _scene(scale, theta, cx, cy, 99)
    d = det.detect(Frame(img, 0.0, 0))                 # first frame: global search
    assert d is not None, "object not found"
    size = 160 * scale
    assert math.hypot(d.x - cx, d.y - cy) < 0.2 * size
    assert abs(math.degrees(wrap_angle(2 * (d.theta - theta)) / 2)) < 15      # axis angle is defined modulo 180 deg
    assert d.confidence > 0.6 and d.contour is not None and len(d.contour) > 8
    # next frame, object moved a little: the crop tracker takes over and follows
    img2, _ = _scene(scale, theta + 0.05, cx + 12, cy + 4, 100)
    d2 = det.detect(Frame(img2, 0.033, 1))
    assert d2 is not None and math.hypot(d2.x - (cx + 12), d2.y - (cy + 4)) < 0.2 * size


@needs_vision
def test_learned_detector_ignores_empty_scene_and_other_shapes(trained):
    path, _ = trained
    det = LearnedObjectDetector(str(path), models_dir=str(MODELS), global_interval=1)
    empty, _ = _scene(1.0, 0.0, -500, -500, 5)         # object far outside the image
    assert det.detect(Frame(empty, 0.0, 0)) is None
    circle = empty.copy()
    cv2.circle(circle, (320, 200), 60, (40, 40, 200), -1)
    assert det.detect(Frame(circle, 0.1, 1)) is None


@needs_vision
def test_learned_detector_recovers_after_object_leaves_and_returns(trained):
    path, _ = trained
    det = LearnedObjectDetector(str(path), models_dir=str(MODELS), global_interval=1)
    a, _ = _scene(1.0, 0.0, 320, 200, 11)
    assert det.detect(Frame(a, 0.0, 0)) is not None
    gone, _ = _scene(1.0, 0.0, -500, -500, 12)
    results = [det.detect(Frame(gone, 0.03 * (i + 1), i + 1)) for i in range(8)]
    assert all(r is None for r in results)
    b, _ = _scene(0.8, 0.6, 420, 240, 13)
    d = det.detect(Frame(b, 1.0, 20))                   # back somewhere else: found by the global search
    assert d is not None and math.hypot(d.x - 420, d.y - 240) < 40


def test_learned_detector_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        LearnedObjectDetector(str(tmp_path / "nope.npz"), models_dir=str(MODELS))


@needs_vision
def test_sam_one_click_segments_the_part_and_negative_click_trims():
    img, gt = _scene(1.0, 0.3, 320, 200, 42)
    sam = SamSegmenter(MODELS)
    sam.set_image(img)
    c, s = math.cos(0.3), math.sin(0.3)
    both = sam.segment([(320, 200), (320 + c * 48, 200 + s * 48)], [1, 1])      # body + marker dot
    iou = (both & gt).sum() / (both | gt).sum()
    assert iou > 0.85
    with pytest.raises(ValueError):
        sam.segment([], [])
    assert GRID == VIEW // PATCH


# ----------------------------------------------------------------------------------------------------------
def _lookalike_scene(scale, theta, cx, cy, seed, size=(640, 400)):
    """Same white body as the real part, but with a BLUE marker dot: a realistic 'almost my object' distractor."""
    img, _ = _scene(scale, theta, cx, cy, seed, size)
    c, s = math.cos(theta), math.sin(theta)
    hl = 80.0 * scale
    dot = (round(cx + c * hl * 0.6), round(cy + s * hl * 0.6))
    cv2.circle(img, dot, round(70 * scale * 0.25) + 1, (200, 90, 20), -1, cv2.LINE_AA)   # BGR blue, covers the red dot
    return img


@needs_vision
def test_empty_scenes_teach_the_head_to_reject_lookalikes():
    """Frames of the real surroundings (here: containing the lookalike) are the most truthful negatives."""
    dino = DinoFeatures(MODELS)
    samples = [_scene(sc, th, 320, 200, i) for i, (sc, th) in enumerate([(1.0, 0.0), (0.7, 0.4), (1.3, -0.3), (0.85, 0.15)])]
    placements = [(1.0, 0.3, 300, 200), (0.8, -0.5, 350, 180), (1.2, 0.8, 280, 220), (0.9, -0.2, 400, 210),
                  (1.1, 0.1, 320, 190), (0.75, 0.6, 360, 230)]
    empties = [_lookalike_scene(*p, 900 + i) for i, p in enumerate(placements * 2)]
    without = train_object_model(samples, dino, "a", seed=1)
    with_empty = train_object_model(samples, dino, "b", seed=1, empty_scenes=empties)

    def false_positive_rate(model):
        hits = 0
        for i, (sc, th, cx, cy) in enumerate([(0.95, 0.45, 310, 205), (1.15, -0.35, 330, 195), (0.8, 0.9, 370, 215),
                                              (1.0, -0.8, 290, 200), (0.9, 0.2, 340, 225), (1.25, 0.05, 320, 185)]):
            view, _ = global_view(_lookalike_scene(sc, th, cx, cy, 500 + i), 448)
            mp = mask_pose(model.prob_map(dino.extract(view)), (640, 400))
            hits += mp is not None and mp.score >= 0.6 and model.aspect_ok(mp.aspect)
        return hits / 6

    def recall(model):
        hits = 0
        for i, (sc, th, cx, cy) in enumerate([(0.95, 0.45, 310, 205), (1.15, -0.35, 330, 195), (0.8, 0.9, 370, 215), (1.0, -0.8, 290, 200)]):
            view, _ = global_view(_scene(sc, th, cx, cy, 600 + i)[0], 448)
            mp = mask_pose(model.prob_map(dino.extract(view)), (640, 400))
            hits += mp is not None and mp.score >= 0.6 and math.hypot(mp.x - cx, mp.y - cy) < 0.25 * 160 * sc
        return hits / 4

    fp_a, fp_b = false_positive_rate(without), false_positive_rate(with_empty)
    assert fp_b <= fp_a and fp_b <= 0.34, f"false positives without/with empty scenes: {fp_a:.2f} / {fp_b:.2f}"
    assert recall(with_empty) >= 0.75, "the real object must still be found"


def test_model_records_and_restores_its_backbone(tmp_path):
    rng = np.random.default_rng(2)
    m = ObjectModel("x", rng.normal(size=384).astype(np.float32), 0.0, np.zeros(384, np.float32), np.ones(384, np.float32),
                    backbone="dinov2_small_int8.onnx")
    m.save(tmp_path / "x.npz")
    assert ObjectModel.load(tmp_path / "x.npz").backbone == "dinov2_small_int8.onnx"
    old = np.load(tmp_path / "x.npz")
    np.savez(tmp_path / "old.npz", **{k: old[k] for k in old.files if k != "backbone"})       # a model saved before the field existed
    assert ObjectModel.load(tmp_path / "old.npz").backbone == "dinov2_small.onnx"


def test_detector_refuses_a_model_whose_backbone_file_is_missing(tmp_path):
    rng = np.random.default_rng(3)
    ObjectModel("x", rng.normal(size=384).astype(np.float32), 0.0, np.zeros(384, np.float32), np.ones(384, np.float32),
                backbone="not_there.onnx").save(tmp_path / "x.npz")
    with pytest.raises(FileNotFoundError, match="not_there.onnx"):
        LearnedObjectDetector(str(tmp_path / "x.npz"), models_dir=str(MODELS))


@needs_vision
def test_trained_model_remembers_the_backbone_it_was_trained_with(trained):
    _, model = trained
    assert model.backbone == DinoFeatures(MODELS).filename


@needs_vision
def test_whole_frame_search_is_throttled_only_while_it_is_slow(trained):
    path, _ = trained
    blank = Frame(np.full((400, 640, 3), 35, np.uint8), 0.0, 0)

    def count_searches(delay_s, frames=12):
        det = LearnedObjectDetector(str(path), models_dir=str(MODELS), global_interval=3)
        calls = []

        def fake(img):
            calls.append(1)
            time.sleep(delay_s)
            return None

        det._global_search = fake
        for i in range(frames):
            det.detect(Frame(blank.image, i / 30, i))
        return len(calls)

    slow, fast = count_searches(0.08), count_searches(0.0)
    assert fast == 12                         # accelerator-like speed: search on every frame
    assert slow <= 5, slow                    # CPU-like speed: every 3rd frame, so the frame loop is not starved
