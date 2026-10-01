import math

import cv2
import numpy as np
import pytest

import teach
from conftest import ROOT
from ctrack.camera import SyntheticConveyorSource
from ctrack.camera.synthetic import render_object
from ctrack.config import load_config
from ctrack.detector import ShapeMatchDetector
from ctrack.pipeline import Pipeline
from ctrack.types import Frame, Pose, TrackState, wrap_angle

TEMPLATE = str(ROOT / "templates" / "synthetic_part.png")  # 201x101, centre = pose origin
OBJ = (160.0, 70.0)


def _frame(pose: Pose, size=(640, 400), noise=6.0, seed=0, fid=0) -> Frame:
    img = np.full((size[1], size[0], 3), 35, np.uint8)
    render_object(img, pose, *OBJ)
    if noise:
        n = np.random.default_rng(seed).normal(0, noise, size[::-1]).astype(np.float32)
        img = np.clip(img + n[..., None], 0, 255).astype(np.uint8)
    return Frame(img, fid / 30.0, fid)


@pytest.fixture(scope="module")
def detector():
    return ShapeMatchDetector(TEMPLATE)


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.6, -2.2, 3.05, -3.1])
def test_pose_accuracy_over_full_rotation(theta):
    """Global search finds position and angle of the part at any rotation (incl. around +-pi)."""
    det = ShapeMatchDetector(TEMPLATE)
    truth = Pose(321.4, 187.8, theta)
    r = det.detect(_frame(truth))
    assert r is not None
    assert math.hypot(r.x - truth.x, r.y - truth.y) < 1.0
    assert abs(wrap_angle(r.theta - truth.theta)) < math.radians(1.0)
    assert r.confidence > 0.9


def test_taught_orientation_is_theta_zero():
    det = ShapeMatchDetector(TEMPLATE)
    r = det.detect(_frame(Pose(300, 200, 0.0)))
    assert abs(r.theta) < math.radians(0.5)


def test_180_degree_ambiguity_is_resolved():
    """The red marker makes the part asymmetric: a part rotated by pi must not be read as theta = 0."""
    det = ShapeMatchDetector(TEMPLATE)
    r = det.detect(_frame(Pose(300, 200, math.pi)))
    assert abs(abs(r.theta) - math.pi) < math.radians(1.0)


def test_empty_image_gives_none(detector):
    img = np.full((400, 640, 3), 35, np.uint8)
    assert detector.detect(Frame(img, 0.0, 0)) is None


def test_other_shape_is_rejected():
    det = ShapeMatchDetector(TEMPLATE)
    img = np.full((400, 640, 3), 35, np.uint8)
    cv2.circle(img, (300, 200), 60, (230, 230, 230), -1)
    assert det.detect(Frame(img, 0.0, 0)) is None


def test_tracking_sequence_local_search_stays_accurate():
    src = SyntheticConveyorSource(width=1280, height=720, fps=30, velocity=(250, 15), omega=0.6,
                                  start=(300, 300, 0.0), realtime=False, max_frames=45, seed=1)
    det = ShapeMatchDetector(TEMPLATE)
    pos, ang = [], []
    while (f := src.read()) is not None:
        r = det.detect(f)
        assert r is not None, f"lost at frame {f.frame_id}"
        gt = f.ground_truth
        pos.append(math.hypot(r.x - gt.x, r.y - gt.y))
        ang.append(math.degrees(abs(wrap_angle(r.theta - gt.theta))))
    assert max(pos) < 1.0 and max(ang) < 1.0


def test_recovers_after_part_leaves_and_returns():
    det = ShapeMatchDetector(TEMPLATE, global_interval=1)
    assert det.detect(_frame(Pose(300, 200, 0.4))) is not None
    assert det.detect(Frame(np.full((400, 640, 3), 35, np.uint8), 0.1, 1)) is None
    r = det.detect(_frame(Pose(480, 120, -0.8), fid=2))
    assert r is not None and math.hypot(r.x - 480, r.y - 120) < 1.0


def test_global_search_is_throttled_when_nothing_is_found():
    det = ShapeMatchDetector(TEMPLATE, global_interval=3)
    empty = Frame(np.full((400, 640, 3), 35, np.uint8), 0.0, 0)
    calls = []
    orig = det._global
    det._global = lambda g: (calls.append(1), orig(g))[1]
    for _ in range(9):
        det.detect(empty)
    assert len(calls) == 3


@pytest.mark.parametrize("roi", [(10, 10, 201, 101), (10, 10, 200, 100), (10, 10, 200, 101)])
def test_template_size_parity_does_not_shift_pose(tmp_path, roi):
    """Pose origin is the template centre ((w-1)/2, (h-1)/2) for odd and even template sizes alike."""
    ref = _frame(Pose(110, 60, 0.0), size=(260, 120), noise=0.0)
    x, y, w, h = roi
    path = tmp_path / "t.png"
    cv2.imwrite(str(path), cv2.cvtColor(ref.image, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w])
    det = ShapeMatchDetector(str(path))
    r = det.detect(_frame(Pose(300, 200, 0.9)))
    ox, oy = (x + (w - 1) / 2) - 110, (y + (h - 1) / 2) - 60  # template centre relative to shape centre
    c, s = math.cos(0.9), math.sin(0.9)
    assert math.hypot(r.x - (300 + c * ox - s * oy), r.y - (200 + s * ox + c * oy)) < 0.3


def test_teach_tool_roundtrip(tmp_path):
    ref = _frame(Pose(300, 200, 0.0), noise=0.0)
    img_path, out = tmp_path / "ref.png", tmp_path / "t" / "part.png"
    cv2.imwrite(str(img_path), ref.image)
    assert teach.main(["--image", str(img_path), "--roi", "195,145,211,111", "--out", str(out), "--mask-auto"]) == 0
    mask = cv2.imread(str(out.with_name("part_mask.png")), cv2.IMREAD_GRAYSCALE)
    expected = 160 * 70 / (211 * 111)  # body area / ROI area = 0.478
    assert mask is not None and abs(mask.mean() / 255 - expected) < 0.03  # body only, not the whole ROI
    det = ShapeMatchDetector(str(out), mask=str(out.with_name("part_mask.png")))
    truth = Pose(420, 150, -1.1)
    r = det.detect(_frame(truth, seed=2))
    assert math.hypot(r.x - truth.x, r.y - truth.y) < 1.0
    assert abs(wrap_angle(r.theta - truth.theta)) < math.radians(1.5)


def test_teach_rejects_roi_outside_image(tmp_path):
    cv2.imwrite(str(tmp_path / "i.png"), np.zeros((50, 50, 3), np.uint8))
    with pytest.raises(SystemExit):
        teach.main(["--image", str(tmp_path / "i.png"), "--roi", "30,30,40,40", "--out", str(tmp_path / "o.png")])


def test_missing_template_and_bad_mask(tmp_path):
    with pytest.raises(FileNotFoundError, match="teach.py"):
        ShapeMatchDetector(str(tmp_path / "nope.png"))
    cv2.imwrite(str(tmp_path / "m.png"), np.zeros((5, 5), np.uint8))
    with pytest.raises(ValueError):
        ShapeMatchDetector(TEMPLATE, mask=str(tmp_path / "m.png"))


def test_pipeline_end_to_end_with_shape_match():
    cfg = load_config(ROOT / "config" / "synthetic_shape.yaml",
                      {"camera": {"realtime": False, "max_frames": 60, "seed": 5}, "output": {"type": "none"}})
    pipe = Pipeline.from_config(cfg)
    pipe.tracker.request_lock()
    errs = []
    while (step := pipe.step()) is not None:
        if step.state is TrackState.TRACKING and step.pose is not None:
            errs.append(step.pose.distance(step.frame.ground_truth))
    assert len(errs) > 50 and np.percentile(errs, 95) < 2.0
