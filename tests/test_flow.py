import math

import cv2
import numpy as np

from ctrack.flow import FlowTracker
from ctrack.predictor import KalmanPredictor
from ctrack.types import Detection, Pose


def _scene(dx=0.0, dy=0.0, rot_deg=0.0):
    """Textured square 'object' on a plain background, moved/rotated about its centre (400, 300)."""
    rng = np.random.default_rng(3)
    tex = (rng.random((40, 40)) * 255).astype(np.uint8)
    tex = cv2.resize(tex, (200, 200), interpolation=cv2.INTER_CUBIC)
    img = np.full((600, 800, 3), 90, np.uint8)
    obj = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    M = cv2.getRotationMatrix2D((100, 100), rot_deg, 1.0)
    M[:, 2] += (400 - 100 + dx, 300 - 100 + dy)
    mask = cv2.warpAffine(np.full((200, 200), 255, np.uint8), M, (800, 600))
    warped = cv2.warpAffine(obj, M, (800, 600))
    img[mask > 0] = warped[mask > 0]
    c = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], np.float32)
    return img, c @ M[:, :2].T + M[:, 2]


def test_flow_follows_translation_and_rotation():
    img0, poly0 = _scene()
    ft = FlowTracker()
    assert ft.reset(img0, 400, 300, 0.0, poly0, 0.0)
    pose = None
    for k in range(1, 9):
        img, _ = _scene(dx=3.0 * k, dy=-2.0 * k, rot_deg=1.5 * k)
        pose = ft.step(img, k / 30)
        assert pose is not None
    x, y, th = pose
    assert abs(x - (400 + 24)) < 2.5 and abs(y - (300 - 16)) < 2.5
    assert abs(math.degrees(th) - (-12.0)) < 1.5 or abs(math.degrees(th) - 12.0) < 1.5      # sign convention of the rotation matrix


def test_flow_gives_up_without_texture_and_when_too_old():
    img0, poly0 = _scene()
    ft = FlowTracker(max_age_s=0.2)
    ft.reset(img0, 400, 300, 0.0, poly0, 0.0)
    assert ft.step(np.full((600, 800, 3), 90, np.uint8), 0.03) is None and not ft.active      # object vanished
    ft.reset(img0, 400, 300, 0.0, poly0, 0.0)
    assert ft.step(img0, 0.5) is None                                                         # no detection for too long


def test_flow_correction_carries_a_late_detection_forward():
    ft = FlowTracker()
    img0, poly0 = _scene()
    ft.reset(img0, 400, 300, 0.0, poly0, 0.0)
    for k in range(1, 5):
        img, _ = _scene(dx=4.0 * k)
        ft.step(img, k / 30)
    # a detection of frame 0 (true position 400) arrives now, at frame 4 where the object really is at 416
    ft.correct(img, 0.0, 400.0, 300.0, 0.0, poly0, 4 / 30)
    x, y, _ = ft.step(_scene(dx=20.0)[0], 5 / 30)
    assert abs(x - 420) < 3 and abs(y - 300) < 3


def test_measurement_noise_scale_weakens_a_measurement():
    def run(scale):
        kf = KalmanPredictor(process_noise=50.0, measurement_noise=2.0)
        kf.update(Pose(0, 0, 0), 0.0)
        kf.update(Pose(10, 0, 0), 0.033)
        before = kf.predict(0.066).x
        kf.update(Pose(before + 10, 0, 0), 0.066, noise_scale=scale)
        return kf.predict(0.066).x - before
    assert run(1.0) > 2 * run(10.0) > 0
