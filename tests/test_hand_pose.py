import math

import numpy as np
import pytest

from ctrack.detector.hand import hand_pose_from_landmarks


def _landmarks(wrist, index_mcp, pinky_mcp):
    pts = np.zeros((21, 2))
    pts[0], pts[5], pts[17] = wrist, index_mcp, pinky_mcp
    # Fingers get random positions — they must not influence the pose.
    pts[[i for i in range(21) if i not in (0, 5, 17)]] = np.random.default_rng(0).uniform(0, 1000, (18, 2))
    return pts


def test_hand_pointing_right():
    pose = hand_pose_from_landmarks(_landmarks((100, 200), (200, 180), (200, 220)))
    assert pose.x == pytest.approx((100 + 200 + 200) / 3)
    assert pose.y == pytest.approx(200)
    assert pose.theta == pytest.approx(0.0)


def test_hand_pointing_up_in_image():
    # Image y grows downward, so "up" is -pi/2.
    pose = hand_pose_from_landmarks(_landmarks((300, 400), (290, 300), (310, 300)))
    assert pose.theta == pytest.approx(-math.pi / 2)


def test_hand_pointing_left_is_plus_pi():
    pose = hand_pose_from_landmarks(_landmarks((300, 300), (200, 300), (200, 300)))
    assert pose.theta == pytest.approx(math.pi)


def test_fingers_ignored():
    a = _landmarks((100, 200), (200, 180), (200, 220))
    b = a.copy()
    b[8] = (-500, -500)  # index fingertip somewhere else
    assert hand_pose_from_landmarks(a) == hand_pose_from_landmarks(b)
