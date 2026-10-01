import json
import math

import pytest

from ctrack.types import MESSAGE_VERSION, Message, Pose, TrackState, Velocity, wrap_angle


@pytest.mark.parametrize("a, expected", [
    (0.0, 0.0),
    (math.pi, math.pi),
    (-math.pi, math.pi),  # (-pi, pi]: -pi maps to +pi
    (3 * math.pi, math.pi),
    (2 * math.pi, 0.0),
    (math.pi + 0.1, -math.pi + 0.1),
    (-math.pi - 0.1, math.pi - 0.1),
    (7.0, 7.0 - 2 * math.pi),
])
def test_wrap_angle(a, expected):
    assert wrap_angle(a) == pytest.approx(expected, abs=1e-12)


def test_angle_diff_across_pi():
    a, b = Pose(0, 0, math.pi - 0.05), Pose(0, 0, -math.pi + 0.05)
    assert a.angle_diff(b) == pytest.approx(0.1)


def _msg(state, pose=Pose(1, 2, 0.3), velocity=Velocity(4, 5, 0.1), **kw):
    return Message(seq=7, state=state, t_exposure=10.0, t_sent=10.05, pose=pose, velocity=velocity,
                   unit="px", frame="image", confidence=0.9, **kw)


def test_tracking_message_requires_pose():
    with pytest.raises(ValueError):
        _msg(TrackState.TRACKING, pose=None, velocity=None)
    with pytest.raises(ValueError):
        _msg(TrackState.TRACKING, pose=Pose(math.nan, 0, 0))


def test_non_tracking_message_may_omit_pose():
    m = _msg(TrackState.SEARCHING, pose=None, velocity=None)
    assert m.to_dict()["pose"] is None and m.to_dict()["velocity"] is None


def test_message_schema_matches_prd():
    d = _msg(TrackState.TRACKING, predicted=Pose(1.5, 2, 0.3), t_predicted=10.15).to_dict()
    assert set(d) == {"v", "seq", "state", "t_exposure", "t_sent", "pose", "velocity", "unit", "frame",
                      "confidence", "predicted"}
    assert d["v"] == MESSAGE_VERSION
    assert d["state"] == "TRACKING"
    assert set(d["pose"]) == {"x", "y", "theta"}
    assert set(d["velocity"]) == {"vx", "vy", "omega"}
    assert set(d["predicted"]) == {"t", "x", "y", "theta"}
    json.dumps(d)  # serializable
