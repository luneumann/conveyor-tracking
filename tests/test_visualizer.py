import numpy as np

from ctrack.pipeline import StepResult
from ctrack.types import Detection, Frame, Message, Pose, TrackState, Velocity
from ctrack.visualizer import STATE_COLORS, OverlayRenderer


def _step(state, det, pose, detection_t):
    f = Frame(np.full((200, 300, 3), 90, np.uint8), 1.0, 0)
    msg = Message(seq=0, state=state, t_exposure=1.0, t_sent=1.0, pose=pose, velocity=Velocity(0, 0, 0) if pose else None,
                  unit="px", frame="image", confidence=1.0)
    return StepResult(f, 1.0, det, state, pose, Velocity(0, 0, 0) if pose else None, None, None, msg, True, 5.0, detection_t, False, 0.0)


def _square(cx, cy, r=20):
    return np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]], float)


def test_segment_is_filled_in_the_state_colour():
    det = Detection(100, 100, 0.0, 0.9, contour=_square(100, 100))
    img = OverlayRenderer(hud=False).render(_step(TrackState.TRACKING, det, Pose(100, 100, 0.0), 1.0), {})
    inside, outside = img[110, 90].astype(int), img[10, 10].astype(int)
    assert np.allclose(outside, 90)
    assert np.linalg.norm(inside - 90) > 20 and inside[1] > inside[0]            # tinted green (TRACKING)
    assert STATE_COLORS[TrackState.TRACKING][1] > STATE_COLORS[TrackState.TRACKING][0]


def test_old_segment_follows_the_tracked_pose():
    """Async detection: the outline belongs to an earlier frame; it must be drawn where the object is now."""
    det = Detection(100, 100, 0.0, 0.9, contour=_square(100, 100))
    img = OverlayRenderer(hud=False).render(_step(TrackState.TRACKING, det, Pose(160, 100, 0.0), 0.9), {})
    assert np.linalg.norm(img[110, 150].astype(int) - 90) > 20                    # new position tinted
    assert np.allclose(img[110, 90].astype(int), 90, atol=3)                      # old position untouched
