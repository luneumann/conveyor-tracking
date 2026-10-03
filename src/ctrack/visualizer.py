"""Live overlay (P0-7). Optional: the pipeline core never imports this module."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .pipeline import StepResult
from .types import Pose, TrackState, wrap_angle

STATE_COLORS = {
    TrackState.SEARCHING: (200, 200, 200),
    TrackState.TRACKING: (246, 130, 59),
    TrackState.COASTING: (0, 190, 255),
    TrackState.LOST: (60, 60, 230),
}
FEATURE_COLOR = (255, 255, 255)   # the small feature region the pose is taken from
X_COLOR, Y_COLOR, PRED_COLOR = (80, 80, 255), (245, 245, 245), (230, 110, 200)


def _dashed_line(img: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], color: tuple[int, int, int],
                 thickness: int = 2, dash: float = 10.0) -> None:
    length = math.dist(p0, p1)
    n = max(int(length / dash), 1)
    for i in range(0, n, 2):
        a = i / n
        b = min((i + 1) / n, 1.0)
        q0 = (round(p0[0] + (p1[0] - p0[0]) * a), round(p0[1] + (p1[1] - p0[1]) * a))
        q1 = (round(p0[0] + (p1[0] - p0[0]) * b), round(p0[1] + (p1[1] - p0[1]) * b))
        cv2.line(img, q0, q1, color, thickness, cv2.LINE_AA)


def draw_axes(img: np.ndarray, pose: Pose, length: float = 90.0, dashed: bool = False,
              color: tuple[int, int, int] | None = None) -> None:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    o = (pose.x, pose.y)
    x_end = (pose.x + c * length, pose.y + s * length)
    y_end = (pose.x - s * length * 0.6, pose.y + c * length * 0.6)
    if dashed:
        _dashed_line(img, o, x_end, color or X_COLOR)
        _dashed_line(img, o, y_end, color or Y_COLOR)
    else:
        po, px, py = (round(o[0]), round(o[1])), (round(x_end[0]), round(x_end[1])), (round(y_end[0]), round(y_end[1]))
        # Dark outline first so the axes stay visible on any background.
        cv2.arrowedLine(img, po, px, (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.2)
        cv2.line(img, po, py, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.arrowedLine(img, po, px, color or X_COLOR, 3, cv2.LINE_AA, tipLength=0.2)
        cv2.line(img, po, py, color or Y_COLOR, 3, cv2.LINE_AA)
    cv2.circle(img, (round(o[0]), round(o[1])), 4, color or (255, 255, 255), -1, cv2.LINE_AA)


class OverlayRenderer:
    """Draws the overlay onto a copy of the frame. No window, usable headless (e.g. by the web GUI)."""

    def __init__(self, horizon_ms: float = 100.0, reacquire_radius_px: float = 80.0, hud: bool = True) -> None:
        self.horizon_ms = horizon_ms
        self.reacquire_radius = reacquire_radius_px
        self.hud = hud  # the web GUI shows these numbers itself

    def render(self, step: StepResult, stats: dict[str, float], reference: Pose | None = None,
               gate_radius: float | None = None) -> np.ndarray:
        img = step.frame.image.copy()
        det = step.detection
        if det is not None:
            if det.contour is not None:
                self._segment(img, det, step)
            if det.keypoints is not None:
                for x, y in det.keypoints[:, :2]:
                    cv2.circle(img, (round(x), round(y)), 3, (180, 180, 180), -1, cv2.LINE_AA)
            if step.state is TrackState.SEARCHING:
                draw_axes(img, det.pose, color=(200, 200, 200))
        if step.state is TrackState.LOST and reference is not None:
            cv2.circle(img, (round(reference.x), round(reference.y)), round(gate_radius or self.reacquire_radius),
                       STATE_COLORS[TrackState.LOST], 1, cv2.LINE_AA)
        compact = det is not None and det.contour is not None      # the segment itself shows the object: keep the axes small
        if step.pose is not None:
            draw_axes(img, step.pose, length=45.0 if compact else 90.0)
        if step.predicted is not None:
            draw_axes(img, step.predicted, length=45.0 if compact else 90.0, dashed=True, color=PRED_COLOR)
        if self.hud:
            self._hud(img, step, stats)
        return img

    @staticmethod
    def _follow(pts: np.ndarray, det, step: StepResult) -> np.ndarray:
        """Outline points of a possibly older detection, moved and turned by the way the tracked pose has moved since."""
        pts = np.asarray(pts, np.float32)
        if step.pose is not None and step.detection_t is not None and step.detection_t < step.frame.t_exposure:
            dth = wrap_angle(step.pose.theta - det.theta)
            if abs(dth) < 0.6:
                c, s_ = math.cos(dth), math.sin(dth)
                rel = pts - (det.x, det.y)
                return np.column_stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]]) + (step.pose.x, step.pose.y)
            return pts + (step.pose.x - det.x, step.pose.y - det.y)
        return pts

    @classmethod
    def _segment(cls, img: np.ndarray, det, step: StepResult) -> None:
        """The detected object's outline as a filled, translucent segment in the state colour (the big thing that is found),
        plus - when a feature anchor is used - the small feature region the position is taken from.

        With asynchronous detection the outline is a few frames old; it is moved and turned by the way the tracked
        pose has moved since, so the segment sits on the object in the live image instead of trailing behind it."""
        poly = np.round(cls._follow(det.contour, det, step)).astype(np.int32)
        color = STATE_COLORS[step.state] if step.state is not TrackState.SEARCHING else (210, 190, 40)
        fill = img.copy()
        cv2.fillPoly(fill, [poly], color)
        cv2.addWeighted(fill, 0.35, img, 0.65, 0, dst=img)
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
        if det.feature is not None:
            fp = np.round(cls._follow(det.feature, det, step)).astype(np.int32)
            cv2.polylines(img, [fp], True, (0, 0, 0), 5, cv2.LINE_AA)          # dark outline first: visible on any background
            if det.feature_matched:
                cv2.polylines(img, [fp], True, FEATURE_COLOR, 2, cv2.LINE_AA)
            else:                                                              # only where the mask predicts it: dashed
                for i in range(4):
                    _dashed_line(img, tuple(fp[i]), tuple(fp[(i + 1) % 4]), FEATURE_COLOR, 2, dash=7.0)

    def _hud(self, img: np.ndarray, step: StepResult, stats: dict[str, float]) -> None:
        lines = [
            (f"{step.state.value}", STATE_COLORS[step.state], 0.9),
            (f"fps {stats['fps']:.1f}   latency p50 {stats['latency_p50']:.0f} / p95 {stats['latency_p95']:.0f} ms",
             (255, 255, 255), 0.6),
            (f"pred err @{self.horizon_ms:.0f}ms p50 {stats['pred_err_p50']:.1f} / p95 {stats['pred_err_p95']:.1f} px"
             f" ({stats['pred_err_p95_pct']:.2f}% width), p95 {stats['pred_err_deg_p95']:.1f} deg",
             (255, 255, 255), 0.6),
        ]
        if not math.isnan(stats.get("gt_err_p95", math.nan)):
            lines.append((f"ground-truth err p95 {stats['gt_err_p95']:.1f} px", (255, 255, 255), 0.6))
        if step.velocity is not None:
            v = step.velocity
            lines.append((f"v ({v.vx:.0f}, {v.vy:.0f}) px/s  omega {v.omega:.2f} rad/s", (255, 255, 255), 0.6))
        lines.append(("[L] lock  [R] reset  [Q] quit", (180, 180, 180), 0.5))

        pad, y = 10, 10
        heights = [int(28 * s / 0.6) for _, _, s in lines]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (620, sum(heights) + 2 * pad), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
        for (text, color, scale), h in zip(lines, heights):
            y += h
            cv2.putText(img, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale > 0.6 else 1,
                        cv2.LINE_AA)



class Visualizer(OverlayRenderer):
    """OpenCV window with keyboard handling."""

    def __init__(self, window: str = "ctrack", horizon_ms: float = 100.0, reacquire_radius_px: float = 80.0) -> None:
        super().__init__(horizon_ms, reacquire_radius_px)
        self.window = window
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def show(self, step: StepResult, stats: dict[str, float], reference: Pose | None = None,
             gate_radius: float | None = None) -> str | None:
        cv2.imshow(self.window, self.render(step, stats, reference, gate_radius))
        key = cv2.waitKey(1) & 0xFF
        return chr(key).lower() if key != 255 else None

    def close(self) -> None:
        cv2.destroyWindow(self.window)
