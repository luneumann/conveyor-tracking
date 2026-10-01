"""Live overlay (P0-7). Optional: the pipeline core never imports this module."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .pipeline import StepResult
from .types import Pose, TrackState

STATE_COLORS = {
    TrackState.SEARCHING: (200, 200, 200),
    TrackState.TRACKING: (80, 200, 80),
    TrackState.COASTING: (0, 190, 255),
    TrackState.LOST: (60, 60, 230),
}
X_COLOR, Y_COLOR, PRED_COLOR = (60, 60, 255), (80, 220, 80), (255, 200, 0)


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


def draw_axes(img: np.ndarray, pose: Pose, length: float = 60.0, dashed: bool = False,
              color: tuple[int, int, int] | None = None) -> None:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    o = (pose.x, pose.y)
    x_end = (pose.x + c * length, pose.y + s * length)
    y_end = (pose.x - s * length * 0.6, pose.y + c * length * 0.6)
    if dashed:
        _dashed_line(img, o, x_end, color or X_COLOR)
        _dashed_line(img, o, y_end, color or Y_COLOR)
    else:
        cv2.arrowedLine(img, (round(o[0]), round(o[1])), (round(x_end[0]), round(x_end[1])),
                        color or X_COLOR, 3, cv2.LINE_AA, tipLength=0.2)
        cv2.line(img, (round(o[0]), round(o[1])), (round(y_end[0]), round(y_end[1])), color or Y_COLOR, 3, cv2.LINE_AA)
    cv2.circle(img, (round(o[0]), round(o[1])), 4, color or (255, 255, 255), -1, cv2.LINE_AA)


class Visualizer:
    def __init__(self, window: str = "ctrack", horizon_ms: float = 100.0, reacquire_radius_px: float = 80.0) -> None:
        self.window = window
        self.horizon_ms = horizon_ms
        self.reacquire_radius = reacquire_radius_px
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def render(self, step: StepResult, stats: dict[str, float], reference: Pose | None = None) -> np.ndarray:
        img = step.frame.image.copy()
        det = step.detection
        if det is not None:
            if det.keypoints is not None:
                for x, y in det.keypoints[:, :2]:
                    cv2.circle(img, (round(x), round(y)), 3, (180, 180, 180), -1, cv2.LINE_AA)
            if step.state is TrackState.SEARCHING:
                draw_axes(img, det.pose, color=(200, 200, 200))
        if step.state is TrackState.LOST and reference is not None:
            cv2.circle(img, (round(reference.x), round(reference.y)), round(self.reacquire_radius),
                       STATE_COLORS[TrackState.LOST], 1, cv2.LINE_AA)
        if step.pose is not None:
            draw_axes(img, step.pose)
        if step.predicted is not None:
            draw_axes(img, step.predicted, dashed=True, color=PRED_COLOR)
        self._hud(img, step, stats)
        return img

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

    def show(self, step: StepResult, stats: dict[str, float], reference: Pose | None = None) -> str | None:
        cv2.imshow(self.window, self.render(step, stats, reference))
        key = cv2.waitKey(1) & 0xFF
        return chr(key).lower() if key != 255 else None

    def close(self) -> None:
        cv2.destroyWindow(self.window)
