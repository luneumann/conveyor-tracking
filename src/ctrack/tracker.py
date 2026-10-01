"""Tracking state machine (PRD 5.2, ADR-004). See docs/SYSTEM-DESIGN.md §3."""

from __future__ import annotations

from .predictor import KalmanPredictor
from .types import Detection, Pose, TrackState, Velocity


class Tracker:
    def __init__(self, predictor: KalmanPredictor, coast_ms: float = 300.0,
                 reacquire_radius_px: float = 80.0) -> None:
        self.predictor = predictor
        self.coast_s = coast_ms / 1000.0
        self.reacquire_radius = reacquire_radius_px
        self.state = TrackState.SEARCHING
        self.lock_requested = False
        self.t_last_seen: float | None = None
        self.t_lost: float | None = None

    def request_lock(self) -> None:
        """Key L: lock on the next valid detection (only meaningful in SEARCHING)."""
        if self.state is TrackState.SEARCHING:
            self.lock_requested = True

    def reset(self) -> None:
        """Key R: drop the target and go back to SEARCHING."""
        self.state = TrackState.SEARCHING
        self.lock_requested = False
        self.t_last_seen = None
        self.t_lost = None
        self.predictor.reset()

    def reference_pose(self, t: float) -> Pose | None:
        """Where we expect the target; extrapolation is capped at coast_ms after the last sighting."""
        if not self.predictor.initialized or self.t_last_seen is None:
            return None
        return self.predictor.predict(min(t, self.t_last_seen + self.coast_s))

    def update(self, detection: Detection | None, t: float) -> TrackState:
        s = self.state
        if s is TrackState.SEARCHING:
            if self.lock_requested and detection is not None:
                self._acquire(detection, t)
        elif s in (TrackState.TRACKING, TrackState.COASTING):
            if detection is not None:
                self.predictor.update(detection.pose, t)
                self.t_last_seen = t
                self.state = TrackState.TRACKING
            elif t - self.t_last_seen > self.coast_s:
                self.state = TrackState.LOST
                self.t_lost = t
            else:
                self.state = TrackState.COASTING
        elif s is TrackState.LOST:
            if detection is not None:
                ref = self.reference_pose(t)
                if ref is not None and ref.distance(detection.pose) <= self.reacquire_radius:
                    self._acquire(detection, t)
        return self.state

    def _acquire(self, detection: Detection, t: float) -> None:
        # Re-init rather than update: after SEARCHING/LOST the velocity estimate is stale.
        self.predictor.init(detection.pose, t)
        self.state = TrackState.TRACKING
        self.lock_requested = False
        self.t_last_seen = t
        self.t_lost = None

    def estimate(self, t: float) -> tuple[Pose, Velocity] | None:
        """Pose to publish at time t: filter estimate in TRACKING, prediction in COASTING, else None."""
        if self.state in (TrackState.TRACKING, TrackState.COASTING):
            return self.predictor.predict(t), self.predictor.velocity()
        return None
