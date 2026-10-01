"""Constant-velocity Kalman filter over (x, y, theta) (P0-4, ADR-005)."""

from __future__ import annotations

import numpy as np

from .types import Pose, Velocity, wrap_angle

# State layout: [x, y, theta, vx, vy, omega]
_THETA = 2
_H = np.hstack([np.eye(3), np.zeros((3, 3))])


class KalmanPredictor:
    def __init__(self, process_noise: float = 50.0, measurement_noise: float = 4.0,
                 process_noise_theta: float = 2.0, measurement_noise_theta: float = 0.05,
                 initial_velocity_std: float = 500.0, initial_omega_std: float = 3.0,
                 horizon_ms: float = 100.0) -> None:
        # Spectral densities of the white-noise acceleration per axis.
        self.q = np.array([process_noise, process_noise, process_noise_theta])
        self.R = np.diag([measurement_noise**2, measurement_noise**2, measurement_noise_theta**2])
        self.initial_std = np.array([measurement_noise, measurement_noise, measurement_noise_theta,
                                     initial_velocity_std, initial_velocity_std, initial_omega_std])
        self.horizon_s = horizon_ms / 1000.0
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.t: float | None = None

    @property
    def initialized(self) -> bool:
        return self.t is not None

    def reset(self) -> None:
        self.t = None

    def init(self, pose: Pose, t: float) -> None:
        self.x = np.array([pose.x, pose.y, pose.theta, 0.0, 0.0, 0.0])
        self.P = np.diag(self.initial_std**2)
        self.t = t

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        # Continuous white noise acceleration model, per axis: q * [[dt^3/3, dt^2/2], [dt^2/2, dt]].
        Q = np.zeros((6, 6))
        for i, q in enumerate(self.q):
            Q[i, i] = q * dt**3 / 3
            Q[i, i + 3] = Q[i + 3, i] = q * dt**2 / 2
            Q[i + 3, i + 3] = q * dt
        return F, Q

    def _propagate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.t is not None, "predictor not initialized"
        dt = t - self.t
        if dt <= 0:
            return self.x.copy(), self.P.copy()
        F, Q = self._transition(dt)
        x = F @ self.x
        x[_THETA] = wrap_angle(x[_THETA])
        return x, F @ self.P @ F.T + Q

    def update(self, pose: Pose, t: float) -> None:
        """Advance to time t and fuse a measurement. Initializes on first call."""
        if not self.initialized:
            self.init(pose, t)
            return
        x, P = self._propagate(t)
        z = np.array([pose.x, pose.y, pose.theta])
        y = z - _H @ x
        y[_THETA] = wrap_angle(y[_THETA])  # innovation across ±pi must not jump
        S = _H @ P @ _H.T + self.R
        K = P @ _H.T @ np.linalg.inv(S)
        x = x + K @ y
        x[_THETA] = wrap_angle(x[_THETA])
        I_KH = np.eye(6) - K @ _H
        self.P = I_KH @ P @ I_KH.T + K @ self.R @ K.T  # Joseph form, stays symmetric PSD
        self.x = x
        self.t = max(t, self.t)

    def predict(self, t_target: float) -> Pose:
        """Pose at an arbitrary (usually future) time, without changing the filter state."""
        x, _ = self._propagate(t_target)
        return Pose(float(x[0]), float(x[1]), float(x[2]))

    def velocity(self) -> Velocity:
        return Velocity(float(self.x[3]), float(self.x[4]), float(self.x[5]))

    def position_std(self, t: float) -> float:
        """1-sigma position uncertainty (px) at time t."""
        _, P = self._propagate(t)
        return float(np.sqrt(P[0, 0] + P[1, 1]))
