import math

import numpy as np
import pytest

from ctrack.predictor import KalmanPredictor
from ctrack.types import Pose, wrap_angle


def _run_constant_velocity(kf, v=(200.0, -50.0, 0.5), n=90, fps=30.0, noise=2.0, rng=None, theta0=0.0):
    rng = rng or np.random.default_rng(1)
    for i in range(n):
        t = i / fps
        kf.update(Pose(100 + v[0] * t + rng.normal(0, noise), 300 + v[1] * t + rng.normal(0, noise),
                       wrap_angle(theta0 + v[2] * t + rng.normal(0, 0.01))), t)
    return (n - 1) / fps


def test_first_update_initializes_with_zero_velocity():
    kf = KalmanPredictor()
    kf.update(Pose(10, 20, 0.5), 1.0)
    assert kf.initialized
    assert kf.predict(1.0) == Pose(10, 20, 0.5)
    assert kf.velocity().vx == 0 and kf.velocity().vy == 0


def test_converges_to_constant_velocity():
    kf = KalmanPredictor(measurement_noise=2.0)
    _run_constant_velocity(kf)
    v = kf.velocity()
    assert v.vx == pytest.approx(200, abs=10)
    assert v.vy == pytest.approx(-50, abs=10)
    assert v.omega == pytest.approx(0.5, abs=0.1)


def test_predict_future_pose():
    kf = KalmanPredictor(measurement_noise=2.0)
    t_last = _run_constant_velocity(kf)
    t = t_last + 0.1
    p = kf.predict(t)
    assert p.x == pytest.approx(100 + 200 * t, abs=4)
    assert p.y == pytest.approx(300 - 50 * t, abs=4)


def test_predict_does_not_mutate_state():
    kf = KalmanPredictor()
    _run_constant_velocity(kf, n=10)
    x, P, t = kf.x.copy(), kf.P.copy(), kf.t
    kf.predict(t + 5.0)
    assert np.array_equal(kf.x, x) and np.array_equal(kf.P, P) and kf.t == t


def test_theta_crossing_pi_has_no_jump():
    """Rotate steadily through +pi -> -pi; estimates must stay close to truth, never jump by ~2pi."""
    kf = KalmanPredictor(measurement_noise_theta=0.01)
    omega, fps = 1.0, 30.0
    theta0 = math.pi - 1.0
    max_err = 0.0
    for i in range(90):  # 3 s -> passes pi after 1 s
        t = i / fps
        truth = wrap_angle(theta0 + omega * t)
        kf.update(Pose(0, 0, truth), t)
        if i > 10:
            max_err = max(max_err, abs(wrap_angle(kf.predict(t).theta - truth)))
        assert -math.pi < kf.predict(t).theta <= math.pi
    assert max_err < 0.05
    assert kf.velocity().omega == pytest.approx(omega, abs=0.05)


def test_predict_across_pi_wraps():
    kf = KalmanPredictor()
    _run_constant_velocity(kf, v=(0, 0, 2.0), theta0=math.pi - 3.0, n=40, noise=0.0)
    p = kf.predict(kf.t + 0.5)
    assert -math.pi < p.theta <= math.pi


def test_out_of_order_timestamp_is_ignored_for_propagation():
    kf = KalmanPredictor()
    kf.update(Pose(0, 0, 0), 1.0)
    kf.update(Pose(1, 0, 0), 0.9)  # stale measurement: fused without propagating backwards
    assert kf.t == 1.0
    assert np.all(np.isfinite(kf.x))


def test_reset():
    kf = KalmanPredictor()
    kf.update(Pose(0, 0, 0), 0.0)
    kf.reset()
    assert not kf.initialized
