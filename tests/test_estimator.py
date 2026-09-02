"""Attitude EKF tests: convergence, bias estimation, covariance health."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.estimator import AttitudeEKF


def level_accel(roll=0.0, pitch=0.0, g=9.81):
    """Specific force measured by an accelerometer at a given static attitude."""
    return np.array(
        [
            -np.sin(pitch) * g,
            np.sin(roll) * np.cos(pitch) * g,
            np.cos(roll) * np.cos(pitch) * g,
        ]
    )


def test_ekf_converges_to_a_static_tilt():
    dt = 0.01
    true_roll, true_pitch = np.deg2rad(15.0), np.deg2rad(-8.0)
    ekf = AttitudeEKF()
    a = level_accel(true_roll, true_pitch)
    for _ in range(int(30.0 / dt)):
        ekf.step([0.0, 0.0, 0.0], a, dt)
    roll, pitch, _ = ekf.euler
    assert roll == pytest.approx(true_roll, abs=np.deg2rad(1.0))
    assert pitch == pytest.approx(true_pitch, abs=np.deg2rad(1.0))


def test_ekf_estimates_gyro_bias():
    """The reason to use an EKF over Madgwick: the bias becomes a state and
    converges, rather than being permanently fought by a fixed gain."""
    dt = 0.02
    bias = np.array([np.deg2rad(2.0), np.deg2rad(-1.5), 0.0])
    ekf = AttitudeEKF(gyro_bias_noise=1e-6)
    a = level_accel(0.0, 0.0)
    for _ in range(int(90.0 / dt)):
        ekf.step(bias, a, dt)
    est = ekf.gyro_bias
    assert est[0] == pytest.approx(bias[0], abs=np.deg2rad(0.5))
    assert est[1] == pytest.approx(bias[1], abs=np.deg2rad(0.5))
    # Attitude stays level despite the bias, which is the point.
    roll, pitch, _ = ekf.euler
    assert abs(roll) < np.deg2rad(1.0)
    assert abs(pitch) < np.deg2rad(1.0)


def test_ekf_covariance_stays_symmetric_and_psd():
    """Joseph form guarantee, checked every single step for a long run."""
    dt = 0.005
    rng = np.random.default_rng(4)
    ekf = AttitudeEKF()
    for i in range(3000):
        gyro = rng.normal(0.0, 0.05, 3)
        a = level_accel(np.deg2rad(5 * np.sin(i * dt)), 0.0) + rng.normal(0.0, 0.15, 3)
        ekf.step(gyro, a, dt)
        assert ekf.is_covariance_psd(), f"covariance lost PSD at step {i}"
    assert np.allclose(ekf.P, ekf.P.T)
    assert np.min(np.linalg.eigvalsh(ekf.P)) > -1e-9


def test_ekf_quaternion_stays_unit_norm():
    dt = 0.004
    ekf = AttitudeEKF()
    rng = np.random.default_rng(9)
    for i in range(2000):
        ekf.step(rng.normal(0.0, 1.0, 3), level_accel(0.0, 0.0), dt)
        assert np.linalg.norm(ekf.quaternion) == pytest.approx(1.0, abs=1e-9)


def test_ekf_rejects_the_accelerometer_during_a_hard_manoeuvre():
    """Gating: under 2 g the accelerometer is not measuring gravity, and using
    it as a tilt observation is how attitude estimators tip over."""
    ekf = AttitudeEKF(accel_gate=0.15)
    ekf.predict([0.0, 0.0, 0.0], 0.005)
    accepted = ekf.update_accel(level_accel(0.0, 0.0) * 2.0)
    assert accepted is False
    assert "gate" in ekf.diagnostics.reason
    # A normal-magnitude reading is accepted.
    assert ekf.update_accel(level_accel(0.0, 0.0)) is True


def test_ekf_nis_is_small_when_healthy_and_large_when_lied_to():
    dt = 0.005
    ekf = AttitudeEKF()
    a = level_accel(0.0, 0.0)
    for _ in range(2000):
        ekf.step([0.0, 0.0, 0.0], a, dt)
    healthy_nis = ekf.diagnostics.nis
    assert ekf.diagnostics.accepted
    assert healthy_nis < 5.0

    # Now feed an attitude 40 degrees away from what the filter believes; the
    # innovation should be large enough that NIS flags it.
    ekf.nis_gate = None
    ekf.predict([0.0, 0.0, 0.0], dt)
    ekf.update_accel(level_accel(np.deg2rad(40.0), 0.0))
    assert ekf.diagnostics.nis > 10.0 * max(healthy_nis, 1e-6)


def test_ekf_nis_gate_rejects_an_outlier():
    dt = 0.005
    ekf = AttitudeEKF(nis_gate=5.0)
    a = level_accel(0.0, 0.0)
    for _ in range(2000):
        ekf.step([0.0, 0.0, 0.0], a, dt)
    before = ekf.quaternion.copy()
    ekf.predict([0.0, 0.0, 0.0], dt)
    accepted = ekf.update_accel(level_accel(np.deg2rad(60.0), 0.0))
    assert accepted is False
    assert np.allclose(ekf.quaternion, before, atol=1e-6)


def test_ekf_tracks_a_rotating_platform():
    """Propagation must be right, not just the correction: rotate at a constant
    rate for a second and the estimate must follow."""
    dt = 0.002
    rate = np.deg2rad(30.0)
    ekf = AttitudeEKF(accel_gate=0.05)
    for i in range(int(1.0 / dt)):
        angle = rate * i * dt
        # Accel is consistent with the true tilt at every instant.
        ekf.step([rate, 0.0, 0.0], level_accel(angle, 0.0), dt)
    roll, _, _ = ekf.euler
    assert roll == pytest.approx(rate * 1.0, rel=0.05)


def test_ekf_magnetometer_constrains_yaw():
    dt = 0.01
    ekf = AttitudeEKF()
    mag = np.array([0.5, 0.0, 0.3])
    for _ in range(int(20.0 / dt)):
        ekf.step([0.0, 0.0, np.deg2rad(1.0)], level_accel(0.0, 0.0), dt, mag=mag)
    _, _, yaw = ekf.euler
    assert abs(yaw) < np.deg2rad(20.0)


def test_ekf_reset_restores_the_initial_state():
    ekf = AttitudeEKF()
    p0 = ekf.P.copy()
    for _ in range(200):
        ekf.step([0.1, 0.0, 0.0], level_accel(0.2, 0.0), 0.005)
    assert not np.allclose(ekf.P, p0)
    ekf.reset()
    assert np.allclose(ekf.P, p0)
    assert np.allclose(ekf.quaternion, [1.0, 0.0, 0.0, 0.0])


def test_ekf_input_validation():
    ekf = AttitudeEKF()
    with pytest.raises(ValueError):
        ekf.predict([0.0, 0.0, 0.0], 0.0)
    with pytest.raises(ValueError):
        ekf.predict([0.0, 0.0], 0.01)
    with pytest.raises(ValueError):
        ekf.update_accel([0.0, 0.0])
    with pytest.raises(ValueError):
        AttitudeEKF(accel_noise=0.0)
