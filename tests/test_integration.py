"""End-to-end tests: the modules have to work together, not just alone.

These are the tests that would catch a sign convention that is self-consistent
inside one module and wrong at the boundary between two.
"""

from __future__ import annotations

import numpy as np
import pytest

import dctk
from dctk.cascade import MultirotorCascade
from dctk.estimator import AttitudeEKF
from dctk.filters import BiquadLowPass, ComplementaryFilter, NotchFilter
from dctk.lqr import brysons_rule, c2d, dlqr
from dctk.metrics import step_metrics
from dctk.mixer import Mixer, quad_x
from dctk.pid import PID, AntiWindup
from dctk.sim import (
    ActuatorDelay,
    MotorLag,
    PitchPlant,
    Quantiser,
    SecondOrderPlant,
    SensorNoise,
    simulate,
)
from dctk.trajectory import min_jerk, min_jerk_duration


def test_package_exports_everything_it_advertises():
    for name in dctk.__all__:
        assert hasattr(dctk, name), f"dctk.__all__ advertises {name} but it is missing"
    assert dctk.__version__


def test_importing_dctk_does_not_import_matplotlib():
    """Tests and embedded use must never need a display or a plotting stack."""
    import sys

    assert "matplotlib" not in sys.modules


def test_pid_closes_the_rate_loop_on_a_realistic_pitch_axis():
    """Everything at once: real plant, gyro noise, ESC quantisation, transport
    delay, motor lag, and a filtered D term. The loop must still track."""
    dt = 0.001
    # measure='rate' -> the loop sees the gyro, which is what a rate loop closes on.
    plant = PitchPlant(inertia=0.02, damping=0.002, measure="rate")
    pid = PID(
        kp=0.35, ki=1.2, kd=0.012,
        output_limits=(-0.6, 0.6),
        anti_windup=AntiWindup.BACK_CALCULATION,
        tracking_time_constant=0.2,
        derivative_cutoff_hz=40.0,
    )

    result = simulate(
        plant,
        pid.update,
        duration=3.0,
        dt=dt,
        setpoint=np.deg2rad(120.0),
        sensor_noise=SensorNoise(sigma=np.deg2rad(1.5), seed=13),
        actuator_delay=ActuatorDelay(0.004, dt),
        motor_lag=MotorLag(0.025),
        actuator_quantiser=Quantiser(step=0.6 / 2048.0),
    )
    m = step_metrics(result.t, result.y, np.deg2rad(120.0))
    assert abs(m.steady_state_error) < np.deg2rad(4.0)
    assert m.overshoot_pct < 40.0
    assert np.isfinite(m.rise_time) and m.rise_time < 0.3


def test_lqr_beats_a_detuned_pid_on_the_same_plant_and_metrics_prove_it():
    """Both controllers, one plant, one metric set. No hand-waving."""
    dt = 0.002
    wn, zeta = 5.0, 0.1
    plant_args = dict(wn=wn, zeta=zeta, dc_gain=1.0)

    ref = SecondOrderPlant.from_modal(**plant_args)
    m_, c_, k_ = ref.mass, ref.damping, ref.stiffness
    A = np.array([[0.0, 1.0], [-k_ / m_, -c_ / m_]])
    B = np.array([[0.0], [1.0 / m_]])
    Ad, Bd = c2d(A, B, dt)
    Q, R = brysons_rule([0.05, 1.0], [20.0], rho=1.0)
    res = dlqr(Ad, Bd, Q, R)
    assert res.is_stable

    def lqr_controller(sp, meas, _dt, plant=None):
        x = np.array([plant.x[0] - sp, plant.x[1]])
        return float((-res.K @ x)[0])

    lqr_plant = SecondOrderPlant.from_modal(**plant_args)
    lqr_result = simulate(
        lqr_plant,
        lambda sp, meas, d: lqr_controller(sp, meas, d, plant=lqr_plant)
        + k_ * sp,  # feed-forward to hold the setpoint against the spring
        duration=6.0,
        dt=dt,
        setpoint=1.0,
    )
    lqr_metrics = step_metrics(lqr_result.t, lqr_result.y, 1.0)

    pid = PID(kp=3.0, ki=0.5, kd=0.2, output_limits=(-50, 50), derivative_cutoff_hz=40.0)
    pid_result = simulate(
        SecondOrderPlant.from_modal(**plant_args), pid.update, duration=6.0, dt=dt, setpoint=1.0
    )
    pid_metrics = step_metrics(pid_result.t, pid_result.y, 1.0)

    assert lqr_metrics.overshoot_pct < pid_metrics.overshoot_pct
    assert lqr_metrics.itae < pid_metrics.itae


def test_mixer_consumes_cascade_output_without_violating_motor_limits():
    """The boundary that matters on a real vehicle: whatever the attitude
    controller demands, the motors stay legal."""
    mc = MultirotorCascade()
    mixer = Mixer(quad_x(), idle=0.06)
    dt = 0.002
    theta, q = 0.0, 0.0
    inertia, damping = 0.01, 0.002
    saturated_samples = 0
    for _ in range(int(3.0 / dt)):
        torque = mc.position[2].update(np.deg2rad(35.0), theta, dt)
        torque = mc.position[3].update(torque, q, dt)
        u = mixer.mix(torque * 2.0, 0.0, 0.0, 0.6)
        assert np.all(u >= 0.06 - 1e-12)
        assert np.all(u <= 1.0 + 1e-12)
        if mixer.last_saturation > 0.0:
            saturated_samples += 1
        alpha = (np.clip(torque, -0.5, 0.5) - damping * q) / inertia
        q += alpha * dt
        theta += q * dt
    assert theta == pytest.approx(np.deg2rad(35.0), abs=np.deg2rad(3.0))


def test_estimator_output_can_drive_a_controller():
    """Estimate attitude from a noisy IMU, feed it to a PID, and check the
    closed loop still holds. This is the coupling that breaks when the
    estimator's sign convention disagrees with the controller's."""
    dt = 0.002
    ekf = AttitudeEKF(accel_gate=0.3)
    pid = PID(kp=4.0, ki=0.5, kd=0.4, output_limits=(-0.5, 0.5), derivative_cutoff_hz=25.0)
    rng = np.random.default_rng(21)

    theta, q = np.deg2rad(12.0), 0.0
    inertia, damping = 0.02, 0.01
    gyro_bias = np.deg2rad(1.0)
    for _ in range(int(6.0 / dt)):
        gyro = np.array([q + gyro_bias + rng.normal(0.0, np.deg2rad(0.5)), 0.0, 0.0])
        accel = np.array([0.0, np.sin(theta), np.cos(theta)]) * 9.81 + rng.normal(0.0, 0.05, 3)
        ekf.step(gyro, accel, dt)
        roll_est, _, _ = ekf.euler
        torque = pid.update(0.0, roll_est, dt)
        alpha = (torque - damping * q) / inertia
        q += alpha * dt
        theta += q * dt
    assert abs(theta) < np.deg2rad(2.5)  # driven back to level using estimated attitude


def test_notch_plus_lowpass_recovers_a_command_buried_in_prop_vibration():
    """The realistic gyro conditioning chain, measured end to end."""
    fs = 2000.0
    t = np.arange(int(4.0 * fs)) / fs
    command = np.sin(2 * np.pi * 3.0 * t)  # what the pilot asked for
    blade_pass = NotchFilter.blade_pass_hz(rpm=12000.0, blades=2)
    assert blade_pass == pytest.approx(400.0)
    rng = np.random.default_rng(6)
    measured = command + 0.8 * np.sin(2 * np.pi * blade_pass * t) + rng.normal(0.0, 0.05, t.size)

    notch = NotchFilter(blade_pass, fs, q=15.0)
    lowpass = BiquadLowPass(80.0, fs)
    cleaned = lowpass.filt(notch.filt(measured))

    err_raw = float(np.std(measured[2000:] - command[2000:]))
    err_clean = float(np.std(cleaned[2000:] - command[2000:]))
    assert err_clean < 0.15 * err_raw
    # And the phase cost of the chain at the command frequency is small.
    assert notch.phase_lag_deg(3.0, fs) + lowpass.phase_lag_deg(3.0, fs) < 10.0


def test_complementary_filter_and_pid_hold_level_under_a_biased_gyro():
    dt = 0.002
    cf = ComplementaryFilter(tau=0.7)
    pid = PID(kp=3.0, ki=0.0, kd=0.3, output_limits=(-0.5, 0.5), derivative_cutoff_hz=30.0)
    theta, q = np.deg2rad(10.0), 0.0
    inertia, damping = 0.02, 0.01
    bias = np.deg2rad(2.0)
    for _ in range(int(10.0 / dt)):
        accel = np.array([0.0, np.sin(theta), np.cos(theta)]) * 9.81
        roll_est, _ = cf.update([q + bias, 0.0, 0.0], accel, dt)
        torque = pid.update(0.0, roll_est, dt)
        alpha = (torque - damping * q) / inertia
        q += alpha * dt
        theta += q * dt
    # The filter's bias * tau offset shows up directly as a levelling error;
    # that is the documented trade, not a failure.
    assert abs(theta) < np.deg2rad(4.0)
    assert abs(theta) > 0.0


def test_min_jerk_reference_reduces_tracking_error_versus_a_step():
    """The reason to generate a trajectory: a feasible reference keeps the loop
    in the small-error regime the linear tune was designed for."""
    dt = 0.001
    distance = 2.0
    duration = min_jerk_duration(distance, max_velocity=1.0, max_acceleration=2.0)
    traj = min_jerk(0.0, distance, duration, n=int(duration / dt) + 1)

    def run(setpoint_fn, sim_time):
        plant = SecondOrderPlant.from_modal(wn=6.0, zeta=0.7, dc_gain=1.0)
        pid = PID(kp=8.0, ki=2.0, kd=1.0, output_limits=(-4.0, 4.0), derivative_cutoff_hz=40.0)
        return simulate(plant, pid.update, duration=sim_time, dt=dt, setpoint=setpoint_fn)

    sim_time = duration + 10.0
    smooth = run(lambda tt: float(traj.at(tt)[0]), sim_time)
    stepped = run(lambda tt: distance, sim_time)

    peak_error_smooth = float(np.max(np.abs(smooth.setpoint - smooth.y)))
    peak_error_step = float(np.max(np.abs(stepped.setpoint - stepped.y)))
    assert peak_error_smooth < 0.5 * peak_error_step
    # Both arrive.
    assert smooth.y[-1] == pytest.approx(distance, abs=0.02)
    assert stepped.y[-1] == pytest.approx(distance, abs=0.02)
    # And the smooth one never asks the actuator for a saturated command.
    assert float(np.max(np.abs(smooth.u))) < 4.0
