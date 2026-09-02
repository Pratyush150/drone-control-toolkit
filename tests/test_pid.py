"""PID behaviour tests.

These assert the *behaviour* each feature was added for, not that the functions
return without raising. Each test names the failure mode it is guarding.
"""

from __future__ import annotations

import numpy as np
import pytest

from dctk.pid import PID, AntiWindup


def test_proportional_only_is_exact():
    pid = PID(kp=2.0, derivative_cutoff_hz=None)
    assert pid.update(1.0, 0.0, 0.01) == pytest.approx(2.0)
    assert pid.update(1.0, 0.5, 0.01) == pytest.approx(1.0)


def test_integral_accumulates_error_over_time():
    pid = PID(kp=0.0, ki=1.0, derivative_cutoff_hz=None)
    for _ in range(100):
        out = pid.update(1.0, 0.0, 0.01)
    # 100 steps of 0.01 s at unit error -> integral of 1.0.
    assert out == pytest.approx(1.0, rel=1e-9)


def test_output_saturation_is_respected():
    pid = PID(kp=100.0, output_limits=(-1.0, 2.0), derivative_cutoff_hz=None)
    assert pid.update(10.0, 0.0, 0.01) == pytest.approx(2.0)
    assert pid.update(-10.0, 0.0, 0.01) == pytest.approx(-1.0)


def test_no_antiwindup_lets_the_integrator_run_away():
    """Baseline for the two tests below: without protection the integrator
    grows without bound while the output is pinned at the limit."""
    pid = PID(kp=1.0, ki=5.0, output_limits=(-1.0, 1.0),
              anti_windup=AntiWindup.NONE, derivative_cutoff_hz=None)
    for _ in range(500):
        pid.update(10.0, 0.0, 0.01)
    assert pid.state.integral > 40.0


def test_clamp_antiwindup_bounds_the_integrator():
    pid = PID(kp=1.0, ki=5.0, output_limits=(-1.0, 1.0),
              anti_windup=AntiWindup.CLAMP, derivative_cutoff_hz=None)
    integrals = []
    for _ in range(500):
        pid.update(10.0, 0.0, 0.01)
        integrals.append(pid.state.integral)
    # Conditional integration freezes the integral after the first saturated
    # sample, so it never exceeds one step's worth of error.
    assert max(integrals) <= 10.0 * 0.01 + 1e-12
    assert pid.state.integral == pytest.approx(integrals[0])


def test_back_calculation_antiwindup_bounds_the_integrator():
    pid = PID(kp=1.0, ki=5.0, output_limits=(-1.0, 1.0),
              anti_windup=AntiWindup.BACK_CALCULATION, tracking_time_constant=0.1,
              derivative_cutoff_hz=None)
    for _ in range(2000):
        pid.update(10.0, 0.0, 0.01)
    # Steady state: the integral settles where error*dt exactly cancels the
    # back-calculation correction, which is far below the runaway value.
    assert abs(pid.state.integral) < 5.0
    assert pid.state.saturated


def test_antiwindup_recovers_faster_than_no_antiwindup():
    """The reason anti-windup exists: a wound-up integrator overshoots badly
    when the setpoint returns to something achievable."""

    def run(strategy):
        pid = PID(kp=1.0, ki=4.0, output_limits=(-1.0, 1.0),
                  anti_windup=strategy, tracking_time_constant=0.2,
                  derivative_cutoff_hz=None)
        y = 0.0
        peak = 0.0
        for i in range(1500):
            sp = 5.0 if i < 500 else 0.5
            u = pid.update(sp, y, 0.01)
            y += 0.01 * (u - 0.2 * y)  # slow integrating plant
            if i >= 500:
                peak = max(peak, y)
        return peak

    peak_none = run(AntiWindup.NONE)
    peak_clamp = run(AntiWindup.CLAMP)
    peak_back = run(AntiWindup.BACK_CALCULATION)
    assert peak_clamp < peak_none
    assert peak_back < peak_none


def test_derivative_on_measurement_gives_no_kick_on_setpoint_step():
    """Derivative kick: with derivative-on-error, a setpoint step produces a
    one-sample impulse of size kd * delta_sp / dt."""
    dt = 0.01
    kd = 0.5
    on_meas = PID(kp=0.0, kd=kd, derivative_on_measurement=True, derivative_cutoff_hz=None)
    on_err = PID(kp=0.0, kd=kd, derivative_on_measurement=False, derivative_cutoff_hz=None)

    # Settle both at setpoint 0 with a constant measurement.
    for _ in range(5):
        on_meas.update(0.0, 0.0, dt)
        on_err.update(0.0, 0.0, dt)

    # Step the setpoint; the measurement has not moved yet (it physically cannot).
    out_meas = on_meas.update(1.0, 0.0, dt)
    out_err = on_err.update(1.0, 0.0, dt)

    assert out_meas == pytest.approx(0.0, abs=1e-12)
    assert out_err == pytest.approx(kd * 1.0 / dt)
    assert abs(out_err) > 10.0 * max(abs(out_meas), 1e-9)


def test_derivative_responds_to_measurement_motion_either_way():
    """Derivative-on-measurement must not be a no-op: it still reacts to the
    plant moving, which is the whole point of the D term."""
    dt = 0.01
    pid = PID(kp=0.0, kd=1.0, derivative_on_measurement=True, derivative_cutoff_hz=None)
    pid.update(0.0, 0.0, dt)
    out = pid.update(0.0, 0.1, dt)
    assert out == pytest.approx(-0.1 / dt)


def test_derivative_filter_attenuates_high_frequency_noise():
    dt = 0.001
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, 4000)

    def run(cutoff):
        pid = PID(kp=0.0, kd=1.0, derivative_cutoff_hz=cutoff)
        outs = [pid.update(0.0, float(n), dt) for n in noise]
        return float(np.std(outs[1000:]))

    raw = run(None)
    filtered = run(20.0)
    assert filtered < 0.2 * raw


def test_derivative_filter_preserves_a_slow_ramp():
    """The filter must remove noise without destroying the signal: on a ramp
    slow compared with the cutoff, the filtered D term converges to the true
    slope."""
    dt = 0.001
    pid = PID(kp=0.0, kd=1.0, derivative_cutoff_hz=20.0)
    for i in range(2000):
        out = pid.update(0.0, 0.5 * i * dt, dt)  # measurement ramping at 0.5 units/s
    assert out == pytest.approx(-0.5, rel=0.02)


def test_feedforward_acts_immediately_with_no_error():
    pid = PID(kp=1.0, kff=0.5, derivative_cutoff_hz=None)
    # Measurement already equals setpoint, so P, I and D are all zero.
    out = pid.update(2.0, 2.0, 0.01)
    assert out == pytest.approx(1.0)


def test_explicit_feedforward_signal_overrides_setpoint():
    pid = PID(kff=2.0, derivative_cutoff_hz=None)
    assert pid.update(1.0, 1.0, 0.01, feedforward=3.0) == pytest.approx(6.0)


def test_bumpless_gain_change_preserves_output():
    pid = PID(kp=1.0, ki=2.0, derivative_cutoff_hz=None)
    for _ in range(50):
        before = pid.update(1.0, 0.0, 0.01)
    pid.set_gains(ki=8.0, bumpless=True)
    after = pid.update(1.0, 0.0, 0.01)
    # One extra sample of integration is expected; the step must not be the
    # 4x jump that a naive gain change would produce.
    assert after == pytest.approx(before, rel=0.05)


def test_non_bumpless_gain_change_does_step():
    pid = PID(kp=1.0, ki=2.0, derivative_cutoff_hz=None)
    for _ in range(50):
        before = pid.update(1.0, 0.0, 0.01)
    pid.set_gains(ki=8.0, bumpless=False)
    after = pid.update(1.0, 0.0, 0.01)
    assert after > 2.0 * before


def test_manual_to_auto_handover_is_bumpless():
    pid = PID(kp=1.0, ki=2.0, derivative_cutoff_hz=None)
    pid.set_manual(0.75)
    for _ in range(20):
        manual_out = pid.update(1.0, 0.4, 0.01)
    assert manual_out == pytest.approx(0.75)
    pid.set_auto()
    first_auto = pid.update(1.0, 0.4, 0.01)
    assert first_auto == pytest.approx(0.75, abs=0.02)


def test_variable_dt_integration_is_consistent():
    """Two short steps must integrate the same as one long step."""
    a = PID(ki=1.0, derivative_cutoff_hz=None)
    a.update(1.0, 0.0, 0.02)
    b = PID(ki=1.0, derivative_cutoff_hz=None)
    b.update(1.0, 0.0, 0.01)
    b.update(1.0, 0.0, 0.01)
    assert a.state.integral == pytest.approx(b.state.integral)


def test_bad_dt_is_rejected():
    pid = PID(kp=1.0)
    with pytest.raises(ValueError):
        pid.update(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        pid.update(1.0, 0.0, -0.01)


def test_reset_clears_state_and_first_derivative_is_zero():
    pid = PID(kp=1.0, ki=1.0, kd=1.0, derivative_cutoff_hz=None)
    for _ in range(20):
        pid.update(1.0, 0.3, 0.01)
    assert pid.state.integral != 0.0
    pid.reset()
    assert pid.state.integral == 0.0
    assert np.isnan(pid.state.last_measurement)
    # First sample after reset must not produce a derivative spike even though
    # the measurement is far from zero.
    out = pid.update(0.0, 5.0, 0.01)
    # P (-5.0) plus one step of I (-0.05); D contributes exactly nothing.
    assert out == pytest.approx(-5.0 - 5.0 * 0.01)


def test_setpoint_weight_softens_the_step_without_changing_disturbance_response():
    dt = 0.01
    full = PID(kp=2.0, setpoint_weight_p=1.0, derivative_cutoff_hz=None)
    soft = PID(kp=2.0, setpoint_weight_p=0.5, derivative_cutoff_hz=None)
    assert soft.update(1.0, 0.0, dt) < full.update(1.0, 0.0, dt)
    # With setpoint at zero (a pure disturbance), both are identical.
    assert soft.update(0.0, 0.3, dt) == pytest.approx(full.update(0.0, 0.3, dt))


def test_invalid_construction_is_rejected():
    with pytest.raises(ValueError):
        PID(output_limits=(1.0, -1.0))
    with pytest.raises(ValueError):
        PID(tracking_time_constant=0.0)
    with pytest.raises(ValueError):
        PID(derivative_cutoff_hz=-5.0)
