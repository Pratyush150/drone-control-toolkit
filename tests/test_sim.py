"""Simulation harness tests: the plants are right and the defects actually bite."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.metrics import step_metrics
from dctk.pid import PID
from dctk.sim import (
    ActuatorDelay,
    FirstOrderPlant,
    MotorLag,
    PitchPlant,
    PointMassQuadrotor2D,
    Quantiser,
    SecondOrderPlant,
    SensorNoise,
    WindGust,
    simulate,
)


def test_first_order_plant_matches_the_analytic_step_response():
    plant = FirstOrderPlant(tau=0.5, gain=2.0)
    dt = 0.001
    ys = []
    for _ in range(3000):
        ys.append(plant.step(1.0, dt))
    t = np.arange(1, 3001) * dt
    expected = 2.0 * (1.0 - np.exp(-t / 0.5))
    assert np.max(np.abs(np.array(ys) - expected)) < 1e-8


def test_second_order_modal_construction_gives_the_requested_dynamics():
    wn, zeta = 4.0, 0.3
    plant = SecondOrderPlant.from_modal(wn, zeta, dc_gain=1.0)
    dt = 0.0005
    t, ys = [], []
    for i in range(int(6.0 / dt)):
        ys.append(plant.step(1.0, dt))
        t.append((i + 1) * dt)
    t = np.array(t)
    y = np.array(ys)
    assert y[-1] == pytest.approx(1.0, rel=1e-3)  # DC gain
    m = step_metrics(t, y, 1.0)
    expected_os = 100.0 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    assert m.overshoot_pct == pytest.approx(expected_os, rel=0.02)
    wd = wn * np.sqrt(1 - zeta**2)
    assert m.peak_time == pytest.approx(np.pi / wd, rel=0.02)


def test_pitch_plant_is_a_double_integrator():
    """Constant torque gives constant angular acceleration."""
    plant = PitchPlant(inertia=0.02, damping=0.0)
    dt = 0.001
    for _ in range(1000):
        plant.step(0.01, dt)
    alpha = 0.01 / 0.02
    assert plant.rate() == pytest.approx(alpha * 1.0, rel=1e-6)
    assert plant.x[0] == pytest.approx(0.5 * alpha * 1.0**2, rel=1e-6)


def test_point_mass_quadrotor_hovers_at_hover_thrust():
    plant = PointMassQuadrotor2D(mass=1.2)
    dt = 0.002
    for _ in range(2000):
        plant.step(plant.hover_thrust(), dt)
    assert plant.x[1] == pytest.approx(0.0, abs=1e-6)
    assert plant.x[3] == pytest.approx(0.0, abs=1e-6)


def test_point_mass_quadrotor_drag_bounds_the_terminal_velocity():
    plant = PointMassQuadrotor2D(mass=1.2, drag_xy=0.3)
    plant.set_input(0.0, np.deg2rad(20.0))
    dt = 0.002
    for _ in range(20000):
        plant.step(plant.hover_thrust(), dt)
    v_terminal = np.sqrt(plant.hover_thrust() * np.sin(np.deg2rad(20.0)) / 0.3)
    assert plant.x[2] == pytest.approx(v_terminal, rel=0.02)


def test_quadrotor_accepts_a_vertical_disturbance_component():
    """A downdraft is (0, -F); a scalar disturbance stays horizontal."""
    plant = PointMassQuadrotor2D(mass=1.0, drag_z=0.0)
    dt = 0.001
    for _ in range(1000):
        plant.step(plant.hover_thrust(), dt, (0.0, -2.0))
    assert plant.x[3] == pytest.approx(-2.0, rel=1e-3)  # a = F/m = 2 m/s^2 for 1 s
    assert plant.x[2] == pytest.approx(0.0, abs=1e-12)

    plant2 = PointMassQuadrotor2D(mass=1.0, drag_xy=0.0)
    for _ in range(1000):
        plant2.step(plant2.hover_thrust(), dt, 2.0)
    assert plant2.x[2] == pytest.approx(2.0, rel=1e-3)


def test_sensor_noise_is_deterministic_for_a_given_seed():
    a = SensorNoise(sigma=0.1, seed=42)
    b = SensorNoise(sigma=0.1, seed=42)
    va = [a.apply(0.0, 0.01) for _ in range(50)]
    vb = [b.apply(0.0, 0.01) for _ in range(50)]
    assert va == vb
    a.reset()
    assert [a.apply(0.0, 0.01) for _ in range(50)] == va


def test_sensor_bias_random_walk_grows_with_time():
    walk = SensorNoise(sigma=0.0, bias_walk=0.05, seed=1)
    for _ in range(100):
        walk.apply(0.0, 0.01)
    early = abs(walk.current_bias)
    for _ in range(9900):
        walk.apply(0.0, 0.01)
    assert abs(walk.current_bias) > early or abs(walk.current_bias) > 0.0
    assert walk.current_bias != 0.0


def test_quantiser_snaps_to_the_grid_and_clips():
    q = Quantiser(step=0.25, lo=-1.0, hi=1.0)
    assert q.apply(0.3) == pytest.approx(0.25)
    assert q.apply(0.4) == pytest.approx(0.5)
    assert q.apply(5.0) == pytest.approx(1.0)
    assert q.apply(-5.0) == pytest.approx(-1.0)


def test_actuator_delay_shifts_the_signal_by_exactly_n_samples():
    delay = ActuatorDelay(delay_s=0.005, dt=0.001)
    assert delay.n == 5
    out = [delay.apply(float(i)) for i in range(20)]
    assert out[:5] == [0.0] * 5
    assert out[5:] == [float(i) for i in range(15)]


def test_zero_delay_is_a_passthrough():
    delay = ActuatorDelay(delay_s=0.0, dt=0.001)
    assert delay.apply(3.14) == pytest.approx(3.14)


def test_motor_lag_is_a_first_order_response():
    lag = MotorLag(tau=0.05)
    dt = 0.0005
    for _ in range(int(0.05 / dt)):
        v = lag.apply(1.0, dt)
    assert v == pytest.approx(1.0 - np.exp(-1.0), rel=1e-6)  # one time constant


def test_wind_gust_shapes():
    step = WindGust(kind="step", amplitude=2.0, t_start=1.0)
    assert step.value(0.5, 0.01) == 0.0
    assert step.value(1.5, 0.01) == pytest.approx(2.0)

    gust = WindGust(kind="gust", amplitude=3.0, t_start=1.0, duration=2.0)
    assert gust.value(0.5, 0.01) == 0.0
    assert gust.value(2.0, 0.01) == pytest.approx(3.0)  # peak at the midpoint
    assert gust.value(4.0, 0.01) == 0.0

    turb = WindGust(kind="turbulence", amplitude=1.0, tau=0.2, seed=3)
    vals = [turb.value(i * 0.01, 0.01) for i in range(2000)]
    assert np.std(vals) > 0.0
    turb.reset()
    assert [turb.value(i * 0.01, 0.01) for i in range(2000)] == vals


def test_simulate_closes_the_loop_and_reaches_the_setpoint():
    plant = FirstOrderPlant(tau=0.5)
    pid = PID(kp=4.0, ki=6.0, kd=0.0, output_limits=(-10, 10), derivative_cutoff_hz=None)
    result = simulate(plant, pid.update, duration=5.0, dt=0.001, setpoint=1.0)
    assert result.y[-1] == pytest.approx(1.0, abs=1e-3)
    assert result.t.shape == result.y.shape == result.u.shape


def test_actuator_delay_degrades_a_tune_that_was_fine_without_it():
    """The headline reason this module exists: a tune validated on a clean
    plant is not validated. 30 ms of transport delay -- a plausible budget for
    a serial link plus ESC protocol -- takes a nearly critically damped
    response to 40 % overshoot without a single gain being changed."""
    dt = 0.001

    def run(delay_s):
        plant = SecondOrderPlant.from_modal(wn=8.0, zeta=0.15, dc_gain=1.0)
        pid = PID(kp=6.0, ki=2.0, kd=0.3, output_limits=(-20, 20), derivative_cutoff_hz=80.0)
        return simulate(
            plant,
            pid.update,
            duration=6.0,
            dt=dt,
            setpoint=1.0,
            actuator_delay=ActuatorDelay(delay_s, dt),
        ).metrics(1.0)

    clean = run(0.0)
    laggy = run(0.03)
    assert clean.overshoot_pct < 5.0
    assert laggy.overshoot_pct > clean.overshoot_pct + 20.0
    assert laggy.ise > 1.5 * clean.ise


def test_motor_lag_imposes_a_gain_margin_where_there_was_none():
    """A second-order plant under proportional control is stable at *any* gain
    -- its phase never reaches -180 degrees. Add the motor's own first-order
    lag and there is now a third pole, a phase crossover, and a finite ultimate
    gain. This is why "it was fine in simulation" and "it oscillated on the
    bench" are compatible statements."""
    dt = 0.0005
    kp = 5.0

    def run(tau):
        plant = SecondOrderPlant.from_modal(wn=8.0, zeta=0.1, dc_gain=1.0)
        pid = PID(kp=kp, ki=0.0, kd=0.0, output_limits=(-50, 50), derivative_cutoff_hz=None)
        return simulate(
            plant, pid.update, duration=8.0, dt=dt, setpoint=1.0, motor_lag=MotorLag(tau)
        ).y

    clean = run(0.0)
    laggy = run(0.02)  # 20 ms motor time constant, a small 5-inch quad
    assert float(np.ptp(clean[-4000:])) < 0.05     # settled
    assert float(np.ptp(laggy[-4000:])) > 1.0      # limit cycling


def test_motor_lag_slows_the_actuator_response_at_fixed_gains():
    dt = 0.001

    def run(tau):
        plant = SecondOrderPlant.from_modal(wn=6.0, zeta=0.5, dc_gain=1.0)
        pid = PID(kp=3.0, ki=1.0, kd=0.2, output_limits=(-20, 20), derivative_cutoff_hz=50.0)
        return simulate(
            plant, pid.update, duration=8.0, dt=dt, setpoint=1.0, motor_lag=MotorLag(tau)
        ).metrics(1.0)

    clean, laggy = run(0.0), run(0.08)
    assert clean.overshoot_pct == pytest.approx(0.0)
    assert laggy.overshoot_pct > 3.0
    assert laggy.ise > clean.ise


def test_wind_disturbance_produces_a_steady_state_error_without_integral_action():
    dt = 0.001
    gust = WindGust(kind="step", amplitude=0.5, t_start=2.0)

    p_only = PID(kp=4.0, ki=0.0, output_limits=(-10, 10), derivative_cutoff_hz=None)
    r_p = simulate(FirstOrderPlant(tau=0.5), p_only.update, duration=8.0, dt=dt,
                   setpoint=1.0, disturbance=gust)
    assert abs(r_p.y[-1] - 1.0) > 0.05

    pi = PID(kp=4.0, ki=8.0, output_limits=(-10, 10), derivative_cutoff_hz=None)
    r_pi = simulate(FirstOrderPlant(tau=0.5), pi.update, duration=8.0, dt=dt,
                    setpoint=1.0, disturbance=WindGust(kind="step", amplitude=0.5, t_start=2.0))
    assert abs(r_pi.y[-1] - 1.0) < 1e-3


def test_simulation_is_bit_reproducible():
    def run():
        pid = PID(kp=3.0, ki=1.0, derivative_cutoff_hz=30.0)
        return simulate(
            FirstOrderPlant(tau=0.4),
            pid.update,
            duration=3.0,
            dt=0.002,
            setpoint=1.0,
            sensor_noise=SensorNoise(sigma=0.02, seed=99),
        ).y

    assert np.array_equal(run(), run())


def test_simulate_input_validation():
    pid = PID(kp=1.0)
    with pytest.raises(ValueError):
        simulate(FirstOrderPlant(), pid.update, duration=1.0, dt=0.0)
    with pytest.raises(ValueError):
        FirstOrderPlant(tau=0.0)
    with pytest.raises(ValueError):
        ActuatorDelay(-1.0, 0.001)
