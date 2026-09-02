"""Cascade tests: nesting, rate limits, and the tuning-order argument."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.cascade import CascadeController, CascadeLoop, LoopSpec, MultirotorCascade
from dctk.metrics import iae, step_metrics
from dctk.sim import PointMassQuadrotor2D, WindGust


def test_output_limit_bounds_the_inner_setpoint():
    """The whole point of putting the limit between loops: a position loop with
    a huge error may not command a velocity the airframe cannot fly."""
    loop = CascadeLoop(LoopSpec("position", kp=10.0, output_limit=3.0, derivative_cutoff_hz=None))
    assert loop.update(100.0, 0.0, 0.01) == pytest.approx(3.0)
    assert loop.update(-100.0, 0.0, 0.01) == pytest.approx(-3.0)


def test_slew_limit_bounds_the_rate_of_change_not_the_value():
    loop = CascadeLoop(
        LoopSpec("velocity", kp=10.0, output_limit=10.0, slew_limit=2.0, derivative_cutoff_hz=None)
    )
    dt = 0.01
    out = [loop.update(5.0, 0.0, dt) for _ in range(10)]
    assert max(np.diff(out)) <= 2.0 * dt + 1e-12
    assert out[0] == pytest.approx(2.0 * dt)
    assert loop.slew_limited


def test_cascade_chains_outputs_into_setpoints():
    outer = LoopSpec("outer", kp=2.0, output_limit=100.0, derivative_cutoff_hz=None)
    inner = LoopSpec("inner", kp=3.0, output_limit=100.0, derivative_cutoff_hz=None)
    c = CascadeController([outer, inner])
    # Outer: kp*(1-0) = 2. Inner: kp*(2-0.5) = 4.5.
    assert c.update(1.0, [0.0, 0.5], 0.01) == pytest.approx(4.5)
    assert c.setpoints[0] == pytest.approx(2.0)
    assert c.errors[0] == pytest.approx(1.0)
    assert c.errors[1] == pytest.approx(1.5)


def test_cascade_indexing_by_name_and_position():
    c = CascadeController([LoopSpec("a", kp=1.0), LoopSpec("b", kp=1.0)])
    assert c["a"] is c[0]
    assert len(c) == 2
    with pytest.raises(KeyError):
        _ = c["missing"]


def test_bandwidth_separation_warns_when_loops_are_too_close():
    close = CascadeController(
        [LoopSpec("outer", kp=1.0, rate_hz=100.0), LoopSpec("inner", kp=1.0, rate_hz=150.0)]
    )
    assert len(close.check_bandwidth_separation()) == 1
    assert "interact" in close.check_bandwidth_separation()[0]

    ok = CascadeController(
        [LoopSpec("outer", kp=1.0, rate_hz=50.0), LoopSpec("inner", kp=1.0, rate_hz=500.0)]
    )
    assert ok.check_bandwidth_separation() == []


def test_default_multirotor_cascade_respects_the_bandwidth_rule():
    mc = MultirotorCascade()
    assert mc.position.check_bandwidth_separation() == []
    assert mc.altitude.check_bandwidth_separation() == []


def test_cascade_reset_clears_every_loop():
    c = CascadeController([LoopSpec("a", kp=1.0, ki=1.0), LoopSpec("b", kp=1.0, ki=1.0)])
    for _ in range(50):
        c.update(1.0, [0.0, 0.0], 0.01)
    assert any(loop.pid.state.integral != 0.0 for loop in c.loops)
    c.reset()
    assert all(loop.pid.state.integral == 0.0 for loop in c.loops)
    assert c.setpoints == [0.0, 0.0]


def test_altitude_hold_reaches_and_holds_the_setpoint():
    mc = MultirotorCascade(hover_thrust=0.5, max_thrust=1.0)
    plant = PointMassQuadrotor2D(mass=1.2)
    dt = 0.004
    hover_newtons = plant.hover_thrust()
    t, z = [], []
    for i in range(int(12.0 / dt)):
        thrust_norm = mc.update_altitude(2.0, plant.x[1], plant.x[3], dt)
        # normalised 0.5 == hover, so full scale is 2 x hover thrust
        plant.step(thrust_norm * 2.0 * hover_newtons, dt)
        t.append(i * dt)
        z.append(plant.x[1])
    m = step_metrics(np.array(t), np.array(z), 2.0)
    assert abs(m.steady_state_error) < 0.05
    assert m.overshoot_pct < 15.0
    assert np.isfinite(m.rise_time)


def test_altitude_hold_rejects_a_sustained_downdraft():
    """Integral action in the climb-rate loop is what holds altitude against a
    steady disturbance; without it the aircraft settles low."""
    dt = 0.004
    gust = WindGust(kind="step", amplitude=-3.0, t_start=6.0)  # newtons, downward

    def run(ki):
        mc = MultirotorCascade(hover_thrust=0.5, max_thrust=1.0)
        mc.altitude[1].pid.set_gains(ki=ki, bumpless=False)
        plant = PointMassQuadrotor2D(mass=1.2)
        hover = plant.hover_thrust()
        for i in range(int(20.0 / dt)):
            u = mc.update_altitude(2.0, plant.x[1], plant.x[3], dt)
            # (fx, fz): a pure downdraft, which is what threatens altitude hold.
            plant.step(u * 2.0 * hover, dt, (0.0, gust.value(i * dt, dt)))
        return plant.x[1]

    with_i = run(0.12)
    without_i = run(0.0)
    assert abs(with_i - 2.0) < 0.05
    assert abs(without_i - 2.0) > 0.15


def test_inner_loop_bandwidth_matters_the_tuning_order_argument():
    """Detune the inner (rate) loop and the outer (attitude) loop that was fine
    before now performs worse -- which is why you never tune outward first."""
    dt = 0.001

    def run(rate_kp):
        mc = MultirotorCascade()
        mc.position[3].pid.set_gains(kp=rate_kp, bumpless=False)
        theta, q = 0.0, 0.0
        inertia, damping = 0.01, 0.002
        errs = []
        for i in range(int(4.0 / dt)):
            torque = mc.position[2].update(np.deg2rad(10.0), theta, dt)
            torque = mc.position[3].update(torque, q, dt)
            alpha = (np.clip(torque, -0.5, 0.5) - damping * q) / inertia
            q += alpha * dt
            theta += q * dt
            errs.append(abs(np.deg2rad(10.0) - theta))
        t_axis = np.arange(len(errs)) * dt
        return iae(t_axis, -np.asarray(errs), 0.0)

    good_inner = run(0.06)
    weak_inner = run(0.004)
    assert weak_inner > good_inner


def test_horizontal_chain_moves_the_vehicle_toward_the_setpoint():
    """Sign check on the full position -> velocity -> attitude -> rate chain.
    Getting this sign wrong makes the aircraft accelerate away from the
    waypoint, which looks like an unstable loop but is a wiring bug."""
    mc = MultirotorCascade()
    dt = 0.002
    x, vx, theta, q = 0.0, 0.0, 0.0, 0.0
    g, inertia, damping, mass = 9.81, 0.01, 0.002, 1.2
    for _ in range(int(6.0 / dt)):
        torque = mc.update_horizontal(3.0, x, vx, theta, q, dt)
        alpha = (np.clip(torque, -0.5, 0.5) - damping * q) / inertia
        q += alpha * dt
        theta += q * dt
        ax = -g * np.tan(np.clip(theta, -0.6, 0.6)) - 0.15 * vx * abs(vx) / mass
        vx += ax * dt
        x += vx * dt
    assert x > 2.0  # moved toward the target, not away from it
    assert abs(x - 3.0) < 1.0


def test_diagnostics_expose_every_loop():
    mc = MultirotorCascade()
    mc.update_altitude(1.0, 0.0, 0.0, 0.01)
    mc.update_horizontal(1.0, 0.0, 0.0, 0.0, 0.0, 0.01)
    d = mc.diagnostics()
    for name in ("position", "velocity", "attitude", "rate", "altitude", "climb_rate"):
        assert f"{name}_error" in d
        assert f"{name}_sat" in d


def test_cascade_rejects_a_wrong_length_measurement_vector():
    c = CascadeController([LoopSpec("a", kp=1.0), LoopSpec("b", kp=1.0)])
    with pytest.raises(ValueError):
        c.update(1.0, [0.0], 0.01)
    with pytest.raises(ValueError):
        CascadeController([])
