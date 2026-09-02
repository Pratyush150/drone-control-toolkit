"""Auto-tuning tests, including the honest one: ZN output is too hot to fly."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.metrics import step_metrics
from dctk.pid import PID
from dctk.sim import ActuatorDelay, SecondOrderPlant, simulate
from dctk.tuning import find_ultimate_gain, relay_autotune, ziegler_nichols


# 4 ms is ~180x oversampled for a plant whose ultimate frequency is ~1.4 Hz,
# and keeps the gain sweep -- which runs a full closed-loop simulation per
# bisection step -- to a few seconds.
DT = 0.004


def make_plant():
    """Second-order plant plus 50 ms of transport delay: a phase crossover
    exists, so an ultimate gain exists."""
    return SecondOrderPlant.from_modal(wn=6.0, zeta=0.2, dc_gain=1.0), ActuatorDelay(0.05, DT)


def closed_loop(kp):
    """Proportional-only closed loop for the gain sweep.

    Deliberately unsaturated: a saturated loop limit-cycles at bounded
    amplitude above the stability boundary, and the envelope test would then
    report every gain as stable.
    """
    plant, delay = make_plant()
    pid = PID(kp, 0.0, 0.0, derivative_cutoff_hz=None)
    return simulate(
        plant, pid.update, duration=14.0, dt=DT, setpoint=1.0, actuator_delay=delay
    ).y


@pytest.fixture(scope="module")
def ultimate():
    """Gain sweep result, computed once and shared.

    The sweep runs a full closed-loop simulation per bisection step, so it is
    the most expensive thing in the suite. Module scope keeps it to one run.
    """
    return find_ultimate_gain(closed_loop, dt=DT, kp_max=100.0, iterations=20)


def test_relay_autotune_finds_the_ultimate_gain_and_period():
    """Cross-check against the analytic answer: for this plant the -180 degree
    crossover is near 8.8 rad/s where |G| is about 0.77, so Ku is about 1.3."""
    plant, delay = make_plant()

    def step(u, dt):
        return plant.step(delay.apply(u), dt)

    result = relay_autotune(step, relay_amplitude=1.0, dt=DT, duration=30.0)
    assert result.converged
    assert result.n_cycles >= 5
    assert result.ku == pytest.approx(1.30, rel=0.15)
    assert result.tu == pytest.approx(2 * np.pi / 8.8, rel=0.15)
    assert result.ultimate_frequency_hz == pytest.approx(1.0 / result.tu)


def test_relay_amplitude_bounds_the_excitation():
    """The safety argument for relay tuning: the oscillation amplitude scales
    with the relay amplitude you chose, so you control how hard the plant is
    shaken."""
    amps = []
    for d in (0.5, 1.0):
        plant, delay = make_plant()

        def step(u, dt, _p=plant, _d=delay):
            return _p.step(_d.apply(u), dt)

        amps.append(relay_autotune(step, relay_amplitude=d, dt=DT, duration=30.0).amplitude)
    assert amps[1] == pytest.approx(2.0 * amps[0], rel=0.1)


def test_relay_with_hysteresis_still_converges():
    plant, delay = make_plant()

    def step(u, dt):
        return plant.step(delay.apply(u), dt)

    result = relay_autotune(step, relay_amplitude=1.0, hysteresis=0.05, dt=DT, duration=30.0)
    assert result.converged
    assert result.ku == pytest.approx(1.30, rel=0.25)


def test_relay_reports_failure_when_it_cannot_see_a_limit_cycle():
    calls = {"n": 0}

    def dead_plant(u, dt):
        calls["n"] += 1
        return 0.0  # no response at all

    result = relay_autotune(dead_plant, relay_amplitude=1.0, dt=DT, duration=1.0)
    assert not result.converged
    assert calls["n"] > 0


def test_gain_sweep_agrees_with_the_relay_method(ultimate):
    assert ultimate.converged
    assert ultimate.ku == pytest.approx(1.30, rel=0.2)


def test_ziegler_nichols_table_conversions():
    ku, tu = 2.0, 0.5
    g = ziegler_nichols(ku, tu, "classic")
    assert g.kp == pytest.approx(0.6 * ku)
    assert g.ki == pytest.approx(g.kp / (0.5 * tu))
    assert g.kd == pytest.approx(g.kp * 0.125 * tu)

    pi = ziegler_nichols(ku, tu, "PI")
    assert pi.kd == 0.0
    p_only = ziegler_nichols(ku, tu, "P")
    assert p_only.ki == 0.0 and p_only.kd == 0.0


def test_less_aggressive_rules_have_lower_gains():
    ku, tu = 2.0, 0.5
    classic = ziegler_nichols(ku, tu, "classic")
    gentle = ziegler_nichols(ku, tu, "no_overshoot")
    assert gentle.kp < classic.kp
    assert gentle.ki < classic.ki


def test_derating_scales_i_by_the_square_of_the_factor():
    g = ziegler_nichols(2.0, 0.5, "classic")
    half = g.derated(0.5)
    assert half.kp == pytest.approx(0.5 * g.kp)
    assert half.kd == pytest.approx(0.5 * g.kd)
    assert half.ki == pytest.approx(0.25 * g.ki)
    assert "x0.5" in half.rule


def test_ziegler_nichols_tune_is_fast_but_fragile_to_unmodelled_lag(ultimate):
    """The honest claim from the module docstring, measured.

    Classic ZN targets roughly 30 degrees of phase margin. That is enough on
    the plant the ultimate-gain test saw and not enough on the real one, which
    always has extra lag the test did not include: a slower ESC, a filter added
    later, a busier scheduler. Add 40 ms of delay that was not present during
    tuning and the ZN response degrades sharply; the derated one does not
    notice.
    """
    zn = ziegler_nichols(ultimate.ku, ultimate.tu, "classic")
    derated = zn.derated(0.5)

    def run(gains, extra_delay_s):
        plant, _ = make_plant()
        delay = ActuatorDelay(0.05 + extra_delay_s, DT)
        pid = PID(
            gains.kp, gains.ki, gains.kd,
            output_limits=(-50.0, 50.0), derivative_cutoff_hz=50.0,
        )
        result = simulate(
            plant, pid.update, duration=40.0, dt=DT, setpoint=1.0, actuator_delay=delay
        )
        return step_metrics(result.t, result.y, 1.0)

    zn_nominal = run(zn, 0.0)
    zn_extra = run(zn, 0.04)
    de_nominal = run(derated, 0.0)
    de_extra = run(derated, 0.04)

    # ZN is faster on the plant it was tuned against.
    assert zn_nominal.rise_time < de_nominal.rise_time

    # ...and pays for it when reality adds lag.
    assert zn_extra.overshoot_pct > 2.0 * max(zn_nominal.overshoot_pct, 1.0)
    assert zn_extra.settling_time > 2.0 * zn_nominal.settling_time

    # The derated tune is barely affected: that is what phase margin buys.
    assert de_extra.overshoot_pct < 1.0
    assert abs(de_extra.settling_time - de_nominal.settling_time) < 0.1 * de_nominal.settling_time


def test_derating_costs_bandwidth_which_is_the_trade_being_made(ultimate):
    zn = ziegler_nichols(ultimate.ku, ultimate.tu, "classic")

    def run(gains):
        plant, delay = make_plant()
        pid = PID(gains.kp, gains.ki, gains.kd,
                  output_limits=(-50.0, 50.0), derivative_cutoff_hz=50.0)
        r = simulate(plant, pid.update, duration=40.0, dt=DT, setpoint=1.0,
                     actuator_delay=delay)
        return step_metrics(r.t, r.y, 1.0)

    fast = run(zn)
    slow = run(zn.derated(0.5))
    assert slow.rise_time > 5.0 * fast.rise_time
    assert slow.overshoot_pct < fast.overshoot_pct


def test_tuning_input_validation():
    with pytest.raises(ValueError):
        ziegler_nichols(0.0, 1.0)
    with pytest.raises(ValueError):
        ziegler_nichols(1.0, 1.0, rule="nonsense")
    with pytest.raises(ValueError):
        ziegler_nichols(1.0, 1.0).derated(0.0)
    with pytest.raises(ValueError):
        relay_autotune(lambda u, dt: 0.0, relay_amplitude=0.0)
    with pytest.raises(ValueError):
        find_ultimate_gain(closed_loop, dt=DT, kp_min=1.0, kp_max=0.5)
