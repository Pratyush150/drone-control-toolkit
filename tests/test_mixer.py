"""Mixer tests: limits are never violated and attitude survives saturation."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.mixer import Mixer, hexa_x, quad_plus, quad_x, vtol_quadplane


GEOMETRIES = [quad_x(), quad_plus(), hexa_x()]


@pytest.mark.parametrize("geom", GEOMETRIES, ids=lambda g: g.name)
def test_hover_command_is_uniform_and_matches_the_thrust_demand(geom):
    m = Mixer(geom, idle=0.05)
    u = m.mix(0.0, 0.0, 0.0, 0.5)
    assert np.allclose(u, 0.5, atol=1e-9)
    assert m.achieved(u)[3] == pytest.approx(0.5)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=lambda g: g.name)
def test_unsaturated_demand_is_reproduced_exactly(geom):
    m = Mixer(geom, idle=0.05)
    demand = (0.15, -0.1, 0.05, 0.5)
    u = m.mix(*demand)
    achieved = m.achieved(u)
    assert np.allclose(achieved, demand, atol=1e-9)
    assert m.last_saturation == pytest.approx(0.0)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=lambda g: g.name)
def test_motor_commands_never_leave_the_limits(geom):
    """Fuzz across the whole demand space. Not one command may fall outside
    [idle, 1] -- an out-of-range command is either a stopped motor or a
    saturated ESC, and both are silent in a log."""
    m = Mixer(geom, idle=0.07, max_command=1.0)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        demand = rng.uniform(-1.5, 1.5, 4)
        demand[3] = rng.uniform(-0.5, 2.0)
        u = m.mix(*demand)
        assert u.shape == (geom.n_motors,)
        assert np.all(u >= 0.07 - 1e-12), f"below idle for demand {demand}"
        assert np.all(u <= 1.0 + 1e-12), f"above max for demand {demand}"


def test_thrust_is_sacrificed_to_preserve_attitude():
    """The headline design decision. Demand more thrust than can coexist with
    the attitude demand: the attitude must come out intact and the thrust must
    be the thing that shrinks."""
    m = Mixer(quad_x(), idle=0.05)
    roll, pitch, yaw, thrust = 0.6, 0.4, 0.0, 0.95
    u = m.mix(roll, pitch, yaw, thrust)
    achieved = m.achieved(u)
    assert achieved[0] == pytest.approx(roll, abs=1e-9)
    assert achieved[1] == pytest.approx(pitch, abs=1e-9)
    assert achieved[3] < thrust - 0.1  # thrust gave way
    assert m.last_saturation == pytest.approx(0.0)  # attitude authority intact


def test_attitude_direction_is_preserved_under_saturation():
    """Naive per-motor clipping rotates the torque vector, turning a roll
    command into roll-plus-pitch. Desaturation here may only scale it."""
    m = Mixer(quad_x(), idle=0.05)
    rng = np.random.default_rng(1)
    for _ in range(1000):
        roll, pitch = rng.uniform(-1.2, 1.2, 2)
        thrust = rng.uniform(0.0, 1.2)
        u = m.mix(roll, pitch, 0.0, thrust)
        err = m.attitude_direction_error(roll, pitch, m.achieved(u))
        assert err < np.deg2rad(1.0), f"torque direction rotated by {np.degrees(err):.2f} deg"


def test_excessive_attitude_demand_is_scaled_not_clipped():
    m = Mixer(quad_x(), idle=0.05)
    u = m.mix(3.0, 0.0, 0.0, 0.5)
    achieved = m.achieved(u)
    assert m.last_saturation > 0.0
    assert 0.0 < achieved[0] < 3.0
    assert abs(achieved[1]) < 1e-9  # no pitch leaked in


def test_yaw_yields_before_roll_by_default():
    """Yaw is the weakest and least urgent axis; with yaw_priority=0 it gives
    way completely before roll/pitch loses anything."""
    m = Mixer(quad_x(), idle=0.05, yaw_priority=0.0)
    roll, yaw = 0.8, 0.9
    u = m.mix(roll, 0.0, yaw, 0.5)
    achieved = m.achieved(u)
    assert achieved[0] == pytest.approx(roll, abs=1e-9)  # roll untouched
    assert achieved[2] < yaw  # yaw reduced
    assert m.last_yaw_scale < 1.0


def test_yaw_priority_one_defends_yaw_at_the_cost_of_roll():
    """The other end of the knob: yaw is not scaled first, so the shortfall is
    shared and roll pays for it."""
    roll, yaw = 0.8, 0.9
    low = Mixer(quad_x(), idle=0.05, yaw_priority=0.0)
    high = Mixer(quad_x(), idle=0.05, yaw_priority=1.0)
    a_low = low.achieved(low.mix(roll, 0.0, yaw, 0.5))
    a_high = high.achieved(high.mix(roll, 0.0, yaw, 0.5))
    assert a_high[2] > a_low[2]  # more yaw retained
    assert a_high[0] < a_low[0]  # less roll retained
    assert a_high[0] < roll
    assert high.last_saturation > 0.0


def test_idle_floor_is_respected_at_zero_thrust():
    """Armed motors are never commanded to zero: a stopped motor is a crash."""
    m = Mixer(quad_x(), idle=0.09)
    u = m.mix(0.0, 0.0, 0.0, 0.0)
    assert np.all(u >= 0.09 - 1e-12)


def test_quad_x_geometry_signs_are_consistent():
    m = Mixer(quad_x(), idle=0.0)
    # Pure roll right: the left-side motors must run harder than the right.
    u = m.mix(0.3, 0.0, 0.0, 0.5)
    # M1 front-right and M3 rear-right down; M2 rear-left and M4 front-left up.
    assert u[0] < u[1] and u[2] < u[3]
    # Pure pitch up: front motors down, rear motors up.
    u = m.mix(0.0, 0.3, 0.0, 0.5)
    assert u[0] > u[2] and u[3] > u[1]


def test_hexa_is_overactuated_and_pinv_gives_a_minimum_norm_solution():
    geom = hexa_x()
    m = Mixer(geom, idle=0.0)
    demand = np.array([0.2, 0.1, 0.05, 0.6])
    u = m.allocate_raw(*demand)
    assert np.allclose(m.achieved(u), demand, atol=1e-9)
    # Any vector in the null space added to u must increase its norm.
    ns = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    ns = ns - geom.effectiveness.T @ np.linalg.pinv(geom.effectiveness.T) @ ns
    if np.linalg.norm(ns) > 1e-9:
        assert np.linalg.norm(u + 0.1 * ns) >= np.linalg.norm(u) - 1e-12


def test_vtol_pusher_is_excluded_from_attitude_allocation():
    m = Mixer(vtol_quadplane(), idle=0.05, passthrough_motors=(4,))
    u = m.mix(0.5, 0.0, 0.0, 0.9, passthrough=[0.7])
    assert u[4] == pytest.approx(0.7)  # pusher untouched by desaturation
    assert np.all(u[:4] >= 0.05 - 1e-12) and np.all(u[:4] <= 1.0 + 1e-12)
    # And it still holds when the lift motors are fully saturated.
    u2 = m.mix(2.0, 0.0, 0.0, 1.0, passthrough=[0.3])
    assert u2[4] == pytest.approx(0.3)


def test_vtol_pusher_contributes_nothing_to_attitude():
    geom = vtol_quadplane()
    assert np.allclose(geom.effectiveness[:, 4], 0.0)


def test_mixer_input_validation():
    with pytest.raises(ValueError):
        Mixer(quad_x(), idle=1.0, max_command=1.0)
    with pytest.raises(ValueError):
        Mixer(quad_x(), yaw_priority=1.5)
    m = Mixer(quad_x())
    with pytest.raises(ValueError):
        m.achieved([0.1, 0.2])
    mv = Mixer(vtol_quadplane(), passthrough_motors=(4,))
    with pytest.raises(ValueError):
        mv.mix(0.0, 0.0, 0.0, 0.5, passthrough=[0.1, 0.2])
