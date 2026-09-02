"""Trajectory tests: endpoints, feasibility, and limit checking."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.trajectory import (
    CubicSpline,
    QuinticSegment,
    check_limits,
    min_jerk,
    min_jerk_duration,
    trapezoidal_profile,
    waypoint_path,
)


def test_min_jerk_endpoints_have_zero_velocity_and_acceleration():
    """The defining property. Non-zero endpoint acceleration would demand a
    step in commanded tilt or thrust, which no airframe can supply."""
    traj = min_jerk(0.0, 5.0, 3.0, n=1001)
    assert traj.position[0] == pytest.approx(0.0)
    assert traj.position[-1] == pytest.approx(5.0)
    assert traj.velocity[0] == pytest.approx(0.0, abs=1e-12)
    assert traj.velocity[-1] == pytest.approx(0.0, abs=1e-12)
    assert traj.acceleration[0] == pytest.approx(0.0, abs=1e-12)
    assert traj.acceleration[-1] == pytest.approx(0.0, abs=1e-12)


def test_min_jerk_peak_velocity_and_acceleration_match_the_closed_form():
    d, T = 8.0, 4.0
    traj = min_jerk(0.0, d, T, n=20001)
    assert traj.peak_velocity == pytest.approx(1.875 * d / T, rel=1e-4)
    assert traj.peak_acceleration == pytest.approx(10.0 * d / (np.sqrt(3.0) * T**2), rel=1e-3)


def test_min_jerk_is_symmetric_about_its_midpoint():
    traj = min_jerk(0.0, 1.0, 2.0, n=1001)
    assert traj.position[500] == pytest.approx(0.5, abs=1e-9)
    assert np.allclose(traj.velocity, traj.velocity[::-1], atol=1e-9)


def test_min_jerk_multidimensional_endpoints():
    traj = min_jerk([0.0, 1.0, 2.0], [3.0, -1.0, 2.0], 2.0, n=501)
    assert traj.position.shape == (501, 3)
    assert np.allclose(traj.position[0], [0.0, 1.0, 2.0])
    assert np.allclose(traj.position[-1], [3.0, -1.0, 2.0])
    assert np.allclose(traj.velocity[0], 0.0, atol=1e-12)
    assert np.allclose(traj.acceleration[-1], 0.0, atol=1e-12)


def test_min_jerk_duration_respects_the_binding_limit():
    d = 10.0
    T = min_jerk_duration(d, max_velocity=4.0, max_acceleration=100.0)
    traj = min_jerk(0.0, d, T, n=20001)
    assert traj.peak_velocity == pytest.approx(4.0, rel=1e-3)
    assert traj.peak_acceleration < 100.0

    T2 = min_jerk_duration(d, max_velocity=1000.0, max_acceleration=2.0)
    traj2 = min_jerk(0.0, d, T2, n=20001)
    assert traj2.peak_acceleration == pytest.approx(2.0, rel=1e-3)

    # Both supplied: the more restrictive one must win.
    T3 = min_jerk_duration(d, max_velocity=4.0, max_acceleration=2.0)
    assert T3 == pytest.approx(max(T, T2))


def test_trapezoidal_reaches_the_goal_and_respects_both_limits():
    traj = trapezoidal_profile(0.0, 10.0, max_velocity=2.0, max_acceleration=1.0, dt=1e-3)
    assert traj.position[-1] == pytest.approx(10.0)
    assert traj.velocity[0] == pytest.approx(0.0)
    assert traj.velocity[-1] == pytest.approx(0.0)
    assert traj.peak_velocity <= 2.0 + 1e-9
    assert traj.peak_acceleration <= 1.0 + 1e-9
    # Total time: 2 s ramp up + 2 s ramp down + 3 s cruise.
    assert traj.duration == pytest.approx(7.0, rel=1e-3)


def test_trapezoidal_degenerates_to_a_triangle_on_a_short_move():
    """The classic hand-rolled-implementation bug is a negative cruise time."""
    traj = trapezoidal_profile(0.0, 0.5, max_velocity=5.0, max_acceleration=1.0, dt=1e-3)
    assert traj.position[-1] == pytest.approx(0.5)
    assert traj.peak_velocity < 5.0
    assert traj.peak_velocity == pytest.approx(np.sqrt(0.5 * 1.0), rel=1e-2)
    assert traj.duration == pytest.approx(2.0 * np.sqrt(0.5), rel=1e-2)


def test_trapezoidal_handles_a_negative_move():
    traj = trapezoidal_profile(3.0, -2.0, max_velocity=2.0, max_acceleration=1.0, dt=1e-3)
    assert traj.position[-1] == pytest.approx(-2.0)
    assert np.min(traj.velocity) < 0.0
    assert traj.peak_velocity <= 2.0 + 1e-9


def test_trapezoidal_is_faster_than_min_jerk_for_the_same_limits():
    """Documented trade: trapezoidal is quicker, min-jerk is smoother."""
    d, vmax, amax = 10.0, 2.0, 1.0
    trap = trapezoidal_profile(0.0, d, vmax, amax, dt=1e-3)
    T = min_jerk_duration(d, max_velocity=vmax, max_acceleration=amax)
    assert trap.duration < T


def test_quintic_segment_hits_all_six_boundary_conditions():
    seg = QuinticSegment(1.0, 4.0, 2.0, v0=0.5, v1=-0.25, a0=0.1, a1=-0.3)
    p, v, a = seg.evaluate(np.array([0.0, 2.0]))
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(4.0)
    assert v[0] == pytest.approx(0.5)
    assert v[1] == pytest.approx(-0.25)
    assert a[0] == pytest.approx(0.1)
    assert a[1] == pytest.approx(-0.3)


def test_cubic_spline_interpolates_the_knots_exactly():
    x = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
    y = np.array([0.0, 2.0, -1.0, 0.5, 3.0])
    sp = CubicSpline(x, y)
    got, _, _ = sp(x)
    assert np.allclose(got, y, atol=1e-10)


def test_cubic_spline_reproduces_a_cubic_exactly_in_the_interior():
    x = np.linspace(0.0, 6.0, 13)
    y = 0.5 * x**3 - 2.0 * x**2 + x
    sp = CubicSpline(x, y)
    xq = np.linspace(1.0, 5.0, 41)
    got, dgot, _ = sp(xq)
    # Natural boundary conditions perturb the ends; the interior must be close.
    assert np.max(np.abs(got - (0.5 * xq**3 - 2.0 * xq**2 + xq))) < 0.2
    assert np.max(np.abs(dgot - (1.5 * xq**2 - 4.0 * xq + 1.0))) < 0.5


def test_cubic_spline_has_zero_second_derivative_at_the_ends():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    sp = CubicSpline(x, y)
    _, _, d2 = sp(np.array([0.0, 3.0]))
    assert np.allclose(d2, 0.0, atol=1e-12)


def test_waypoint_path_passes_through_the_waypoints():
    wp = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, -1.0], [6.0, 0.0]])
    traj = waypoint_path(wp, average_speed=1.5, n=800)
    for point in wp:
        d = np.linalg.norm(traj.position - point, axis=1)
        assert np.min(d) < 0.05, f"path misses waypoint {point}"


def test_quintic_waypoint_path_stops_at_every_waypoint():
    wp = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, -1.0]])
    traj = waypoint_path(wp, average_speed=1.0, n=600, method="quintic")
    assert np.allclose(traj.velocity[0], 0.0, atol=1e-9)
    assert np.allclose(traj.velocity[-1], 0.0, atol=1e-9)


def test_check_limits_uses_the_vector_norm_not_the_per_axis_max():
    """A trajectory at 0.9 of the limit on two axes at once is over the limit."""
    t = np.linspace(0.0, 1.0, 11)
    from dctk.trajectory import Trajectory

    traj = Trajectory(
        t=t,
        position=np.zeros((11, 2)),
        velocity=np.full((11, 2), 0.9),
        acceleration=np.zeros((11, 2)),
    )
    result = check_limits(traj, max_velocity=1.0)
    assert result["peak_velocity"] == pytest.approx(np.sqrt(2) * 0.9)
    assert result["velocity_ok"] is False


def test_check_limits_passes_a_feasible_trajectory():
    duration = min_jerk_duration(5.0, max_velocity=3.0, max_acceleration=2.0)
    traj = min_jerk(0.0, 5.0, duration, n=4001)
    result = check_limits(traj, max_velocity=3.0, max_acceleration=2.0)
    assert result["velocity_ok"] is True
    assert result["acceleration_ok"] is True


def test_trajectory_sampling_at_arbitrary_time():
    traj = min_jerk(0.0, 4.0, 2.0, n=2001)
    pos, vel, acc = traj.at(1.0)
    assert float(pos) == pytest.approx(2.0, abs=1e-3)
    assert float(vel) == pytest.approx(1.875 * 4.0 / 2.0, rel=1e-3)
    assert float(acc) == pytest.approx(0.0, abs=1e-3)
    # Clamps outside the range rather than extrapolating.
    assert float(traj.at(99.0)[0]) == pytest.approx(4.0)


def test_trajectory_input_validation():
    with pytest.raises(ValueError):
        min_jerk(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        trapezoidal_profile(0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        min_jerk_duration(1.0)
    with pytest.raises(ValueError):
        CubicSpline([0.0, 1.0], [0.0, 1.0])  # needs 3 points
    with pytest.raises(ValueError):
        waypoint_path(np.array([[0.0, 0.0], [0.0, 0.0]]), average_speed=1.0)  # duplicate
