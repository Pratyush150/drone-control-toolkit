"""LQR tests: the gain must actually stabilise, not merely be produced."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.lqr import (
    brysons_rule,
    c2d,
    closed_loop_poles,
    controllability_matrix,
    dlqr,
    expm,
    is_controllable,
)


def test_expm_matches_known_closed_forms():
    # Nilpotent: exp([[0,1],[0,0]]*t) = [[1,t],[0,1]]
    assert np.allclose(expm(np.array([[0.0, 1.0], [0.0, 0.0]]) * 0.7),
                       np.array([[1.0, 0.7], [0.0, 1.0]]))
    # Diagonal
    assert np.allclose(expm(np.diag([-1.0, 2.0])), np.diag([np.exp(-1.0), np.exp(2.0)]))
    # Rotation generator: exp([[0,-w],[w,0]]*t) is a rotation by w*t
    w, t = 3.0, 0.4
    R = expm(np.array([[0.0, -w], [w, 0.0]]) * t)
    assert np.allclose(R, np.array([[np.cos(w * t), -np.sin(w * t)],
                                    [np.sin(w * t), np.cos(w * t)]]))


def test_expm_handles_a_large_norm_via_scaling_and_squaring():
    A = np.array([[-50.0, 10.0], [0.0, -30.0]])
    got = expm(A * 0.5)
    # exp of an upper-triangular matrix has exp of the diagonal on its diagonal.
    assert got[0, 0] == pytest.approx(np.exp(-25.0), rel=1e-9)
    assert got[1, 1] == pytest.approx(np.exp(-15.0), rel=1e-9)
    assert got[1, 0] == pytest.approx(0.0, abs=1e-15)


def test_c2d_zoh_matches_the_analytic_double_integrator():
    """ZOH of a double integrator is exactly [[1,dt],[0,1]], [[dt^2/2],[dt]]."""
    dt = 0.02
    Ad, Bd = c2d([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], dt)
    assert np.allclose(Ad, [[1.0, dt], [0.0, 1.0]])
    assert np.allclose(Bd, [[dt * dt / 2.0], [dt]])


def test_c2d_euler_is_measurably_worse_than_zoh():
    """Documented error of the Euler option: the discrete pole is 1 - a*dt
    instead of exp(-a*dt)."""
    a, dt = 20.0, 0.02
    Ad_z, _ = c2d([[-a]], [[1.0]], dt, method="zoh")
    Ad_e, _ = c2d([[-a]], [[1.0]], dt, method="euler")
    assert Ad_z[0, 0] == pytest.approx(np.exp(-a * dt))
    assert Ad_e[0, 0] == pytest.approx(1.0 - a * dt)
    assert abs(Ad_e[0, 0] - Ad_z[0, 0]) > 0.05


def test_controllability_detects_a_dead_input():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert is_controllable(A, np.array([[0.0], [1.0]]))
    # Input that cannot touch the second state.
    assert not is_controllable(A, np.array([[1.0], [0.0]]))
    assert controllability_matrix(A, np.array([[0.0], [1.0]])).shape == (2, 2)


def test_dlqr_refuses_an_uncontrollable_plant():
    A = np.array([[0.5, 0.0], [0.0, 0.9]])
    B = np.array([[1.0], [0.0]])  # second mode has no input at all
    with pytest.raises(ValueError, match="controllable"):
        dlqr(A, B, np.eye(2), np.eye(1))


def test_dlqr_stabilises_an_unstable_plant():
    """The core claim: LQR takes a plant with a pole outside the unit circle
    and places every closed-loop pole strictly inside it."""
    A = np.array([[1.2, 0.3], [0.0, 1.05]])  # both poles unstable
    B = np.array([[0.0], [1.0]])
    assert np.max(np.abs(np.linalg.eigvals(A))) > 1.0
    res = dlqr(A, B, np.eye(2), np.array([[1.0]]))
    assert res.converged
    assert res.is_stable
    assert res.spectral_radius < 1.0
    assert np.all(np.abs(closed_loop_poles(A, B, res.K)) < 1.0)


def test_dlqr_stabilises_a_discretised_inverted_pendulum():
    """A physical unstable plant: inverted pendulum linearised about upright,
    torque input, discretised at 100 Hz."""
    g, ell = 9.81, 0.5
    A = np.array([[0.0, 1.0], [g / ell, 0.0]])  # theta'' = (g/l) theta + u
    B = np.array([[0.0], [1.0]])
    Ad, Bd = c2d(A, B, 0.01)
    assert np.max(np.abs(np.linalg.eigvals(Ad))) > 1.0
    Q, R = brysons_rule([np.deg2rad(5.0), np.deg2rad(60.0)], [5.0])
    res = dlqr(Ad, Bd, Q, R)
    assert res.is_stable

    # Simulate from a 5 degree tilt; it must come back to upright.
    x = np.array([np.deg2rad(5.0), 0.0])
    for _ in range(2000):
        x = Ad @ x + Bd @ (-res.K @ x)
    assert abs(x[0]) < 1e-3


def test_higher_r_produces_a_smaller_gain():
    """Sanity on the tuning knob: more expensive control means less of it."""
    Ad, Bd = c2d([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], 0.01)
    cheap = dlqr(Ad, Bd, np.eye(2), np.array([[0.01]]))
    dear = dlqr(Ad, Bd, np.eye(2), np.array([[100.0]]))
    assert np.linalg.norm(dear.K) < np.linalg.norm(cheap.K)
    assert cheap.spectral_radius < dear.spectral_radius  # cheaper control is faster


def test_riccati_solution_satisfies_the_equation():
    """Residual check: P must actually solve the DARE, not just stop moving."""
    Ad, Bd = c2d([[0.0, 1.0], [-2.0, -0.3]], [[0.0], [1.0]], 0.01)
    Q, R = np.eye(2), np.array([[0.5]])
    res = dlqr(Ad, Bd, Q, R)
    P = res.P
    S = R + Bd.T @ P @ Bd
    residual = Ad.T @ P @ Ad - P - Ad.T @ P @ Bd @ np.linalg.solve(S, Bd.T @ P @ Ad) + Q
    assert np.max(np.abs(residual)) < 1e-8
    assert np.allclose(P, P.T)


def test_brysons_rule_normalises_by_the_squared_limits():
    Q, R = brysons_rule([0.1, 2.0], [5.0], rho=2.0)
    assert Q[0, 0] == pytest.approx(100.0)
    assert Q[1, 1] == pytest.approx(0.25)
    assert R[0, 0] == pytest.approx(2.0 / 25.0)
    with pytest.raises(ValueError):
        brysons_rule([0.0], [1.0])


def test_singular_r_is_rejected():
    Ad, Bd = c2d([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], 0.01)
    with pytest.raises(ValueError, match="positive definite"):
        dlqr(Ad, Bd, np.eye(2), np.array([[0.0]]))


def test_multi_input_plant_is_handled():
    A = np.array([[1.1, 0.1], [0.0, 0.95]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = dlqr(A, B, np.eye(2), np.eye(2))
    assert res.K.shape == (2, 2)
    assert res.is_stable
