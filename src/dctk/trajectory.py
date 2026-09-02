"""Trajectory generation: minimum-jerk, trapezoidal, and spline paths.

Why generate a trajectory at all instead of feeding the controller a step:

A step setpoint asks for infinite velocity and infinite acceleration. The
controller cannot deliver either, so it saturates, the integrator winds up, and
the resulting motion is decided by your saturation limits rather than by
anything you designed. Feeding a *feasible* reference -- one whose velocity and
acceleration stay inside the airframe's limits -- means the feedback loop only
ever has to correct small errors, which is the regime where the linear tuning
you did is actually valid.

The velocity and acceleration profiles produced here are also the natural
feed-forward signals for a cascade (see :mod:`dctk.cascade`): hand the position
to the position loop, the velocity to the velocity loop as feed-forward, and the
acceleration to the attitude loop as a tilt feed-forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Trajectory",
    "min_jerk",
    "min_jerk_duration",
    "trapezoidal_profile",
    "CubicSpline",
    "QuinticSegment",
    "waypoint_path",
    "check_limits",
]


@dataclass
class Trajectory:
    """Sampled position/velocity/acceleration reference.

    Arrays are ``(n,)`` for a scalar trajectory or ``(n, d)`` for ``d``
    dimensions.
    """

    t: NDArray[np.float64]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def peak_velocity(self) -> float:
        return float(np.max(np.abs(self.velocity)))

    @property
    def peak_acceleration(self) -> float:
        return float(np.max(np.abs(self.acceleration)))

    def at(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Linearly interpolate the trajectory at time ``t`` (clamped to range).

        Linear interpolation of an already-smooth trajectory sampled finely is
        fine; do not use this to resample a coarse trajectory and then expect
        the acceleration to still be continuous.
        """
        tt = float(np.clip(t, self.t[0], self.t[-1]))
        if self.position.ndim == 1:
            return (
                np.interp(tt, self.t, self.position),
                np.interp(tt, self.t, self.velocity),
                np.interp(tt, self.t, self.acceleration),
            )
        def col(arr):
            return np.array(
                [np.interp(tt, self.t, arr[:, i]) for i in range(arr.shape[1])]
            )

        return col(self.position), col(self.velocity), col(self.acceleration)


# ======================================================================
# minimum jerk
# ======================================================================
def min_jerk(
    start: ArrayLike, goal: ArrayLike, duration: float, *, n: int = 200, t0: float = 0.0
) -> Trajectory:
    """Minimum-jerk point-to-point trajectory.

    Minimising ``integral(jerk^2)`` subject to zero velocity *and* zero
    acceleration at both ends gives a unique quintic:

    ``s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5``,  ``tau = t / T``

    which is why every implementation you have ever seen has those three
    coefficients in it. The endpoint acceleration constraint is what makes it
    useful on an aircraft: a trajectory that starts with non-zero acceleration
    demands a step change in thrust or tilt at ``t=0``, and neither is
    physically available. Zero acceleration at the endpoints means the
    reference is continuously realisable from a standstill.

    Peak velocity is ``1.875 * distance / T`` and peak acceleration is
    ``5.7735 * distance / T^2`` (both exact, at ``tau = 0.5`` and
    ``tau = 0.5 -/+ 1/(2*sqrt(3))`` respectively). Use
    :func:`min_jerk_duration` to invert those and pick ``T`` from limits.
    """
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    if n < 2:
        raise ValueError("n must be >= 2")
    p0 = np.atleast_1d(np.asarray(start, dtype=float))
    p1 = np.atleast_1d(np.asarray(goal, dtype=float))
    if p0.shape != p1.shape:
        raise ValueError("start and goal must have the same shape")

    t = np.linspace(0.0, duration, n)
    tau = t / duration
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2

    delta = p1 - p0
    pos = p0[None, :] + s[:, None] * delta[None, :]
    vel = sd[:, None] * delta[None, :]
    acc = sdd[:, None] * delta[None, :]
    if p0.size == 1:
        pos, vel, acc = pos.ravel(), vel.ravel(), acc.ravel()
    return Trajectory(t=t + t0, position=pos, velocity=vel, acceleration=acc)


def min_jerk_duration(
    distance: float,
    *,
    max_velocity: Optional[float] = None,
    max_acceleration: Optional[float] = None,
) -> float:
    """Shortest minimum-jerk duration that respects the given limits.

    Solves ``1.875 d / T <= v_max`` and ``5.7735 d / T^2 <= a_max`` and returns
    the binding one. At least one limit must be supplied.
    """
    d = abs(float(distance))
    if d == 0.0:
        return 0.0
    candidates = []
    if max_velocity is not None:
        if max_velocity <= 0.0:
            raise ValueError("max_velocity must be > 0")
        candidates.append(1.875 * d / max_velocity)
    if max_acceleration is not None:
        if max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be > 0")
        candidates.append(float(np.sqrt(10.0 * d / (np.sqrt(3.0) * max_acceleration))))
    if not candidates:
        raise ValueError("supply at least one of max_velocity or max_acceleration")
    return float(max(candidates))


# ======================================================================
# trapezoidal
# ======================================================================
def trapezoidal_profile(
    start: float,
    goal: float,
    max_velocity: float,
    max_acceleration: float,
    *,
    dt: float = 0.01,
    t0: float = 0.0,
) -> Trajectory:
    """Constant-acceleration / constant-velocity / constant-deceleration profile.

    The workhorse of every motion controller, and the fastest way to cover a
    distance under a velocity *and* an acceleration limit. Compared with
    minimum jerk it is quicker for the same limits, but its acceleration is
    discontinuous at the three corners, which excites every lightly-damped mode
    in the structure. On a rigid gantry that is fine. On an airframe with a
    flexible arm or a gimbal hanging off it, the corners ring.

    If the distance is too short to reach ``max_velocity`` the profile
    degenerates to a triangle (accelerate, decelerate) and this function
    handles that case rather than producing a negative cruise time -- the
    classic bug in hand-rolled implementations.
    """
    if max_velocity <= 0.0 or max_acceleration <= 0.0:
        raise ValueError("limits must be > 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    distance = float(goal) - float(start)
    direction = float(np.sign(distance)) if distance != 0.0 else 1.0
    d = abs(distance)

    t_acc = max_velocity / max_acceleration
    d_acc = 0.5 * max_acceleration * t_acc**2
    if 2.0 * d_acc >= d:
        # Triangular: never reaches cruise velocity.
        t_acc = float(np.sqrt(d / max_acceleration)) if d > 0 else 0.0
        t_cruise = 0.0
        v_peak = max_acceleration * t_acc
    else:
        t_cruise = (d - 2.0 * d_acc) / max_velocity
        v_peak = max_velocity

    total = 2.0 * t_acc + t_cruise
    n = max(2, int(round(total / dt)) + 1)
    t = np.linspace(0.0, total, n)

    pos = np.zeros(n)
    vel = np.zeros(n)
    acc = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < t_acc:
            acc[i] = max_acceleration
            vel[i] = max_acceleration * ti
            pos[i] = 0.5 * max_acceleration * ti**2
        elif ti < t_acc + t_cruise:
            tc = ti - t_acc
            acc[i] = 0.0
            vel[i] = v_peak
            pos[i] = 0.5 * max_acceleration * t_acc**2 + v_peak * tc
        else:
            td = min(ti - t_acc - t_cruise, t_acc)
            acc[i] = -max_acceleration
            vel[i] = v_peak - max_acceleration * td
            pos[i] = (
                0.5 * max_acceleration * t_acc**2
                + v_peak * t_cruise
                + v_peak * td
                - 0.5 * max_acceleration * td**2
            )
    pos[-1] = d
    vel[-1] = 0.0
    acc[-1] = 0.0
    return Trajectory(
        t=t + t0,
        position=float(start) + direction * pos,
        velocity=direction * vel,
        acceleration=direction * acc,
    )


# ======================================================================
# splines
# ======================================================================
class QuinticSegment:
    """Quintic polynomial through given boundary position/velocity/acceleration.

    Six coefficients, six constraints. This is the building block for stitching
    waypoints together with continuous acceleration, which matters because a
    discontinuity in acceleration is a discontinuity in commanded tilt, and the
    attitude loop cannot follow a step.
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        duration: float,
        *,
        v0: float = 0.0,
        v1: float = 0.0,
        a0: float = 0.0,
        a1: float = 0.0,
    ) -> None:
        if duration <= 0.0:
            raise ValueError("duration must be > 0")
        T = float(duration)
        self.duration = T
        c0, c1, c2 = float(p0), float(v0), 0.5 * float(a0)
        # Solve the remaining three from the terminal conditions.
        M = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ]
        )
        rhs = np.array(
            [
                p1 - (c0 + c1 * T + c2 * T**2),
                v1 - (c1 + 2 * c2 * T),
                a1 - 2 * c2,
            ]
        )
        c3, c4, c5 = np.linalg.solve(M, rhs)
        self.coeffs = np.array([c0, c1, c2, c3, c4, c5])

    def evaluate(self, t: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        tt = np.asarray(t, dtype=float)
        c = self.coeffs
        pos = c[0] + c[1] * tt + c[2] * tt**2 + c[3] * tt**3 + c[4] * tt**4 + c[5] * tt**5
        vel = c[1] + 2 * c[2] * tt + 3 * c[3] * tt**2 + 4 * c[4] * tt**3 + 5 * c[5] * tt**4
        acc = 2 * c[2] + 6 * c[3] * tt + 12 * c[4] * tt**2 + 20 * c[5] * tt**3
        return pos, vel, acc


class CubicSpline:
    """Natural cubic spline interpolation, one axis, from scratch.

    Solves the standard tridiagonal system for the second derivatives with a
    Thomas sweep. "Natural" boundary conditions (zero second derivative at the
    ends) mean zero acceleration at the path endpoints, which is what you want
    for a path that starts and ends at rest.

    Used by :func:`waypoint_path`. It is a *geometric* interpolant: it gives a
    smooth curve through the waypoints but says nothing about whether the
    resulting velocities are flyable. That is what :func:`check_limits` is for.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        xs = np.asarray(x, dtype=float).ravel()
        ys = np.asarray(y, dtype=float).ravel()
        if xs.size != ys.size:
            raise ValueError("x and y must be the same length")
        if xs.size < 3:
            raise ValueError("need at least 3 points for a cubic spline")
        if np.any(np.diff(xs) <= 0.0):
            raise ValueError("x must be strictly increasing")
        self.x, self.y = xs, ys
        n = xs.size
        h = np.diff(xs)

        # Tridiagonal system for the interior second derivatives.
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.zeros(n)
        for i in range(1, n - 1):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])

        # Thomas algorithm.
        cp = np.zeros(n)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        m = np.zeros(n)
        m[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            m[i] = dp[i] - cp[i] * m[i + 1]
        self.m = m
        self.h = h

    def __call__(self, xq: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        """Return ``(y, dy/dx, d2y/dx2)`` at the query points."""
        q = np.atleast_1d(np.asarray(xq, dtype=float))
        idx = np.clip(np.searchsorted(self.x, q) - 1, 0, self.x.size - 2)
        h = self.h[idx]
        xl, xr = self.x[idx], self.x[idx + 1]
        yl, yr = self.y[idx], self.y[idx + 1]
        ml, mr = self.m[idx], self.m[idx + 1]
        A = (xr - q) / h
        B = (q - xl) / h
        y = A * yl + B * yr + ((A**3 - A) * ml + (B**3 - B) * mr) * (h**2) / 6.0
        dy = (yr - yl) / h + ((-3 * A**2 + 1) * ml + (3 * B**2 - 1) * mr) * h / 6.0
        d2y = A * ml + B * mr
        return y, dy, d2y


def waypoint_path(
    waypoints: ArrayLike,
    *,
    average_speed: float = 1.0,
    n: int = 400,
    method: str = "cubic",
) -> Trajectory:
    """Smooth path through waypoints, time-parameterised by path length.

    Parameters
    ----------
    waypoints:
        ``(k, d)`` array. ``k >= 3`` for the cubic method.
    average_speed:
        Used to assign a time to each waypoint from the chord length between
        them. This is a *nominal* speed, not a guarantee -- the spline is
        longer than the chords, so the realised speed is higher. Always run
        :func:`check_limits` on the result.
    method:
        ``'cubic'`` for a natural cubic spline (C2 through the waypoints, zero
        acceleration at the ends), or ``'quintic'`` for per-segment quintics
        that additionally come to a full stop at each waypoint. Quintic gives
        you a stop-and-go path; cubic gives you a fly-through path. Pick based
        on whether the waypoints are places you need to *be* or places you need
        to *pass*.
    """
    wp = np.atleast_2d(np.asarray(waypoints, dtype=float))
    if wp.ndim != 2 or wp.shape[0] < 2:
        raise ValueError("waypoints must be (k, d) with k >= 2")
    if average_speed <= 0.0:
        raise ValueError("average_speed must be > 0")

    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    if np.any(seg_len <= 0.0):
        raise ValueError("waypoints must be distinct")
    knots = np.concatenate([[0.0], np.cumsum(seg_len / average_speed)])

    if method == "quintic":
        segments = []
        times = []
        for i in range(wp.shape[0] - 1):
            T = knots[i + 1] - knots[i]
            k = max(2, int(round(n / (wp.shape[0] - 1))))
            ts = np.linspace(0.0, T, k, endpoint=(i == wp.shape[0] - 2))
            per_dim = [
                QuinticSegment(wp[i, j], wp[i + 1, j], T).evaluate(ts) for j in range(wp.shape[1])
            ]
            segments.append(per_dim)
            times.append(ts + knots[i])
        t = np.concatenate(times)
        dims = range(wp.shape[1])
        pos, vel, acc = (
            np.column_stack(
                [np.concatenate([seg[j][k] for seg in segments]) for j in dims]
            )
            for k in (0, 1, 2)
        )
        return Trajectory(t=t, position=pos, velocity=vel, acceleration=acc)

    if method != "cubic":
        raise ValueError(f"unknown method {method!r}")
    if wp.shape[0] < 3:
        raise ValueError("cubic method needs at least 3 waypoints; use method='quintic'")

    t = np.linspace(knots[0], knots[-1], n)
    cols = [CubicSpline(knots, wp[:, j])(t) for j in range(wp.shape[1])]
    pos = np.column_stack([c[0] for c in cols])
    vel = np.column_stack([c[1] for c in cols])
    acc = np.column_stack([c[2] for c in cols])
    return Trajectory(t=t, position=pos, velocity=vel, acceleration=acc)


def check_limits(
    traj: Trajectory,
    *,
    max_velocity: Optional[float] = None,
    max_acceleration: Optional[float] = None,
) -> dict[str, float | bool]:
    """Check a trajectory against airframe limits.

    Returns the peaks and a per-limit pass flag. For a multi-dimensional
    trajectory the norm across dimensions is used, not the per-axis maximum:
    an airframe's acceleration limit is on the *vector*, and a trajectory that
    puts 0.9 g on x and 0.9 g on y simultaneously is asking for 1.27 g.
    """
    if traj.velocity.ndim == 1:
        v = np.abs(traj.velocity)
        a = np.abs(traj.acceleration)
    else:
        v = np.linalg.norm(traj.velocity, axis=1)
        a = np.linalg.norm(traj.acceleration, axis=1)
    result: dict[str, float | bool] = {
        "peak_velocity": float(np.max(v)),
        "peak_acceleration": float(np.max(a)),
    }
    if max_velocity is not None:
        result["velocity_ok"] = bool(result["peak_velocity"] <= max_velocity * (1 + 1e-9))
    if max_acceleration is not None:
        result["acceleration_ok"] = bool(
            result["peak_acceleration"] <= max_acceleration * (1 + 1e-9)
        )
    return result
