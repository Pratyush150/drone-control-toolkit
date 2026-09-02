"""Cascaded control loops, arranged the way a flight stack actually arranges them.

::

    position --> velocity --> attitude --> rate --> mixer --> motors
     ~5 Hz       ~20 Hz       ~50 Hz     ~500 Hz

Each loop's output is the *setpoint* of the loop inside it. The outer loops
never touch the actuators. This buys you three things that a single monolithic
controller does not give you:

1. **Rate limiting where it belongs.** A position loop that wants 40 m/s cannot
   command it, because the velocity setpoint is clamped to what the airframe
   can do. Limits sit between loops, in engineering units you can reason about,
   instead of being an abstract output clamp.
2. **Disturbance rejection at the right bandwidth.** A gust disturbs body rate
   long before it disturbs position. The rate loop sees it first and rejects it
   at 500 Hz, before the outer loops know anything happened.
3. **Debuggability.** When the aircraft misbehaves you can look at each loop's
   tracking error separately and find the one that is not following its
   setpoint.

The rate-loop-fastest convention
--------------------------------
Each loop must be roughly 3-5x slower in bandwidth than the loop inside it. If
the attitude loop is anywhere near as fast as the rate loop, the two interact:
the attitude loop commands a rate the rate loop has not achieved yet, integrates
the resulting error, and you get a low-frequency oscillation that looks like a
badly tuned attitude loop but is actually a bandwidth-separation problem. The
factor of 3-5 is what makes the "inner loop is instantaneous" approximation --
which is what lets you tune each loop against a simple plant -- actually true.

Tuning order
------------
**Always inner loop first, and never move outward until the inner one is
finished.** Rate, then attitude, then velocity, then position. Tuning an outer
loop against an inner loop that is still wrong is fitting noise: every change
you make to the inner loop invalidates the outer tune. Concretely, on a
multirotor:

1. Rate loop, in acro/manual, on a test stand or in a large open space. P until
   it is crisp then oscillates, back off ~30 %, add D to damp, add just enough
   I to hold against a static imbalance.
2. Attitude loop, in stabilised mode. Almost always P-only. If you need I here,
   your rate loop's I is wrong.
3. Velocity loop, then position loop, both usually P with a small I.

Full procedure in ``docs/CONTROL_NOTES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .pid import PID, AntiWindup

__all__ = ["LoopSpec", "CascadeLoop", "CascadeController", "MultirotorCascade"]


@dataclass
class LoopSpec:
    """Declarative description of one loop in a cascade.

    Attributes
    ----------
    name:
        Loop label, used in diagnostics.
    kp, ki, kd, kff:
        Gains for this loop's PID.
    output_limit:
        Symmetric limit on this loop's output, i.e. on the *setpoint handed to
        the next loop in*. This is the rate limit that makes the cascade safe:
        a position loop with ``output_limit=5.0`` can never ask for more than
        5 m/s no matter how large the position error is.
    slew_limit:
        Maximum rate of change of this loop's output, in output-units per
        second. ``None`` disables. Distinct from ``output_limit``: the latter
        bounds the value, this bounds how fast it moves. Use it to stop a step
        in the outer setpoint from arriving at the inner loop as a step, which
        is what excites the inner loop's transient response for no good reason.
    rate_hz:
        Nominal execution rate. Recorded for the bandwidth-separation check in
        :meth:`CascadeController.check_bandwidth_separation`; it does not by
        itself schedule anything.
    derivative_cutoff_hz:
        D-term filter cutoff. Outer loops run slower and can use lower cutoffs.
    """

    name: str
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    kff: float = 0.0
    output_limit: Optional[float] = None
    slew_limit: Optional[float] = None
    rate_hz: float = 100.0
    derivative_cutoff_hz: Optional[float] = 20.0
    anti_windup: AntiWindup = AntiWindup.BACK_CALCULATION


class CascadeLoop:
    """One stage: a PID plus an output rate limiter."""

    def __init__(self, spec: LoopSpec) -> None:
        self.spec = spec
        limits = (
            (-spec.output_limit, spec.output_limit)
            if spec.output_limit is not None
            else (None, None)
        )
        self.pid = PID(
            spec.kp,
            spec.ki,
            spec.kd,
            spec.kff,
            output_limits=limits,
            anti_windup=spec.anti_windup,
            derivative_cutoff_hz=spec.derivative_cutoff_hz,
            name=spec.name,
        )
        self.last_output = 0.0
        self.last_error = 0.0
        self.slew_limited = False

    def reset(self) -> None:
        self.pid.reset()
        self.last_output = 0.0
        self.last_error = 0.0
        self.slew_limited = False

    def update(
        self, setpoint: float, measurement: float, dt: float, feedforward: Optional[float] = None
    ) -> float:
        raw = self.pid.update(setpoint, measurement, dt, feedforward=feedforward)
        out = raw
        self.slew_limited = False
        if self.spec.slew_limit is not None:
            max_step = self.spec.slew_limit * dt
            delta = out - self.last_output
            if abs(delta) > max_step:
                out = self.last_output + np.sign(delta) * max_step
                self.slew_limited = True
        self.last_output = float(out)
        self.last_error = float(setpoint - measurement)
        return float(out)

    @property
    def saturated(self) -> bool:
        return self.pid.state.saturated or self.slew_limited


class CascadeController:
    """A chain of :class:`CascadeLoop` stages, outermost first.

    Call :meth:`update` with the outermost setpoint and the measurement vector
    (one measurement per loop, outermost first). Each loop's output becomes the
    next loop's setpoint; the innermost loop's output is the actuator command.
    """

    def __init__(self, specs: Sequence[LoopSpec]) -> None:
        if not specs:
            raise ValueError("need at least one loop")
        self.loops = [CascadeLoop(s) for s in specs]

    def __len__(self) -> int:
        return len(self.loops)

    def __getitem__(self, key):
        if isinstance(key, str):
            for loop in self.loops:
                if loop.spec.name == key:
                    return loop
            raise KeyError(key)
        return self.loops[key]

    def reset(self) -> None:
        for loop in self.loops:
            loop.reset()

    def update(
        self,
        setpoint: float,
        measurements: Sequence[float],
        dt: float,
        *,
        feedforwards: Optional[Sequence[Optional[float]]] = None,
    ) -> float:
        """Run the full chain and return the innermost output."""
        meas = list(measurements)
        if len(meas) != len(self.loops):
            raise ValueError(f"expected {len(self.loops)} measurements, got {len(meas)}")
        ff = list(feedforwards) if feedforwards is not None else [None] * len(self.loops)
        sp = float(setpoint)
        for loop, m, f in zip(self.loops, meas, ff):
            sp = loop.update(sp, float(m), dt, feedforward=f)
        return sp

    @property
    def setpoints(self) -> list[float]:
        """The setpoint each inner loop was last handed. Log this."""
        return [loop.last_output for loop in self.loops]

    @property
    def errors(self) -> list[float]:
        return [loop.last_error for loop in self.loops]

    def check_bandwidth_separation(self, *, min_ratio: float = 3.0) -> list[str]:
        """Return a warning for every adjacent pair that is too close in rate.

        Not a hard failure, because there are legitimate designs that break the
        rule. But if a cascade is misbehaving and this returns anything, start
        there before you touch a gain.
        """
        warnings: list[str] = []
        for outer, inner in zip(self.loops[:-1], self.loops[1:]):
            ratio = inner.spec.rate_hz / outer.spec.rate_hz
            if ratio < min_ratio:
                warnings.append(
                    f"{outer.spec.name} ({outer.spec.rate_hz:g} Hz) -> "
                    f"{inner.spec.name} ({inner.spec.rate_hz:g} Hz): ratio {ratio:.1f} "
                    f"< {min_ratio:g}; the loops will interact"
                )
        return warnings


@dataclass
class MultirotorCascade:
    """Position -> velocity -> attitude -> rate for one horizontal axis, plus altitude.

    This is a *reference wiring*, not a flight controller. It exists so the
    structure and the limits are written down somewhere concrete, and so the
    example scripts can close a realistic multi-loop loop against
    :mod:`dctk.sim`.

    The horizontal chain maps as follows on a real airframe:

    ====================  ===============  =========================
    loop                  output           physical meaning
    ====================  ===============  =========================
    position (m)          velocity (m/s)   "fly toward the waypoint"
    velocity (m/s)        tilt (rad)       "lean to accelerate"
    attitude (rad)        body rate (rad/s) "rotate to that lean"
    rate (rad/s)          torque (N.m)     "apply torque"
    ====================  ===============  =========================

    Note the velocity -> attitude step: on a multirotor, horizontal
    acceleration *is* tilt, because the only force you can steer is the thrust
    vector. ``a_x = g * tan(theta)`` for small angles. That is why
    ``max_tilt_rad`` is the real limit on horizontal acceleration and why an
    aggressive position tune shows up as an aggressive lean.

    The altitude chain is shorter (altitude -> climb rate -> thrust) because
    vertical acceleration is directly commanded by collective thrust; no
    attitude is involved.
    """

    max_velocity: float = 5.0
    max_tilt_rad: float = float(np.deg2rad(30.0))
    max_rate: float = float(np.deg2rad(220.0))
    max_torque: float = 0.5
    max_climb_rate: float = 3.0
    max_thrust: float = 1.0
    hover_thrust: float = 0.5

    position: CascadeController = field(init=False)
    altitude: CascadeController = field(init=False)

    def __post_init__(self) -> None:
        self.position = CascadeController(
            [
                LoopSpec(
                    "position",
                    kp=1.0,
                    ki=0.0,
                    kd=0.0,
                    output_limit=self.max_velocity,
                    slew_limit=self.max_velocity * 2.0,
                    rate_hz=50.0,
                    derivative_cutoff_hz=5.0,
                ),
                LoopSpec(
                    "velocity",
                    kp=0.25,
                    ki=0.08,
                    kd=0.01,
                    output_limit=self.max_tilt_rad,
                    rate_hz=250.0,
                    derivative_cutoff_hz=10.0,
                ),
                LoopSpec(
                    "attitude",
                    kp=8.0,
                    ki=0.0,
                    kd=0.0,
                    output_limit=self.max_rate,
                    rate_hz=1000.0,
                    derivative_cutoff_hz=30.0,
                ),
                LoopSpec(
                    "rate",
                    kp=0.06,
                    ki=0.15,
                    kd=0.002,
                    output_limit=self.max_torque,
                    rate_hz=4000.0,
                    derivative_cutoff_hz=60.0,
                ),
            ]
        )
        self.altitude = CascadeController(
            [
                LoopSpec(
                    "altitude",
                    kp=1.2,
                    ki=0.0,
                    kd=0.0,
                    output_limit=self.max_climb_rate,
                    rate_hz=50.0,
                    derivative_cutoff_hz=5.0,
                ),
                LoopSpec(
                    "climb_rate",
                    kp=0.25,
                    ki=0.12,
                    kd=0.01,
                    output_limit=self.max_thrust,
                    rate_hz=250.0,
                    derivative_cutoff_hz=15.0,
                ),
            ]
        )

    def reset(self) -> None:
        self.position.reset()
        self.altitude.reset()

    def update_horizontal(
        self, x_setpoint: float, x: float, vx: float, theta: float, q: float, dt: float
    ) -> float:
        """Full position chain. Returns commanded pitch torque (N.m).

        The sign convention here is "pitch nose down to move forward", which is
        why the velocity loop's output is negated before it becomes the
        attitude setpoint. Getting this sign wrong is the single most common
        cascade bug and it presents as the aircraft accelerating away from the
        waypoint -- which looks like an unstable position loop but is not.
        """
        v_sp = self.position[0].update(x_setpoint, x, dt)
        tilt_sp = self.position[1].update(v_sp, vx, dt)
        theta_sp = -tilt_sp
        rate_sp = self.position[2].update(theta_sp, theta, dt)
        return self.position[3].update(rate_sp, q, dt)

    def update_altitude(self, z_setpoint: float, z: float, vz: float, dt: float) -> float:
        """Altitude chain. Returns normalised collective thrust in [0, max].

        ``hover_thrust`` is applied as a feed-forward, not learned by the
        integrator. Two reasons: the integrator would need seconds to wind up
        to hover thrust on every mode entry (a visible sag on takeoff), and a
        wound-up integrator carrying the entire hover command is exactly the
        state that produces a large transient when the loop is re-enabled.
        """
        vz_sp = self.altitude[0].update(z_setpoint, z, dt)
        thrust = self.altitude[1].update(vz_sp, vz, dt)
        return float(np.clip(thrust + self.hover_thrust, 0.0, self.max_thrust))

    def diagnostics(self) -> dict[str, float]:
        """Per-loop errors and saturation flags, ready to log."""
        out: dict[str, float] = {}
        for chain in (self.position, self.altitude):
            for loop in chain.loops:
                out[f"{loop.spec.name}_error"] = loop.last_error
                out[f"{loop.spec.name}_sat"] = float(loop.saturated)
        return out
