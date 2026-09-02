"""Plants and hardware-realism injectors.

A controller that only ever runs against ``y' = -y + u`` is a maths exercise.
The reason a tune that looks perfect in a clean simulation falls apart on the
bench is almost never the plant model: it is the 15 ms of latency in the
actuator path, the 30 ms motor time constant, the 12-bit quantiser on the ESC
command, and the fact that the gyro carries 0.05 rad/s of noise on top of a
slowly walking bias.

This module gives you both halves. :class:`Plant` subclasses are the physics;
the injector classes are the defects. :func:`simulate` wires a controller to a
plant through whichever defects you enable, so an example script can show the
same gains behaving well on a clean plant and marginally on a realistic one.

Everything is deterministic given a seed. Tests depend on that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Plant",
    "FirstOrderPlant",
    "SecondOrderPlant",
    "PitchPlant",
    "PointMassQuadrotor2D",
    "SensorNoise",
    "Quantiser",
    "ActuatorDelay",
    "MotorLag",
    "WindGust",
    "SimResult",
    "simulate",
]


# ======================================================================
# Plants
# ======================================================================
class Plant(ABC):
    """Base class for a continuous plant integrated with fixed-step RK4.

    RK4 rather than forward Euler because the whole point of this module is to
    let the *injected* defects dominate the result. If the integrator itself
    contributed error at the same order, you could not tell whether a marginal
    response came from the actuator lag you added or from your own numerics.
    """

    #: Names of the state elements, for logging and plotting.
    state_names: Sequence[str] = ()

    def __init__(self, x0: Optional[NDArray[np.float64]] = None) -> None:
        self.x0 = np.zeros(self.n_states) if x0 is None else np.asarray(x0, dtype=float).copy()
        self.x = self.x0.copy()

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Dimension of the state vector."""

    @abstractmethod
    def derivative(
        self, x: NDArray[np.float64], u: float, disturbance: float = 0.0
    ) -> NDArray[np.float64]:
        """Return ``dx/dt`` for state ``x`` and input ``u``."""

    def output(self, x: Optional[NDArray[np.float64]] = None) -> float:
        """Measured output. Defaults to the first state."""
        xx = self.x if x is None else x
        return float(xx[0])

    def reset(self, x0: Optional[NDArray[np.float64]] = None) -> None:
        self.x = self.x0.copy() if x0 is None else np.asarray(x0, dtype=float).copy()

    def step(self, u: float, dt: float, disturbance: float = 0.0) -> float:
        """Integrate one step of ``dt`` with a zero-order-hold input."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        x = self.x
        k1 = self.derivative(x, u, disturbance)
        k2 = self.derivative(x + 0.5 * dt * k1, u, disturbance)
        k3 = self.derivative(x + 0.5 * dt * k2, u, disturbance)
        k4 = self.derivative(x + dt * k3, u, disturbance)
        self.x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.output()


class FirstOrderPlant(Plant):
    """``tau * y' + y = K * u``.

    Stands in for a thermal loop, a motor speed loop with the electrical
    dynamics ignored, or a well-behaved velocity loop. No overshoot is possible
    with proportional control alone, which makes it the right plant for
    demonstrating integral windup in isolation.
    """

    state_names = ("y",)

    def __init__(self, tau: float = 1.0, gain: float = 1.0, y0: float = 0.0) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be > 0")
        self.tau = float(tau)
        self.gain = float(gain)
        super().__init__(np.array([y0], dtype=float))

    @property
    def n_states(self) -> int:
        return 1

    def derivative(self, x, u, disturbance=0.0):
        return np.array([(self.gain * (u + disturbance) - x[0]) / self.tau])


class SecondOrderPlant(Plant):
    """Mass-spring-damper: ``m x'' + c x' + k x = u``.

    Parameterised either by physical ``(m, c, k)`` or by
    ``(natural_frequency, damping_ratio, dc_gain)``. This is the plant to reach
    for when you want a resonance to excite: set ``zeta`` around 0.1 and the
    loop will happily show you what an over-aggressive D term does to a lightly
    damped airframe mode.
    """

    state_names = ("x", "xdot")

    def __init__(
        self,
        mass: float = 1.0,
        damping: float = 0.4,
        stiffness: float = 1.0,
        x0: float = 0.0,
        v0: float = 0.0,
    ) -> None:
        if mass <= 0.0:
            raise ValueError("mass must be > 0")
        self.mass = float(mass)
        self.damping = float(damping)
        self.stiffness = float(stiffness)
        super().__init__(np.array([x0, v0], dtype=float))

    @classmethod
    def from_modal(
        cls, wn: float, zeta: float, dc_gain: float = 1.0, **kwargs
    ) -> "SecondOrderPlant":
        """Build from natural frequency (rad/s), damping ratio and DC gain."""
        if wn <= 0.0:
            raise ValueError("wn must be > 0")
        k = 1.0 / dc_gain if dc_gain != 0.0 else 1.0
        m = k / (wn * wn)
        c = 2.0 * zeta * wn * m
        return cls(mass=m, damping=c, stiffness=k, **kwargs)

    @property
    def n_states(self) -> int:
        return 2

    def derivative(self, x, u, disturbance=0.0):
        accel = (u + disturbance - self.damping * x[1] - self.stiffness * x[0]) / self.mass
        return np.array([x[1], accel])


class PitchPlant(Plant):
    """1-DOF pitch axis: ``I theta'' = tau_cmd - b theta'' ... ``

    Specifically ``I * theta_ddot = u - damping * theta_dot``, with state
    ``[theta, theta_dot]`` in radians. There is no restoring term, which makes
    this a *double integrator with damping*: open-loop marginally stable at
    best, and unstable in attitude the instant you add any lag. That is exactly
    the character of a real multirotor axis, and it is why a rate loop with
    pure P will not hold attitude and why an attitude loop with too much lag
    oscillates at a frequency set by the loop, not by the airframe.

    Parameters
    ----------
    measure:
        ``'angle'`` returns theta from :meth:`output` (attitude loop);
        ``'rate'`` returns theta_dot (rate loop, i.e. what a gyro gives you).
        Having both lets the same plant be used for the inner and the outer
        loop without wrapping it.
    """

    state_names = ("theta", "theta_dot")

    def __init__(
        self,
        inertia: float = 0.01,
        damping: float = 0.002,
        theta0: float = 0.0,
        rate0: float = 0.0,
        measure: str = "angle",
    ) -> None:
        if inertia <= 0.0:
            raise ValueError("inertia must be > 0")
        if measure not in {"angle", "rate"}:
            raise ValueError("measure must be 'angle' or 'rate'")
        self.inertia = float(inertia)
        self.damping = float(damping)
        self.measure = measure
        super().__init__(np.array([theta0, rate0], dtype=float))

    @property
    def n_states(self) -> int:
        return 2

    def derivative(self, x, u, disturbance=0.0):
        alpha = (u + disturbance - self.damping * x[1]) / self.inertia
        return np.array([x[1], alpha])

    def output(self, x=None):
        xx = self.x if x is None else x
        return float(xx[1] if self.measure == "rate" else xx[0])

    def rate(self) -> float:
        """Body pitch rate in rad/s, i.e. what a gyro measures."""
        return float(self.x[1])


class PointMassQuadrotor2D(Plant):
    """Planar (x, z) point-mass quadrotor with quadratic drag.

    State: ``[x, z, vx, vz]``. Input is the pair ``(thrust, pitch)`` supplied
    through :meth:`set_input`, or a scalar thrust with the pitch held at the
    value last set. The vehicle is modelled as if the attitude loop is much
    faster than the translation loop, which is the standard cascade assumption
    and is true to within a factor of five on real hardware.

    Drag is ``-0.5 * rho * Cd * A * v * |v|`` collapsed into one coefficient
    per axis. It matters: without drag a point mass reaches implausible speeds
    and any velocity controller looks better than it is.
    """

    state_names = ("x", "z", "vx", "vz")

    def __init__(
        self,
        mass: float = 1.2,
        gravity: float = 9.81,
        drag_xy: float = 0.15,
        drag_z: float = 0.25,
        x0: Optional[Sequence[float]] = None,
    ) -> None:
        if mass <= 0.0:
            raise ValueError("mass must be > 0")
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.drag_xy = float(drag_xy)
        self.drag_z = float(drag_z)
        self.pitch = 0.0
        super().__init__(np.asarray(x0 if x0 is not None else [0.0, 0.0, 0.0, 0.0], dtype=float))

    @property
    def n_states(self) -> int:
        return 4

    def set_input(self, thrust: float, pitch: float) -> None:
        """Set the commanded pitch angle (rad); thrust is passed to :meth:`step`."""
        self.pitch = float(pitch)

    def derivative(self, x, u, disturbance=0.0):
        # u is total thrust in newtons; self.pitch tilts it.
        #
        # ``disturbance`` is a wind force in newtons. A scalar is taken as
        # horizontal, which is what :func:`simulate` supplies and is the common
        # case (a gust front). A 2-element sequence is ``(fx, fz)``, so a
        # downdraft -- the disturbance that actually threatens an altitude hold
        # -- can be injected as ``(0, -F)``.
        thrust = float(u)
        d = np.atleast_1d(np.asarray(disturbance, dtype=float))
        fx = float(d[0])
        fz = float(d[1]) if d.size > 1 else 0.0
        ax = (thrust * np.sin(self.pitch) + fx) / self.mass
        az = (thrust * np.cos(self.pitch) + fz) / self.mass - self.gravity
        ax -= self.drag_xy * x[2] * abs(x[2]) / self.mass
        az -= self.drag_z * x[3] * abs(x[3]) / self.mass
        return np.array([x[2], x[3], ax, az])

    def output(self, x=None):
        """Altitude, because altitude hold is the usual demo."""
        xx = self.x if x is None else x
        return float(xx[1])

    def hover_thrust(self) -> float:
        """Thrust that exactly cancels gravity. The natural feed-forward."""
        return self.mass * self.gravity


# ======================================================================
# Realism injectors
# ======================================================================
@dataclass
class SensorNoise:
    """Additive white noise plus a random-walk bias.

    Real gyros are not "noisy": they are *biased and slowly drifting* with
    white noise on top. A controller can tolerate a lot of white noise (the
    plant integrates it away) but a bias walks your integrator, and that is the
    part that shows up as slow drift in altitude hold or in a heading that
    creeps.

    Parameters
    ----------
    sigma:
        Standard deviation of the white component, in measurement units.
    bias:
        Initial bias.
    bias_walk:
        Standard deviation of the bias increment per second (random walk
        intensity). Set to 0 for a constant bias.
    seed:
        Seeds a private ``Generator`` so simulations are reproducible.
    """

    sigma: float = 0.0
    bias: float = 0.0
    bias_walk: float = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _bias: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bias = float(self.bias)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bias = float(self.bias)

    @property
    def current_bias(self) -> float:
        return self._bias

    def apply(self, value: float, dt: float) -> float:
        if self.bias_walk:
            self._bias += float(self._rng.normal(0.0, self.bias_walk * np.sqrt(dt)))
        noise = float(self._rng.normal(0.0, self.sigma)) if self.sigma else 0.0
        return value + self._bias + noise


@dataclass
class Quantiser:
    """Uniform quantiser with optional range clipping.

    Models an ADC on a sensor line or the integer command word going out to an
    ESC. The failure it reproduces: with a coarse quantiser the derivative term
    sees a staircase, so D output becomes a train of spikes at the sample rate
    even with a perfectly noise-free plant. If your D term is buzzing on the
    bench with the props off, check your resolution before you blame vibration.
    """

    step: float = 0.0
    lo: Optional[float] = None
    hi: Optional[float] = None

    def apply(self, value: float) -> float:
        v = float(value)
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        if self.step > 0.0:
            v = float(np.round(v / self.step) * self.step)
        return v


class ActuatorDelay:
    """Pure transport delay of ``n`` samples on the command path.

    This is the defect that kills tunes. Serial link latency, a companion
    computer's scheduling jitter, ESC protocol frame time and the FC's own
    mixer all add up to somewhere between 3 and 20 ms. Pure delay costs phase
    linearly with frequency (``phi = -omega * T``) while contributing no gain
    attenuation at all, so it eats phase margin without warning you in the gain
    plot. A loop tuned with the delay disabled and then flown with it will
    oscillate.
    """

    def __init__(self, delay_s: float, dt: float, initial: float = 0.0) -> None:
        if delay_s < 0.0:
            raise ValueError("delay_s must be >= 0")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.delay_s = float(delay_s)
        self.dt = float(dt)
        self.n = int(round(delay_s / dt))
        self.initial = float(initial)
        self._buf: deque[float] = deque([self.initial] * self.n, maxlen=max(1, self.n))

    def reset(self) -> None:
        self._buf = deque([self.initial] * self.n, maxlen=max(1, self.n))

    def apply(self, value: float) -> float:
        if self.n == 0:
            return float(value)
        out = self._buf[0]
        self._buf.append(float(value))
        return float(out)


class MotorLag:
    """First-order motor/ESC response: ``tau * u' + u = u_cmd``.

    A 5-inch quad motor+prop has a spin-up time constant in the 20-40 ms range;
    a large multirotor is well over 100 ms. This is a real pole in your loop.
    It is why rate-loop D gain has a ceiling that has nothing to do with your
    gyro: past a point the motor simply cannot follow the command, and the loop
    starts fighting the motor's own lag.
    """

    def __init__(self, tau: float, initial: float = 0.0) -> None:
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        self.tau = float(tau)
        self.initial = float(initial)
        self.value = float(initial)

    def reset(self) -> None:
        self.value = self.initial

    def apply(self, command: float, dt: float) -> float:
        if self.tau <= 0.0:
            self.value = float(command)
        else:
            alpha = 1.0 - float(np.exp(-dt / self.tau))
            self.value += alpha * (float(command) - self.value)
        return self.value


@dataclass
class WindGust:
    """Disturbance force: a step, a 1-cosine gust, or band-limited turbulence.

    ``kind='step'``      constant ``amplitude`` from ``t_start`` onwards.
    ``kind='gust'``      1-cosine profile of length ``duration`` (the classic
                         discrete gust shape from certification work).
    ``kind='turbulence'``first-order-filtered white noise with time constant
                         ``tau``, which is a serviceable stand-in for Dryden
                         without dragging in the full spectrum.
    """

    kind: str = "step"
    amplitude: float = 0.0
    t_start: float = 0.0
    duration: float = 1.0
    tau: float = 0.5
    seed: int = 1
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        if self.kind not in {"step", "gust", "turbulence"}:
            raise ValueError(f"unknown gust kind {self.kind!r}")
        self._rng = np.random.default_rng(self.seed)
        self._state = 0.0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state = 0.0

    def value(self, t: float, dt: float) -> float:
        if self.kind == "step":
            return self.amplitude if t >= self.t_start else 0.0
        if self.kind == "gust":
            if not (self.t_start <= t < self.t_start + self.duration):
                return 0.0
            phase = (t - self.t_start) / self.duration
            return 0.5 * self.amplitude * (1.0 - np.cos(2.0 * np.pi * phase))
        # turbulence
        alpha = 1.0 - float(np.exp(-dt / self.tau)) if self.tau > 0 else 1.0
        target = float(self._rng.normal(0.0, self.amplitude))
        self._state += alpha * (target - self._state)
        return self._state


# ======================================================================
# Harness
# ======================================================================
@dataclass
class SimResult:
    """Time histories produced by :func:`simulate`."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    y_meas: NDArray[np.float64]
    u: NDArray[np.float64]
    u_applied: NDArray[np.float64]
    setpoint: NDArray[np.float64]
    disturbance: NDArray[np.float64]
    states: NDArray[np.float64]

    def metrics(self, target: Optional[float] = None, **kwargs):
        """Convenience wrapper around :func:`dctk.metrics.step_metrics`."""
        from .metrics import step_metrics

        tgt = float(self.setpoint[-1]) if target is None else float(target)
        return step_metrics(self.t, self.y, tgt, **kwargs)


def simulate(
    plant: Plant,
    controller: Callable[[float, float, float], float],
    *,
    duration: float,
    dt: float,
    setpoint: Callable[[float], float] | float = 1.0,
    sensor_noise: Optional[SensorNoise] = None,
    sensor_quantiser: Optional[Quantiser] = None,
    actuator_delay: Optional[ActuatorDelay] = None,
    motor_lag: Optional[MotorLag] = None,
    actuator_quantiser: Optional[Quantiser] = None,
    disturbance: Optional[WindGust] = None,
    reset: bool = True,
) -> SimResult:
    """Close the loop on ``plant`` with ``controller`` through the given defects.

    ``controller`` is called as ``controller(setpoint, measurement, dt)`` and
    must return the raw command. The signal path is:

    ``plant output -> quantiser -> noise -> controller -> actuator quantiser ->
    transport delay -> motor lag -> plant``

    which is the physical order: you quantise at the ADC, add noise at the
    sensor, and the delay sits in front of the motor's own dynamics.

    Parameters
    ----------
    setpoint:
        Either a constant or a callable of time, so you can drive steps, ramps
        or chirps without writing a second harness.
    """
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if duration <= 0.0:
        raise ValueError("duration must be > 0")

    if reset:
        plant.reset()
        for obj in (sensor_noise, actuator_delay, motor_lag, disturbance):
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()

    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    sp_fn = setpoint if callable(setpoint) else (lambda _t, v=float(setpoint): v)

    y = np.zeros(n)
    y_meas = np.zeros(n)
    u = np.zeros(n)
    u_applied = np.zeros(n)
    sp = np.zeros(n)
    dist = np.zeros(n)
    states = np.zeros((n, plant.n_states))

    for i, ti in enumerate(t):
        y[i] = plant.output()
        states[i] = plant.x

        meas = y[i]
        if sensor_quantiser is not None:
            meas = sensor_quantiser.apply(meas)
        if sensor_noise is not None:
            meas = sensor_noise.apply(meas, dt)
        y_meas[i] = meas

        sp[i] = sp_fn(ti)
        cmd = float(controller(sp[i], meas, dt))
        u[i] = cmd

        applied = cmd
        if actuator_quantiser is not None:
            applied = actuator_quantiser.apply(applied)
        if actuator_delay is not None:
            applied = actuator_delay.apply(applied)
        if motor_lag is not None:
            applied = motor_lag.apply(applied, dt)
        u_applied[i] = applied

        dist[i] = disturbance.value(ti, dt) if disturbance is not None else 0.0

        if i < n - 1:
            plant.step(applied, dt, dist[i])

    return SimResult(
        t=t,
        y=y,
        y_meas=y_meas,
        u=u,
        u_applied=u_applied,
        setpoint=sp,
        disturbance=dist,
        states=states,
    )
