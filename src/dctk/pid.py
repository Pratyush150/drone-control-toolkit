"""Production-grade PID controller.

The textbook PID is three lines of algebra. The thing that actually flies an
aircraft is mostly the code *around* those three lines: what happens when the
actuator saturates, what happens when the setpoint jumps, what happens when the
IMU feeding the D term is sitting on a vibrating airframe, and what happens the
moment a human flips from manual to auto.

Every feature in this module exists because of a specific failure mode. Each is
documented with the failure it prevents, so you can decide which ones you need
rather than copying the whole thing blindly.

Conventions
-----------
* Positive error means "measurement is below setpoint".
* ``dt`` is seconds and is supplied per call; nothing here assumes a fixed loop
  rate, because real loops jitter and occasionally drop a frame.
* All state is plain floats. No globals, no hidden singletons. One controller
  object per loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

__all__ = ["AntiWindup", "PIDGains", "PIDState", "PID"]


class AntiWindup(str, Enum):
    """Selectable integral anti-windup strategy.

    ``NONE``
        No protection. Included only so tests can demonstrate the failure.
    ``CLAMP``
        Conditional integration: stop accumulating error whenever the output is
        already saturated *and* the error would push it further into
        saturation. Cheap, no extra tuning parameter, slightly conservative
        because it also freezes the integrator during transients that would
        have recovered on their own.
    ``BACK_CALCULATION``
        Feed the saturation excess ``(u_sat - u_unsat)`` back into the
        integrator through a gain ``1 / Tt``. The integrator unwinds smoothly
        and proportionally to how badly you are saturated, which gives a softer
        recovery than clamping. Needs one extra constant, the tracking time
        constant ``Tt``; a common starting point is ``Tt = sqrt(Ti * Td)`` or,
        for a PI loop, ``Tt = Ti``.
    """

    NONE = "none"
    CLAMP = "clamp"
    BACK_CALCULATION = "back_calculation"


@dataclass
class PIDGains:
    """Gain set for :class:`PID`.

    Attributes
    ----------
    kp, ki, kd:
        Parallel-form gains. ``ki`` is in output-units per (error-unit *
        second); ``kd`` is output-units per (error-unit / second). Parallel form
        is used rather than the "standard" ``Kp(1 + 1/Ti s + Td s)`` form
        because it is what every flight stack exposes and because it lets you
        set ``ki = 0`` without a division by infinity.
    kff:
        Feed-forward gain applied to the *setpoint* (or to an explicit
        feed-forward signal). See :meth:`PID.update`.
    """

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    kff: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.kp, self.ki, self.kd, self.kff)


@dataclass
class PIDState:
    """Introspectable controller state, exposed for logging and tests."""

    integral: float = 0.0
    derivative: float = 0.0
    last_measurement: float = float("nan")
    last_output: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    ff_term: float = 0.0
    saturated: bool = False
    unsaturated_output: float = 0.0


class PID:
    """A PID controller that survives contact with hardware.

    Parameters
    ----------
    kp, ki, kd, kff:
        Initial gains. See :class:`PIDGains`.
    output_limits:
        ``(low, high)`` saturation applied to the final output. Use ``None`` on
        either side for unbounded. Saturation is not optional in practice: a
        motor cannot produce 130 % thrust, and a controller that pretends
        otherwise will integrate against a wall.
    anti_windup:
        Strategy from :class:`AntiWindup`.
    tracking_time_constant:
        ``Tt`` for :attr:`AntiWindup.BACK_CALCULATION`, in seconds. Smaller
        means more aggressive unwinding. Must be > 0.
    derivative_on_measurement:
        If ``True`` (default) the derivative is computed as
        ``-d(measurement)/dt`` instead of ``d(error)/dt``.

        *Failure prevented:* derivative kick. A step change in setpoint makes
        ``d(error)/dt`` an impulse; multiplied by ``kd`` it slams the actuator
        to a limit for one sample. On a multirotor that is an audible motor
        snap and a visible twitch on every stick input. Differentiating the
        measurement instead gives an identical response to disturbances and
        plant motion, with zero response to a setpoint step.
    derivative_cutoff_hz:
        Cutoff of the first-order low-pass filter on the derivative term.
        ``None`` disables filtering (do not do this on real hardware).

        *Failure prevented:* amplified sensor noise. Differentiation multiplies
        a signal's amplitude by its frequency. A gyro on a multirotor carries
        prop-blade-pass content at 100-400 Hz that is small in rate units and
        enormous after differentiation, so raw D on a real IMU is not "a bit
        noisy", it is unusable: the D output is pure noise, the motors heat up,
        and you cannot raise ``kd`` far enough to do anything useful before the
        aircraft starts buzzing. A cutoff between 20 and 60 Hz is typical for a
        rate loop; every Hz you remove also removes phase margin, so this is a
        direct trade (see ``docs/CONTROL_NOTES.md``).
    setpoint_weight_p:
        Proportional setpoint weight ``b`` in ``kp * (b * sp - meas)``. ``b=1``
        is classic PID; ``b<1`` softens the response to setpoint steps without
        changing disturbance rejection. Useful when you want a gentler feel
        without detuning the loop.
    name:
        Free-form label used in ``repr`` and logs.

    Notes
    -----
    Bumpless transfer is handled two ways, both automatic:

    * **On gain change** (:meth:`set_gains`): the integrator is rebalanced so
      the total output is unchanged at the instant of the change. Without this,
      changing ``ki`` from 0.1 to 0.2 with a stored integral of 40 doubles the
      I contribution instantly and the aircraft lurches. This is exactly what
      happens if you tune gains in flight over telemetry.
    * **On manual -> auto handover** (:meth:`set_manual` / :meth:`set_auto`):
      while in manual the integrator is continuously back-solved so that the
      PID output would equal the manual output. When you hand control over, the
      first automatic output equals the last manual output, and the loop takes
      over without a step.
    """

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        kff: float = 0.0,
        *,
        output_limits: tuple[Optional[float], Optional[float]] = (None, None),
        anti_windup: AntiWindup = AntiWindup.BACK_CALCULATION,
        tracking_time_constant: float = 1.0,
        derivative_on_measurement: bool = True,
        derivative_cutoff_hz: Optional[float] = 30.0,
        setpoint_weight_p: float = 1.0,
        integral_limits: tuple[Optional[float], Optional[float]] = (None, None),
        name: str = "pid",
    ) -> None:
        if tracking_time_constant <= 0.0:
            raise ValueError("tracking_time_constant must be > 0")
        if derivative_cutoff_hz is not None and derivative_cutoff_hz <= 0.0:
            raise ValueError("derivative_cutoff_hz must be > 0 or None")
        lo, hi = output_limits
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("output_limits must satisfy low < high")

        self.gains = PIDGains(kp, ki, kd, kff)
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.anti_windup = AntiWindup(anti_windup)
        self.tracking_time_constant = float(tracking_time_constant)
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.derivative_cutoff_hz = derivative_cutoff_hz
        self.setpoint_weight_p = float(setpoint_weight_p)
        self.name = name

        self.state = PIDState()
        self._manual = False
        self._manual_output = 0.0
        self._last_error = float("nan")

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @property
    def kp(self) -> float:
        return self.gains.kp

    @property
    def ki(self) -> float:
        return self.gains.ki

    @property
    def kd(self) -> float:
        return self.gains.kd

    def set_gains(
        self,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
        kff: Optional[float] = None,
        *,
        bumpless: bool = True,
    ) -> None:
        """Change gains, optionally rebalancing the integrator (bumpless).

        With ``bumpless=True`` the stored integral is rescaled so that the I
        contribution ``ki * integral`` is preserved across the change. If the
        new ``ki`` is zero the accumulated action cannot be preserved, so the
        integral is simply zeroed (there is no term left to carry it).

        *Failure prevented:* the output step you get when tuning gains live.
        """
        old_ki = self.gains.ki
        if kp is not None:
            self.gains.kp = float(kp)
        if ki is not None:
            self.gains.ki = float(ki)
        if kd is not None:
            self.gains.kd = float(kd)
        if kff is not None:
            self.gains.kff = float(kff)

        if bumpless and ki is not None:
            i_contribution = old_ki * self.state.integral
            if self.gains.ki != 0.0:
                self.state.integral = i_contribution / self.gains.ki
            else:
                self.state.integral = 0.0

    def set_manual(self, manual_output: float) -> None:
        """Enter manual mode and track ``manual_output``.

        While manual, :meth:`update` returns ``manual_output`` and back-solves
        the integrator so the automatic output would match it.
        """
        self._manual = True
        self._manual_output = float(manual_output)

    def set_auto(self) -> None:
        """Return to automatic control. The handover is bumpless by
        construction because the integrator was tracked during manual mode."""
        self._manual = False

    @property
    def manual(self) -> bool:
        return self._manual

    def reset(self, *, keep_gains: bool = True) -> None:
        """Clear all internal state.

        Call this whenever the loop has been disengaged for more than a couple
        of samples: after an arm/disarm cycle, a mode change that bypasses the
        loop, or a sensor dropout. A stale integrator and a stale
        ``last_measurement`` are the two classic causes of "it kicked on the
        first sample after re-enable".
        """
        self.state = PIDState()
        self._last_error = float("nan")
        self._manual = False
        self._manual_output = 0.0
        if not keep_gains:
            self.gains = PIDGains()

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
        *,
        feedforward: Optional[float] = None,
    ) -> float:
        """Advance the controller one step and return the saturated output.

        Parameters
        ----------
        setpoint, measurement:
            In the same engineering units.
        dt:
            Elapsed time since the previous call, in seconds. Must be > 0.

            *Failure prevented:* variable-rate loops. Scheduler jitter, a
            blocking log write, or a USB hiccup will hand you a ``dt`` two or
            three times nominal. Integrating and differentiating with a
            hard-coded nominal ``dt`` silently mis-scales I and D exactly when
            the loop is already struggling. A non-positive ``dt`` (duplicate
            timestamp, clock step) is rejected rather than producing a division
            by zero.
        feedforward:
            Explicit feed-forward signal. If ``None`` the setpoint is used.
            The term added is ``kff * ff``.

            *Failure prevented:* lag on fast setpoint tracking. Feedback only
            acts after an error exists. On a rate loop tracking stick input, or
            a velocity loop tracking a trajectory, feed-forward supplies most
            of the actuator command immediately and leaves the feedback terms
            to clean up the residual, which lets you run lower gains for the
            same tracking performance.

        Returns
        -------
        float
            The command after saturation.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")
        if not (np.isfinite(setpoint) and np.isfinite(measurement)):
            raise ValueError("setpoint and measurement must be finite")

        g = self.gains
        error = setpoint - measurement

        # --- proportional (with setpoint weighting) -----------------------
        p_term = g.kp * (self.setpoint_weight_p * setpoint - measurement)

        # --- derivative ---------------------------------------------------
        if np.isnan(self.state.last_measurement):
            # First sample after construction or reset: no valid history, so
            # the derivative is defined as zero. Prevents a huge first-sample
            # D spike from differencing against garbage.
            raw_derivative = 0.0
        elif self.derivative_on_measurement:
            raw_derivative = -(measurement - self.state.last_measurement) / dt
        else:
            prev_error = self._last_error
            raw_derivative = 0.0 if np.isnan(prev_error) else (error - prev_error) / dt

        self.state.derivative = self._filter_derivative(raw_derivative, dt)
        d_term = g.kd * self.state.derivative

        # --- feed-forward --------------------------------------------------
        ff = setpoint if feedforward is None else feedforward
        ff_term = g.kff * ff

        # --- integral ------------------------------------------------------
        # Provisional integration; may be undone below by the anti-windup
        # strategy. Trapezoidal would be marginally more accurate but forward
        # Euler is what every flight stack ships and it keeps the clamp logic
        # exact.
        candidate_integral = self.state.integral + error * dt
        candidate_integral = self._clamp_integral(candidate_integral)

        i_term = g.ki * candidate_integral
        unsat = p_term + i_term + d_term + ff_term
        out = self._saturate(unsat)
        saturated = out != unsat

        if self.anti_windup is AntiWindup.CLAMP:
            # Conditional integration: only accept the new integral if we are
            # not saturated, or if the error is trying to pull us back out of
            # saturation.
            pushing_further = saturated and (np.sign(error) == np.sign(unsat - out))
            if not pushing_further:
                self.state.integral = candidate_integral
            # else: integral is frozen at its previous value.
            i_term = g.ki * self.state.integral
            unsat = p_term + i_term + d_term + ff_term
            out = self._saturate(unsat)
            saturated = out != unsat
        elif self.anti_windup is AntiWindup.BACK_CALCULATION:
            # Standard back-calculation: the excess (out - unsat) is fed back
            # into the integrator scaled by dt / Tt. When not saturated the
            # correction is exactly zero, so nominal behaviour is untouched.
            self.state.integral = candidate_integral + (dt / self.tracking_time_constant) * (
                out - unsat
            ) / (g.ki if g.ki != 0.0 else 1.0)
            self.state.integral = self._clamp_integral(self.state.integral)
        else:  # AntiWindup.NONE
            self.state.integral = candidate_integral

        # --- manual mode / bumpless tracking --------------------------------
        if self._manual:
            out = self._saturate(self._manual_output)
            if g.ki != 0.0:
                # Back-solve the integral so that auto output == manual output.
                self.state.integral = (out - p_term - d_term - ff_term) / g.ki
                self.state.integral = self._clamp_integral(self.state.integral)
            i_term = g.ki * self.state.integral
            unsat = out
            saturated = False

        # --- bookkeeping -----------------------------------------------------
        self.state.p_term = p_term
        self.state.i_term = i_term
        self.state.d_term = d_term
        self.state.ff_term = ff_term
        self.state.unsaturated_output = unsat
        self.state.saturated = saturated
        self.state.last_measurement = measurement
        self.state.last_output = out
        self._last_error = error
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _filter_derivative(self, raw: float, dt: float) -> float:
        """First-order low-pass on the derivative term.

        Discretised as an exponential-decay IIR, ``alpha = 1 - exp(-dt/tau)``,
        rather than the more common ``dt / (tau + dt)``. The exponential form
        stays correct (and stable) when ``dt`` occasionally spikes to several
        times ``tau``, which the bilinear/Euler form does not.
        """
        if self.derivative_cutoff_hz is None:
            return raw
        tau = 1.0 / (2.0 * np.pi * self.derivative_cutoff_hz)
        alpha = 1.0 - float(np.exp(-dt / tau))
        return self.state.derivative + alpha * (raw - self.state.derivative)

    def _saturate(self, value: float) -> float:
        lo, hi = self.output_limits
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def _clamp_integral(self, value: float) -> float:
        lo, hi = self.integral_limits
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        g = self.gains
        return (
            f"PID(name={self.name!r}, kp={g.kp}, ki={g.ki}, kd={g.kd}, kff={g.kff}, "
            f"limits={self.output_limits}, anti_windup={self.anti_windup.value})"
        )
