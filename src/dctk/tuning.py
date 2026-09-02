"""Automatic tuning: Ziegler-Nichols and relay (Astrom-Hagglund) auto-tuning.

Read this before you use either of them
---------------------------------------
Ziegler-Nichols was derived in 1942 for process control loops -- temperature,
level, flow -- where a 25 % overshoot and a quarter-amplitude decay were
perfectly acceptable and the plant took minutes to respond. Applied to an
aircraft it produces a tune that is roughly **twice as aggressive as anything
you would want to fly**. Specifically:

* The classic ZN PID rule targets a gain margin around 2 and a phase margin
  near 30 degrees. For a multirotor rate loop you want a phase margin of
  45-60 degrees, because you have unmodelled dynamics -- motor lag, ESC
  latency, prop flex, filter phase -- that ZN's open-loop test does not see.
* ZN assumes a plant that is well described by a first-order-plus-dead-time
  model. A multirotor rate axis is a double integrator with lag. The rule does
  not apply, it just happens to produce numbers.
* The ultimate-gain test itself requires driving the loop into sustained
  oscillation. On a bench with props off that is fine. On a flying aircraft it
  is how you break props, and the oscillation amplitude is not something you
  can control once you are at the stability boundary.

So: **treat ZN output as a starting point in the right order of magnitude,
then halve the P gain and re-tune by hand.** It is genuinely useful for that.
It is not a tuning method for aircraft.

Relay auto-tuning is the better of the two here. It excites the plant at its
ultimate frequency using a bounded relay rather than by pushing gain until it
oscillates, so the oscillation amplitude is set by the relay amplitude you
chose and stays bounded. That makes it safe enough to run on a test stand. It
still gives you ZN-family numbers at the end, so the same "halve it" advice
applies.

Both functions here are offline/simulation tools. Running an auto-tuner on a
flying aircraft needs an abort path, an amplitude limit, and a human on the
switch, none of which belongs in a library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "UltimateGain",
    "ZNGains",
    "ziegler_nichols",
    "relay_autotune",
    "find_ultimate_gain",
]


@dataclass
class UltimateGain:
    """Ultimate gain ``Ku`` and ultimate period ``Tu`` (seconds)."""

    ku: float
    tu: float
    amplitude: float = 0.0
    n_cycles: int = 0
    converged: bool = True

    @property
    def ultimate_frequency_hz(self) -> float:
        return 1.0 / self.tu if self.tu > 0 else float("nan")


@dataclass
class ZNGains:
    """Gains from a Ziegler-Nichols rule, plus the safety derating."""

    kp: float
    ki: float
    kd: float
    rule: str
    ku: float
    tu: float

    def derated(self, factor: float = 0.5) -> "ZNGains":
        """Scale P and D by ``factor`` and I by ``factor**2``.

        You are buying phase margin with bandwidth: the derated loop is
        slower, and in exchange its response barely changes when you add the
        unmodelled lag that the ultimate-gain test never saw.

        Why these exponents: reducing loop gain by ``f`` scales ``kp`` and
        ``kd`` by ``f`` directly. The integral gain in the parallel form is
        ``kp / Ti``, and a slower loop also wants a proportionally longer
        integral time, so ``ki`` picks up the factor twice. Applying ``f`` to
        all three equally leaves the loop relatively *more* integral-heavy than
        the original tune, which is exactly the direction that produces the
        slow limit cycle people complain about after "just turning the gains
        down".
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError("factor must be in (0, 1]")
        return ZNGains(
            kp=self.kp * factor,
            ki=self.ki * factor * factor,
            kd=self.kd * factor,
            rule=f"{self.rule} x{factor:g}",
            ku=self.ku,
            tu=self.tu,
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.kp, self.ki, self.kd)


# Classic ZN table, expressed as (kp_factor, Ti_factor, Td_factor).
_ZN_RULES: dict[str, tuple[float, Optional[float], Optional[float]]] = {
    # rule            Kp        Ti          Td
    "P": (0.50, None, None),
    "PI": (0.45, 1.0 / 1.2, None),
    "PD": (0.80, None, 0.125),
    "classic": (0.60, 0.5, 0.125),
    # Less aggressive variants from the same family. "no_overshoot" is the one
    # to start from on an aircraft; it is roughly a third of classic P gain and
    # is still usually too hot for a rate loop.
    "pessen": (0.70, 0.4, 0.15),
    "some_overshoot": (1.0 / 3.0, 0.5, 1.0 / 3.0),
    "no_overshoot": (0.20, 0.5, 1.0 / 3.0),
}


def ziegler_nichols(ku: float, tu: float, rule: str = "classic") -> ZNGains:
    """Apply a Ziegler-Nichols rule to an ultimate gain/period pair.

    Parameters
    ----------
    ku, tu:
        From :func:`find_ultimate_gain` or :func:`relay_autotune`.
    rule:
        One of ``P``, ``PI``, ``PD``, ``classic``, ``pessen``,
        ``some_overshoot``, ``no_overshoot``.

    Returns
    -------
    ZNGains
        In *parallel* form: ``kp``, ``ki = kp / Ti``, ``kd = kp * Td``. Note the
        conversion, because the ZN table is stated in terms of integral and
        derivative *times* and it is very easy to plug ``Ti`` into a ``ki``
        slot and wonder why the aircraft is unflyable.
    """
    if ku <= 0.0 or tu <= 0.0:
        raise ValueError("ku and tu must be > 0")
    if rule not in _ZN_RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {sorted(_ZN_RULES)}")
    kp_f, ti_f, td_f = _ZN_RULES[rule]
    kp = kp_f * ku
    ki = kp / (ti_f * tu) if ti_f else 0.0
    kd = kp * (td_f * tu) if td_f else 0.0
    return ZNGains(kp=kp, ki=ki, kd=kd, rule=rule, ku=ku, tu=tu)


def find_ultimate_gain(
    closed_loop: Callable[[float], NDArray[np.float64]],
    *,
    dt: float,
    kp_min: float = 1e-3,
    kp_max: float = 1e4,
    iterations: int = 40,
    oscillation_threshold: float = 0.02,
) -> UltimateGain:
    """Bisect for the proportional gain that produces sustained oscillation.

    ``closed_loop(kp)`` must run a fixed-length simulation with proportional
    gain ``kp`` and return the output time series, uniformly sampled at ``dt``
    seconds. The growth of the envelope over the second half of the record
    decides whether the loop is stable at that gain.

    This is the *simulation* version of the ZN test. Doing the same thing on
    hardware means deliberately destabilising the aircraft, which is why the
    relay method below exists.

    ``oscillation_threshold`` is the relative envelope growth above which the
    response counts as unstable.

    Important: ``closed_loop`` must **not** saturate the actuator. A saturated
    loop limit-cycles at a bounded amplitude for every gain above the stability
    boundary, so the envelope stops growing and this test reports "stable" all
    the way to infinity. Leave the output limits off for the sweep and put them
    back for the verification run.
    """
    if kp_min <= 0.0 or kp_max <= kp_min:
        raise ValueError("require 0 < kp_min < kp_max")

    def unstable(kp: float) -> tuple[bool, float]:
        # A gain far past the stability boundary can overflow to inf/nan, or
        # make the controller itself reject a non-finite measurement. Both mean
        # "unstable", not "broken test".
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                y = np.asarray(closed_loop(kp), dtype=float).ravel()
        except (ValueError, FloatingPointError, OverflowError):
            return True, float("inf")
        if not np.all(np.isfinite(y)):
            return True, float("inf")
        half = y.size // 2
        window = slice(half, half + max(1, half // 2))
        first = float(np.max(y[window]) - np.min(y[window]))
        last = float(np.max(y[-max(1, half // 2) :]) - np.min(y[-max(1, half // 2) :]))
        if first < 1e-12:
            return False, 0.0
        growth = (last - first) / first
        return growth > oscillation_threshold, last

    lo, hi = kp_min, kp_max
    if not unstable(hi)[0]:
        return UltimateGain(ku=hi, tu=float("nan"), converged=False)
    if unstable(lo)[0]:
        return UltimateGain(ku=lo, tu=float("nan"), converged=False)

    for _ in range(iterations):
        mid = float(np.sqrt(lo * hi))  # geometric bisection: gain is a ratio
        if unstable(mid)[0]:
            hi = mid
        else:
            lo = mid

    ku = float(np.sqrt(lo * hi))
    y = np.asarray(closed_loop(ku), dtype=float).ravel()
    return UltimateGain(
        ku=ku, tu=_estimate_period(y, dt), amplitude=float(np.ptp(y)), converged=True
    )


def _estimate_period(y: NDArray[np.float64], dt: float = 1.0) -> float:
    """Period from mean-crossings of the second half of a record."""
    seg = y[y.size // 2 :]
    seg = seg - np.mean(seg)
    sign = np.sign(seg)
    crossings = np.flatnonzero((sign[:-1] < 0) & (sign[1:] >= 0))
    if crossings.size < 2:
        return float("nan")
    return float(np.mean(np.diff(crossings)) * dt)


def relay_autotune(
    plant_step: Callable[[float, float], float],
    *,
    setpoint: float = 0.0,
    relay_amplitude: float = 1.0,
    hysteresis: float = 0.0,
    dt: float = 0.001,
    duration: float = 20.0,
    settle_fraction: float = 0.5,
) -> UltimateGain:
    """Astrom-Hagglund relay feedback test.

    Replace the controller with a relay: output ``+d`` when the error is
    positive, ``-d`` when negative. The loop limit-cycles at the frequency
    where the plant's phase is -180 degrees -- exactly the ultimate frequency
    -- and the describing-function approximation of the relay gives

    ``Ku = 4 d / (pi a)``

    where ``a`` is the amplitude of the resulting output oscillation and ``d``
    the relay amplitude. ``Tu`` is read off the limit-cycle period directly.

    Parameters
    ----------
    plant_step:
        ``plant_step(u, dt) -> y``. Advance the plant one step and return the
        measurement. Any callable with state works, including a closure over a
        :class:`dctk.sim.Plant`.
    relay_amplitude:
        ``d``. This bounds the excitation. Pick the smallest value that gives a
        clean limit cycle; that is the entire safety argument for this method
        over the ZN gain sweep.
    hysteresis:
        Deadband on the relay switching. Set it to ~2x the noise amplitude on
        the measurement, otherwise sensor noise chatters the relay at the
        sample rate and you measure your own noise instead of the plant. The
        describing function with hysteresis is
        ``Ku = 4 d / (pi sqrt(a^2 - h^2))``, which this function uses when
        ``hysteresis > 0``.
    settle_fraction:
        Fraction of the record discarded before measuring, so the initial
        transient does not contaminate the amplitude and period estimates.

    Returns
    -------
    UltimateGain
        With ``converged=False`` if fewer than two full cycles were observed,
        which usually means ``duration`` is too short or the relay amplitude is
        too small to overcome friction/deadband in the plant.
    """
    if relay_amplitude <= 0.0:
        raise ValueError("relay_amplitude must be > 0")
    if dt <= 0.0 or duration <= 0.0:
        raise ValueError("dt and duration must be > 0")
    if hysteresis < 0.0:
        raise ValueError("hysteresis must be >= 0")
    if not 0.0 <= settle_fraction < 1.0:
        raise ValueError("settle_fraction must be in [0, 1)")

    n = int(round(duration / dt))
    y_hist = np.zeros(n)
    t_hist = np.arange(n) * dt
    u = relay_amplitude
    y = 0.0
    for i in range(n):
        error = setpoint - y
        if error > hysteresis:
            u = relay_amplitude
        elif error < -hysteresis:
            u = -relay_amplitude
        # else: hold the previous relay state (this is the hysteresis)
        y = float(plant_step(u, dt))
        y_hist[i] = y

    start = int(settle_fraction * n)
    seg = y_hist[start:]
    t_seg = t_hist[start:]
    if seg.size < 4:
        return UltimateGain(ku=float("nan"), tu=float("nan"), converged=False)

    centred = seg - np.mean(seg)
    amplitude = 0.5 * float(np.ptp(seg))

    sign = np.sign(centred)
    up = np.flatnonzero((sign[:-1] <= 0) & (sign[1:] > 0))
    if up.size < 2:
        return UltimateGain(ku=float("nan"), tu=float("nan"), amplitude=amplitude, converged=False)
    # Linear interpolation of each crossing, then average the spacing.
    times = []
    for i in up:
        y0, y1 = centred[i], centred[i + 1]
        frac = 0.0 if y1 == y0 else -y0 / (y1 - y0)
        times.append(t_seg[i] + frac * dt)
    tu = float(np.mean(np.diff(times)))

    if amplitude <= hysteresis:
        return UltimateGain(ku=float("nan"), tu=tu, amplitude=amplitude, converged=False)
    denom = np.sqrt(amplitude**2 - hysteresis**2) if hysteresis > 0 else amplitude
    ku = 4.0 * relay_amplitude / (np.pi * denom)
    return UltimateGain(
        ku=float(ku), tu=tu, amplitude=amplitude, n_cycles=int(up.size - 1), converged=True
    )
