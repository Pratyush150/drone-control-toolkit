"""Step-response metrics.

Claims about a controller are worth nothing without numbers attached. These are
the numbers: rise time, settling time, overshoot, steady-state error, and the
integral error criteria. Tests and examples both use this module so the plots
and the assertions agree.

All functions take a uniformly-or-non-uniformly sampled ``(t, y)`` pair and a
target value. Non-uniform time vectors are handled by trapezoidal integration,
because real logs are not uniformly sampled.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

# numpy 2.0 renamed ``trapz`` to ``trapezoid``. Support both so the package
# works on the numpy that ships with a given Jetson/ROS image without forcing an
# upgrade of the whole stack.
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "StepMetrics",
    "rise_time",
    "settling_time",
    "overshoot_percent",
    "steady_state_error",
    "peak_time",
    "iae",
    "ise",
    "itae",
    "step_metrics",
]


def _as_pair(t: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t_arr = np.asarray(t, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if t_arr.shape != y_arr.shape:
        raise ValueError(f"t and y must have the same shape, got {t_arr.shape} and {y_arr.shape}")
    if t_arr.size < 2:
        raise ValueError("need at least two samples")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("t must be strictly increasing")
    return t_arr, y_arr


def _crossing_time(
    t: NDArray[np.float64], y: NDArray[np.float64], level: float, rising: bool
) -> Optional[float]:
    """First time ``y`` crosses ``level``, linearly interpolated.

    Interpolation matters: at 50 Hz logging, reporting the sample time instead
    of the crossing time quantises rise time to 20 ms, which is the same order
    as the quantity being measured on a fast loop.
    """
    if rising:
        idx = np.flatnonzero(y >= level)
    else:
        idx = np.flatnonzero(y <= level)
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(t[i])
    frac = (level - y0) / (y1 - y0)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def rise_time(
    t: ArrayLike,
    y: ArrayLike,
    target: float,
    *,
    low: float = 0.1,
    high: float = 0.9,
    y0: Optional[float] = None,
) -> float:
    """Time to go from ``low`` to ``high`` fraction of the step size.

    Defaults are the 10-90 % convention. Returns ``nan`` if the response never
    reaches the high threshold (an overdamped loop that has not settled inside
    the window, or an unstable one heading the other way).
    """
    t_arr, y_arr = _as_pair(t, y)
    start = y_arr[0] if y0 is None else float(y0)
    span = target - start
    if span == 0.0:
        return 0.0
    rising = span > 0.0
    t_low = _crossing_time(t_arr, y_arr, start + low * span, rising)
    t_high = _crossing_time(t_arr, y_arr, start + high * span, rising)
    if t_low is None or t_high is None:
        return float("nan")
    return float(t_high - t_low)


def settling_time(
    t: ArrayLike,
    y: ArrayLike,
    target: float,
    *,
    tolerance: float = 0.02,
    y0: Optional[float] = None,
) -> float:
    """Time after which ``y`` stays within ``tolerance`` of ``target`` forever.

    ``tolerance`` is a fraction of the step size (2 % by default), not of the
    target, so a step from 5 to 6 is judged on the 1-unit change and not on the
    absolute value of 6. Returns ``nan`` if the response leaves the band before
    the end of the record: that is the honest answer, and it is what you want a
    test to catch.
    """
    t_arr, y_arr = _as_pair(t, y)
    start = y_arr[0] if y0 is None else float(y0)
    span = abs(target - start)
    band = tolerance * span if span > 0.0 else tolerance
    outside = np.flatnonzero(np.abs(y_arr - target) > band)
    if outside.size == 0:
        return float(t_arr[0])
    last = int(outside[-1])
    if last >= t_arr.size - 1:
        return float("nan")
    return float(t_arr[last + 1])


def overshoot_percent(y: ArrayLike, target: float, *, y0: Optional[float] = None) -> float:
    """Peak excursion beyond ``target``, as a percentage of the step size.

    Zero if the response never crosses the target. Sign-aware, so a downward
    step reports overshoot for undershooting below the target.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("empty response")
    start = y_arr[0] if y0 is None else float(y0)
    span = target - start
    if span == 0.0:
        return 0.0
    if span > 0.0:
        peak = float(np.max(y_arr))
        excess = peak - target
    else:
        peak = float(np.min(y_arr))
        excess = target - peak
    return max(0.0, 100.0 * excess / abs(span))


def peak_time(t: ArrayLike, y: ArrayLike, target: float, *, y0: Optional[float] = None) -> float:
    """Time of the extremum in the direction of the step."""
    t_arr, y_arr = _as_pair(t, y)
    start = y_arr[0] if y0 is None else float(y0)
    idx = int(np.argmax(y_arr)) if target >= start else int(np.argmin(y_arr))
    return float(t_arr[idx])


def steady_state_error(y: ArrayLike, target: float, *, fraction: float = 0.1) -> float:
    """Mean error over the final ``fraction`` of the record.

    Averaging instead of taking the last sample keeps the number meaningful in
    the presence of measurement noise or a small limit cycle.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    n = max(1, int(round(fraction * y_arr.size)))
    return float(target - np.mean(y_arr[-n:]))


def iae(t: ArrayLike, y: ArrayLike, target: float) -> float:
    """Integral of Absolute Error. Penalises all error equally."""
    t_arr, y_arr = _as_pair(t, y)
    return float(_trapz(np.abs(target - y_arr), t_arr))


def ise(t: ArrayLike, y: ArrayLike, target: float) -> float:
    """Integral of Squared Error. Punishes large excursions hardest, so it
    favours fast, twitchy tuning."""
    t_arr, y_arr = _as_pair(t, y)
    return float(_trapz((target - y_arr) ** 2, t_arr))


def itae(t: ArrayLike, y: ArrayLike, target: float, *, t0: Optional[float] = None) -> float:
    """Integral of Time-weighted Absolute Error.

    Late error costs more than early error, so ITAE rewards a response that
    actually settles rather than one that gets close fast and then rings. It is
    the criterion that usually agrees with a pilot's opinion of a tune.
    """
    t_arr, y_arr = _as_pair(t, y)
    origin = t_arr[0] if t0 is None else float(t0)
    return float(_trapz((t_arr - origin) * np.abs(target - y_arr), t_arr))


@dataclass
class StepMetrics:
    """Bundle returned by :func:`step_metrics`."""

    rise_time: float
    settling_time: float
    overshoot_pct: float
    peak_time: float
    steady_state_error: float
    iae: float
    ise: float
    itae: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def summary(self, label: str = "") -> str:
        """One-line human-readable summary, used by the example scripts."""
        head = f"{label:<28}" if label else ""
        return (
            f"{head}rise={self.rise_time:6.3f}s  settle={self.settling_time:6.3f}s  "
            f"OS={self.overshoot_pct:5.1f}%  ess={self.steady_state_error:+.4f}  "
            f"IAE={self.iae:8.4f}  ITAE={self.itae:9.4f}"
        )


def step_metrics(
    t: ArrayLike,
    y: ArrayLike,
    target: float,
    *,
    tolerance: float = 0.02,
    y0: Optional[float] = None,
) -> StepMetrics:
    """Compute the full metric set for one step response."""
    return StepMetrics(
        rise_time=rise_time(t, y, target, y0=y0),
        settling_time=settling_time(t, y, target, tolerance=tolerance, y0=y0),
        overshoot_pct=overshoot_percent(y, target, y0=y0),
        peak_time=peak_time(t, y, target, y0=y0),
        steady_state_error=steady_state_error(y, target),
        iae=iae(t, y, target),
        ise=ise(t, y, target),
        itae=itae(t, y, target),
    )
