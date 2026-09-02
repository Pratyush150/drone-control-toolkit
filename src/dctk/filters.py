"""Filters for attitude estimation and for cleaning up a vibrating airframe.

Two different jobs live in this file and it is worth keeping them apart:

* **Sensor fusion** -- complementary filter, Madgwick, Kalman. These combine
  sensors with complementary error characteristics (a gyro that is smooth but
  drifts, an accelerometer that is absolute but noisy and confounded by linear
  acceleration).
* **Signal conditioning** -- moving average, biquad low-pass, notch. These
  remove content you do not want from a single channel, and every one of them
  costs you phase lag, which is a direct subtraction from your loop's phase
  margin. See ``docs/CONTROL_NOTES.md``.

All filters are causal, sample-by-sample, and hold their own state, so they can
be run in a real-time loop rather than only over a stored array. Each also
offers ``filt(array)`` for offline use in tests and examples.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ComplementaryFilter",
    "MadgwickAHRS",
    "KalmanFilter1D",
    "KalmanFilter",
    "MovingAverage",
    "BiquadLowPass",
    "NotchFilter",
    "quaternion_to_euler",
    "euler_to_quaternion",
    "quaternion_multiply",
    "quaternion_normalize",
]


# ======================================================================
# quaternion helpers  (w, x, y, z convention throughout)
# ======================================================================
def quaternion_normalize(q: ArrayLike) -> NDArray[np.float64]:
    """Unit-normalise a quaternion; falls back to identity if degenerate."""
    q_arr = np.asarray(q, dtype=float).ravel()
    n = float(np.linalg.norm(q_arr))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q_arr / n


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``q1 * q2`` with ``(w, x, y, z)`` ordering."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).ravel()
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).ravel()
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_to_euler(q: ArrayLike) -> tuple[float, float, float]:
    """Convert to ``(roll, pitch, yaw)`` in radians, aerospace 3-2-1 order.

    Pitch is clamped before ``arcsin`` so a numerically-just-over-unity
    argument returns +/- 90 degrees instead of ``nan``. Near +/-90 degrees
    pitch the roll/yaw split is ill-conditioned (gimbal lock in the *Euler
    representation*, not in the filter); that is a reason to keep the state as
    a quaternion and convert only for display.
    """
    w, x, y, z = quaternion_normalize(q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Inverse of :func:`quaternion_to_euler`."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


# ======================================================================
# complementary filter
# ======================================================================
class ComplementaryFilter:
    """Accel + gyro -> roll and pitch, the two-line fusion that actually works.

    ``angle = alpha * (angle + gyro * dt) + (1 - alpha) * angle_accel``

    The gyro path is a high-pass (it is trusted at high frequency, where it is
    smooth, and its DC drift is rejected) and the accel path is a low-pass (it
    is trusted at DC, where it is absolute, and its high-frequency content --
    vibration and the vehicle's own linear acceleration -- is rejected). The
    two transfer functions sum to exactly 1 at every frequency, which is where
    the name comes from and why there is no gain hole.

    tau vs cutoff
    -------------
    ``alpha = tau / (tau + dt)``, and ``tau`` is the time constant of the
    crossover between the two paths:

    ``f_c = 1 / (2 * pi * tau)``

    So ``tau = 1 s`` puts the crossover at 0.16 Hz: below that you follow the
    accelerometer, above it you follow the gyro. Choosing ``tau`` is choosing
    how long you are willing to believe the gyro before the accelerometer pulls
    you back. Longer ``tau`` rejects more linear-acceleration error (a
    hovering-to-forward-flight transition tilts the accel vector by exactly the
    amount you do *not* want to believe) but takes proportionally longer to
    wash out gyro bias. On a multirotor, 0.5-2 s is the usual range. Note that
    ``alpha`` depends on ``dt``, so it is recomputed every call rather than
    frozen at construction -- freezing it is the classic bug that makes a
    filter behave differently at 250 Hz than it did at 100 Hz.

    Steady-state behaviour under a constant gyro bias ``b``: the filter settles
    at an angle offset of ``b * tau`` from truth. That is the design trade in
    one equation. A 1 deg/s bias with ``tau = 1 s`` costs you 1 degree of tilt.
    """

    def __init__(self, tau: float = 1.0, roll: float = 0.0, pitch: float = 0.0) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be > 0")
        self.tau = float(tau)
        self.roll = float(roll)
        self.pitch = float(pitch)

    @property
    def cutoff_hz(self) -> float:
        """Crossover frequency between the accel and gyro paths."""
        return 1.0 / (2.0 * np.pi * self.tau)

    @classmethod
    def from_cutoff(cls, cutoff_hz: float, **kwargs) -> "ComplementaryFilter":
        if cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be > 0")
        return cls(tau=1.0 / (2.0 * np.pi * cutoff_hz), **kwargs)

    def alpha(self, dt: float) -> float:
        return self.tau / (self.tau + dt)

    def reset(self, roll: float = 0.0, pitch: float = 0.0) -> None:
        self.roll = float(roll)
        self.pitch = float(pitch)

    @staticmethod
    def accel_angles(accel: ArrayLike) -> tuple[float, float]:
        """Roll and pitch from a 3-axis accelerometer, in radians.

        Assumes the body frame is FRD-ish with gravity read as ``+g`` on ``az``
        when level. ``arctan2`` on the full vector rather than ``arcsin`` on a
        single axis, so the result stays well-conditioned past 45 degrees.

        This is only valid when the vehicle's linear acceleration is small
        compared with ``g``. Under a hard acceleration the accelerometer is
        measuring specific force, not gravity, and these angles are simply
        wrong -- which is precisely why they are low-passed by the fusion.
        """
        ax, ay, az = np.asarray(accel, dtype=float).ravel()
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.hypot(ay, az))
        return float(roll), float(pitch)

    def update(self, gyro: ArrayLike, accel: ArrayLike, dt: float) -> tuple[float, float]:
        """One fusion step.

        Parameters
        ----------
        gyro:
            ``(gx, gy, gz)`` body rates in rad/s.
        accel:
            ``(ax, ay, az)`` specific force, any consistent unit (only the
            direction is used).
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        gx, gy, _gz = np.asarray(gyro, dtype=float).ravel()
        a = self.alpha(dt)
        roll_a, pitch_a = self.accel_angles(accel)
        self.roll = a * (self.roll + gx * dt) + (1.0 - a) * roll_a
        self.pitch = a * (self.pitch + gy * dt) + (1.0 - a) * pitch_a
        return self.roll, self.pitch


# ======================================================================
# Madgwick
# ======================================================================
class MadgwickAHRS:
    """Madgwick's gradient-descent AHRS filter (IMU and MARG variants).

    The idea: propagate the quaternion with the gyro, then take one
    gradient-descent step on the objective "rotate the reference gravity (and
    magnetic) vector by ``q`` and match the measurement". The step size is
    ``beta``, which has a physical reading -- it is the maximum rate at which
    the filter will correct, in the same units as gyro error. Madgwick's own
    suggestion is ``beta = sqrt(3/4) * gyro_drift_rate_in_rad_per_s``.

    Why use it over a full EKF: it costs a few dozen flops, has one tuning
    parameter, and does not require you to maintain a covariance. Why not: it
    gives you no uncertainty estimate, so you cannot tell whether it is
    diverging. Use :mod:`dctk.estimator` when you need that.

    Tuning ``beta`` is the same trade as ``tau`` in the complementary filter.
    Too low and gyro bias walks the estimate; too high and every bump in linear
    acceleration is interpreted as a tilt. 0.03-0.1 is a normal range for a
    consumer MEMS IMU.
    """

    def __init__(self, beta: float = 0.05, q0: Optional[ArrayLike] = None) -> None:
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        self.beta = float(beta)
        self.q = quaternion_normalize(q0 if q0 is not None else [1.0, 0.0, 0.0, 0.0])

    def reset(self, q0: Optional[ArrayLike] = None) -> None:
        self.q = quaternion_normalize(q0 if q0 is not None else [1.0, 0.0, 0.0, 0.0])

    @property
    def euler(self) -> tuple[float, float, float]:
        return quaternion_to_euler(self.q)

    def update(
        self, gyro: ArrayLike, accel: ArrayLike, dt: float, mag: Optional[ArrayLike] = None
    ) -> NDArray[np.float64]:
        """Advance the filter. Supply ``mag`` for the MARG (yaw-observable) form.

        Without a magnetometer, yaw is unobservable from accel+gyro alone --
        gravity carries no heading information. The IMU-only branch therefore
        lets yaw drift freely, and that is correct behaviour, not a bug. If you
        need heading without a magnetometer you need a different observation:
        GPS course, optical flow, or a visual system.
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        q = self.q
        gx, gy, gz = np.asarray(gyro, dtype=float).ravel()

        # Rate of change from the gyro: 0.5 * q (x) omega
        qdot = 0.5 * quaternion_multiply(q, np.array([0.0, gx, gy, gz]))

        a = np.asarray(accel, dtype=float).ravel()
        a_norm = float(np.linalg.norm(a))
        if a_norm > 1e-9:
            a = a / a_norm
            q0, q1, q2, q3 = q
            if mag is None:
                # Objective: rotate reference gravity [0,0,1] into the body
                # frame and match the normalised accel reading.
                f = np.array(
                    [
                        2.0 * (q1 * q3 - q0 * q2) - a[0],
                        2.0 * (q0 * q1 + q2 * q3) - a[1],
                        2.0 * (0.5 - q1 * q1 - q2 * q2) - a[2],
                    ]
                )
                J = np.array(
                    [
                        [-2.0 * q2, 2.0 * q3, -2.0 * q0, 2.0 * q1],
                        [2.0 * q1, 2.0 * q0, 2.0 * q3, 2.0 * q2],
                        [0.0, -4.0 * q1, -4.0 * q2, 0.0],
                    ]
                )
            else:
                m = np.asarray(mag, dtype=float).ravel()
                m_norm = float(np.linalg.norm(m))
                if m_norm < 1e-9:
                    return self.update(gyro, accel, dt, mag=None)
                m = m / m_norm
                # Rotate the measured field into the earth frame and flatten it
                # onto the horizontal plane; this removes the local inclination
                # so the filter does not need it as a parameter.
                h = quaternion_multiply(
                    quaternion_multiply(q, np.array([0.0, m[0], m[1], m[2]])),
                    np.array([q[0], -q[1], -q[2], -q[3]]),
                )
                bx = float(np.hypot(h[1], h[2]))
                bz = float(h[3])
                f = np.array(
                    [
                        2.0 * (q1 * q3 - q0 * q2) - a[0],
                        2.0 * (q0 * q1 + q2 * q3) - a[1],
                        2.0 * (0.5 - q1 * q1 - q2 * q2) - a[2],
                        2.0 * bx * (0.5 - q2 * q2 - q3 * q3)
                        + 2.0 * bz * (q1 * q3 - q0 * q2)
                        - m[0],
                        2.0 * bx * (q1 * q2 - q0 * q3)
                        + 2.0 * bz * (q0 * q1 + q2 * q3)
                        - m[1],
                        2.0 * bx * (q0 * q2 + q1 * q3)
                        + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2)
                        - m[2],
                    ]
                )
                J = np.array(
                    [
                        [-2.0 * q2, 2.0 * q3, -2.0 * q0, 2.0 * q1],
                        [2.0 * q1, 2.0 * q0, 2.0 * q3, 2.0 * q2],
                        [0.0, -4.0 * q1, -4.0 * q2, 0.0],
                        [
                            -2.0 * bz * q2,
                            2.0 * bz * q3,
                            -4.0 * bx * q2 - 2.0 * bz * q0,
                            -4.0 * bx * q3 + 2.0 * bz * q1,
                        ],
                        [
                            -2.0 * bx * q3 + 2.0 * bz * q1,
                            2.0 * bx * q2 + 2.0 * bz * q0,
                            2.0 * bx * q1 + 2.0 * bz * q3,
                            -2.0 * bx * q0 + 2.0 * bz * q2,
                        ],
                        [
                            2.0 * bx * q2,
                            2.0 * bx * q3 - 4.0 * bz * q1,
                            2.0 * bx * q0 - 4.0 * bz * q2,
                            2.0 * bx * q1,
                        ],
                    ]
                )
            grad = J.T @ f
            g_norm = float(np.linalg.norm(grad))
            if g_norm > 1e-12:
                qdot = qdot - self.beta * (grad / g_norm)

        self.q = quaternion_normalize(q + qdot * dt)
        return self.q


# ======================================================================
# Kalman
# ======================================================================
class KalmanFilter1D:
    """Scalar constant-value Kalman filter (a tuned exponential smoother).

    Model: ``x[k+1] = x[k] + w``, ``z = x + v``. With ``Q`` process variance and
    ``R`` measurement variance, this is exactly a first-order low-pass whose
    gain adapts: it starts fast (large ``P``, trusts the measurement) and
    settles to a steady-state gain of ``K = P/(P+R)``. Its usefulness over a
    fixed alpha is that the transient is optimal and ``P`` tells you how
    confident it is.

    The ratio ``Q/R`` is the only thing that matters in steady state. Doubling
    both changes nothing.
    """

    def __init__(self, q: float = 1e-3, r: float = 1e-1, x0: float = 0.0, p0: float = 1.0) -> None:
        if q < 0.0 or r <= 0.0 or p0 < 0.0:
            raise ValueError("require q >= 0, r > 0, p0 >= 0")
        self.q = float(q)
        self.r = float(r)
        self.x = float(x0)
        self.p = float(p0)
        self._x0, self._p0 = float(x0), float(p0)

    def reset(self) -> None:
        self.x, self.p = self._x0, self._p0

    def update(self, z: float, dt: float = 1.0, u: float = 0.0) -> float:
        """Predict with optional rate input ``u`` (units/second), then correct."""
        self.x += u * dt
        self.p += self.q * dt
        k = self.p / (self.p + self.r)
        innovation = float(z) - self.x
        self.x += k * innovation
        self.p = (1.0 - k) * self.p
        self.gain = k
        self.innovation = innovation
        return self.x


class KalmanFilter:
    """Linear multi-dimensional Kalman filter with Joseph-form covariance update.

    ``x[k+1] = F x[k] + B u[k] + w``,  ``z = H x + v``.

    Deliberately uses the Joseph form

    ``P = (I - KH) P (I - KH)' + K R K'``

    rather than the cheaper ``P = (I - KH) P``. The short form is algebraically
    equivalent but numerically fragile: it subtracts two similar matrices, so
    round-off can push ``P`` asymmetric and eventually indefinite. When that
    happens the filter does not crash -- it quietly starts producing negative
    variances and a gain that makes no sense, and you find out in the flight
    log. The Joseph form is a sum of two quadratic forms, so it stays symmetric
    positive-semi-definite by construction. It costs one extra matrix multiply.
    """

    def __init__(
        self,
        F: ArrayLike,
        H: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        B: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
        P0: Optional[ArrayLike] = None,
    ) -> None:
        self.F = np.atleast_2d(np.asarray(F, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        n = self.F.shape[0]
        if self.F.shape != (n, n):
            raise ValueError("F must be square")
        if self.H.shape[1] != n:
            raise ValueError("H column count must match the state dimension")
        if self.Q.shape != (n, n):
            raise ValueError("Q must match the state dimension")
        m = self.H.shape[0]
        if self.R.shape != (m, m):
            raise ValueError("R must match the measurement dimension")
        self.B = None if B is None else np.atleast_2d(np.asarray(B, dtype=float))
        self.n, self.m = n, m
        self._x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
        self._P0 = np.eye(n) if P0 is None else np.atleast_2d(np.asarray(P0, dtype=float)).copy()
        self.x = self._x0.copy()
        self.P = self._P0.copy()
        self.innovation = np.zeros(m)
        self.innovation_cov = np.eye(m)

    def reset(self) -> None:
        self.x = self._x0.copy()
        self.P = self._P0.copy()

    def predict(self, u: Optional[ArrayLike] = None) -> NDArray[np.float64]:
        self.x = self.F @ self.x
        if u is not None:
            if self.B is None:
                raise ValueError("control input supplied but B was not given")
            self.x = self.x + self.B @ np.asarray(u, dtype=float).ravel()
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x

    def update(self, z: ArrayLike) -> NDArray[np.float64]:
        z_arr = np.asarray(z, dtype=float).ravel()
        if z_arr.size != self.m:
            raise ValueError(f"measurement must have {self.m} elements")
        y = z_arr - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + K @ y
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.innovation = y
        self.innovation_cov = S
        self.K = K
        return self.x

    def nis(self) -> float:
        """Normalised Innovation Squared, ``y' S^-1 y``.

        Should average to the measurement dimension ``m`` if the filter is
        consistent. Persistently much larger means the filter is overconfident
        (``Q`` or ``R`` too small, or the model is wrong); persistently much
        smaller means it is throwing away information.
        """
        return float(self.innovation @ np.linalg.solve(self.innovation_cov, self.innovation))


# ======================================================================
# signal conditioning
# ======================================================================
class MovingAverage:
    """N-sample boxcar.

    Honest assessment: a moving average is a bad low-pass. Its stopband is
    terrible (first sidelobe only ~13 dB down) and it has nulls at multiples of
    ``fs/N``, so it attenuates some frequencies completely and passes their
    neighbours. It is included because it is what people reach for, and because
    for a *known* periodic disturbance whose period you can match exactly to
    ``N`` samples it is genuinely optimal. Its one real virtue is exactly
    linear phase: the delay is exactly ``(N-1)/2`` samples at every frequency.
    For everything else, use :class:`BiquadLowPass`.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self._buf = np.zeros(self.n)
        self._count = 0
        self._idx = 0
        self._sum = 0.0

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._count = 0
        self._idx = 0
        self._sum = 0.0

    @property
    def group_delay_samples(self) -> float:
        return (self.n - 1) / 2.0

    def update(self, x: float) -> float:
        self._sum -= self._buf[self._idx]
        self._buf[self._idx] = float(x)
        self._sum += float(x)
        self._idx = (self._idx + 1) % self.n
        self._count = min(self._count + 1, self.n)
        return float(self._sum / self._count)

    def filt(self, data: ArrayLike) -> NDArray[np.float64]:
        self.reset()
        return np.array([self.update(v) for v in np.asarray(data, dtype=float).ravel()])


class _Biquad:
    """Direct-form-I biquad section. Base for the low-pass and notch."""

    def __init__(self, b: Sequence[float], a: Sequence[float]) -> None:
        b = np.asarray(b, dtype=float)
        a = np.asarray(a, dtype=float)
        if a[0] == 0.0:
            raise ValueError("a[0] must be non-zero")
        self.b = b / a[0]
        self.a = a / a[0]
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0

    def reset(self, value: float = 0.0) -> None:
        """Reset to a steady state consistent with a constant input ``value``.

        Zeroing the history instead would make the filter ring on the first
        sample after re-enable, which on a rate loop is an audible motor snap.
        """
        self._x1 = self._x2 = float(value)
        dc = float(np.sum(self.b) / np.sum(self.a))
        self._y1 = self._y2 = float(value) * dc

    def update(self, x: float) -> float:
        x = float(x)
        y = (
            self.b[0] * x
            + self.b[1] * self._x1
            + self.b[2] * self._x2
            - self.a[1] * self._y1
            - self.a[2] * self._y2
        )
        self._x2, self._x1 = self._x1, x
        self._y2, self._y1 = self._y1, y
        return float(y)

    def filt(self, data: ArrayLike, *, prime: bool = True) -> NDArray[np.float64]:
        data_arr = np.asarray(data, dtype=float).ravel()
        self.reset(float(data_arr[0]) if (prime and data_arr.size) else 0.0)
        return np.array([self.update(v) for v in data_arr])

    def response(self, freqs_hz: ArrayLike, fs: float) -> NDArray[np.complex128]:
        """Complex frequency response at the given frequencies."""
        w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) / fs
        z = np.exp(-1j * w)
        num = self.b[0] + self.b[1] * z + self.b[2] * z**2
        den = self.a[0] + self.a[1] * z + self.a[2] * z**2
        return num / den

    def gain_at(self, freq_hz: float, fs: float) -> float:
        """Magnitude response at one frequency (linear, not dB)."""
        return float(np.abs(self.response([freq_hz], fs)[0]))

    def phase_lag_deg(self, freq_hz: float, fs: float) -> float:
        """Phase lag in degrees at one frequency. Positive means lag.

        This is the number to look at before you drop a filter into a control
        loop. Lag here is subtracted directly from your phase margin.
        """
        return float(-np.degrees(np.angle(self.response([freq_hz], fs)[0])))


class BiquadLowPass(_Biquad):
    """Second-order Butterworth low-pass (RBJ cookbook, bilinear transform).

    A single biquad at ``Q = 1/sqrt(2)`` is maximally flat in the passband --
    the standard 2nd-order Butterworth. It rolls off at 40 dB/decade, which is
    twice what a first-order filter gives you for roughly the same phase cost
    near the cutoff, and that is why every flight stack uses biquads rather
    than chained RC filters on gyro data.

    The frequency is pre-warped (``tan(pi*f/fs)``) so the analogue cutoff lands
    where you asked for it after the bilinear transform. Without pre-warping
    the realised cutoff creeps low as ``f_c`` approaches ``fs/2`` -- a 100 Hz
    filter at 500 Hz sampling would come out closer to 92 Hz.

    Phase cost, which is the part that matters: this filter has 90 degrees of
    lag *at* the cutoff and approaches 180 degrees above it. Put a 40 Hz
    low-pass on a rate loop that crosses over at 20 Hz and you have spent about
    50 degrees of phase margin.
    """

    def __init__(self, cutoff_hz: float, fs: float, q: float = 0.7071067811865476) -> None:
        if cutoff_hz <= 0.0 or fs <= 0.0:
            raise ValueError("cutoff_hz and fs must be > 0")
        if cutoff_hz >= 0.5 * fs:
            raise ValueError("cutoff_hz must be below the Nyquist frequency")
        if q <= 0.0:
            raise ValueError("q must be > 0")
        self.cutoff_hz = float(cutoff_hz)
        self.fs = float(fs)
        self.q = float(q)
        w0 = 2.0 * np.pi * cutoff_hz / fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        b = np.array([(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0])
        a = np.array([1.0 + alpha, -2.0 * cw, 1.0 - alpha])
        super().__init__(b, a)


class NotchFilter(_Biquad):
    """Second-order band-stop for a narrow mechanical/aerodynamic tone.

    Parameters
    ----------
    freq_hz:
        Centre of the notch.
    fs:
        Sample rate.
    q:
        Quality factor, ``f_centre / bandwidth_-3dB``. High ``Q`` (20-30) is a
        surgical notch that removes almost no phase from the rest of the band;
        low ``Q`` (2-5) is a wide bite that also costs you phase either side.

    Why fixed notches disappoint
    ----------------------------
    The dominant vibration tone on a multirotor is blade-pass frequency:

    ``f_bp = motor_rpm / 60 * n_blades``

    A 5-inch quad on 6S idles around 5 000 rpm and pulls 25 000+ rpm on a punch
    out. With a two-blade prop that is roughly 170 Hz to 830 Hz -- the tone
    sweeps across almost the entire useful band within a single throttle
    input. A fixed notch centred where you found the peak in a hover log is
    correct for hover and useless everywhere else; worse, the phase lag it adds
    is *permanent* even when the tone has moved away, so you pay the stability
    cost all the time and get the benefit occasionally.

    Widening ``Q`` to cover the range does not save you: a notch broad enough
    to span 170-830 Hz is just a bad low-pass with a lot of phase lag.

    The real fix is an RPM-tracked notch: take motor RPM from ESC telemetry
    (DShot bidirectional, or a dedicated telemetry line), compute blade-pass
    per motor, and retune the notch centre every loop. Modern stacks run
    several of those plus a harmonic. If you have no RPM telemetry, an FFT-
    driven dynamic notch is the fallback -- it finds the peak in a running
    spectrum instead of being told where it is. See ``flight-log-analyzer``
    for extracting the actual vibration spectrum from a flight log so you know
    what you are chasing before you start filtering.

    :meth:`retune` exists so you can drive the centre frequency from RPM in a
    live loop without reallocating the filter or losing its state.
    """

    def __init__(self, freq_hz: float, fs: float, q: float = 20.0) -> None:
        if fs <= 0.0:
            raise ValueError("fs must be > 0")
        if q <= 0.0:
            raise ValueError("q must be > 0")
        self.fs = float(fs)
        self.q = float(q)
        self.freq_hz = 0.0
        b, a = self._coeffs(freq_hz, fs, q)
        super().__init__(b, a)
        self.freq_hz = float(freq_hz)

    @staticmethod
    def _coeffs(freq_hz: float, fs: float, q: float):
        if not 0.0 < freq_hz < 0.5 * fs:
            raise ValueError("freq_hz must be in (0, fs/2)")
        w0 = 2.0 * np.pi * freq_hz / fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        b = np.array([1.0, -2.0 * cw, 1.0])
        a = np.array([1.0 + alpha, -2.0 * cw, 1.0 - alpha])
        return b, a

    def retune(self, freq_hz: float, q: Optional[float] = None) -> None:
        """Move the notch centre without resetting the delay line.

        Recomputing coefficients in place is safe for a biquad as long as you
        do not also zero the history: the state is in units of the signal, not
        of the old coefficients, so a modest frequency step produces a small
        transient rather than a discontinuity. Step the centre smoothly (rate-
        limit it, or filter the RPM signal) if you are tracking a fast throttle
        change.
        """
        if q is not None:
            if q <= 0.0:
                raise ValueError("q must be > 0")
            self.q = float(q)
        b, a = self._coeffs(freq_hz, self.fs, self.q)
        self.b = b / a[0]
        self.a = a / a[0]
        self.freq_hz = float(freq_hz)

    @staticmethod
    def blade_pass_hz(rpm, blades: int = 2):
        """Blade-pass frequency for a given motor RPM and blade count.

        Accepts a scalar or an array of RPM values, so a whole throttle sweep
        from a log can be converted in one call.
        """
        if blades < 1:
            raise ValueError("blades must be >= 1")
        rpm_arr = np.asarray(rpm, dtype=float)
        result = rpm_arr / 60.0 * float(blades)
        return float(result) if result.ndim == 0 else result
