"""Extended Kalman Filter for attitude from an IMU.

State: unit quaternion plus gyro bias, ``[qw, qx, qy, qz, bx, by, bz]``.
Propagation is by gyro integration; correction is by accelerometer (gravity
direction, which fixes roll and pitch) and optionally magnetometer (which fixes
yaw).

Why bother, when :class:`dctk.filters.MadgwickAHRS` is a tenth of the code:

* The EKF estimates **gyro bias** as part of the state. Madgwick's ``beta``
  fights bias but never learns it, so it always sits at a bias-dependent
  offset. The EKF converges the offset to zero.
* The EKF carries a **covariance**, so it can weight the accelerometer less
  when the vehicle is manoeuvring and more when it is still, and it can tell
  you how confident it is.
* The EKF gives you **innovation and NIS**, which is how you diagnose a
  diverging filter from a log instead of from a crash.

Design decisions that keep it stable in floating point:

* Covariance updates use the **Joseph form**, so ``P`` stays symmetric
  positive-semi-definite even after millions of samples.
* ``P`` is explicitly re-symmetrised after every operation. Cheap insurance.
* The quaternion is renormalised every step and the corresponding covariance
  rows/columns are left alone; for small corrections this is standard practice
  and is far more robust than trying to maintain a constrained covariance.
* The accelerometer update is **rejected** when ``|a|`` differs from ``g`` by
  more than a configurable margin. Under a hard manoeuvre the accelerometer is
  measuring specific force, not gravity, and folding that in as a tilt
  observation is how attitude estimators tip over during aggressive flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filters import quaternion_normalize, quaternion_to_euler

__all__ = ["AttitudeEKF", "EKFDiagnostics"]


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _quat_to_dcm(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix that takes a vector from body frame to earth frame."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class EKFDiagnostics:
    """Per-update diagnostics. Log these; they are how you debug a filter."""

    innovation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    innovation_cov: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    nis: float = 0.0
    accepted: bool = True
    reason: str = ""


class AttitudeEKF:
    """Quaternion + gyro-bias EKF.

    Parameters
    ----------
    gyro_noise:
        Angular random walk, rad/s per sqrt(Hz)-ish. Drives ``Q`` on the
        quaternion block. Bigger means "trust the gyro less", which speeds up
        accel correction at the cost of noisier attitude.
    gyro_bias_noise:
        Rate random walk on the bias states. Set this from the datasheet's bias
        instability. Too small and the estimated bias freezes and cannot track
        thermal drift; too large and the bias state absorbs real motion.
    accel_noise:
        Measurement variance for the normalised gravity observation.
    mag_noise:
        Measurement variance for the normalised magnetic observation.
    gravity:
        Local ``g`` in the same units as the accelerometer input.
    accel_gate:
        Fractional tolerance on ``|a| / g``. A reading outside
        ``[1 - gate, 1 + gate]`` is rejected. 0.1 (10 %) is a reasonable start.
    nis_gate:
        Chi-square style gate on the normalised innovation squared. An update
        whose NIS exceeds this is rejected as an outlier. ``None`` disables.
        For a 3-element innovation the 99.9th percentile of chi-square is about
        16.3, so a gate around 20-30 rejects genuine outliers without
        discarding healthy updates.
    """

    N_STATES = 7

    def __init__(
        self,
        *,
        gyro_noise: float = 1e-4,
        gyro_bias_noise: float = 1e-7,
        accel_noise: float = 5e-2,
        mag_noise: float = 1e-1,
        gravity: float = 9.81,
        accel_gate: float = 0.15,
        nis_gate: Optional[float] = 25.0,
        q0: Optional[ArrayLike] = None,
        bias0: Optional[ArrayLike] = None,
        p0_attitude: float = 1e-2,
        p0_bias: float = 1e-4,
    ) -> None:
        if min(gyro_noise, gyro_bias_noise) < 0.0:
            raise ValueError("noise densities must be >= 0")
        if accel_noise <= 0.0 or mag_noise <= 0.0:
            raise ValueError("measurement noises must be > 0")
        if gravity <= 0.0:
            raise ValueError("gravity must be > 0")

        self.gyro_noise = float(gyro_noise)
        self.gyro_bias_noise = float(gyro_bias_noise)
        self.accel_noise = float(accel_noise)
        self.mag_noise = float(mag_noise)
        self.gravity = float(gravity)
        self.accel_gate = float(accel_gate)
        self.nis_gate = nis_gate

        self._q0 = quaternion_normalize(q0 if q0 is not None else [1.0, 0.0, 0.0, 0.0])
        self._b0 = (
            np.zeros(3) if bias0 is None else np.asarray(bias0, dtype=float).ravel().copy()
        )
        self._p0 = np.diag(
            [p0_attitude] * 4 + [p0_bias] * 3
        )
        self.x = np.concatenate([self._q0, self._b0])
        self.P = self._p0.copy()
        self.diagnostics = EKFDiagnostics()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.x = np.concatenate([self._q0, self._b0])
        self.P = self._p0.copy()
        self.diagnostics = EKFDiagnostics()

    @property
    def quaternion(self) -> NDArray[np.float64]:
        return self.x[:4].copy()

    @property
    def gyro_bias(self) -> NDArray[np.float64]:
        return self.x[4:].copy()

    @property
    def euler(self) -> tuple[float, float, float]:
        """``(roll, pitch, yaw)`` in radians."""
        return quaternion_to_euler(self.x[:4])

    def is_covariance_psd(self, *, tol: float = 1e-9) -> bool:
        """True if ``P`` is symmetric and has no meaningfully negative eigenvalue.

        Worth asserting in a test and worth checking periodically in flight: a
        ``P`` that has gone indefinite is a filter that has already failed, and
        it will happily keep producing numbers.
        """
        sym = float(np.max(np.abs(self.P - self.P.T)))
        if sym > tol:
            return False
        return bool(np.min(np.linalg.eigvalsh(0.5 * (self.P + self.P.T))) >= -tol)

    # ------------------------------------------------------------------
    def predict(self, gyro: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Propagate with the measured body rates.

        The bias estimate is subtracted from the raw gyro before integration,
        which is the entire reason the bias state exists. The quaternion
        kinematics ``qdot = 0.5 * Omega(w) q`` are integrated with the exact
        first-order-hold rotation (axis-angle) rather than Euler, so the
        quaternion stays close to unit norm even at large rates.
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        w_meas = np.asarray(gyro, dtype=float).ravel()
        if w_meas.size != 3:
            raise ValueError("gyro must have 3 elements")
        q = self.x[:4]
        b = self.x[4:]
        w = w_meas - b

        angle = float(np.linalg.norm(w) * dt)
        if angle > 1e-12:
            axis = w / np.linalg.norm(w)
            dq = np.concatenate([[np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        qw, qx, qy, qz = q
        dw, dx, dy, dz = dq
        q_new = np.array(
            [
                qw * dw - qx * dx - qy * dy - qz * dz,
                qw * dx + qx * dw + qy * dz - qz * dy,
                qw * dy - qx * dz + qy * dw + qz * dx,
                qw * dz + qx * dy - qy * dx + qz * dw,
            ]
        )
        q_new = quaternion_normalize(q_new)

        # Jacobian of the quaternion kinematics with respect to (q, b).
        Omega = 0.5 * np.array(
            [
                [0.0, -w[0], -w[1], -w[2]],
                [w[0], 0.0, w[2], -w[1]],
                [w[1], -w[2], 0.0, w[0]],
                [w[2], w[1], -w[0], 0.0],
            ]
        )
        Xi = 0.5 * np.array(
            [
                [-q[1], -q[2], -q[3]],
                [q[0], -q[3], q[2]],
                [q[3], q[0], -q[1]],
                [-q[2], q[1], q[0]],
            ]
        )
        F = np.eye(self.N_STATES)
        F[:4, :4] = np.eye(4) + Omega * dt
        F[:4, 4:] = -Xi * dt

        Q = np.zeros((self.N_STATES, self.N_STATES))
        Q[:4, :4] = (self.gyro_noise * dt) * (Xi @ Xi.T) * 4.0 + np.eye(4) * (
            self.gyro_noise * dt * 1e-3
        )
        Q[4:, 4:] = np.eye(3) * (self.gyro_bias_noise * dt)

        self.x = np.concatenate([q_new, b])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x

    # ------------------------------------------------------------------
    def _joseph_update(
        self, H: NDArray[np.float64], y: NDArray[np.float64], R: NDArray[np.float64], label: str
    ) -> bool:
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            S_inv_y = np.linalg.solve(S, y)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate S
            self.diagnostics = EKFDiagnostics(y, S, float("inf"), False, f"{label}: singular S")
            return False
        nis = float(y @ S_inv_y)

        if self.nis_gate is not None and nis > self.nis_gate:
            self.diagnostics = EKFDiagnostics(y, S, nis, False, f"{label}: NIS gate")
            return False

        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ y
        I_KH = np.eye(self.N_STATES) - K @ H
        # Joseph form: a sum of two quadratic forms, so the result is
        # symmetric PSD by construction even with round-off.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x[:4] = quaternion_normalize(self.x[:4])
        self.diagnostics = EKFDiagnostics(y, S, nis, True, label)
        return True

    def update_accel(self, accel: ArrayLike) -> bool:
        """Correct roll and pitch from the gravity direction.

        Returns ``True`` if the update was accepted. Rejection is normal and
        expected during manoeuvres; a filter that *never* rejects has its gate
        set too wide.
        """
        a = np.asarray(accel, dtype=float).ravel()
        if a.size != 3:
            raise ValueError("accel must have 3 elements")
        norm = float(np.linalg.norm(a))
        if norm < 1e-9:
            self.diagnostics = EKFDiagnostics(np.zeros(3), np.eye(3), 0.0, False, "accel: zero")
            return False
        if abs(norm / self.gravity - 1.0) > self.accel_gate:
            self.diagnostics = EKFDiagnostics(
                np.zeros(3), np.eye(3), 0.0, False, "accel: magnitude gate (manoeuvring)"
            )
            return False

        q = self.x[:4]
        R_be = _quat_to_dcm(q)  # body -> earth
        g_earth = np.array([0.0, 0.0, 1.0])
        predicted = R_be.T @ g_earth  # expected normalised accel in body frame
        y = (a / norm) - predicted

        H = np.zeros((3, self.N_STATES))
        H[:, :4] = self._dcm_transpose_jacobian(q, g_earth)
        R = np.eye(3) * self.accel_noise
        return self._joseph_update(H, y, R, "accel")

    def update_mag(self, mag: ArrayLike, declination: float = 0.0) -> bool:
        """Correct yaw from the magnetometer.

        The reference field is taken as horizontal-north after removing the
        inclination numerically, so you do not have to supply the local dip
        angle. Hard-iron and soft-iron calibration are *your* problem and are
        not optional: an uncalibrated magnetometer on a multirotor sees the
        current in the power leads far more strongly than it sees the Earth.
        """
        m = np.asarray(mag, dtype=float).ravel()
        if m.size != 3:
            raise ValueError("mag must have 3 elements")
        norm = float(np.linalg.norm(m))
        if norm < 1e-9:
            self.diagnostics = EKFDiagnostics(np.zeros(3), np.eye(3), 0.0, False, "mag: zero")
            return False
        m_unit = m / norm

        q = self.x[:4]
        R_be = _quat_to_dcm(q)
        m_earth = R_be @ m_unit
        horiz = float(np.hypot(m_earth[0], m_earth[1]))
        ref_earth = np.array(
            [horiz * np.cos(declination), horiz * np.sin(declination), m_earth[2]]
        )
        n = float(np.linalg.norm(ref_earth))
        if n < 1e-9:  # pragma: no cover - degenerate field
            return False
        ref_earth = ref_earth / n

        predicted = R_be.T @ ref_earth
        y = m_unit - predicted
        H = np.zeros((3, self.N_STATES))
        H[:, :4] = self._dcm_transpose_jacobian(q, ref_earth)
        R = np.eye(3) * self.mag_noise
        return self._joseph_update(H, y, R, "mag")

    @staticmethod
    def _dcm_transpose_jacobian(
        q: NDArray[np.float64], v_earth: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """d(R(q)' v) / dq, computed analytically (3x4)."""
        w, x, y, z = q
        vx, vy, vz = v_earth
        # R' v where R is body->earth, i.e. earth vector expressed in body frame.
        dw = 2.0 * np.array(
            [w * vx + z * vy - y * vz, -z * vx + w * vy + x * vz, y * vx - x * vy + w * vz]
        )
        dx = 2.0 * np.array(
            [x * vx + y * vy + z * vz, y * vx - x * vy + w * vz, z * vx - w * vy - x * vz]
        )
        dy = 2.0 * np.array(
            [-y * vx + x * vy - w * vz, x * vx + y * vy + z * vz, w * vx + z * vy - y * vz]
        )
        dz = 2.0 * np.array(
            [-z * vx + w * vy + x * vz, -w * vx - z * vy + y * vz, x * vx + y * vy + z * vz]
        )
        return np.column_stack([dw, dx, dy, dz])

    # ------------------------------------------------------------------
    def step(
        self,
        gyro: ArrayLike,
        accel: ArrayLike,
        dt: float,
        mag: Optional[ArrayLike] = None,
        declination: float = 0.0,
    ) -> NDArray[np.float64]:
        """Predict then correct. Convenience wrapper for a fixed-rate loop."""
        self.predict(gyro, dt)
        self.update_accel(accel)
        if mag is not None:
            self.update_mag(mag, declination)
        return self.x
