#!/usr/bin/env python3
"""Four attitude estimators on the same synthetic IMU stream.

The IMU is realistic in the ways that matter: the gyro carries a constant bias
plus white noise, and the accelerometer carries white noise *and* is corrupted
by the vehicle's own linear acceleration during a manoeuvre window -- the case
that breaks naive tilt-from-accelerometer.

Estimators compared:

* **Accelerometer only** -- absolute, no drift, unusable during manoeuvres.
* **Gyro integration only** -- smooth, and the bias walks it away linearly.
* **Complementary filter** -- crossover between the two. Settles at an offset
  of ``bias * tau``, which is the whole design trade in one number.
* **Attitude EKF** -- estimates the bias as a state, so the offset converges to
  zero, and gates the accelerometer during the manoeuvre.

Run: ``python3 examples/complementary_vs_kalman.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.estimator import AttitudeEKF
from dctk.filters import ComplementaryFilter

DT = 0.005
DURATION = 60.0
GYRO_BIAS = np.deg2rad(2.0)  # 2 deg/s, a poor but entirely plausible MEMS gyro
GYRO_NOISE = np.deg2rad(0.4)
ACCEL_NOISE = 0.08
TAU = 1.0
MANOEUVRE = (30.0, 34.0)  # window of sustained lateral acceleration
LATERAL_ACCEL = 6.0  # m/s^2; enough to push |a| outside the EKF's magnitude gate


def truth(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True roll angle and true roll rate."""
    roll = np.deg2rad(15.0) * np.sin(2 * np.pi * 0.05 * t)
    rate = np.deg2rad(15.0) * 2 * np.pi * 0.05 * np.cos(2 * np.pi * 0.05 * t)
    return roll, rate


def main() -> None:
    header("Attitude estimation: accel, gyro, complementary filter, EKF")

    n = int(DURATION / DT) + 1
    t = np.arange(n) * DT
    roll_true, rate_true = truth(t)
    rng = np.random.default_rng(2026)

    cf = ComplementaryFilter(tau=TAU)
    ekf = AttitudeEKF(gyro_bias_noise=1e-6, accel_noise=0.08, accel_gate=0.10)

    est_accel = np.zeros(n)
    est_gyro = np.zeros(n)
    est_cf = np.zeros(n)
    est_ekf = np.zeros(n)
    bias_est = np.zeros(n)
    accel_used = np.zeros(n, dtype=bool)

    gyro_only = 0.0
    for i, ti in enumerate(t):
        # --- synthesise the IMU -------------------------------------------
        gyro = rate_true[i] + GYRO_BIAS + rng.normal(0.0, GYRO_NOISE)
        lateral = LATERAL_ACCEL if MANOEUVRE[0] <= ti < MANOEUVRE[1] else 0.0
        accel = np.array(
            [
                0.0,
                np.sin(roll_true[i]) * 9.81 + lateral,
                np.cos(roll_true[i]) * 9.81,
            ]
        ) + rng.normal(0.0, ACCEL_NOISE, 3)

        # --- estimators ----------------------------------------------------
        est_accel[i] = ComplementaryFilter.accel_angles(accel)[0]
        gyro_only += gyro * DT
        est_gyro[i] = gyro_only
        est_cf[i] = cf.update([gyro, 0.0, 0.0], accel, DT)[0]
        ekf.step([gyro, 0.0, 0.0], accel, DT)
        est_ekf[i] = ekf.euler[0]
        bias_est[i] = ekf.gyro_bias[0]
        accel_used[i] = ekf.diagnostics.accepted

    # --- report ------------------------------------------------------------
    manoeuvre = (t >= MANOEUVRE[0]) & (t < MANOEUVRE[1])
    # One full period of the truth sine, after the manoeuvre and after every
    # filter has settled. Averaging over a whole period keeps the mean of the
    # true angle at zero, so a mean error is a genuine offset and not an
    # artefact of where the window happened to start.
    quiet = (t >= 40.0) & (t <= 60.0)

    def rms(est, mask):
        return float(np.rad2deg(np.sqrt(np.mean((est[mask] - roll_true[mask]) ** 2))))

    print()
    rows = []
    for label, est in [
        ("accelerometer only", est_accel),
        ("gyro integration only", est_gyro),
        (f"complementary (tau={TAU:g} s)", est_cf),
        ("attitude EKF", est_ekf),
    ]:
        rows.append(
            (
                label,
                f"RMS error (quiet) = {rms(est, quiet):6.3f} deg   "
                f"RMS during manoeuvre = {rms(est, manoeuvre):6.3f} deg   "
                f"final error = {np.rad2deg(est[-1] - roll_true[-1]):+7.3f} deg",
            )
        )
    table(rows)

    predicted_offset = np.rad2deg(GYRO_BIAS * TAU)
    actual_offset = float(np.rad2deg(np.mean(est_cf[quiet] - roll_true[quiet])))
    print()
    table(
        [
            ("gyro bias (truth)", f"{np.rad2deg(GYRO_BIAS):.3f} deg/s"),
            ("gyro bias (EKF estimate)", f"{np.rad2deg(bias_est[-1]):.3f} deg/s"),
            (
                "complementary offset: bias*tau predicted vs measured",
                f"{predicted_offset:.3f} deg vs {actual_offset:.3f} deg",
            ),
            (
                "EKF mean offset over the same window",
                f"{float(np.rad2deg(np.mean(est_ekf[quiet] - roll_true[quiet]))):+.3f} deg",
            ),
            (
                "accel updates rejected by the EKF gate",
                f"{int(np.sum(~accel_used))} of {n} samples "
                f"({100 * np.mean(~accel_used):.1f} %)",
            ),
            (
                "gyro-only drift over 60 s",
                f"{np.rad2deg(est_gyro[-1] - roll_true[-1]):+.1f} deg",
            ),
        ]
    )
    print(
        "\nThe complementary filter's steady offset is bias*tau, exactly as predicted --\n"
        "that is the price of its simplicity. The EKF makes the bias a state and drives\n"
        "the offset to zero, and its magnitude gate throws away the accelerometer\n"
        "during the manoeuvre window instead of believing it."
    )

    fig, (ax1, ax2, ax3) = figure(3, 1, figsize=(11, 10), sharex=True)
    ax1.plot(t, np.rad2deg(roll_true), "k", lw=2, label="truth")
    ax1.plot(t, np.rad2deg(est_accel), lw=0.5, alpha=0.5, label="accel only")
    ax1.plot(t, np.rad2deg(est_cf), lw=1.2, label=f"complementary (tau={TAU:g}s)")
    ax1.plot(t, np.rad2deg(est_ekf), lw=1.2, label="EKF")
    ax1.axvspan(*MANOEUVRE, color="orange", alpha=0.2, label="lateral acceleration")
    ax1.set_ylabel("roll (deg)")
    ax1.set_title("Attitude estimate vs truth")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(t, np.rad2deg(est_cf - roll_true), label="complementary error")
    ax2.plot(t, np.rad2deg(est_ekf - roll_true), label="EKF error")
    ax2.axhline(predicted_offset, color="r", ls="--", lw=0.8,
                label=f"predicted bias*tau = {predicted_offset:.2f} deg")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_ylabel("error (deg)")
    ax2.set_title("Estimation error -- the complementary filter sits at bias*tau")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.plot(t, np.rad2deg(bias_est), label="EKF gyro bias estimate")
    ax3.axhline(np.rad2deg(GYRO_BIAS), color="k", ls="--", lw=0.8, label="true bias")
    ax3.set_ylabel("gyro bias (deg/s)")
    ax3.set_xlabel("time (s)")
    ax3.set_title("The EKF learns the bias; the complementary filter never can")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    save(fig, "complementary_vs_kalman.png")


if __name__ == "__main__":
    main()
