#!/usr/bin/env python3
"""Four tunes on one realistic plant, with the numbers that justify the ranking.

The plant is a lightly damped second-order system measured through a noisy
sensor and driven through 15 ms of transport delay and a 25 ms motor lag -- an
actuator that is neither instant nor exact, which is the situation on every
real vehicle.

* **P only** -- the plant has a spring term, so holding a steady output needs a
  steady force, and with no integrator that force can only come from a
  permanent error. It never arrives.
* **PI** -- arrives, but the integrator adds phase lag on top of the delay and
  the response rings.
* **PID, raw D** -- damps the ringing, and turns the actuator command into
  noise. Look at the command-jitter number, not the tracking number.
* **PID, filtered D** -- identical tracking, an order of magnitude less command
  jitter. This is the one you would fly.

Run: ``python3 examples/pid_tuning_comparison.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.metrics import step_metrics
from dctk.pid import PID
from dctk.sim import ActuatorDelay, MotorLag, SecondOrderPlant, SensorNoise, simulate

DT = 0.001
DURATION = 6.0
TARGET = 1.0


def run(label: str, **pid_kwargs):
    plant = SecondOrderPlant.from_modal(wn=6.0, zeta=0.2, dc_gain=1.0)
    pid = PID(output_limits=(-8.0, 8.0), **pid_kwargs)
    result = simulate(
        plant,
        pid.update,
        duration=DURATION,
        dt=DT,
        setpoint=TARGET,
        sensor_noise=SensorNoise(sigma=0.01, seed=5),
        actuator_delay=ActuatorDelay(0.015, DT),
        motor_lag=MotorLag(0.025),
    )
    metrics = step_metrics(result.t, result.y, TARGET)
    # Command jitter: RMS of the sample-to-sample change in the raw command,
    # measured after the transient. This is what heats motors and what you hear
    # as a buzz on the bench.
    jitter = float(np.std(np.diff(result.u[int(2.0 / DT):])))
    return label, result, metrics, jitter


def main() -> None:
    header("PID tuning comparison -- noisy sensor, 15 ms delay, 25 ms motor lag")

    runs = [
        run("P only", kp=1.2, ki=0.0, kd=0.0, derivative_cutoff_hz=None),
        run("PI", kp=1.2, ki=1.6, kd=0.0, derivative_cutoff_hz=None),
        run("PID, raw D", kp=1.2, ki=1.6, kd=0.15, derivative_cutoff_hz=None),
        run("PID, D filtered 15 Hz", kp=1.2, ki=1.6, kd=0.15, derivative_cutoff_hz=15.0),
    ]

    print()
    for label, _, m, jitter in runs:
        print(f"{m.summary(label)}  cmd_jitter={jitter:7.4f}")

    raw, filt = runs[2], runs[3]
    print()
    table(
        [
            ("P-only steady-state error", f"{runs[0][2].steady_state_error:+.4f}"),
            ("PI overshoot", f"{runs[1][2].overshoot_pct:.1f} %"),
            (
                "ITAE, raw D vs filtered D",
                f"{raw[2].itae:.4f} vs {filt[2].itae:.4f} (indistinguishable)",
            ),
            (
                "command jitter, raw D vs filtered D",
                f"{raw[3]:.4f} vs {filt[3]:.4f} "
                f"({raw[3] / filt[3]:.1f}x reduction for free)",
            ),
        ]
    )
    print(
        "\nThe D filter costs nothing in tracking here and removes almost all of the\n"
        "command noise. On a real airframe that noise is motor heat and audible buzz,\n"
        "and it is the reason raw D on an IMU is unusable rather than merely untidy."
    )

    fig, (ax1, ax2) = figure(2, 1, figsize=(10, 8), sharex=True)
    for label, result, m, _ in runs:
        ax1.plot(result.t, result.y, label=f"{label}  (ITAE {m.itae:.3f})")
    ax1.axhline(TARGET, color="k", ls="--", lw=0.8, label="setpoint")
    ax1.set_ylabel("output")
    ax1.set_title("Step response -- 2nd order plant, noisy sensor, delay + motor lag")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    for label, result, _, jitter in runs[2:]:
        ax2.plot(result.t, result.u, lw=0.7, label=f"{label}  (jitter {jitter:.3f})")
    ax2.set_ylabel("raw command")
    ax2.set_xlabel("time (s)")
    ax2.set_title("Controller output: raw D is noise, filtered D is a command")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    save(fig, "pid_tuning_comparison.png")


if __name__ == "__main__":
    main()
