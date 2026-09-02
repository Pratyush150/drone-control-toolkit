#!/usr/bin/env python3
"""Trajectory generation, and what it does to tracking error.

Three references to the same destination:

* a **step** -- asks for infinite velocity and infinite acceleration, so the
  actuator saturates and the resulting motion is decided by your saturation
  limits, not by your design;
* a **trapezoidal profile** -- fastest thing that respects a velocity and an
  acceleration limit, at the cost of discontinuous acceleration at the corners;
* a **minimum-jerk profile** -- smooth, with zero velocity *and* zero
  acceleration at both ends, so the reference is realisable from a standstill.

Each is fed to the same controller on the same plant. The measured numbers are
peak tracking error, whether the actuator saturated, and how long the whole
manoeuvre took -- which is the trade: min-jerk is smoothest, trapezoidal is
fastest, and the step is neither.

Then a multi-waypoint path is generated and checked against velocity and
acceleration limits, because a smooth-looking spline is not automatically a
flyable one.

Run: ``python3 examples/min_jerk_trajectory.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.metrics import iae
from dctk.pid import PID
from dctk.sim import SecondOrderPlant, simulate
from dctk.trajectory import (
    check_limits,
    min_jerk,
    min_jerk_duration,
    trapezoidal_profile,
    waypoint_path,
)

DT = 0.001
DISTANCE = 3.0
V_MAX = 1.2
A_MAX = 1.5
COMMAND_LIMIT = 6.0


def track(setpoint_fn, sim_time: float, label: str):
    plant = SecondOrderPlant.from_modal(wn=7.0, zeta=0.7, dc_gain=1.0)
    pid = PID(
        kp=9.0, ki=3.0, kd=1.2,
        output_limits=(-COMMAND_LIMIT, COMMAND_LIMIT),
        derivative_cutoff_hz=40.0,
    )
    result = simulate(plant, pid.update, duration=sim_time, dt=DT, setpoint=setpoint_fn)
    error = result.setpoint - result.y
    peak = float(np.max(np.abs(error)))
    sat = float(np.mean(np.abs(result.u) >= COMMAND_LIMIT - 1e-9))
    arrival_idx = np.flatnonzero(np.abs(result.y - DISTANCE) < 0.02 * DISTANCE)
    arrival = float(result.t[arrival_idx[0]]) if arrival_idx.size else float("nan")
    # IAE of the *tracking* error (reference minus output), not of the error
    # against the final target -- the latter would simply punish any profile
    # for being slower than a step, which is not what is being measured here.
    tracking_iae = iae(result.t, error, 0.0)
    return label, result, peak, sat, arrival, tracking_iae


def main() -> None:
    header("Trajectory generation -- step vs trapezoidal vs minimum jerk")

    mj_duration = min_jerk_duration(DISTANCE, max_velocity=V_MAX, max_acceleration=A_MAX)
    mj = min_jerk(0.0, DISTANCE, mj_duration, n=int(mj_duration / DT) + 1)
    trap = trapezoidal_profile(0.0, DISTANCE, V_MAX, A_MAX, dt=DT)

    print()
    table(
        [
            ("distance", f"{DISTANCE:.2f} m"),
            ("limits", f"v <= {V_MAX:.2f} m/s, a <= {A_MAX:.2f} m/s^2"),
            (
                "min-jerk duration",
                f"{mj_duration:.3f} s  (peak v {mj.peak_velocity:.3f} m/s, "
                f"peak a {mj.peak_acceleration:.3f} m/s^2)",
            ),
            (
                "trapezoidal duration",
                f"{trap.duration:.3f} s  (peak v {trap.peak_velocity:.3f} m/s, "
                f"peak a {trap.peak_acceleration:.3f} m/s^2)",
            ),
            (
                "min-jerk endpoint conditions",
                f"v(0)={mj.velocity[0]:.2e}, a(0)={mj.acceleration[0]:.2e}, "
                f"v(T)={mj.velocity[-1]:.2e}, a(T)={mj.acceleration[-1]:.2e}",
            ),
        ]
    )
    print(
        f"\nTrapezoidal is {100 * (1 - trap.duration / mj_duration):.1f} % faster for the "
        "same limits. It pays for that with\ndiscontinuous acceleration at three corners, "
        "which rings anything flexible."
    )

    sim_time = max(mj_duration, trap.duration) + 3.0
    runs = [
        track(lambda tt: DISTANCE, sim_time, "step"),
        track(lambda tt: float(trap.at(tt)[0]), sim_time, "trapezoidal"),
        track(lambda tt: float(mj.at(tt)[0]), sim_time, "minimum jerk"),
    ]

    print()
    for label, _r, peak, sat, arrival, err in runs:
        print(
            f"  {label:<14}  peak tracking error = {peak:6.4f} m   "
            f"command saturated = {100 * sat:5.1f} % of the run   "
            f"arrival = {arrival:5.3f} s   tracking IAE = {err:.4f}"
        )

    step_run, trap_run, mj_run = runs
    print()
    table(
        [
            (
                "peak tracking error, step vs min-jerk",
                f"{step_run[2]:.4f} m vs {mj_run[2]:.4f} m "
                f"({step_run[2] / mj_run[2]:.1f}x smaller)",
            ),
            (
                "actuator saturation",
                f"step {100 * step_run[3]:.1f} %, trapezoidal {100 * trap_run[3]:.1f} %, "
                f"min-jerk {100 * mj_run[3]:.1f} %",
            ),
            (
                "arrival time (within 2 % of target)",
                f"step {step_run[4]:.3f} s, trapezoidal {trap_run[4]:.3f} s, "
                f"min-jerk {mj_run[4]:.3f} s",
            ),
            (
                "tracking IAE",
                f"step {step_run[5]:.4f}, trapezoidal {trap_run[5]:.4f}, "
                f"min-jerk {mj_run[5]:.4f}",
            ),
        ]
    )
    print(
        "\nThe step arrives first here because the actuator has headroom and the plant is\n"
        "well damped -- on a real airframe that headroom is the margin you were saving\n"
        "for gusts, and a saturated loop is an open loop. The generated references keep\n"
        "the tracking error an order of magnitude smaller, which is the regime the\n"
        "linear tune was designed for."
    )

    # --- waypoint path -------------------------------------------------------
    waypoints = np.array([[0.0, 0.0], [3.0, 2.0], [7.0, 1.5], [10.0, 4.0], [13.0, 3.0]])
    path = waypoint_path(waypoints, average_speed=2.0, n=1200)
    limits = check_limits(path, max_velocity=4.0, max_acceleration=3.0)
    print()
    table(
        [
            ("waypoints", f"{waypoints.shape[0]} points, {path.duration:.2f} s of path"),
            ("peak path speed", f"{limits['peak_velocity']:.3f} m/s (limit 4.00)"),
            ("peak path acceleration", f"{limits['peak_acceleration']:.3f} m/s^2 (limit 3.00)"),
            ("velocity limit respected", str(limits["velocity_ok"])),
            ("acceleration limit respected", str(limits["acceleration_ok"])),
        ]
    )
    print(
        "\ncheck_limits uses the vector norm across axes, not the per-axis maximum: a\n"
        "path at 0.9 of the limit on x and on y at the same instant is asking for 1.27\n"
        "of the limit, and a per-axis check would pass it."
    )

    # --- plots ---------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = figure(2, 2, figsize=(13, 9))
    ax1.plot(mj.t, mj.position, label="minimum jerk")
    ax1.plot(trap.t, trap.position, label="trapezoidal")
    ax1.axhline(DISTANCE, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("position (m)")
    ax1.set_xlabel("time (s)")
    ax1.set_title("Reference position")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(mj.t, mj.acceleration, label="minimum jerk")
    ax2.plot(trap.t, trap.acceleration, label="trapezoidal")
    ax2.set_ylabel("acceleration (m/s^2)")
    ax2.set_xlabel("time (s)")
    ax2.set_title("Reference acceleration -- trapezoidal has three corners")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    for label, result, peak, _sat, _arr, _err in runs:
        ax3.plot(result.t, result.setpoint - result.y, lw=1.0,
                 label=f"{label} (peak {peak:.4f} m)")
    ax3.set_ylabel("tracking error (m)")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Tracking error: a feasible reference keeps the loop linear")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4.plot(path.position[:, 0], path.position[:, 1], lw=1.5, label="cubic spline path")
    ax4.plot(waypoints[:, 0], waypoints[:, 1], "o--", color="k", lw=0.8, ms=6,
             label="waypoints")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_title(
        f"Waypoint path -- peak {limits['peak_velocity']:.2f} m/s, "
        f"{limits['peak_acceleration']:.2f} m/s^2"
    )
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.axis("equal")
    save(fig, "min_jerk_trajectory.png")


if __name__ == "__main__":
    main()
