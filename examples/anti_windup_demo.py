#!/usr/bin/env python3
"""Integral windup, and the two ways to stop it.

Scenario: an altitude-hold-shaped loop is commanded to a setpoint it cannot
reach, because the actuator saturates long before it gets there. This is not a
contrived situation -- it is a climb command at full throttle in a downdraft,
or a heavily loaded aircraft asked for a rate it cannot achieve.

While the output is pinned at the limit the error stays large and the
integrator keeps accumulating. Nothing is visibly wrong: the aircraft is doing
its best. Then the setpoint comes back to something achievable, and the
integrator has to *unwind* before the output can leave the limit. For the
duration of that unwind the controller is effectively open loop, which is why
the vehicle sails straight past the new setpoint.

Run: ``python3 examples/anti_windup_demo.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.pid import PID, AntiWindup
from dctk.sim import FirstOrderPlant, simulate

DT = 0.002
DURATION = 90.0
HIGH_SETPOINT = 6.0  # unreachable: the actuator saturates at 1.0
LOW_SETPOINT = 1.0  # comfortably reachable
SWITCH_TIME = 12.0


def setpoint(t: float) -> float:
    return HIGH_SETPOINT if t < SWITCH_TIME else LOW_SETPOINT


def run(strategy: AntiWindup, label: str):
    plant = FirstOrderPlant(tau=2.0, gain=2.0)
    pid = PID(
        kp=1.0,
        ki=0.8,
        kd=0.0,
        output_limits=(-1.0, 1.0),
        anti_windup=strategy,
        tracking_time_constant=1.0,
        derivative_cutoff_hz=None,
    )
    integrals = []

    def controller(sp, meas, dt):
        u = pid.update(sp, meas, dt)
        integrals.append(pid.state.integral)
        return u

    result = simulate(plant, controller, duration=DURATION, dt=DT, setpoint=setpoint)
    integrals = np.asarray(integrals)

    mask = result.t >= SWITCH_TIME
    t_after = result.t[mask]
    y_after = result.y[mask]
    u_after = result.u[mask]

    # Time from the setpoint drop until the output first arrives at the new
    # setpoint. This is the number that matters: while the integrator unwinds
    # the controller is saturated and therefore open loop.
    arrived = np.flatnonzero(y_after <= LOW_SETPOINT * 1.02)
    arrival = float(t_after[arrived[0]] - SWITCH_TIME) if arrived.size else float("nan")

    # Fraction of the post-switch window spent with the command still pinned at
    # a limit -- the open-loop time.
    saturated = float(np.mean(np.abs(u_after) >= 1.0 - 1e-9))

    undershoot = float(max(0.0, LOW_SETPOINT - np.min(y_after)))
    return label, result, integrals, arrival, saturated, undershoot


def main() -> None:
    header("Integral anti-windup -- unreachable setpoint, then a reachable one")
    print(
        f"\nSetpoint is {HIGH_SETPOINT} (unreachable, output saturates at 1.0) until "
        f"t={SWITCH_TIME:.0f} s,\nthen drops to {LOW_SETPOINT}. Watch the integrator."
    )

    runs = [
        run(AntiWindup.NONE, "no anti-windup"),
        run(AntiWindup.CLAMP, "clamping"),
        run(AntiWindup.BACK_CALCULATION, "back-calculation"),
    ]

    print()
    rows = []
    for label, _result, integrals, arrival, saturated, undershoot in runs:
        peak_i = float(np.max(np.abs(integrals)))
        rows.append(
            (
                label,
                f"peak |integral| = {peak_i:8.3f}   "
                f"time to reach new setpoint = {arrival:6.2f} s   "
                f"saturated after switch = {100 * saturated:5.1f} %   "
                f"undershoot = {undershoot:.3f}",
            )
        )
    table(rows)

    none_peak = float(np.max(np.abs(runs[0][2])))
    clamp_peak = float(np.max(np.abs(runs[1][2])))
    back_peak = float(np.max(np.abs(runs[2][2])))
    print()
    table(
        [
            ("integrator reduction, clamping", f"{none_peak / clamp_peak:.0f}x"),
            ("integrator reduction, back-calculation", f"{none_peak / back_peak:.0f}x"),
            (
                "time to reach the new setpoint",
                f"{runs[0][3]:.2f} s (none) -> {runs[1][3]:.2f} s (clamp), "
                f"{runs[2][3]:.2f} s (back-calc)",
            ),
            (
                "open-loop time after the switch",
                f"{100 * runs[0][4]:.1f} % -> {100 * runs[1][4]:.1f} % / "
                f"{100 * runs[2][4]:.1f} %",
            ),
        ]
    )
    print(
        "\nClamping freezes the integrator the moment it would push further into\n"
        "saturation. Back-calculation bleeds it off in proportion to how far past the\n"
        "limit the unsaturated command is, which unwinds more smoothly at the cost of\n"
        "one extra constant (the tracking time Tt)."
    )

    fig, (ax1, ax2, ax3) = figure(3, 1, figsize=(10, 10), sharex=True)
    for label, result, integrals, _, _, _ in runs:
        ax1.plot(result.t, result.y, label=label)
        ax2.plot(result.t, integrals, label=label)
        ax3.plot(result.t, result.u, label=label)
    ax1.plot(runs[0][1].t, runs[0][1].setpoint, "k--", lw=0.8, label="setpoint")
    ax1.set_ylabel("output")
    ax1.set_title("Plant output -- the wound-up loop sails past the new setpoint")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("integrator state")
    ax2.set_title("Integrator: unbounded without protection")
    ax2.grid(alpha=0.3)
    ax3.axhline(1.0, color="k", ls=":", lw=0.8)
    ax3.axhline(-1.0, color="k", ls=":", lw=0.8)
    ax3.set_ylabel("command")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Command, with the saturation limits")
    ax3.grid(alpha=0.3)
    save(fig, "anti_windup_demo.png")


if __name__ == "__main__":
    main()
