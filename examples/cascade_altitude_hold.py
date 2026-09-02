#!/usr/bin/env python3
"""Cascaded altitude hold against a wind gust, with and without the inner loop.

Two points are made here.

1. **Cascade beats single-loop.** The same vertical plant is controlled twice:
   once with a single position PID going straight to thrust, and once with the
   altitude -> climb-rate cascade. The cascade rejects a gust better because
   the inner loop sees the disturbance as a climb-rate error immediately,
   instead of waiting for it to become an altitude error.

2. **The rate limit lives between the loops.** The cascade's outer loop cannot
   command a climb rate above ``max_climb_rate`` no matter how large the
   altitude error is. That bound is in m/s, a unit you can reason about, rather
   than in normalised thrust.

Run: ``python3 examples/cascade_altitude_hold.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.cascade import MultirotorCascade
from dctk.metrics import iae, step_metrics
from dctk.pid import PID
from dctk.sim import PointMassQuadrotor2D, WindGust

DT = 0.004
DURATION = 30.0
TARGET_ALT = 5.0
GUST_START = 15.0
GUST_NEWTONS = -4.0  # sustained downdraft


def simulate_vertical(controller, label: str):
    """Run the vertical axis. ``controller(z_sp, z, vz, dt) -> normalised thrust``."""
    plant = PointMassQuadrotor2D(mass=1.2, drag_z=0.25)
    hover_n = plant.hover_thrust()
    gust = WindGust(kind="step", amplitude=GUST_NEWTONS, t_start=GUST_START)

    n = int(DURATION / DT) + 1
    t = np.arange(n) * DT
    z = np.zeros(n)
    vz = np.zeros(n)
    u = np.zeros(n)
    for i, ti in enumerate(t):
        z[i] = plant.x[1]
        vz[i] = plant.x[3]
        u[i] = controller(TARGET_ALT, z[i], vz[i], DT)
        # Normalised thrust: 0.5 is hover, so full scale is twice hover thrust.
        plant.step(u[i] * 2.0 * hover_n, DT, (0.0, gust.value(ti, DT)))
    return label, t, z, vz, u


def main() -> None:
    header("Cascaded altitude hold -- takeoff to 5 m, then a sustained downdraft")

    # --- single loop: altitude error straight to thrust ---------------------
    single = PID(
        kp=0.20, ki=0.06, kd=0.22, kff=0.0,
        output_limits=(-0.5, 0.5), derivative_cutoff_hz=8.0,
    )

    def single_loop(z_sp, z, _vz, dt):
        return float(np.clip(single.update(z_sp, z, dt) + 0.5, 0.0, 1.0))

    # --- cascade: altitude -> climb rate -> thrust --------------------------
    mc = MultirotorCascade(hover_thrust=0.5, max_thrust=1.0, max_climb_rate=3.0)

    def cascade_loop(z_sp, z, vz, dt):
        return mc.update_altitude(z_sp, z, vz, dt)

    runs = [
        simulate_vertical(single_loop, "single loop"),
        simulate_vertical(cascade_loop, "cascade"),
    ]

    print()
    rows = []
    for label, t, z, vz, _u in runs:
        climb = t < GUST_START
        m = step_metrics(t[climb], z[climb], TARGET_ALT)
        after = t >= GUST_START
        sag = TARGET_ALT - float(np.min(z[after]))
        recovered = float(np.abs(z[-1] - TARGET_ALT))
        gust_iae = iae(t[after], z[after], TARGET_ALT)
        rows.append((label, m, sag, recovered, gust_iae, float(np.max(np.abs(vz)))))
        print(m.summary(label))

    print()
    table(
        [
            (
                "peak altitude sag under gust",
                " / ".join(f"{r[0]}: {r[2]:.3f} m" for r in rows),
            ),
            (
                "residual error at t=30 s",
                " / ".join(f"{r[0]}: {r[3]:.4f} m" for r in rows),
            ),
            (
                "IAE over the gust window",
                " / ".join(f"{r[0]}: {r[4]:.3f}" for r in rows),
            ),
            (
                "peak climb rate commanded",
                " / ".join(f"{r[0]}: {r[5]:.2f} m/s" for r in rows),
            ),
        ]
    )
    improvement = rows[0][4] / rows[1][4] if rows[1][4] > 0 else float("nan")
    print(f"\nCascade IAE over the gust window is {improvement:.2f}x better than the single loop.")
    print(
        "The cascade's climb-rate loop reacts to the downdraft as soon as vertical\n"
        "velocity changes, which is one inner-loop time constant before the altitude\n"
        "error the single loop is waiting for even exists."
    )

    fig, (ax1, ax2, ax3) = figure(3, 1, figsize=(10, 10), sharex=True)
    for label, t, z, vz, u in runs:
        ax1.plot(t, z, label=label)
        ax2.plot(t, vz, label=label)
        ax3.plot(t, u, label=label)
    ax1.axhline(TARGET_ALT, color="k", ls="--", lw=0.8, label="setpoint")
    ax1.axvline(GUST_START, color="r", ls=":", lw=1.0, label="downdraft starts")
    ax1.set_ylabel("altitude (m)")
    ax1.set_title("Altitude hold: single loop vs cascade")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(3.0, color="k", ls=":", lw=0.8)
    ax2.axhline(-3.0, color="k", ls=":", lw=0.8)
    ax2.set_ylabel("climb rate (m/s)")
    ax2.set_title("Climb rate -- the cascade's inner setpoint is limited to +/-3 m/s")
    ax2.grid(alpha=0.3)
    ax3.set_ylabel("normalised thrust")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Thrust command (0.5 = hover)")
    ax3.grid(alpha=0.3)
    save(fig, "cascade_altitude_hold.png")


if __name__ == "__main__":
    main()
