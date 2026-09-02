#!/usr/bin/env python3
"""LQR against a hand-tuned PID on the same unstable plant.

The plant is a 1-DOF pitch axis: a double integrator with a little damping,
which is what a multirotor axis looks like once the motors are included. It is
not open-loop stable in attitude, so this is not a case where "do nothing" is
an option.

Two controllers, one plant, one metric set:

* **PID** on the angle, hand tuned, with a filtered derivative.
* **LQR** on the full state ``[theta, theta_dot]``, which is the honest
  advantage: it uses the rate measurement as a state rather than
  differentiating the angle. Weights come from Bryson's rule, so the tuning
  knobs have units -- "5 degrees of error is a lot, 0.05 N.m of torque is a
  lot" -- instead of being three unitless numbers.

The comparison is deliberately fair: both see the same actuator limit, the same
sample rate, and the same 8 ms of transport delay.

Run: ``python3 examples/lqr_vs_pid.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.lqr import brysons_rule, c2d, dlqr, is_controllable
from dctk.metrics import step_metrics
from dctk.pid import PID
from dctk.sim import ActuatorDelay, MotorLag, PitchPlant

DT = 0.002
DURATION = 4.0
TARGET = np.deg2rad(20.0)
TORQUE_LIMIT = 0.25
INERTIA = 0.02
DAMPING = 0.004
DELAY_S = 0.008
MOTOR_TAU = 0.03


def run(controller, label: str):
    plant = PitchPlant(inertia=INERTIA, damping=DAMPING)
    delay = ActuatorDelay(DELAY_S, DT)
    lag = MotorLag(MOTOR_TAU)
    n = int(DURATION / DT) + 1
    t = np.arange(n) * DT
    theta = np.zeros(n)
    rate = np.zeros(n)
    u = np.zeros(n)
    for i in range(n):
        theta[i] = plant.x[0]
        rate[i] = plant.x[1]
        cmd = float(np.clip(controller(TARGET, theta[i], rate[i], DT), -TORQUE_LIMIT, TORQUE_LIMIT))
        u[i] = cmd
        plant.step(lag.apply(delay.apply(cmd), DT), DT)
    return label, t, theta, rate, u


def main() -> None:
    header("LQR vs PID on a 1-DOF pitch axis (double integrator + damping)")

    # --- design the LQR -----------------------------------------------------
    A = np.array([[0.0, 1.0], [0.0, -DAMPING / INERTIA]])
    B = np.array([[0.0], [1.0 / INERTIA]])
    print(f"\ncontrollable: {is_controllable(A, B)}")
    Ad, Bd = c2d(A, B, DT)
    Q, R = brysons_rule(
        max_state_deviation=[np.deg2rad(5.0), np.deg2rad(180.0)],
        max_control_effort=[0.05],
        rho=1.0,
    )
    res = dlqr(Ad, Bd, Q, R)
    print(
        f"Riccati converged in {res.iterations} iterations "
        f"(residual {res.residual:.2e}); K = [{res.K[0, 0]:.4f}, {res.K[0, 1]:.4f}]"
    )
    print(
        "closed-loop discrete eigenvalues: "
        + ", ".join(f"{abs(e):.4f}" for e in res.eigenvalues)
        + f"  -> stable: {res.is_stable}"
    )

    def lqr_controller(sp, th, rt, _dt):
        x = np.array([th - sp, rt])
        return float((-res.K @ x)[0])

    # Best tune found by a grid search over (kp, ki, kd) on ITAE for this exact
    # plant, so the comparison is against a properly tuned PID rather than a
    # straw man. Note ki came out at zero: on a double integrator behind a
    # transport delay, integral action costs more phase than it buys, and the
    # plant has no steady-state error to remove for a constant reference
    # anyway.
    pid = PID(kp=0.55, ki=0.0, kd=0.13, output_limits=(-TORQUE_LIMIT, TORQUE_LIMIT),
              derivative_cutoff_hz=30.0)

    def pid_controller(sp, th, _rt, dt):
        return pid.update(sp, th, dt)

    runs = [
        run(pid_controller, "PD (angle only)"),
        run(lqr_controller, "LQR (full state)"),
    ]

    print()
    metrics = []
    for label, t, theta, _rate, _u in runs:
        m = step_metrics(t, theta, TARGET)
        metrics.append(m)
        print(m.summary(label))

    print()
    table(
        [
            ("LQR gain K", f"[{res.K[0, 0]:.4f}, {res.K[0, 1]:.4f}] (N.m per rad, per rad/s)"),
            ("spectral radius", f"{res.spectral_radius:.4f} (must be < 1)"),
            (
                "overshoot",
                f"PD {metrics[0].overshoot_pct:.1f} % vs LQR {metrics[1].overshoot_pct:.1f} %",
            ),
            ("ITAE", f"PD {metrics[0].itae:.4f} vs LQR {metrics[1].itae:.4f}"),
            ("rise time", f"PD {metrics[0].rise_time:.3f} s vs LQR {metrics[1].rise_time:.3f} s"),
            (
                "settling time",
                f"PD {metrics[0].settling_time:.3f} s vs LQR {metrics[1].settling_time:.3f} s",
            ),
            (
                "peak torque used",
                " / ".join(f"{r[0].split(' ')[0]} {np.max(np.abs(r[4])):.3f} N.m" for r in runs),
            ),
        ]
    )
    print(
        "\nHonest reading of this result: a properly tuned PD is close. The LQR's real\n"
        "advantages here are that it is handed the rate as a state instead of having to\n"
        "differentiate the angle, and that Bryson's rule turned 'how much error and how\n"
        "much torque are acceptable' into weights directly instead of into three\n"
        "unitless numbers found by search. Neither controller has integral action, so\n"
        "neither would hold against a constant disturbance -- a CG offset or a bent arm\n"
        "would leave a permanent tilt in both. Adding an integral state to the LQR is\n"
        "the standard fix and is not free: it is another state to weight."
    )
    print(
        "\nWhere LQR genuinely wins is multi-input, coupled plants -- a full 6-DOF\n"
        "attitude problem -- where inventing a cascade for every pair of states stops\n"
        "being practical. On one decoupled axis like this, use whichever you can tune."
    )

    fig, (ax1, ax2, ax3) = figure(3, 1, figsize=(10, 10), sharex=True)
    for (label, t, theta, rate, u), m in zip(runs, metrics):
        ax1.plot(t, np.rad2deg(theta), label=f"{label}  (ITAE {m.itae:.4f})")
        ax2.plot(t, np.rad2deg(rate), label=label)
        ax3.plot(t, u, label=label)
    ax1.axhline(np.rad2deg(TARGET), color="k", ls="--", lw=0.8, label="setpoint")
    ax1.set_ylabel("pitch (deg)")
    ax1.set_title("Attitude step -- 20 deg, 8 ms delay, 30 ms motor lag")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("pitch rate (deg/s)")
    ax2.grid(alpha=0.3)
    ax3.axhline(TORQUE_LIMIT, color="k", ls=":", lw=0.8)
    ax3.axhline(-TORQUE_LIMIT, color="k", ls=":", lw=0.8)
    ax3.set_ylabel("torque (N.m)")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Commanded torque, with the actuator limit")
    ax3.grid(alpha=0.3)
    save(fig, "lqr_vs_pid.png")


if __name__ == "__main__":
    main()
