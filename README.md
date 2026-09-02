# drone-control-toolkit

Control and estimation building blocks for small unmanned aircraft. PID that
survives contact with hardware, cascaded loops, discrete LQR, complementary /
Madgwick / Kalman / EKF attitude estimation, motor mixing with correct
saturation behaviour, trajectory generation, and a simulation harness that
injects the defects real vehicles actually have.

**numpy only.** matplotlib is used by the example scripts and imported lazily,
so the library imports and the tests run on a headless machine with no plotting
stack.

---

## The problem this solves

Most control code you find online is correct and useless. The PID is three
lines of algebra that work perfectly against `y' = -y + u` and fall apart the
first time a motor saturates, a setpoint steps, or a gyro is bolted to a
vibrating airframe. The gap between "the maths is right" and "the aircraft
flies" is entirely made of things that are missing from the three lines:

- the integrator that keeps winding while the actuator is already at its limit,
  and then holds the loop open for the whole time it takes to unwind;
- the derivative kick that slams the actuator on every setpoint step;
- the D term that is pure noise because differentiation multiplies amplitude by
  frequency and your gyro carries 300 Hz of blade-pass content;
- the 15 ms of transport delay that eats 30 degrees of phase margin without
  showing up in a gain plot at all;
- the mixer that clips per-motor and quietly turns a roll command into a
  roll-plus-yaw command exactly when you are asking for it hardest.

Every module here handles one of those, and the docstring says which failure it
prevents. The `sim` module then lets you *demonstrate* it: controllers are
exercised against plants with sensor noise, actuator latency, motor lag,
quantisation and wind, and `metrics` turns the result into numbers so claims
can be checked rather than asserted.

---

## Quickstart

```bash
git clone https://github.com/Pratyush150/drone-control-toolkit
cd drone-control-toolkit
pip install -r requirements.txt

python3 -m pytest -q          # 173 tests, ~45 s, no network, no display needed
python3 examples/pid_tuning_comparison.py
```

Use it from source (no install needed):

```python
import sys; sys.path.insert(0, "src")

from dctk import PID, AntiWindup, FirstOrderPlant, simulate, step_metrics

pid = PID(
    kp=2.0, ki=1.5, kd=0.1,
    output_limits=(-1.0, 1.0),        # a motor cannot do 130 % thrust
    anti_windup=AntiWindup.BACK_CALCULATION,
    derivative_on_measurement=True,   # no kick on a setpoint step
    derivative_cutoff_hz=30.0,        # raw D on a real IMU is unusable
)

result = simulate(FirstOrderPlant(tau=0.5), pid.update, duration=5.0, dt=0.002, setpoint=1.0)
print(step_metrics(result.t, result.y, 1.0).summary("first order"))
```

Or install it:

```bash
pip install -e .
```

---

## Modules

| Module | Real-world problem it addresses |
|---|---|
| `dctk.pid` | The integrator that winds up while the actuator is saturated; the derivative kick on a setpoint step; raw D on a noisy IMU; the output step when you change gains in flight or hand over from manual; loop rate jitter. |
| `dctk.cascade` | Rate limits belong *between* loops in engineering units, not as an abstract output clamp. Bandwidth separation, and why tuning outward before the inner loop is finished is fitting noise. |
| `dctk.lqr` | Multi-state plants where inventing a cascade per state pair stops being practical. Discrete Riccati iteration implemented directly (no scipy), controllability check, ZOH discretisation, Bryson's rule so `Q` and `R` have units. |
| `dctk.filters` | Fusing a gyro that drifts with an accelerometer that lies under acceleration; removing blade-pass vibration without spending all your phase margin; why a fixed notch fails across the throttle range. |
| `dctk.estimator` | Attitude from an IMU with the gyro bias as an estimated state, Joseph-form covariance so `P` stays symmetric PSD over millions of samples, accelerometer gating during manoeuvres, and NIS output so you can tell a diverging filter from a healthy one. |
| `dctk.mixer` | Control allocation for quad X / quad plus / hexa X / quadplane VTOL, with saturation that sacrifices thrust to preserve attitude direction -- the choice that keeps the aircraft upright. |
| `dctk.trajectory` | A step setpoint asks for infinite acceleration; the loop saturates and the motion is decided by your limits, not your design. Minimum-jerk, trapezoidal and spline references with limit checking. |
| `dctk.sim` | Plants plus the defects that break tunes: sensor noise with a walking bias, transport delay, motor lag, quantisation, wind gusts and turbulence. Deterministic given a seed. |
| `dctk.metrics` | Rise time, settling time, overshoot, steady-state error, IAE/ISE/ITAE. Every claim in this README is one of these numbers. |
| `dctk.tuning` | Ziegler-Nichols and relay auto-tuning, with an honest account of why ZN output is a starting point in the right order of magnitude and not a tune you would fly. |

---

## What the examples actually measure

Every number below is printed by the script named next to it. Run them and you
will get the same numbers -- the simulations are deterministic.

### `pid_tuning_comparison.py`
Second-order plant, noisy sensor, 15 ms transport delay, 25 ms motor lag.

| Tune | Overshoot | ITAE | Command jitter |
|---|---|---|---|
| P only | 1.4 % | 8.2199 | 0.0172 |
| PI | 49.8 % | 6.0643 | 0.0176 |
| PID, raw D | 0.0 % | 0.6732 | 3.7299 |
| PID, D filtered at 15 Hz | 0.0 % | 0.6681 | 0.2194 |

P alone leaves a steady-state error of **+0.4578**. Filtering the derivative
costs nothing measurable in tracking (ITAE 0.6732 -> 0.6681) and cuts command
jitter by **17x**.

### `anti_windup_demo.py`
Setpoint held at an unreachable value until the actuator has been saturated for
12 s, then dropped to something achievable.

| Strategy | Peak integrator | Time to reach the new setpoint | Saturated after the switch |
|---|---|---|---|
| None | 51.99 | 54.07 s | 63.5 % of the window |
| Clamping | 0.63 | 0.74 s | 0.0 % |
| Back-calculation | 0.63 | 0.85 s | 0.0 % |

Without protection the loop spends nearly two thirds of the recovery window
still pinned at its limit -- open loop.

### `cascade_altitude_hold.py`
Climb to 5 m, then a sustained 4 N downdraft.

| Controller | Overshoot | Settling | Peak sag under gust | IAE over the gust window |
|---|---|---|---|---|
| Single loop (altitude -> thrust) | 18.6 % | 9.696 s | 0.646 m | 2.840 |
| Cascade (altitude -> climb rate -> thrust) | 0.0 % | 2.612 s | 0.356 m | 1.178 |

**2.41x** better gust rejection, because the inner loop sees the disturbance as
a climb-rate error before it has become an altitude error.

### `complementary_vs_kalman.py`
Synthetic IMU: 2 deg/s gyro bias, white noise on both sensors, and a 4 s window
of 6 m/s² lateral acceleration that corrupts the accelerometer.

| Estimator | RMS error (quiet) | RMS during the manoeuvre | Final error |
|---|---|---|---|
| Accelerometer only | 0.471 deg | 33.538 deg | -0.080 deg |
| Gyro integration only | 100.470 deg | 64.041 deg | +119.801 deg |
| Complementary, tau = 1 s | 1.998 deg | 28.480 deg | +2.031 deg |
| Attitude EKF | 0.058 deg | 20.820 deg | +0.005 deg |

The complementary filter's steady offset is `bias * tau`: **predicted 2.000
deg, measured 1.998 deg**. The EKF estimates the bias at 2.025 deg/s against a
true 2.000 and drives its own offset to -0.049 deg, and its magnitude gate
rejects 2.9 % of accelerometer updates -- the manoeuvre window.

### `lqr_vs_pid.py`
1-DOF pitch axis (double integrator plus damping), 8 ms delay, 30 ms motor lag,
0.25 N.m torque limit. The PD gains are the best found by a grid search on
ITAE, so the comparison is not against a straw man.

| Controller | Rise | Settling | Overshoot | ITAE | Peak torque |
|---|---|---|---|---|---|
| PD, angle only | 0.309 s | 0.996 s | 8.8 % | 0.0244 | 0.192 N.m |
| LQR, full state | 0.341 s | 0.978 s | 4.6 % | 0.0215 | 0.199 N.m |

Riccati converged in 2190 iterations to a residual of 4.55e-13; closed-loop
spectral radius 0.9924. The honest reading: on one decoupled axis a properly
tuned PD is close, and LQR's real advantages are that it is handed the rate as
a state rather than differentiating the angle, and that Bryson's rule turns
"5 degrees of error and 0.05 N.m are a lot" into weights directly.

### `vibration_notch_filter.py`
Blade-pass tone sweeping 200 -> 400 Hz with throttle, on top of a 4 Hz command.
Phase lag is quoted at a 20 Hz rate-loop crossover.

| Chain | Residual vs command | D-term RMS | Phase lag @ 20 Hz |
|---|---|---|---|
| Raw gyro | 0.2481 rad/s | 9.4761 (26.7x ideal) | 0.00 deg |
| 150 Hz low-pass | 0.0666 rad/s | 2.1324 (6.0x) | 10.82 deg |
| 40 Hz low-pass | 0.1004 rad/s | 0.3875 (1.1x) | 43.30 deg |
| Fixed notch @200 Hz + 150 Hz LP | 0.0665 rad/s | 2.1180 (6.0x) | 11.20 deg |
| RPM-tracked notch + 150 Hz LP | 0.0278 rad/s | 0.3671 (1.0x) | 11.20 deg |

The 40 Hz low-pass and the tracked notch clean the signal about equally well.
The low-pass costs **32.1 degrees more phase margin** for the same result. The
fixed notch is correct only while the tone sits inside it and pays its phase
cost for the entire flight regardless.

### `min_jerk_trajectory.py`
3 m move under a 1.2 m/s and 1.5 m/s² limit.

| Reference | Peak tracking error | Actuator saturated | Arrival | Tracking IAE |
|---|---|---|---|---|
| Step | 3.0000 m | 0.9 % of the run | 4.415 s | 1.1715 |
| Trapezoidal | 0.2782 m | 0.0 % | 5.326 s | 0.9088 |
| Minimum jerk | 0.2522 m | 0.0 % | 6.053 s | 0.8855 |

Minimum-jerk endpoint velocity and acceleration are zero to floating-point
precision. Trapezoidal is 29.6 % faster than minimum jerk for the same limits,
paid for with discontinuous acceleration at three corners.

Each script writes a PNG to `examples/output/` (gitignored) and prints the
metrics above.

---

## Architecture

```
src/dctk/
  pid.py          PID: anti-windup, D-on-measurement, D filter, FF, bumpless transfer
  cascade.py      LoopSpec / CascadeController / MultirotorCascade
  lqr.py          expm, c2d, controllability, Bryson's rule, dlqr (Riccati iteration)
  filters.py      complementary, Madgwick, Kalman 1D/nD, biquad LP, notch, quaternions
  estimator.py    AttitudeEKF: quaternion + gyro bias, Joseph form, NIS diagnostics
  mixer.py        quad X / quad plus / hexa X / quadplane VTOL, pinv allocation, desat
  trajectory.py   min-jerk, trapezoidal, quintic segments, cubic spline, limit checks
  sim.py          plants + SensorNoise / Quantiser / ActuatorDelay / MotorLag / WindGust
  metrics.py      rise, settling, overshoot, ss error, IAE / ISE / ITAE
  tuning.py       Ziegler-Nichols tables, relay auto-tune, ultimate-gain sweep
tests/            11 files; behaviour, not smoke tests
examples/         7 runnable scripts, each printing metrics and writing a PNG
docs/             CONTROL_NOTES.md -- field notes on tuning, diagnosis and filtering
```

Design conventions used throughout:

- **Sample-by-sample.** Every filter and controller has an `update(...)` that
  advances one step, so it can run in a real-time loop and not only over a
  stored array. Offline `filt(array)` helpers exist for tests and plots.
- **`dt` is a parameter, never a constant.** Real loops jitter and occasionally
  drop a frame; hard-coding the nominal rate silently mis-scales I and D
  exactly when the loop is already struggling.
- **State is explicit and inspectable.** `PID.state`, `AttitudeEKF.diagnostics`,
  `Mixer.last_saturation`, `CascadeController.errors`. These exist to be
  logged; most control bugs are diagnosed from a log, not from a debugger.
- **Deterministic given a seed.** Every noise source owns a private
  `numpy.random.Generator`.

---

## Testing

```bash
python3 -m pytest -q
```

173 tests across 11 files, roughly 360 assertions, no network, no display, no
plotting. They prove behaviour rather than absence of exceptions:

- anti-windup actually bounds the integrator under saturation, and the
  unprotected case is asserted to run away so the comparison is real;
- derivative-on-measurement produces exactly zero output on a setpoint step,
  while derivative-on-error produces `kd * dsp / dt`;
- the LQR gain places every closed-loop eigenvalue of a genuinely unstable
  plant (an inverted pendulum discretised at 100 Hz) inside the unit circle,
  and the returned `P` is checked to satisfy the discrete Riccati equation to
  1e-8;
- the complementary filter converges under a constant gyro bias to
  `bias * tau`, the value the theory predicts, not merely "to something";
- the EKF's covariance is asserted symmetric and PSD at **every step** of a
  6000-step run with noise;
- 3000 random demands per airframe geometry, and no motor command ever leaves
  `[idle, 1]`; under saturation the achieved torque direction stays within
  1 degree of the demanded direction;
- minimum-jerk endpoints have zero velocity and acceleration, and peak velocity
  and acceleration match the closed forms `1.875 d/T` and `10 d/(sqrt(3) T^2)`;
- metrics are checked against hand-computable answers: first-order rise time is
  `tau ln 9`, 2 % settling is `tau ln 50`, second-order overshoot at
  `zeta = 0.5` is `exp(-pi zeta / sqrt(1-zeta^2))`.

There is also an integration file that closes real loops across module
boundaries -- estimator into controller, cascade into mixer -- because that is
where sign conventions that are self-consistent within one module go wrong.

---

## What this is and is not

**It is:**

- a readable, tested reference implementation of the control and estimation
  pieces a small UAV needs, with the hardware-driven details included rather
  than elided;
- a simulation harness for demonstrating that a controller survives latency,
  noise and saturation before it goes anywhere near an airframe;
- a place to look up why a particular piece of flight-stack behaviour exists.

**It is not:**

- a flight controller. There is no hardware abstraction, no scheduler, no
  failsafe logic, no arming state machine. Use PX4 or ArduPilot; this is for
  understanding, prototyping and offline analysis.
- a replacement for `scipy` or `python-control`. The LQR solver here is an
  iterative Riccati solve, chosen so the module has no heavy dependency and so
  the algorithm is visible. For large or ill-conditioned problems use a proper
  Schur-decomposition solver.
- a full 6-DOF simulator. `sim` provides low-order plants deliberately: the
  point is to expose controllers to *defects*, not to model aerodynamics.
  Gazebo, AirSim or PX4 SITL are the right tools for vehicle dynamics.
- validated against a specific airframe. The gains in `MultirotorCascade` are a
  plausible starting structure with sensible limits, not a tune for your
  aircraft. Read `docs/CONTROL_NOTES.md` before putting any of it near
  hardware.

**Known limitations:**

- The EKF's process noise model for the quaternion block is a simplification;
  it is adequate and stable but a rigorously derived `Q` from the gyro's
  Allan-variance parameters would be better.
- The magnetometer update assumes calibrated input. Hard-iron and soft-iron
  calibration are not implemented and are not optional on a real vehicle.
- `MadgwickAHRS` has no covariance, so it cannot report whether it is
  diverging. That is the reason `estimator.AttitudeEKF` exists.
- The VTOL mixer models a quadplane in hover only. Transition scheduling --
  blending lift and wing authority as airspeed builds -- is out of scope.
- Trajectory generation is kinematic. It respects velocity and acceleration
  limits but knows nothing about thrust-to-weight, wind, or attitude limits.

---

## Related repositories

- **[flight-log-analyzer](https://github.com/Pratyush150/flight-log-analyzer)**
  -- pull the real vibration spectrum, EKF innovations, mixer saturation and
  mode timeline out of a PX4 ULog or ArduPilot log. Use it to find out what you
  are actually filtering before you choose a notch frequency here.
- **[px4-mavlink-companion](https://github.com/Pratyush150/px4-mavlink-companion)**
  -- MAVLink bridge and offboard control between a flight controller and a
  companion computer. That link is where the transport delay modelled by
  `sim.ActuatorDelay` actually comes from.
- **[ros2-drone-bringup](https://github.com/Pratyush150/ros2-drone-bringup)**
  -- ROS 2 bringup and SITL, for testing a control change without risking
  hardware.

---

## License

MIT. See [LICENSE](LICENSE).
