# Control notes

Field notes, not theory. These are the things that cost time on real hardware
and that are hard to find written down in one place.

Everything here assumes a small multirotor with a modern flight stack (PX4,
ArduPilot or Betaflight). The physics is the same on any airframe; the numbers
scale with size.

---

## 1. Tuning a rate loop on a real multirotor, safely

The rate loop is the innermost loop and everything else is built on top of it.
Tune it first, tune it properly, and do not touch the outer loops until it is
finished.

### Before you arm

- **Props off, on the bench, first.** You can validate the sign of every axis,
  the D-term noise floor and your logging setup with the props off. If the
  motors buzz with props off, you have an electrical or resolution problem, not
  a vibration problem, and no amount of filtering will fix it.
- **Fix the mechanical problems first.** A bent shaft, a chipped prop or a
  loose arm produces vibration that no filter will remove without eating your
  phase margin. Balance props, check motor bell play, re-torque the arms.
  Tuning a mechanically bad aircraft is a way to spend an afternoon and end up
  with a worse tune than you started with.
- **Set your logging rate high enough.** You cannot diagnose a 60 Hz
  oscillation from a 50 Hz log. Log gyro and motor outputs at the loop rate for
  the tuning flights, then turn it back down.
- **Have a way to abort.** A mode switch back to a known-good tune, or a
  known-good rate on a second profile. "I'll just land it" is not an abort
  plan when the aircraft is oscillating.

### The procedure

1. **Start low.** Halve whatever gains you were going to start with. It is much
   faster to walk a gain up than to recover an aircraft that is oscillating.
2. **P first, I and D at zero (or near zero).** Hover, then make small, sharp
   stick inputs on one axis. Raise P until the aircraft feels crisp and then
   begins a fast oscillation after the input. Back off by about 30 %.
3. **Add D.** D damps the P-induced oscillation and lets you raise P further.
   Raise D until you hear the motors start to buzz or feel a high-frequency
   twitchiness -- that is the D term amplifying gyro noise, not the airframe.
   Back off by about 25 %. If you cannot get useful D before it buzzes, your
   problem is vibration, not the gain (see section 3).
4. **Re-raise P** now that D is damping it, then re-check D. One or two rounds
   of this converges.
5. **Add I last, and only as much as you need.** I fights a static imbalance --
   a heavy battery mounted off-centre, a bent arm, a slight motor mismatch. Too
   much I is a slow wallow after a stick input, and it is the term that winds
   up during a saturated manoeuvre.
6. **Test with a punch out and a hard flip.** Gains that are fine in a hover
   are often not fine when the motors are near their limits, because motor
   authority is not linear with throttle: at full throttle a motor cannot go
   any *further* up, so the loop's effective gain in that direction collapses.

### One axis at a time

Roll and pitch are usually close enough to share gains on a symmetric frame.
Yaw is different: it acts on prop drag torque rather than thrust differential,
so it is roughly an order of magnitude weaker and needs its own, much higher, P
and much lower D. Do not copy roll gains to yaw.

---

## 2. What the oscillation frequency tells you

The frequency of an oscillation identifies which loop and which term is wrong.
This is the single most useful diagnostic in the air, and it costs nothing --
just listen, or look at the gyro trace in a log.

| Frequency | What it looks like | Almost always |
|---|---|---|
| **> 100 Hz** | Audible buzz, motors get hot, no visible motion | D term amplifying gyro noise. Lower D, or fix the vibration/filtering. |
| **30-100 Hz** | High-pitched twitch, visible blur on video | D too high, or the D filter cutoff is too high for the noise floor. Sometimes a resonant frame mode. |
| **10-30 Hz** | Fast, visible shake; motors clearly working | Rate loop P too high. This is the classic "too much P" oscillation. |
| **3-10 Hz** | Slower, obvious wobble after a stick input | Attitude loop P too high, or the rate loop is too slow to serve it (bandwidth separation is broken). |
| **1-3 Hz** | Slow bounce or wallow, especially after a manoeuvre | Rate-loop I too high, or the velocity loop is too aggressive. |
| **< 1 Hz** | Slow drift, hunting around a position | Position loop too aggressive, or an integrator winding against a persistent disturbance. |

Two rules of thumb behind the table:

- **Fast oscillation = inner loop.** The frequency an unstable loop oscillates
  at is roughly its own crossover frequency. A rate loop crosses over at
  15-30 Hz on a 5-inch quad, so a 20 Hz oscillation is a rate-loop problem, not
  a position-loop problem, no matter what the position looks like.
- **P oscillates faster than I.** Raising P moves the crossover up; the
  oscillation it causes is at that frequency. Integral instability is a
  low-frequency phenomenon because the integrator's contribution rolls off with
  frequency.

If you cannot tell, halve the D gain and fly again. If the oscillation
frequency drops noticeably, it was a D/noise problem. If it does not change,
look at P.

---

## 3. Why prop vibration destroys the D term

Differentiation multiplies a signal's amplitude by its frequency. That is the
whole story, but it is worth writing down concretely.

Suppose your gyro carries 0.02 rad/s of blade-pass vibration at 300 Hz on top
of a command signal of 2 rad/s at 5 Hz. In the raw signal the vibration is 1 %
of the command -- invisible. After differentiation:

- command derivative: `2 * 2*pi*5` = 63 rad/s²
- vibration derivative: `0.02 * 2*pi*300` = 38 rad/s²

The vibration is now 60 % of the signal. Raise the vibration to 0.1 rad/s --
still only 5 % of the command in the raw signal, and entirely normal on a
worn prop -- and the D term is three times more vibration than command.

This is why:

- **Raw D on a real IMU is not "a bit noisy", it is unusable.** The D output is
  dominated by content you do not want, the motors are being commanded at
  hundreds of Hz, and they heat up. Motors that are hot after a flight with no
  aggressive manoeuvring are almost always a D-term noise problem.
- **There is a ceiling on useful D that has nothing to do with your gyro.**
  Past a point the motor's own time constant (20-40 ms on a 5-inch, over
  100 ms on a large multirotor) means it physically cannot follow the D
  command, and the loop starts fighting the motor's lag instead of the
  airframe.
- **Blade-pass frequency moves with throttle.** `f_bp = rpm/60 * n_blades`. A
  5-inch quad on 6S runs roughly 5 000 rpm at idle to 25 000+ rpm on a punch
  out, so with a two-blade prop the tone sweeps roughly 170 Hz to 830 Hz within
  a single throttle input. A notch set from a hover log is correct for hover
  and wrong everywhere else.

The fixes, in order of preference:

1. **Mechanical.** Balance the props, replace chipped ones, soft-mount the FC.
   Every dB of vibration you remove mechanically is a dB you do not have to pay
   for in phase lag.
2. **RPM-tracked notch.** Take motor RPM from bidirectional-DShot ESC
   telemetry, compute blade-pass per motor, retune the notch centre every loop.
   `dctk.filters.NotchFilter.retune()` exists for exactly this.
3. **Dynamic (FFT-driven) notch.** If you have no RPM telemetry, find the peak
   in a running spectrum and follow it. Works, but it is chasing rather than
   knowing.
4. **Low-pass.** Always needed as a backstop, but every Hz you take off the
   cutoff is phase margin gone (see section 5). Lowering the cutoff until the
   buzz goes away is the reflex, and it is how people end up with a tune that
   cannot be raised any further.

`examples/vibration_notch_filter.py` measures this trade: on a throttle sweep,
a 40 Hz low-pass and an RPM-tracked notch clean the signal about equally well,
and the low-pass costs about 32 degrees more phase at a 20 Hz crossover.

Cross-reference: **flight-log-analyzer** extracts the actual vibration spectrum
and the clipping/aliasing indicators from a PX4 ULog or ArduPilot log. Look at
the real spectrum before deciding where to put a filter. Guessing from the
sound of the aircraft is how people end up with four notches and no phase
margin.

---

## 4. Spotting a saturated integrator in a flight log

A wound-up integrator is invisible in the attitude plot -- the aircraft looks
like it is doing its best -- and it is one of the most common causes of "it
overshot on the way back". Here is how to find it.

**The signature:**

1. **Actuator output pinned at a limit** for a sustained period. On a
   multirotor look at the individual motor outputs, not just the collective:
   one motor at 100 % and another at idle means the mixer is saturated even if
   the throttle looks reasonable.
2. **A persistent error in the same direction** during that period. If the
   error changes sign the integrator is not winding, it is doing its job.
3. **The recovery is the tell.** When the demand comes back into range, the
   output stays pinned for a while *after* the error has already reversed. That
   delay is the integrator unwinding, and during it the loop is open. It is
   usually the largest overshoot in the whole log.

**Where to look, per stack:**

- PX4: `rate_ctrl_status` publishes the integrator state per axis directly.
  `actuator_outputs` and the mixer saturation flags in `actuator_controls`
  tell you when allocation ran out of authority.
- ArduPilot: `PIDR`/`PIDP`/`PIDY` messages log the P, I and D contributions
  separately. Plotting the I contribution against the total output is the
  fastest way to see windup.
- Betaflight blackbox: `axisI[n]` is logged directly, alongside `motor[n]`.

**What to do about it:**

- Turn on anti-windup, which `dctk.pid.PID` does by default. Clamping is
  cheap and adequate; back-calculation unwinds more smoothly at the cost of one
  extra constant.
- Bound the integrator explicitly as well (`integral_limits`). Anti-windup
  handles saturation; an explicit bound handles the case where a sensor fault
  produces a persistent large error that never saturates the output.
- Reset the integrator on mode changes and on re-arm. A stale integrator from
  the previous flight mode is the classic "it kicked on the first sample after
  re-enable".

`examples/anti_windup_demo.py` reproduces the whole failure: with the
protection off, the loop spends 63 % of the post-manoeuvre window still
saturated -- open loop -- and takes 54 seconds to reach the new setpoint
instead of 0.7.

---

## 5. When a filter's phase lag becomes instability

Every filter you add to the measurement path subtracts phase from your loop's
phase margin at the crossover frequency. This is not a subtlety; it is the main
practical constraint on how much filtering you can use.

**The numbers you need:**

| Filter | Phase lag |
|---|---|
| First-order low-pass at cutoff `fc` | 45 deg at `fc`, ~26 deg at `fc/2`, ~11 deg at `fc/5` |
| Second-order (Butterworth biquad) at `fc` | 90 deg at `fc`, ~53 deg at `fc/2`, ~23 deg at `fc/5` |
| Notch, quality factor Q, at centre `f0` | ~0 deg well away from `f0`; the lag is confined to roughly `f0/Q` either side |
| Pure transport delay `T` | `-omega*T` radians, i.e. **linear in frequency and unbounded** |
| Moving average, N samples | exactly `(N-1)/2` samples of delay, at every frequency |

**Budget it.** You want 45-60 degrees of phase margin on a rate loop. Start
from the phase margin the bare loop has and subtract:

- gyro low-pass: use the table above at your crossover frequency,
- each notch: a few degrees if `Q` is high and `f0` is far from crossover,
- the sample-and-hold of a discrete controller: `180 * f_crossover / f_sample`
  degrees, which at a 20 Hz crossover and a 2 kHz loop is only 1.8 degrees, but
  at a 500 Hz loop is 7.2,
- transport delay from ESC protocol, link latency and scheduling: `360 * f * T`
  degrees. **This is the one people forget.** 5 ms at a 20 Hz crossover is
  36 degrees. It is most of your budget, and it does not show up in a gain plot
  at all -- pure delay attenuates nothing.

**The failure mode.** A cascaded low-pass at 40 Hz plus a 20 ms transport delay
at a 20 Hz crossover is roughly 53 + 144 = 197 degrees of lag from those two
alone. You are already past 180 degrees before the plant contributes anything,
and no gain setting is stable. Long before that, at around 20-30 degrees of
margin, the loop is technically stable and behaves badly: every disturbance
rings, the aircraft feels "loose", and raising P makes it worse rather than
crisper.

**Practical rules:**

- Put the gyro low-pass cutoff at **5-10x your loop crossover**, not at
  "whatever makes the buzz stop".
- Prefer removing vibration mechanically or with a tracked notch over lowering
  the low-pass. A notch spends its phase in a narrow band you are not crossing
  over in; a low-pass spends it everywhere.
- If you add a filter, re-tune. A filter added to a tuned loop has changed the
  loop.
- The D-term filter and the gyro filter are separate. The D term needs more
  filtering than P, because differentiation amplifies noise; filtering the
  whole gyro signal that hard would slow the P path unnecessarily.

---

## 6. Bandwidth separation between cascaded loops

Each loop should be **3-5x slower in bandwidth than the loop inside it**. This
is what makes "the inner loop is instantaneous" -- the assumption that lets you
tune each loop against a simple plant -- actually true.

If the separation is too small, the loops interact: the outer loop commands a
setpoint the inner loop has not achieved yet, integrates the resulting error,
and produces a low-frequency oscillation that looks like a badly tuned outer
loop but is not. Retuning the outer loop will not fix it; you have to either
speed up the inner loop or slow down the outer one.

`dctk.cascade.CascadeController.check_bandwidth_separation()` flags adjacent
pairs that are too close.

Typical values on a small multirotor:

| Loop | Execution rate | Closed-loop bandwidth |
|---|---|---|
| Rate | 1-8 kHz | 15-30 Hz |
| Attitude | 500 Hz - 1 kHz | 5-10 Hz |
| Velocity | 50-250 Hz | 1-3 Hz |
| Position | 20-50 Hz | 0.3-1 Hz |

Note that execution rate and bandwidth are different things. Running a loop
faster does not make it faster; it only removes the sample-and-hold phase lag
and lets you *tune* it faster.

---

## 7. Mixer saturation: give up thrust, keep attitude

When the allocation does not fit inside the motors' range you must give
something up. The correct choice is thrust, and the argument is short:

- Lose attitude authority and a multirotor flips. There is no recovery at low
  altitude, and the loss is immediate.
- Lose a few percent of collective thrust and it descends slightly. You add
  throttle, life continues.

Per-motor clipping is the wrong answer even though it is the easiest to write:
clipping one motor and not another changes the *direction* of the commanded
torque, so a roll command comes out partly as pitch or yaw. The aircraft feels
like it has a bent airframe, and only when you are pushing hard.

Yaw should yield before roll and pitch. Yaw on a multirotor comes from prop
drag torque, which is roughly an order of magnitude weaker than the thrust
differential that drives roll and pitch, so it is the axis that saturates
first -- and it is the least urgent. A slow yaw is annoying; a slow roll is a
crash. `dctk.mixer.Mixer` does this by default (`yaw_priority=0.0`).

**Log the saturation flag.** A rate loop that spends its time with the mixer
saturated is a rate loop that is not in control of the aircraft, and it is
completely invisible in an attitude plot.

---

## 8. Things that look like control problems and are not

- **Slow drift in altitude hold.** Usually barometer drift or a gyro bias
  walking the estimator, not the altitude loop. Check the estimator's output
  against the raw sensor before touching a gain.
- **Toilet-bowling in position hold.** Almost always magnetometer
  interference -- current in the power leads, or the mag mounted too close to
  the ESCs -- not the position loop. The heading estimate is wrong, so "north"
  rotates as the aircraft flies.
- **Twitching only at certain throttle settings.** Resonance. Some frame or
  prop mode is being excited at that RPM. Move the notch, or change the prop.
- **Fine in the air, oscillates near the ground.** Ground effect changes the
  thrust curve. This is a real plant change, not a tuning error, and the fix is
  a gain schedule or simply accepting it.
- **Fine on the bench, oscillates in flight.** Something in the flight path
  that is not on the bench: vibration, transport delay under load, a companion
  computer stealing CPU. Compare the loop timing in the log, not just the gyro.
- **Worked yesterday, does not today.** Check the battery. Motor authority
  scales with voltage, so the same gains at 3.7 V/cell are a different loop
  from the same gains at 4.2 V/cell. This is why serious stacks scale P by
  measured battery voltage.

---

## Related repositories

- **flight-log-analyzer** -- pull the vibration spectrum, EKF innovation
  history, mixer saturation and mode timeline out of a PX4 ULog or ArduPilot
  log. That is where the numbers for sections 3, 4 and 5 come from on a real
  aircraft.
- **px4-mavlink-companion** -- the offboard control path, where the transport
  delay in section 5 actually lives.
- **ros2-drone-bringup** -- SITL setup for testing a control change without
  risking hardware.
