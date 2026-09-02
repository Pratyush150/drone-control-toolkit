#!/usr/bin/env python3
"""Prop vibration on a gyro: notch vs low-pass, and what each one costs.

Synthesises a gyro signal containing:

* the pilot's actual command (a few Hz),
* blade-pass vibration whose frequency **sweeps with throttle** -- 6 000 rpm at
  idle to 12 000 rpm at full, two blades, so 200 Hz to 400 Hz,
* broadband sensor noise.

The naive fix for vibration is to lower the low-pass cutoff until the buzz goes
away. That works, and it costs phase margin at the frequency your rate loop
actually crosses over. A notch removes a narrow band without touching the phase
elsewhere -- if it is pointed at the right frequency, which on a real aircraft
it usually is not, because the tone follows throttle.

Five conditioning chains are compared on residual error, on what a rate
controller's D term does with the result (that is where vibration becomes motor
heat), and on the phase lag each chain spends at a 20 Hz loop crossover.

Run: ``python3 examples/vibration_notch_filter.py``
"""

from __future__ import annotations

import numpy as np

from _common import figure, header, save, table

from dctk.filters import BiquadLowPass, NotchFilter

FS = 4000.0
DURATION = 4.0
COMMAND_HZ = 4.0
VIB_AMPLITUDE = 0.35  # rad/s of blade-pass content on the gyro
NOISE_SIGMA = 0.02
RPM_IDLE = 6000.0
RPM_MAX = 12000.0
BLADES = 2
CROSSOVER_HZ = 20.0  # where a typical multirotor rate loop crosses over
FIXED_NOTCH_HZ = NotchFilter.blade_pass_hz(RPM_IDLE, BLADES)  # set from a hover log
LP_WIDE_HZ = 150.0
LP_TIGHT_HZ = 40.0


def synthesise():
    """Gyro signal, true command, and the instantaneous blade-pass frequency."""
    n = int(DURATION * FS)
    t = np.arange(n) / FS
    rpm = RPM_IDLE + (RPM_MAX - RPM_IDLE) * (t / DURATION)
    blade_pass = NotchFilter.blade_pass_hz(rpm, BLADES)
    # Integrate the instantaneous frequency to get a continuous phase; using
    # 2*pi*f*t directly would produce a discontinuous chirp.
    phase = 2 * np.pi * np.cumsum(blade_pass) / FS
    command = np.sin(2 * np.pi * COMMAND_HZ * t)
    rng = np.random.default_rng(31)
    gyro = command + VIB_AMPLITUDE * np.sin(phase) + rng.normal(0.0, NOISE_SIGMA, n)
    return t, gyro, command, blade_pass


def d_term(signal: np.ndarray, kd: float = 0.02) -> np.ndarray:
    """What a rate controller's derivative term does with this signal."""
    return kd * np.gradient(signal, 1.0 / FS)


def tracked_notch(gyro: np.ndarray, blade_pass: np.ndarray, q: float = 15.0) -> np.ndarray:
    """RPM-tracked notch: retune the centre every sample from the motor RPM."""
    notch = NotchFilter(float(blade_pass[0]), FS, q=q)
    out = np.zeros_like(gyro)
    for i, sample in enumerate(gyro):
        # In a real loop this frequency comes from bidirectional-DShot ESC
        # telemetry, once per motor, updated every few milliseconds.
        notch.retune(float(blade_pass[i]))
        out[i] = notch.update(float(sample))
    return out


def main() -> None:
    header("Prop vibration: notch vs low-pass on a throttle sweep")
    t, gyro, command, blade_pass = synthesise()
    print(
        f"\nblade-pass sweeps {blade_pass[0]:.0f} Hz -> {blade_pass[-1]:.0f} Hz "
        f"({RPM_IDLE:.0f} to {RPM_MAX:.0f} rpm, {BLADES} blades)"
    )
    print(f"fixed notch was set from a hover log at {FIXED_NOTCH_HZ:.0f} Hz")

    lp_wide = BiquadLowPass(LP_WIDE_HZ, FS)
    lp_tight = BiquadLowPass(LP_TIGHT_HZ, FS)
    fixed = NotchFilter(FIXED_NOTCH_HZ, FS, q=15.0)

    chains = [
        ("raw gyro", gyro, 0.0),
        (
            f"{LP_WIDE_HZ:.0f} Hz low-pass",
            lp_wide.filt(gyro),
            lp_wide.phase_lag_deg(CROSSOVER_HZ, FS),
        ),
        (
            f"{LP_TIGHT_HZ:.0f} Hz low-pass",
            lp_tight.filt(gyro),
            lp_tight.phase_lag_deg(CROSSOVER_HZ, FS),
        ),
        (
            f"fixed notch @{FIXED_NOTCH_HZ:.0f} Hz + {LP_WIDE_HZ:.0f} Hz LP",
            BiquadLowPass(LP_WIDE_HZ, FS).filt(fixed.filt(gyro)),
            lp_wide.phase_lag_deg(CROSSOVER_HZ, FS) + fixed.phase_lag_deg(CROSSOVER_HZ, FS),
        ),
        (
            f"RPM-tracked notch + {LP_WIDE_HZ:.0f} Hz LP",
            BiquadLowPass(LP_WIDE_HZ, FS).filt(tracked_notch(gyro, blade_pass)),
            lp_wide.phase_lag_deg(CROSSOVER_HZ, FS) + fixed.phase_lag_deg(CROSSOVER_HZ, FS),
        ),
    ]

    settle = t > 0.5
    d_clean = float(np.sqrt(np.mean(d_term(command)[settle] ** 2)))
    print()
    rows = []
    for label, signal, lag in chains:
        residual = float(np.std(signal[settle] - command[settle]))
        d_rms = float(np.sqrt(np.mean(d_term(signal)[settle] ** 2)))
        rows.append((label, residual, d_rms, d_rms / d_clean, lag))
        print(
            f"  {label:<36}  residual = {residual:6.4f} rad/s   "
            f"D-term RMS = {d_rms:7.4f} ({d_rms / d_clean:5.1f}x ideal)   "
            f"phase lag @{CROSSOVER_HZ:.0f} Hz = {lag:5.2f} deg"
        )

    raw_row, wide, tight, fixed_row, tracked_row = rows
    print()
    table(
        [
            ("vibration on the raw gyro", f"D term is {raw_row[3]:.0f}x the ideal"),
            (
                f"{LP_TIGHT_HZ:.0f} Hz LP: it works, and it costs",
                f"D {tight[3]:.2f}x ideal, but {tight[4]:.1f} deg of phase at "
                f"{CROSSOVER_HZ:.0f} Hz",
            ),
            (
                "tracked notch: same result, less phase",
                f"D {tracked_row[3]:.2f}x ideal for {tracked_row[4]:.1f} deg",
            ),
            (
                "phase margin saved by using a notch",
                f"{tight[4] - tracked_row[4]:.1f} deg",
            ),
            (
                "fixed vs tracked notch (D-term RMS)",
                f"{fixed_row[2]:.4f} vs {tracked_row[2]:.4f} "
                f"({100 * (1 - tracked_row[2] / fixed_row[2]):+.1f} %)",
            ),
        ]
    )
    print(
        f"\nThe {LP_TIGHT_HZ:.0f} Hz low-pass and the tracked notch clean the signal about\n"
        f"equally well. The low-pass pays {tight[4]:.0f} degrees of phase at the loop\n"
        f"crossover for it; the notch pays {tracked_row[4]:.0f}. That difference is\n"
        "phase margin you get to spend on gain instead.\n"
        "\n"
        "The fixed notch is only correct while the tone sits inside it. On this sweep\n"
        "the tone leaves the notch within the first second and never comes back, and\n"
        "the notch's phase cost is paid for the whole flight regardless.\n"
        "\n"
        "Use flight-log-analyzer to get the real spectrum out of a flight log before\n"
        "you decide where to put a notch. Guessing from the sound of the aircraft is\n"
        "how people end up with four notches and no phase margin."
    )

    # --- plots ---------------------------------------------------------------
    fig, (ax1, ax2, ax3) = figure(3, 1, figsize=(11, 11))
    window = (t > 1.0) & (t < 1.3)
    ax1.plot(t[window], gyro[window], lw=0.6, alpha=0.5, label="raw gyro")
    ax1.plot(t[window], command[window], "k", lw=1.8, label="true command")
    for label, signal, _ in chains[2:]:
        ax1.plot(t[window], signal[window], lw=1.0, label=label)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("rate (rad/s)")
    ax1.set_title("Gyro signal, 300 ms window")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    half = t > DURATION / 2
    freqs = np.fft.rfftfreq(int(np.sum(half)), 1.0 / FS)
    for label, signal, _ in chains:
        spectrum = np.abs(np.fft.rfft(signal[half])) / np.sum(half)
        ax2.semilogy(freqs, spectrum + 1e-9, lw=0.8, label=label)
    ax2.set_xlim(0, 600)
    ax2.set_ylim(1e-6, 1e0)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("magnitude")
    ax2.set_title("Spectrum over the second half of the record (tone is sweeping 300-400 Hz)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    lag_freqs = np.logspace(0, np.log10(FS / 2 - 1), 400)
    ax3.semilogx(lag_freqs, [lp_wide.phase_lag_deg(f, FS) for f in lag_freqs],
                 label=f"{LP_WIDE_HZ:.0f} Hz LP")
    ax3.semilogx(lag_freqs, [lp_tight.phase_lag_deg(f, FS) for f in lag_freqs],
                 label=f"{LP_TIGHT_HZ:.0f} Hz LP")
    ax3.semilogx(lag_freqs, [fixed.phase_lag_deg(f, FS) for f in lag_freqs],
                 label=f"notch @{FIXED_NOTCH_HZ:.0f} Hz")
    ax3.axvline(CROSSOVER_HZ, color="r", ls=":", lw=1.0, label="rate loop crossover")
    ax3.set_xlabel("frequency (Hz)")
    ax3.set_ylabel("phase lag (deg)")
    ax3.set_title("Phase cost: this is what you subtract from your phase margin")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, which="both")
    save(fig, "vibration_notch_filter.png")


if __name__ == "__main__":
    main()
