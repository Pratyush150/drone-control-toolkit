"""Filter tests: convergence, attenuation, and the phase cost."""

from __future__ import annotations

import numpy as np
import pytest

from dctk.filters import (
    BiquadLowPass,
    ComplementaryFilter,
    KalmanFilter,
    KalmanFilter1D,
    MadgwickAHRS,
    MovingAverage,
    NotchFilter,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_euler,
)


# ----------------------------------------------------------------------
# quaternions
# ----------------------------------------------------------------------
def test_euler_quaternion_roundtrip():
    for rpy in [(0.1, -0.2, 0.3), (-0.5, 0.4, -1.2), (0.0, 0.0, 0.0)]:
        q = euler_to_quaternion(*rpy)
        assert np.allclose(quaternion_to_euler(q), rpy, atol=1e-12)
        assert np.linalg.norm(q) == pytest.approx(1.0)


def test_quaternion_multiply_identity_and_inverse():
    q = euler_to_quaternion(0.3, -0.4, 0.5)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion_multiply(q, identity), q)
    conj = np.array([q[0], -q[1], -q[2], -q[3]])
    assert np.allclose(quaternion_multiply(q, conj), identity, atol=1e-12)


def test_quaternion_normalize_survives_a_degenerate_input():
    assert np.allclose(quaternion_normalize([0.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# complementary filter
# ----------------------------------------------------------------------
def test_complementary_tau_cutoff_relationship():
    cf = ComplementaryFilter(tau=1.0 / (2 * np.pi))
    assert cf.cutoff_hz == pytest.approx(1.0)
    cf2 = ComplementaryFilter.from_cutoff(5.0)
    assert cf2.cutoff_hz == pytest.approx(5.0)
    # alpha must depend on dt, not be frozen at construction.
    assert cf.alpha(0.001) > cf.alpha(0.01)


def test_complementary_filter_converges_to_the_true_angle_despite_gyro_bias():
    """The headline property: a constant gyro bias would integrate to infinity
    on its own, but the accelerometer path bounds the error at bias * tau."""
    dt = 0.005
    tau = 0.5
    true_roll = np.deg2rad(12.0)
    bias = np.deg2rad(3.0)  # 3 deg/s of gyro bias, a bad but realistic sensor
    cf = ComplementaryFilter(tau=tau)
    accel = np.array([0.0, np.sin(true_roll), np.cos(true_roll)]) * 9.81

    for _ in range(int(40.0 / dt)):
        roll, _ = cf.update([bias, 0.0, 0.0], accel, dt)

    # Steady state offset is bias * tau. Assert both bounds: it converges, and
    # it converges to the value the theory predicts.
    expected_offset = bias * tau
    assert roll == pytest.approx(true_roll + expected_offset, abs=np.deg2rad(0.5))
    assert abs(roll - true_roll) < np.deg2rad(2.0)


def test_complementary_filter_bias_error_scales_with_tau():
    dt = 0.005
    bias = np.deg2rad(3.0)
    accel = np.array([0.0, 0.0, 9.81])
    errs = []
    for tau in (0.25, 1.0):
        cf = ComplementaryFilter(tau=tau)
        for _ in range(int(60.0 / dt)):
            roll, _ = cf.update([bias, 0.0, 0.0], accel, dt)
        errs.append(abs(roll))
    assert errs[1] > 3.0 * errs[0]


def test_complementary_filter_tracks_a_pure_gyro_rotation_at_high_frequency():
    """Fast motion must come from the gyro; the accel path is too slow to
    follow it and must not be allowed to drag the estimate back."""
    dt = 0.002
    cf = ComplementaryFilter(tau=2.0)
    rate = np.deg2rad(60.0)
    accel = np.array([0.0, 0.0, 9.81])  # accel says "level" the whole time
    for i in range(int(0.2 / dt)):
        roll, _ = cf.update([rate, 0.0, 0.0], accel, dt)
    assert roll == pytest.approx(rate * 0.2, rel=0.1)


def test_accel_angles_are_correct_and_well_conditioned_past_45_degrees():
    for angle_deg in (0.0, 30.0, 60.0, 85.0):
        a = np.deg2rad(angle_deg)
        roll, pitch = ComplementaryFilter.accel_angles([0.0, np.sin(a), np.cos(a)])
        assert roll == pytest.approx(a)
        assert pitch == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# Madgwick
# ----------------------------------------------------------------------
def test_madgwick_converges_to_the_accelerometer_attitude():
    dt = 0.005
    true_roll = np.deg2rad(20.0)
    accel = np.array([0.0, np.sin(true_roll), np.cos(true_roll)]) * 9.81
    m = MadgwickAHRS(beta=0.2)
    for _ in range(int(20.0 / dt)):
        m.update([0.0, 0.0, 0.0], accel, dt)
    roll, pitch, _ = m.euler
    assert roll == pytest.approx(true_roll, abs=np.deg2rad(1.0))
    assert pitch == pytest.approx(0.0, abs=np.deg2rad(1.0))
    assert np.linalg.norm(m.q) == pytest.approx(1.0)


def test_madgwick_yaw_is_unobservable_without_a_magnetometer():
    """Documented behaviour, not a bug: gravity carries no heading."""
    dt = 0.005
    m = MadgwickAHRS(beta=0.1)
    for _ in range(int(5.0 / dt)):
        m.update([0.0, 0.0, np.deg2rad(20.0)], [0.0, 0.0, 9.81], dt)
    _, _, yaw = m.euler
    assert abs(yaw) > np.deg2rad(80.0)  # it followed the gyro, uncorrected


def test_madgwick_with_magnetometer_holds_yaw():
    dt = 0.005
    m = MadgwickAHRS(beta=0.3)
    mag = np.array([0.4, 0.0, 0.6])  # north-ish with some inclination
    for _ in range(int(20.0 / dt)):
        m.update([0.0, 0.0, np.deg2rad(2.0)], [0.0, 0.0, 9.81], dt, mag=mag)
    _, _, yaw = m.euler
    assert abs(yaw) < np.deg2rad(15.0)


# ----------------------------------------------------------------------
# Kalman
# ----------------------------------------------------------------------
def test_kalman1d_reduces_noise_and_covariance_decreases():
    rng = np.random.default_rng(3)
    kf = KalmanFilter1D(q=1e-5, r=0.1, p0=1.0)
    truth = 2.0
    out = []
    for _ in range(500):
        out.append(kf.update(truth + rng.normal(0.0, 0.3)))
    assert kf.p < 1.0
    assert abs(out[-1] - truth) < 0.1
    assert np.std(out[100:]) < 0.3


def test_kalman_multidim_tracks_a_constant_velocity_target():
    dt = 0.01
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    kf = KalmanFilter(F, H, Q=np.diag([1e-6, 1e-4]), R=np.array([[0.04]]), P0=np.eye(2))
    rng = np.random.default_rng(11)
    v_true = 1.5
    for i in range(2000):
        kf.predict()
        kf.update([v_true * i * dt + rng.normal(0.0, 0.2)])
    assert kf.x[1] == pytest.approx(v_true, abs=0.1)


def test_kalman_covariance_stays_symmetric_psd_with_joseph_form():
    dt = 0.01
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    kf = KalmanFilter(F, H, Q=np.diag([1e-12, 1e-12]), R=np.array([[1e-8]]), P0=np.eye(2) * 1e3)
    rng = np.random.default_rng(5)
    for _ in range(5000):
        kf.predict()
        kf.update([rng.normal(0.0, 1e-4)])
        assert np.allclose(kf.P, kf.P.T, atol=1e-18)
        assert np.min(np.linalg.eigvalsh(kf.P)) >= -1e-18


def test_kalman_nis_is_near_the_measurement_dimension_when_consistent():
    dt = 0.01
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    sigma = 0.2
    kf = KalmanFilter(F, H, Q=np.diag([1e-8, 1e-6]), R=np.array([[sigma**2]]), P0=np.eye(2))
    rng = np.random.default_rng(17)
    nis = []
    for i in range(4000):
        kf.predict()
        kf.update([1.0 + rng.normal(0.0, sigma)])
        if i > 500:
            nis.append(kf.nis())
    assert 0.5 < float(np.mean(nis)) < 2.0  # dimension is 1


# ----------------------------------------------------------------------
# signal conditioning
# ----------------------------------------------------------------------
def test_moving_average_passes_dc_and_nulls_its_design_frequency():
    ma = MovingAverage(10)
    out = ma.filt(np.full(200, 3.0))
    assert out[-1] == pytest.approx(3.0)
    assert ma.group_delay_samples == pytest.approx(4.5)

    fs = 1000.0
    n = 10
    t = np.arange(4000) / fs
    # A boxcar of N samples has an exact null at fs/N.
    ma2 = MovingAverage(n)
    y = ma2.filt(np.sin(2 * np.pi * (fs / n) * t))
    assert np.max(np.abs(y[100:])) < 1e-9


def test_biquad_lowpass_is_minus_3db_at_cutoff_and_rolls_off_at_40db_per_decade():
    fs, fc = 1000.0, 40.0
    lp = BiquadLowPass(fc, fs)
    assert lp.gain_at(fc, fs) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6)
    assert lp.gain_at(0.01, fs) == pytest.approx(1.0, rel=1e-3)
    # Two octaves above cutoff a 2nd-order filter is about 1/16 of the gain.
    assert lp.gain_at(4 * fc, fs) == pytest.approx(1.0 / 16.0, rel=0.25)


def test_biquad_lowpass_phase_lag_is_90_degrees_at_cutoff():
    """The number to subtract from your phase margin."""
    fs, fc = 1000.0, 40.0
    lp = BiquadLowPass(fc, fs)
    assert lp.phase_lag_deg(fc, fs) == pytest.approx(90.0, abs=1e-6)
    assert 0.0 < lp.phase_lag_deg(fc / 4, fs) < 45.0


def test_biquad_lowpass_actually_attenuates_noise_in_the_time_domain():
    fs = 1000.0
    t = np.arange(4000) / fs
    signal = np.sin(2 * np.pi * 2.0 * t)
    rng = np.random.default_rng(2)
    noisy = signal + rng.normal(0.0, 0.5, t.size)
    lp = BiquadLowPass(20.0, fs)
    out = lp.filt(noisy)
    assert np.std(out[500:] - signal[500:]) < 0.35 * np.std(noisy - signal)


def test_notch_removes_its_centre_frequency_and_passes_the_rest():
    fs, f0 = 1000.0, 200.0
    nf = NotchFilter(f0, fs, q=20.0)
    assert nf.gain_at(f0, fs) == pytest.approx(0.0, abs=1e-9)
    assert nf.gain_at(50.0, fs) == pytest.approx(1.0, rel=0.01)
    assert nf.gain_at(400.0, fs) == pytest.approx(1.0, rel=0.01)

    t = np.arange(6000) / fs
    y = np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * f0 * t)
    out = nf.filt(y)
    residual = out[2000:] - np.sin(2 * np.pi * 5.0 * t[2000:])
    assert np.max(np.abs(residual)) < 0.05


def test_fixed_notch_fails_when_the_tone_moves_but_a_retuned_one_does_not():
    """This is the documented failure mode of a fixed notch across the throttle
    range, demonstrated rather than asserted in prose."""
    fs = 2000.0
    t = np.arange(8000) / fs
    tone_hz = 350.0  # the motor spun up; the notch was set for 200 Hz
    y = 0.5 * np.sin(2 * np.pi * tone_hz * t)

    fixed = NotchFilter(200.0, fs, q=20.0)
    out_fixed = fixed.filt(y)
    assert np.max(np.abs(out_fixed[2000:])) > 0.4  # tone passes straight through

    tracked = NotchFilter(200.0, fs, q=20.0)
    tracked.retune(tone_hz)
    out_tracked = tracked.filt(y)
    assert np.max(np.abs(out_tracked[2000:])) < 0.05


def test_blade_pass_frequency_formula():
    assert NotchFilter.blade_pass_hz(6000.0, blades=2) == pytest.approx(200.0)
    assert NotchFilter.blade_pass_hz(24000.0, blades=3) == pytest.approx(1200.0)


def test_filter_construction_rejects_nonsense():
    with pytest.raises(ValueError):
        BiquadLowPass(600.0, 1000.0)  # above Nyquist
    with pytest.raises(ValueError):
        NotchFilter(0.0, 1000.0)
    with pytest.raises(ValueError):
        MovingAverage(0)
    with pytest.raises(ValueError):
        ComplementaryFilter(tau=0.0)
