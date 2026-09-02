"""Metric tests against hand-computable responses.

A metrics module that is wrong makes every other claim in the repo wrong, so
these are checked against closed-form answers rather than against themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from dctk.metrics import (
    iae,
    ise,
    itae,
    overshoot_percent,
    peak_time,
    rise_time,
    settling_time,
    steady_state_error,
    step_metrics,
)


def first_order(tau=1.0, duration=20.0, n=20001):
    t = np.linspace(0.0, duration, n)
    return t, 1.0 - np.exp(-t / tau)


def test_rise_time_of_first_order_is_tau_ln9():
    """10-90 % rise time of 1 - exp(-t/tau) is exactly tau * ln(9)."""
    t, y = first_order(tau=1.0)
    assert rise_time(t, y, 1.0) == pytest.approx(np.log(9.0), rel=1e-4)


def test_rise_time_scales_with_tau():
    t, y = first_order(tau=0.25)
    assert rise_time(t, y, 1.0) == pytest.approx(0.25 * np.log(9.0), rel=1e-4)


def test_settling_time_of_first_order_is_tau_ln50():
    """2 % settling of a first-order step is exactly tau * ln(50)."""
    t, y = first_order(tau=1.0)
    assert settling_time(t, y, 1.0, tolerance=0.02) == pytest.approx(np.log(50.0), rel=1e-3)


def test_first_order_has_no_overshoot():
    _, y = first_order()
    assert overshoot_percent(y, 1.0) == pytest.approx(0.0)


def test_second_order_overshoot_matches_the_closed_form():
    """For zeta = 0.5, peak overshoot is exp(-pi*zeta/sqrt(1-zeta^2)) = 16.30 %."""
    zeta, wn = 0.5, 5.0
    t = np.linspace(0.0, 6.0, 60001)
    wd = wn * np.sqrt(1 - zeta**2)
    y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
    expected = 100.0 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    assert overshoot_percent(y, 1.0) == pytest.approx(expected, rel=1e-3)
    # Peak time is pi / wd.
    assert peak_time(t, y, 1.0) == pytest.approx(np.pi / wd, rel=1e-3)


def test_steady_state_error_on_a_constant_offset():
    y = np.full(1000, 0.9)
    assert steady_state_error(y, 1.0) == pytest.approx(0.1)


def test_iae_ise_itae_on_a_constant_error():
    """Constant error e over [0, T]: IAE = e*T, ISE = e^2*T, ITAE = e*T^2/2."""
    t = np.linspace(0.0, 10.0, 10001)
    y = np.full_like(t, 0.8)
    assert iae(t, y, 1.0) == pytest.approx(0.2 * 10.0, rel=1e-9)
    assert ise(t, y, 1.0) == pytest.approx(0.04 * 10.0, rel=1e-9)
    assert itae(t, y, 1.0) == pytest.approx(0.2 * 100.0 / 2.0, rel=1e-9)


def test_itae_penalises_late_error_more_than_early_error():
    t = np.linspace(0.0, 10.0, 10001)
    early = np.where(t < 2.0, 0.0, 1.0)  # error only in the first 2 s
    late = np.where(t < 8.0, 1.0, 0.0)  # error only in the last 2 s
    assert iae(t, early, 1.0) == pytest.approx(iae(t, late, 1.0), rel=1e-3)
    assert itae(t, late, 1.0) > 3.0 * itae(t, early, 1.0)


def test_rise_time_is_nan_when_the_response_never_arrives():
    t = np.linspace(0.0, 1.0, 101)
    y = 0.5 * t  # only reaches 0.5, never 0.9 of the step
    assert np.isnan(rise_time(t, y, 1.0))


def test_settling_time_is_nan_for_a_response_that_leaves_the_band():
    t = np.linspace(0.0, 10.0, 1001)
    y = 1.0 + 0.5 * np.sin(5 * t)  # never settles
    assert np.isnan(settling_time(t, y, 1.0))


def test_metrics_handle_a_downward_step():
    t = np.linspace(0.0, 10.0, 10001)
    y = 1.0 - (1.0 - np.exp(-t))  # 1 -> 0
    m = step_metrics(t, y, 0.0)
    assert m.rise_time == pytest.approx(np.log(9.0), rel=1e-3)
    assert m.overshoot_pct == pytest.approx(0.0)


def test_step_metrics_bundle_is_self_consistent():
    t, y = first_order(tau=0.5)
    m = step_metrics(t, y, 1.0)
    assert m.rise_time == pytest.approx(rise_time(t, y, 1.0))
    assert m.iae == pytest.approx(iae(t, y, 1.0))
    assert "rise=" in m.summary("label")
    assert set(m.as_dict()) >= {"rise_time", "settling_time", "overshoot_pct", "itae"}


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError):
        rise_time([0, 1, 2], [0, 1], 1.0)
    with pytest.raises(ValueError):
        iae([0, 1, 1], [0, 1, 2], 1.0)  # not strictly increasing
