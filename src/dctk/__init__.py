"""dctk -- drone control toolkit.

Control and estimation building blocks for small unmanned aircraft, written to
be read: every non-obvious line exists because of a failure mode that shows up
on real hardware, and the docstring says which one.

Modules
-------
``pid``         PID with anti-windup, derivative-on-measurement, D filtering,
                feed-forward and bumpless transfer.
``cascade``     Position -> velocity -> attitude -> rate loop nesting.
``lqr``         Discrete infinite-horizon LQR, scipy-free.
``filters``     Complementary, Madgwick, Kalman, biquad low-pass, notch.
``estimator``   Quaternion + gyro-bias attitude EKF with NIS diagnostics.
``mixer``       Control allocation for quad X / plus, hexa X, quadplane VTOL.
``trajectory``  Minimum-jerk, trapezoidal, spline waypoint paths.
``sim``         Plants plus injectable noise, latency, motor lag, quantisation
                and wind, so controllers can be demonstrated against defects.
``metrics``     Rise/settling time, overshoot, IAE/ISE/ITAE.
``tuning``      Ziegler-Nichols and relay auto-tuning.

Dependencies: numpy only. matplotlib is imported lazily by the example scripts
and is never needed to import this package or to run the tests.
"""

from __future__ import annotations

__version__ = "0.1.0"
__license__ = "MIT"

from .pid import PID, PIDGains, PIDState, AntiWindup
from .cascade import CascadeController, CascadeLoop, LoopSpec, MultirotorCascade
from .lqr import LQRResult, brysons_rule, c2d, dlqr, is_controllable
from .filters import (
    BiquadLowPass,
    ComplementaryFilter,
    KalmanFilter,
    KalmanFilter1D,
    MadgwickAHRS,
    MovingAverage,
    NotchFilter,
)
from .estimator import AttitudeEKF, EKFDiagnostics
from .mixer import Mixer, MixerGeometry, hexa_x, quad_plus, quad_x, vtol_quadplane
from .trajectory import (
    Trajectory,
    check_limits,
    min_jerk,
    min_jerk_duration,
    trapezoidal_profile,
    waypoint_path,
)
from .sim import (
    ActuatorDelay,
    FirstOrderPlant,
    MotorLag,
    PitchPlant,
    PointMassQuadrotor2D,
    Quantiser,
    SecondOrderPlant,
    SensorNoise,
    WindGust,
    simulate,
)
from .metrics import StepMetrics, step_metrics
from .tuning import UltimateGain, ZNGains, relay_autotune, ziegler_nichols

__all__ = [
    "__version__",
    # pid
    "PID",
    "PIDGains",
    "PIDState",
    "AntiWindup",
    # cascade
    "CascadeController",
    "CascadeLoop",
    "LoopSpec",
    "MultirotorCascade",
    # lqr
    "LQRResult",
    "brysons_rule",
    "c2d",
    "dlqr",
    "is_controllable",
    # filters
    "BiquadLowPass",
    "ComplementaryFilter",
    "KalmanFilter",
    "KalmanFilter1D",
    "MadgwickAHRS",
    "MovingAverage",
    "NotchFilter",
    # estimator
    "AttitudeEKF",
    "EKFDiagnostics",
    # mixer
    "Mixer",
    "MixerGeometry",
    "quad_x",
    "quad_plus",
    "hexa_x",
    "vtol_quadplane",
    # trajectory
    "Trajectory",
    "check_limits",
    "min_jerk",
    "min_jerk_duration",
    "trapezoidal_profile",
    "waypoint_path",
    # sim
    "ActuatorDelay",
    "FirstOrderPlant",
    "MotorLag",
    "PitchPlant",
    "PointMassQuadrotor2D",
    "Quantiser",
    "SecondOrderPlant",
    "SensorNoise",
    "WindGust",
    "simulate",
    # metrics
    "StepMetrics",
    "step_metrics",
    # tuning
    "UltimateGain",
    "ZNGains",
    "relay_autotune",
    "ziegler_nichols",
]
