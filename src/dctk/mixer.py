"""Motor mixing and control allocation.

The mixer turns four numbers the controller cares about -- roll, pitch and yaw
torque plus collective thrust -- into ``n`` motor commands. On paper it is one
matrix multiply. In practice everything interesting happens at the limits, and
that is what this module is about.

The one decision that matters
-----------------------------
When the demanded mix does not fit inside ``[idle, 1]`` on every motor, you
must give something up. **Give up thrust, keep attitude.** The reasoning is
short and it is not a matter of taste:

* A multirotor that loses attitude authority flips. There is no recovery from
  an inverted multirotor at low altitude, and the loss is immediate.
* A multirotor that loses a few percent of collective thrust descends slightly.
  You notice, you add throttle, life continues.

So when the allocation saturates, this module scales the *thrust* component
until the attitude component fits, rather than scaling everything uniformly
(which throws away attitude authority exactly when the aircraft is asking for
it hardest) or clipping per-motor (which silently distorts the torque
direction, so a roll command comes out partly as yaw).

This is what PX4's ``ControlAllocation`` and Betaflight's "airmode" are both
doing underneath, expressed as a small amount of readable numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MixerGeometry", "Mixer", "quad_x", "quad_plus", "hexa_x", "vtol_quadplane"]

_SQRT_HALF = float(np.sqrt(0.5))


@dataclass(frozen=True)
class MixerGeometry:
    """A control-effectiveness matrix and its metadata.

    ``effectiveness`` is ``(4, n_motors)``: rows are
    ``[roll, pitch, yaw, thrust]``, columns are motors. Entry ``(i, j)`` is the
    contribution of motor ``j`` at unit command to axis ``i``.

    Sign conventions (aerospace body frame, x forward, y right, z down):

    * positive roll torque rolls right,
    * positive pitch torque pitches nose up,
    * positive yaw torque yaws right (nose to starboard),
    * positive thrust is up (so it is ``-z``; the sign is handled here so the
      caller works in "more thrust is more positive").
    """

    name: str
    effectiveness: NDArray[np.float64]
    motor_names: Sequence[str] = ()

    @property
    def n_motors(self) -> int:
        return int(self.effectiveness.shape[1])

    def __post_init__(self) -> None:
        e = np.asarray(self.effectiveness, dtype=float)
        if e.ndim != 2 or e.shape[0] != 4:
            raise ValueError("effectiveness must be a (4, n_motors) array")
        object.__setattr__(self, "effectiveness", e)


def _geom(
    name: str, rows: Sequence[Sequence[float]], motor_names: Sequence[str]
) -> MixerGeometry:
    return MixerGeometry(
        name=name,
        effectiveness=np.asarray(rows, dtype=float),
        motor_names=tuple(motor_names),
    )


def quad_x() -> MixerGeometry:
    """Quad in X, motors numbered PX4-style.

    ::

        M4 (CCW)  front  M1 (CW)
                \\   ^   /
                 \\  |  /
                  \\ | /
                   \\|/
                   /|\\
                  / | \\
                 /  |  \\
        M2 (CW)  rear   M3 (CCW)

    Arms at 45 degrees, so each motor contributes equally to roll and pitch,
    scaled by ``sqrt(1/2)``. Yaw comes from the reaction torque of the props,
    so motors alternate rotation direction and the yaw row alternates sign.

    Yaw authority on a quad X is roughly an order of magnitude weaker than
    roll/pitch authority: roll uses thrust differential on a lever arm, yaw
    uses only prop drag torque. That asymmetry is why yaw is the axis that
    saturates first on a hard manoeuvre, and why giving yaw the lowest priority
    in the desaturation order is the standard (and correct) choice.
    """
    r = _SQRT_HALF
    return _geom(
        "quad_x",
        [
            # M1 front-right, M2 rear-left, M3 rear-right, M4 front-left
            [-r, +r, -r, +r],  # roll
            [+r, -r, -r, +r],  # pitch
            [+1, +1, -1, -1],  # yaw
            [0.25, 0.25, 0.25, 0.25],  # thrust (normalised: demand 1.0 = all motors at 1.0)
        ],
        ("M1 front-right CW", "M2 rear-left CCW", "M3 rear-right CCW", "M4 front-left CW"),
    )


def quad_plus() -> MixerGeometry:
    """Quad in plus configuration: arms along the body axes.

    Each motor affects exactly one of roll or pitch, which makes the mixing
    trivially readable and is why it is the configuration in every textbook.
    It is rarely flown because the front motor sits in the camera's view and
    because X gives you ``sqrt(2)`` more effective arm length per axis for the
    same frame size.
    """
    return _geom(
        "quad_plus",
        [
            [0.0, 0.0, -1.0, +1.0],  # roll: right motor down, left motor up
            [+1.0, -1.0, 0.0, 0.0],  # pitch: front/rear
            [+1.0, +1.0, -1.0, -1.0],  # yaw
            [0.25, 0.25, 0.25, 0.25],  # thrust (normalised)
        ],
        ("M1 front CW", "M2 rear CW", "M3 right CCW", "M4 left CCW"),
    )


def hexa_x() -> MixerGeometry:
    """Hexacopter in X: six arms at 60 degree spacing, starting 30 degrees off
    the nose.

    Motor ``j`` sits at azimuth ``theta_j``; its roll contribution is
    ``-sin(theta_j)`` and its pitch contribution is ``+cos(theta_j)``. Rotation
    directions alternate around the ring so the net reaction torque cancels in
    hover.

    A hexa is over-actuated: six motors for four control axes, so the
    allocation has a two-dimensional null space. That is the whole point --
    lose one motor and the remaining five can still (marginally) span the four
    axes, which a quad cannot. The pseudo-inverse used here picks the
    minimum-norm solution within that null space, which is also the
    minimum-power one.
    """
    angles = np.deg2rad(np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]))
    roll = -np.sin(angles)
    pitch = np.cos(angles)
    yaw = np.array([+1.0, -1.0, +1.0, -1.0, +1.0, -1.0])
    thrust = np.ones(6) / 6.0  # normalised: demand 1.0 = all motors at 1.0
    return _geom(
        "hexa_x",
        [roll, pitch, yaw, thrust],
        tuple(f"M{i + 1} @ {np.degrees(a):.0f}deg" for i, a in enumerate(angles)),
    )


def vtol_quadplane() -> MixerGeometry:
    """Quadplane VTOL: four lift motors in X plus one forward pusher.

    The pusher contributes no roll or pitch (it is on the centreline and
    roughly on the thrust axis) and no lift. It is listed as a fifth column
    with a zero column in the attitude rows, which means the pseudo-inverse
    will never use it to fix an attitude error -- correct, because it
    physically cannot.

    The consequence for allocation: in hover the pusher is *outside* the
    attitude allocation and should be commanded directly by the transition
    logic, not by the mixer's saturation handling. This class exposes it
    through :attr:`Mixer.passthrough_motors` so desaturation leaves it alone.
    A transition controller that lets attitude desaturation steal from the
    pusher will stall the wing halfway through a transition, which is an
    expensive way to learn about control allocation.
    """
    r = _SQRT_HALF
    return _geom(
        "vtol_quadplane",
        [
            [-r, +r, -r, +r, 0.0],
            [+r, -r, -r, +r, 0.0],
            [+1, +1, -1, -1, 0.0],
            [0.25, 0.25, 0.25, 0.25, 0.0],
        ],
        ("M1 lift FR", "M2 lift RL", "M3 lift RR", "M4 lift FL", "M5 pusher"),
    )


class Mixer:
    """Control allocation with saturation handling.

    Parameters
    ----------
    geometry:
        A :class:`MixerGeometry`, typically from :func:`quad_x` and friends.
    idle:
        Minimum motor command while armed. Real ESCs stop commutating below
        some duty cycle and a stopped motor on a multirotor is a crash, so
        armed motors are never commanded to zero. 0.05-0.10 is typical.
    max_command:
        Upper limit, normally 1.0.
    yaw_priority:
        How hard yaw defends itself against roll/pitch when the allocation
        saturates, in ``[0, 1]``.

        ``0.0`` (default) means yaw yields completely before a single percent
        of roll/pitch authority is given up. That is the right choice for a
        multirotor: losing yaw for 200 ms costs you a heading wobble, losing
        roll costs you the aircraft.

        ``1.0`` means yaw is never scaled and roll/pitch pay instead. Raise it
        above zero only when heading is genuinely load-bearing -- a gimbal-less
        mapping payload, or an airframe whose yaw authority is so weak that
        losing it entirely means losing heading control for seconds. Values in
        between linearly interpolate.
    passthrough_motors:
        Indices excluded from attitude allocation and from desaturation (a VTOL
        pusher, for example). Their commands are supplied separately.
    """

    def __init__(
        self,
        geometry: MixerGeometry,
        *,
        idle: float = 0.05,
        max_command: float = 1.0,
        yaw_priority: float = 0.4,
        passthrough_motors: Sequence[int] = (),
    ) -> None:
        if not 0.0 <= idle < max_command:
            raise ValueError("require 0 <= idle < max_command")
        if not 0.0 <= yaw_priority <= 1.0:
            raise ValueError("yaw_priority must be in [0, 1]")
        self.geometry = geometry
        self.idle = float(idle)
        self.max_command = float(max_command)
        self.yaw_priority = float(yaw_priority)
        self.passthrough_motors = tuple(int(i) for i in passthrough_motors)

        n = geometry.n_motors
        self._active = np.array(
            [j for j in range(n) if j not in self.passthrough_motors], dtype=int
        )
        self.E = geometry.effectiveness
        self.E_active = self.E[:, self._active]
        # Minimum-norm allocation. pinv rather than inv because a hexa is
        # over-actuated (wide E) and a VTOL's attitude rows are rank-deficient
        # in the pusher column; pinv handles both without a special case.
        self.E_pinv = np.linalg.pinv(self.E_active)
        self.last_saturation = 0.0
        self.last_thrust_scale = 1.0
        self.last_yaw_scale = 1.0

    # ------------------------------------------------------------------
    @property
    def n_motors(self) -> int:
        return self.geometry.n_motors

    def allocate_raw(
        self, roll: float, pitch: float, yaw: float, thrust: float
    ) -> NDArray[np.float64]:
        """Unsaturated pseudo-inverse allocation over the active motors."""
        demand = np.array([roll, pitch, yaw, thrust], dtype=float)
        return self.E_pinv @ demand

    def mix(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        thrust: float,
        *,
        passthrough: Optional[ArrayLike] = None,
    ) -> NDArray[np.float64]:
        """Allocate and desaturate. Returns commands in ``[idle, max_command]``.

        Algorithm, in priority order:

        1. Allocate roll/pitch, yaw and thrust separately with the
           pseudo-inverse, so each can be scaled independently.
        2a. **Yaw yields first.** It is scaled down to whatever command range
            is left after roll and pitch have taken their share;
            ``yaw_priority`` blends that back toward "yaw untouched".
        2b. **Only if that is not enough**, scale roll, pitch and yaw together
            by one common factor. They are never scaled individually: scaling
            one axis alone rotates the commanded torque vector and turns a roll
            input into a roll-plus-pitch input, which feels like a bent
            airframe and only shows up when you push hard.
        3. Shift the whole set by adjusting thrust until it fits. This is the
           step that sacrifices thrust to preserve attitude.
        4. Clip, as a final guarantee. After steps 2-3 the clip is a no-op to
           within floating point; it is kept because "should be" is not "is"
           when a caller passes a non-physical demand.

        Diagnostics, all set on every call and all worth logging:

        * :attr:`last_saturation` -- fraction of roll/pitch authority given up
          in step 2b (0 = none, 1 = everything). A rate loop that spends its
          time at ``last_saturation > 0`` is a rate loop that is not in control
          of the aircraft, and it is invisible in an attitude plot.
        * :attr:`last_yaw_scale` -- fraction of yaw demand retained in step 2a.
        * :attr:`last_thrust_scale` -- fraction of thrust demand delivered
          after step 3.
        """
        att_only = self.allocate_raw(roll, pitch, 0.0, 0.0)
        yaw_only = self.allocate_raw(0.0, 0.0, yaw, 0.0)
        thrust_only = self.allocate_raw(0.0, 0.0, 0.0, thrust)

        lo, hi = self.idle, self.max_command
        span = hi - lo
        self.last_yaw_scale = 1.0
        self.last_thrust_scale = 1.0

        # --- step 2a: yaw yields first ---------------------------------------
        # Yaw is scaled to whatever room is left after roll/pitch has taken its
        # share. ``yaw_priority`` then blends that back toward "yaw untouched".
        att_span = float(np.max(att_only) - np.min(att_only))
        yaw_span = float(np.max(yaw_only) - np.min(yaw_only))
        if yaw_span > 1e-12 and att_span + yaw_span > span:
            fits = float(np.clip((span - att_span) / yaw_span, 0.0, 1.0))
            scale = fits + self.yaw_priority * (1.0 - fits)
            self.last_yaw_scale = scale
            yaw_only = yaw_only * scale
        combined = att_only + yaw_only

        # --- step 2b: roll/pitch/yaw scaled together as a last resort --------
        excess = float(np.max(combined) - np.min(combined))
        att_scale = 1.0
        if excess > span and excess > 1e-12:
            att_scale = span / excess
            combined = combined * att_scale

        # --- step 3: move thrust to fit -------------------------------------
        u = combined + thrust_only
        low_violation = lo - float(np.min(u))
        high_violation = float(np.max(u)) - hi
        if low_violation > 0.0:
            u = u + low_violation
            high_violation = float(np.max(u)) - hi
        if high_violation > 0.0:
            u = u - high_violation

        # --- step 4: hard clip ------------------------------------------------
        out_full = np.full(self.n_motors, lo, dtype=float)
        out_full[self._active] = np.clip(u, lo, hi)

        if self.passthrough_motors:
            if passthrough is None:
                pt = np.full(len(self.passthrough_motors), lo)
            else:
                pt = np.asarray(passthrough, dtype=float).ravel()
                if pt.size != len(self.passthrough_motors):
                    raise ValueError(
                        f"passthrough must have {len(self.passthrough_motors)} elements"
                    )
            out_full[list(self.passthrough_motors)] = np.clip(pt, 0.0, hi)

        thrust_achieved = float(self.E[3, self._active] @ np.clip(u, lo, hi))
        self.last_thrust_scale = (
            thrust_achieved / thrust if abs(thrust) > 1e-12 else 1.0
        )
        self.last_saturation = float(1.0 - att_scale)
        return out_full

    # ------------------------------------------------------------------
    def achieved(self, commands: ArrayLike) -> NDArray[np.float64]:
        """Forward map: what ``[roll, pitch, yaw, thrust]`` the commands produce.

        Use this to check that desaturation preserved the *direction* of the
        attitude demand. That is the property that matters; the magnitude can
        legitimately shrink.
        """
        u = np.asarray(commands, dtype=float).ravel()
        if u.size != self.n_motors:
            raise ValueError(f"expected {self.n_motors} commands")
        return self.E @ u

    def attitude_direction_error(
        self, roll: float, pitch: float, achieved: ArrayLike
    ) -> float:
        """Angle in radians between demanded and achieved roll/pitch torque.

        Zero means desaturation only scaled the demand. Anything large means it
        rotated it, which is the failure mode naive per-motor clipping causes.
        """
        want = np.array([roll, pitch], dtype=float)
        got = np.asarray(achieved, dtype=float).ravel()[:2]
        nw, ng = float(np.linalg.norm(want)), float(np.linalg.norm(got))
        if nw < 1e-12 or ng < 1e-12:
            return 0.0
        cos = float(np.clip(np.dot(want, got) / (nw * ng), -1.0, 1.0))
        return float(np.arccos(cos))
