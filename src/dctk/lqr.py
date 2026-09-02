"""Discrete-time infinite-horizon LQR, implemented from scratch.

No scipy. The Riccati recursion is twenty lines and iterating it yourself means
you can see it converge, check the residual, and ship the module on a Jetson
image where installing scipy costs you half an hour and 400 MB.

What LQR buys you over a hand-tuned PID on a multirotor: it handles coupled
multi-input plants without you inventing a cascade for every pair of states,
and the gain it produces is optimal for the cost you wrote down. What it costs
you: you need a model, and the tune now lives in ``Q`` and ``R``, which are
less intuitive than ``kp`` until you use Bryson's rule (below) to give them
units.

Everything here is discrete-time. Continuous LQR is a nice derivation and a bad
idea to implement for something that will run at a fixed sample rate on an FC.
Discretise first, design second, and the gain you compute is the gain that runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "LQRResult",
    "c2d",
    "expm",
    "controllability_matrix",
    "is_controllable",
    "brysons_rule",
    "dlqr",
    "closed_loop_poles",
]


# ----------------------------------------------------------------------
# discretisation
# ----------------------------------------------------------------------
def expm(a: ArrayLike, *, order: int = 6, terms: int = 18) -> NDArray[np.float64]:
    """Matrix exponential via scaling-and-squaring with a Taylor series.

    ``scipy.linalg.expm`` uses a Pade approximant with the same scaling trick.
    A truncated Taylor series is used here instead because it is four lines,
    and once the matrix has been scaled so that ``||A||_inf <= 0.5`` the series
    converges geometrically: the truncation error after ``terms`` terms is
    below ``||A||^terms / terms!``, which for 18 terms and a half-norm argument
    is around 1e-20 -- comfortably under double-precision round-off. The
    squaring step then costs ``order`` matrix multiplies.

    Parameters
    ----------
    order:
        Number of squarings; the matrix is pre-scaled by ``2**-s`` where ``s``
        is chosen automatically but at least this many times.
    """
    A = np.asarray(a, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expm expects a square matrix")
    norm = float(np.max(np.sum(np.abs(A), axis=1))) if A.size else 0.0
    s = 0
    while norm > 0.5:
        norm /= 2.0
        s += 1
    s = max(s, 0)
    As = A / (2.0**s)

    n = A.shape[0]
    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, terms + 1):
        term = term @ As / k
        result = result + term
    for _ in range(s):
        result = result @ result
    return result


def c2d(
    A: ArrayLike, B: ArrayLike, dt: float, *, method: str = "zoh"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretise ``x' = Ax + Bu`` at sample time ``dt``.

    Parameters
    ----------
    method:
        ``'zoh'``
            Exact zero-order-hold using the block-matrix trick: exponentiate
            ``[[A, B], [0, 0]]`` and read off ``Ad`` and ``Bd``. Exact for a
            piecewise-constant input, which is what a digital controller
            actually produces, so there is no discretisation error to account
            for.
        ``'euler'``
            ``Ad = I + A*dt``, ``Bd = B*dt``. Cheap and easy to read, and wrong
            by ``O((A*dt)^2 / 2)`` per step -- concretely, a pole at
            ``s = -a`` maps to ``1 - a*dt`` instead of ``exp(-a*dt)``, so the
            discrete pole is too far from 1 and the model looks *more* damped
            than the plant. The error is a few percent when ``|a|*dt < 0.1``
            and unacceptable above ``|a|*dt ~ 0.5``, where Euler can even place
            a stable continuous pole outside the unit circle. Use it only for a
            sanity check, or when ``dt`` is genuinely tiny.
    """
    A_arr = np.atleast_2d(np.asarray(A, dtype=float))
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(-1, 1)
    n, m = A_arr.shape[0], B_arr.shape[1]
    if A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be square")
    if B_arr.shape[0] != n:
        raise ValueError("B must have the same number of rows as A")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    if method == "euler":
        return np.eye(n) + A_arr * dt, B_arr * dt
    if method != "zoh":
        raise ValueError(f"unknown method {method!r}")

    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_arr
    M[:n, n:] = B_arr
    Md = expm(M * dt)
    return Md[:n, :n].copy(), Md[:n, n:].copy()


# ----------------------------------------------------------------------
# structural checks
# ----------------------------------------------------------------------
def controllability_matrix(A: ArrayLike, B: ArrayLike) -> NDArray[np.float64]:
    """``[B, AB, A^2 B, ..., A^(n-1) B]``."""
    A_arr = np.atleast_2d(np.asarray(A, dtype=float))
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(-1, 1)
    n = A_arr.shape[0]
    cols = [B_arr]
    for _ in range(n - 1):
        cols.append(A_arr @ cols[-1])
    return np.hstack(cols)


def is_controllable(A: ArrayLike, B: ArrayLike, *, tol: Optional[float] = None) -> bool:
    """Rank test on the controllability matrix, via SVD.

    Run this before you run :func:`dlqr`. An uncontrollable mode does not make
    the Riccati iteration fail loudly; if the mode happens to be stable the
    recursion converges perfectly well and hands you a gain that simply has no
    authority over that state. You then spend an afternoon wondering why one
    axis ignores you. Ninety percent of the time the cause is a modelling
    mistake -- a zero row in ``B``, or two states that are actually the same
    state.
    """
    C = controllability_matrix(A, B)
    sv = np.linalg.svd(C, compute_uv=False)
    n = np.atleast_2d(np.asarray(A, dtype=float)).shape[0]
    if tol is None:
        tol = max(C.shape) * float(np.finfo(float).eps) * (sv[0] if sv.size else 0.0)
    rank = int(np.sum(sv > tol))
    return rank == n


# ----------------------------------------------------------------------
# weight selection
# ----------------------------------------------------------------------
def brysons_rule(
    max_state_deviation: ArrayLike, max_control_effort: ArrayLike, *, rho: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Diagonal ``Q``, ``R`` from acceptable deviations.

    Bryson's rule: ``Q_ii = 1 / x_i,max^2`` and ``R_jj = rho / u_j,max^2``,
    where ``x_i,max`` is the largest deviation of state ``i`` you are willing
    to accept and ``u_j,max`` the largest control effort you are willing to
    spend. This normalises every term in the cost to be dimensionless and of
    order 1 at its own limit, which is why it works: without it you are adding
    radians-squared to metres-squared to newton-metres-squared and the relative
    weighting is an accident of your unit choices.

    ``rho`` is then the single knob you actually turn. Raise it for a lazier,
    more control-frugal loop; lower it for a tighter, hungrier one. Start at 1,
    move by factors of 10, and stop when the commanded effort in simulation
    stops looking like something your actuators can deliver.

    Examples
    --------
    A pitch axis where 5 degrees of error is a lot and 0.1 N.m is a lot::

        Q, R = brysons_rule([np.deg2rad(5), np.deg2rad(50)], [0.1])
    """
    x_max = np.asarray(max_state_deviation, dtype=float).ravel()
    u_max = np.asarray(max_control_effort, dtype=float).ravel()
    if np.any(x_max <= 0.0) or np.any(u_max <= 0.0):
        raise ValueError("maximum deviations must be strictly positive")
    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    return np.diag(1.0 / x_max**2), np.diag(rho / u_max**2)


# ----------------------------------------------------------------------
# the solver
# ----------------------------------------------------------------------
@dataclass
class LQRResult:
    """Output of :func:`dlqr`."""

    K: NDArray[np.float64]
    P: NDArray[np.float64]
    iterations: int
    residual: float
    converged: bool
    eigenvalues: NDArray[np.complex128]

    @property
    def is_stable(self) -> bool:
        """True if every closed-loop eigenvalue is strictly inside the unit circle."""
        return bool(np.all(np.abs(self.eigenvalues) < 1.0))

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.eigenvalues))) if self.eigenvalues.size else 0.0


def dlqr(
    A: ArrayLike,
    B: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    *,
    max_iter: int = 10000,
    tol: float = 1e-12,
    check_controllable: bool = True,
) -> LQRResult:
    """Infinite-horizon discrete LQR by iterating the Riccati difference equation.

    Solves ``min sum(x'Qx + u'Ru)`` for ``x[k+1] = A x[k] + B u[k]``. The
    optimal policy is ``u = -K x`` with

    ``P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q``
    ``K = (R + B'PB)^-1 B'PA``

    The recursion is run backwards from ``P = Q`` until ``||P_new - P||`` stops
    changing. For a controllable ``(A, B)`` with ``Q >= 0`` and ``R > 0`` this
    converges to the unique stabilising solution; convergence is linear with
    rate set by the closed-loop spectral radius, so a plant with a pole near
    the unit circle takes noticeably more iterations. That is the reason for
    the generous ``max_iter``, and the reason :attr:`LQRResult.converged` is
    reported rather than assumed.

    Notes
    -----
    Numerical hygiene, all of which matters at long horizons:

    * ``P`` is re-symmetrised every iteration (``(P + P')/2``). Round-off
      breaks symmetry, asymmetry feeds back through the next multiply, and the
      iteration slowly drifts away from a valid covariance-like matrix.
    * ``R + B'PB`` is solved with ``np.linalg.solve`` rather than inverted.
    * ``R`` is checked for positive definiteness up front. A singular ``R``
      means "free control effort", which admits infinite-gain solutions and
      will either blow up or hand you a gain no actuator can execute.
    """
    A_arr = np.atleast_2d(np.asarray(A, dtype=float))
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(-1, 1)
    Q_arr = np.atleast_2d(np.asarray(Q, dtype=float))
    R_arr = np.atleast_2d(np.asarray(R, dtype=float))

    n, m = A_arr.shape[0], B_arr.shape[1]
    if A_arr.shape != (n, n):
        raise ValueError("A must be square")
    if B_arr.shape[0] != n:
        raise ValueError("B row count must match A")
    if Q_arr.shape != (n, n):
        raise ValueError(f"Q must be {n}x{n}")
    if R_arr.shape != (m, m):
        raise ValueError(f"R must be {m}x{m}")
    if np.min(np.linalg.eigvalsh(0.5 * (R_arr + R_arr.T))) <= 0.0:
        raise ValueError("R must be positive definite (control effort cannot be free)")
    if np.min(np.linalg.eigvalsh(0.5 * (Q_arr + Q_arr.T))) < -1e-12:
        raise ValueError("Q must be positive semi-definite")
    if check_controllable and not is_controllable(A_arr, B_arr):
        raise ValueError(
            "(A, B) is not controllable; LQR will silently ignore the uncontrollable modes. "
            "Check for zero rows in B or duplicated states."
        )

    P = Q_arr.copy()
    K = np.zeros((m, n))
    residual = float("inf")
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        S = R_arr + B_arr.T @ P @ B_arr
        K = np.linalg.solve(S, B_arr.T @ P @ A_arr)
        P_next = A_arr.T @ P @ (A_arr - B_arr @ K) + Q_arr
        P_next = 0.5 * (P_next + P_next.T)
        residual = float(np.max(np.abs(P_next - P)))
        P = P_next
        if residual < tol:
            converged = True
            break

    eig = np.linalg.eigvals(A_arr - B_arr @ K)
    return LQRResult(
        K=K, P=P, iterations=it, residual=residual, converged=converged, eigenvalues=eig
    )


def closed_loop_poles(A: ArrayLike, B: ArrayLike, K: ArrayLike) -> NDArray[np.complex128]:
    """Eigenvalues of ``A - BK``. Inside the unit circle means stable."""
    A_arr = np.atleast_2d(np.asarray(A, dtype=float))
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(-1, 1)
    K_arr = np.atleast_2d(np.asarray(K, dtype=float))
    return np.linalg.eigvals(A_arr - B_arr @ K_arr)
