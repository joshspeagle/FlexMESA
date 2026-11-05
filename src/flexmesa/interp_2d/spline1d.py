# SPDX-License-Identifier: LGPL-3.0-or-later
"""One-dimensional C^2 cubic spline helpers (`mkspline_db`, `splinck_db`).

Ports the 1D $C^2$ tridiagonal spline solver (`V_SPLINE_db`, `cspline_db`,
`mkspline_db`) and grid uniformity checker (`splinck_db`) from upstream
PSPLINE / MESA ``interp_2d/private/bicub_db.f`` into ``float64`` NumPy/JAX.
"""

from __future__ import annotations

import enum

import jax.numpy as jnp
import numpy as np


class BicubicBc(enum.IntEnum):
    """Boundary condition codes for 1D and 2D cubic splines (`-1..=7`).

    Attributes:
        PERIODIC: Periodic boundary condition (`-1`).
        NOT_A_KNOT: Not-a-knot boundary condition (`0`, MESA default).
        FIRST_DERIV: Match specified first derivative (`1`).
        SECOND_DERIV: Match specified second derivative (`2`).
        ZERO_FIRST_DERIV: Zero first derivative (`3`).
        ZERO_SECOND_DERIV: Natural spline / zero second derivative (`4`).
        DIVIDED_DIFF_1: Match first divided difference (`5`).
        DIVIDED_DIFF_2: Match second divided difference (`6`).
        DIVIDED_DIFF_3: Match third divided difference (`7`).
    """

    PERIODIC = -1
    NOT_A_KNOT = 0
    FIRST_DERIV = 1
    SECOND_DERIV = 2
    ZERO_FIRST_DERIV = 3
    ZERO_SECOND_DERIV = 4
    DIVIDED_DIFF_1 = 5
    DIVIDED_DIFF_2 = 6
    DIVIDED_DIFF_3 = 7


def splinck_db(
    x: np.ndarray | jnp.ndarray,
    ztol: float = 1.0e-3,
    axis: str = "x",
) -> bool:
    """Checks strict ascending order and approximate uniform spacing of `x`.

    Args:
        x: 1D coordinate array of shape ``(N,)`` in physical or dimensionless
            units.
        ztol: Relative spacing tolerance (default ``1e-3``).
        axis: Axis label for error reporting.

    Returns:
        ``True`` (`ilinx = 1`) if uniformly spaced within ``ztol``, ``False``
        (`ilinx = 2`) otherwise.

    Raises:
        ValueError: If ``x`` contains non-finite values or is not strictly
            ascending.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{axis} grid must be a finite 1D float64 array.")
    if arr.size <= 1:
        return True
    diffs = np.diff(arr)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{axis} grid must be strictly ascending.")
    dxavg = (arr[-1] - arr[0]) / float(arr.size - 1)
    zeps = abs(ztol * dxavg)
    return bool(np.all(np.abs(diffs - dxavg) <= zeps))


def reshape_grid_2d(
    z: np.ndarray | jnp.ndarray,
    nx: int,
    ny: int,
    name: str = "z",
) -> np.ndarray:
    """Validates and reshapes a 2D or flat grid array to ``(nx, ny)``.

    Args:
        z: Grid values of shape ``(nx, ny)`` or flat ``(nx * ny,)``.
        nx: Expected x-grid length.
        ny: Expected y-grid length.
        name: Parameter name for error reporting.

    Returns:
        2D ``float64`` NumPy array of shape ``(nx, ny)``.

    Raises:
        ValueError: If ``z`` does not have shape ``(nx, ny)`` or ``(nx * ny,)``
            or contains non-finite values.
    """
    arr = np.asarray(z, dtype=np.float64)
    if arr.shape not in ((nx, ny), (nx * ny,)):
        raise ValueError(
            f"{name} must have shape ({nx}, {ny}) or ({nx * ny},), "
            f"got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite float64 values.")
    return arr.reshape(nx, ny)


def _v_spline_db(
    k_bc1: int,
    k_bcn: int,
    x: np.ndarray,
    f: np.ndarray,
) -> None:
    """In-place port of PSPLINE ``V_SPLINE_db`` on ``f`` of shape ``(N, 4)``.

    Solves the tridiagonal (or cyclic tridiagonal when ``k_bc1 == -1``)
    continuity system for $C^2$ cubic spline moments and stores ``f'(x_i)``,
    ``f''(x_i)``, and ``f'''(x_i)`` in ``f[:, 1]``, ``f[:, 2]``, and
    ``f[:, 3]``.

    Args:
        k_bc1: Left boundary condition code in ``-1..=7``.
        k_bcn: Right boundary condition code in ``0..=7``.
        x: 1D ascending knot array of shape ``(N,)`` in physical units.
        f: Coefficient array of shape ``(N, 4)`` modified in place so
            ``f[:, 2]`` holds ``f''(x_i)`` on exit.
    """
    n = x.size
    imin = 0
    imax = n - 1

    # Evaluate boundary derivative or divided-difference targets at x[0], x[-1].
    a1 = (
        float(f[0, 1])
        if k_bc1 == 1
        else float((f[1, 0] - f[0, 0]) / (x[1] - x[0]))
        if k_bc1 == 5
        else 0.0
    )
    b1 = (
        float(f[0, 2])
        if k_bc1 == 2
        else float(
            2.0
            * (
                (f[2, 0] - f[1, 0]) / (x[2] - x[1])
                - (f[1, 0] - f[0, 0]) / (x[1] - x[0])
            )
            / (x[2] - x[0])
        )
        if k_bc1 == 6
        else 0.0
    )
    an = (
        float(f[-1, 1])
        if k_bcn == 1
        else float((f[-1, 0] - f[-2, 0]) / (x[-1] - x[-2]))
        if k_bcn == 5
        else 0.0
    )
    bn = (
        float(f[-1, 2])
        if k_bcn == 2
        else float(
            2.0
            * (
                (f[-1, 0] - f[-2, 0]) / (x[-1] - x[-2])
                - (f[-2, 0] - f[-3, 0]) / (x[-2] - x[-3])
            )
            / (x[-1] - x[-3])
        )
        if k_bcn == 6
        else 0.0
    )

    # Assemble interior tridiagonal matrix diagonals and RHS differences.
    f[-1, 1:] = 0.0
    f[0, 3] = x[1] - x[0]
    f[1, 2] = (f[1, 0] - f[0, 0]) / f[0, 3]
    for i in range(1, n - 1):
        f[i, 3] = x[i + 1] - x[i]
        f[i, 1] = 2.0 * (f[i - 1, 3] + f[i, 3])
        f[i + 1, 2] = (f[i + 1, 0] - f[i, 0]) / f[i, 3]
        f[i, 2] = f[i + 1, 2] - f[i, 2]

    elem21 = float(f[0, 3])
    elemnn1 = float(f[n - 2, 3])

    # Apply left boundary condition row at index 0 (or condense into row 1).
    if k_bc1 == -1:
        f[0, 1] = 2.0 * (f[0, 3] + f[n - 2, 3])
        f[0, 2] = (f[1, 0] - f[0, 0]) / f[0, 3] - (f[-1, 0] - f[-2, 0]) / f[
            n - 2, 3
        ]
    elif k_bc1 in (1, 3, 5):
        f[0, 1] = 2.0 * f[0, 3]
        f[0, 2] = (f[1, 0] - f[0, 0]) / f[0, 3] - a1
    elif k_bc1 in (2, 4, 6):
        f[0, 1] = 2.0 * f[0, 3]
        f[0, 2] = f[0, 3] * b1 / 3.0
        f[0, 3] = 0.0
    elif k_bc1 == 7:
        f[0, 1] = -f[0, 3]
        f[0, 2] = f[2, 2] / (x[3] - x[1]) - f[1, 2] / (x[2] - x[0])
        f[0, 2] = f[0, 2] * f[0, 3] * f[0, 3] / (x[3] - x[0])
    else:
        imin = 1
        f[1, 1] = f[0, 3] + 2.0 * f[1, 3]
        f[1, 2] = f[1, 2] * f[1, 3] / (f[0, 3] + f[1, 3])

    # Apply right boundary condition row at n - 1 (or condense into n - 2).
    if k_bcn in (1, 3, 5):
        f[-1, 1] = 2.0 * f[n - 2, 3]
        f[-1, 2] = -(f[-1, 0] - f[-2, 0]) / f[n - 2, 3] + an
    elif k_bcn in (2, 4, 6):
        f[-1, 1] = 2.0 * f[n - 2, 3]
        f[-1, 2] = f[n - 2, 3] * bn / 3.0
        elemnn1 = 0.0
    elif k_bcn == 7:
        f[-1, 1] = -f[n - 2, 3]
        f[-1, 2] = f[n - 2, 2] / (x[-1] - x[-3]) - f[n - 3, 2] / (x[-2] - x[-4])
        f[-1, 2] = -f[-1, 2] * f[n - 2, 3] * f[n - 2, 3] / (x[-1] - x[-4])
    elif k_bc1 != -1:
        imax = n - 2
        f[n - 2, 1] = 2.0 * f[n - 3, 3] + f[n - 2, 3]
        f[n - 2, 2] = f[n - 2, 2] * f[n - 3, 3] / (f[n - 2, 3] + f[n - 3, 3])

    # Forward elimination and back-substitution for cyclic or standard system.
    if k_bc1 == -1:
        wk = np.zeros(n, dtype=np.float64)
        wk[0] = f[n - 2, 3]
        wk[n - 3] = f[n - 3, 3]
        wk[n - 2] = f[n - 2, 3]
        for i in range(1, n - 2):
            t = f[i - 1, 3] / f[i - 1, 1]
            f[i, 1] -= t * f[i - 1, 3]
            f[i, 2] -= t * f[i - 1, 2]
            wk[i] -= t * wk[i - 1]
            q = wk[n - 2] / f[i - 1, 1]
            wk[n - 2] = -q * f[i - 1, 3]
            f[n - 2, 1] -= q * wk[i - 1]
            f[n - 2, 2] -= q * f[i - 1, 2]
        wk[n - 2] += f[n - 3, 3]
        t = wk[n - 2] / f[n - 3, 1]
        f[n - 2, 1] -= t * wk[n - 3]
        f[n - 2, 2] -= t * f[n - 3, 2]
        f[n - 2, 2] /= f[n - 2, 1]
        f[n - 3, 2] = (f[n - 3, 2] - wk[n - 3] * f[n - 2, 2]) / f[n - 3, 1]
        for ib in range(3, n):
            i = n - 1 - ib
            f[i, 2] = (
                f[i, 2] - f[i, 3] * f[i + 1, 2] - wk[i] * f[n - 2, 2]
            ) / f[i, 1]
        f[-1, 2] = f[0, 2]
    else:
        for i in range(imin + 1, imax + 1):
            if i == n - 2 and imax == n - 2:
                t = (f[i - 1, 3] - f[i, 3]) / f[i - 1, 1]
            elif i == 1:
                t = elem21 / f[i - 1, 1]
            elif i == n - 1:
                t = elemnn1 / f[i - 1, 1]
            else:
                t = f[i - 1, 3] / f[i - 1, 1]
            if i == imin + 1 and imin == 1:
                f[i, 1] -= t * (f[i - 1, 3] - f[i - 2, 3])
            else:
                f[i, 1] -= t * f[i - 1, 3]
            f[i, 2] -= t * f[i - 1, 2]
        f[imax, 2] /= f[imax, 1]
        for ib in range(1, imax - imin + 1):
            i = imax - ib
            if i == 1 and imin == 1:
                f[i, 2] = (f[i, 2] - (f[i, 3] - f[i - 1, 3]) * f[i + 1, 2]) / f[
                    i, 1
                ]
            else:
                f[i, 2] = (f[i, 2] - f[i, 3] * f[i + 1, 2]) / f[i, 1]
        f[0, 3] = x[1] - x[0]
        f[n - 2, 3] = x[-1] - x[-2]
        if k_bc1 <= 0 or k_bc1 > 7:
            f[0, 2] = (f[1, 2] * (f[0, 3] + f[1, 3]) - f[2, 2] * f[0, 3]) / f[
                1, 3
            ]
        if k_bcn <= 0 or k_bcn > 7:
            f[-1, 2] = (
                f[-2, 2] + (f[-2, 2] - f[-3, 2]) * f[n - 2, 3] / f[n - 3, 3]
            )

    # Convert moments sigma_i = f''(x_i) / 6 into f', f'', and f'''.
    for i in range(n - 1):
        h_i = f[i, 3]
        sig_i = f[i, 2]
        sig_next = f[i + 1, 2]
        f[i, 1] = (f[i + 1, 0] - f[i, 0]) / h_i - h_i * (sig_next + 2.0 * sig_i)
        f[i, 3] = 6.0 * (sig_next - sig_i) / h_i
        f[i, 2] = 6.0 * sig_i

    if k_bc1 == -1:
        f[-1, 1:] = f[0, 1:]
    else:
        hn = x[-1] - x[-2]
        f[-1, 1] = f[-2, 1] + hn * (f[-2, 2] + 0.5 * hn * f[-2, 3])
        f[-1, 2] = f[-2, 2] + hn * f[-2, 3]
        f[-1, 3] = f[-2, 3]
        if k_bcn in (1, 3, 5):
            f[-1, 1] = an
        elif k_bcn in (2, 4, 6):
            f[-1, 2] = bn


def interp_mkspline_db(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    ibcxmin: BicubicBc | int = BicubicBc.NOT_A_KNOT,
    bcxmin: float = 0.0,
    ibcxmax: BicubicBc | int = BicubicBc.NOT_A_KNOT,
    bcxmax: float = 0.0,
) -> tuple[jnp.ndarray, bool]:
    """Computes second derivatives ``f''(x_i)`` of a 1D C^2 cubic spline.

    Args:
        x: Strictly ascending knot coordinates of shape ``(N,)`` (``N >= 4``)
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.
        ibcxmin: Left boundary condition code in ``-1..=7``.
        bcxmin: Left boundary value (used when ``ibcxmin in (1, 2)``).
        ibcxmax: Right boundary condition code in ``0..=7``.
        bcxmax: Right boundary value (used when ``ibcxmax in (1, 2)``).

    Returns:
        Tuple ``(fxx, uniform)`` where ``fxx`` has shape ``(N,)`` (`float64`)
        and ``uniform`` indicates even spacing within ``1e-3``.

    Raises:
        ValueError: If ``x`` or ``y`` has fewer than 4 points, mismatched
            shapes, non-ascending knot coordinates, non-finite values, or
            invalid boundary condition codes.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.shape != x_arr.shape or x_arr.size < 4:
        raise ValueError("x and y must be 1D arrays of matching length >= 4.")
    if not (
        np.all(np.isfinite(y_arr))
        and np.isfinite(bcxmin)
        and np.isfinite(bcxmax)
    ):
        raise ValueError(
            "y, bcxmin, and bcxmax must contain finite float64 values."
        )
    bc1 = int(ibcxmin)
    bcn = int(ibcxmax)
    if (
        bc1 < -1
        or bc1 > 7
        or bcn < -1
        or bcn > 7
        or (bc1 >= 0 and bcn < 0)
    ):
        raise ValueError("Invalid boundary condition code.")
    uniform = splinck_db(x_arr, 1.0e-3, "x")

    fspl4 = np.zeros((x_arr.size, 4), dtype=np.float64)
    fspl4[:, 0] = y_arr
    if bc1 == 1:
        fspl4[0, 1] = bcxmin
    elif bc1 == 2:
        fspl4[0, 2] = bcxmin
    if bcn == 1:
        fspl4[-1, 1] = bcxmax
    elif bcn == 2:
        fspl4[-1, 2] = bcxmax

    _v_spline_db(bc1, bcn, x_arr, fspl4)
    fxx = np.zeros(x_arr.size, dtype=np.float64)
    fxx[:-1] = fspl4[:-1, 2]
    fxx[-1] = fspl4[-2, 2] + (x_arr[-1] - x_arr[-2]) * fspl4[-2, 3]
    return jnp.asarray(fxx, dtype=jnp.float64), uniform


mkspline_db = interp_mkspline_db
