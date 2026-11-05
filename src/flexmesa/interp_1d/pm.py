# SPDX-License-Identifier: LGPL-3.0-or-later
"""Steffen (1990) piecewise-monotonic cubic interpolation (`interp_pm`).

Reference:
    Steffen, M. (1990), "A simple method for monotonic interpolation in one
    dimension", Astronomy and Astrophysics, 239, 443-450.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from flexmesa import auto_diff

SMACHEPS: float = 1.0e-13
PM_WORK_SIZE: int = 3
MP_WORK_SIZE: int = 14


def validate_1d_grid(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    min_points: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates strictly monotonic 1D knot coordinates and finite values.

    Args:
        x: 1D array of knot coordinates of shape ``(N,)`` in physical or
            dimensionless units (strictly increasing or strictly decreasing).
        y: 1D array of knot values of shape ``(N,)`` in physical or
            dimensionless units.
        min_points: Minimum number of knots required.

    Returns:
        Tuple ``(x_arr, y_arr)`` converted to 1D ``float64`` NumPy arrays of
        shape ``(N,)``.

    Raises:
        ValueError: If shapes mismatch, ``N < min_points``, values are
            non-finite, or ``x`` is not strictly monotonic.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
        raise ValueError("x and y must be 1D arrays of identical length.")
    if x_arr.size < min_points:
        raise ValueError(
            f"1D interpolation requires at least {min_points} points."
        )
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("x and y must contain only finite float64 values.")
    dx = np.diff(x_arr)
    if not (np.all(dx > 0.0) or np.all(dx < 0.0)):
        raise ValueError("x must be strictly increasing or decreasing.")
    return x_arr, y_arr


def _steffen_slope(s0: float, s1: float, p: float) -> float:
    """Applies Steffen (1990) monotonicity limiter to parabolic slope `p`.

    Args:
        s0: Left secant slope in physical units.
        s1: Right secant slope in physical units.
        p: Parabolic slope estimate at the shared knot.

    Returns:
        Monotonicity-limited first derivative at the knot.
    """
    if s0 * s1 <= 0.0:
        return 0.0
    if s0 * p <= 0.0:
        return 0.0
    lim = 2.0 * min(abs(s0), abs(s1))
    if abs(p) > lim:
        return float(np.copysign(lim, s0))
    return p


def _mk_pmcub(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    *,
    slope_only: bool,
) -> jnp.ndarray:
    """Shared implementation for `interp_pm` and `interp_pm_slopes_only`."""
    x_arr, y_arr = validate_1d_grid(x, y, min_points=2)
    nx = x_arr.size
    coeffs = np.zeros((nx, 4), dtype=np.float64)
    coeffs[:, 0] = y_arr

    if nx == 2:
        coeffs[0, 1] = (y_arr[1] - y_arr[0]) / (x_arr[1] - x_arr[0])
        coeffs[1, 1] = 0.0
        return jnp.asarray(coeffs, dtype=jnp.float64)

    h = x_arr[1:] - x_arr[:-1]
    s = (y_arr[1:] - y_arr[:-1]) / h

    p = (s[:-1] * h[1:] + s[1:] * h[:-1]) / (h[:-1] + h[1:])
    for i in range(1, nx - 1):
        coeffs[i, 1] = _steffen_slope(
            float(s[i - 1]), float(s[i]), float(p[i - 1])
        )

    p_left = float(
        s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1])
    )
    coeffs[0, 1] = _steffen_slope(float(s[0]), float(s[0]), p_left)

    p_right = float(
        s[-1] * (1.0 + h[-1] / (h[-1] + h[-2]))
        - s[-2] * h[-1] / (h[-1] + h[-2])
    )
    coeffs[-1, 1] = _steffen_slope(float(s[-1]), float(s[-1]), p_right)

    if slope_only:
        return jnp.asarray(coeffs, dtype=jnp.float64)

    coeffs[:-1, 2] = (3.0 * s - 2.0 * coeffs[:-1, 1] - coeffs[1:, 1]) / h
    coeffs[:-1, 3] = (coeffs[:-1, 1] + coeffs[1:, 1] - 2.0 * s) / (h * h)
    return jnp.asarray(coeffs, dtype=jnp.float64)


def interp_pm(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Computes Steffen (1990) piecewise-monotonic cubic coefficients.

    On subinterval ``[x[i], x[i + 1]]`` with offset ``dx = x_eval - x[i]``, the
    interpolant is evaluated via Horner's rule:
    ``f(x_eval) = c[i, 0] + dx * (c[i, 1] + dx * (c[i, 2] + dx * c[i, 3]))``.

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.

    Returns:
        JAX ``float64`` array of shape ``(N, 4)`` containing the local cubic
        coefficients ``[c0, c1, c2, c3]`` at each knot.

    Raises:
        ValueError: If ``x`` and ``y`` fail grid validation.
    """
    return _mk_pmcub(x, y, slope_only=False)


def interp_pm_slopes_only(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Computes only the knot values and Steffen (1990) monotonic slopes.

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.

    Returns:
        JAX ``float64`` array of shape ``(N, 4)`` with ``[:, 0] = y``,
        ``[:, 1] = df/dx``, and ``[:, 2:] = 0``.

    Raises:
        ValueError: If ``x`` and ``y`` fail grid validation.
    """
    return _mk_pmcub(x, y, slope_only=True)


def interp_4pt_pm(
    x: Sequence[float] | np.ndarray | jnp.ndarray,
    y: Sequence[float] | np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Computes cubic coefficients on the middle interval of a 4-point stencil.

    For ``x[1] <= x_eval <= x[2]`` and ``dx = x_eval - x[1]``, the interpolated
    value is ``y[1] + dx * (a[0] + dx * (a[1] + dx * a[2]))``.

    Args:
        x: 4-element strictly monotonic knot coordinate sequence of shape
            ``(4,)`` in physical or dimensionless units.
        y: 4-element knot value sequence of shape ``(4,)`` in physical or
            dimensionless units.

    Returns:
        JAX ``float64`` array ``[a0, a1, a2]`` of shape ``(3,)``.

    Raises:
        ValueError: If ``x`` and ``y`` do not have length 4 or fail grid
            validation.
    """
    x_arr, y_arr = validate_1d_grid(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        min_points=4,
    )
    if x_arr.size != 4:
        raise ValueError("interp_4pt_pm requires exactly 4 points.")
    return interp_pm(x_arr, y_arr)[1, 1:]


def _steffen_slope_autodiff(
    s0: auto_diff.DualOrder1,
    s1: auto_diff.DualOrder1,
    p: auto_diff.DualOrder1,
) -> auto_diff.DualOrder1:
    """Applies the Steffen (1990) limiter to ``DualOrder1`` slopes."""
    n_vars = int(s0.d1.shape[0])
    if s0.val * s1.val <= 0.0 or s0.val * p.val <= 0.0:
        return auto_diff.DualOrder1.constant(0.0, n_vars)
    two_s0 = 2.0 * abs(s0)
    two_s1 = 2.0 * abs(s1)
    if abs(p.val) > two_s0.val or abs(p.val) > two_s1.val:
        lim = two_s0 if two_s0.val <= two_s1.val else two_s1
        return lim if s0.val >= 0.0 else -lim
    return p


def interp_pm_autodiff(
    x: Sequence[auto_diff.DualOrder1],
    y: Sequence[auto_diff.DualOrder1],
) -> list[
    tuple[
        auto_diff.DualOrder1,
        auto_diff.DualOrder1,
        auto_diff.DualOrder1,
        auto_diff.DualOrder1,
    ]
]:
    """Computes Steffen (1990) cubic coefficients on ``DualOrder1`` knots.

    Args:
        x: Sequence of ``N >= 2`` strictly monotonic ``DualOrder1`` knot
            coordinates in physical or dimensionless units.
        y: Sequence of ``N`` ``DualOrder1`` knot values in physical or
            dimensionless units.

    Returns:
        List of ``N`` 4-tuples ``(c0, c1, c2, c3)`` of ``DualOrder1`` values.

    Raises:
        ValueError: If ``x`` and ``y`` have mismatched lengths, fewer than 2
            points, non-finite values/derivatives, or non-monotonic abscissas.
    """
    nx = len(x)
    if nx != len(y) or nx < 2:
        raise ValueError("x and y must have matching length >= 2.")
    if any(not np.all(np.isfinite(pt.d1)) for pt in x) or any(
        not np.all(np.isfinite(pt.d1)) for pt in y
    ):
        raise ValueError("x and y dual derivatives must be finite.")
    x_primal = np.array([pt.val for pt in x], dtype=np.float64)
    y_primal = np.array([pt.val for pt in y], dtype=np.float64)
    validate_1d_grid(x_primal, y_primal, min_points=2)

    n_vars = int(x[0].d1.shape[0])
    zero = auto_diff.DualOrder1.constant(0.0, n_vars)
    if nx == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return [(y[0], slope, zero, zero), (y[1], zero, zero, zero)]

    h = [x[i + 1] - x[i] for i in range(nx - 1)]
    s = [(y[i + 1] - y[i]) / h[i] for i in range(nx - 1)]
    d: list[auto_diff.DualOrder1] = [zero] * nx
    for i in range(1, nx - 1):
        p_i = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i])
        d[i] = _steffen_slope_autodiff(s[i - 1], s[i], p_i)

    p0 = s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1])
    d[0] = _steffen_slope_autodiff(s[0], s[0], p0)

    pn = s[-1] * (1.0 + h[-1] / (h[-1] + h[-2])) - s[-2] * h[-1] / (
        h[-1] + h[-2]
    )
    d[-1] = _steffen_slope_autodiff(s[-1], s[-1], pn)

    out: list[
        tuple[
            auto_diff.DualOrder1,
            auto_diff.DualOrder1,
            auto_diff.DualOrder1,
            auto_diff.DualOrder1,
        ]
    ] = []
    for i in range(nx - 1):
        c2 = (3.0 * s[i] - 2.0 * d[i] - d[i + 1]) / h[i]
        c3 = (d[i] + d[i + 1] - 2.0 * s[i]) / (h[i] * h[i])
        out.append((y[i], d[i], c2, c3))
    out.append((y[-1], d[-1], zero, zero))
    return out
