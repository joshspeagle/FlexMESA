# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spline evaluation, definite integration, and remeshing (`interp_1d`)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np

from flexmesa import auto_diff
from flexmesa.interp_1d import pm


def _locate_intervals_jax(
    init_x: jnp.ndarray,
    xs: jnp.ndarray,
) -> jnp.ndarray:
    """Locates 0-based subinterval indices ``i in [0, nx - 2]`` for queries."""
    nx = init_x.shape[0]
    increasing = init_x[-1] > init_x[0]
    grid_asc = jnp.where(increasing, init_x, init_x[::-1])
    pos = jnp.searchsorted(grid_asc, xs, side="right") - 1
    idx_asc = jnp.clip(pos, 0, nx - 2)
    return jnp.where(increasing, idx_asc, (nx - 2) - idx_asc)


def interp_values_and_slopes(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates a 1D cubic spline and its first derivative at query points.

    Supports ``jax.grad``, ``jax.hessian``, and ``jax.vmap`` with respect to
    both ``xs`` and ``coeffs``.

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)`` in
            physical or dimensionless units.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        xs: Query coordinates of shape ``(M,)`` in physical or dimensionless
            units.

    Returns:
        Tuple ``(vals, slopes)`` of ``float64`` JAX arrays of shape ``(M,)``.
    """
    x_grid = jnp.asarray(init_x, dtype=jnp.float64)
    c_arr = jnp.asarray(coeffs, dtype=jnp.float64)
    x_q = jnp.asarray(xs, dtype=jnp.float64)
    if x_grid.ndim != 1 or c_arr.shape != (x_grid.shape[0], 4):
        raise ValueError("coeffs must have shape (len(init_x), 4).")

    idx = _locate_intervals_jax(x_grid, x_q)
    dx = x_q - x_grid[idx]
    c0 = c_arr[idx, 0]
    c1 = c_arr[idx, 1]
    c2 = c_arr[idx, 2]
    c3 = c_arr[idx, 3]

    vals = c0 + dx * (c1 + dx * (c2 + dx * c3))
    slopes = c1 + dx * (2.0 * c2 + 3.0 * dx * c3)

    lo = jnp.minimum(x_grid[0], x_grid[-1])
    hi = jnp.maximum(x_grid[0], x_grid[-1])
    left_idx = jnp.where(x_grid[-1] > x_grid[0], 0, x_grid.shape[0] - 1)
    right_idx = jnp.where(x_grid[-1] > x_grid[0], x_grid.shape[0] - 1, 0)
    vals = jnp.where(x_q <= lo, c_arr[left_idx, 0], vals)
    vals = jnp.where(x_q >= hi, c_arr[right_idx, 0], vals)
    slopes = jnp.where((x_q < lo) | (x_q > hi), 0.0, slopes)
    return vals, slopes


def interp_values(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates a 1D cubic spline at query points ``xs`` (`interp_values`)."""
    vals, _ = interp_values_and_slopes(init_x, coeffs, xs)
    return vals


def interp_value_and_slope(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    x: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates a 1D cubic spline value and slope at a single scalar ``x``."""
    vals, slopes = interp_values_and_slopes(
        init_x, coeffs, jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
    )
    return vals[0], slopes[0]


def interp_value(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    x: float | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates a 1D cubic spline value at a single scalar ``x``."""
    val, _ = interp_value_and_slope(init_x, coeffs, x)
    return val


def _cubic_antideriv(c: np.ndarray, dx: float) -> float:
    """Evaluates the cubic antiderivative polynomial at displacement ``dx``."""
    return float(
        dx * (c[0] + dx * (0.5 * c[1] + dx * (c[2] / 3.0 + 0.25 * c[3] * dx)))
    )


def integrate_values(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Integrates a 1D cubic spline over adjacent segments ``[xs[k-1], xs[k]]``.

    Returns an array ``integrals`` of shape ``(M,)`` with ``integrals[0] = 0``
    and ``integrals[k] = \\int_{xs[k-1]}^{xs[k]} f(x)\\,dx`` for ``k >= 1``.

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)`` in
            physical or dimensionless units.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        xs: Monotonic integration bounds of shape ``(M,)`` aligned with
            ``init_x`` direction and lying within ``init_x`` bounds.

    Returns:
        JAX ``float64`` array of shape ``(M,)`` containing segment integrals.

    Raises:
        ValueError: If ``xs`` lies outside ``init_x`` or reverses direction.
    """
    x_grid = np.asarray(init_x, dtype=np.float64)
    c_arr = np.asarray(coeffs, dtype=np.float64)
    x_q = np.asarray(xs, dtype=np.float64)
    nx = x_grid.size
    if c_arr.shape != (nx, 4):
        raise ValueError("coeffs must have shape (len(init_x), 4).")
    increasing = bool(x_grid[-1] > x_grid[0])
    out = np.zeros(x_q.size, dtype=np.float64)

    for k in range(1, x_q.size):
        xa = float(x_q[k - 1])
        xb = float(x_q[k])
        if increasing:
            if xb < xa or xa < x_grid[0] or xb > x_grid[-1]:
                raise ValueError("xs must be increasing within init_x range.")
        else:
            if xb > xa or xa > x_grid[0] or xb < x_grid[-1]:
                raise ValueError("xs must be decreasing within init_x range.")
        if xa == xb:
            continue

        total = 0.0
        for i in range(nx - 1):
            x0 = float(x_grid[i])
            x1 = float(x_grid[i + 1])
            if increasing:
                if x1 <= xa:
                    continue
                if x0 >= xb:
                    break
                seg_start = max(xa, x0)
                seg_end = min(xb, x1)
            else:
                if x1 >= xa:
                    continue
                if x0 <= xb:
                    break
                seg_start = min(xa, x0)
                seg_end = max(xb, x1)
            a2 = _cubic_antideriv(c_arr[i], seg_end - x0)
            a1 = (
                _cubic_antideriv(c_arr[i], seg_start - x0)
                if seg_start != x0
                else 0.0
            )
            total += a2 - a1
        out[k] = total
    return jnp.asarray(out, dtype=jnp.float64)


def interpolate_vector(
    old_x: np.ndarray | jnp.ndarray,
    old_y: np.ndarray | jnp.ndarray,
    new_x: np.ndarray | jnp.ndarray,
    builder: Callable[
        [np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray], jnp.ndarray
    ] = pm.interp_pm,
) -> jnp.ndarray:
    """Interpolates ``(old_x, old_y)`` onto ``new_x`` using ``builder``."""
    coeffs = builder(old_x, old_y)
    return interp_values(old_x, coeffs, new_x)


def interpolate_vector_pm(
    old_x: np.ndarray | jnp.ndarray,
    old_y: np.ndarray | jnp.ndarray,
    new_x: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Interpolates ``(old_x, old_y)`` onto ``new_x`` using ``interp_pm``."""
    return interpolate_vector(old_x, old_y, new_x, builder=pm.interp_pm)


def interpolate_vector_pm_autodiff(
    old_x: Sequence[auto_diff.DualOrder1],
    old_y: Sequence[auto_diff.DualOrder1],
    new_x: Sequence[auto_diff.DualOrder1],
) -> list[auto_diff.DualOrder1]:
    """Interpolates ``DualOrder1`` vectors via Steffen (1990) ``interp_pm``."""
    coeffs = pm.interp_pm_autodiff(old_x, old_y)
    nx = len(old_x)
    increasing = old_x[-1].val > old_x[0].val
    out: list[auto_diff.DualOrder1] = []

    for xq in new_x:
        if (increasing and xq.val <= old_x[0].val) or (
            not increasing and xq.val >= old_x[0].val
        ):
            out.append(coeffs[0][0])
            continue
        if (increasing and xq.val >= old_x[-1].val) or (
            not increasing and xq.val <= old_x[-1].val
        ):
            out.append(coeffs[-1][0])
            continue
        idx = nx - 2
        for i in range(nx - 1):
            in_cell = (
                old_x[i].val <= xq.val <= old_x[i + 1].val
                if increasing
                else old_x[i].val >= xq.val >= old_x[i + 1].val
            )
            if in_cell:
                idx = i
                break
        dx = xq - old_x[idx]
        c0, c1, c2, c3 = coeffs[idx]
        out.append(c0 + dx * (c1 + dx * (c2 + dx * c3)))
    return out


def interp_3_to_1(
    dm_m1: float,
    dm_00: float,
    dm_p1: float,
    f_m1: float,
    f_00: float,
    f_p1: float,
) -> float:
    """Interpolates cell-averaged ``(f_m1, f_00, f_p1)`` to the central face."""
    old_x = np.array(
        [0.0, 0.5 * (dm_m1 + dm_00), 0.5 * dm_m1 + dm_00 + 0.5 * dm_p1],
        dtype=np.float64,
    )
    old_y = np.array([f_m1, f_00, f_p1], dtype=np.float64)
    new_x = np.array([0.5 * dm_m1], dtype=np.float64)
    return float(interpolate_vector_pm(old_x, old_y, new_x)[0])


def interp_3_to_2(
    dm_m1: float,
    dm_00: float,
    dm_p1: float,
    dm_p2: float,
    f_m1: float,
    f_00: float,
    f_p1: float,
) -> tuple[float, float]:
    """Interpolates cell-averaged ``(f_m1, f_00, f_p1)`` to two inner faces."""
    old_x = np.array(
        [
            0.0,
            0.5 * (dm_m1 + dm_00),
            0.5 * dm_m1 + dm_00 + 0.5 * dm_p1,
            0.5 * dm_m1 + dm_00 + dm_p1 + 0.5 * dm_p2,
        ],
        dtype=np.float64,
    )
    old_y = np.array([f_m1, f_00, f_p1, f_p1], dtype=np.float64)
    new_x = np.array([0.5 * dm_m1, 0.5 * dm_m1 + dm_00], dtype=np.float64)
    res = interpolate_vector_pm(old_x, old_y, new_x)
    return float(res[0]), float(res[1])


def interp_4_to_1(
    dm_m2: float,
    dm_m1: float,
    dm_00: float,
    dm_pt: float,
    f_m2: float,
    f_m1: float,
    f_00: float,
    f_p1: float,
) -> float:
    """Interpolates 4 cell points to ``dm_pt`` using ``interp_pm``."""
    old_x = np.array(
        [0.0, dm_m2, dm_m2 + dm_m1, dm_m2 + dm_m1 + dm_00],
        dtype=np.float64,
    )
    old_y = np.array([f_m2, f_m1, f_00, f_p1], dtype=np.float64)
    new_x = np.array([dm_pt], dtype=np.float64)
    return float(interpolate_vector_pm(old_x, old_y, new_x)[0])
