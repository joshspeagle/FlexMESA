# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spline evaluation and definite integration (`interp_1d`).

Provides vectorized ``float64`` JAX evaluation of 1D cubic splines and exact
piecewise-analytic definite integration across monotonic knot grids.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.interp_1d import pm


def _locate_intervals_jax(
    init_x: jnp.ndarray,
    xs: jnp.ndarray,
) -> jnp.ndarray:
    """Locates 0-based subinterval indices ``i in [0, nx - 2]`` for queries.

    Handles both strictly increasing and strictly decreasing knot arrays
    ``init_x`` of shape ``(N,)`` via binary search (`jnp.searchsorted`).
    """
    nx = init_x.shape[0]
    increasing = init_x[-1] > init_x[0]
    grid_asc = jnp.where(increasing, init_x, init_x[::-1])
    pos = jnp.searchsorted(grid_asc, xs, side="right") - 1
    idx_asc = jnp.clip(pos, 0, nx - 2)
    return jnp.where(increasing, idx_asc, (nx - 2) - idx_asc)


def _eval_horner(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
    *,
    with_slopes: bool,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Shared Horner evaluator for 1D cubic splines and optional derivatives.

    Clamps values outside ``[min(init_x), max(init_x)]`` to the nearest
    endpoint knot value and zeroes exterior derivatives, matching upstream MESA
    ``do_interp_values``.
    """
    x_grid = jnp.asarray(init_x, dtype=jnp.float64)
    c_arr = jnp.asarray(coeffs, dtype=jnp.float64)
    x_q = jnp.asarray(xs, dtype=jnp.float64)
    nx = x_grid.shape[0]
    if x_grid.ndim != 1 or nx == 0 or c_arr.shape != (nx, 4):
        raise ValueError("coeffs must have shape (len(init_x), 4) with N >= 1.")
    if nx == 1:
        vals = jnp.full_like(x_q, c_arr[0, 0])
        slopes = jnp.zeros_like(x_q) if with_slopes else None
        return vals, slopes

    idx = _locate_intervals_jax(x_grid, x_q)
    dx = x_q - x_grid[idx]
    c0, c1, c2, c3 = (
        c_arr[idx, 0],
        c_arr[idx, 1],
        c_arr[idx, 2],
        c_arr[idx, 3],
    )

    vals = c0 + dx * (c1 + dx * (c2 + dx * c3))
    lo = jnp.minimum(x_grid[0], x_grid[-1])
    hi = jnp.maximum(x_grid[0], x_grid[-1])
    left_idx = jnp.where(x_grid[-1] > x_grid[0], 0, nx - 1)
    right_idx = jnp.where(x_grid[-1] > x_grid[0], nx - 1, 0)
    vals = jnp.where(x_q <= lo, c_arr[left_idx, 0], vals)
    vals = jnp.where(x_q >= hi, c_arr[right_idx, 0], vals)
    if not with_slopes:
        return vals, None
    slopes = c1 + dx * (2.0 * c2 + 3.0 * dx * c3)
    slopes = jnp.where((x_q < lo) | (x_q > hi), 0.0, slopes)
    return vals, slopes


def _prepare_rust_eval_1d(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Validates and packs 1D spline inputs as C-contiguous ``float64`` arrays.

    Returns ``(xg, ca_flat, xq)`` when ``nx >= 2`` and ``xq`` is 1D, or ``None``
    when the single-knot / higher-rank fallback path in ``_eval_horner`` should
    be used.
    """
    xg = np.ascontiguousarray(init_x, dtype=np.float64)
    ca = np.ascontiguousarray(coeffs, dtype=np.float64)
    xq = np.ascontiguousarray(xs, dtype=np.float64)
    if xg.ndim != 1 or xg.size == 0 or ca.shape != (xg.size, 4):
        raise ValueError("coeffs must have shape (len(init_x), 4) with N >= 1.")
    if xg.size >= 2 and xq.ndim == 1:
        return xg, ca.ravel(), xq
    return None


def interp_values_and_slopes(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray]:
    """Evaluates a 1D cubic spline and its first derivative at query points.

    Supports ``jax.grad``, ``jax.hessian``, and ``jax.vmap`` with respect to
    both ``xs`` and ``coeffs``.

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)`` in
            physical or dimensionless units.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        xs: Query coordinates of shape ``(M,)`` in physical or dimensionless
            units.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Tuple ``(vals, slopes)`` of ``float64`` arrays of shape ``(M,)``.

    Raises:
        ValueError: If ``init_x`` and ``coeffs`` shapes are incompatible.
    """
    if _dispatch.resolve_backend(backend, init_x, coeffs, xs) == "rust":
        packed = _prepare_rust_eval_1d(init_x, coeffs, xs)
        if packed is not None:
            return _rust.interp_values_and_slopes(*packed)
    vals, slopes = _eval_horner(init_x, coeffs, xs, with_slopes=True)
    assert slopes is not None
    return vals, slopes


def interp_values(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Evaluates a 1D cubic spline at query points ``xs`` (`interp_values`).

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)``.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        xs: Query coordinates of shape ``(M,)``.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of interpolated values of shape ``(M,)``.

    Raises:
        ValueError: If ``init_x`` and ``coeffs`` shapes are incompatible.
    """
    if _dispatch.resolve_backend(backend, init_x, coeffs, xs) == "rust":
        packed = _prepare_rust_eval_1d(init_x, coeffs, xs)
        if packed is not None:
            return _rust.interp_values(*packed)
    vals, _ = _eval_horner(init_x, coeffs, xs, with_slopes=False)
    return vals


def interp_value_and_slope(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    x: float | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray]:
    """Evaluates a 1D cubic spline value and slope at a single scalar ``x``.

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)``.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        x: Scalar query coordinate in physical or dimensionless units.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Tuple ``(val, slope)`` of scalar ``float64`` values.

    Raises:
        ValueError: If ``init_x`` and ``coeffs`` shapes are incompatible.
    """
    resolved = _dispatch.resolve_backend(backend, init_x, coeffs, x)
    xq = (
        np.atleast_1d(np.asarray(x, dtype=np.float64))
        if resolved == "rust"
        else jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
    )
    vals, slopes = interp_values_and_slopes(
        init_x, coeffs, xq, backend=resolved
    )
    return vals[0], slopes[0]


def interp_value(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    x: float | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Evaluates a 1D cubic spline value at a single scalar ``x``.

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)``.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        x: Scalar query coordinate in physical or dimensionless units.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Scalar ``float64`` value containing ``f(x)``.

    Raises:
        ValueError: If ``init_x`` and ``coeffs`` shapes are incompatible.
    """
    resolved = _dispatch.resolve_backend(backend, init_x, coeffs, x)
    xq = (
        np.atleast_1d(np.asarray(x, dtype=np.float64))
        if resolved == "rust"
        else jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
    )
    vals = interp_values(init_x, coeffs, xq, backend=resolved)
    return vals[0]


def _cubic_antideriv(c: np.ndarray, dx: float) -> float:
    """Evaluates the cubic antiderivative polynomial at displacement ``dx``."""
    return float(
        dx * (c[0] + dx * (0.5 * c[1] + dx * (c[2] / 3.0 + 0.25 * c[3] * dx)))
    )


def integrate_values(
    init_x: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    xs: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Integrates a 1D cubic spline over adjacent segments ``[xs[k-1], xs[k]]``.

    Returns an array ``integrals`` of shape ``(M,)`` with ``integrals[0] = 0``
    and ``integrals[k] = \\int_{xs[k-1]}^{xs[k]} f(x)\\,dx`` for ``k >= 1``.

    Args:
        init_x: Strictly monotonic knot coordinates of shape ``(N,)``
            (``N >= 2``) in physical or dimensionless units.
        coeffs: Spline coefficient array of shape ``(N, 4)``.
        xs: Monotonic integration bounds of shape ``(M,)`` aligned with
            ``init_x`` direction and lying within ``init_x`` bounds.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of shape ``(M,)`` containing segment integrals.

    Raises:
        ValueError: If ``init_x``, ``coeffs``, or ``xs`` fail validation or
            ``xs`` lies outside ``init_x`` or reverses direction.
    """
    x_grid, _ = pm.validate_1d_grid(init_x, init_x, min_points=2)
    c_arr = np.asarray(coeffs, dtype=np.float64)
    x_q = np.asarray(xs, dtype=np.float64)
    nx = x_grid.size
    if c_arr.shape != (nx, 4) or not np.all(np.isfinite(c_arr)):
        raise ValueError("coeffs must be a finite array of shape (N, 4).")
    if x_q.ndim != 1 or not np.all(np.isfinite(x_q)):
        raise ValueError("xs must be a finite 1D float64 array.")
    if x_q.size == 0:
        return jnp.zeros(0, dtype=jnp.float64)

    if _dispatch.resolve_backend(backend, init_x, coeffs, xs) == "rust":
        return _rust.integrate_values(
            _dispatch.as_c_f64(x_grid),
            _dispatch.as_c_f64(c_arr),
            _dispatch.as_c_f64(x_q),
        )

    increasing = bool(x_grid[-1] > x_grid[0])
    lo = float(min(x_grid[0], x_grid[-1]))
    hi = float(max(x_grid[0], x_grid[-1]))
    if float(np.min(x_q)) < lo or float(np.max(x_q)) > hi:
        raise ValueError("xs must lie within the init_x domain.")

    grid_asc = x_grid if increasing else x_grid[::-1]
    out = np.zeros(x_q.size, dtype=np.float64)

    for k in range(1, x_q.size):
        xa, xb = float(x_q[k - 1]), float(x_q[k])
        if increasing:
            if xb < xa:
                raise ValueError("xs must be increasing within init_x range.")
            pos = int(np.searchsorted(grid_asc, xa, side="right")) - 1
            start_i = max(0, min(pos, nx - 2))
        else:
            if xb > xa:
                raise ValueError("xs must be decreasing within init_x range.")
            pos = int(np.searchsorted(grid_asc, xa, side="left"))
            start_i = max(0, min((nx - 2) - pos, nx - 2))
        if xa == xb:
            continue

        total = 0.0
        for i in range(start_i, nx - 1):
            x0, x1 = float(x_grid[i]), float(x_grid[i + 1])
            if increasing:
                if x1 <= xa:
                    continue
                if x0 >= xb:
                    break
                seg_start, seg_end = max(xa, x0), min(xb, x1)
            else:
                if x1 >= xa:
                    continue
                if x0 <= xb:
                    break
                seg_start, seg_end = min(xa, x0), max(xb, x1)
            a2 = _cubic_antideriv(c_arr[i], seg_end - x0)
            a1 = (
                _cubic_antideriv(c_arr[i], seg_start - x0)
                if seg_start != x0
                else 0.0
            )
            total += a2 - a1
        out[k] = total
    return jnp.asarray(out, dtype=jnp.float64)
