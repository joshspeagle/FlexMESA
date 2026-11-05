# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional piecewise-monotonic cubic interpolation (`bipm_db`)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa import interp_1d
from flexmesa.interp_2d import spline1d


def interp_mkbipm_db(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    z: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Constructs 2D piecewise-monotonic cubic spline coefficients along `x`.

    Args:
        x: Strictly ascending x-grid of shape ``(nx,)`` (``nx >= 4``) in
            physical or dimensionless units.
        y: Strictly ascending y-grid of shape ``(ny,)`` (``ny >= 4``) in
            physical or dimensionless units.
        z: Grid values of shape ``(nx, ny)`` or flat ``(nx * ny,)`` in
            row-major order.

    Returns:
        JAX ``float64`` array of shape ``(nx, ny, 4)`` containing 1D
        Steffen (1990) ``x``-spline coefficients for each ``y`` slice.

    Raises:
        ValueError: If grid dimensions are ``< 4``, non-ascending, or
            non-finite.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nx, ny = x_arr.size, y_arr.size
    if nx < 4 or ny < 4:
        raise ValueError("interp_mkbipm_db requires nx >= 4 and ny >= 4.")
    spline1d.splinck_db(x_arr, 1.0e-3, "x")
    spline1d.splinck_db(y_arr, 1.0e-3, "y")
    z_arr = np.asarray(z, dtype=np.float64).reshape(nx, ny)
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must contain only finite float64 values.")

    coeffs = np.zeros((nx, ny, 4), dtype=np.float64)
    for iy in range(ny):
        c_x = interp_1d.interp_pm(x_arr, z_arr[:, iy])
        coeffs[:, iy, :] = np.asarray(c_x, dtype=np.float64)
    return jnp.asarray(coeffs, dtype=jnp.float64)


def interp_evbipm_db(
    xget: float | jnp.ndarray,
    yget: float | jnp.ndarray,
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates the 2D piecewise-monotonic cubic interpolant at `(xget, yget)`.

    Performs four 1D cubic evaluations along ``x`` on a 4-point ``y`` stencil
    followed by a 4-point 1D ``interp_pm`` along ``y``.

    Args:
        xget: Query x-coordinate in ``[x[0], x[-1])``.
        yget: Query y-coordinate in ``[y[0], y[-1])``.
        x: Strictly ascending x-grid of shape ``(nx,)`` (``nx >= 4``).
        y: Strictly ascending y-grid of shape ``(ny,)`` (``ny >= 4``).
        coeffs: Coefficient array of shape ``(nx, ny, 4)`` from
            :func:`interp_mkbipm_db`.

    Returns:
        Scalar ``float64`` JAX array containing the interpolated value.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nx, ny = x_arr.size, y_arr.size
    if nx < 4 or ny < 4:
        raise ValueError("interp_evbipm_db requires nx >= 4 and ny >= 4.")
    spline1d.splinck_db(x_arr, 1.0e-3, "x")
    spline1d.splinck_db(y_arr, 1.0e-3, "y")
    c_arr = np.asarray(coeffs, dtype=np.float64).reshape(nx, ny, 4)
    xq = float(xget)
    yq = float(yget)
    if xq < x_arr[0] or xq >= x_arr[-1] or yq < y_arr[0] or yq >= y_arr[-1]:
        raise ValueError("Query (xget, yget) is outside the grid domain.")

    ix = int(np.clip(np.searchsorted(x_arr, xq, side="right") - 1, 0, nx - 2))
    jy = int(np.clip(np.searchsorted(y_arr, yq, side="right") - 1, 0, ny - 2))

    if jy == 0:
        jlo = 0
    elif jy >= ny - 2:
        jlo = jy - 2
    else:
        jlo = jy - 1

    dx = xq - x_arr[ix]
    ys = y_arr[jlo : jlo + 4]
    c_sub = c_arr[ix, jlo : jlo + 4, :]
    ff = c_sub[:, 0] + dx * (
        c_sub[:, 1] + dx * (c_sub[:, 2] + dx * c_sub[:, 3])
    )
    y_coeffs = interp_1d.interp_pm(ys, ff)
    return interp_1d.interp_value(ys, y_coeffs, yq)
