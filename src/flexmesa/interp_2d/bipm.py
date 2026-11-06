# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional piecewise-monotonic cubic interpolation (`bipm_db`)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, interp_1d
from flexmesa.interp_2d import spline1d


def interp_mkbipm_db(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    z: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Constructs 2D piecewise-monotonic cubic spline coefficients along `x`.

    Args:
        x: Strictly ascending x-grid of shape ``(nx,)`` (``nx >= 4``) in
            physical or dimensionless units.
        y: Strictly ascending y-grid of shape ``(ny,)`` (``ny >= 4``) in
            physical or dimensionless units.
        z: Grid values of shape ``(nx, ny)`` or flat ``(nx * ny,)`` in
            row-major order.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of shape ``(nx, ny, 4)`` containing 1D
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
    z_arr = spline1d.reshape_grid_2d(z, nx, ny, "z")

    if _dispatch.resolve_backend(backend, x, y, z) == "rust":
        return _rust.interp_mkbipm_db(
            _dispatch.as_c_f64(x_arr),
            _dispatch.as_c_f64(y_arr),
            _dispatch.as_c_f64(z_arr),
        )

    coeffs = np.zeros((nx, ny, 4), dtype=np.float64)
    for iy in range(ny):
        c_x = interp_1d.interp_pm(x_arr, z_arr[:, iy], backend="jax")
        coeffs[:, iy, :] = np.asarray(c_x, dtype=np.float64)
    return jnp.asarray(coeffs, dtype=jnp.float64)


def interp_evbipm_db(
    xget: float | jnp.ndarray,
    yget: float | jnp.ndarray,
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    coeffs: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | np.ndarray | jnp.ndarray:
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
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Scalar ``float64`` value containing the interpolated value.

    Raises:
        ValueError: If grid dimensions are ``< 4``, non-ascending, mismatched
            with ``coeffs``, or if ``(xget, yget)`` is non-finite or outside
            ``[x[0], x[-1]) x [y[0], y[-1])``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nx, ny = x_arr.size, y_arr.size
    if nx < 4 or ny < 4:
        raise ValueError("interp_evbipm_db requires nx >= 4 and ny >= 4.")
    spline1d.splinck_db(x_arr, 1.0e-3, "x")
    spline1d.splinck_db(y_arr, 1.0e-3, "y")
    c_raw = np.asarray(coeffs, dtype=np.float64)
    if c_raw.shape not in ((nx, ny, 4), (nx * ny, 4)) or not np.all(
        np.isfinite(c_raw)
    ):
        raise ValueError(
            f"coeffs must be a finite array of shape ({nx}, {ny}, 4)."
        )
    c_arr = c_raw.reshape(nx, ny, 4)
    xq = float(xget)
    yq = float(yget)
    if (
        not (np.isfinite(xq) and np.isfinite(yq))
        or xq < x_arr[0]
        or xq >= x_arr[-1]
        or yq < y_arr[0]
        or yq >= y_arr[-1]
    ):
        raise ValueError("Query (xget, yget) is outside the grid domain.")

    if _dispatch.resolve_backend(backend, xget, yget, x, y, coeffs) == "rust":
        return jnp.asarray(
            _rust.interp_evbipm_db(
                xq,
                yq,
                _dispatch.as_c_f64(x_arr),
                _dispatch.as_c_f64(y_arr),
                _dispatch.as_c_f64(c_arr),
            ),
            dtype=jnp.float64,
        )

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
    y_coeffs = interp_1d.interp_pm(ys, ff, backend="jax")
    return interp_1d.interp_value(ys, y_coeffs, yq, backend="jax")
