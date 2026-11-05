# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional C^2 bicubic spline setup (`interp_mkbicub_db`).

Constructs $C^2$ bicubic spline coefficient tables ``[f, f_xx, f_yy, f_xxyy]``
over a rectangular grid ``(x, y)`` of shape ``(nx, ny)`` by solving 1D cubic
splines along ``x`` and ``y`` with optional inhomogeneous boundary-condition
correction matching upstream PSPLINE / MESA ``do_mkbicub_db``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa.interp_2d import bicub_eval, spline1d

BicubicBc = spline1d.BicubicBc
BicubicEval = bicub_eval.BicubicEval
herm2xy_db = bicub_eval.herm2xy_db
interp_evbicub_db = bicub_eval.interp_evbicub_db
interp_evbicub_mask_db = bicub_eval.interp_evbicub_mask_db
interp_mkspline_db = spline1d.interp_mkspline_db
mkspline_db = spline1d.mkspline_db
splinck_db = spline1d.splinck_db


def _validate_bc_slice(
    bc: np.ndarray | jnp.ndarray | None,
    code: int,
    expected_len: int,
    name: str,
) -> np.ndarray:
    """Validates and normalizes a 1D boundary-condition array.

    Args:
        bc: Optional 1D boundary array of shape ``(expected_len,)``.
        code: Boundary condition integer code in ``-1..=7``.
        expected_len: Required length along the transverse grid axis.
        name: Parameter name for error reporting.

    Returns:
        1D ``float64`` NumPy array of shape ``(expected_len,)``.
    """
    if bc is None:
        return np.zeros(expected_len, dtype=np.float64)
    arr = np.asarray(bc, dtype=np.float64)
    if code in (1, 2) and arr.shape != (expected_len,):
        raise ValueError(
            f"{name} must have shape ({expected_len},) when boundary code "
            f"is {code}."
        )
    if arr.size == 0:
        return np.zeros(expected_len, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite float64 values.")
    return arr


def interp_mkbicub_db(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    z: np.ndarray | jnp.ndarray,
    ibcxmin: BicubicBc | int = BicubicBc.NOT_A_KNOT,
    bcxmin: np.ndarray | jnp.ndarray | None = None,
    ibcxmax: BicubicBc | int = BicubicBc.NOT_A_KNOT,
    bcxmax: np.ndarray | jnp.ndarray | None = None,
    ibcymin: BicubicBc | int = BicubicBc.NOT_A_KNOT,
    bcymin: np.ndarray | jnp.ndarray | None = None,
    ibcymax: BicubicBc | int = BicubicBc.NOT_A_KNOT,
    bcymax: np.ndarray | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, bool, bool]:
    """Constructs 2D C^2 bicubic spline coefficients on grid ``(x, y)``.

    Args:
        x: Strictly ascending x-grid of shape ``(nx,)`` (``nx >= 4``) in
            physical or dimensionless units.
        y: Strictly ascending y-grid of shape ``(ny,)`` (``ny >= 4``) in
            physical or dimensionless units.
        z: Grid values of shape ``(nx, ny)`` or flat ``(nx * ny,)`` in
            row-major order.
        ibcxmin: Boundary condition at ``x[0]``.
        bcxmin: Boundary data of shape ``(ny,)`` at ``x[0]``.
        ibcxmax: Boundary condition at ``x[-1]``.
        bcxmax: Boundary data of shape ``(ny,)`` at ``x[-1]``.
        ibcymin: Boundary condition at ``y[0]``.
        bcymin: Boundary data of shape ``(nx,)`` at ``y[0]``.
        ibcymax: Boundary condition at ``y[-1]``.
        bcymax: Boundary data of shape ``(nx,)`` at ``y[-1]``.

    Returns:
        Tuple ``(coeffs, ilinx, iliny)`` where ``coeffs`` is a ``float64`` JAX
        array of shape ``(nx, ny, 4)`` containing ``[f, f_xx, f_yy, f_xxyy]``.

    Raises:
        ValueError: If grid dimensions are ``< 4``, non-ascending, mismatched
            with ``z`` or boundary arrays, or contain non-finite values.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    nx, ny = x_arr.size, y_arr.size
    if nx < 4 or ny < 4:
        raise ValueError("2D bicubic spline requires nx >= 4 and ny >= 4.")
    z_arr = np.asarray(z, dtype=np.float64).reshape(nx, ny)
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must contain only finite float64 values.")

    ilinx = splinck_db(x_arr, 1.0e-3, "x")
    iliny = splinck_db(y_arr, 1.0e-3, "y")

    bc_x0 = _validate_bc_slice(bcxmin, int(ibcxmin), ny, "bcxmin")
    bc_x1 = _validate_bc_slice(bcxmax, int(ibcxmax), ny, "bcxmax")
    bc_y0 = _validate_bc_slice(bcymin, int(ibcymin), nx, "bcymin")
    bc_y1 = _validate_bc_slice(bcymax, int(ibcymax), nx, "bcymax")

    coeffs = np.zeros((nx, ny, 4), dtype=np.float64)
    coeffs[:, :, 0] = z_arr

    # Step 1: Spline along x for each y-slice to obtain f_xx.
    for iy in range(ny):
        fxx, _ = interp_mkspline_db(
            x_arr,
            coeffs[:, iy, 0],
            ibcxmin,
            float(bc_x0[iy]) if int(ibcxmin) in (1, 2) else 0.0,
            ibcxmax,
            float(bc_x1[iy]) if int(ibcxmax) in (1, 2) else 0.0,
        )
        coeffs[:, iy, 1] = np.asarray(fxx, dtype=np.float64)

    iflg2 = int(ibcymin) != -1 and (
        (int(ibcymin) in (1, 2) and np.any(bc_y0 != 0.0))
        or (int(ibcymax) in (1, 2) and np.any(bc_y1 != 0.0))
    )
    ibcy0_h = 0 if (iflg2 and int(ibcymin) in (1, 2)) else int(ibcymin)
    ibcy1_h = 0 if (iflg2 and int(ibcymax) in (1, 2)) else int(ibcymax)

    # Step 2: Homogeneous spline along y for each x-slice (f_yy and f_xxyy).
    for ix in range(nx):
        fyy, _ = interp_mkspline_db(
            y_arr, coeffs[ix, :, 0], ibcy0_h, 0.0, ibcy1_h, 0.0
        )
        fxxyy, _ = interp_mkspline_db(
            y_arr, coeffs[ix, :, 1], ibcy0_h, 0.0, ibcy1_h, 0.0
        )
        coeffs[ix, :, 2] = np.asarray(fyy, dtype=np.float64)
        coeffs[ix, :, 3] = np.asarray(fxxyy, dtype=np.float64)

    # Step 3: Correct for inhomogeneous y boundary conditions when active.
    if iflg2:
        fcorr1 = np.zeros((nx, ny), dtype=np.float64)
        zeros_y = np.zeros(ny, dtype=np.float64)
        for ix in range(nx):
            zdiff1 = 0.0
            if int(ibcymin) == 1:
                hy = y_arr[1] - y_arr[0]
                cur = (coeffs[ix, 1, 0] - coeffs[ix, 0, 0]) / hy + hy * (
                    -2.0 * coeffs[ix, 0, 2] - coeffs[ix, 1, 2]
                ) / 6.0
                zdiff1 = float(bc_y0[ix] - cur)
            elif int(ibcymin) == 2:
                zdiff1 = float(bc_y0[ix] - coeffs[ix, 0, 2])
            zdiff2 = 0.0
            if int(ibcymax) == 1:
                hy = y_arr[-1] - y_arr[-2]
                cur = (coeffs[ix, -1, 0] - coeffs[ix, -2, 0]) / hy + hy * (
                    2.0 * coeffs[ix, -1, 2] + coeffs[ix, -2, 2]
                ) / 6.0
                zdiff2 = float(bc_y1[ix] - cur)
            elif int(ibcymax) == 2:
                zdiff2 = float(bc_y1[ix] - coeffs[ix, -1, 2])
            corr_y, _ = interp_mkspline_db(
                y_arr, zeros_y, ibcymin, zdiff1, ibcymax, zdiff2
            )
            fcorr1[ix, :] = np.asarray(corr_y, dtype=np.float64)
        for iy in range(ny):
            corr_xy, _ = interp_mkspline_db(
                x_arr, fcorr1[:, iy], ibcxmin, 0.0, ibcxmax, 0.0
            )
            coeffs[:, iy, 2] += fcorr1[:, iy]
            coeffs[:, iy, 3] += np.asarray(corr_xy, dtype=np.float64)

    return jnp.asarray(coeffs, dtype=jnp.float64), ilinx, iliny
