# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional C^2 bicubic spline interpolation (`interp_mkbicub_db`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from flexmesa.interp_2d import spline1d

BicubicBc = spline1d.BicubicBc
interp_mkspline_db = spline1d.interp_mkspline_db
splinck_db = spline1d.splinck_db


@dataclasses.dataclass(frozen=True)
class BicubicEval:
    """Evaluated value and partial derivatives of a 2D bicubic spline.

    Attributes:
        f: Interpolated value ``f(x, y)`` (dimensionless or physical units).
        df_dx: First partial derivative ``df/dx``.
        df_dy: First partial derivative ``df/dy``.
        d2f_dx2: Second partial derivative ``d^2f/dx^2``.
        d2f_dy2: Second partial derivative ``d^2f/dy^2``.
        d2f_dxdy: Mixed second partial derivative ``d^2f/(dx dy)``.
    """

    f: jnp.ndarray
    df_dx: jnp.ndarray
    df_dy: jnp.ndarray
    d2f_dx2: jnp.ndarray
    d2f_dy2: jnp.ndarray
    d2f_dxdy: jnp.ndarray


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
        x: Strictly ascending x-grid of shape ``(nx,)`` (``nx >= 4``).
        y: Strictly ascending y-grid of shape ``(ny,)`` (``ny >= 4``).
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

    bc_x0 = (
        np.zeros(ny, dtype=np.float64)
        if bcxmin is None
        else np.asarray(bcxmin, dtype=np.float64)
    )
    bc_x1 = (
        np.zeros(ny, dtype=np.float64)
        if bcxmax is None
        else np.asarray(bcxmax, dtype=np.float64)
    )
    bc_y0 = (
        np.zeros(nx, dtype=np.float64)
        if bcymin is None
        else np.asarray(bcymin, dtype=np.float64)
    )
    bc_y1 = (
        np.zeros(nx, dtype=np.float64)
        if bcymax is None
        else np.asarray(bcymax, dtype=np.float64)
    )

    coeffs = np.zeros((nx, ny, 4), dtype=np.float64)
    coeffs[:, :, 0] = z_arr

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

    for ix in range(nx):
        fyy, _ = interp_mkspline_db(
            y_arr, coeffs[ix, :, 0], ibcy0_h, 0.0, ibcy1_h, 0.0
        )
        fxxyy, _ = interp_mkspline_db(
            y_arr, coeffs[ix, :, 1], ibcy0_h, 0.0, ibcy1_h, 0.0
        )
        coeffs[ix, :, 2] = np.asarray(fyy, dtype=np.float64)
        coeffs[ix, :, 3] = np.asarray(fxxyy, dtype=np.float64)

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


def herm2xy_db(
    xget: float | jnp.ndarray,
    yget: float | jnp.ndarray,
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    ilinx: bool = True,
    iliny: bool = True,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Locates the 2D grid cell and local coordinates (`herm2xy_db`)."""
    del ilinx, iliny
    x_grid = jnp.asarray(x, dtype=jnp.float64)
    y_grid = jnp.asarray(y, dtype=jnp.float64)
    xq = jnp.asarray(xget, dtype=jnp.float64)
    yq = jnp.asarray(yget, dtype=jnp.float64)
    zx = jnp.clip(xq, x_grid[0], x_grid[-1])
    zy = jnp.clip(yq, y_grid[0], y_grid[-1])

    ix = jnp.clip(
        jnp.searchsorted(x_grid, zx, side="right") - 1, 0, x_grid.shape[0] - 2
    )
    jy = jnp.clip(
        jnp.searchsorted(y_grid, zy, side="right") - 1, 0, y_grid.shape[0] - 2
    )
    hx = x_grid[ix + 1] - x_grid[ix]
    hxi = 1.0 / hx
    xp = (zx - x_grid[ix]) * hxi
    hy = y_grid[jy + 1] - y_grid[jy]
    hyi = 1.0 / hy
    yp = (zy - y_grid[jy]) * hyi
    return ix, jy, xp, yp, hx, hxi, hy, hyi


def interp_evbicub_db(
    xget: float | jnp.ndarray,
    yget: float | jnp.ndarray,
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    ilinx: bool,
    iliny: bool,
    coeffs: np.ndarray | jnp.ndarray,
) -> BicubicEval:
    """Evaluates 2D bicubic spline value and derivatives (`fvbicub_db`).

    Fully differentiable in ``float64`` via ``jax.grad`` and ``jax.hessian``.
    """
    x_grid = jnp.asarray(x, dtype=jnp.float64)
    y_grid = jnp.asarray(y, dtype=jnp.float64)
    c_arr = jnp.asarray(coeffs, dtype=jnp.float64).reshape(
        x_grid.shape[0], y_grid.shape[0], 4
    )
    ix, jy, xp, yp, hx, hxi, hy, hyi = herm2xy_db(
        xget, yget, x_grid, y_grid, ilinx, iliny
    )
    sixth = 1.0 / 6.0
    z36th = sixth * sixth

    xpi = 1.0 - xp
    xp2 = xp * xp
    xpi2 = xpi * xpi
    cx = xp * (xp2 - 1.0)
    cxi = xpi * (xpi2 - 1.0)
    cxd = 3.0 * xp2 - 1.0
    cxdi = -3.0 * xpi2 + 1.0
    hx2 = hx * hx

    ypi = 1.0 - yp
    yp2 = yp * yp
    ypi2 = ypi * ypi
    cy = yp * (yp2 - 1.0)
    cyi = ypi * (ypi2 - 1.0)
    cyd = 3.0 * yp2 - 1.0
    cydi = -3.0 * ypi2 + 1.0
    hy2 = hy * hy

    c00 = c_arr[ix, jy]
    c01 = c_arr[ix, jy + 1]
    c10 = c_arr[ix + 1, jy]
    c11 = c_arr[ix + 1, jy + 1]

    f = (
        xpi * (ypi * c00[0] + yp * c01[0])
        + xp * (ypi * c10[0] + yp * c11[0])
        + sixth
        * hx2
        * (
            cxi * (ypi * c00[1] + yp * c01[1])
            + cx * (ypi * c10[1] + yp * c11[1])
        )
        + sixth
        * hy2
        * (
            xpi * (cyi * c00[2] + cy * c01[2])
            + xp * (cyi * c10[2] + cy * c11[2])
        )
        + z36th
        * hx2
        * hy2
        * (
            cxi * (cyi * c00[3] + cy * c01[3])
            + cx * (cyi * c10[3] + cy * c11[3])
        )
    )
    df_dx = (
        hxi * (-(ypi * c00[0] + yp * c01[0]) + (ypi * c10[0] + yp * c11[0]))
        + sixth
        * hx
        * (
            cxdi * (ypi * c00[1] + yp * c01[1])
            + cxd * (ypi * c10[1] + yp * c11[1])
        )
        + sixth
        * hxi
        * hy2
        * (-(cyi * c00[2] + cy * c01[2]) + (cyi * c10[2] + cy * c11[2]))
        + z36th
        * hx
        * hy2
        * (
            cxdi * (cyi * c00[3] + cy * c01[3])
            + cxd * (cyi * c10[3] + cy * c11[3])
        )
    )
    df_dy = (
        hyi * (xpi * (-c00[0] + c01[0]) + xp * (-c10[0] + c11[0]))
        + sixth
        * hx2
        * hyi
        * (cxi * (-c00[1] + c01[1]) + cx * (-c10[1] + c11[1]))
        + sixth
        * hy
        * (
            xpi * (cydi * c00[2] + cyd * c01[2])
            + xp * (cydi * c10[2] + cyd * c11[2])
        )
        + z36th
        * hx2
        * hy
        * (
            cxi * (cydi * c00[3] + cyd * c01[3])
            + cx * (cydi * c10[3] + cyd * c11[3])
        )
    )
    d2f_dx2 = (
        xpi * (ypi * c00[1] + yp * c01[1])
        + xp * (ypi * c10[1] + yp * c11[1])
        + sixth
        * hy2
        * (
            xpi * (cyi * c00[3] + cy * c01[3])
            + xp * (cyi * c10[3] + cy * c11[3])
        )
    )
    d2f_dy2 = (
        xpi * (ypi * c00[2] + yp * c01[2])
        + xp * (ypi * c10[2] + yp * c11[2])
        + sixth
        * hx2
        * (
            cxi * (ypi * c00[3] + yp * c01[3])
            + cx * (ypi * c10[3] + yp * c11[3])
        )
    )
    d2f_dxdy = (
        hxi * hyi * (c00[0] - c01[0] - c10[0] + c11[0])
        + sixth
        * hx
        * hyi
        * (cxdi * (-c00[1] + c01[1]) + cxd * (-c10[1] + c11[1]))
        + sixth
        * hxi
        * hy
        * (-(cydi * c00[2] + cyd * c01[2]) + (cydi * c10[2] + cyd * c11[2]))
        + z36th
        * hx
        * hy
        * (
            cxdi * (cydi * c00[3] + cyd * c01[3])
            + cxd * (cydi * c10[3] + cyd * c11[3])
        )
    )
    return BicubicEval(
        f=f,
        df_dx=df_dx,
        df_dy=df_dy,
        d2f_dx2=d2f_dx2,
        d2f_dy2=d2f_dy2,
        d2f_dxdy=d2f_dxdy,
    )


def interp_evbicub_mask_db(
    xget: float | jnp.ndarray,
    yget: float | jnp.ndarray,
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    ilinx: bool,
    iliny: bool,
    coeffs: np.ndarray | jnp.ndarray,
    ict: Sequence[int | bool],
) -> jnp.ndarray:
    """Packs selected bicubic outputs according to 6-element mask ``ict``."""
    ev = interp_evbicub_db(xget, yget, x, y, ilinx, iliny, coeffs)
    items = [ev.f, ev.df_dx, ev.df_dy, ev.d2f_dx2, ev.d2f_dy2, ev.d2f_dxdy]
    out = [0.0] * 6
    iadr = 0
    for k in range(6):
        if bool(ict[k]):
            out[iadr] = float(items[k])
            iadr += 1
    return jnp.asarray(out, dtype=jnp.float64)
