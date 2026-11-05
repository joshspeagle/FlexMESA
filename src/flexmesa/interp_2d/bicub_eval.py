# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional bicubic spline evaluation (`herm2xy_db`, `fvbicub_db`).

Evaluates $C^2$ bicubic splines and their first and second partial derivatives
in ``float64`` JAX arithmetic, compatible with ``jax.grad`` and ``jax.hessian``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np


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


def _locate_axis_jax(
    axis: jnp.ndarray,
    val: jnp.ndarray,
    uniform: bool,
) -> jnp.ndarray:
    """Locates the 0-based grid cell index along ``axis``."""
    n = axis.shape[0]
    nm = n - 1
    if uniform:
        raw = jnp.floor(
            float(nm) * (val - axis[0]) / (axis[-1] - axis[0])
        ).astype(jnp.int32)
        idx = jnp.clip(raw, 0, nm - 1)
        idx = jnp.where((val < axis[idx]) & (idx > 0), idx - 1, idx)
        return jnp.where(
            (val > axis[idx + 1]) & (idx + 1 < nm), idx + 1, idx
        )
    return jnp.clip(jnp.searchsorted(axis, val, side="right") - 1, 0, nm - 1)


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
    """Locates the 2D grid cell and local coordinates (`herm2xy_db`).

    Args:
        xget: Query x-coordinate in physical or dimensionless units.
        yget: Query y-coordinate in physical or dimensionless units.
        x: Strictly ascending x-grid of shape ``(nx,)``.
        y: Strictly ascending y-grid of shape ``(ny,)``.
        ilinx: Whether ``x`` is uniformly spaced within ``1e-3``.
        iliny: Whether ``y`` is uniformly spaced within ``1e-3``.

    Returns:
        8-tuple ``(ix, jy, xp, yp, hx, hxi, hy, hyi)`` of ``float64``/``int32``
        JAX scalars.
    """
    x_grid = jnp.asarray(x, dtype=jnp.float64)
    y_grid = jnp.asarray(y, dtype=jnp.float64)
    xq = jnp.asarray(xget, dtype=jnp.float64)
    yq = jnp.asarray(yget, dtype=jnp.float64)
    zx = jnp.clip(xq, x_grid[0], x_grid[-1])
    zy = jnp.clip(yq, y_grid[0], y_grid[-1])

    ix = _locate_axis_jax(x_grid, zx, bool(ilinx))
    jy = _locate_axis_jax(y_grid, zy, bool(iliny))
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

    Args:
        xget: Query x-coordinate in physical or dimensionless units.
        yget: Query y-coordinate in physical or dimensionless units.
        x: Strictly ascending x-grid of shape ``(nx,)``.
        y: Strictly ascending y-grid of shape ``(ny,)``.
        ilinx: Uniform spacing flag for ``x``.
        iliny: Uniform spacing flag for ``y``.
        coeffs: Bicubic coefficient array of shape ``(nx, ny, 4)`` from
            :func:`flexmesa.interp_2d.bicub.interp_mkbicub_db`.

    Returns:
        :class:`BicubicEval` containing ``(f, df_dx, df_dy, d2f_dx2, d2f_dy2,
        d2f_dxdy)``.
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

    # Shared y-interpolated combinations along left and right x-edges.
    y0_f = ypi * c00[0] + yp * c01[0]
    y1_f = ypi * c10[0] + yp * c11[0]
    y0_fxx = ypi * c00[1] + yp * c01[1]
    y1_fxx = ypi * c10[1] + yp * c11[1]
    y0_fyy = ypi * c00[2] + yp * c01[2]
    y1_fyy = ypi * c10[2] + yp * c11[2]
    y0_fxxyy = ypi * c00[3] + yp * c01[3]
    y1_fxxyy = ypi * c10[3] + yp * c11[3]

    cy0_fyy = cyi * c00[2] + cy * c01[2]
    cy1_fyy = cyi * c10[2] + cy * c11[2]
    cy0_fxxyy = cyi * c00[3] + cy * c01[3]
    cy1_fxxyy = cyi * c10[3] + cy * c11[3]

    cyd0_fyy = cydi * c00[2] + cyd * c01[2]
    cyd1_fyy = cydi * c10[2] + cyd * c11[2]
    cyd0_fxxyy = cydi * c00[3] + cyd * c01[3]
    cyd1_fxxyy = cydi * c10[3] + cyd * c11[3]

    df0_y = c01[0] - c00[0]
    df1_y = c11[0] - c10[0]
    dfxx0_y = c01[1] - c00[1]
    dfxx1_y = c11[1] - c10[1]

    f = (
        xpi * y0_f
        + xp * y1_f
        + sixth * hx2 * (cxi * y0_fxx + cx * y1_fxx)
        + sixth * hy2 * (xpi * cy0_fyy + xp * cy1_fyy)
        + z36th * hx2 * hy2 * (cxi * cy0_fxxyy + cx * cy1_fxxyy)
    )
    df_dx = (
        hxi * (y1_f - y0_f)
        + sixth * hx * (cxdi * y0_fxx + cxd * y1_fxx)
        + sixth * hxi * hy2 * (cy1_fyy - cy0_fyy)
        + z36th * hx * hy2 * (cxdi * cy0_fxxyy + cxd * cy1_fxxyy)
    )
    df_dy = (
        hyi * (xpi * df0_y + xp * df1_y)
        + sixth * hx2 * hyi * (cxi * dfxx0_y + cx * dfxx1_y)
        + sixth * hy * (xpi * cyd0_fyy + xp * cyd1_fyy)
        + z36th * hx2 * hy * (cxi * cyd0_fxxyy + cx * cyd1_fxxyy)
    )
    d2f_dx2 = (
        xpi * y0_fxx
        + xp * y1_fxx
        + sixth * hy2 * (xpi * cy0_fxxyy + xp * cy1_fxxyy)
    )
    d2f_dy2 = (
        xpi * y0_fyy
        + xp * y1_fyy
        + sixth * hx2 * (cxi * y0_fxxyy + cx * y1_fxxyy)
    )
    d2f_dxdy = (
        hxi * hyi * (df1_y - df0_y)
        + sixth * hx * hyi * (cxdi * dfxx0_y + cxd * dfxx1_y)
        + sixth * hxi * hy * (cyd1_fyy - cyd0_fyy)
        + z36th * hx * hy * (cxdi * cyd0_fxxyy + cxd * cyd1_fxxyy)
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
    """Packs selected bicubic outputs according to 6-element mask ``ict``.

    Args:
        xget: Query x-coordinate.
        yget: Query y-coordinate.
        x: Strictly ascending x-grid of shape ``(nx,)``.
        y: Strictly ascending y-grid of shape ``(ny,)``.
        ilinx: Uniform spacing flag for ``x``.
        iliny: Uniform spacing flag for ``y``.
        coeffs: Bicubic coefficient array of shape ``(nx, ny, 4)``.
        ict: 6-element selector sequence for ``[f, df_dx, df_dy, d2f_dx2,
            d2f_dy2, d2f_dxdy]``.

    Returns:
        JAX ``float64`` array of shape ``(6,)`` containing requested values
        packed from index 0.
    """
    ev = interp_evbicub_db(xget, yget, x, y, ilinx, iliny, coeffs)
    items = [ev.f, ev.df_dx, ev.df_dy, ev.d2f_dx2, ev.d2f_dy2, ev.d2f_dxdy]
    out = [0.0] * 6
    iadr = 0
    for k in range(6):
        if bool(ict[k]):
            out[iadr] = float(items[k])
            iadr += 1
    return jnp.asarray(out, dtype=jnp.float64)
