# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pre-validated JAX `float64` 2D bicubic spline evaluation (`bicub_fast`).

Hoists $O(n_x n_y)$ grid monotonicity (`splinck_db`) and coefficient
finiteness checks outside the zone loop so Stage 1, Stage 2 (`@jax.jit`), and
Stage 3 (`jax.vmap`) benchmarks measure pure 16-term Hermite cell evaluation
across $N_{\\mathrm{zones}}$ queries.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def locate_uniform_cell_jax(
    query: jax.Array,
    grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Locates cell index `idx`, normalized offset `xp`, `h`, and `1 / h`.

    Args:
        query: Query coordinate scalar (`float64` `jax.Array`).
        grid: Strictly ascending uniform 1D grid of shape `(N,)`.

    Returns:
        4-tuple `(idx, xp, h, hi)` of `int32`/`float64` JAX scalars.
    """
    n = grid.shape[0]
    zq = jnp.clip(query, grid[0], grid[-1])
    d_avg = (grid[-1] - grid[0]) / float(n - 1)
    idx = jnp.clip(
        jnp.floor((zq - grid[0]) / d_avg).astype(jnp.int32), 0, n - 2
    )
    idx = jnp.where((zq < grid[idx]) & (idx > 0), idx - 1, idx)
    idx = jnp.where((zq > grid[idx + 1]) & (idx + 1 < n - 1), idx + 1, idx)
    h = grid[idx + 1] - grid[idx]
    hi = 1.0 / h
    xp = (zq - grid[idx]) * hi
    return idx, xp, h, hi


def eval_bicub_point_jax(
    xq: jax.Array,
    yq: jax.Array,
    x_grid: jax.Array,
    y_grid: jax.Array,
    coeffs: jax.Array,
) -> jax.Array:
    """Evaluates `[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]` in pure JAX.

    Args:
        xq: Query x-coordinate (`float64` scalar, $\\log_{10}\\rho$).
        yq: Query y-coordinate (`float64` scalar, $\\log_{10}T$).
        x_grid: Uniform x-grid of shape `(nx,)`.
        y_grid: Uniform y-grid of shape `(ny,)`.
        coeffs: Bicubic spline coefficients of shape `(nx, ny, 4)`.

    Returns:
        6-element `float64` JAX array `[f, df_dx, df_dy, d2f_dx2, d2f_dy2,
        d2f_dxdy]`.
    """
    i, xp, hx, hxi = locate_uniform_cell_jax(xq, x_grid)
    j, yp, hy, hyi = locate_uniform_cell_jax(yq, y_grid)
    xpi, ypi = 1.0 - xp, 1.0 - yp
    xp2, xpi2 = xp * xp, xpi * xpi
    yp2, ypi2 = yp * yp, ypi * ypi
    cx, cxi = xp * (xp2 - 1.0), xpi * (xpi2 - 1.0)
    cxd, cxdi = 3.0 * xp2 - 1.0, -3.0 * xpi2 + 1.0
    cy, cyi = yp * (yp2 - 1.0), ypi * (ypi2 - 1.0)
    cyd, cydi = 3.0 * yp2 - 1.0, -3.0 * ypi2 + 1.0
    hx2, hy2 = hx * hx, hy * hy

    c00, c01 = coeffs[i, j], coeffs[i, j + 1]
    c10, c11 = coeffs[i + 1, j], coeffs[i + 1, j + 1]
    y0 = ypi * c00 + yp * c01
    y1 = ypi * c10 + yp * c11
    cy0 = cyi * c00 + cy * c01
    cy1 = cyi * c10 + cy * c11
    cyd0 = cydi * c00 + cyd * c01
    cyd1 = cydi * c10 + cyd * c11
    df0 = c01 - c00
    df1 = c11 - c10

    sixth, z36th = 1.0 / 6.0, 1.0 / 36.0
    f = (
        xpi * y0[0]
        + xp * y1[0]
        + sixth * hx2 * (cxi * y0[1] + cx * y1[1])
        + sixth * hy2 * (xpi * cy0[2] + xp * cy1[2])
        + z36th * hx2 * hy2 * (cxi * cy0[3] + cx * cy1[3])
    )
    df_dx = (
        hxi * (y1[0] - y0[0])
        + sixth * hx * (cxdi * y0[1] + cxd * y1[1])
        + sixth * hxi * hy2 * (cy1[2] - cy0[2])
        + z36th * hx * hy2 * (cxdi * cy0[3] + cxd * cy1[3])
    )
    df_dy = (
        hyi * (xpi * df0[0] + xp * df1[0])
        + sixth * hx2 * hyi * (cxi * df0[1] + cx * df1[1])
        + sixth * hy * (xpi * cyd0[2] + xp * cyd1[2])
        + z36th * hx2 * hy * (cxi * cyd0[3] + cx * cyd1[3])
    )
    d2f_dx2 = (
        xpi * y0[1] + xp * y1[1] + sixth * hy2 * (xpi * cy0[3] + xp * cy1[3])
    )
    d2f_dy2 = (
        xpi * y0[2] + xp * y1[2] + sixth * hx2 * (cxi * y0[3] + cx * y1[3])
    )
    d2f_dxdy = (
        hxi * hyi * (df1[0] - df0[0])
        + sixth * hx * hyi * (cxdi * df0[1] + cxd * df1[1])
        + sixth * hxi * hy * (cyd1[2] - cyd0[2])
        + z36th * hx * hy * (cxdi * cyd0[3] + cxd * cyd1[3])
    )
    return jnp.stack([f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy])


bicub_vmap = jax.vmap(eval_bicub_point_jax, in_axes=(0, 0, None, None, None))
bicub_jit = jax.jit(bicub_vmap)
bicub_batch_jit = jax.jit(
    jax.vmap(bicub_vmap, in_axes=(0, 0, None, None, None))
)
