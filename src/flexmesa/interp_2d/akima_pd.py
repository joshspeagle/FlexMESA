# SPDX-License-Identifier: LGPL-3.0-or-later
"""Akima (1996, ACM Algorithm 760) partial-derivative estimation (`rgpd3p_db`).

Estimates first partial derivatives $(z_x, z_y)$ and cross partial derivative
$z_{xy}$ at every node of a rectangular grid $(x_d, y_d)$ by taking volatility-
weighted averages over up to four 4-point 1D stencils along each coordinate and
up to sixteen 4x4 2D stencils across the grid.

Reference:
    Akima, H. (1996), "Algorithm 760: Rectangular-Grid-Data Surface Fitting
    that Has the Accuracy of a Bicubic Polynomial", ACM Transactions on
    Mathematical Software, 22(3), 357-361.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa.interp_2d import spline1d

# Relative index offsets for the 4 candidate 4-point stencils around i0.
_IDLT: tuple[tuple[int, int, int], ...] = (
    (-3, -2, -1),
    (-2, -1, 1),
    (-1, 1, 2),
    (1, 2, 3),
)


def _z2f(xx1: float, xx2: float, zz0: float, zz1: float) -> float:
    """Linear extrapolation helper for 2-point grids in ``RGPD3P_db``.

    Extrapolates a linear profile passing through ``(0, zz0)`` and
    ``(xx1, zz1)`` to displacement ``xx2``.
    """
    return (zz1 - zz0) * xx2 / xx1 + zz0


def _z3f(
    xx1: float, xx2: float, xx3: float, zz0: float, zz1: float, zz2: float
) -> float:
    """Quadratic extrapolation helper for 3-point grids in ``RGPD3P_db``.

    Extrapolates a quadratic polynomial passing through ``(0, zz0)``,
    ``(xx1, zz1)``, and ``(xx2, zz2)`` to displacement ``xx3``.
    """
    return (
        (zz2 - zz0) * (xx3 - xx1) / xx2 - (zz1 - zz0) * (xx3 - xx2) / xx1
    ) * (xx3 / (xx2 - xx1)) + zz0


def _eval_1d_stencils(
    grid: np.ndarray, i0: int, z_line: np.ndarray, eps_scale: float
) -> tuple[tuple[np.ndarray, ...], float]:
    """Evaluates the 4 Akima 1D stencils along a single grid line.

    Computes the primary derivative estimate ``pez``, least-squares linear fit
    ``(b00, b1)``, and volatility ``volf`` for each of the four 4-point
    stencils around ``i0``, returning the cached stencil arrays and weighted
    1D derivative estimate.
    """
    n = grid.size
    n0 = max(4, n)
    c0 = float(grid[i0])
    z00 = float(z_line[i0])
    da = np.zeros((4, 3), dtype=np.float64)
    za = np.zeros((4, 3), dtype=np.float64)
    ca = np.zeros((4, 3), dtype=np.float64)
    sa = np.zeros(4, dtype=np.float64)
    ssa = np.zeros(4, dtype=np.float64)
    b00a = np.zeros(4, dtype=np.float64)
    b1a = np.zeros(4, dtype=np.float64)
    smpef = smwtf = smpei = smwti = 0.0

    for ipe, (d_1, d_2, d_3) in enumerate(_IDLT):
        i1, i2, i3 = i0 + d_1, i0 + d_2, i0 + d_3
        if i1 < 0 or i2 < 0 or i3 < 0 or i1 >= n0 or i2 >= n0 or i3 >= n0:
            continue
        d1 = float(grid[i1] - c0)
        z1 = float(z_line[i1])
        # Supplement stencil coordinates and values when grid has < 4 points.
        if n >= 4:
            d2, d3 = float(grid[i2] - c0), float(grid[i3] - c0)
            z2, z3 = float(z_line[i2]), float(z_line[i3])
        elif n == 3:
            d2 = float(grid[i2] - c0)
            z2 = float(z_line[i2])
            d3 = float(2.0 * grid[2] - grid[1] - c0)
            z3 = _z3f(d1, d2, d3, z00, z1, z2)
        else:
            d2 = float(2.0 * grid[1] - grid[0] - c0)
            d3 = float(2.0 * grid[0] - grid[1] - c0)
            z2 = _z2f(d1, d2, z00, z1)
            z3 = _z2f(d1, d3, z00, z1)

        # Lagrange 4-point cubic derivative coefficients at displacement 0.
        c1 = d2 * d3 / ((d1 - d2) * (d1 - d3))
        c2 = d3 * d1 / ((d2 - d3) * (d2 - d1))
        c3 = d1 * d2 / ((d3 - d1) * (d3 - d2))
        pez = c1 * (z1 - z00) / d1 + c2 * (z2 - z00) / d2 + c3 * (z3 - z00) / d3

        # Volatility factor volf from linear least-squares fit residuals.
        # When volf exceeds machine-precision threshold epsln, weight by
        # 1 / (volf * ss); otherwise accumulate unweighted exact linear fit.
        s = d1 + d2 + d3
        sz = z00 + z1 + z2 + z3
        ss = d1 * d1 + d2 * d2 + d3 * d3
        szd = d1 * z1 + d2 * z2 + d3 * z3
        dnm = 4.0 * ss - s * s
        b00 = (ss * sz - s * szd) / dnm
        b1 = (4.0 * szd - s * sz) / dnm
        volf = (
            (z00 - b00) ** 2
            + (z1 - (b00 + b1 * d1)) ** 2
            + (z2 - (b00 + b1 * d2)) ** 2
            + (z3 - (b00 + b1 * d3)) ** 2
        )
        epsln = (z00 * z00 + z1 * z1 + z2 * z2 + z3 * z3) * eps_scale
        if volf > epsln:
            wt = 1.0 / (volf * ss)
            smpef += wt * pez
            smwtf += wt
        else:
            smpei += pez
            smwti += 1.0
        da[ipe] = (d1, d2, d3)
        za[ipe] = (z1, z2, z3)
        ca[ipe] = (c1, c2, c3)
        sa[ipe], ssa[ipe], b00a[ipe], b1a[ipe] = s, ss, b00, b1

    deriv = smpef / smwtf if smwti < 0.5 else smpei / smwti
    return (da, za, ca, sa, ssa, b00a, b1a), deriv


def _supplement_z_grid(
    nxd: int,
    nyd: int,
    z_arr: np.ndarray,
    ixs: tuple[int, int, int],
    iys: tuple[int, int, int],
    xs_loc: np.ndarray,
    ys_loc: np.ndarray,
    zs_x: np.ndarray,
    zs_y: np.ndarray,
) -> np.ndarray:
    """Builds the 3x3 stencil value matrix with boundary extrapolation.

    Populates interior entries from ``z_arr`` and extrapolates missing rows or
    columns via ``_z2f`` or ``_z3f`` when ``nxd < 4`` or ``nyd < 4``.
    """
    z_grid = np.zeros((3, 3), dtype=np.float64)
    nx_in = min(nxd - 1, 3)
    ny_in = min(nyd - 1, 3)
    for r in range(nx_in):
        for c in range(ny_in):
            z_grid[r, c] = z_arr[ixs[r], iys[c]]
    x1, x2, x3 = float(xs_loc[0]), float(xs_loc[1]), float(xs_loc[2])
    y1, y2, y3 = float(ys_loc[0]), float(ys_loc[1]), float(ys_loc[2])
    for c in range(ny_in):
        if nxd == 3:
            z_grid[2, c] = _z3f(
                x1,
                x2,
                x3,
                float(zs_y[c]),
                float(z_grid[0, c]),
                float(z_grid[1, c]),
            )
        elif nxd == 2:
            z_grid[1, c] = _z2f(x1, x2, float(zs_y[c]), float(z_grid[0, c]))
            z_grid[2, c] = _z2f(x1, x3, float(zs_y[c]), float(z_grid[0, c]))
    for r in range(3):
        if nyd == 3:
            z_grid[r, 2] = _z3f(
                y1,
                y2,
                y3,
                float(zs_x[r]),
                float(z_grid[r, 0]),
                float(z_grid[r, 1]),
            )
        elif nyd == 2:
            z_grid[r, 1] = _z2f(y1, y2, float(zs_x[r]), float(z_grid[r, 0]))
            z_grid[r, 2] = _z2f(y1, y3, float(zs_x[r]), float(z_grid[r, 0]))
    return z_grid


def rgpd3p_db(
    xd: np.ndarray | jnp.ndarray,
    yd: np.ndarray | jnp.ndarray,
    zd: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Estimates partial derivatives ``(z_x, z_y, z_xy)`` on grid `(xd, yd)`.

    Implements Akima (1996) subroutine ``RGPD3P_db`` in ``float64``. At each
    grid point ``(xd[ix], yd[iy])``, evaluates up to four 4-point 1D Lagrange
    stencils along ``x`` and ``y`` and up to sixteen 4x4 2D tensor-product
    stencils to estimate ``(z_x, z_y, z_xy)`` without spurious oscillations.

    Args:
        xd: Strictly ascending x-grid of shape ``(nxd,)`` (``nxd >= 2``) in
            physical or dimensionless units.
        yd: Strictly ascending y-grid of shape ``(nyd,)`` (``nyd >= 2``) in
            physical or dimensionless units.
        zd: Grid values of shape ``(nxd, nyd)`` or flat ``(nxd * nyd,)``.

    Returns:
        JAX ``float64`` array of shape ``(nxd, nyd, 3)`` containing
        ``[zx, zy, zxy]`` at each grid point.

    Raises:
        ValueError: If ``nxd < 2``, ``nyd < 2``, grids are non-ascending, or
            ``zd`` contains non-finite values.
    """
    x_arr = np.asarray(xd, dtype=np.float64)
    y_arr = np.asarray(yd, dtype=np.float64)
    nxd, nyd = x_arr.size, y_arr.size
    if nxd < 2 or nyd < 2:
        raise ValueError("rgpd3p_db requires nxd >= 2 and nyd >= 2.")
    spline1d.splinck_db(x_arr, 1.0e-3, "x")
    spline1d.splinck_db(y_arr, 1.0e-3, "y")
    z_arr = np.asarray(zd, dtype=np.float64).reshape(nxd, nyd)
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("zd must contain only finite float64 values.")

    nx0, ny0 = max(4, nxd), max(4, nyd)
    eps_scale = float(np.float32(1.0e-12))
    pdd = np.zeros((nxd, nyd, 3), dtype=np.float64)

    for iy0 in range(nyd):
        for ix0 in range(nxd):
            z00 = float(z_arr[ix0, iy0])
            (xa, zi0a, cxa, sxa, sxxa, b00xa, b10a), zxdi = _eval_1d_stencils(
                x_arr, ix0, z_arr[:, iy0], eps_scale
            )
            (ya, z0ia, cya, sya, syya, b00ya, b01a), zydi = _eval_1d_stencils(
                y_arr, iy0, z_arr[ix0, :], eps_scale
            )
            # Combine valid x and y stencils to estimate cross derivative z_xy.
            smpef = smwtf = smpei = smwti = 0.0
            for ipex, (dx1, dx2, dx3) in enumerate(_IDLT):
                ix1, ix2, ix3 = ix0 + dx1, ix0 + dx2, ix0 + dx3
                if min(ix1, ix2, ix3) < 0 or max(ix1, ix2, ix3) >= nx0:
                    continue
                xs_loc, zs_x, cxs = xa[ipex], zi0a[ipex], cxa[ipex]
                sx, sxx, b00x, b10 = (
                    sxa[ipex],
                    sxxa[ipex],
                    b00xa[ipex],
                    b10a[ipex],
                )
                for ipey, (dy1, dy2, dy3) in enumerate(_IDLT):
                    iy1, iy2, iy3 = iy0 + dy1, iy0 + dy2, iy0 + dy3
                    if min(iy1, iy2, iy3) < 0 or max(iy1, iy2, iy3) >= ny0:
                        continue
                    ys_loc, zs_y, cys = ya[ipey], z0ia[ipey], cya[ipey]
                    sy, syy, b00y, b01 = (
                        sya[ipey],
                        syya[ipey],
                        b00ya[ipey],
                        b01a[ipey],
                    )
                    z_grid = _supplement_z_grid(
                        nxd,
                        nyd,
                        z_arr,
                        (ix1, ix2, ix3),
                        (iy1, iy2, iy3),
                        xs_loc,
                        ys_loc,
                        zs_x,
                        zs_y,
                    )
                    dzxy = (
                        z_grid - zs_x[:, None] - zs_y[None, :] + z00
                    ) / np.outer(xs_loc, ys_loc)
                    pezxy = float(cxs @ (dzxy @ cys))

                    # Evaluate 2D bilinear least-squares plane and volatility
                    # across the 16-point stencil (center + 3x1 + 1x3 + 3x3).
                    b00 = 0.5 * (b00x + b00y)
                    sxxyy = sxx * syy
                    sxyz = float(xs_loc @ (z_grid @ ys_loc))
                    b11 = (
                        sxyz - b00 * sx * sy - b10 * sxx * sy - b01 * sx * syy
                    ) / sxxyy
                    pred = (
                        b00
                        + b01 * ys_loc[None, :]
                        + xs_loc[:, None] * (b10 + b11 * ys_loc[None, :])
                    )
                    # Sum squared deviations over all 16 points of the 2D
                    # stencil and compare against relative machine threshold.
                    volf = (
                        (z00 - b00) ** 2
                        + float(np.sum((zs_y - (b00 + b01 * ys_loc)) ** 2))
                        + float(np.sum((zs_x - (b00 + b10 * xs_loc)) ** 2))
                        + float(np.sum((z_grid - pred) ** 2))
                    )
                    z_sq = (
                        z00 * z00
                        + float(np.sum(zs_y * zs_y))
                        + float(np.sum(zs_x * zs_x))
                        + float(np.sum(z_grid * z_grid))
                    )
                    if volf > z_sq * eps_scale:
                        wt = 1.0 / (volf * sxxyy)
                        smpef += wt * pezxy
                        smwtf += wt
                    else:
                        smpei += pezxy
                        smwti += 1.0
            zxydi = smpef / smwtf if smwti < 0.5 else smpei / smwti
            pdd[ix0, iy0] = (zxdi, zydi, zxydi)
    return jnp.asarray(pdd, dtype=jnp.float64)
