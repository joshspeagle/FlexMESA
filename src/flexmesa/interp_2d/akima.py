# SPDX-License-Identifier: LGPL-3.0-or-later
"""Akima (1996, ACM Algorithm 760) rectangular-grid bivariate interpolation.

Reference:
    Akima, H. (1996), "Algorithm 760: Rectangular-Grid-Data Surface Fitting
    that Has the Accuracy of a Bicubic Polynomial", ACM Transactions on
    Mathematical Software, 22(3), 357-361.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa.interp_2d import bicub

_IDLT: tuple[tuple[int, int, int], ...] = (
    (-3, -2, -1),
    (-2, -1, 1),
    (-1, 1, 2),
    (1, 2, 3),
)


def _z2f(xx1: float, xx2: float, zz0: float, zz1: float) -> float:
    return (zz1 - zz0) * xx2 / xx1 + zz0


def _z3f(
    xx1: float, xx2: float, xx3: float, zz0: float, zz1: float, zz2: float
) -> float:
    return (
        (zz2 - zz0) * (xx3 - xx1) / xx2 - (zz1 - zz0) * (xx3 - xx2) / xx1
    ) * (xx3 / (xx2 - xx1)) + zz0


def _eval_1d_stencils(
    grid: np.ndarray, i0: int, z_line: np.ndarray, eps_scale: float
) -> tuple[tuple[np.ndarray, ...], float]:
    """Evaluates the 4 Akima 1D stencils along a single grid line."""
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

        c1 = d2 * d3 / ((d1 - d2) * (d1 - d3))
        c2 = d3 * d1 / ((d2 - d3) * (d2 - d1))
        c3 = d1 * d2 / ((d3 - d1) * (d3 - d2))
        pez = c1 * (z1 - z00) / d1 + c2 * (z2 - z00) / d2 + c3 * (z3 - z00) / d3

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


def rgpd3p_db(
    xd: np.ndarray | jnp.ndarray,
    yd: np.ndarray | jnp.ndarray,
    zd: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Estimates partial derivatives ``(z_x, z_y, z_xy)`` on grid `(xd, yd)`.

    Args:
        xd: Strictly ascending x-grid of shape ``(nxd,)`` (``nxd >= 2``).
        yd: Strictly ascending y-grid of shape ``(nyd,)`` (``nyd >= 2``).
        zd: Grid values of shape ``(nxd, nyd)`` or flat ``(nxd * nyd,)``.

    Returns:
        JAX ``float64`` array of shape ``(nxd, nyd, 3)`` containing
        ``[zx, zy, zxy]`` at each grid point.
    """
    x_arr = np.asarray(xd, dtype=np.float64)
    y_arr = np.asarray(yd, dtype=np.float64)
    nxd, nyd = x_arr.size, y_arr.size
    if nxd < 2 or nyd < 2:
        raise ValueError("rgpd3p_db requires nxd >= 2 and nyd >= 2.")
    bicub.splinck_db(x_arr, 1.0e-3, "x")
    bicub.splinck_db(y_arr, 1.0e-3, "y")
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
            smpef = smwtf = smpei = smwti = 0.0
            for ipex, (dx1, dx2, dx3) in enumerate(_IDLT):
                ix1, ix2, ix3 = ix0 + dx1, ix0 + dx2, ix0 + dx3
                if min(ix1, ix2, ix3) < 0 or max(ix1, ix2, ix3) >= nx0:
                    continue
                xs_loc = xa[ipex]
                zs_x = zi0a[ipex]
                cxs = cxa[ipex]
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
                    ys_loc = ya[ipey]
                    zs_y = z0ia[ipey]
                    cys = cya[ipey]
                    sy, syy, b00y, b01 = (
                        sya[ipey],
                        syya[ipey],
                        b00ya[ipey],
                        b01a[ipey],
                    )
                    z_grid = z_arr[
                        np.ix_(
                            [
                                min(ix1, nxd - 1),
                                min(ix2, nxd - 1),
                                min(ix3, nxd - 1),
                            ],
                            [
                                min(iy1, nyd - 1),
                                min(iy2, nyd - 1),
                                min(iy3, nyd - 1),
                            ],
                        )
                    ].copy()
                    dzxy = (
                        z_grid - zs_x[:, None] - zs_y[None, :] + z00
                    ) / np.outer(xs_loc, ys_loc)
                    pezxy = float(cxs @ (dzxy @ cys))

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


def _eval_akima_point(
    xd: np.ndarray,
    yd: np.ndarray,
    zd: np.ndarray,
    pdd: np.ndarray,
    xii: float,
    yii: float,
) -> float:
    """Evaluates the Akima (1996) polynomial at a single point `(xii, yii)`."""
    nxd, nyd = xd.size, yd.size
    ixdi = (
        0
        if xii <= xd[0]
        else (
            int(np.searchsorted(xd, xii, side="right")) if xii < xd[-1] else nxd
        )
    )
    iydi = (
        0
        if yii <= yd[0]
        else (
            int(np.searchsorted(yd, yii, side="right")) if yii < yd[-1] else nyd
        )
    )
    ixd0, iyd0 = max(1, ixdi) - 1, max(1, iydi) - 1

    x0, y0 = float(xd[ixd0]), float(yd[iyd0])
    z00 = float(zd[ixd0, iyd0])
    zx00, zy00, zxy00 = (float(v) for v in pdd[ixd0, iyd0])
    u, v = xii - x0, yii - y0

    if 0 < ixdi < nxd and 0 < iydi < nyd:
        ixd1, iyd1 = ixd0 + 1, iyd0 + 1
        dx, dy = float(xd[ixd1] - x0), float(yd[iyd1] - y0)
        dxsq, dysq = dx * dx, dy * dy
        z10, z01, z11 = (
            float(zd[ixd1, iyd0]),
            float(zd[ixd0, iyd1]),
            float(zd[ixd1, iyd1]),
        )
        zx10, zy10, zxy10 = (float(val) for val in pdd[ixd1, iyd0])
        zx01, zy01, zxy01 = (float(val) for val in pdd[ixd0, iyd1])
        zx11, zy11, zxy11 = (float(val) for val in pdd[ixd1, iyd1])

        z0dx = (z10 - z00) / dx
        z0dy = (z01 - z00) / dy
        z1dy = (z11 - z10) / dy
        zx0dy = (zx01 - zx00) / dy
        zx1dy = (zx11 - zx10) / dy
        zy0dx = (zy10 - zy00) / dx
        zy1dx = (zy11 - zy01) / dx
        zdxdy = (z1dy - z0dy) / dx

        a = zdxdy - zx0dy - zy0dx + zxy00
        b = zx1dy - zx0dy - zxy10 + zxy00
        c = zy1dx - zy0dx - zxy01 + zxy00
        d = zxy11 - zxy10 - zxy01 + zxy00

        p02 = (2.0 * (z0dy - zy00) + z0dy - zy01) / dy
        p03 = (-2.0 * z0dy + zy01 + zy00) / dysq
        p12 = (2.0 * (zx0dy - zxy00) + zx0dy - zxy01) / dy
        p13 = (-2.0 * zx0dy + zxy01 + zxy00) / dysq
        p20 = (2.0 * (z0dx - zx00) + z0dx - zx10) / dx
        p21 = (2.0 * (zy0dx - zxy00) + zy0dx - zxy10) / dx
        p22 = (3.0 * (3.0 * a - b - c) + d) / (dx * dy)
        p23 = (-6.0 * a + 2.0 * b + 3.0 * c - d) / (dx * dysq)
        p30 = (-2.0 * z0dx + zx10 + zx00) / dxsq
        p31 = (-2.0 * zy0dx + zxy10 + zxy00) / dxsq
        p32 = (-6.0 * a + 3.0 * b + 2.0 * c - d) / (dxsq * dy)
        p33 = (2.0 * (2.0 * a - b - c) + d) / (dxsq * dysq)

        q0 = z00 + v * (zy00 + v * (p02 + v * p03))
        q1 = zx00 + v * (zxy00 + v * (p12 + v * p13))
        q2 = p20 + v * (p21 + v * (p22 + v * p23))
        q3 = p30 + v * (p31 + v * (p32 + v * p33))
        return q0 + u * (q1 + u * (q2 + u * q3))
    if 0 < ixdi < nxd:
        ixd1 = ixd0 + 1
        dx = float(xd[ixd1] - x0)
        dxsq = dx * dx
        z10 = float(zd[ixd1, iyd0])
        zx10, zy10, zxy10 = (float(val) for val in pdd[ixd1, iyd0])
        z0dx = (z10 - z00) / dx
        zy0dx = (zy10 - zy00) / dx
        p20 = (2.0 * (z0dx - zx00) + z0dx - zx10) / dx
        p21 = (2.0 * (zy0dx - zxy00) + zy0dx - zxy10) / dx
        p30 = (-2.0 * z0dx + zx10 + zx00) / dxsq
        p31 = (-2.0 * zy0dx + zxy10 + zxy00) / dxsq
        return (z00 + v * zy00) + u * (
            (zx00 + v * zxy00) + u * ((p20 + v * p21) + u * (p30 + v * p31))
        )
    if 0 < iydi < nyd:
        iyd1 = iyd0 + 1
        dy = float(yd[iyd1] - y0)
        dysq = dy * dy
        z01 = float(zd[ixd0, iyd1])
        zx01, zy01, zxy01 = (float(val) for val in pdd[ixd0, iyd1])
        z0dy = (z01 - z00) / dy
        zx0dy = (zx01 - zx00) / dy
        p02 = (2.0 * (z0dy - zy00) + z0dy - zy01) / dy
        p03 = (-2.0 * z0dy + zy01 + zy00) / dysq
        p12 = (2.0 * (zx0dy - zxy00) + zx0dy - zxy01) / dy
        p13 = (-2.0 * zx0dy + zxy01 + zxy00) / dysq
        q0 = z00 + v * (zy00 + v * (p02 + v * p03))
        q1 = zx00 + v * (zxy00 + v * (p12 + v * p13))
        return q0 + u * q1
    return (z00 + v * zy00) + u * (zx00 + v * zxy00)


def interp_rgbi3p_db(
    xd: np.ndarray | jnp.ndarray,
    yd: np.ndarray | jnp.ndarray,
    zd: np.ndarray | jnp.ndarray,
    xi: np.ndarray | jnp.ndarray,
    yi: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates Akima (1996) bivariate interpolation at points `(xi, yi)`."""
    x_arr = np.asarray(xd, dtype=np.float64)
    y_arr = np.asarray(yd, dtype=np.float64)
    z_arr = np.asarray(zd, dtype=np.float64).reshape(x_arr.size, y_arr.size)
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    if xi_arr.shape != yi_arr.shape or xi_arr.size == 0:
        raise ValueError(
            "xi and yi must be non-empty arrays of identical shape."
        )
    pdd = np.asarray(rgpd3p_db(x_arr, y_arr, z_arr), dtype=np.float64)
    out = [
        _eval_akima_point(x_arr, y_arr, z_arr, pdd, float(xq), float(yq))
        for xq, yq in zip(xi_arr.ravel(), yi_arr.ravel(), strict=True)
    ]
    return jnp.asarray(out, dtype=jnp.float64).reshape(xi_arr.shape)


def interp_rgsf3p_db(
    xd: np.ndarray | jnp.ndarray,
    yd: np.ndarray | jnp.ndarray,
    zd: np.ndarray | jnp.ndarray,
    xi: np.ndarray | jnp.ndarray,
    yi: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates Akima (1996) surface fitting on target grid `(xi, yi)`."""
    x_arr = np.asarray(xd, dtype=np.float64)
    y_arr = np.asarray(yd, dtype=np.float64)
    z_arr = np.asarray(zd, dtype=np.float64).reshape(x_arr.size, y_arr.size)
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    if xi_arr.size == 0 or yi_arr.size == 0:
        raise ValueError("xi and yi must be non-empty 1D arrays.")
    pdd = np.asarray(rgpd3p_db(x_arr, y_arr, z_arr), dtype=np.float64)
    zi = np.zeros((xi_arr.size, yi_arr.size), dtype=np.float64)
    for ixi, xq in enumerate(xi_arr):
        for iyi, yq in enumerate(yi_arr):
            zi[ixi, iyi] = _eval_akima_point(
                x_arr, y_arr, z_arr, pdd, float(xq), float(yq)
            )
    return jnp.asarray(zi, dtype=jnp.float64)
