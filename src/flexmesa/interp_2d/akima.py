# SPDX-License-Identifier: LGPL-3.0-or-later
"""Akima (1996, ACM Algorithm 760) rectangular-grid bivariate interpolation.

Implements bivariate bicubic polynomial evaluation (`interp_rgbi3p_db`) and
rectangular surface fitting (`interp_rgsf3p_db`) from the estimated grid
partial derivatives $(z_x, z_y, z_{xy})$ produced by `rgpd3p_db`.

Reference:
    Akima, H. (1996), "Algorithm 760: Rectangular-Grid-Data Surface Fitting
    that Has the Accuracy of a Bicubic Polynomial", ACM Transactions on
    Mathematical Software, 22(3), 357-361.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa.interp_2d import akima_pd, spline1d

rgpd3p_db = akima_pd.rgpd3p_db


def _hermite_pair(
    secant: float,
    d0: float,
    d1: float,
    h: float,
    h_sq: float,
) -> tuple[float, float]:
    """Computes quadratic and cubic Hermite coefficients ``(p2, p3)``.

    Given cell width ``h``, squared width ``h_sq = h**2``, secant slope
    ``secant = (f1 - f0) / h``, and endpoint derivatives ``d0`` and ``d1``,
    returns ``(p2, p3)`` for $p(u) = f_0 + d_0 u + p_2 u^2 + p_3 u^3$.
    """
    return (
        (2.0 * (secant - d0) + secant - d1) / h,
        (-2.0 * secant + d1 + d0) / h_sq,
    )


def _eval_akima_point(
    xd: np.ndarray,
    yd: np.ndarray,
    zd: np.ndarray,
    pdd: np.ndarray,
    xii: float,
    yii: float,
) -> float:
    """Evaluates the Akima (1996) polynomial at a single point `(xii, yii)`.

    Handles interior cells (`0 < ixdi < nxd` and `0 < iydi < nyd`) via the
    16-coefficient bicubic tensor product and outside-grid queries via linear
    or cubic-linear extrapolation matching upstream `RGBI3P_db` / `RGSF3P_db`.
    """
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

    # Case 1: Query lies strictly inside an interior grid cell.
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

        p02, p03 = _hermite_pair(z0dy, zy00, zy01, dy, dysq)
        p12, p13 = _hermite_pair(zx0dy, zxy00, zxy01, dy, dysq)
        p20, p30 = _hermite_pair(z0dx, zx00, zx10, dx, dxsq)
        p21, p31 = _hermite_pair(zy0dx, zxy00, zxy10, dx, dxsq)
        p22 = (3.0 * (3.0 * a - b - c) + d) / (dx * dy)
        p23 = (-6.0 * a + 2.0 * b + 3.0 * c - d) / (dx * dysq)
        p32 = (-6.0 * a + 3.0 * b + 2.0 * c - d) / (dxsq * dy)
        p33 = (2.0 * (2.0 * a - b - c) + d) / (dxsq * dysq)

        q0 = z00 + v * (zy00 + v * (p02 + v * p03))
        q1 = zx00 + v * (zxy00 + v * (p12 + v * p13))
        q2 = p20 + v * (p21 + v * (p22 + v * p23))
        q3 = p30 + v * (p31 + v * (p32 + v * p33))
        return q0 + u * (q1 + u * (q2 + u * q3))

    # Case 2: Query is interior in x and extrapolated linearly in y.
    if 0 < ixdi < nxd:
        ixd1 = ixd0 + 1
        dx = float(xd[ixd1] - x0)
        dxsq = dx * dx
        z10 = float(zd[ixd1, iyd0])
        zx10, zy10, zxy10 = (float(val) for val in pdd[ixd1, iyd0])
        p20, p30 = _hermite_pair((z10 - z00) / dx, zx00, zx10, dx, dxsq)
        p21, p31 = _hermite_pair((zy10 - zy00) / dx, zxy00, zxy10, dx, dxsq)
        return (z00 + v * zy00) + u * (
            (zx00 + v * zxy00) + u * ((p20 + v * p21) + u * (p30 + v * p31))
        )

    # Case 3: Query is extrapolated linearly in x and interior in y.
    if 0 < iydi < nyd:
        iyd1 = iyd0 + 1
        dy = float(yd[iyd1] - y0)
        dysq = dy * dy
        z01 = float(zd[ixd0, iyd1])
        zx01, zy01, zxy01 = (float(val) for val in pdd[ixd0, iyd1])
        p02, p03 = _hermite_pair((z01 - z00) / dy, zy00, zy01, dy, dysq)
        p12, p13 = _hermite_pair((zx01 - zx00) / dy, zxy00, zxy01, dy, dysq)
        q0 = z00 + v * (zy00 + v * (p02 + v * p03))
        q1 = zx00 + v * (zxy00 + v * (p12 + v * p13))
        return q0 + u * q1

    # Case 4: Query lies in a corner quadrant outside both x and y ranges.
    return (z00 + v * zy00) + u * (zx00 + v * zxy00)


def interp_rgbi3p_db(
    xd: np.ndarray | jnp.ndarray,
    yd: np.ndarray | jnp.ndarray,
    zd: np.ndarray | jnp.ndarray,
    xi: np.ndarray | jnp.ndarray,
    yi: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluates Akima (1996) bivariate interpolation at points `(xi, yi)`.

    Args:
        xd: Strictly ascending x-grid of shape ``(nxd,)`` (``nxd >= 2``).
        yd: Strictly ascending y-grid of shape ``(nyd,)`` (``nyd >= 2``).
        zd: Grid values of shape ``(nxd, nyd)`` or flat ``(nxd * nyd,)``.
        xi: Query x-coordinates of shape ``(*batch,)``.
        yi: Query y-coordinates of shape ``(*batch,)`` matching ``xi``.

    Returns:
        JAX ``float64`` array of shape ``(*batch,)`` containing interpolated
        values.

    Raises:
        ValueError: If grid or query arrays have mismatched shapes, fewer than
            2 grid points per axis, non-ascending knot coordinates, or
            non-finite values.
    """
    x_arr = np.asarray(xd, dtype=np.float64)
    y_arr = np.asarray(yd, dtype=np.float64)
    z_arr = spline1d.reshape_grid_2d(zd, x_arr.size, y_arr.size, "zd")
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    if xi_arr.shape != yi_arr.shape or xi_arr.size == 0:
        raise ValueError(
            "xi and yi must be non-empty arrays of identical shape."
        )
    if not (np.all(np.isfinite(xi_arr)) and np.all(np.isfinite(yi_arr))):
        raise ValueError("xi and yi must contain only finite float64 values.")
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
    """Evaluates Akima (1996) surface fitting on target grid `(xi, yi)`.

    Args:
        xd: Strictly ascending x-grid of shape ``(nxd,)`` (``nxd >= 2``).
        yd: Strictly ascending y-grid of shape ``(nyd,)`` (``nyd >= 2``).
        zd: Grid values of shape ``(nxd, nyd)`` or flat ``(nxd * nyd,)``.
        xi: 1D target x-grid of shape ``(nxi,)``.
        yi: 1D target y-grid of shape ``(nyi,)``.

    Returns:
        JAX ``float64`` array of shape ``(nxi, nyi)`` containing surface values.

    Raises:
        ValueError: If grid or target arrays fail validation or contain
            non-finite values.
    """
    x_arr = np.asarray(xd, dtype=np.float64)
    y_arr = np.asarray(yd, dtype=np.float64)
    z_arr = spline1d.reshape_grid_2d(zd, x_arr.size, y_arr.size, "zd")
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    if (
        xi_arr.ndim != 1
        or yi_arr.ndim != 1
        or xi_arr.size == 0
        or yi_arr.size == 0
    ):
        raise ValueError("xi and yi must be non-empty 1D arrays.")
    if not (np.all(np.isfinite(xi_arr)) and np.all(np.isfinite(yi_arr))):
        raise ValueError("xi and yi must contain only finite float64 values.")
    pdd = np.asarray(rgpd3p_db(x_arr, y_arr, z_arr), dtype=np.float64)
    zi = np.zeros((xi_arr.size, yi_arr.size), dtype=np.float64)
    for ixi, xq in enumerate(xi_arr):
        for iyi, yq in enumerate(yi_arr):
            zi[ixi, iyi] = _eval_akima_point(
                x_arr, y_arr, z_arr, pdd, float(xq), float(yq)
            )
    return jnp.asarray(zi, dtype=jnp.float64)
