# SPDX-License-Identifier: LGPL-3.0-or-later
"""Huynh (1993) and Suresh & Huynh (1997) monotonicity-preserving splines.

References:
    Huynh, H. T. (1993), "Accurate Monotone Cubic Interpolation", SIAM Journal
    on Numerical Analysis, 30(1), 57-100.
    Suresh, A., & Huynh, H. T. (1997), "Accurate Monotonicity-Preserving
    Schemes with Runge-Kutta Time Stepping", Journal of Computational Physics,
    136, 83-99.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from flexmesa.interp_1d import pm


def _fortran_sign1(val: float) -> float:
    """Returns ``+1.0`` for ``val >= 0.0``, else ``-1.0``."""
    return 1.0 if val >= 0.0 else -1.0


def minmod(a: float, b: float) -> float:
    """Two-argument minmod limiter matching upstream ``minmod1``."""
    return 0.5 * (_fortran_sign1(a) + _fortran_sign1(b)) * min(abs(a), abs(b))


def minmod4(a: float, b: float, c: float, d: float) -> float:
    """Four-argument sequential minmod limiter matching ``minmod4_1``."""
    return minmod(minmod(minmod(a, b), c), d)


def median(a: float, b: float, c: float) -> float:
    """Median of three numbers via ``a + minmod(b - a, c - a)``."""
    return a + minmod(b - a, c - a)


def interp_m3(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    which: str = "quartic",
) -> jnp.ndarray:
    """Computes Huynh (1993) monotonicity-preserving cubic spline coefficients.

    For ``N <= 3``, delegates to :func:`flexmesa.interp_1d.pm.interp_pm`
    matching upstream MESA ``m3`` (`mk_pmlinear` / `mk_pmquad`).

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.
        which: Slope-limiter variant in ``{"average", "quartic", "super_bee"}``.

    Returns:
        JAX ``float64`` array of shape ``(N, 4)`` with ``[c0, c1, c2, c3]``
        at each knot.

    Raises:
        ValueError: If ``N < 2``, ``x`` is not strictly monotonic, or ``which``
            is unknown.
    """
    if which not in ("average", "quartic", "super_bee"):
        raise ValueError(
            f"Unknown m3 variant {which!r}; expected 'average', 'quartic', "
            "or 'super_bee'."
        )
    x_arr, y_arr = pm.validate_1d_grid(x, y, min_points=2)
    n = x_arr.size
    if n <= 3:
        return pm.interp_pm(x_arr, y_arr)

    h = x_arr[1:] - x_arr[:-1]
    coeffs = np.zeros((n, 4), dtype=np.float64)
    coeffs[:, 0] = y_arr

    s_mid = (y_arr[1:] - y_arr[:-1]) / h
    d = np.zeros(n, dtype=np.float64)
    d[1:-1] = (s_mid[1:] - s_mid[:-1]) / (x_arr[2:] - x_arr[:-2])
    d[0] = d[1]
    d[-1] = d[-2]

    d_mid = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        d0 = float(d[i])
        d1 = float(d[i + 1])
        d_mid[i] = minmod4(d1, d0, 4.0 * d1 - d0, 4.0 * d0 - d1)

    sp_l = np.zeros(n, dtype=np.float64)
    sp_r = np.zeros(n, dtype=np.float64)
    hd_mid = h * d_mid
    sp_l[1:] = s_mid + hd_mid
    sp_r[:-1] = s_mid - hd_mid

    s = np.zeros(n, dtype=np.float64)
    t = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        s[i] = minmod(float(s_mid[i - 1]), float(s_mid[i]))
        t[i] = minmod(float(sp_l[i]), float(sp_r[i]))

    if which == "average":
        for i in range(1, n - 1):
            bound = max(3.0 * abs(float(s[i])), 1.5 * abs(float(t[i])))
            avg = 0.5 * abs(float(sp_l[i] + sp_r[i]))
            coeffs[i, 1] = _fortran_sign1(float(t[i])) * min(avg, bound)
    elif which == "quartic":
        e_mid = np.zeros(n - 1, dtype=np.float64)
        e_mid[1:-1] = (d[2:-1] - d[1:-2]) / (x_arr[3:] - x_arr[:-3])
        for i in range(2, n - 2):
            numer = e_mid[i - 1] * (x_arr[i + 2] - x_arr[i]) + e_mid[i] * (
                x_arr[i] - x_arr[i - 2]
            )
            denom = x_arr[i + 2] - x_arr[i - 2]
            coeffs[i, 1] = s_mid[i] - h[i] * (d[i] + h[i - 1] * numer / denom)
        for idx in (1, n - 2):
            bound = max(3.0 * abs(float(s[idx])), 1.5 * abs(float(t[idx])))
            avg = 0.5 * abs(float(sp_l[idx] + sp_r[idx]))
            coeffs[idx, 1] = _fortran_sign1(float(t[idx])) * min(avg, bound)
        for i in range(1, n - 1):
            med = median(float(coeffs[i, 1]), float(sp_l[i]), float(sp_r[i]))
            tmax = _fortran_sign1(float(t[i])) * max(
                3.0 * abs(float(s[i])), 1.5 * abs(float(t[i]))
            )
            coeffs[i, 1] = minmod(med, tmax)
    else:
        for i in range(1, n - 1):
            bound = max(3.0 * abs(float(s[i])), 1.5 * abs(float(t[i])))
            max_sp = max(abs(float(sp_l[i])), abs(float(sp_r[i])))
            coeffs[i, 1] = _fortran_sign1(float(t[i])) * min(max_sp, bound)

    coeffs[0, 1] = minmod(float(s_mid[0]), float(s_mid[1]))
    coeffs[-1, 1] = minmod(float(s_mid[-2]), float(s_mid[-1]))

    coeffs[:-1, 2] = (3.0 * s_mid - 2.0 * coeffs[:-1, 1] - coeffs[1:, 1]) / h
    coeffs[:-1, 3] = (coeffs[:-1, 1] + coeffs[1:, 1] - 2.0 * s_mid) / (h * h)

    hn = float(h[-1])
    yn_prev = float(coeffs[-2, 0])
    yn = float(coeffs[-1, 0])
    s_prev = float(coeffs[-2, 1])
    s_end = float(coeffs[-1, 1])
    coeffs[-1, 2] = (3.0 * yn_prev - 3.0 * yn + (s_prev + 2.0 * s_end) * hn) / (
        hn * hn
    )
    coeffs[-1, 3] = (-2.0 * yn_prev + 2.0 * yn - (s_prev + s_end) * hn) / (
        hn * hn * hn
    )
    return jnp.asarray(coeffs, dtype=jnp.float64)


def interp_m3a(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Computes Huynh (1993) average-slope MP coefficients (`interp_m3a`)."""
    return interp_m3(x, y, which="average")


def interp_m3b(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Computes Huynh (1993) superbee-slope MP coefficients (`interp_m3b`)."""
    return interp_m3(x, y, which="super_bee")


def interp_m3q(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Computes Huynh (1993) quartic-slope MP coefficients (`interp_m3q`)."""
    return interp_m3(x, y, which="quartic")
