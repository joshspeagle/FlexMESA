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

import enum

import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.interp_1d import pm


class MpLimiter(enum.IntEnum):
    """Slope-limiter variant for Huynh (1993) MP cubic interpolation."""

    AVERAGE = 1
    QUARTIC = 2
    SUPER_BEE = 3


def _fortran_sign1(val: float) -> float:
    """Returns ``+1.0`` for ``val >= 0.0``, else ``-1.0``."""
    return 1.0 if val >= 0.0 else -1.0


def minmod(a: float, b: float) -> float:
    """Two-argument minmod limiter matching upstream ``minmod1``.

    Args:
        a: First scalar slope argument in physical or dimensionless units.
        b: Second scalar slope argument in physical or dimensionless units.

    Returns:
        Limited scalar ``0.5 * (sign(1, a) + sign(1, b)) * min(|a|, |b|)``.
    """
    return 0.5 * (_fortran_sign1(a) + _fortran_sign1(b)) * min(abs(a), abs(b))


def minmod4(a: float, b: float, c: float, d: float) -> float:
    """Four-argument sequential minmod limiter matching ``minmod4_1``.

    Args:
        a: First scalar argument.
        b: Second scalar argument.
        c: Third scalar argument.
        d: Fourth scalar argument.

    Returns:
        Sequentially limited value ``minmod(minmod(minmod(a, b), c), d)``.
    """
    return minmod(minmod(minmod(a, b), c), d)


def median(a: float, b: float, c: float) -> float:
    """Median of three numbers via ``a + minmod(b - a, c - a)``.

    Args:
        a: First scalar argument.
        b: Second scalar argument.
        c: Third scalar argument.

    Returns:
        Median of ``(a, b, c)``.
    """
    return a + minmod(b - a, c - a)


def _normalize_limiter(which: str | MpLimiter) -> MpLimiter:
    """Normalizes string or enum slope-limiter selector to `MpLimiter`."""
    if isinstance(which, MpLimiter):
        return which
    mapping = {
        "average": MpLimiter.AVERAGE,
        "quartic": MpLimiter.QUARTIC,
        "super_bee": MpLimiter.SUPER_BEE,
    }
    if which not in mapping:
        raise ValueError(
            f"Unknown m3 variant {which!r}; expected 'average', 'quartic', "
            "or 'super_bee'."
        )
    return mapping[which]


def interp_m3(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    which: str | MpLimiter = MpLimiter.QUARTIC,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Computes Huynh (1993) monotonicity-preserving cubic spline coefficients.

    For ``N <= 3``, delegates to :func:`flexmesa.interp_1d.pm.interp_pm`
    matching upstream MESA ``m3`` (`mk_pmlinear` / `mk_pmquad`).

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.
        which: Slope-limiter variant (`MpLimiter` or ``"average"``,
            ``"quartic"``, ``"super_bee"``).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of shape ``(N, 4)`` with ``[c0, c1, c2, c3]``
        at each knot.

    Raises:
        ValueError: If ``N < 2``, ``x`` is not strictly monotonic, or ``which``
            is unknown.
    """
    limiter = _normalize_limiter(which)
    x_arr, y_arr = pm.validate_1d_grid(x, y, min_points=2)
    n = x_arr.size
    if n <= 3:
        return pm.interp_pm(x_arr, y_arr, backend=backend)

    if _dispatch.resolve_backend(backend, x, y) == "rust":
        xc = _dispatch.as_c_f64(x_arr)
        yc = _dispatch.as_c_f64(y_arr)
        if limiter == MpLimiter.AVERAGE:
            return _rust.interp_m3a(xc, yc)
        if limiter == MpLimiter.QUARTIC:
            return _rust.interp_m3q(xc, yc)
        return _rust.interp_m3b(xc, yc)

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

    if limiter == MpLimiter.AVERAGE:
        for i in range(1, n - 1):
            bound = max(3.0 * abs(float(s[i])), 1.5 * abs(float(t[i])))
            avg = 0.5 * abs(float(sp_l[i] + sp_r[i]))
            coeffs[i, 1] = _fortran_sign1(float(t[i])) * min(avg, bound)
    elif limiter == MpLimiter.QUARTIC:
        e_mid = np.zeros(n - 1, dtype=np.float64)
        e_mid[1:-1] = (d[2:-1] - d[1:-2]) / (x_arr[3:] - x_arr[:-3])
        if n > 4:
            numer = e_mid[1:-2] * (x_arr[4:] - x_arr[2:-2]) + e_mid[2:-1] * (
                x_arr[2:-2] - x_arr[:-4]
            )
            denom = x_arr[4:] - x_arr[:-4]
            coeffs[2:-2, 1] = s_mid[2:-1] - h[2:-1] * (
                d[2:-2] + h[1:-2] * numer / denom
            )
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
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Computes Huynh (1993) average-slope MP coefficients (`interp_m3a`).

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of shape ``(N, 4)`` with ``[c0, c1, c2, c3]`` at
        each knot.

    Raises:
        ValueError: If ``x`` and ``y`` fail grid validation.
    """
    return interp_m3(x, y, which=MpLimiter.AVERAGE, backend=backend)


def interp_m3b(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Computes Huynh (1993) superbee-slope MP coefficients (`interp_m3b`).

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of shape ``(N, 4)`` with ``[c0, c1, c2, c3]`` at
        each knot.

    Raises:
        ValueError: If ``x`` and ``y`` fail grid validation.
    """
    return interp_m3(x, y, which=MpLimiter.SUPER_BEE, backend=backend)


def interp_m3q(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    *,
    backend: _dispatch.Backend = "auto",
) -> np.ndarray | jnp.ndarray:
    """Computes Huynh (1993) quartic-slope MP coefficients (`interp_m3q`).

    Args:
        x: Knot coordinates of shape ``(N,)`` (``N >= 2``), strictly monotonic,
            in physical or dimensionless units.
        y: Knot values of shape ``(N,)`` in physical or dimensionless units.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ``float64`` array of shape ``(N, 4)`` with ``[c0, c1, c2, c3]`` at
        each knot.

    Raises:
        ValueError: If ``x`` and ``y`` fail grid validation.
    """
    return interp_m3(x, y, which=MpLimiter.QUARTIC, backend=backend)
