# SPDX-License-Identifier: LGPL-3.0-or-later
"""Ridders differentiation and adaptive Simpson 3/8 quadrature (``num``).

Ports ``num/private/mod_num_lib.f90`` numerical differentiation (``dfridr``)
and adaptive recursive Simpson 3/8 quadrature with Richardson extrapolation
(``integrate``) into ``float64`` Python.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def dfridr(hx: float, func: Callable[[float], float]) -> tuple[float, float]:
    """Estimates ``f'(0)`` and its error via Ridders polynomial extrapolation.

    Evaluates symmetric finite-difference quotients at geometrically decreasing
    step sizes $h_i = h_0 / 1.4^i$ ($0 \\le i < 10$) and builds a Neville
    extrapolation tableau to eliminate leading truncation error orders.

    Args:
        hx: Initial non-zero step size (same units as the argument of ``func``).
        func: Scalar callable ``func(dx)`` evaluated at symmetric offsets.

    Returns:
        Tuple ``(derivative, error_estimate)`` in ``float64``.

    Raises:
        ValueError: If ``hx == 0.0`` or ``hx`` is non-finite.
    """
    if hx == 0.0 or not np.isfinite(hx):
        raise ValueError("hx must be finite and non-zero in dfridr")
    ntab = 10
    con = 1.4
    con2 = con * con
    safe = 2.0
    a = np.zeros((ntab, ntab), dtype=np.float64)
    hh = float(hx)
    a[0, 0] = (func(hh) - func(-hh)) / (2.0 * hh)
    err = float("inf")
    ans = float(a[0, 0])
    for i in range(1, ntab):
        hh /= con
        a[0, i] = (func(hh) - func(-hh)) / (2.0 * hh)
        fac = con2
        for j in range(1, i + 1):
            a[j, i] = (a[j - 1, i] * fac - a[j - 1, i - 1]) / (fac - 1.0)
            fac *= con2
            errt = max(
                abs(float(a[j, i] - a[j - 1, i])),
                abs(float(a[j, i] - a[j - 1, i - 1])),
            )
            if errt <= err:
                err = errt
                ans = float(a[j, i])
        if abs(float(a[i, i] - a[i - 1, i - 1])) >= safe * err:
            break
    return ans, err


def integrate(
    func: Callable[[float], float],
    min_x: float,
    max_x: float,
    eps: float = 1e-8,
    max_steps: int = 20_000,
) -> tuple[float, int]:
    """Evaluates definite integral via adaptive Simpson 3/8 quadrature.

    Subdivides $[\\min_x, \\max_x]$ into three equal subintervals per recursion
    level and applies a 4th-order Richardson correction $(S_f - S_c) / 15$
    once $|S_f - S_c| \\le \\epsilon$.

    Args:
        func: Scalar integrand callable ``func(x)``.
        min_x: Lower integration limit.
        max_x: Upper integration limit.
        eps: Local truncation error tolerance (``> 0``).
        max_steps: Maximum integrand evaluations allowed.

    Returns:
        Tuple ``(integral_value, num_evaluations)``.

    Raises:
        ValueError: If limits are non-finite or ``eps <= 0``.
        RuntimeError: If ``max_steps`` is exceeded.
    """
    if not (np.isfinite(min_x) and np.isfinite(max_x)) or eps <= 0.0:
        raise ValueError("bounds must be finite and eps > 0 in integrate")
    if min_x == max_x:
        return 0.0, 0
    ncalls = 4
    fa = float(func(min_x))
    fb = float(func(max_x))
    h = max_x - min_x
    f1 = float(func(min_x + h / 3.0))
    f2 = float(func(min_x + 2.0 * h / 3.0))
    s_whole = (h / 8.0) * (fa + 3.0 * f1 + 3.0 * f2 + fb)

    def _rec(
        a: float,
        b: float,
        va: float,
        v1: float,
        v2: float,
        vb: float,
        s_prev: float,
        tol: float,
        depth: int,
    ) -> float:
        nonlocal ncalls
        span = b - a
        m1 = a + span / 3.0
        m2 = a + 2.0 * span / 3.0
        fa1 = float(func(a + span / 9.0))
        fa2 = float(func(a + 2.0 * span / 9.0))
        fb1 = float(func(m1 + span / 9.0))
        fb2 = float(func(m1 + 2.0 * span / 9.0))
        fc1 = float(func(m2 + span / 9.0))
        fc2 = float(func(m2 + 2.0 * span / 9.0))
        ncalls += 6
        s1 = ((m1 - a) / 8.0) * (va + 3.0 * fa1 + 3.0 * fa2 + v1)
        s2 = ((m2 - m1) / 8.0) * (v1 + 3.0 * fb1 + 3.0 * fb2 + v2)
        s3 = ((b - m2) / 8.0) * (v2 + 3.0 * fc1 + 3.0 * fc2 + vb)
        s_new = s1 + s2 + s3
        if (depth >= 2 and abs(s_new - s_prev) <= tol) or depth >= 28:
            return float(s_new + (s_new - s_prev) / 15.0)
        if max_steps > 0 and ncalls > max_steps:
            raise RuntimeError("max_steps exceeded in integrate")
        t3 = tol / 3.0
        return (
            _rec(a, m1, va, fa1, fa2, v1, s1, t3, depth + 1)
            + _rec(m1, m2, v1, fb1, fb2, v2, s2, t3, depth + 1)
            + _rec(m2, b, v2, fc1, fc2, vb, s3, t3, depth + 1)
        )

    val = _rec(min_x, max_x, fa, f1, f2, fb, s_whole, eps, 0)
    return float(val), ncalls
