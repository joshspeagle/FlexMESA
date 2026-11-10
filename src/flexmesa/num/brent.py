# SPDX-License-Identifier: LGPL-3.0-or-later
"""Brent (1973) root-finding and univariate local/global minimization (``num``).

Ports ``num/private/mod_brent.f90`` (``eval_brent_safe_zero``,
``eval_brent_local_min``, and ``eval_brent_global_min``) into ``float64``
Python.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def brent_safe_zero(
    a: float,
    b: float,
    eps: float,
    t: float,
    epsy: float,
    func: Callable[[float], float],
    df_a: float | None = None,
    df_b: float | None = None,
    max_iter: int = 200,
) -> tuple[float, float]:
    """Finds a root $x \\in [a, b]$ of $f(x) = 0$ via Brent's (1973) method.

    Combines bisection, linear secant interpolation, and inverse quadratic
    interpolation so that the bracket contracts superlinearly on smooth
    functions while never converging slower than bisection.

    Args:
        a: Lower or first bracket endpoint.
        b: Upper or second bracket endpoint.
        eps: Relative abscissa tolerance ($\\ge 0$).
        t: Absolute abscissa tolerance ($\\ge 0$).
        epsy: Residual stopping tolerance on $|f(x)|$.
        func: Scalar callable ``func(x)`` returning $f(x)$ in ``float64``.
        df_a: Optional pre-evaluated $f(a)$.
        df_b: Optional pre-evaluated $f(b)$.
        max_iter: Maximum Brent iterations allowed.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.

    Raises:
        ValueError: If $f(a)$ and $f(b)$ have the same strict sign.
        RuntimeError: If ``max_iter`` is exceeded.
    """
    sa = float(a)
    sb = float(b)
    fa = float(func(sa) if df_a is None else df_a)
    fb = float(func(sb) if df_b is None else df_b)
    if not (
        math.isfinite(sa)
        and math.isfinite(sb)
        and math.isfinite(fa)
        and math.isfinite(fb)
    ):
        raise ValueError("non-finite endpoint or value in brent_safe_zero")
    if fa == 0.0:
        return sa, 0.0
    if fb == 0.0:
        return sb, 0.0
    if (fa > 0.0 and fb > 0.0) or (fa < 0.0 and fb < 0.0):
        raise ValueError("f(a) and f(b) must have opposite signs")

    c = sa
    fc = fa
    e = sb - sa
    d = e
    limit = 200 if max_iter <= 0 else max_iter

    for _ in range(limit):
        if (fc > 0.0 and fb > 0.0) or (fc < 0.0 and fb < 0.0):
            c = sa
            fc = fa
            e = sb - sa
            d = e
        if abs(fc) < abs(fb):
            sa = sb
            sb = c
            c = sa
            fa = fb
            fb = fc
            fc = fa
        tol = 2.0 * eps * abs(sb) + t
        m = 0.5 * (c - sb)
        if abs(m) <= tol or fb == 0.0 or (epsy > 0.0 and abs(fb) <= epsy):
            return sb, fb
        if abs(e) < tol or abs(fa) <= abs(fb):
            e = m
            d = e
        else:
            s = fb / fa
            if sa == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q0 = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q0 * (q0 - r) - (sb - sa) * (r - 1.0))
                q = (q0 - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            s_old = e
            e = d
            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(
                0.5 * s_old * q
            ):
                d = p / q
            else:
                e = m
                d = e
        sa = sb
        fa = fb
        if abs(d) > tol:
            sb += d
        elif m > 0.0:
            sb += tol
        else:
            sb -= tol
        fb = float(func(sb))
        if not math.isfinite(fb):
            raise ValueError("non-finite f(x) in brent_safe_zero")

    raise RuntimeError("brent_safe_zero exceeded max_iter")


def brent_local_min(
    max_tries: int,
    a: float,
    b: float,
    eps: float,
    t: float,
    func: Callable[[float], float],
) -> tuple[float, float]:
    """Finds a local minimum $(x_{\\min}, f(x_{\\min}))$ of $f$ on $[a, b]$.

    Combines golden-section search with successive parabolic interpolation
    (``eval_brent_local_min`` in ``mod_brent.f90``).

    Args:
        max_tries: Maximum function evaluations allowed.
        a: Lower search bound ($a < b$).
        b: Upper search bound ($b > a$).
        eps: Relative abscissa tolerance ($> 0$).
        t: Absolute abscissa tolerance ($\\ge 0$).
        func: Objective callable ``func(x)`` returning $f(x)$ in ``float64``.

    Returns:
        Tuple ``(x_min, f_min)`` in ``float64``.

    Raises:
        ValueError: If $a \\ge b$, $\\epsilon \\le 0$, or $t < 0$.
    """
    if (
        not (math.isfinite(a) and math.isfinite(b))
        or a >= b
        or eps <= 0.0
        or t < 0.0
    ):
        raise ValueError(
            "require finite a < b, eps > 0, t >= 0 in brent_local_min"
        )
    c_gold = 0.5 * (3.0 - math.sqrt(5.0))
    sa = float(a)
    sb = float(b)
    x = sa + c_gold * (sb - sa)
    w = x
    v = w
    e = 0.0
    d = 0.0
    fx = float(func(x))
    if not math.isfinite(fx):
        raise ValueError("non-finite objective value in brent_local_min")
    fw = fx
    fv = fx
    limit = 200 if max_tries <= 0 else max_tries

    for _ in range(limit):
        m = 0.5 * (sa + sb)
        tol = eps * abs(x) + t
        t2 = 2.0 * tol
        if abs(x - m) <= t2 - 0.5 * (sb - sa):
            return x, fx
        r = 0.0
        q = 0.0
        p = 0.0
        if abs(e) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) < abs(0.5 * q * r) and p > q * (sa - x) and p < q * (sb - x):
            d = p / q
            u = x + d
            if (u - sa) < t2 or (sb - u) < t2:
                d = tol if x < m else -tol
        else:
            e = (sb - x) if x < m else (sa - x)
            d = c_gold * e
        if abs(d) >= tol:
            u = x + d
        elif d > 0.0:
            u = x + tol
        else:
            u = x - tol
        fu = float(func(u))
        if not math.isfinite(fu):
            raise ValueError("non-finite objective value in brent_local_min")
        if fu <= fx:
            if u < x:
                sb = x
            else:
                sa = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                sa = u
            else:
                sb = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


def local_min(
    max_tries: int,
    a: float,
    b: float,
    eps: float,
    t: float,
    func: Callable[[float], float],
) -> tuple[float, float]:
    """Alias for :func:`brent_local_min` matching upstream ``local_min``.

    Args:
        max_tries: Maximum function evaluations allowed.
        a: Lower search bound ($a < b$).
        b: Upper search bound ($b > a$).
        eps: Relative abscissa tolerance ($> 0$).
        t: Absolute abscissa tolerance ($\\ge 0$).
        func: Objective callable ``func(x)`` returning $f(x)$ in ``float64``.

    Returns:
        Tuple ``(x_min, f_min)`` in ``float64``.
    """
    return brent_local_min(max_tries, a, b, eps, t, func)


def brent_global_min(
    max_tries: int,
    a: float,
    b: float,
    c: float,
    m: float,
    e: float,
    t: float,
    func: Callable[[float], float],
) -> tuple[float, float]:
    """Seeks the global minimum $(x_{\\min}, f(x_{\\min}))$ of $f$ on $[a, b]$.

    Implements Brent's (1973) global minimization algorithm
    (``eval_global_min`` / ``glomin`` in ``mod_brent.f90``) using parabolic
    under-estimators with second-derivative bound $m$.

    Args:
        max_tries: Maximum outer interval-advance iterations.
        a: Lower search bound ($a < b$).
        b: Upper search bound ($b > a$).
        c: Initial candidate abscissa in $[a, b]$.
        m: Upper bound on $f''(x)$ on $[a, b]$.
        e: Positive bound on absolute evaluation error in $f(x)$.
        t: Positive absolute abscissa tolerance.
        func: Objective callable ``func(x)`` returning $f(x)$ in ``float64``.

    Returns:
        Tuple ``(x_min, f_min)`` in ``float64``.

    Raises:
        ValueError: If $a \\ge b$ or inputs/outputs are non-finite.
    """
    if (
        not (
            math.isfinite(a)
            and math.isfinite(b)
            and math.isfinite(c)
            and math.isfinite(m)
            and math.isfinite(e)
            and math.isfinite(t)
        )
        or a >= b
    ):
        raise ValueError("require finite a < b and c in brent_global_min")
    machep = 2.220446049250313e-16
    a0 = float(b)
    x = a0
    a2 = float(a)
    y0 = float(func(a0))
    yb = y0
    y2 = float(func(a2))
    y = y2
    if not (math.isfinite(y0) and math.isfinite(y2)):
        raise ValueError("non-finite endpoint value in brent_global_min")
    if y0 < y:
        y = y0
    else:
        x = a2
    if m <= 0.0 or b <= a:
        return x, y

    m2 = 0.5 * (1.0 + 16.0 * machep) * float(m)
    sc = 0.5 * (a2 + a0) if (c <= a2 or a0 <= c) else float(c)
    y1 = float(func(sc))
    if not math.isfinite(y1):
        raise ValueError("non-finite interior value in brent_global_min")
    k = 3
    d0 = a2 - sc
    h = 9.0 / 11.0
    if y1 < y:
        x = sc
        y = y1

    limit = 100_000 if max_tries <= 0 else int(max_tries)
    for _ in range(limit):
        d1 = a2 - a0
        d2 = sc - a0
        z2 = a0_end = float(b) - a2
        z0 = y2 - y1
        z1 = y2 - y0
        r = d1 * d1 * z0 - d0 * d0 * z1
        p = r
        qs = 2.0 * (d0 * z1 - d1 * z0)
        q = qs

        if k < 1_000_000 or y2 <= y:
            while True:
                if q * (r * (yb - y2) + z2 * q * ((y2 - y) + t)) < (
                    z2 * m2 * r * (z2 * q - r)
                ):
                    a3 = a2 + r / q
                    y3 = float(func(a3))
                    if not math.isfinite(y3):
                        raise ValueError("non-finite f(x) in brent_global_min")
                    if y3 < y:
                        x = a3
                        y = y3
                k = (1611 * k) % 1_048_576
                q = 1.0
                r = (float(b) - float(a)) * 0.00001 * float(k)
                if a0_end <= r:
                    break
        else:
            k = (1611 * k) % 1_048_576
            q = 1.0
            r = (float(b) - float(a)) * 0.00001 * float(k)
            while r < z2:
                if q * (r * (yb - y2) + z2 * q * ((y2 - y) + t)) < (
                    z2 * m2 * r * (z2 * q - r)
                ):
                    a3 = a2 + r / q
                    y3 = float(func(a3))
                    if not math.isfinite(y3):
                        raise ValueError("non-finite f(x) in brent_global_min")
                    if y3 < y:
                        x = a3
                        y = y3
                k = (1611 * k) % 1_048_576
                q = 1.0
                r = (float(b) - float(a)) * 0.00001 * float(k)

        r = m2 * d0 * d1 * d2
        s = math.sqrt(((y2 - y) + t) / m2)
        h = 0.5 * (1.0 + h)
        p = h * (p + 2.0 * r * s)
        q = q + 0.5 * qs
        r = -0.5 * (d0 + (z0 + 2.01 * e) / (d0 * m2))
        r = (a2 + s) if (r < s or d0 < 0.0) else (a2 + r)
        a3 = (a2 + p / q) if (p * q > 0.0) else r

        while True:
            a3 = max(a3, r)
            if float(b) <= a3:
                a3 = float(b)
                y3 = yb
            else:
                y3 = float(func(a3))
                if not math.isfinite(y3):
                    raise ValueError("non-finite f(x) in brent_global_min")
            if y3 < y:
                x = a3
                y = y3
            d0 = a3 - a2
            if a3 <= r:
                break
            p = 2.0 * (y2 - y3) / (float(m) * d0)
            if (1.0 + 9.0 * machep) * d0 <= abs(p):
                break
            if 0.5 * m2 * (d0 * d0 + p * p) <= (y2 - y) + (y3 - y) + 2.0 * t:
                break
            a3 = 0.5 * (a2 + a3)
            h = 0.9 * h

        if float(b) <= a3:
            break
        a0 = sc
        sc = a2
        a2 = a3
        y0 = y1
        y1 = y2
        y2 = y3

    return x, y


def glomin(
    max_tries: int,
    a: float,
    b: float,
    c: float,
    m: float,
    e: float,
    t: float,
    func: Callable[[float], float],
) -> tuple[float, float]:
    """Alias for :func:`brent_global_min` matching Burkardt/MESA ``glomin``.

    Args:
        max_tries: Maximum outer interval-advance iterations.
        a: Lower search bound ($a < b$).
        b: Upper search bound ($b > a$).
        c: Initial candidate abscissa in $[a, b]$.
        m: Upper bound on $f''(x)$ on $[a, b]$.
        e: Positive bound on absolute evaluation error in $f(x)$.
        t: Positive absolute abscissa tolerance.
        func: Objective callable ``func(x)`` returning $f(x)$ in ``float64``.

    Returns:
        Tuple ``(x_min, f_min)`` in ``float64``.
    """
    return brent_global_min(max_tries, a, b, c, m, e, t, func)
