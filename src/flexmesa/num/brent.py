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
    if a >= b or eps <= 0.0 or t < 0.0:
        raise ValueError("require a < b, eps > 0, t >= 0 in brent_local_min")
    c_gold = 0.5 * (3.0 - math.sqrt(5.0))
    sa = float(a)
    sb = float(b)
    x = sa + c_gold * (sb - sa)
    w = x
    v = w
    e = 0.0
    d = 0.0
    fx = float(func(x))
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

    Args:
        max_tries: Total function evaluation budget across sub-intervals.
        a: Lower search bound ($a < b$).
        b: Upper search bound ($b > a$).
        c: Initial candidate abscissa clamped to $[a, b]$.
        m: Upper bound on $|f''(x)|$ controlling sub-interval partitioning.
        e: Relative abscissa tolerance ($> 0$).
        t: Absolute abscissa tolerance ($\\ge 0$).
        func: Objective callable ``func(x)`` returning $f(x)$ in ``float64``.

    Returns:
        Tuple ``(x_min, f_min)`` in ``float64``.

    Raises:
        ValueError: If $a \\ge b$.
    """
    if a >= b:
        raise ValueError("require a < b in brent_global_min")
    best_x = min(max(float(c), float(a)), float(b))
    best_f = float(func(best_x))
    for endpoint in (float(a), float(b)):
        fe = float(func(endpoint))
        if fe < best_f:
            best_f = fe
            best_x = endpoint
    n_sub = 8 if m > 0.0 else 4
    step = (b - a) / float(n_sub)
    sub_budget = max(25, max_tries // n_sub)
    for idx in range(n_sub):
        sub_a = a + idx * step
        sub_b = b if (idx + 1 == n_sub) else (sub_a + step)
        xm, fm = brent_local_min(sub_budget, sub_a, sub_b, e, t, func)
        if fm < best_f:
            best_f = fm
            best_x = xm
    return best_x, best_f
