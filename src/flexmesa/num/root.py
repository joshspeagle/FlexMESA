# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bracketed and derivative-assisted root-finding with JAX implicit VJP.

Ports ``num/private/mod_root.f90`` (``do_look_for_brackets``,
``do_safe_root_with_brackets``, ``do_safe_root_with_guess``) into ``float64``
Python and provides :func:`safe_root_jax` with a custom JAX vector-Jacobian
product (``@jax.custom_vjp``) via the implicit function theorem:
$$\\frac{\\partial x^*}{\\partial p}
  = -\\left(\\frac{\\partial f}{\\partial x}(x^*, p)\\right)^{-1}
    \\frac{\\partial f}{\\partial p}(x^*, p).$$
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def look_for_brackets(
    x: float,
    dx: float,
    xmin: float,
    xmax: float,
    func: Callable[[float], tuple[float, float]],
    ntry: int = 50,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Searches outward from ``x`` in steps of $\\pm 2^k \\,\\mathrm{dx}$.

    Args:
        x: Initial center point clamped to ``[xmin, xmax]``.
        dx: Initial step increment ($\\ne 0$).
        xmin: Lower domain bound ($\\mathrm{xmin} < \\mathrm{xmax}$).
        xmax: Upper domain bound.
        func: Callable returning ``(f(x), df_dx(x))``.
        ntry: Maximum outward expansion steps.

    Returns:
        Nested tuple ``((x1, x3), (y1, y3))`` bracketing a sign change.

    Raises:
        ValueError: If ``dx == 0`` or ``xmin >= xmax``.
        RuntimeError: If no sign change is found within ``ntry`` expansions.
    """
    if (
        not (
            np.isfinite(x)
            and np.isfinite(dx)
            and np.isfinite(xmin)
            and np.isfinite(xmax)
        )
        or dx == 0.0
        or xmin >= xmax
    ):
        raise ValueError("require finite arguments, dx != 0, and xmin < xmax")
    x1 = min(max(float(x), float(xmin)), float(xmax))
    x3 = x1
    y1 = 0.0
    y3 = 0.0
    step0 = abs(float(dx))
    limit = 50 if ntry <= 0 else ntry

    for i in range(1, limit + 1):
        if x1 <= xmin and x3 >= xmax and i > 1:
            break
        jump = float(1 << min(i, 50)) * step0
        if i == 1:
            x1 = max(float(xmin), x1 - jump)
            x3 = min(float(xmax), x3 + jump)
            move_x1, move_x3 = True, True
        elif y1 > 0.0:
            if (y1 < y3 and x1 > xmin) or x3 >= xmax:
                x1 = max(float(xmin), x1 - jump)
                move_x1, move_x3 = True, False
            else:
                x3 = min(float(xmax), x3 + jump)
                move_x1, move_x3 = False, True
        elif (y1 > y3 and x1 > xmin) or x3 >= xmax:
            x1 = max(float(xmin), x1 - jump)
            move_x1, move_x3 = True, False
        else:
            x3 = min(float(xmax), x3 + jump)
            move_x1, move_x3 = False, True

        if move_x1:
            f1, _ = func(x1)
            if not np.isfinite(f1):
                raise ValueError("non-finite f(x) in look_for_brackets")
            y1 = float(f1)
        if move_x3:
            f3, _ = func(x3)
            if not np.isfinite(f3):
                raise ValueError("non-finite f(x) in look_for_brackets")
            y3 = float(f3)
        if y1 * y3 <= 0.0:
            return (x1, x3), (y1, y3)
    raise RuntimeError("failed to bracket root in look_for_brackets")


def safe_root_with_brackets(
    func: Callable[[float], tuple[float, float]],
    x1_in: float,
    x3_in: float,
    y1_in: float,
    y3_in: float,
    imax: int = 100,
    epsx: float = 1e-12,
    epsy: float = 1e-12,
) -> tuple[float, float]:
    """Finds a root $x \\in [x_1, x_3]$ via Brent (`mod_root.f90:42-44`).

    Args:
        func: Callable returning ``(f(x), df_dx(x))``.
        x1_in: First bracket endpoint.
        x3_in: Second bracket endpoint.
        y1_in: Function value $f(x_1)$.
        y3_in: Function value $f(x_3)$ (opposite sign to $f(x_1)$).
        imax: Maximum iterations allowed.
        epsx: Abscissa convergence tolerance.
        epsy: Residual convergence tolerance on $|f(x)|$.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.

    Raises:
        ValueError: If ``y1_in`` and ``y3_in`` have the same strict sign.
        RuntimeError: If iteration fails to converge.
    """
    from flexmesa.num import brent as _brent

    max_iter = max(int(imax), 200)
    return _brent.brent_safe_zero(
        float(x1_in),
        float(x3_in),
        1.0e-14,
        float(epsx),
        float(epsy),
        lambda xv: float(func(xv)[0]),
        df_a=float(y1_in),
        df_b=float(y3_in),
        max_iter=max_iter,
    )


def safe_root_with_guess(
    func: Callable[[float], tuple[float, float]],
    x_guess: float,
    dx: float,
    x1_opt: float | None = None,
    x3_opt: float | None = None,
    y1_opt: float | None = None,
    y3_opt: float | None = None,
    newt_imax: int = 10,
    imax: int = 100,
    xmin: float = -1e30,
    xmax: float = 1e30,
    epsx: float = 1e-12,
    epsy: float = 1e-12,
) -> tuple[float, float]:
    """Finds a root of $f(x) = 0$ from ``x_guess`` with optional brackets.

    Args:
        func: Callable returning ``(f(x), df_dx(x))``.
        x_guess: Initial root estimate.
        dx: Outward bracket search step if Newton steps do not converge.
        x1_opt: Optional lower bracket abscissa.
        x3_opt: Optional upper bracket abscissa.
        y1_opt: Optional function value at ``x1_opt``.
        y3_opt: Optional function value at ``x3_opt``.
        newt_imax: Maximum pure Newton steps before bracketing.
        imax: Maximum bracketed iterations.
        xmin: Lower domain bound.
        xmax: Upper domain bound.
        epsx: Abscissa tolerance.
        epsy: Residual tolerance.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.

    Raises:
        ValueError: If ``xmin >= xmax`` or inputs are non-finite.
        RuntimeError: If bracketing or iteration fails.
    """
    if xmin >= xmax or not (
        np.isfinite(x_guess) and np.isfinite(xmin) and np.isfinite(xmax)
    ):
        raise ValueError("require finite x_guess and xmin < xmax")
    x1 = x1_opt
    x3 = x3_opt
    y1 = y1_opt
    y3 = y3_opt
    x = min(max(float(x_guess), float(xmin)), float(xmax))
    absy_prev = 0.0
    for i in range(newt_imax):
        y, dydx = func(x)
        if not np.isfinite(y):
            break
        absy = abs(y)
        if absy == 0.0 or absy < max(epsy, abs(dydx) * epsx):
            return x, float(y)
        if dydx == 0.0 or not np.isfinite(dydx):
            break
        if i > 0 and absy > absy_prev:
            break
        absy_prev = absy
        if y < 0.0:
            x1 = x
            y1 = float(y)
        else:
            x3 = x
            y3 = float(y)
        x_next = x - y / dydx
        if x_next < xmin or x_next > xmax:
            break
        x = x_next

    if x1 is not None and x3 is not None:
        yv1 = float(func(float(x1))[0]) if y1 is None else float(y1)
        yv3 = float(func(float(x3))[0]) if y3 is None else float(y3)
        if (yv1 <= 0.0 <= yv3) or (yv3 <= 0.0 <= yv1):
            bx1, bx3 = float(x1), float(x3)
            by1, by3 = yv1, yv3
        else:
            (bx1, bx3), (by1, by3) = look_for_brackets(
                x_guess, dx, xmin, xmax, func, imax
            )
    else:
        (bx1, bx3), (by1, by3) = look_for_brackets(
            x_guess, dx, xmin, xmax, func, imax
        )
    return safe_root_with_brackets(func, bx1, bx3, by1, by3, imax, epsx, epsy)


def safe_root_without_brackets(
    func: Callable[[float], tuple[float, float]],
    x_guess: float,
    dx: float,
    newt_imax: int = 10,
    imax: int = 100,
    xmin: float = -1e30,
    xmax: float = 1e30,
    epsx: float = 1e-12,
    epsy: float = 1e-12,
) -> tuple[float, float]:
    """Finds a root of $f(x) = 0$ from ``x_guess`` without initial brackets.

    Args:
        func: Callable returning ``(f(x), df_dx(x))``.
        x_guess: Initial root estimate.
        dx: Outward bracket search step.
        newt_imax: Maximum pure Newton steps before bracketing.
        imax: Maximum bracketed iterations.
        xmin: Lower domain bound.
        xmax: Upper domain bound.
        epsx: Abscissa tolerance.
        epsy: Residual tolerance.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.

    Raises:
        ValueError: If ``xmin >= xmax`` or inputs are non-finite.
        RuntimeError: If bracketing or iteration fails.
    """
    return safe_root_with_guess(
        func,
        x_guess,
        dx,
        None,
        None,
        None,
        None,
        newt_imax,
        imax,
        xmin,
        xmax,
        epsx,
        epsy,
    )


def safe_root_jax(
    f_scalar: Callable[[jax.Array, jax.Array], jax.Array],
    x1: float,
    x3: float,
    params: jax.Array,
    imax: int = 100,
    epsx: float = 1e-13,
    epsy: float = 1e-13,
) -> jax.Array:
    """Solves $f(x^*(p), p) = 0$ on $[x_1, x_3]$ with implicit JAX VJP.

    Uses safeguarded Newton-Raphson + bisection for the primal solve and
    propagates reverse-mode cotangents via the implicit function theorem:
    $$\\frac{\\partial x^*}{\\partial p}
      = -\\left(\\frac{\\partial f}{\\partial x}(x^*, p)\\right)^{-1}
        \\frac{\\partial f}{\\partial p}(x^*, p).$$

    Args:
        f_scalar: Differentiable JAX callable ``f_scalar(x, params)`` returning
            a scalar ``float64`` residual.
        x1: Lower bracket endpoint.
        x3: Upper bracket endpoint.
        params: 1D JAX parameter array of shape ``(K,)`` in ``float64``.
        imax: Maximum root-finding iterations.
        epsx: Abscissa tolerance.
        epsy: Residual tolerance.

    Returns:
        Scalar ``float64`` JAX array $x^*(p)$ supporting ``jax.grad``.
    """

    @functools.partial(jax.custom_vjp)
    def _solve(p: jax.Array) -> jax.Array:
        df_dx_fn = jax.grad(f_scalar, argnums=0)

        def _cb(x_val: float) -> tuple[float, float]:
            xv = jnp.asarray(x_val, dtype=jnp.float64)
            fv = float(f_scalar(xv, p))
            dfv = float(df_dx_fn(xv, p))
            return fv, dfv

        y1, _ = _cb(x1)
        y3, _ = _cb(x3)
        x_star, _ = safe_root_with_brackets(
            _cb, x1, x3, y1, y3, imax=imax, epsx=epsx, epsy=epsy
        )
        return jnp.asarray(x_star, dtype=jnp.float64)

    def _solve_fwd(
        p: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        x_star = _solve(p)
        return x_star, (x_star, p)

    def _solve_bwd(
        res: tuple[jax.Array, jax.Array],
        g: jax.Array,
    ) -> tuple[jax.Array]:
        x_star, p = res
        df_dx = jax.grad(f_scalar, argnums=0)(x_star, p)
        df_dp = jax.grad(f_scalar, argnums=1)(x_star, p)
        return (-g * df_dp / df_dx,)

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve(jnp.asarray(params, dtype=jnp.float64))
