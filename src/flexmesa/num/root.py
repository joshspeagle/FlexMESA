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
    if dx == 0.0 or xmin >= xmax:
        raise ValueError("require dx != 0 and xmin < xmax")
    x1 = min(max(float(x), float(xmin)), float(xmax))
    x3 = x1
    y0, _ = func(x1)
    if y0 == 0.0:
        return (x1, x1), (0.0, 0.0)
    y1 = float(y0)
    y3 = float(y0)
    step = abs(float(dx))
    limit = 50 if ntry <= 0 else ntry

    for _ in range(limit):
        can_left = x1 > xmin
        can_right = x3 < xmax
        if not can_left and not can_right:
            break
        if can_left:
            x_new = max(float(xmin), x1 - step)
            y_new, _ = func(x_new)
            if y_new == 0.0 or (y_new > 0.0) != (y1 > 0.0):
                return (x_new, x1), (float(y_new), y1)
            x1, y1 = x_new, float(y_new)
        if can_right:
            x_new = min(float(xmax), x3 + step)
            y_new, _ = func(x_new)
            if y_new == 0.0 or (y_new > 0.0) != (y3 > 0.0):
                return (x3, x_new), (y3, float(y_new))
            x3, y3 = x_new, float(y_new)
        step *= 2.0
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
    """Finds a root $x \\in [x_1, x_3]$ via safeguarded Newton + bisection.

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
        RuntimeError: If ``imax`` is exceeded.
    """
    if y1_in == 0.0:
        return float(x1_in), 0.0
    if y3_in == 0.0:
        return float(x3_in), 0.0
    if (y1_in > 0.0 and y3_in > 0.0) or (y1_in < 0.0 and y3_in < 0.0):
        raise ValueError("y1_in and y3_in must have opposite signs")
    if y1_in < 0.0:
        xl, xh = float(x1_in), float(x3_in)
    else:
        xl, xh = float(x3_in), float(x1_in)

    rts = 0.5 * (x1_in + x3_in)
    dxold = abs(x3_in - x1_in)
    dx = dxold
    f, df = func(rts)
    limit = 100 if imax <= 0 else imax

    for _ in range(limit):
        if abs(f) <= epsy or abs(dx) <= epsx or abs(xh - xl) <= epsx:
            return float(rts), float(f)
        out_of_bracket = ((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0
        slow_decrease = abs(2.0 * f) > abs(dxold * df)
        if df == 0.0 or not np.isfinite(df) or out_of_bracket or slow_decrease:
            dxold = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return float(rts), float(f)
        else:
            dxold = dx
            dx = f / df
            temp = rts
            rts -= dx
            if temp == rts:
                return float(rts), float(f)
        if abs(dx) <= epsx:
            f_end, _ = func(rts)
            return float(rts), float(f_end)
        f, df = func(rts)
        if f < 0.0:
            xl = rts
        else:
            xh = rts

    raise RuntimeError("safe_root_with_brackets exceeded imax")


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
    """
    x = min(max(float(x_guess), float(xmin)), float(xmax))
    for _ in range(newt_imax):
        fx, dfx = func(x)
        if abs(fx) <= epsy:
            return x, float(fx)
        if dfx == 0.0 or not np.isfinite(dfx):
            break
        step = fx / dfx
        x_next = x - step
        if x_next < xmin or x_next > xmax:
            break
        x = x_next
        if abs(step) <= epsx:
            fx_end, _ = func(x)
            if abs(fx_end) <= max(epsy, 1e-12):
                return x, float(fx_end)
            break

    if (
        x1_opt is not None
        and x3_opt is not None
        and y1_opt is not None
        and y3_opt is not None
        and ((y1_opt <= 0.0 <= y3_opt) or (y3_opt <= 0.0 <= y1_opt))
    ):
        bx1, bx3 = float(x1_opt), float(x3_opt)
        by1, by3 = float(y1_opt), float(y3_opt)
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
