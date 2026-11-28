# SPDX-License-Identifier: GPL-3.0-or-later
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

from flexmesa.num import brent as _brent


def look_for_brackets(
    x: float,
    dx: float,
    func: Callable[[float], tuple[float, float]],
    imax: int = 50,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Brackets a root outward from ``x`` matching ``do_look_for_brackets``.

    Mirrors ``num/private/mod_root.f90:168-256``: starts from ``x1 = x3 = x``
    and widens by ``jump = 2**i * dx`` for ``i = 1..imax`` (moving both ends
    at ``i = 1`` and then whichever endpoint value is closer to zero) until
    ``y1 * y3 <= 0``. No bounds, no finiteness checks, and signed ``dx``.

    Args:
        x: Initial center point (arbitrary units).
        dx: Initial step increment (used with its sign).
        func: Callable returning ``(f(x), df_dx(x))``.
        imax: Maximum outward expansion steps.

    Returns:
        Nested tuple ``((x1, x3), (y1, y3))`` bracketing a root.

    Raises:
        ValueError: If ``i >= 31`` where upstream's 32-bit integer ``2**i``
            overflows (``num/private/mod_root.f90:199``, ``#669``), or if no
            bracket is found within ``imax`` steps.
    """
    y1 = 0.0
    y3 = 0.0
    x1 = float(x)
    x3 = x1
    dx_f = float(dx)
    for i in range(1, imax + 1):
        # `jump = (2**i)*dx` (`mod_root.f90:199`): `2**i` overflows the
        # default 32-bit integer for `i >= 31` (#669, `DECISIONS #43`).
        if i >= 31:
            raise ValueError(
                "look_for_brackets reached i = 31, where upstream's integer "
                "2**i overflows (num/private/mod_root.f90:199, #669)"
            )
        jump = float(1 << i) * dx_f
        if i == 1:
            x1 -= jump
            x3 += jump
            move_x1 = True
            move_x3 = True
        elif y1 > 0.0:
            if y1 < y3:
                x1 -= jump
                move_x1 = True
                move_x3 = False
            else:
                x3 += jump
                move_x3 = True
                move_x1 = False
        elif y1 > y3:
            x1 -= jump
            move_x1 = True
            move_x3 = False
        else:
            x3 += jump
            move_x3 = True
            move_x1 = False
        if move_x1:
            y1 = float(func(x1)[0])
        if move_x3:
            y3 = float(func(x3)[0])
        if y1 * y3 <= 0.0:
            return (x1, x3), (y1, y3)
    raise ValueError("failed to bracket root in look_for_brackets")


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
    """Finds a root via ``eval_brent_safe_zero`` (``mod_root.f90:27-45``).

    Args:
        func: Callable returning ``(f(x), df_dx(x))``.
        x1_in: First bracket endpoint.
        x3_in: Second bracket endpoint.
        y1_in: Function value ``f(x1_in)``.
        y3_in: Function value ``f(x3_in)``.
        imax: Ignored, matching ``do_safe_root_with_brackets``
            (``num/private/mod_root.f90:42-43``).
        epsx: Abscissa convergence tolerance.
        epsy: Residual convergence tolerance on ``|f(x)|``.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.
    """
    # `do_safe_root_with_brackets` ignores `imax`
    # (`num/private/mod_root.f90:42-43`).
    _ = imax
    return _brent.eval_brent_safe_zero(
        float(x1_in),
        float(x3_in),
        1.0e-14,
        float(epsx),
        float(epsy),
        lambda xv: float(func(xv)[0]),
        float(y1_in),
        float(y3_in),
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
    epsx: float = 1e-12,
    epsy: float = 1e-12,
) -> tuple[float, float]:
    """Finds a root from ``x_guess`` matching ``do_safe_root_with_guess``.

    Mirrors ``num/private/mod_root.f90:47-164``: runs up to ``newt_imax``
    unbounded Newton steps from ``x_guess``, stopping only on
    ``abs(y) < max(epsy, abs(dydx) * epsx)`` (``:102``), exiting Newton on
    ``ValueError``, ``dydx == 0``, or increasing ``abs(y)`` (``:89-108``),
    then falling back to :func:`look_for_brackets` and delegating to
    :func:`safe_root_with_brackets`.

    Args:
        func: Callable returning ``(f(x), df_dx(x))``.
        x_guess: Initial root estimate.
        dx: Outward bracket search step if Newton steps do not bracket.
        x1_opt: Optional first bracket abscissa.
        x3_opt: Optional second bracket abscissa.
        y1_opt: Optional function value at ``x1_opt``.
        y3_opt: Optional function value at ``x3_opt``.
        newt_imax: Maximum pure Newton steps before bracketing.
        imax: Bracketing step budget passed to :func:`look_for_brackets`.
        epsx: Abscissa tolerance.
        epsy: Residual tolerance.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.

    Raises:
        ValueError: Propagated from ``func`` outside the Newton loop or from
            :func:`look_for_brackets`.
    """
    # `have_x1 = (x1 /= arg_not_provided)` etc. (`mod_root.f90:80-83`).
    have_x1 = x1_opt is not None
    have_x3 = x3_opt is not None
    have_y1 = y1_opt is not None
    have_y3 = y3_opt is not None
    x1 = 0.0 if x1_opt is None else float(x1_opt)
    x3 = 0.0 if x3_opt is None else float(x3_opt)
    y1 = 0.0 if y1_opt is None else float(y1_opt)
    y3 = 0.0 if y3_opt is None else float(y3_opt)
    x = float(x_guess)
    epsx_f = float(epsx)
    epsy_f = float(epsy)
    absy_prev = 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(1, newt_imax + 1):
            # `if (ierr /= 0) ... exit ! try safe_root` (`:89-96`).
            try:
                y_raw, dydx_raw = func(x)
            except ValueError:
                break
            y = float(y_raw)
            dydx = float(dydx_raw)
            absy = abs(y)
            if absy < float(np.fmax(epsy_f, abs(dydx) * epsx_f)):
                return x, y
            if dydx == 0.0:
                break
            if i > 1 and absy > absy_prev:
                break
            absy_prev = absy
            if y < 0.0:
                x1 = x
                y1 = y
                have_x1 = True
            else:
                x3 = x
                y3 = y
                have_x3 = True
            x = float(np.float64(x) - np.float64(y) / np.float64(dydx))

    if not (have_x1 and have_x3):
        (x1, x3), (y1, y3) = look_for_brackets(x_guess, dx, func, imax)
    else:
        if not have_y1:
            y1 = float(func(x1)[0])
        if not have_y3:
            y3 = float(func(x3)[0])

    return safe_root_with_brackets(func, x1, x3, y1, y3, imax, epsx, epsy)


def safe_root_without_brackets(
    func: Callable[[float], tuple[float, float]],
    x_guess: float,
    dx: float,
    newt_imax: int = 10,
    imax: int = 100,
    epsx: float = 1e-12,
    epsy: float = 1e-12,
) -> tuple[float, float]:
    """Finds a root from ``x_guess`` matching ``safe_root_without_brackets``.

    Mirrors ``num/public/num_safe_root.dek:120-144``: calls
    :func:`safe_root_with_guess` with no initial brackets and
    ``newt_imax = min(imax, newt_imax)`` (``:143``).

    Args:
        func: Callable returning ``(f(x), df_dx(x))``.
        x_guess: Initial root estimate.
        dx: Outward bracket search step.
        newt_imax: Maximum pure Newton steps before bracketing.
        imax: Bracketing step budget (also caps ``newt_imax``).
        epsx: Abscissa tolerance.
        epsy: Residual tolerance.

    Returns:
        Tuple ``(root_x, f_at_root)`` in ``float64``.

    Raises:
        ValueError: Propagated from :func:`safe_root_with_guess`.
    """
    return safe_root_with_guess(
        func,
        x_guess,
        dx,
        None,
        None,
        None,
        None,
        min(imax, newt_imax),
        imax,
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

    Uses safeguarded Brent root-finding (`brent_safe_zero`) for the primal solve
    and propagates reverse-mode cotangents via the implicit function theorem:
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
