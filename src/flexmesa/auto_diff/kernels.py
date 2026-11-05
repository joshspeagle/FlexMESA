# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared JAX `float64` first- and second-derivative kernels for `auto_diff`.

Each unary kernel returns a 3-tuple `(f(x), f'(x), f''(x))` of `float64` JAX
arrays so that `DualOrder1` and `AutoDiffReal2VarOrder2` share identical
elementary derivative formulas.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa import math as fmath

type UnaryTriplet = tuple[jax.Array, jax.Array, jax.Array]
type UnaryKernel = Callable[[jax.Array], UnaryTriplet]

_INV_LN10: float = 1.0 / constants.LN10
_INV_LN2: float = 1.0 / float(jnp.log(jnp.asarray(2.0, dtype=jnp.float64)))
_INV_PI: float = 1.0 / constants.PI


def unary_sign(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\operatorname{sgn}(x)$."""
    z0 = jnp.zeros_like(val)
    return fmath.sgn(val), z0, z0


def unary_abs(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $|x|$."""
    return jnp.abs(val), fmath.sgn(val), jnp.zeros_like(val)


def unary_sqrt(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\sqrt{x}$."""
    q0 = jnp.sqrt(val)
    return q0, 0.5 / q0, -0.25 / (val * q0)


def unary_safe_sqrt(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\sqrt{x H(x)}$."""
    q0 = jnp.sqrt(val * fmath.heaviside(val))
    inv_x = fmath.powm1(val)
    return q0, 0.5 * q0 * inv_x, -0.25 * q0 * fmath.pow2(inv_x)


def unary_exp(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\exp(x)$."""
    q0 = fmath.exp(val)
    return q0, q0, q0


def unary_expm1(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $e^{x} - 1$."""
    q0 = fmath.exp(val)
    return fmath.expm1(val), q0, q0


def unary_exp10(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $10^{x}$."""
    q0 = fmath.exp10(val)
    d1 = q0 * constants.LN10
    return q0, d1, d1 * constants.LN10


def unary_powm1(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $x^{-1}$."""
    inv = fmath.powm1(val)
    inv2 = fmath.pow2(inv)
    return inv, -inv2, 2.0 * inv2 * inv


def unary_log(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\ln(x)$."""
    inv = fmath.powm1(val)
    return fmath.log(val), inv, -fmath.pow2(inv)


def unary_safe_log(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\operatorname{safe\\_log}(x)$."""
    inv = fmath.powm1(val)
    return fmath.safe_log(val), inv, -fmath.pow2(inv)


def unary_log1p(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\ln(1 + x)$."""
    inv = fmath.powm1(1.0 + val)
    return fmath.log1p(val), inv, -fmath.pow2(inv)


def unary_log10(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\log_{10}(x)$."""
    inv = fmath.powm1(val)
    return fmath.log10(val), _INV_LN10 * inv, -_INV_LN10 * fmath.pow2(inv)


def unary_safe_log10(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\operatorname{safe\\_log10}(x)$."""
    inv = fmath.powm1(val)
    return fmath.safe_log10(val), _INV_LN10 * inv, -_INV_LN10 * fmath.pow2(inv)


def unary_log2(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\log_{2}(x)$."""
    inv = fmath.powm1(val)
    return fmath.log2(val), _INV_LN2 * inv, -_INV_LN2 * fmath.pow2(inv)


def unary_sin(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\sin(x)$."""
    s = fmath.sin(val)
    c = fmath.cos(val)
    return s, c, -s


def unary_cos(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\cos(x)$."""
    s = fmath.sin(val)
    c = fmath.cos(val)
    return c, -s, -c


def unary_tan(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\tan(x)$."""
    t = fmath.tan(val)
    sec2 = fmath.pow2(t) + 1.0
    return t, sec2, 2.0 * t * sec2


def unary_sinpi(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\sin(\\pi x)$."""
    s = fmath.sinpi(val)
    c = fmath.cospi(val)
    return s, constants.PI * c, -(constants.PI * constants.PI) * s


def unary_cospi(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\cos(\\pi x)$."""
    s = fmath.sinpi(val)
    c = fmath.cospi(val)
    return c, -constants.PI * s, -(constants.PI * constants.PI) * c


def unary_tanpi(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\tan(\\pi x)$."""
    t = fmath.tanpi(val)
    sec2 = fmath.pow2(t) + 1.0
    return (
        t,
        constants.PI * sec2,
        2.0 * (constants.PI * constants.PI) * t * sec2,
    )


def unary_sinh(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\sinh(x)$."""
    sh = fmath.sinh(val)
    ch = fmath.cosh(val)
    return sh, ch, sh


def unary_cosh(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\cosh(x)$."""
    sh = fmath.sinh(val)
    ch = fmath.cosh(val)
    return ch, sh, ch


def unary_tanh(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\tanh(x)$."""
    th = fmath.tanh(val)
    sech2 = 1.0 - fmath.pow2(th)
    return th, sech2, -2.0 * th * sech2


def unary_asin(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\arcsin(x)$."""
    q1 = 1.0 - fmath.pow2(val)
    sq = jnp.sqrt(q1)
    return fmath.asin(val), 1.0 / sq, val / (q1 * sq)


def unary_acos(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\arccos(x)$."""
    q1 = 1.0 - fmath.pow2(val)
    sq = jnp.sqrt(q1)
    return fmath.acos(val), -1.0 / sq, -val / (q1 * sq)


def unary_atan(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\arctan(x)$."""
    q0 = 1.0 + fmath.pow2(val)
    return fmath.atan(val), 1.0 / q0, -2.0 * val / fmath.pow2(q0)


def unary_asinpi(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\arcsin(x) / \\pi$."""
    _, d1, d2 = unary_asin(val)
    return fmath.asinpi(val), d1 * _INV_PI, d2 * _INV_PI


def unary_acospi(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\arccos(x) / \\pi$."""
    _, d1, d2 = unary_acos(val)
    return fmath.acospi(val), d1 * _INV_PI, d2 * _INV_PI


def unary_atanpi(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\arctan(x) / \\pi$."""
    _, d1, d2 = unary_atan(val)
    return fmath.atanpi(val), d1 * _INV_PI, d2 * _INV_PI


def unary_asinh(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\operatorname{arsinh}(x)$."""
    q0 = fmath.pow2(val) + 1.0
    sq = jnp.sqrt(q0)
    return fmath.asinh(val), 1.0 / sq, -val / (q0 * sq)


def unary_acosh(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\operatorname{arcosh}(x)$."""
    q0 = fmath.pow2(val) - 1.0
    sq = jnp.sqrt(q0)
    return fmath.acosh(val), 1.0 / sq, -val / (q0 * sq)


def unary_atanh(val: jax.Array) -> UnaryTriplet:
    """Return `(f, f', f'')` for $\\operatorname{artanh}(x)$."""
    q0 = 1.0 - fmath.pow2(val)
    return fmath.atanh(val), 1.0 / q0, 2.0 * val / fmath.pow2(q0)


def unary_pown(val: jax.Array, degree: int) -> UnaryTriplet:
    """Return `(f, f', f'')` for $x^{2}$ through $x^{8}$."""
    if degree == 2:
        return fmath.pow2(val), 2.0 * val, jnp.full_like(val, 2.0)
    if degree == 3:
        return fmath.pow3(val), 3.0 * fmath.pow2(val), 6.0 * val
    if degree == 4:
        return fmath.pow4(val), 4.0 * fmath.pow3(val), 12.0 * fmath.pow2(val)
    if degree == 5:
        return fmath.pow5(val), 5.0 * fmath.pow4(val), 20.0 * fmath.pow3(val)
    if degree == 6:
        return fmath.pow6(val), 6.0 * fmath.pow5(val), 30.0 * fmath.pow4(val)
    if degree == 7:
        return fmath.pow7(val), 7.0 * fmath.pow6(val), 42.0 * fmath.pow5(val)
    if degree == 8:
        return fmath.pow8(val), 8.0 * fmath.pow7(val), 56.0 * fmath.pow6(val)
    return unary_pow_scalar(val, float(degree))


def unary_pow_scalar(
    val: jax.Array, exponent: fmath.intrinsics.ArrayLike
) -> UnaryTriplet:
    """Return `(f, f', f'')` for $x^{p}$ with constant exponent $p$."""
    pa = fmath.intrinsics.as_f64(exponent)
    res = fmath.pow(val, pa)
    d1 = pa * fmath.pow(val, pa - 1.0)
    d2 = pa * (pa - 1.0) * fmath.pow(val, pa - 2.0)
    return res, d1, d2


def unary_rpow_scalar(
    base: fmath.intrinsics.ArrayLike, exponent: jax.Array
) -> UnaryTriplet:
    """Return `(f, f', f'')` for $b^{x}$ with constant base $b > 0$."""
    ba = fmath.intrinsics.as_f64(base)
    res = fmath.pow(ba, exponent)
    lnb = fmath.log(ba)
    d1 = res * lnb
    return res, d1, d1 * lnb


def binary_pow(
    lhs: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return `(z, dz/dx, dz/dy, d2z/dx2, d2z/dxdy, d2z/dy2)` for $x^{y}$."""
    res = fmath.pow(lhs, rhs)
    lnx = fmath.log(lhs)
    xm1 = fmath.pow(lhs, rhs - 1.0)
    xm2 = fmath.pow(lhs, rhs - 2.0)
    d1x = rhs * xm1
    d1y = res * lnx
    d2x = rhs * (rhs - 1.0) * xm2
    d1x_d1y = xm1 * (1.0 + rhs * lnx)
    d2y = res * fmath.pow2(lnx)
    return res, d1x, d1y, d2x, d1x_d1y, d2y
