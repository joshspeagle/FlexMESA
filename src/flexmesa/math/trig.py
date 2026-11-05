# SPDX-License-Identifier: LGPL-3.0-or-later
"""Trigonometric, half-turn (`pi`-scaled), hyperbolic, sign, and string helpers.

Provides radian and half-turn trigonometric functions, hyperbolic functions,
MESA's symmetric `heaviside` and `sgn` step functions, and Fortran string
parsers (`str_to_double`, `str_to_vector`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.math import intrinsics


def cos(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\cos(x)$ for $x$ in radians."""
    return jnp.cos(intrinsics.as_f64(val))


def sin(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\sin(x)$ for $x$ in radians."""
    return jnp.sin(intrinsics.as_f64(val))


def tan(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\tan(x)$ for $x$ in radians."""
    return jnp.tan(intrinsics.as_f64(val))


def cospi(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\cos(\\pi x)$ for half-turn argument $x$."""
    return jnp.cos(intrinsics.as_f64(val) * constants.PI)


def sinpi(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\sin(\\pi x)$ for half-turn argument $x$."""
    return jnp.sin(intrinsics.as_f64(val) * constants.PI)


def tanpi(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\tan(\\pi x)$ for half-turn argument $x$."""
    return jnp.tan(intrinsics.as_f64(val) * constants.PI)


def acos(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\arccos(x)$ in radians."""
    return jnp.arccos(intrinsics.as_f64(val))


def asin(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\arcsin(x)$ in radians."""
    return jnp.arcsin(intrinsics.as_f64(val))


def atan(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\arctan(x)$ in radians."""
    return jnp.arctan(intrinsics.as_f64(val))


def acospi(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\arccos(x) / \\pi$ in half-turns."""
    return jnp.arccos(intrinsics.as_f64(val)) / constants.PI


def asinpi(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\arcsin(x) / \\pi$ in half-turns."""
    return jnp.arcsin(intrinsics.as_f64(val)) / constants.PI


def atanpi(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\arctan(x) / \\pi$ in half-turns."""
    return jnp.arctan(intrinsics.as_f64(val)) / constants.PI


def cosh(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\cosh(x)$."""
    return jnp.cosh(intrinsics.as_f64(val))


def sinh(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\sinh(x)$."""
    return jnp.sinh(intrinsics.as_f64(val))


def tanh(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\tanh(x)$."""
    return jnp.tanh(intrinsics.as_f64(val))


def acosh(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\operatorname{arcosh}(x)$."""
    return jnp.arccosh(intrinsics.as_f64(val))


def asinh(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\operatorname{arsinh}(x)$."""
    return jnp.arcsinh(intrinsics.as_f64(val))


def atanh(val: intrinsics.ArrayLike) -> jax.Array:
    """Compute $\\operatorname{artanh}(x)$."""
    return jnp.arctanh(intrinsics.as_f64(val))


def sgn(val: intrinsics.ArrayLike) -> jax.Array:
    """Evaluate the three-valued signum function $\\operatorname{sgn}(x)$.

    Returns `-1.0` where `val < 0.0`, `0.0` where `val == 0.0`, `+1.0` where
    `val > 0.0`, and `NaN` where `val` is `NaN`.

    Args:
        val: Scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.sign(intrinsics.as_f64(val))


def heaviside(val: intrinsics.ArrayLike) -> jax.Array:
    """Evaluate the symmetric Heaviside step function $H(x)$.

    Returns `0.0` where `val < 0.0`, `0.5` where `val == 0.0`, `1.0` where
    `val > 0.0`, and `NaN` where `val` is `NaN`.

    Args:
        val: Scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.heaviside(intrinsics.as_f64(val), 0.5)


def str_to_double(token: str) -> float:
    """Parse a single Fortran or IEEE-754 floating-point string into `float`.

    Args:
        token: String representation of a floating-point value (supports `D`/`d`
            exponent markers).

    Returns:
        Parsed Python `float`.

    Raises:
        ValueError: If `token` cannot be parsed as a valid float.
    """
    normalized = token.strip().replace("D", "e").replace("d", "e")
    return float(normalized)


def str_to_vector(text: str) -> jax.Array:
    """Parse whitespace-delimited Fortran/IEEE-754 numbers into a 1D JAX array.

    Args:
        text: Whitespace-separated string of floating-point tokens.

    Returns:
        1D `float64` JAX array of shape `(N,)`.

    Raises:
        ValueError: If any token cannot be parsed as a valid float.
    """
    vals = [str_to_double(tok) for tok in text.split()]
    return jnp.asarray(vals, dtype=jnp.float64)
