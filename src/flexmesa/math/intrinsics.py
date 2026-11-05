# SPDX-License-Identifier: LGPL-3.0-or-later
"""IEEE-754 float64 elementary and domain-safe mathematical intrinsics.

This module ports upstream MESA's `math` module (`math_lib_intrinsic.f90`,
`math_pown.f90`, and `math_io.f90`) to JAX-traceable 64-bit floating-point
functions. Domain-safe functions (`safe_sqrt`, `safe_log`, `safe_log10`) use
the double-`jnp.where` pattern so that neither primal values nor forward/reverse
automatic differentiation gradients ever evaluate `0.0 * NaN` or `0.0 * Inf`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants

type ArrayLike = float | int | Sequence[float] | np.ndarray | jax.Array

SAFE_LOG_FLOOR: float = 1.0e-99
SAFE_LOG_BAD_RETURN: float = -99.0


def as_f64(val: ArrayLike) -> jax.Array:
    """Convert scalar or array input `val` to a `float64` JAX array."""
    if isinstance(val, jax.Array) and val.dtype == jnp.float64:
        return val
    return jnp.asarray(val, dtype=jnp.float64)


def _safe_log_apply(
    val: ArrayLike, log_fn: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """Apply `log_fn` with MESA's `1e-99` floor and `-99.0` bad-value guard."""
    x_arr = as_f64(val)
    finite = jnp.isfinite(x_arr)
    clamped = jnp.where(
        finite & (x_arr > SAFE_LOG_FLOOR), x_arr, SAFE_LOG_FLOOR
    )
    return jnp.where(finite, log_fn(clamped), SAFE_LOG_BAD_RETURN)


def safe_sqrt(val: ArrayLike) -> jax.Array:
    """Compute the domain-safe square root $\\sqrt{\\max(x, 0)}$.

    Uses the double-`where` guard so negative inputs evaluate to `0.0` with
    zero derivative, while `NaN` inputs propagate `NaN`.

    Args:
        val: Input scalar or array of shape `(*batch,)` (dimensionless).

    Returns:
        `float64` JAX array of shape `(*batch,)` with $\\sqrt{\\max(x, 0)}$.

    Examples:
        >>> import flexmesa.math as fmath
        >>> float(fmath.safe_sqrt(4.0))
        2.0
        >>> float(fmath.safe_sqrt(-9.0))
        0.0
    """
    x_arr = as_f64(val)
    pos = x_arr > 0.0
    safe_x = jnp.where(pos, x_arr, 1.0)
    res = jnp.where(pos, jnp.sqrt(safe_x), 0.0)
    return jnp.where(jnp.isnan(x_arr), jnp.nan, res)


def safe_log(val: ArrayLike) -> jax.Array:
    """Compute the domain-safe natural logarithm $\\ln(\\max(10^{-99}, x))$.

    Returns `-99.0` where `val` is non-finite (`NaN` or `±Inf`), matching MESA's
    `safe_log` safeguard without propagating `NaN` through `jnp.where`.

    Args:
        val: Input scalar or array of shape `(*batch,)` (dimensionless).

    Returns:
        `float64` JAX array of shape `(*batch,)`.

    Examples:
        >>> import flexmesa.math as fmath
        >>> float(fmath.safe_log(1.0))
        0.0
        >>> float(fmath.safe_log(float("nan")))
        -99.0
    """
    return _safe_log_apply(val, jnp.log)


def safe_log10(val: ArrayLike) -> jax.Array:
    """Compute the domain-safe base-10 log $\\log_{10}(\\max(10^{-99}, x))$.

    Returns `-99.0` where `val` is non-finite (`NaN` or `±Inf`) or `<= 1e-99`.

    Args:
        val: Input scalar or array of shape `(*batch,)` (dimensionless).

    Returns:
        `float64` JAX array of shape `(*batch,)`.

    Examples:
        >>> import flexmesa.math as fmath
        >>> float(fmath.safe_log10(10.0))
        1.0
        >>> float(fmath.safe_log10(0.0))
        -99.0
    """
    return _safe_log_apply(val, jnp.log10)


def log(val: ArrayLike) -> jax.Array:
    """Compute the natural logarithm $\\ln(x)$ in `float64`.

    Args:
        val: Positive scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.log(as_f64(val))


def log10(val: ArrayLike) -> jax.Array:
    """Compute the base-10 logarithm $\\log_{10}(x)$ in `float64`.

    Args:
        val: Positive scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.log10(as_f64(val))


def log1p(val: ArrayLike) -> jax.Array:
    """Compute $\\ln(1 + x)$ matching MESA's `log1p` definition.

    Args:
        val: Scalar or array of shape `(*batch,)` with $x > -1$.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.log(1.0 + as_f64(val))


def log2(val: ArrayLike) -> jax.Array:
    """Compute $\\log_{2}(x) = \\ln(x) / \\ln(2)$ matching MESA's `log2`.

    Args:
        val: Positive scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.log(as_f64(val)) / constants.LN2


def exp(val: ArrayLike) -> jax.Array:
    """Compute the natural exponential $e^{x}$ in `float64`.

    Args:
        val: Scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.exp(as_f64(val))


def expm1(val: ArrayLike) -> jax.Array:
    """Compute $e^{x} - 1$ matching MESA's `expm1` definition.

    Args:
        val: Scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.exp(as_f64(val)) - 1.0


def exp10(val: ArrayLike) -> jax.Array:
    """Compute the base-10 exponential $10^{x} = \\exp(x \\ln 10)$.

    Args:
        val: Scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    x_arr = as_f64(val)
    is_int = (
        jnp.isfinite(x_arr)
        & (x_arr == jnp.floor(x_arr))
        & (jnp.abs(x_arr) <= 308.0)
    )
    safe_int = jnp.where(is_int, x_arr, 0.0)
    return jnp.where(
        is_int, jnp.power(10.0, safe_int), jnp.exp(x_arr * constants.LN10)
    )


def hypot(lhs: ArrayLike, rhs: ArrayLike) -> jax.Array:
    """Compute the Euclidean hypotenuse $\\sqrt{x^{2} + y^{2}}$ in `float64`."""
    return jnp.hypot(as_f64(lhs), as_f64(rhs))


def pow(base: ArrayLike, exponent: ArrayLike) -> jax.Array:
    """Raise `base` to power `exponent` ($x^{y}$) in `float64`.

    Args:
        base: Base scalar or array of shape `(*batch,)`.
        exponent: Exponent scalar or array broadcastable to `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return jnp.power(as_f64(base), as_f64(exponent))


def powi(base: ArrayLike, exponent: int) -> jax.Array:
    """Raise `base` to an integer power `exponent` ($x^{n}$) in `float64`."""
    return jnp.power(as_f64(base), float(exponent))


def powm1(val: ArrayLike) -> jax.Array:
    """Compute the reciprocal $x^{-1} = 1 / x$.

    Args:
        val: Nonzero scalar or array of shape `(*batch,)`.

    Returns:
        `float64` JAX array of shape `(*batch,)`.
    """
    return 1.0 / as_f64(val)


def pow2(val: ArrayLike) -> jax.Array:
    """Compute $x^{2}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa


def pow3(val: ArrayLike) -> jax.Array:
    """Compute $x^{3}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa * xa


def pow4(val: ArrayLike) -> jax.Array:
    """Compute $x^{4}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa * xa * xa


def pow5(val: ArrayLike) -> jax.Array:
    """Compute $x^{5}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa * xa * xa * xa


def pow6(val: ArrayLike) -> jax.Array:
    """Compute $x^{6}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa * xa * xa * xa * xa


def pow7(val: ArrayLike) -> jax.Array:
    """Compute $x^{7}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa * xa * xa * xa * xa * xa


def pow8(val: ArrayLike) -> jax.Array:
    """Compute $x^{8}$ via repeated multiplication."""
    xa = as_f64(val)
    return xa * xa * xa * xa * xa * xa * xa * xa
