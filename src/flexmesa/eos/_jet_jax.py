# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat `jnp.float64` bivariate 3rd-order Taylor jet kernels (`(..., 10)`).

Ports MESA's `auto_diff_real_2var_order3` (`flexmesa-autodiff/src/order3.rs`,
`order3_ops.rs`, and `kernels3.rs`) as pure branchless JAX array functions
with zero Python wrapper classes or nested `jax.jacfwd` graph expansion.

Component order along `axis=-1` (size 10):
`[val, d1, d2, d11, d12, d22, d111, d112, d122, d222]`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bucket_size(n_species: int) -> int:
    """Return padded species bucket width (`8`, `32`, or multiple of `32`).

    Args:
        n_species: Active species count `N_species >= 1`.

    Returns:
        Bucket size (`8` for `1..8`, `32` for `9..32`, or `32 * ceil(N / 32)`).
    """
    if n_species <= 8:
        return 8
    if n_species <= 32:
        return 32
    return ((n_species + 31) // 32) * 32


def pad_trailing(arr: jax.Array, target: int, fill: float = 0.0) -> jax.Array:
    """Pad trailing species axis of `arr` to `target` with constant `fill`.

    Args:
        arr: Input JAX array of shape `(*batch, N_species)`.
        target: Target trailing dimension `>= N_species`.
        fill: Constant scalar fill value for padded trailing entries.

    Returns:
        Padded `jnp.float64` array of shape `(*batch, target)`.
    """
    pad = target - int(arr.shape[-1])
    if pad <= 0:
        return arr
    widths = [(0, 0)] * (arr.ndim - 1) + [(0, pad)]
    return jnp.pad(arr, widths, constant_values=fill)


def j3_const(val: jax.Array) -> jax.Array:
    """Construct a constant 3rd-order jet `[val, 0, ..., 0]` (`(..., 10)`).

    Args:
        val: Primal value array of shape `(*batch,)`.

    Returns:
        Jet array of shape `(*batch, 10)` with zero partial derivatives.
    """
    v = jnp.asarray(val, dtype=jnp.float64)
    z = jnp.zeros_like(v)
    return jnp.stack([v, z, z, z, z, z, z, z, z, z], axis=-1)


def j3_var1(val: jax.Array) -> jax.Array:
    """Seed independent variable 1 (`d1 = 1`) as shape `(*batch, 10)`.

    Args:
        val: Primal coordinate $x_1$ (typically $T$ in `K`), shape `(*batch,)`.

    Returns:
        Jet array of shape `(*batch, 10)` with `d1val1 = 1.0`.
    """
    v = jnp.asarray(val, dtype=jnp.float64)
    z = jnp.zeros_like(v)
    o = jnp.ones_like(v)
    return jnp.stack([v, o, z, z, z, z, z, z, z, z], axis=-1)


def j3_var2(val: jax.Array) -> jax.Array:
    """Seed independent variable 2 (`d2 = 1`) as shape `(*batch, 10)`.

    Args:
        val: Primal coordinate $x_2$ (typically $\\rho$ in `g cm^-3`),
            shape `(*batch,)`.

    Returns:
        Jet array of shape `(*batch, 10)` with `d1val2 = 1.0`.
    """
    v = jnp.asarray(val, dtype=jnp.float64)
    z = jnp.zeros_like(v)
    o = jnp.ones_like(v)
    return jnp.stack([v, z, o, z, z, z, z, z, z, z], axis=-1)


def j3_where(cond: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Select jet `x` or `y` elementwise from boolean mask `cond`.

    Args:
        cond: Boolean condition array of shape `(*batch,)`.
        x: True-branch jet array of shape `(*batch, 10)`.
        y: False-branch jet array of shape `(*batch, 10)`.

    Returns:
        Selected jet array of shape `(*batch, 10)`.
    """
    return jnp.where(jnp.expand_dims(cond, axis=-1), x, y)


def j3_diff1(x: jax.Array) -> jax.Array:
    """Extract $\\partial f / \\partial x_1$ (`differentiate_1` in `order3.rs`).

    Args:
        x: Input jet array of shape `(*batch, 10)`.

    Returns:
        First-variable derivative jet of shape `(*batch, 10)` with zero 3rd
        partials.
    """
    z = jnp.zeros_like(x[..., 0])
    return jnp.stack(
        [
            x[..., 1],
            x[..., 3],
            x[..., 4],
            x[..., 6],
            x[..., 7],
            x[..., 8],
            z,
            z,
            z,
            z,
        ],
        axis=-1,
    )


def j3_diff2(x: jax.Array) -> jax.Array:
    """Extract $\\partial f / \\partial x_2$ (`differentiate_2` in `order3.rs`).

    Args:
        x: Input jet array of shape `(*batch, 10)`.

    Returns:
        Second-variable derivative jet of shape `(*batch, 10)` with zero 3rd
        partials.
    """
    z = jnp.zeros_like(x[..., 0])
    return jnp.stack(
        [
            x[..., 2],
            x[..., 4],
            x[..., 5],
            x[..., 7],
            x[..., 8],
            x[..., 9],
            z,
            z,
            z,
            z,
        ],
        axis=-1,
    )


def j3_unary(
    x: jax.Array,
    z_val: jax.Array,
    z_d1x: jax.Array,
    z_d2x: jax.Array,
    z_d3x: jax.Array,
) -> jax.Array:
    """Propagate scalar derivatives `(f, f', f'', f''')` through jet `x`.

    Args:
        x: Argument jet array of shape `(*batch, 10)`.
        z_val: Scalar value $f(x_0)$, shape `(*batch,)`.
        z_d1x: First derivative $f'(x_0)$, shape `(*batch,)`.
        z_d2x: Second derivative $f''(x_0)$, shape `(*batch,)`.
        z_d3x: Third derivative $f'''(x_0)$, shape `(*batch,)`.

    Returns:
        Composite jet array of shape `(*batch, 10)`.
    """
    x1, x2 = x[..., 1], x[..., 2]
    x11, x12, x22 = x[..., 3], x[..., 4], x[..., 5]
    x111, x112, x122, x222 = x[..., 6], x[..., 7], x[..., 8], x[..., 9]
    q0 = x1 * x1
    q1 = x2 * z_d2x
    q2 = x2 * x2
    q3 = x1 * z_d2x
    q4 = 2.0 * x12
    return jnp.stack(
        [
            z_val,
            x1 * z_d1x,
            x2 * z_d1x,
            q0 * z_d2x + x11 * z_d1x,
            q1 * x1 + x12 * z_d1x,
            q2 * z_d2x + x22 * z_d1x,
            3.0 * q3 * x11 + x111 * z_d1x + z_d3x * (q0 * x1),
            q0 * x2 * z_d3x + q1 * x11 + q3 * q4 + x112 * z_d1x,
            q1 * q4 + q2 * x1 * z_d3x + q3 * x22 + x122 * z_d1x,
            3.0 * q1 * x22 + x222 * z_d1x + z_d3x * (q2 * x2),
        ],
        axis=-1,
    )


def j3_compose(jet: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Substitute `(x, y)` into bivariate Taylor `jet` (`compose_binary`).

    Args:
        jet: Outer Taylor jet of shape `(*batch, 10)`.
        x: First inner argument jet of shape `(*batch, 10)`.
        y: Second inner argument jet of shape `(*batch, 10)`.

    Returns:
        Composed 3rd-order jet of shape `(*batch, 10)`.
    """
    z_val = jet[..., 0]
    z_d1x, z_d1y = jet[..., 1], jet[..., 2]
    z_d2x, z_d1x_d1y, z_d2y = jet[..., 3], jet[..., 4], jet[..., 5]
    z_d3x, z_d2x_d1y = jet[..., 6], jet[..., 7]
    z_d1x_d2y, z_d3y = jet[..., 8], jet[..., 9]

    x1, x2 = x[..., 1], x[..., 2]
    x11, x12, x22 = x[..., 3], x[..., 4], x[..., 5]
    x111, x112, x122, x222 = x[..., 6], x[..., 7], x[..., 8], x[..., 9]
    y1, y2 = y[..., 1], y[..., 2]
    y11, y12, y22 = y[..., 3], y[..., 4], y[..., 5]
    y111, y112, y122, y222 = y[..., 6], y[..., 7], y[..., 8], y[..., 9]

    q0, q1 = x1 * x1, y1 * y1
    q2 = x1 * z_d1x_d1y
    q3 = 2.0 * q2
    q4, q5 = x2 * z_d2x, y2 * z_d1x_d1y
    q6, q7 = x2 * z_d1x_d1y, y2 * z_d2y
    q8, q9 = x2 * x2, y2 * y2
    q10 = 2.0 * q5
    q11 = x1 * z_d2x
    q12, q13 = 3.0 * x11, 3.0 * y11
    q14, q15 = y1 * z_d1x_d1y, y1 * z_d2y
    q16, q17 = q1 * z_d1x_d2y, q0 * z_d2x_d1y
    q18, q19 = 2.0 * x12, 2.0 * y12
    q20 = 2.0 * x1 * y1
    q21, q22 = x2 * z_d2x_d1y, y2 * z_d1x_d2y
    q23, q24 = q9 * z_d1x_d2y, q8 * z_d2x_d1y
    q25, q26 = 3.0 * x22, 3.0 * y22

    return jnp.stack(
        [
            z_val,
            x1 * z_d1x + y1 * z_d1y,
            x2 * z_d1x + y2 * z_d1y,
            q0 * z_d2x + q1 * z_d2y + q3 * y1 + x11 * z_d1x + y11 * z_d1y,
            (q4 + q5) * x1 + (q6 + q7) * y1 + x12 * z_d1x + y12 * z_d1y,
            q10 * x2 + q8 * z_d2x + q9 * z_d2y + x22 * z_d1x + y22 * z_d1y,
            (
                3.0 * q16 * x1
                + 3.0 * q17 * y1
                + q12 * (q11 + q14)
                + q13 * (q15 + q2)
                + x111 * z_d1x
                + y111 * z_d1y
                + z_d3x * (q0 * x1)
                + z_d3y * (q1 * y1)
            ),
            (
                q0 * x2 * z_d3x
                + q1 * y2 * z_d3y
                + (q11 + q14) * q18
                + q15 * q19
                + q16 * x2
                + q17 * y2
                + q20 * (q21 + q22)
                + q3 * y12
                + (q4 + q5) * x11
                + (q6 + q7) * y11
                + x112 * z_d1x
                + y112 * z_d1y
            ),
            (
                2.0 * q21 * x1 * y2
                + 2.0 * q22 * x2 * y1
                + q10 * x12
                + (q11 + q14) * x22
                + (q15 + q2) * y22
                + q18 * q4
                + q19 * (q6 + q7)
                + q23 * x1
                + q24 * y1
                + q8 * x1 * z_d3x
                + q9 * y1 * z_d3y
                + x122 * z_d1x
                + y122 * z_d1y
            ),
            (
                3.0 * q23 * x2
                + 3.0 * q24 * y2
                + q25 * (q4 + q5)
                + q26 * (q6 + q7)
                + x222 * z_d1x
                + y222 * z_d1y
                + z_d3x * (q8 * x2)
                + z_d3y * (q9 * y2)
            ),
        ],
        axis=-1,
    )


def j3_add_s(x: jax.Array, s: float | jax.Array) -> jax.Array:
    """Return `x + s` shifting only the primal slot `x[..., 0]`."""
    return x.at[..., 0].add(jnp.asarray(s, dtype=jnp.float64))


def j3_rsub_s(s: float | jax.Array, x: jax.Array) -> jax.Array:
    """Return `s - x` negating all derivatives and shifting primal by `s`."""
    return (-x).at[..., 0].add(jnp.asarray(s, dtype=jnp.float64))


def j3_scale(x: jax.Array, s: float | jax.Array) -> jax.Array:
    """Scale all 10 jet components by constant factor `s`."""
    return x * jnp.expand_dims(jnp.asarray(s, dtype=jnp.float64), axis=-1)


def j3_mul(x: jax.Array, y: jax.Array) -> jax.Array:
    """Multiply two 3rd-order jets `x * y` via the Leibniz product rule."""
    xv, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    x11, x12, x22 = x[..., 3], x[..., 4], x[..., 5]
    x111, x112, x122, x222 = x[..., 6], x[..., 7], x[..., 8], x[..., 9]
    yv, y1, y2 = y[..., 0], y[..., 1], y[..., 2]
    y11, y12, y22 = y[..., 3], y[..., 4], y[..., 5]
    y111, y112, y122, y222 = y[..., 6], y[..., 7], y[..., 8], y[..., 9]
    return jnp.stack(
        [
            xv * yv,
            x1 * yv + y1 * xv,
            x2 * yv + y2 * xv,
            2.0 * x1 * y1 + x11 * yv + y11 * xv,
            y2 * x1 + x2 * y1 + x12 * yv + y12 * xv,
            2.0 * y2 * x2 + x22 * yv + y22 * xv,
            3.0 * x11 * y1 + 3.0 * y11 * x1 + x111 * yv + y111 * xv,
            (
                2.0 * y1 * x12
                + 2.0 * x1 * y12
                + y2 * x11
                + x2 * y11
                + x112 * yv
                + y112 * xv
            ),
            (
                2.0 * y2 * x12
                + y1 * x22
                + x1 * y22
                + 2.0 * y12 * x2
                + x122 * yv
                + y122 * xv
            ),
            3.0 * x22 * y2 + 3.0 * y22 * x2 + x222 * yv + y222 * xv,
        ],
        axis=-1,
    )


def j3_inv(y: jax.Array) -> jax.Array:
    """Return reciprocal jet `1 / y` (`powm1` in `kernels3.rs`)."""
    inv = 1.0 / y[..., 0]
    inv2 = inv * inv
    return j3_unary(y, inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2)


def j3_div(x: jax.Array, y: jax.Array) -> jax.Array:
    """Return quotient jet `x / y`."""
    return j3_mul(x, j3_inv(y))


def j3_rdiv_s(s: float | jax.Array, y: jax.Array) -> jax.Array:
    """Return scalar-over-jet quotient `s / y`."""
    return j3_scale(j3_inv(y), s)


def j3_sqrt(x: jax.Array) -> jax.Array:
    """Return square-root jet $\\sqrt{x}$ (`unary3_sqrt` in `kernels3.rs`)."""
    xv = x[..., 0]
    q0 = jnp.sqrt(xv)
    return j3_unary(x, q0, 0.5 / q0, -0.25 / (xv * q0), 0.375 / (xv * xv * q0))


def j3_exp(x: jax.Array) -> jax.Array:
    """Return exponential jet $\\exp(x)$ (`unary3_exp` in `kernels3.rs`)."""
    q0 = jnp.exp(x[..., 0])
    return j3_unary(x, q0, q0, q0, q0)


def j3_log(x: jax.Array) -> jax.Array:
    """Return natural-log jet $\\ln(x)$ (`unary3_log` in `kernels3.rs`)."""
    xv = x[..., 0]
    inv = 1.0 / xv
    inv2 = inv * inv
    return j3_unary(x, jnp.log(xv), inv, -inv2, 2.0 * inv2 * inv)


def j3_pow2(x: jax.Array) -> jax.Array:
    """Return squared jet $x^2$ (`unary3_pown(..., 2)` in `kernels3.rs`)."""
    xv = x[..., 0]
    return j3_unary(
        x, xv * xv, 2.0 * xv, jnp.full_like(xv, 2.0), jnp.zeros_like(xv)
    )


def j3_pow3(x: jax.Array) -> jax.Array:
    """Return cubed jet $x^3$ (`unary3_pown(..., 3)` in `kernels3.rs`)."""
    xv = x[..., 0]
    x2 = xv * xv
    return j3_unary(x, x2 * xv, 3.0 * x2, 6.0 * xv, jnp.full_like(xv, 6.0))


def j3_pow4(x: jax.Array) -> jax.Array:
    """Return quartic jet $x^4$ (`unary3_pown(..., 4)` in `kernels3.rs`)."""
    xv = x[..., 0]
    x2 = xv * xv
    return j3_unary(x, x2 * x2, 4.0 * x2 * xv, 12.0 * x2, 24.0 * xv)


def j3_powf(x: jax.Array, p: float | jax.Array) -> jax.Array:
    """Return real-power jet $x^p$ (`unary3_pow_scalar` in `kernels3.rs`)."""
    xv = x[..., 0]
    pa = jnp.asarray(p, dtype=jnp.float64)
    d1 = pa * jnp.power(xv, pa - 1.0)
    d2 = pa * (pa - 1.0) * jnp.power(xv, pa - 2.0)
    d3 = pa * (pa - 1.0) * (pa - 2.0) * jnp.power(xv, pa - 3.0)
    return j3_unary(x, jnp.power(xv, pa), d1, d2, d3)


def j3_atan(x: jax.Array) -> jax.Array:
    """Return arctangent jet $\\arctan(x)$ (`unary3_atan` in `kernels3.rs`)."""
    xv = x[..., 0]
    x2 = xv * xv
    q0 = 1.0 + x2
    q0_2 = q0 * q0
    return j3_unary(
        x,
        jnp.arctan(xv),
        1.0 / q0,
        -2.0 * xv / q0_2,
        (6.0 * x2 - 2.0) / (q0_2 * q0),
    )


def j3_tanh(x: jax.Array) -> jax.Array:
    """Return hyperbolic-tangent jet $\\tanh(x)$ (`unary3_tanh`)."""
    th = jnp.tanh(x[..., 0])
    th2 = th * th
    sech2 = 1.0 - th2
    return j3_unary(
        x, th, sech2, -2.0 * th * sech2, -2.0 * sech2 * (1.0 - 3.0 * th2)
    )


def j3_min(x: jax.Array, y: jax.Array) -> jax.Array:
    """Return $\\min(x, y)$ with MESA's symmetric Heaviside subgradient."""
    xv, yv = x[..., 0], y[..., 0]
    q0 = jnp.expand_dims(jnp.where(yv > xv, 1.0, 0.0), axis=-1)
    q1 = jnp.expand_dims(jnp.where(xv > yv, 1.0, 0.0), axis=-1)
    return (q0 * x + q1 * y).at[..., 0].set(jnp.minimum(xv, yv))


def j3_max_s(x: jax.Array, c: float | jax.Array) -> jax.Array:
    """Return $\\max(x, c)$ with MESA's Heaviside subgradient."""
    xv = x[..., 0]
    ca = jnp.asarray(c, dtype=jnp.float64)
    q0 = jnp.expand_dims(jnp.where(xv > ca, 1.0, 0.0), axis=-1)
    return (x * q0).at[..., 0].set(jnp.maximum(xv, ca))


def j3_min_s(x: jax.Array, c: float | jax.Array) -> jax.Array:
    """Return $\\min(x, c)$ with MESA's Heaviside subgradient."""
    xv = x[..., 0]
    ca = jnp.asarray(c, dtype=jnp.float64)
    q0 = jnp.expand_dims(jnp.where(ca > xv, 1.0, 0.0), axis=-1)
    return (x * q0).at[..., 0].set(jnp.minimum(xv, ca))
