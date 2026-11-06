# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D and 2D biquintic (`h5`) and bicubic (`h3`) Hermite basis kernels in JAX.

Ports `eos/private/helm_polynomials.f90` (Timmes & Swesty 2000, ApJS, 126, 501)
into `@jax.jit`-compatible `jnp.float64` polynomial evaluation kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_H5_T_IDX: tuple[int, ...] = (
    0,
    0,
    3,
    3,
    1,
    1,
    4,
    4,
    2,
    2,
    5,
    5,
    0,
    0,
    3,
    3,
    0,
    0,
    3,
    3,
    1,
    1,
    4,
    4,
    1,
    1,
    4,
    4,
    2,
    2,
    5,
    5,
    2,
    2,
    5,
    5,
)
_H5_D_IDX: tuple[int, ...] = (
    0,
    3,
    0,
    3,
    0,
    3,
    0,
    3,
    0,
    3,
    0,
    3,
    1,
    4,
    1,
    4,
    2,
    5,
    2,
    5,
    1,
    4,
    1,
    4,
    2,
    5,
    2,
    5,
    1,
    4,
    1,
    4,
    2,
    5,
    2,
    5,
)
_H3_T_IDX: tuple[int, ...] = (0, 0, 2, 2, 1, 1, 3, 3, 0, 0, 2, 2, 1, 1, 3, 3)
_H3_D_IDX: tuple[int, ...] = (0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3)


def quintic_basis_1d(
    z: jax.Array,
    d: jax.Array,
    d2: jax.Array,
    di: jax.Array,
    d2i: jax.Array,
    d3i: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate 1D quintic Hermite basis weights `(w0, w1, w2, w3)`.

    Args:
        z: Normalized cell coordinate in `[0, 1]`, shape `(*batch,)`.
        d: Physical cell width, shape `(*batch,)`.
        d2: Squared cell width `d^2`, shape `(*batch,)`.
        di: Inverse cell width `1 / d`, shape `(*batch,)`.
        d2i: Inverse squared cell width `1 / d^2`, shape `(*batch,)`.
        d3i: Inverse cubed cell width `1 / d^3`, shape `(*batch,)`.

    Returns:
        Tuple `(w0, w1, w2, w3)` of 6-element basis weight arrays of shape
        `(*batch, 6)` for value, first, second, and third derivatives.
    """
    x = jnp.stack([z, 1.0 - z], axis=-1)
    p0 = x * x * x * (x * (-6.0 * x + 15.0) - 10.0) + 1.0
    dp0 = x * x * (x * (-30.0 * x + 60.0) - 30.0)
    ddp0 = x * (x * (-120.0 * x + 180.0) - 60.0)
    dddp0 = x * (-360.0 * x + 360.0) - 60.0
    p1 = x * (x * x * (x * (-3.0 * x + 8.0) - 6.0) + 1.0)
    dp1 = x * x * (x * (-15.0 * x + 32.0) - 18.0) + 1.0
    ddp1 = x * (x * (-60.0 * x + 96.0) - 36.0)
    dddp1 = x * (-180.0 * x + 192.0) - 36.0
    p2 = 0.5 * x * x * (x * (x * (-x + 3.0) - 3.0) + 1.0)
    dp2 = 0.5 * x * (x * (x * (-5.0 * x + 12.0) - 9.0) + 2.0)
    ddp2 = 0.5 * (x * (x * (-20.0 * x + 36.0) - 18.0) + 2.0)
    dddp2 = 0.5 * (x * (-60.0 * x + 72.0) - 18.0)

    w0 = jnp.stack(
        [
            p0[..., 0],
            p1[..., 0] * d,
            p2[..., 0] * d2,
            p0[..., 1],
            -p1[..., 1] * d,
            p2[..., 1] * d2,
        ],
        axis=-1,
    )
    w1 = jnp.stack(
        [
            dp0[..., 0] * di,
            dp1[..., 0],
            dp2[..., 0] * d,
            -dp0[..., 1] * di,
            dp1[..., 1],
            -dp2[..., 1] * d,
        ],
        axis=-1,
    )
    w2 = jnp.stack(
        [
            ddp0[..., 0] * d2i,
            ddp1[..., 0] * di,
            ddp2[..., 0],
            ddp0[..., 1] * d2i,
            -ddp1[..., 1] * di,
            ddp2[..., 1],
        ],
        axis=-1,
    )
    w3 = jnp.stack(
        [
            dddp0[..., 0] * d3i,
            dddp1[..., 0] * d2i,
            dddp2[..., 0] * di,
            -dddp0[..., 1] * d3i,
            dddp1[..., 1] * d2i,
            -dddp2[..., 1] * di,
        ],
        axis=-1,
    )
    return w0, w1, w2, w3


def cubic_basis_1d(
    z: jax.Array, d: jax.Array, di: jax.Array, d2i: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 1D cubic Hermite basis weights `(w0, w1, w2)`.

    Args:
        z: Normalized cell coordinate in `[0, 1]`, shape `(*batch,)`.
        d: Physical cell width, shape `(*batch,)`.
        di: Inverse cell width `1 / d`, shape `(*batch,)`.
        d2i: Inverse squared cell width `1 / d^2`, shape `(*batch,)`.

    Returns:
        Tuple `(w0, w1, w2)` of 4-element basis weight arrays of shape
        `(*batch, 4)` for value, first, and second derivatives.
    """
    x = jnp.stack([z, 1.0 - z], axis=-1)
    xp0 = x * x * (2.0 * x - 3.0) + 1.0
    xdp0 = x * (6.0 * x - 6.0)
    xddp0 = 12.0 * x - 6.0
    xp1 = x * (x * (x - 2.0) + 1.0)
    xdp1 = x * (3.0 * x - 4.0) + 1.0
    xddp1 = 6.0 * x - 4.0

    w0 = jnp.stack(
        [xp0[..., 0], xp1[..., 0] * d, xp0[..., 1], -xp1[..., 1] * d], axis=-1
    )
    w1 = jnp.stack(
        [xdp0[..., 0] * di, xdp1[..., 0], -xdp0[..., 1] * di, xdp1[..., 1]],
        axis=-1,
    )
    w2 = jnp.stack(
        [
            xddp0[..., 0] * d2i,
            xddp1[..., 0] * di,
            xddp0[..., 1] * d2i,
            -xddp1[..., 1] * di,
        ],
        axis=-1,
    )
    return w0, w1, w2


def h5(fi: jax.Array, wt: jax.Array, wd: jax.Array) -> jax.Array:
    """Evaluate the 36-term biquintic Hermite tensor product `h5`."""
    w36 = wt[..., _H5_T_IDX] * wd[..., _H5_D_IDX]
    return jnp.sum(fi * w36, axis=-1)


def h3(fi: jax.Array, wt: jax.Array, wd: jax.Array) -> jax.Array:
    """Evaluate the 16-term bicubic Hermite tensor product `h3`."""
    w16 = wt[..., _H3_T_IDX] * wd[..., _H3_D_IDX]
    return jnp.sum(fi * w16, axis=-1)
