# SPDX-License-Identifier: LGPL-3.0-or-later
"""Block-tridiagonal matrix-vector products and Thomas solvers (`mtx`).

Represents a block-tridiagonal matrix across `nz` shells and `nvar` variables
per shell via three arrays `lblk`, `dblk`, and `ublk` of shape
`(nz, nvar, nvar)` in `float64`, where shell `k` has sub-diagonal block
`lblk[k]` ($k > 0$), diagonal block `dblk[k]`, and super-diagonal block
`ublk[k]` ($k + 1 < n_z$). Includes a JAX-differentiable block Thomas solver
(`solve_block_tridiagonal_jax`) built on `jax.lax.scan`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa.mtx import dense


def _validate_blocks(
    lblk: npt.ArrayLike,
    dblk: npt.ArrayLike,
    ublk: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int,
    int,
]:
    """Validates `(nz, nvar, nvar)` block-tridiagonal arrays in `float64`."""
    l_arr = np.asarray(lblk, dtype=np.float64)
    d_arr = np.asarray(dblk, dtype=np.float64)
    u_arr = np.asarray(ublk, dtype=np.float64)
    if (
        d_arr.ndim != 3
        or d_arr.shape[0] == 0
        or d_arr.shape[1] == 0
        or d_arr.shape[1] != d_arr.shape[2]
        or l_arr.shape != d_arr.shape
        or u_arr.shape != d_arr.shape
    ):
        raise ValueError(
            "lblk, dblk, ublk must share non-empty shape (nz, nvar, nvar)."
        )
    if not (
        np.all(np.isfinite(l_arr))
        and np.all(np.isfinite(d_arr))
        and np.all(np.isfinite(u_arr))
    ):
        raise ValueError("Block-tridiagonal arrays contain NaN or Inf.")
    return l_arr, d_arr, u_arr, int(d_arr.shape[0]), int(d_arr.shape[1])


def block_dble_mv(
    lblk: npt.ArrayLike,
    dblk: npt.ArrayLike,
    ublk: npt.ArrayLike,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates $y_k = L_k b_{k-1} + D_k b_k + U_k b_{k+1}$ in `float64`.

    Args:
        lblk: Sub-diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        ublk: Super-diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        b: Input vector of shape `(nz, nvar)` or `(nz * nvar,)` in `float64`
            (dimensionless).

    Returns:
        Product array matching the shape of `b` in `float64`.

    Raises:
        ValueError: If block or vector dimensions do not match.
    """
    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(lblk, dblk, ublk)
    b_arr = np.asarray(b, dtype=np.float64)
    is_1d = b_arr.ndim == 1
    if is_1d:
        if b_arr.shape != (nz * nvar,):
            raise ValueError("1D vector b must have shape (nz * nvar,).")
        b_2d = b_arr.reshape(nz, nvar)
    elif b_arr.shape == (nz, nvar):
        b_2d = b_arr
    else:
        raise ValueError("b must have shape (nz, nvar) or (nz * nvar,).")

    prod = np.einsum("kij,kj->ki", d_arr, b_2d)
    if nz > 1:
        prod[1:] += np.einsum("kij,kj->ki", l_arr[1:], b_2d[:-1])
        prod[:-1] += np.einsum("kij,kj->ki", u_arr[:-1], b_2d[1:])
    return prod.reshape(-1) if is_1d else prod


def block_multiply_xa(
    lblk: npt.ArrayLike,
    dblk: npt.ArrayLike,
    ublk: npt.ArrayLike,
    x: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates left product $b = x^{\\top} A$ for a block-tridiagonal system.

    Args:
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`.
        dblk: Diagonal blocks `(nz, nvar, nvar)` in `float64`.
        ublk: Super-diagonal blocks `(nz, nvar, nvar)` in `float64`.
        x: Left vector of shape `(nz, nvar)` or `(nz * nvar,)` in `float64`.

    Returns:
        Row-vector product $b = x^{\\top} A$ matching the shape of `x`.

    Raises:
        ValueError: If block or vector dimensions do not match.
    """
    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(lblk, dblk, ublk)
    x_arr = np.asarray(x, dtype=np.float64)
    is_1d = x_arr.ndim == 1
    if is_1d:
        if x_arr.shape != (nz * nvar,):
            raise ValueError("1D vector x must have shape (nz * nvar,).")
        x_2d = x_arr.reshape(nz, nvar)
    elif x_arr.shape == (nz, nvar):
        x_2d = x_arr
    else:
        raise ValueError("x must have shape (nz, nvar) or (nz * nvar,).")

    out = np.einsum("ki,kij->kj", x_2d, d_arr)
    if nz > 1:
        out[1:] += np.einsum("ki,kij->kj", x_2d[:-1], u_arr[:-1])
        out[:-1] += np.einsum("ki,kij->kj", x_2d[1:], l_arr[1:])
    return out.reshape(-1) if is_1d else out


def block_thomas_factor(
    lblk: npt.ArrayLike,
    dblk: npt.ArrayLike,
    ublk: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Factors a block-tridiagonal system via the block Thomas algorithm.

    Args:
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`.
        dblk: Diagonal blocks `(nz, nvar, nvar)` in `float64`.
        ublk: Super-diagonal blocks `(nz, nvar, nvar)` in `float64`.

    Returns:
        Tuple `(d_factored, u_factored, ipiv)` of shapes `(nz, nvar, nvar)`,
        `(nz, nvar, nvar)`, and `(nz, nvar)`.

    Raises:
        ValueError: If any diagonal block is singular or shapes are invalid.
    """
    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(lblk, dblk, ublk)
    d_work = d_arr.copy()
    u_work = u_arr.copy()
    ipiv = np.zeros((nz, nvar), dtype=np.int64)
    for k in range(nz):
        lu_k, piv_k = dense.dgetrf(d_work[k])
        d_work[k] = lu_k
        ipiv[k] = piv_k
        if k + 1 < nz:
            u_work[k] = dense.dgetrs(lu_k, piv_k, u_work[k])
            d_work[k + 1] -= l_arr[k + 1] @ u_work[k]
    return d_work, u_work, ipiv


def block_thomas_solve(
    lblk: npt.ArrayLike,
    d_factored: npt.ArrayLike,
    u_factored: npt.ArrayLike,
    ipiv: npt.ArrayLike,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Solves a block-tridiagonal system from [`block_thomas_factor`] factors.

    Args:
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`.
        d_factored: Factored diagonal blocks `(nz, nvar, nvar)` in `float64`.
        u_factored: Factored super-diagonal blocks `(nz, nvar, nvar)` in
            `float64`.
        ipiv: Pivot array `(nz, nvar)` in `int64`.
        b: Right-hand side of shape `(nz, nvar)` or `(nz * nvar,)` in
            `float64`.

    Returns:
        Solution array `x` matching the shape of `b` in `float64`.
    """
    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(
        lblk, d_factored, u_factored
    )
    piv = np.asarray(ipiv, dtype=np.int64)
    rhs = np.asarray(b, dtype=np.float64)
    is_1d = rhs.ndim == 1
    if is_1d:
        if rhs.shape != (nz * nvar,):
            raise ValueError("1D rhs b must have shape (nz * nvar,).")
        work = rhs.reshape(nz, nvar).copy()
    elif rhs.shape == (nz, nvar):
        work = rhs.copy()
    else:
        raise ValueError("b must have shape (nz, nvar) or (nz * nvar,).")

    for k in range(nz):
        if k > 0:
            work[k] -= l_arr[k] @ work[k - 1]
        work[k] = dense.dgetrs(d_arr[k], piv[k], work[k])
    for k in range(nz - 2, -1, -1):
        work[k] -= u_arr[k] @ work[k + 1]
    return work.reshape(-1) if is_1d else work


def solve_block_tridiagonal(
    lblk: npt.ArrayLike,
    dblk: npt.ArrayLike,
    ublk: npt.ArrayLike,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Factors and solves a block-tridiagonal linear system $A x = b$.

    Args:
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`.
        dblk: Diagonal blocks `(nz, nvar, nvar)` in `float64`.
        ublk: Super-diagonal blocks `(nz, nvar, nvar)` in `float64`.
        b: Right-hand side `(nz, nvar)` or `(nz * nvar,)` in `float64`.

    Returns:
        Solution array `x` matching the shape of `b` in `float64`.
    """
    d_fac, u_fac, ipiv = block_thomas_factor(lblk, dblk, ublk)
    return block_thomas_solve(lblk, d_fac, u_fac, ipiv, b)


def _thomas_fwd_step(
    carry: tuple[jax.Array, jax.Array],
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Forward block Thomas elimination step for `jax.lax.scan`."""
    u_prev, y_prev = carry
    l_k, d_k, u_k, b_k = inputs
    d_hat = d_k - l_k @ u_prev
    rhs_k = b_k - l_k @ y_prev
    u_hat = jnp.linalg.solve(d_hat, u_k)
    y_hat = jnp.linalg.solve(d_hat, rhs_k)
    return (u_hat, y_hat), (u_hat, y_hat)


def _thomas_bwd_step(
    x_next: jax.Array,
    inputs: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Backward block Thomas substitution step for `jax.lax.scan`."""
    u_hat_k, y_hat_k = inputs
    x_k = y_hat_k - u_hat_k @ x_next
    return x_k, x_k


def solve_block_tridiagonal_jax(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Differentiable block-tridiagonal Thomas solver in `float64` via `scan`.

    Args:
        lblk: Sub-diagonal JAX array `(nz, nvar, nvar)` in `float64`.
        dblk: Diagonal JAX array `(nz, nvar, nvar)` in `float64`.
        ublk: Super-diagonal JAX array `(nz, nvar, nvar)` in `float64`.
        b: Right-hand side JAX array `(nz, nvar)` in `float64`.

    Returns:
        Solution JAX array `x` of shape `(nz, nvar)` in `float64`,
        differentiable via `jax.grad`, `jax.jacfwd`, and `jax.jacrev`.
    """
    l64 = jnp.asarray(lblk, dtype=jnp.float64)
    d64 = jnp.asarray(dblk, dtype=jnp.float64)
    u64 = jnp.asarray(ublk, dtype=jnp.float64)
    b64 = jnp.asarray(b, dtype=jnp.float64)

    u_hat_0 = jnp.linalg.solve(d64[0], u64[0])
    y_hat_0 = jnp.linalg.solve(d64[0], b64[0])

    _, (u_hats_rest, y_hats_rest) = jax.lax.scan(
        _thomas_fwd_step,
        (u_hat_0, y_hat_0),
        (l64[1:], d64[1:], u64[1:], b64[1:]),
    )
    u_hats = jnp.concatenate([u_hat_0[None, ...], u_hats_rest], axis=0)
    y_hats = jnp.concatenate([y_hat_0[None, ...], y_hats_rest], axis=0)

    _, x_rev = jax.lax.scan(
        _thomas_bwd_step,
        y_hats[-1],
        (u_hats[:-1], y_hats[:-1]),
        reverse=True,
    )
    return jnp.concatenate([x_rev, y_hats[-1:]], axis=0)
