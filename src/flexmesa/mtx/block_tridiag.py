# SPDX-License-Identifier: LGPL-3.0-or-later
"""Block-tridiagonal matrix-vector products and linear solvers (`mtx`).

Represents a block-tridiagonal matrix across `nz` shells and `nvar` variables
per shell via three arrays `lblk`, `dblk`, and `ublk` of shape
`(nz, nvar, nvar)` in `float64`, where shell `k` has sub-diagonal block
`lblk[k]` ($k > 0$), diagonal block `dblk[k]`, and super-diagonal block
`ublk[k]` ($k + 1 < n_z$). Supports Safe Rust sequential Thomas and Hirshman et
al. (2010) `bcyclic` odd-even cyclic reduction via `flexmesa._rust`, as well as
JAX-differentiable solvers (`solver_mode="pcr"`, `"scan"`, or `"rust_ffi"`).
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa.mtx import dense, pcr


def _validate_blocks(
    lblk: npt.ArrayLike, dblk: npt.ArrayLike, ublk: npt.ArrayLike
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int,
    int,
]:
    """Validates `(nz, nvar, nvar)` block-tridiagonal arrays in `float64`."""
    l_a, d_a, u_a = (
        np.asarray(lblk, dtype=np.float64),
        np.asarray(dblk, dtype=np.float64),
        np.asarray(ublk, dtype=np.float64),
    )
    if (
        d_a.ndim != 3
        or 0 in d_a.shape[:2]
        or d_a.shape[1] != d_a.shape[2]
        or l_a.shape != d_a.shape
        or u_a.shape != d_a.shape
    ):
        raise ValueError(
            "lblk, dblk, ublk must share non-empty shape (nz, nvar, nvar)."
        )
    if not (
        np.all(np.isfinite(l_a))
        and np.all(np.isfinite(d_a))
        and np.all(np.isfinite(u_a))
    ):
        raise ValueError("Block-tridiagonal arrays contain NaN or Inf.")
    return l_a, d_a, u_a, int(d_a.shape[0]), int(d_a.shape[1])


def _as_block_vec(
    vec: npt.ArrayLike, nz: int, nvar: int, name: str, *, copy: bool = False
) -> tuple[npt.NDArray[np.float64], bool]:
    """Reshapes and validates a 1D `(nz * nvar,)` or 2D `(nz, nvar)` vector."""
    arr = np.asarray(vec, dtype=np.float64)
    is_1d = arr.ndim == 1
    if is_1d and arr.shape == (nz * nvar,):
        out = arr.reshape(nz, nvar)
    elif not is_1d and arr.shape == (nz, nvar):
        out = arr
    else:
        raise ValueError(f"{name} must have shape (nz, nvar) or (nz * nvar,).")
    return (out.copy() if copy else out), is_1d


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
    b_2d, is_1d = _as_block_vec(b, nz, nvar, "b")

    prod: npt.NDArray[np.float64] = np.einsum("kij,kj->ki", d_arr, b_2d)
    if nz > 1:
        prod[1:] += np.einsum("kij,kj->ki", l_arr[1:], b_2d[:-1])
        prod[:-1] += np.einsum("kij,kj->ki", u_arr[:-1], b_2d[1:])
    return prod.reshape(-1) if is_1d else prod


def block_multiply_xa(
    lblk: npt.ArrayLike,
    dblk: npt.ArrayLike,
    ublk: npt.ArrayLike,
    x: npt.ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> npt.NDArray[np.float64]:
    """Evaluates left product $b = x^{\\top} A$ for a block-tridiagonal system.

    Args:
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal blocks `(nz, nvar, nvar)` in `float64` (dimensionless).
        ublk: Super-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        x: Left vector of shape `(nz, nvar)` or `(nz * nvar,)` in `float64`
            (dimensionless).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Row-vector product $b = x^{\\top} A$ matching the shape of `x`.

    Raises:
        ValueError: If block or vector dimensions do not match.
    """
    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(lblk, dblk, ublk)
    x_2d, is_1d = _as_block_vec(x, nz, nvar, "x")

    if _dispatch.resolve_backend(backend, lblk, dblk, ublk, x) == "rust":
        out_r = _rust.block_multiply_xa(
            nvar,
            nz,
            _dispatch.as_c_f64(l_arr),
            _dispatch.as_c_f64(d_arr),
            _dispatch.as_c_f64(u_arr),
            _dispatch.as_c_f64(x_2d),
        )
        return out_r.reshape(-1) if is_1d else out_r

    out: npt.NDArray[np.float64] = np.einsum("ki,kij->kj", x_2d, d_arr)
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
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal blocks `(nz, nvar, nvar)` in `float64` (dimensionless).
        ublk: Super-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).

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
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        d_factored: Factored diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        u_factored: Factored super-diagonal blocks `(nz, nvar, nvar)` in
            `float64` (dimensionless).
        ipiv: Pivot array `(nz, nvar)` in `int64` (dimensionless).
        b: Right-hand side of shape `(nz, nvar)` or `(nz * nvar,)` in
            `float64` (dimensionless).

    Returns:
        Solution array `x` matching the shape of `b` in `float64`.

    Raises:
        ValueError: If block dimensions mismatch or a diagonal block is
            singular.
    """
    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(
        lblk, d_factored, u_factored
    )
    piv = np.asarray(ipiv, dtype=np.int64)
    work, is_1d = _as_block_vec(b, nz, nvar, "b", copy=True)

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
    *,
    method: Literal["thomas", "bcyclic"] = "thomas",
    parallel: bool = False,
    backend: _dispatch.Backend = "auto",
) -> npt.NDArray[np.float64]:
    """Factors and solves a block-tridiagonal linear system $A x = b$.

    Args:
        lblk: Sub-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal blocks `(nz, nvar, nvar)` in `float64` (dimensionless).
        ublk: Super-diagonal blocks `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        b: Right-hand side `(nz, nvar)` or `(nz * nvar,)` in `float64`
            (dimensionless).
        method: Solver algorithm (`"thomas"` sequential block-Thomas or
            `"bcyclic"` Hirshman et al. 2010 odd-even cyclic reduction).
        parallel: When `method == "bcyclic"`, whether to run the reduction
            levels across the persistent 4-thread `rayon` pool.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Solution array `x` matching the shape of `b` in `float64`
        (dimensionless).

    Raises:
        ValueError: If block dimensions mismatch, inputs are non-finite, or a
            diagonal block is singular.
    """
    if method not in ("thomas", "bcyclic"):
        msg = f"Invalid method {method!r}; expected 'thomas' or 'bcyclic'."
        raise ValueError(msg)

    l_arr, d_arr, u_arr, nz, nvar = _validate_blocks(lblk, dblk, ublk)
    b_2d, is_1d = _as_block_vec(b, nz, nvar, "b")
    if not np.all(np.isfinite(b_2d)):
        raise ValueError("Right-hand side b contains NaN or Inf.")

    if _dispatch.resolve_backend(backend, lblk, dblk, ublk, b) == "rust":
        l_c = _dispatch.as_c_f64(l_arr)
        d_c = _dispatch.as_c_f64(d_arr)
        u_c = _dispatch.as_c_f64(u_arr)
        b_c = _dispatch.as_c_f64(b_2d)
        if method == "bcyclic":
            x_r = _rust.solve_block_tridiag_bcyclic(
                nvar, nz, l_c, d_c, u_c, b_c, parallel
            )
        else:
            x_r = _rust.solve_block_tridiagonal(nvar, nz, l_c, d_c, u_c, b_c)
        return x_r.reshape(-1) if is_1d else x_r

    mode: pcr.SolverMode = "pcr" if method == "bcyclic" else "scan"
    x_j = np.asarray(
        solve_block_tridiagonal_jax(
            jnp.asarray(l_arr),
            jnp.asarray(d_arr),
            jnp.asarray(u_arr),
            jnp.asarray(b_2d),
            solver_mode=mode,
        ),
        dtype=np.float64,
    )
    return x_j.reshape(-1) if is_1d else x_j


def _thomas_fwd_step(
    carry: tuple[jax.Array, jax.Array],
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Forward block Thomas elimination step for `jax.lax.scan`."""
    u_prev, y_prev = carry
    l_k, d_k, u_k, b_k = inputs
    d_hat = d_k - l_k @ u_prev
    rhs_k = b_k - l_k @ y_prev
    uy_hat = jnp.linalg.solve(
        d_hat, jnp.concatenate([u_k, rhs_k[:, None]], axis=-1)
    )
    pair = (uy_hat[:, :-1], uy_hat[:, -1])
    return pair, pair


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
    *,
    solver_mode: pcr.SolverMode = "pcr",
) -> jax.Array:
    """Differentiable block-tridiagonal linear solver in JAX `float64`.

    Args:
        lblk: Sub-diagonal JAX array `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal JAX array `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        ublk: Super-diagonal JAX array `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        b: Right-hand side JAX array `(nz, nvar)` in `float64` (dimensionless).
        solver_mode: JAX solver strategy (`"pcr"` vectorized Parallel Cyclic
            Reduction, `"scan"` sequential `jax.lax.scan` block-Thomas, or
            `"rust_ffi"` Safe Rust `bcyclic` with analytical `custom_vjp`).

    Returns:
        Solution JAX array `x` of shape `(nz, nvar)` in `float64`
        (dimensionless), differentiable via `jax.grad`, `jax.jacfwd`, and
        `jax.jacrev`.

    Raises:
        ValueError: If `solver_mode` is not `"pcr"`, `"scan"`, or `"rust_ffi"`.
    """
    if solver_mode == "pcr":
        return pcr.solve_block_tridiag_pcr(lblk, dblk, ublk, b)
    if solver_mode == "rust_ffi":
        return pcr.solve_block_tridiag_rust_ffi(lblk, dblk, ublk, b)
    if solver_mode != "scan":
        msg = (
            f"Invalid solver_mode {solver_mode!r}; expected 'pcr', 'scan', "
            "or 'rust_ffi'."
        )
        raise ValueError(msg)

    l64, d64, u64, b64, nz, _ = pcr.validate_jax_blocks(lblk, dblk, ublk, b)
    uy_0: jax.Array = jnp.linalg.solve(
        d64[0], jnp.concatenate([u64[0], b64[0, :, None]], axis=-1)
    )
    u_hat_0, y_hat_0 = uy_0[:, :-1], uy_0[:, -1]
    if nz == 1:
        return y_hat_0[None, :]

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
