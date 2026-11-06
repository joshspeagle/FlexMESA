# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parallel Cyclic Reduction (PCR) and Rust FFI block-tridiagonal solvers.

Provides two high-throughput, differentiable block-tridiagonal solvers for
Tier C coupled stellar structure and burn/mix systems ($A x = b$ across $n_z$
shells and $n_{\\mathrm{var}}$ variables per shell):

1. `solve_block_tridiag_pcr` (`solver_mode="pcr"`):
   Vectorized Hockney & Jesshope (1981) / Hirshman et al. (2010) parallel
   block cyclic reduction in pure JAX `float64`. Replaces the $2 n_z$
   sequential `jax.lax.scan` steps of block-Thomas elimination with
   $\\lceil \\log_2 n_z \\rceil$ parallel block-row reduction steps, each
   performing a single batched `(nz, nvar, nvar)` solve and two fused
   matrix-matrix updates across all $n_z$ zones simultaneously on GPU or CPU.
2. `solve_block_tridiag_rust_ffi` (`solver_mode="rust_ffi"`):
   Zero-copy host callback to Safe Rust's Hirshman et al. (2010)
   `solve_block_tridiag_bcyclic` via `jax.pure_callback`, paired with an
   analytical Implicit Function Theorem `@jax.custom_vjp` rule that solves
   the transpose block-tridiagonal adjoint system $A^\\top \\lambda = \\bar{x}$.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust

SolverMode = Literal["pcr", "scan", "rust_ffi"]


def validate_jax_blocks(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, int]:
    """Coerces `(lblk, dblk, ublk, b)` to JAX `float64` and validates shapes.

    Args:
        lblk: Sub-diagonal blocks of shape `(nz, nvar, nvar)` (dimensionless).
        dblk: Diagonal blocks of shape `(nz, nvar, nvar)` (dimensionless).
        ublk: Super-diagonal blocks of shape `(nz, nvar, nvar)` (dimensionless).
        b: Right-hand side of shape `(nz, nvar)` (dimensionless).

    Returns:
        Six-tuple `(l64, d64, u64, b64, nz, nvar)` in `jnp.float64`.

    Raises:
        ValueError: If array ranks or dimensions are incompatible.
    """
    l64 = jnp.asarray(lblk, dtype=jnp.float64)
    d64 = jnp.asarray(dblk, dtype=jnp.float64)
    u64 = jnp.asarray(ublk, dtype=jnp.float64)
    b64 = jnp.asarray(b, dtype=jnp.float64)
    if (
        d64.ndim != 3
        or l64.shape != d64.shape
        or u64.shape != d64.shape
        or d64.shape[0] == 0
        or d64.shape[1] == 0
        or d64.shape[1] != d64.shape[2]
        or b64.shape != (d64.shape[0], d64.shape[1])
    ):
        msg = (
            "Block-tridiagonal solver requires lblk, dblk, ublk of shape "
            "(nz, nvar, nvar) and b of shape (nz, nvar) with nz >= 1, nvar >= 1"
        )
        raise ValueError(msg)
    return l64, d64, u64, b64, int(d64.shape[0]), int(d64.shape[1])


def solve_block_tridiag_pcr(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Solves a block-tridiagonal system in JAX `float64` via $\\log_2$ PCR.

    At reduction level $m \\in \\{0, \\dots, \\lceil \\log_2 n_z \\rceil - 1\\}$
    with stride $s = 2^m$, normalizes each block row by $D_k^{(m)}$ in a
    single batched `(nz, nvar, nvar)` solve:

    $$\\hat{L}_k = \\left(D_k^{(m)}\\right)^{-1} L_k^{(m)}, \\quad
      \\hat{U}_k = \\left(D_k^{(m)}\\right)^{-1} U_k^{(m)}, \\quad
      \\hat{b}_k = \\left(D_k^{(m)}\\right)^{-1} b_k^{(m)},$$

    and eliminates couplings at distance $\\pm s$ across all $n_z$ shells in
    parallel via two fused batched matrix-matrix products:

    $$D_k^{(m+1)} = D_k^{(m)} - L_k^{(m)} \\hat{U}_{k-s}
      - U_k^{(m)} \\hat{L}_{k+s},$$
    $$b_k^{(m+1)} = b_k^{(m)} - L_k^{(m)} \\hat{b}_{k-s}
      - U_k^{(m)} \\hat{b}_{k+s},$$
    $$L_k^{(m+1)} = -L_k^{(m)} \\hat{L}_{k-s}, \\quad
      U_k^{(m+1)} = -U_k^{(m)} \\hat{U}_{k+s}.$$

    After $P = \\lceil \\log_2 n_z \\rceil$ steps, $2^P \\ge n_z$ so all
    off-diagonal couplings vanish and $x_k = (D_k^{(P)})^{-1} b_k^{(P)}$.

    Args:
        lblk: Sub-diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        ublk: Super-diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        b: Right-hand side of shape `(nz, nvar)` in `float64` (dimensionless).

    Returns:
        Solution array `x` of shape `(nz, nvar)` in `float64` (dimensionless),
        compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`, and
        `jax.jacrev`.

    Raises:
        ValueError: If input array ranks or dimensions are incompatible.
    """
    l_cur, d_cur, u_cur, b_cur, nz, nvar = validate_jax_blocks(
        lblk, dblk, ublk, b
    )
    if nz == 1:
        x1: jax.Array = jnp.linalg.solve(d_cur[0], b_cur[0])[None, :]
        return x1

    n_levels = (nz - 1).bit_length()
    n_cols = 2 * nvar + 1
    for level in range(n_levels):
        stride = 1 << level
        rhs_pack = jnp.concatenate([l_cur, u_cur, b_cur[..., None]], axis=-1)
        solved = jnp.linalg.solve(d_cur, rhs_pack)
        z_pack = jnp.zeros((stride, nvar, n_cols), dtype=jnp.float64)
        s_prev = jnp.concatenate([z_pack, solved[:-stride]], axis=0)
        s_next = jnp.concatenate([solved[stride:], z_pack], axis=0)

        l_prod = l_cur @ s_prev
        u_prod = u_cur @ s_next
        d_cur = d_cur - l_prod[:, :, nvar : 2 * nvar] - u_prod[:, :, :nvar]
        b_cur = b_cur - l_prod[:, :, 2 * nvar] - u_prod[:, :, 2 * nvar]
        if level + 1 < n_levels:
            l_cur = -l_prod[:, :, :nvar]
            u_cur = -u_prod[:, :, nvar : 2 * nvar]

    x_out: jax.Array = jnp.linalg.solve(d_cur, b_cur[..., None])[..., 0]
    return x_out


def _rust_bcyclic_host_cb(
    lblk: npt.NDArray[np.float64],
    dblk: npt.NDArray[np.float64],
    ublk: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Host callback invoking `flexmesa._rust.solve_block_tridiag_bcyclic`."""
    nz = int(dblk.shape[0])
    nvar = int(dblk.shape[1])
    return _rust.solve_block_tridiag_bcyclic(
        nvar,
        nz,
        _dispatch.as_c_f64(lblk),
        _dispatch.as_c_f64(dblk),
        _dispatch.as_c_f64(ublk),
        _dispatch.as_c_f64(b),
        nz >= 64,
    )


def _solve_rust_ffi_primal(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluates the primal block-tridiagonal solve via `jax.pure_callback`."""
    l64, d64, u64, b64, _, _ = validate_jax_blocks(lblk, dblk, ublk, b)
    out_shape = jax.ShapeDtypeStruct(  # type: ignore[no-untyped-call]
        b64.shape, jnp.float64
    )
    out: jax.Array = jax.pure_callback(
        _rust_bcyclic_host_cb,
        out_shape,
        l64,
        d64,
        u64,
        b64,
        vmap_method="sequential",
    )
    return out, l64, d64, u64


@jax.custom_vjp
def solve_block_tridiag_rust_ffi(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Solves $A x = b$ via Safe Rust `bcyclic` FFI with analytical IFT VJP.

    Executes the forward pass in compiled Safe Rust
    (`flexmesa._rust.solve_block_tridiag_bcyclic`) and computes reverse
    adjoints via the Implicit Function Theorem transpose system
    $A^\\top \\lambda = \\bar{x}$:

    $$\\bar{b}_k = \\lambda_k, \\quad
      \\bar{D}_k = -\\lambda_k x_k^\\top, \\quad
      \\bar{L}_k = -\\lambda_k x_{k-1}^\\top, \\quad
      \\bar{U}_k = -\\lambda_k x_{k+1}^\\top.$$

    Args:
        lblk: Sub-diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        dblk: Diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        ublk: Super-diagonal blocks of shape `(nz, nvar, nvar)` in `float64`
            (dimensionless).
        b: Right-hand side of shape `(nz, nvar)` in `float64` (dimensionless).

    Returns:
        Solution array `x` of shape `(nz, nvar)` in `float64` (dimensionless).
    """
    return _solve_rust_ffi_primal(lblk, dblk, ublk, b)[0]


def _rust_ffi_fwd(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    b: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Forward pass for `solve_block_tridiag_rust_ffi` custom VJP."""
    x, l64, d64, u64 = _solve_rust_ffi_primal(lblk, dblk, ublk, b)
    return x, (l64, d64, u64, x)


def _rust_ffi_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent_x: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Backward pass solving the transpose block-tridiagonal system."""
    l64, d64, u64, x = residuals
    g64 = jnp.asarray(cotangent_x, dtype=jnp.float64)

    nvar = int(d64.shape[1])
    z1_blk = jnp.zeros((1, nvar, nvar), dtype=jnp.float64)
    l_t = jnp.concatenate([z1_blk, jnp.swapaxes(u64[:-1], -1, -2)], axis=0)
    d_t = jnp.swapaxes(d64, -1, -2)
    u_t = jnp.concatenate([jnp.swapaxes(l64[1:], -1, -2), z1_blk], axis=0)

    lam = solve_block_tridiag_rust_ffi(l_t, d_t, u_t, g64)
    z1_vec = jnp.zeros((1, nvar), dtype=jnp.float64)
    x_prev = jnp.concatenate([z1_vec, x[:-1]], axis=0)
    x_next = jnp.concatenate([x[1:], z1_vec], axis=0)

    neg_lam = -lam[:, :, None]
    grad_l = neg_lam * x_prev[:, None, :]
    grad_d = neg_lam * x[:, None, :]
    grad_u = neg_lam * x_next[:, None, :]
    return grad_l, grad_d, grad_u, lam


solve_block_tridiag_rust_ffi.defvjp(_rust_ffi_fwd, _rust_ffi_bwd)

__all__ = [
    "SolverMode",
    "solve_block_tridiag_pcr",
    "solve_block_tridiag_rust_ffi",
    "validate_jax_blocks",
]
