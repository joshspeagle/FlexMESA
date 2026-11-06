# SPDX-License-Identifier: LGPL-3.0-or-later
"""Four-stage $5 \\times 5$ block-tridiagonal solver benchmark (`tridiag_fast`).

Evaluates block-tridiagonal Henyey linear system factorization and solve
across:
- Stage 0: Pure-Python `solve_block_tridiagonal` (`dgetrf`/`dgetrs`).
- Stage 1: Contiguous pre-allocated LAPACK `np.linalg.solve` Thomas sweep.
- Stage 2: XLA `@jax.jit` fused `solve_block_tridiagonal_jax` (`lax.scan`).
- Stage 3: Batched `@jax.jit(jax.vmap)` across `STAGE3_BATCH_SIZE` systems.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import mtx as _mtx
from flexmesa.bench import harness as _h

_tridiag_jit = jax.jit(_mtx.solve_block_tridiagonal_jax)
_tridiag_batch_jit = jax.jit(
    jax.vmap(_mtx.solve_block_tridiagonal_jax, in_axes=(0, 0, 0, 0))
)


def fast_block_thomas_numpy(
    lblk: npt.NDArray[np.float64],
    dblk: npt.NDArray[np.float64],
    ublk: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    d_w: npt.NDArray[np.float64],
    u_w: npt.NDArray[np.float64],
    x_w: npt.NDArray[np.float64],
) -> None:
    """Solves a $5 \\times 5$ block-tridiagonal system in-place via LAPACK.

    Args:
        lblk: Sub-diagonal blocks `(nz, 5, 5)` in `float64`.
        dblk: Diagonal blocks `(nz, 5, 5)` in `float64`.
        ublk: Super-diagonal blocks `(nz, 5, 5)` in `float64`.
        rhs: Right-hand side `(nz, 5)` in `float64`.
        d_w: Pre-allocated diagonal workspace `(nz, 5, 5)`.
        u_w: Pre-allocated super-diagonal workspace `(nz, 5, 5)`.
        x_w: Pre-allocated solution buffer `(nz, 5)`.
    """
    nz = lblk.shape[0]
    np.copyto(d_w, dblk)
    np.copyto(u_w, ublk)
    np.copyto(x_w, rhs)
    for k in range(nz):
        if k > 0:
            d_w[k] -= lblk[k] @ u_w[k - 1]
            x_w[k] -= lblk[k] @ x_w[k - 1]
        if k + 1 < nz:
            u_w[k] = np.linalg.solve(d_w[k], u_w[k])
        x_w[k] = np.linalg.solve(d_w[k], x_w[k])
    for k in range(nz - 2, -1, -1):
        x_w[k] -= u_w[k] @ x_w[k + 1]


def bench_mtx_block_tridiag(
    nz: int,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks $5 \\times 5$ block-tridiagonal solves across Stages 0..3.

    Args:
        nz: Number of stellar shells ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    lblk, dblk, ublk, rhs = _h.make_block_tridiag_fixture(nz)
    stage0, out0 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _mtx.solve_block_tridiagonal(lblk, dblk, ublk, rhs),
    )

    d_w = np.empty_like(dblk)
    u_w = np.empty_like(ublk)
    out1 = np.empty_like(rhs)
    stage1 = _h.measure_us(
        n_warmup,
        n_repeats,
        lambda: fast_block_thomas_numpy(lblk, dblk, ublk, rhs, d_w, u_w, out1),
    )

    l_j = jnp.asarray(lblk, dtype=jnp.float64)
    d_j = jnp.asarray(dblk, dtype=jnp.float64)
    u_j = jnp.asarray(ublk, dtype=jnp.float64)
    r_j = jnp.asarray(rhs, dtype=jnp.float64)
    stage2, out2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _tridiag_jit(l_j, d_j, u_j, r_j),
        record_first_as_compile=True,
    )

    b_sz = _h.STAGE3_BATCH_SIZE
    l_b = jnp.broadcast_to(l_j[None, ...], (b_sz, *l_j.shape))
    d_b = jnp.broadcast_to(d_j[None, ...], (b_sz, *d_j.shape))
    u_b = jnp.broadcast_to(u_j[None, ...], (b_sz, *u_j.shape))
    r_b = jnp.broadcast_to(r_j[None, ...], (b_sz, *r_j.shape))
    stage3, out3_j = _h.measure_stage3_batch(
        n_warmup,
        n_repeats,
        lambda: _tridiag_batch_jit(l_b, d_b, u_b, r_b),
    )

    out2 = np.asarray(out2_j, dtype=np.float64)
    out3 = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0, out1),
        _h.max_rel_diff(out0, out2),
        _h.max_rel_diff(out0, out3),
    )
    return _h.BenchRecord(
        workload="mtx_tridiag",
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2)),
    )
