# SPDX-License-Identifier: LGPL-3.0-or-later
"""Four-stage $5 \\times 5$ block-tridiagonal solver benchmark (`tridiag_fast`).

Evaluates block-tridiagonal Henyey linear system factorization and solve
across the Three-Tier Hybrid Architecture:
- Stage 0: Zero-copy `flexmesa._rust.solve_block_tridiagonal` (1-thread
  block-Thomas in Safe Rust).
- Stage 1: Zero-copy `flexmesa._rust.solve_block_tridiag_bcyclic` (4-thread
  persistent `rayon` Hirshman et al. 2010 odd-even cyclic reduction).
- Stage 2: XLA `@jax.jit` fused `solve_block_tridiag_pcr` ($\\lceil \\log_2
  N_{\\mathrm{zones}} \\rceil$ vectorized Parallel Cyclic Reduction).
- Stage 3: Batched `@jax.jit(jax.vmap)` `solve_block_tridiag_pcr` across
  `STAGE3_BATCH_SIZE` simultaneous stellar systems.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa import mtx as _mtx
from flexmesa.bench import harness as _h

_tridiag_jit = jax.jit(_mtx.solve_block_tridiag_pcr)
_tridiag_batch_jit = jax.jit(
    jax.vmap(_mtx.solve_block_tridiag_pcr, in_axes=(0, 0, 0, 0))
)


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
    l_f = _dispatch.as_c_f64(lblk)
    d_f = _dispatch.as_c_f64(dblk)
    u_f = _dispatch.as_c_f64(ublk)
    r_f = _dispatch.as_c_f64(rhs)

    stage0, out0 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.solve_block_tridiagonal(5, nz, l_f, d_f, u_f, r_f),
    )
    stage1, out1 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.solve_block_tridiag_bcyclic(
            5, nz, l_f, d_f, u_f, r_f, True
        ),
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
