# SPDX-License-Identifier: LGPL-3.0-or-later
"""Four-stage solver and convective transport benchmarks (`flexmesa.bench`).

Evaluates the Three-Tier Hybrid Architecture across Stages 0..3 for:
1. `bench_mtx_block_tridiag`: $5 \\times 5$ block-tridiagonal Henyey system
   factorization and solve (`mtx`) across `flexmesa._rust` (sequential Thomas
   Stage 0, 4-thread `rayon` Hirshman `bcyclic` Stage 1) and JAX
   `solve_block_tridiag_pcr` (`@jax.jit` Stage 2, `@jax.jit(vmap)` Stage 3).
2. `bench_turb_mlt`: Cox Mixing-Length Theory (`set_mlt` / `set_mlt_jax`) with
   first-order sensitivities across `flexmesa._rust.set_mlt_batch` (1-thread
   Stage 0, 4-thread `rayon` Stage 1) and `@jax.jit` / `@jax.jit(vmap)`
   (Stages 2..3).
3. `bench_turb_tdc`: Kuhfuss 1D Time-Dependent Convection (`set_tdc`) with
   Implicit Function Theorem sensitivities across `flexmesa._rust.set_tdc_batch`
   (Stages 0..1) and `@jax.jit` / `@jax.jit(vmap)` (Stages 2..3).
"""

from __future__ import annotations

import collections.abc

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa import turb as _turb
from flexmesa.bench import harness as _h
from flexmesa.bench import kernels_micro as _micro
from flexmesa.bench import tdc_fast as _tdc_fast
from flexmesa.bench import tridiag_fast as _tridiag_fast

bench_mtx_block_tridiag = _tridiag_fast.bench_mtx_block_tridiag


def _mlt_point_jax(s_frac: jax.Array) -> jax.Array:
    """Computes `[grad_t, d, conv_vel, d_grad_t_d_gradr]` in JAX `float64`.

    Args:
        s_frac: Normalized radial shell coordinate $s \\in [0, 1]$ (`float64`
            scalar).

    Returns:
        4-element `float64` JAX array `[grad_t, d, conv_vel, d_grad_t_d_gradr]`
        of shape `(4,)`.
    """
    t_val = 1.0e5 * (1.0 + 0.5 * s_frac)
    rho_val = 1.0e-5 * (1.0 + 0.5 * s_frac)
    grav_val = jnp.float64(_tdc_fast.GRAV_SOLAR)
    cp_val = jnp.float64(_tdc_fast.CP_IDEAL)
    p_val = rho_val * t_val * jnp.float64(_tdc_fast.RGAS_IDEAL)
    lam_val = 2.0 * p_val / (rho_val * grav_val)
    gradr_val = 0.45 + 0.25 * s_frac
    one, g04 = jnp.float64(1.0), jnp.float64(0.40)

    def _mlt_with_aux(
        gr: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        _, gt, _, cv, dv = _turb.set_mlt_jax(
            one,
            one,
            cp_val,
            grav_val,
            lam_val,
            rho_val,
            p_val,
            t_val,
            one,
            gr,
            g04,
            g04,
            2.0,
        )
        return gt, (dv, cv)

    (grad_t, (d_val, conv_vel)), d_gt_d_gr = jax.value_and_grad(
        _mlt_with_aux, has_aux=True
    )(gradr_val)
    return jnp.stack([grad_t, d_val, conv_vel, d_gt_d_gr])


_mlt_vmap = jax.vmap(_mlt_point_jax)
_mlt_jit = jax.jit(_mlt_vmap)
_mlt_batch_jit = jax.jit(jax.vmap(_mlt_vmap))


def _bench_turb_4stage(
    workload: str,
    nz: int,
    n_warmup: int,
    n_repeats: int,
    rust_batch_fn: collections.abc.Callable[
        [npt.NDArray[np.float64], bool], npt.NDArray[np.float64]
    ],
    jit_fn: collections.abc.Callable[[jax.Array], jax.Array],
    batch_jit_fn: collections.abc.Callable[[jax.Array], jax.Array],
) -> _h.BenchRecord:
    """Shared 4-stage benchmark driver for `turb_mlt` and `turb_tdc`.

    Args:
        workload: Workload identifier (`"turb_mlt"` or `"turb_tdc"`).
        nz: Number of radial shells ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.
        rust_batch_fn: Zero-copy `flexmesa._rust` batch evaluator taking
            `(s_frac, parallel)`.
        jit_fn: `@jax.jit`-compiled kernel for Stage 2.
        batch_jit_fn: `@jax.jit(jax.vmap)` batched kernel for Stage 3.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    s_np = _dispatch.as_c_f64(
        np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    )
    s_j = jnp.asarray(s_np, dtype=jnp.float64)

    stage0, out0_np = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: rust_batch_fn(s_np, False),
    )
    stage1, out1_np = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: rust_batch_fn(s_np, True),
    )
    stage2, out2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: jit_fn(s_j),
        record_first_as_compile=True,
    )

    b_sz = _h.STAGE3_BATCH_SIZE
    s_b = jnp.broadcast_to(s_j[None, :], (b_sz, nz))
    stage3, out3_j = _h.measure_stage3_batch(
        n_warmup,
        n_repeats,
        lambda: batch_jit_fn(s_b),
    )

    out2_np = np.asarray(out2_j, dtype=np.float64)
    out3_np = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0_np, out1_np),
        _h.max_rel_diff(out0_np, out2_np),
        _h.max_rel_diff(out0_np, out3_np),
    )
    return _h.BenchRecord(
        workload=workload,
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2_np[:, 0])),
    )


def bench_turb_mlt(
    grid: _h.ThermoGrid,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks Cox Mixing-Length Theory (`set_mlt`) across Stages 0..3.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    return _bench_turb_4stage(
        "turb_mlt",
        grid.n_zones,
        n_warmup,
        n_repeats,
        _rust.set_mlt_batch,
        _mlt_jit,
        _mlt_batch_jit,
    )


def bench_turb_tdc(
    grid: _h.ThermoGrid,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks Kuhfuss 1D Time-Dependent Convection (`set_tdc`) Stages 0..3.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    return _bench_turb_4stage(
        "turb_tdc",
        grid.n_zones,
        n_warmup,
        n_repeats,
        _rust.set_tdc_batch,
        _tdc_fast.tdc_jit,
        _tdc_fast.tdc_batch_jit,
    )


def run_all_wave1_benchmarks(
    zone_counts: collections.abc.Sequence[int] = (128, 1024, 4096),
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> list[_h.BenchRecord]:
    """Runs all six Wave 1 benchmark workloads across `zone_counts`.

    Args:
        zone_counts: Sequence of stellar zone counts $N_{\\mathrm{zones}}$.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        List of `BenchRecord` instances across all workloads and zone counts.
    """
    table = _h.BicubTable.create()
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        grid = _h.ThermoGrid.create(nz)
        records.append(_micro.bench_autodiff(grid, n_warmup, n_repeats))
        records.append(_micro.bench_interp2d(grid, table, n_warmup, n_repeats))
        records.append(_micro.bench_neu(grid, n_warmup, n_repeats))
        records.append(bench_mtx_block_tridiag(nz, n_warmup, n_repeats))
        records.append(bench_turb_mlt(grid, n_warmup, n_repeats))
        records.append(bench_turb_tdc(grid, n_warmup, n_repeats))
    return records
