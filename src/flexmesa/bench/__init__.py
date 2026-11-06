# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 1 four-stage performance benchmark suite (`flexmesa.bench`).

Evaluates the Wave 1 leaf utilities and microphysics kernels (`auto_diff`,
`interp_2d`, `mtx`, `neu`, and `turb`) across four progressive optimization
stages in Python/NumPy and JAX `float64`:

- **Stage 0 (Idiomatic Scalar Baseline)**: Zone-by-zone scalar Python loops.
- **Stage 1 (Contiguous Structure-of-Arrays Layout)**: Vectorized NumPy or
  un-fused `jax.vmap` evaluation over contiguous `float64` arrays of shape
  `(N_zones,)`.
- **Stage 2 (Fused / JIT Kernel Execution)**: XLA-compiled `@jax.jit` fused
  kernels (recording both first-call compilation latency and warm steady-state
  execution latency) and hoisted-invariant NumPy kernels.
- **Stage 3 (Batched / Multi-Profile Execution)**: Batched `jax.jit(jax.vmap)`
  execution across multiple stellar profiles `(B, N_zones)` on the active XLA
  device (amortized per profile).
"""

from __future__ import annotations

from flexmesa.bench import harness as _harness
from flexmesa.bench import kernels_micro as _micro
from flexmesa.bench import kernels_solver as _solver

BenchRecord = _harness.BenchRecord
BicubTable = _harness.BicubTable
LatencyStats = _harness.LatencyStats
ThermoGrid = _harness.ThermoGrid
make_block_tridiag_fixture = _harness.make_block_tridiag_fixture
max_rel_diff = _harness.max_rel_diff
measure_and_capture = _harness.measure_and_capture
measure_us = _harness.measure_us

bench_autodiff = _micro.bench_autodiff
bench_interp2d = _micro.bench_interp2d
bench_neu = _micro.bench_neu

bench_mtx_block_tridiag = _solver.bench_mtx_block_tridiag
bench_turb_mlt = _solver.bench_turb_mlt
bench_turb_tdc = _solver.bench_turb_tdc
run_all_wave1_benchmarks = _solver.run_all_wave1_benchmarks

__all__: list[str] = [
    "BenchRecord",
    "BicubTable",
    "LatencyStats",
    "ThermoGrid",
    "bench_autodiff",
    "bench_interp2d",
    "bench_mtx_block_tridiag",
    "bench_neu",
    "bench_turb_mlt",
    "bench_turb_tdc",
    "make_block_tridiag_fixture",
    "max_rel_diff",
    "measure_and_capture",
    "measure_us",
    "run_all_wave1_benchmarks",
]
