# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 1 four-stage performance benchmark suite (`flexmesa.bench`).

Evaluates the Wave 1 leaf utilities and microphysics kernels (`auto_diff`,
`interp_2d`, `mtx`, `neu`, and `turb`) across four progressive execution
stages in the Three-Tier Hybrid Architecture (`flexmesa._rust` + JAX `float64`):

- **Stage 0 (Sequential Safe Rust PyO3 Baseline)**: Zero-copy single-thread
  `flexmesa._rust` execution (`parallel=False`).
- **Stage 1 (Parallel 4-Thread `rayon` Safe Rust PyO3)**: Zero-copy 4-thread
  persistent `rayon` `flexmesa._rust` execution (`parallel=True`, and Hirshman
  et al. 2010 `bcyclic` for `mtx_tridiag`).
- **Stage 2 (Fused / JIT Kernel Execution)**: XLA-compiled `@jax.jit` fused
  kernels (recording both first-call compilation latency and warm steady-state
  execution latency), including $\\lceil \\log_2 N_{\\mathrm{zones}} \\rceil$
  Parallel Cyclic Reduction (`solve_block_tridiag_pcr`) for `mtx_tridiag`.
- **Stage 3 (Batched / Multi-Profile Execution)**: Batched `jax.jit(jax.vmap)`
  execution across `STAGE3_BATCH_SIZE` stellar profiles `(B, N_zones)` on the
  active XLA device (amortized per profile).
"""

from __future__ import annotations

from flexmesa.bench import harness as _harness
from flexmesa.bench import kernels_micro as _micro
from flexmesa.bench import kernels_solver as _solver
from flexmesa.bench import kernels_wave2 as _wave2
from flexmesa.bench import kernels_wave3a as _wave3a

STAGE3_BATCH_SIZE = _harness.STAGE3_BATCH_SIZE
BenchRecord = _harness.BenchRecord
BicubTable = _harness.BicubTable
LatencyStats = _harness.LatencyStats
ThermoGrid = _harness.ThermoGrid
make_block_tridiag_fixture = _harness.make_block_tridiag_fixture
max_err3 = _harness.max_err3
max_rel_diff = _harness.max_rel_diff
measure_and_capture = _harness.measure_and_capture
measure_stage3_batch = _harness.measure_stage3_batch
measure_us = _harness.measure_us

bench_autodiff = _micro.bench_autodiff
bench_interp2d = _micro.bench_interp2d
bench_neu = _micro.bench_neu

bench_mtx_block_tridiag = _solver.bench_mtx_block_tridiag
bench_turb_mlt = _solver.bench_turb_mlt
bench_turb_tdc = _solver.bench_turb_tdc
run_all_wave1_benchmarks = _solver.run_all_wave1_benchmarks

WAVE2_ISO_NAMES = _wave2.WAVE2_ISO_NAMES
bench_chem_comp = _wave2.bench_chem_comp
bench_colors_sed = _wave2.bench_colors_sed
bench_ion_4d = _wave2.bench_ion_4d
bench_ion_fe56 = _wave2.bench_ion_fe56
make_wave2_composition_profile = _wave2.make_wave2_composition_profile
run_all_wave2_benchmarks = _wave2.run_all_wave2_benchmarks

bench_eos_helm = _wave3a.bench_eos_helm
bench_eos_ideal = _wave3a.bench_eos_ideal
bench_eos_pc = _wave3a.bench_eos_pc
bench_eos_skye = _wave3a.bench_eos_skye
run_all_wave3a_benchmarks = _wave3a.run_all_wave3a_benchmarks

__all__: list[str] = [
    "STAGE3_BATCH_SIZE",
    "WAVE2_ISO_NAMES",
    "BenchRecord",
    "BicubTable",
    "LatencyStats",
    "ThermoGrid",
    "bench_autodiff",
    "bench_chem_comp",
    "bench_colors_sed",
    "bench_eos_helm",
    "bench_eos_ideal",
    "bench_eos_pc",
    "bench_eos_skye",
    "bench_interp2d",
    "bench_ion_4d",
    "bench_ion_fe56",
    "bench_mtx_block_tridiag",
    "bench_neu",
    "bench_turb_mlt",
    "bench_turb_tdc",
    "make_block_tridiag_fixture",
    "make_wave2_composition_profile",
    "max_err3",
    "max_rel_diff",
    "measure_and_capture",
    "measure_stage3_batch",
    "measure_us",
    "run_all_wave1_benchmarks",
    "run_all_wave2_benchmarks",
    "run_all_wave3a_benchmarks",
]
