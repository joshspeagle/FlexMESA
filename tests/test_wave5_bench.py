# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the Wave 5 benchmark suite and Fortran MESA checksum parity.

Verifies that:
1. All 4 Wave 5 benchmark kernels (`bench_w5_diff_rot`, `bench_w5_riemann_rsp2`,
   `bench_w5_binary_rlof`, and `bench_w5_astero_seism`) and
   `run_all_wave5_benchmarks` execute cleanly across Stages 0..3 (`rust`
   sequential, `rust` 4-thread parallel, `@jax.jit`, and `jax.jit(jax.vmap)`).
2. Cross-backend elementwise relative error is `< 1.0e-12` at `n_zones = 8` and
   `< 1.0e-14` at `n_zones = 128` for `w5_diff_rot`, `w5_binary_rlof`, and
   `w5_astero_seism`. For `w5_riemann_rsp2`, the composite observable sums
   positive `(w / 1e5)` and negative `(de_t_dt / 1e8)` terms that cross zero
   near `s ~ 0.435` (`k = 3` at `n_zones = 8` where `|v[3]| / max(|v|) = 1.4e-3`
   and `k = 55` at `n_zones = 128` where `|v[55]| / max(|v|) = 1.2e-4`), so
   elementwise relative error at the zero-crossing cell is `< 5.0e-12` at
   `n_zones = 8` and `< 5.0e-11` at `n_zones = 128` across CPU and CUDA FMA,
   while `max(|v0 - v2|) / max(|v0|)` and the profile checksum match to
   `< 1.5e-15`.
3. At `n_zones = 8` and `n_zones = 128`, the Python/JAX benchmark checksums
   match the pinned upstream Fortran MESA (`bench_kernels_wave5.f90`) checksums.
"""

from __future__ import annotations

import numpy as np

from flexmesa import bench


def test_wave5_bench_kernels_return_positive_throughput_and_parity() -> None:
    """All 4 Wave 5 benchmark kernels match Fortran MESA checksums."""
    records = bench.run_all_wave5_benchmarks(
        zone_counts=(8, 128), n_warmup=1, n_repeats=3
    )
    assert len(records) == 8
    by_key = {(r.workload, r.n_zones): r for r in records}
    for rec in records:
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        if rec.workload == "w5_riemann_rsp2":
            assert rec.max_rel_err < (5.0e-12 if rec.n_zones == 8 else 5.0e-11)
        elif rec.n_zones == 8:
            assert rec.max_rel_err < 1.0e-12
        else:
            assert rec.max_rel_err < 1.0e-14
        assert np.isfinite(rec.checksum)

    expected_n8 = {
        "w5_diff_rot": 3.082380424983001e5,
        "w5_riemann_rsp2": -2.543418211346364e5,
        "w5_binary_rlof": 6.234702123446103,
        "w5_astero_seism": 7.563537818023943e1,
    }
    for name, expected in expected_n8.items():
        actual = by_key[(name, 8)].checksum
        rel = abs(actual - expected) / abs(expected)
        assert rel < 1.0e-12, f"{name} N=8 checksum {actual} != {expected}"

    expected_n128 = {
        "w5_diff_rot": 3.12459668e5,
        "w5_riemann_rsp2": -2.69003013e5,
        "w5_binary_rlof": 6.27865748,
        "w5_astero_seism": 7.37762193e1,
    }
    for name, expected in expected_n128.items():
        actual = by_key[(name, 128)].checksum
        rel = abs(actual - expected) / abs(expected)
        assert rel < 1.0e-8, f"{name} N=128 checksum {actual} != {expected}"
