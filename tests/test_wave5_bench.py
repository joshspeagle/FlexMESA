# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the Wave 5 benchmark suite and FlexMESA regression snapshots.

Checks that:
1. All 4 Wave 5 benchmark kernels (`bench_w5_diff_rot`, `bench_w5_riemann_rsp2`,
   `bench_w5_binary_rlof`, and `bench_w5_astero_seism`) and
   `run_all_wave5_benchmarks` execute across Stages 0..3 (`rust` sequential,
   `rust` 4-thread parallel, `@jax.jit`, and `jax.jit(jax.vmap)`).
2. Internal Rust-versus-JAX relative difference is `< 1.0e-12` at `n_zones = 8`
   and `< 1.0e-14` at `n_zones = 128` for `w5_diff_rot`, `w5_binary_rlof`, and
   `w5_astero_seism`, and `< 5.0e-12` (`n_zones = 8`) / `< 5.0e-11`
   (`n_zones = 128`) near the `s ~ 0.435` zero crossing of `w5_riemann_rsp2`.
3. At `n_zones = 8` and `n_zones = 128`, outputs match self-generated FlexMESA
   regression snapshots (not upstream MESA outputs; `w5_diff_rot` shifted in
   `642da54` from `3.082380424983001e5` to `2.551374595308515e3` at `N=8` and
   from `3.12459668e5` to `2.60059307e3` at `N=128`; tracked in #146).
"""

from __future__ import annotations

import numpy as np

from flexmesa import bench


def test_wave5_bench_kernels_return_positive_throughput_and_parity() -> None:
    """All 4 Wave 5 benchmark kernels match FlexMESA regression snapshots."""
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

    regression_snapshots_n8 = {
        "w5_diff_rot": 2.551374595308515e3,
        "w5_riemann_rsp2": -2.543418211346364e5,
        "w5_binary_rlof": 6.234702123446103,
        "w5_astero_seism": 7.563537818023943e1,
    }
    for name, snap in regression_snapshots_n8.items():
        actual = by_key[(name, 8)].checksum
        rel = abs(actual - snap) / abs(snap)
        assert rel < 1.0e-12, (
            f"{name} N=8 regression snapshot {actual} != {snap}"
        )

    regression_snapshots_n128 = {
        "w5_diff_rot": 2.60059307e3,
        "w5_riemann_rsp2": -2.69003013e5,
        "w5_binary_rlof": 6.27865748,
        "w5_astero_seism": 7.37762193e1,
    }
    for name, snap in regression_snapshots_n128.items():
        actual = by_key[(name, 128)].checksum
        rel = abs(actual - snap) / abs(snap)
        assert rel < 1.0e-8, (
            f"{name} N=128 regression snapshot {actual} != {snap}"
        )
