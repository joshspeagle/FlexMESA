# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for the Wave 1 four-stage benchmark suite.

Verifies that Stages 0, 1, 2, and 3 produce machine-precision agreement
($\\le 10^{-13}$ relative discrepancy) across all six Wave 1 workloads
(`auto_diff`, `interp_2d`, `neu`, `mtx_tridiag`, `turb_mlt`, and `turb_tdc`).
"""

from __future__ import annotations

import math

from flexmesa import bench

# Pinned cross-language reference checksums at N_zones = 8 (shared with
# crates/flexmesa-bench/tests/bench_tests.rs).
_EXPECTED_CHECKSUMS_N8: tuple[tuple[str, float], ...] = (
    ("auto_diff", 1.0892741555532441e04),
    ("interp_2d", 3.6227753491539829e02),
    ("neu", 9.8689004129113184e12),
    ("mtx_tridiag", 1.0656505308237591e01),
    ("turb_mlt", 3.4190485461509601e00),
    ("turb_tdc", 3.3187342241650115e00),
)


def test_wave1_four_stage_benchmarks_parity_across_stages() -> None:
    """All six Wave 1 workloads agree across Stages 0..3 and match Rust.

    Verifies stage-to-stage relative error <= 1e-12 and pinned cross-language
    reference checksums at N_zones = 8.
    """
    records = bench.run_all_wave1_benchmarks(
        zone_counts=(4, 8),
        n_warmup=1,
        n_repeats=3,
    )
    assert len(records) == 12
    for rec in records:
        assert rec.max_rel_err <= 1.0e-12, (
            f"workload {rec.workload} (N={rec.n_zones}) "
            f"stage parity error {rec.max_rel_err:.3e} > 1e-12"
        )
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        assert math.isfinite(rec.checksum)

    for rec, (exp_name, exp_sum) in zip(
        records[6:], _EXPECTED_CHECKSUMS_N8, strict=True
    ):
        assert rec.workload == exp_name
        rel_err = abs(rec.checksum - exp_sum) / abs(exp_sum)
        assert rel_err <= 1.0e-12, (
            f"workload {rec.workload} (N=8) checksum {rec.checksum:.15e} "
            f"vs {exp_sum:.15e} (rel {rel_err:.3e})"
        )


def test_measure_us_records_positive_median_and_compile_latency() -> None:
    """`measure_us` records positive median and first-call compile latency."""
    stats = bench.measure_us(
        n_warmup=1,
        n_repeats=3,
        op=lambda: sum(range(100)),
        record_first_as_compile=True,
    )
    assert stats.median_us > 0.0
    assert stats.iqr_us >= 0.0
    assert stats.compile_us > 0.0
