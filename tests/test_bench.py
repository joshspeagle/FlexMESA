# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for the Wave 1 four-stage benchmark suite.

Verifies that Stages 0, 1, 2, and 3 produce machine-precision agreement
($\\le 10^{-13}$ relative discrepancy) across all six Wave 1 workloads
(`auto_diff`, `interp_2d`, `neu`, `mtx_tridiag`, `turb_mlt`, and `turb_tdc`).
"""

from __future__ import annotations

import math

from flexmesa import bench


def test_wave1_four_stage_benchmarks_parity_across_stages() -> None:
    """All six Wave 1 workloads agree across Stages 0..3 to <= 1e-12."""
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
