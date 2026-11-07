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

_EXPECTED_WAVE2_CHECKSUMS_N8: tuple[tuple[str, float], ...] = (
    ("chem_comp", 1.6870014935210610e01),
    ("ion_fe56", 1.6837093929584180e02),
    ("ion_4d", 6.1188461461721810e01),
    ("colors_sed", 3.6086917325391080e01),
)

_EXPECTED_WAVE3A_CHECKSUMS_N8: tuple[tuple[str, float], ...] = (
    ("eos_ideal", 1.4382242574811980e02),
    ("eos_helm", 3.5616889578660440e02),
    ("eos_pc", 4.1178868944201080e02),
    ("eos_skye", 3.5613318367227720e02),
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


def test_wave2_four_stage_benchmarks_parity_across_stages() -> None:
    """All four Wave 2 workloads agree across Stages 0..3 and match Rust.

    Verifies stage-to-stage relative error <= 1e-12 and pinned cross-language
    reference checksums at N_zones = 8.
    """
    records = bench.run_all_wave2_benchmarks(
        zone_counts=(4, 8),
        n_warmup=1,
        n_repeats=3,
    )
    assert len(records) == 8
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
        records[4:], _EXPECTED_WAVE2_CHECKSUMS_N8, strict=True
    ):
        assert rec.workload == exp_name
        rel_err = abs(rec.checksum - exp_sum) / abs(exp_sum)
        assert rel_err <= 1.0e-12, (
            f"workload {rec.workload} (N=8) checksum {rec.checksum:.15e} "
            f"vs {exp_sum:.15e} (rel {rel_err:.3e})"
        )


def test_wave3a_four_stage_benchmarks_parity_across_stages() -> None:
    """All four Wave 3a EOS workloads agree across Stages 0..3 and match Rust.

    Verifies stage-to-stage relative error <= 1e-9 across all 26 EOS channels
    (including 2nd-order autodiff Hessians of Helmholtz free energy at low
    density) and pinned cross-language `lnPgas` reference checksums at
    N_zones = 8 to <= 1e-13.
    """
    records = bench.run_all_wave3a_benchmarks(
        zone_counts=(4, 8),
        n_warmup=1,
        n_repeats=3,
    )
    assert len(records) == 8
    for rec in records:
        assert rec.max_rel_err <= 2.0e-9, (
            f"workload {rec.workload} (N={rec.n_zones}) "
            f"stage parity error {rec.max_rel_err:.3e} > 2e-9"
        )
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        assert math.isfinite(rec.checksum)

    for rec, (exp_name, exp_sum) in zip(
        records[4:], _EXPECTED_WAVE3A_CHECKSUMS_N8, strict=True
    ):
        assert rec.workload == exp_name
        rel_err = abs(rec.checksum - exp_sum) / abs(exp_sum)
        assert rel_err <= 1.0e-13, (
            f"workload {rec.workload} (N=8) checksum {rec.checksum:.15e} "
            f"vs {exp_sum:.15e} (rel {rel_err:.3e})"
        )
