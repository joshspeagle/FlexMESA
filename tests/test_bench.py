# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for the four-stage benchmark suites (Waves 1 through 3b).

Verifies that Stages 0, 1, 2, and 3 produce machine-precision agreement and
match the pinned cross-language reference checksums at N_zones = 8 across all
Wave 1, Wave 2, Wave 3a, and Wave 3b workloads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from flexmesa import bench

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

_EXPECTED_WAVE3B_CHECKSUMS_N8: tuple[tuple[str, float], ...] = (
    ("eos_4d", 2.0373645646893260e02),
    ("kap_fixed", 2.9091957571119350e00),
    ("rates_screen", -1.1391889147248130e02),
    ("net_eval", 9.6196498481513540e01),
    ("atm_table", 8.9854111757167640e01),
)


def _assert_records_parity(
    records: Sequence[bench.BenchRecord],
    expected_n8: Sequence[tuple[str, float]],
    max_stage_rel_err: float = 1.0e-12,
    max_chk_rel_err: float = 1.0e-12,
) -> None:
    """Assert stage parity and N_zones = 8 reference checksums."""
    n_workloads = len(expected_n8)
    assert len(records) == 2 * n_workloads
    for rec in records:
        assert rec.max_rel_err <= max_stage_rel_err, (
            f"workload {rec.workload} (N={rec.n_zones}) "
            f"stage parity {rec.max_rel_err:.3e} > {max_stage_rel_err:.1e}"
        )
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        assert math.isfinite(rec.checksum)

    for rec, (exp_name, exp_sum) in zip(
        records[n_workloads:], expected_n8, strict=True
    ):
        assert rec.workload == exp_name
        rel_err = abs(rec.checksum - exp_sum) / abs(exp_sum)
        assert rel_err <= max_chk_rel_err, (
            f"workload {rec.workload} (N=8) checksum {rec.checksum:.15e} "
            f"vs {exp_sum:.15e} (rel {rel_err:.3e})"
        )


def test_wave1_four_stage_benchmarks_parity_across_stages() -> None:
    """All six Wave 1 workloads agree across Stages 0..3 and match Rust."""
    records = bench.run_all_wave1_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_parity(records, _EXPECTED_CHECKSUMS_N8)


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
    """All four Wave 2 workloads agree across Stages 0..3 and match Rust."""
    records = bench.run_all_wave2_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_parity(records, _EXPECTED_WAVE2_CHECKSUMS_N8)


def test_wave3a_four_stage_benchmarks_parity_across_stages() -> None:
    """All four Wave 3a workloads agree across Stages 0..3 and match Rust."""
    records = bench.run_all_wave3a_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_parity(
        records,
        _EXPECTED_WAVE3A_CHECKSUMS_N8,
        max_stage_rel_err=2.0e-9,
        max_chk_rel_err=1.0e-13,
    )


def test_wave3b_four_stage_benchmarks_parity_across_stages() -> None:
    """All five Wave 3b workloads agree across Stages 0..3 and match Rust."""
    records = bench.run_all_wave3b_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_parity(
        records, _EXPECTED_WAVE3B_CHECKSUMS_N8, max_stage_rel_err=5.0e-12
    )
