# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for the four-stage benchmark suites (Waves 1 through 3b).

Checks that Stages 0, 1, 2, and 3 agree internally and match self-generated
FlexMESA regression snapshots at N_zones = 8 across all Wave 1, Wave 2,
Wave 3a, and Wave 3b workloads (internal regression checks, not upstream MESA
outputs; tracked in #146).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from flexmesa import bench

_WAVE1_REGRESSION_SNAPSHOTS_N8: tuple[tuple[str, float], ...] = (
    ("auto_diff", 1.0892741555532441e04),
    ("interp_2d", 3.6227753491539829e02),
    ("neu", 9.873195638253145e12),
    ("mtx_tridiag", 1.0656505308237591e01),
    ("turb_mlt", 3.4190485461509601e00),
    ("turb_tdc", 3.3187342241650115e00),
)

_WAVE2_REGRESSION_SNAPSHOTS_N8: tuple[tuple[str, float], ...] = (
    ("chem_comp", 1.6870014935210610e01),
    ("ion_fe56", 1.6837093929584180e02),
    ("ion_4d", 6.1188461461721810e01),
    ("colors_sed", 3.6086917325391080e01),
)

_WAVE3A_REGRESSION_SNAPSHOTS_N8: tuple[tuple[str, float], ...] = (
    ("eos_ideal", 1.4382242574811980e02),
    ("eos_helm", 3.5616889578660440e02),
    ("eos_pc", 4.1178868944201080e02),
    ("eos_skye", 3.5613318367227720e02),
)

_WAVE3B_REGRESSION_SNAPSHOTS_N8: tuple[tuple[str, float], ...] = (
    ("eos_4d", 2.0373645646893260e02),
    ("kap_fixed", 2.9091957571119350e00),
    ("rates_screen", -1.1391889147248130e02),
    ("net_eval", 9.6196498481513540e01),
    ("atm_table", 8.9854111757167640e01),
)


def _assert_records_match_snapshots(
    records: Sequence[bench.BenchRecord],
    snapshots_n8: Sequence[tuple[str, float]],
    max_stage_rel_err: float = 1.0e-12,
    max_chk_rel_err: float = 1.0e-12,
    stage_check_elsewhere: frozenset[str] = frozenset(),
) -> None:
    """Assert cross-stage agreement and N_zones = 8 regression snapshots.

    Workloads named in `stage_check_elsewhere` have their cross-stage
    agreement asserted by a test of their own; every other check here still
    applies to them.
    """
    n_workloads = len(snapshots_n8)
    assert len(records) == 2 * n_workloads
    for rec in records:
        if rec.workload not in stage_check_elsewhere:
            assert rec.max_rel_err <= max_stage_rel_err, (
                f"workload {rec.workload} (N={rec.n_zones}) "
                f"cross-stage rel_err {rec.max_rel_err:.3e} > "
                f"{max_stage_rel_err:.1e}"
            )
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        assert math.isfinite(rec.checksum)

    for rec, (exp_name, exp_sum) in zip(
        records[n_workloads:], snapshots_n8, strict=True
    ):
        assert rec.workload == exp_name
        rel_err = abs(rec.checksum - exp_sum) / abs(exp_sum)
        assert rel_err <= max_chk_rel_err, (
            f"workload {rec.workload} (N=8) snapshot {rec.checksum:.15e} "
            f"!= {exp_sum:.15e} (rel {rel_err:.3e})"
        )


def test_wave1_four_stage_benchmarks_match_snapshots() -> None:
    """All six Wave 1 workloads agree across Stages 0..3 and match snapshots."""
    records = bench.run_all_wave1_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_match_snapshots(
        records,
        _WAVE1_REGRESSION_SNAPSHOTS_N8,
        stage_check_elsewhere=frozenset({"turb_tdc"}),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: Rust set_tdc carries the root's partials through the Newton "
        "iteration as MESA 26.4.1 does (turb/private/tdc.f90:253-345, #321); "
        "the Python stages apply the implicit function theorem at the "
        "converged root."
    ),
)
def test_wave1_turb_tdc_stages_agree() -> None:
    """The `turb_tdc` workload agrees across Stages 0..3 to 1e-12."""
    for n_zones in (4, 8):
        rec = bench.bench_turb_tdc(
            bench.ThermoGrid.create(n_zones), n_warmup=1, n_repeats=3
        )
        assert rec.max_rel_err <= 1.0e-12, (
            f"workload {rec.workload} (N={rec.n_zones}) "
            f"cross-stage rel_err {rec.max_rel_err:.3e} > 1.0e-12"
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


def test_wave2_four_stage_benchmarks_match_snapshots() -> None:
    """Wave 2 workloads agree across Stages 0..3 and match snapshots."""
    records = bench.run_all_wave2_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_match_snapshots(records, _WAVE2_REGRESSION_SNAPSHOTS_N8)


def test_wave3a_four_stage_benchmarks_match_snapshots() -> None:
    """Wave 3a workloads agree across Stages 0..3 and match snapshots."""
    records = bench.run_all_wave3a_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_match_snapshots(
        records,
        _WAVE3A_REGRESSION_SNAPSHOTS_N8,
        max_stage_rel_err=2.0e-9,
        max_chk_rel_err=1.0e-13,
    )


def test_wave3b_four_stage_benchmarks_match_snapshots() -> None:
    """Wave 3b workloads agree across Stages 0..3 and match snapshots."""
    records = bench.run_all_wave3b_benchmarks(
        zone_counts=(4, 8), n_warmup=1, n_repeats=3
    )
    _assert_records_match_snapshots(
        records, _WAVE3B_REGRESSION_SNAPSHOTS_N8, max_stage_rel_err=5.0e-12
    )
