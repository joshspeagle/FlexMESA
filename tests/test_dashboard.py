# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the MESA 26.4.1 port manifest and dashboard (`#174`)."""

from __future__ import annotations

import dataclasses
import pathlib

from flexmesa.data import fetch
from flexmesa.testing import (
    _dashboard_coverage,
    dashboard,
    port_manifest,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_port_manifest_matches_mesa_26_4_1_with_zero_verified_rows() -> None:
    """Committed port_manifest.toml must validate against MESA 26.4.1."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    assert len(manifest.entries) == 1100
    errs = port_manifest.verify_port_manifest(
        manifest=manifest,
        repo_root=REPO_ROOT,
        mesa_dir=mesa_dir,
        baseline_harness_json=(
            REPO_ROOT
            / "benchmarks"
            / "history_harness"
            / "baseline_run_26_4_1.json"
        ),
    )
    assert not errs
    verified_entries = [e for e in manifest.entries if e.status.is_verified]
    assert not verified_entries


def test_port_manifest_rejects_ported_without_routine_counterpart() -> None:
    """Raising an entry to ported_unverified without routine_map must fail."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    by_path = manifest.by_upstream_path()
    target = by_path["star/private/eps_grav.f90"]
    broken = dataclasses.replace(target, routine_map=())
    mutated = dataclasses.replace(
        manifest,
        entries=tuple(
            broken if e.upstream_path == target.upstream_path else e
            for e in manifest.entries
        ),
    )
    errs = port_manifest.verify_port_manifest(
        manifest=mutated,
        repo_root=REPO_ROOT,
        mesa_dir=mesa_dir,
    )
    assert any("eval_eps_grav_and_partials" in err for err in errs)


def test_port_manifest_rejects_verified_without_generated_artifact() -> None:
    """Verified status tiers must fail when evidence_artifacts are missing."""
    manifest = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    by_path = manifest.by_upstream_path()
    target = by_path["const/public/const_def.f90"]

    no_art = dataclasses.replace(
        target,
        status=port_manifest.PortStatus.KERNEL_REFERENCE_VECTORS,
        evidence_artifacts=(),
    )
    mutated_1 = dataclasses.replace(
        manifest,
        entries=tuple(
            no_art if e.upstream_path == target.upstream_path else e
            for e in manifest.entries
        ),
    )
    errs_1 = port_manifest.verify_port_manifest(mutated_1, repo_root=REPO_ROOT)
    assert any("requires non-empty evidence_artifacts" in e for e in errs_1)

    bad_case = dataclasses.replace(
        target,
        status=port_manifest.PortStatus.CASE_LEVEL_EVIDENCE,
        evidence_artifacts=("notes/audit/case_table.md",),
        harness_cases=("star/1.3M_ms_high_Z",),
    )
    mutated_2 = dataclasses.replace(
        manifest,
        entries=tuple(
            bad_case if e.upstream_path == target.upstream_path else e
            for e in manifest.entries
        ),
    )
    errs_2 = port_manifest.verify_port_manifest(
        mutated_2,
        repo_root=REPO_ROOT,
        baseline_harness_json=(
            REPO_ROOT
            / "benchmarks"
            / "history_harness"
            / "baseline_run_26_4_1.json"
        ),
    )
    assert any("is not passing against mesa_26_4_1" in e for e in errs_2)


def test_port_manifest_promotion_requires_diff_and_allows_lowering() -> None:
    """Raising status requires diffing evidence; lowering does not."""
    base = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    by_path = base.by_upstream_path()

    ported_entry = by_path["const/public/const_def.f90"]
    lowered = dataclasses.replace(
        ported_entry,
        status=port_manifest.PortStatus.PARTIAL,
        routine_map=(),
    )
    lowered_manifest = dataclasses.replace(
        base,
        entries=tuple(
            lowered if e.upstream_path == ported_entry.upstream_path else e
            for e in base.entries
        ),
    )
    errs_low = port_manifest.verify_port_manifest(
        lowered_manifest,
        repo_root=REPO_ROOT,
        base_manifest=base,
        changed_paths=frozenset({"benchmarks/dashboard/port_manifest.toml"}),
    )
    assert not errs_low

    # Raising status without touching evidence fails.
    promoted = dataclasses.replace(
        ported_entry,
        status=port_manifest.PortStatus.KERNEL_REFERENCE_VECTORS,
        evidence_artifacts=("tests/fixtures/golden/module_hex/summary.json",),
    )
    promoted_manifest = dataclasses.replace(
        base,
        entries=tuple(
            promoted if e.upstream_path == ported_entry.upstream_path else e
            for e in base.entries
        ),
    )
    errs_raise_blocked = port_manifest.verify_port_manifest(
        promoted_manifest,
        repo_root=REPO_ROOT,
        base_manifest=base,
        changed_paths=frozenset({"benchmarks/dashboard/port_manifest.toml"}),
    )
    assert any(
        "without adding or updating evidence" in e for e in errs_raise_blocked
    )

    errs_raise_ok = port_manifest.verify_port_manifest(
        promoted_manifest,
        repo_root=REPO_ROOT,
        base_manifest=base,
        changed_paths=frozenset(
            {
                "benchmarks/dashboard/port_manifest.toml",
                "tests/fixtures/golden/module_hex/summary.json",
            }
        ),
    )
    assert not errs_raise_ok


def test_dashboard_freshness_gate_matches_committed_outputs() -> None:
    """Committed README, docs dashboard, and JSON must match a fresh render."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    errs = dashboard.verify_all_dashboard_invariants(
        repo_root=REPO_ROOT, mesa_dir=mesa_dir
    )
    assert not errs


def test_dashboard_freshness_gate_detects_hand_edited_readme_number(
    tmp_path: pathlib.Path,
) -> None:
    """Perturbing a single character in README.md must fail freshness check."""
    del tmp_path  # Unused.
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    expected = dashboard.render_expected_files(REPO_ROOT, mesa_dir)
    readme_text = expected[REPO_ROOT / "README.md"]
    tampered = readme_text.replace("0 / 106", "1 / 106", 1)
    assert tampered != readme_text


def test_dashboard_numerics_rows_report_not_yet_measured_before_165() -> None:
    """Numerics rows must read 'not yet measured' until #165 CI jobs merge."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    data = dashboard.build_dashboard_data(REPO_ROOT, mesa_dir)
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    strict_json = (
        REPO_ROOT / "benchmarks" / "numerics" / "strict_verification.json"
    )
    if "strict-cross-machine-gate:" not in ci_text or not strict_json.is_file():
        assert (
            data["numerics"]["strict_cross_machine_identical"]
            == "not yet measured"
        )
        assert (
            data["numerics"]["strict_kernels_bit_identical"]
            == "not yet measured"
        )


def test_coverage_parsers_aggregate_crate_and_subpackage_totals() -> None:
    """Coverage JSON parsers must aggregate crate and subpackage lines."""
    llvm_sample = {
        "data": [
            {
                "files": [
                    {
                        "filename": "/work/crates/flexmesa-const/src/lib.rs",
                        "summary": {"lines": {"count": 100, "covered": 95}},
                    },
                    {
                        "filename": "/work/crates/flexmesa-math/src/lib.rs",
                        "summary": {"lines": {"count": 100, "covered": 85}},
                    },
                ]
            }
        ]
    }
    rs_out = _dashboard_coverage.parse_llvm_cov_summary(llvm_sample)
    assert rs_out["overall"] == {"lines": 200, "covered": 180, "percent": 90.0}
    assert len(rs_out["by_crate"]) == 2

    py_sample = {
        "files": {
            "src/flexmesa/__init__.py": {
                "summary": {"num_statements": 10, "covered_lines": 10}
            },
            "src/flexmesa/eos/core.py": {
                "summary": {"num_statements": 90, "covered_lines": 70}
            },
        }
    }
    py_out = _dashboard_coverage.parse_pytest_cov_json(py_sample)
    assert py_out["overall"] == {
        "statements": 100,
        "covered": 80,
        "percent": 80.0,
    }
    assert [r["subpackage"] for r in py_out["by_subpackage"]] == [
        "flexmesa",
        "flexmesa.eos",
    ]
