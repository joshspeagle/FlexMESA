# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the MESA 26.4.1 port manifest and dashboard (`#174`)."""

from __future__ import annotations

import dataclasses
import pathlib
import re

from flexmesa.data import fetch
from flexmesa.testing import (
    _dashboard_collect,
    _dashboard_coverage,
    _dashboard_render,
    _port_parsers,
    dashboard,
    port_manifest,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_port_manifest_matches_mesa_26_4_1_and_verified_evidence() -> None:
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
    expected_kernel_verified = {
        "star/private/diffusion.f90",
        "star/private/diffusion_support.f90",
        "star/private/paquette_coeffs.f90",
        "star/private/phase_separation.f90",
        "star/private/solve_omega_mix.f90",
    }
    assert {e.upstream_path for e in verified_entries} == (
        expected_kernel_verified
    )
    for entry in verified_entries:
        assert entry.status == port_manifest.PortStatus.KERNEL_REFERENCE_VECTORS
        assert entry.evidence_artifacts
        assert entry.evidence_tests
    case_verified = [
        e
        for e in verified_entries
        if e.status == port_manifest.PortStatus.CASE_LEVEL_EVIDENCE
    ]
    assert not case_verified


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


def test_port_manifest_rejects_verified_without_artifact_or_test() -> None:
    """Verified status tiers must fail when artifacts or tests are missing."""
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

    no_test = dataclasses.replace(
        target,
        status=port_manifest.PortStatus.KERNEL_REFERENCE_VECTORS,
        evidence_artifacts=("tests/fixtures/golden/module_hex/summary.json",),
        evidence_tests=(),
    )
    mutated_1b = dataclasses.replace(
        manifest,
        entries=tuple(
            no_test if e.upstream_path == target.upstream_path else e
            for e in manifest.entries
        ),
    )
    errs_1b = port_manifest.verify_port_manifest(
        mutated_1b, repo_root=REPO_ROOT
    )
    assert any("requires non-empty evidence_tests" in e for e in errs_1b)

    traversal_art = dataclasses.replace(
        target,
        status=port_manifest.PortStatus.KERNEL_REFERENCE_VECTORS,
        evidence_artifacts=("benchmarks/../README.md",),
        evidence_tests=("crates/flexmesa-const/tests/const_tests.rs",),
    )
    mutated_1c = dataclasses.replace(
        manifest,
        entries=tuple(
            traversal_art if e.upstream_path == target.upstream_path else e
            for e in manifest.entries
        ),
    )
    errs_1c = port_manifest.verify_port_manifest(
        mutated_1c, repo_root=REPO_ROOT
    )
    assert any("must be a safe path" in e for e in errs_1c)

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


def test_port_manifest_promotion_and_base_metadata_immutability() -> None:
    """Raising status requires diffing evidence; upstream metadata is fixed."""
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

    absent_entry = by_path["math/private/math_io.f90"]
    unver_promoted = dataclasses.replace(
        absent_entry,
        status=port_manifest.PortStatus.PARTIAL,
        flexmesa_files=("crates/flexmesa-math/src/lib.rs",),
    )
    unver_manifest = dataclasses.replace(
        base,
        entries=tuple(
            unver_promoted
            if e.upstream_path == absent_entry.upstream_path
            else e
            for e in base.entries
        ),
    )
    errs_unver_blocked = port_manifest.verify_port_manifest(
        unver_manifest,
        repo_root=REPO_ROOT,
        base_manifest=base,
        changed_paths=frozenset({"benchmarks/dashboard/port_manifest.toml"}),
    )
    assert any("implementing flexmesa_files" in e for e in errs_unver_blocked)

    tampered_lines = dataclasses.replace(
        ported_entry, upstream_lines=ported_entry.upstream_lines + 10
    )
    tampered_manifest = dataclasses.replace(
        base,
        entries=tuple(
            tampered_lines
            if e.upstream_path == ported_entry.upstream_path
            else e
            for e in base.entries
        ),
    )
    errs_tamper = port_manifest.verify_port_manifest(
        tampered_manifest,
        repo_root=REPO_ROOT,
        base_manifest=base,
        changed_paths=frozenset({"benchmarks/dashboard/port_manifest.toml"}),
    )
    assert any("upstream metadata" in e for e in errs_tamper)

    promoted = dataclasses.replace(
        ported_entry,
        status=port_manifest.PortStatus.KERNEL_REFERENCE_VECTORS,
        evidence_artifacts=("tests/fixtures/golden/module_hex/summary.json",),
        evidence_tests=("crates/flexmesa-const/tests/const_tests.rs",),
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
        changed_paths=frozenset(
            {
                "benchmarks/dashboard/port_manifest.toml",
                "benchmarks/history_harness/baseline_run_26_4_1.json",
            }
        ),
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


def test_dashboard_renders_and_manifest_invariants_hold() -> None:
    """The dashboard must render and the port manifest must be valid.

    The committed README, docs page and JSON are not compared here: `main`
    regenerates them after each merge (`DECISIONS #47`), so a pull request
    does not carry them. The comparison itself is exercised by
    `test_dashboard_freshness_gate_detects_hand_edited_readme_number`.
    """
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    errs = dashboard.verify_all_dashboard_invariants(
        repo_root=REPO_ROOT, mesa_dir=mesa_dir, check_outputs=False
    )
    assert not errs


def test_readme_dashboard_is_compact_and_covers_every_module() -> None:
    """The README view stays short and names each manifest module once."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    data = dashboard.build_dashboard_data(REPO_ROOT, mesa_dir)
    block = _dashboard_render.render_readme_block(data)
    lines = block.splitlines()
    assert len(lines) <= 160
    for row in data["module_coverage"]["rows"]:
        mod = row["module"]
        assert sum(1 for ln in lines if ln.startswith(f"| `{mod}` | `")) == 1
    assert block.count("```mermaid") == 2
    assert "img.shields.io/badge/engine%20ported-" in block
    hh = _dashboard_render.summarize_headline(data)["hh_tot"]
    assert f"**{hh['passing']} / {hh['total']}**" in block


def test_dashboard_freshness_gate_detects_hand_edited_readme_number(
    tmp_path: pathlib.Path,
) -> None:
    """Perturbing a single number in README.md must fail freshness check."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    expected = dashboard.render_expected_files(REPO_ROOT, mesa_dir)
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    for child in REPO_ROOT.iterdir():
        if child.name in {".git", "README.md", "docs", "benchmarks"}:
            continue
        (fake_root / child.name).symlink_to(child)

    for sub in ("docs", "benchmarks"):
        (fake_root / sub).mkdir()
        for child in (REPO_ROOT / sub).iterdir():
            (fake_root / sub / child.name).symlink_to(child)

    readme_fresh = expected[REPO_ROOT / "README.md"]
    tampered = re.sub(
        r"(\d+) / 106",
        lambda m: f"{int(m.group(1)) + 1} / 106",
        readme_fresh,
        count=1,
    )
    assert tampered != readme_fresh
    (fake_root / "README.md").write_text(tampered, encoding="utf-8")

    errs = dashboard.verify_all_dashboard_invariants(
        repo_root=fake_root, mesa_dir=mesa_dir
    )
    assert any("README.md: committed content differs" in e for e in errs)


def test_fortran_and_rust_parsers_handle_interfaces_fixed_form_and_macros(
    tmp_path: pathlib.Path,
) -> None:
    """Parsers must extract generic interfaces, fixed-form .f, and const fn."""
    f90_src = """
    module demo_mod
      implicit none
      private
      public :: generic_op, direct_sub
      interface generic_op
        module procedure impl_op_a
      end interface generic_op
    contains
      subroutine direct_sub(x)
        real :: x
      end subroutine direct_sub
      subroutine impl_op_a(x)
        real :: x
      end subroutine impl_op_a
    end module demo_mod
    """
    assert _port_parsers.extract_fortran_public_routines(f90_src, ".f90") == [
        "generic_op",
        "direct_sub",
    ]

    f77_src = (
        "      character*(*) function f77_fn(\n"
        "     &  arg1)\n"
        "      integer arg1\n"
        "      f77_fn = 'x'\n"
        "      end\n"
        "      subroutine f77_sub(y)\n"
        "      real y\n"
        "      end\n"
    )
    assert _port_parsers.extract_fortran_public_routines(f77_src, ".f") == [
        "f77_fn",
        "f77_sub",
    ]

    rs_file = tmp_path / "sample.rs"
    rs_file.write_text(
        "// pub fn fake_comment() {}\n"
        'const DOC: &str = "pub fn fake_string() {}";\n'
        "pub const fn real_const_fn() {}\n"
        'unary_method!(exp, kernels::u_exp, "doc");\n',
        encoding="utf-8",
    )
    syms = _port_parsers.extract_flexmesa_symbols(rs_file)
    assert "real_const_fn" in syms
    assert "exp" in syms
    assert "fake_comment" not in syms
    assert "fake_string" not in syms

    rs_skips = _dashboard_collect.extract_rust_ignores_from_text(
        '/// #[ignore]\n#[test] #[ignore = "tracked in #152"]\n'
        '#[cfg_attr(unix, ignore = "tracked in #155")]\nfn t() {}\n',
        "crates/x/tests/t.rs",
    )
    assert len(rs_skips) == 2
    assert rs_skips[0]["issues"] == "#152"
    assert rs_skips[1]["issues"] == "#155"

    py_skips = _dashboard_collect.extract_py_decorator_skips_from_text(
        "import pytest\n"
        '@pytest.mark.skipif(True, reason="pending #153")\n'
        "def test_x(): pass\n",
        "tests/test_x.py",
    )
    assert len(py_skips) == 1
    assert py_skips[0]["issues"] == "#153"


def test_dashboard_numerics_rows_reflect_165_ci_presence(
    tmp_path: pathlib.Path,
) -> None:
    """Numerics rows report #165 data when CI jobs exist, else unmeasured."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    data = dashboard.build_dashboard_data(REPO_ROOT, mesa_dir)
    assert "gated in CI" in data["numerics"]["strict_cross_machine_identical"]
    assert (
        "403,258/403,258 CRMATH"
        in (data["numerics"]["strict_kernels_bit_identical"])
    )

    fake_root = tmp_path / "pre165"
    (fake_root / ".github" / "workflows").mkdir(parents=True)
    (fake_root / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  rust:\n    runs-on: ubuntu-latest\n", encoding="utf-8"
    )
    (fake_root / "tests").symlink_to(REPO_ROOT / "tests")
    pre_165 = _dashboard_collect.collect_numerics_and_module_verification(
        fake_root
    )
    assert pre_165["strict_cross_machine_identical"] == "not yet measured"
    assert pre_165["strict_kernels_bit_identical"] == "not yet measured"


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
