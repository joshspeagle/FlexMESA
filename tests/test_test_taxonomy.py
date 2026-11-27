# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit and live-manifest tests for W4 test-suite taxonomy (`#563`)."""

from __future__ import annotations

import pathlib

import pytest

from flexmesa.data import fetch
from flexmesa.testing import (
    _dashboard_render,
    dashboard,
    port_manifest,
    test_taxonomy,
)

pytestmark = pytest.mark.invariant

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write_mini_manifest(repo: pathlib.Path) -> pathlib.Path:
    man_path = repo / "benchmarks" / "dashboard" / "port_manifest.toml"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(
        "\n".join(
            [
                "[metadata]",
                "schema_version = 1",
                'upstream_version = "r26.4.1"',
                'upstream_commit = "c8c2026"',
                f'archive_md5 = "{"0" * 32}"',
                "",
                "[[files]]",
                'upstream_path = "turb/private/mlt.f90"',
                'module = "turb"',
                "upstream_lines = 100",
                'upstream_sha256 = "' + "0" * 64 + '"',
                'status = "ported_unverified"',
                'notes = "fixture"',
                'rust_targets = ["crates/flexmesa-turb/src/mlt.rs"]',
                "",
                "[files.routine_map]",
                'calc_MLT = "calc_mlt"',
                'set_MLT = "set_mlt"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return man_path


def _write_mini_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    (repo / "crates" / "flexmesa-turb" / "tests").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    _write_mini_manifest(repo)
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "use flexmesa_turb::calc_mlt;\n"
        "#[test]\n"
        "fn matches_mesa_hex() {\n"
        "    let got = f64::from_bits(0x3ff0_0000_0000_0000);\n"
        "    let _ = calc_mlt(got);\n"
        "    assert_eq!(got, 1.0);\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.invariant\n"
        "def test_positive() -> None:\n"
        "    assert 1 == 1\n",
        encoding="utf-8",
    )
    return repo


def test_live_repository_test_taxonomy_is_valid_and_bijective() -> None:
    """Live `test_taxonomy.toml` validates cleanly across all test files."""
    errors = test_taxonomy.verify_test_taxonomy(REPO_ROOT)
    assert errors == []

    tax_path = REPO_ROOT / test_taxonomy.DEFAULT_TAXONOMY_REL
    tax = test_taxonomy.load_test_taxonomy(tax_path)
    discovered = set(test_taxonomy.discover_test_files(REPO_ROOT))
    by_path = tax.by_path()
    assert set(by_path) == discovered
    assert by_path["tests/test_test_taxonomy.py"].kind == "invariant"
    assert by_path["tests/test_test_taxonomy.py"].module == "tooling"

    man = port_manifest.load_port_manifest(
        REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml"
    )
    summary = test_taxonomy.summarize_test_taxonomy(REPO_ROOT, tax, man)
    assert summary["total_files"] == len(discovered)
    assert summary["total_tests"] > 0
    assert summary["mislabeled_count"] > 0
    assert summary["mapped_routines_total"] > 0
    assert summary["listed_oracle_routines_total"] > 0
    assert summary["listed_oracle_routines_pct"] > 0.0
    assert (
        len(summary["listed_oracle_routine_ids"])
        == (summary["listed_oracle_routines_total"])
    )
    by_kind = summary["by_kind"]
    for kind in test_taxonomy.VALID_KINDS:
        assert list(by_kind[kind].keys()) == ["files"]
    assert (
        sum(by_kind[k]["files"] for k in test_taxonomy.VALID_KINDS)
        == summary["total_files"]
    )


def test_taxonomy_detects_bijection_and_duplicate_errors(
    tmp_path: pathlib.Path,
) -> None:
    """Missing, stale, and duplicate entries fail with DECISIONS #72 hint."""
    repo = _write_mini_repo(tmp_path)
    tax_path = repo / test_taxonomy.DEFAULT_TAXONOMY_REL
    valid = [
        test_taxonomy.TestFileRecord(
            path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
            language="rust",
            module="turb",
            kind="oracle",
            provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex oracle",
            routines=("turb/private/mlt.f90::calc_MLT",),
        ),
        test_taxonomy.TestFileRecord(
            path="tests/test_turb_invariants.py",
            language="python",
            module="turb",
            kind="invariant",
        ),
    ]

    # Duplicate + missing
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(
                schema_version=1, entries=(valid[0], valid[0])
            )
        ),
        encoding="utf-8",
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("duplicate entry" in e for e in errs)
    assert any("missing from" in e and "DECISIONS #72" in e for e in errs)

    # Stale
    stale = test_taxonomy.TestFileRecord(
        path="tests/test_deleted.py",
        language="python",
        module="turb",
        kind="invariant",
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(
                schema_version=1, entries=(*valid, stale)
            )
        ),
        encoding="utf-8",
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("missing on disk" in e and "DECISIONS #72" in e for e in errs)


def test_taxonomy_enforces_oracle_provenance_and_evidence_in_code(
    tmp_path: pathlib.Path,
) -> None:
    """Oracle claims require existing MESA artifacts and code references."""
    repo = _write_mini_repo(tmp_path)
    # 1. Self-generated synthetic float snapshot without MESA provenance/code.
    (repo / "tests" / "test_turb_invariants.py").write_text(
        '"""Fake docstring mentioning MESA 26.4.1 test_output and golden."""\n'
        "def test_fake() -> None:\n"
        "    # benchmarks/fortran_mesa/fake.tsv in comment only\n"
        "    snapshot_val = 1.23456789012345\n"
        "    assert abs(snapshot_val - 1.23456789012345) < 1e-14\n",
        encoding="utf-8",
    )
    tax = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance="",
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="Self-generated snapshot 26.4.1 synthetic values",
            ),
        ),
    )
    (repo / test_taxonomy.DEFAULT_TAXONOMY_REL).write_text(
        test_taxonomy.dump_test_taxonomy(tax), encoding="utf-8"
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("requires non-empty provenance" in e for e in errs)
    assert any("must cite a concrete MESA 26.4.1 artifact" in e for e in errs)
    assert any("lacks code-level MESA oracle evidence" in e for e in errs)

    # 2. Synthetic snapshot remains rejected even with valid-looking anchor.
    tax_fake_anchor = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance=(
                    "Upstream MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex"
                ),
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="Upstream MESA 26.4.1 turb/test/test_output",
            ),
        ),
    )
    (repo / test_taxonomy.DEFAULT_TAXONOMY_REL).write_text(
        test_taxonomy.dump_test_taxonomy(tax_fake_anchor), encoding="utf-8"
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert errs == [
        "tests/test_turb_invariants.py: kind='oracle' lacks code-level MESA "
        "oracle evidence"
    ]

    # 3. Nonexistent benchmark path & unreferenced existing benchmark artifact.
    existing_tsv = (
        repo / "benchmarks" / "fortran_mesa" / "turb_oracles" / "mlt.tsv"
    )
    existing_tsv.parent.mkdir(parents=True)
    existing_tsv.write_text("# mesa_version: r26.4.1\n", encoding="utf-8")
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "def test_other_golden_table() -> None:\n"
        "    golden_output = 1.0\n"
        "    assert golden_output == 1.0\n",
        encoding="utf-8",
    )
    tax_bad_paths = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance="benchmarks/fortran_mesa/does_not_exist.tsv",
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="benchmarks/fortran_mesa/turb_oracles/mlt.tsv",
            ),
        ),
    )
    (repo / test_taxonomy.DEFAULT_TAXONOMY_REL).write_text(
        test_taxonomy.dump_test_taxonomy(tax_bad_paths), encoding="utf-8"
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("does not exist on disk" in e for e in errs)
    assert any(
        "does not reference cited provenance artifact" in e for e in errs
    )

    # 4. Golden fixture missing MESA origin metadata fails verification.
    bad_golden = repo / "tests" / "fixtures" / "golden" / "turb" / "out.mod"
    bad_golden.parent.mkdir(parents=True)
    bad_golden.write_text(
        "version_number   'flexmesa-local'\n", encoding="utf-8"
    )
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "import pathlib\n"
        "def test_loads_fixture() -> None:\n"
        "    p = pathlib.Path('tests/fixtures/golden/turb/out.mod')\n"
        "    assert p.name == 'out.mod'\n",
        encoding="utf-8",
    )
    (repo / test_taxonomy.DEFAULT_TAXONOMY_REL).write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(
                schema_version=1,
                entries=(
                    test_taxonomy.TestFileRecord(
                        path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                        language="rust",
                        module="turb",
                        kind="oracle",
                        provenance="MESA 26.4.1 turb/private/mlt.f90 hex",
                    ),
                    test_taxonomy.TestFileRecord(
                        path="tests/test_turb_invariants.py",
                        language="python",
                        module="turb",
                        kind="oracle",
                        provenance="tests/fixtures/golden/turb/out.mod",
                    ),
                ),
            )
        ),
        encoding="utf-8",
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("without MESA 26.4.1 origin" in e for e in errs)

    # 5. Replay fixtures require MESA 26.4.1 origin headers.
    replay_dir = repo / "crates" / "flexmesa-replay" / "tests" / "data"
    replay_dir.mkdir(parents=True)
    replay_sample = replay_dir / "sample.replay"
    replay_sample.write_text("# compiler: local\nRUN\n", encoding="utf-8")
    replay_test = (
        repo / "crates" / "flexmesa-replay" / "tests" / "mesa_sample_test.rs"
    )
    replay_test.write_text(
        'const SAMPLE: &str = include_str!("data/sample.replay");\n'
        "#[test]\n"
        "fn replays_sample() {\n"
        "    assert!(!SAMPLE.is_empty());\n"
        "}\n",
        encoding="utf-8",
    )
    replay_entries = (
        test_taxonomy.TestFileRecord(
            path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
            language="rust",
            module="turb",
            kind="oracle",
            provenance="MESA 26.4.1 turb/private/mlt.f90 hex",
        ),
        test_taxonomy.TestFileRecord(
            path="tests/test_turb_invariants.py",
            language="python",
            module="turb",
            kind="invariant",
        ),
        test_taxonomy.TestFileRecord(
            path="crates/flexmesa-replay/tests/mesa_sample_test.rs",
            language="rust",
            module="tooling",
            kind="oracle",
            provenance=(
                "crates/flexmesa-replay/tests/data/sample.replay "
                "(MESA r26.4.1; #552)"
            ),
        ),
    )
    (repo / test_taxonomy.DEFAULT_TAXONOMY_REL).write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(schema_version=1, entries=replay_entries)
        ),
        encoding="utf-8",
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("cited replay fixture" in e for e in errs)
    replay_sample.write_text(
        "# mesa_version: r26.4.1\n# compiler: gfortran 15.2.0\nRUN\n",
        encoding="utf-8",
    )
    assert test_taxonomy.verify_test_taxonomy(repo) == []


def test_taxonomy_enforces_restricted_vocabulary_mislabel_gate(
    tmp_path: pathlib.Path,
) -> None:
    """Non-oracle filenames with restricted tokens need `#<issue>` reason."""
    repo = _write_mini_repo(tmp_path)
    misnamed = (
        repo / "crates" / "flexmesa-turb" / "tests" / "fake_parity_test.rs"
    )
    misnamed.write_text(
        "#[test]\nfn check_invariant() {\n    assert!(1 + 1 == 2);\n}\n",
        encoding="utf-8",
    )
    allowlisted = repo / "tests" / "test_golden.py"
    allowlisted.write_text(
        "def test_parser() -> None:\n    assert True\n", encoding="utf-8"
    )

    base_entries = [
        test_taxonomy.TestFileRecord(
            path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
            language="rust",
            module="turb",
            kind="oracle",
            provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex oracle",
        ),
        test_taxonomy.TestFileRecord(
            path="tests/test_turb_invariants.py",
            language="python",
            module="turb",
            kind="invariant",
        ),
        test_taxonomy.TestFileRecord(
            path="tests/test_golden.py",
            language="python",
            module="tooling",
            kind="invariant",
        ),
    ]

    # Without mislabel_reason -> rejected
    bad_rec = test_taxonomy.TestFileRecord(
        path="crates/flexmesa-turb/tests/fake_parity_test.rs",
        language="rust",
        module="turb",
        kind="invariant",
        mislabel_reason="missing issue id",
    )
    tax_path = repo / test_taxonomy.DEFAULT_TAXONOMY_REL
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(
                schema_version=1, entries=(*base_entries, bad_rec)
            )
        ),
        encoding="utf-8",
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("requires mislabel_reason citing #<issue>" in e for e in errs)

    # With #<issue> -> accepted
    good_rec = test_taxonomy.TestFileRecord(
        path="crates/flexmesa-turb/tests/fake_parity_test.rs",
        language="rust",
        module="turb",
        kind="invariant",
        mislabel_reason="#563: unit invariant test pending rename",
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(
                schema_version=1, entries=(*base_entries, good_rec)
            )
        ),
        encoding="utf-8",
    )
    assert test_taxonomy.verify_test_taxonomy(repo) == []


def test_taxonomy_detects_inline_markers_and_invalid_routines(
    tmp_path: pathlib.Path,
) -> None:
    """Check inline markers, manifest mapping, and code symbol presence."""
    repo = _write_mini_repo(tmp_path)
    tax_path = repo / test_taxonomy.DEFAULT_TAXONOMY_REL
    entries = (
        test_taxonomy.TestFileRecord(
            path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
            language="rust",
            module="turb",
            kind="invariant",
            routines=("turb/private/mlt.f90::nonexistent_routine",),
        ),
        test_taxonomy.TestFileRecord(
            path="tests/test_turb_invariants.py",
            language="python",
            module="turb",
            kind="regression",
        ),
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(schema_version=1, entries=entries)
        ),
        encoding="utf-8",
    )
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any(
        "inline marker 'oracle' disagrees with taxonomy kind 'invariant'" in e
        for e in errs
    )
    assert any(
        "inline marker 'invariant' disagrees with taxonomy kind 'regression'"
        in e
        for e in errs
    )
    assert any("cannot claim oracle routines" in e for e in errs)
    assert any("not in port_manifest.toml routine_map" in e for e in errs)

    # Mapped routine present only in comments/strings/test name fails.
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "// Comment mentions set_MLT and set_mlt\n"
        "use flexmesa_turb::set_mlt;\n"
        "#[test]\n"
        "fn test_set_mlt_and_set_MLT_in_name_only() {\n"
        '    let _tag = "set_MLT";\n'
        "    let z_dummy = (0.0_f64,);\n"
        "    let got = f64::from_bits(0x3ff0_0000_0000_0000) + z_dummy.0;\n"
        "    assert_eq!(got, 1.0);\n"
        "}\n",
        encoding="utf-8",
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(
            test_taxonomy.TestTaxonomy(
                schema_version=1,
                entries=(
                    test_taxonomy.TestFileRecord(
                        path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                        language="rust",
                        module="turb",
                        kind="oracle",
                        provenance=(
                            "MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex"
                        ),
                        routines=("turb/private/mlt.f90::set_MLT",),
                    ),
                    test_taxonomy.TestFileRecord(
                        path="tests/test_turb_invariants.py",
                        language="python",
                        module="turb",
                        kind="invariant",
                    ),
                ),
            )
        ),
        encoding="utf-8",
    )
    sym_errs = test_taxonomy.verify_test_taxonomy(repo)
    assert len(sym_errs) == 1
    assert "not called in non-comment test code" in sym_errs[0]


def test_taxonomy_enforces_call_sites_and_rust_imports(
    tmp_path: pathlib.Path,
) -> None:
    """Reject field reads, missing crate refs, and JAX-only Python."""
    repo = _write_mini_repo(tmp_path)
    man_path = repo / "benchmarks" / "dashboard" / "port_manifest.toml"
    man_text = man_path.read_text(encoding="utf-8") + (
        "\n"
        "[[files]]\n"
        'upstream_path = "turb/private/mlt_meth.f90"\n'
        'module = "turb"\n'
        "upstream_lines = 10\n"
        f'upstream_sha256 = "{"2" * 64}"\n'
        'status = "ported_unverified"\n'
        'notes = "method fixture"\n'
        "[files.routine_map]\n"
        'run_MLT = "crates/flexmesa-turb/src/mlt.rs::run_mlt"\n'
        'mod_MLT = "crates/flexmesa-turb/src/mlt.rs::mod_mlt"\n'
        "\n"
        "[[files]]\n"
        'upstream_path = "auto_diff/private/o1.f90"\n'
        'module = "auto_diff"\n'
        "upstream_lines = 10\n"
        f'upstream_sha256 = "{"1" * 64}"\n'
        'status = "ported_unverified"\n'
        'notes = "fixture"\n'
        "[files.routine_map]\n"
        'diff1 = "crates/flexmesa-autodiff/src/order1.rs::d1val1"\n'
        'fn_ref_op = "crates/flexmesa-turb/src/mlt.rs::eval_mlt_ref"\n'
    )
    man_path.write_text(man_text, encoding="utf-8")
    turb_src = repo / "crates" / "flexmesa-turb" / "src"
    turb_src.mkdir(parents=True)
    (turb_src / "mlt.rs").write_text(
        "pub struct MltSolver;\n"
        "impl MltSolver {\n"
        "    pub fn new() -> Self { Self }\n"
        "    pub fn run_mlt(&self) {}\n"
        "}\n"
        "pub fn mod_mlt(x: f64) -> f64 { x }\n"
        "pub fn eval_mlt_ref(x: f64) -> f64 { x }\n",
        encoding="utf-8",
    )

    # 1. Rust field read & missing crate fail; fn-item passes.
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "use flexmesa_turb::{calc_mlt, eval_mlt_ref};\n"
        "#[test]\n"
        "fn field_and_fn_item() {\n"
        "    let got = eval3(eval_mlt_ref) + z.d1val1;\n"
        "    let _ = calc_mlt(got);\n"
        "    let _ = 0x3ff0_0000_0000_0000_u64;\n"
        "}\n",
        encoding="utf-8",
    )
    # 2. Python without `flexmesa._rust` import + attr read fails.
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "def test_jax_only(obj: object) -> None:\n"
        "    _ = obj.set_mlt\n"
        "    golden_output = 0x3ff0_0000_0000_0000\n"
        "    assert golden_output > 0\n",
        encoding="utf-8",
    )
    tax_path = repo / test_taxonomy.DEFAULT_TAXONOMY_REL
    tax = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=(
                    "turb/private/mlt.f90::calc_MLT",
                    "auto_diff/private/o1.f90::fn_ref_op",
                    "auto_diff/private/o1.f90::diff1",
                ),
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=("turb/private/mlt.f90::set_MLT",),
            ),
        ),
    )
    tax_path.write_text(test_taxonomy.dump_test_taxonomy(tax), encoding="utf-8")
    errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("targets crate 'flexmesa_autodiff'" in e for e in errs)
    assert any("diff1" in e and "not called" in e for e in errs)
    assert not any("fn_ref_op" in e for e in errs)
    assert any("must import `flexmesa._rust`" in e for e in errs)
    assert any("set_MLT" in e and "not called" in e for e in errs)

    # 3. Importing `flexmesa._rust` + calling `_rust.set_mlt` passes.
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "from flexmesa import _rust\n"
        "def test_pyo3_call() -> None:\n"
        "    _ = _rust.set_mlt(1.0)\n"
        "    golden_output = 0x3ff0_0000_0000_0000\n"
        "    assert golden_output > 0\n",
        encoding="utf-8",
    )
    tax_ok = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=(
                    "turb/private/mlt.f90::calc_MLT",
                    "auto_diff/private/o1.f90::fn_ref_op",
                ),
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=("turb/private/mlt.f90::set_MLT",),
            ),
        ),
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(tax_ok), encoding="utf-8"
    )
    assert test_taxonomy.verify_test_taxonomy(repo) == []

    # 4. Crate `use ... as alias` + qualified `alias::sym` passes.
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "use flexmesa_turb as t;\n"
        "#[test]\n"
        "fn alias_qualified_call_and_fn_item() {\n"
        "    let got = eval3(t::eval_mlt_ref);\n"
        "    let _ = t::calc_mlt(got);\n"
        "    let _ = 0x3ff0_0000_0000_0000_u64;\n"
        "}\n",
        encoding="utf-8",
    )
    assert test_taxonomy.verify_test_taxonomy(repo) == []

    # 5. Reject unrelated `.sym()`, local `fn sym`, cross-crate calls, and
    #    method calls without target type/constructor (#605 review).
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "use flexmesa_math::calc_mlt;\n"
        "use flexmesa_turb::eval_mlt_ref;\n"
        "fn mod_mlt(x: f64) -> f64 { x }\n"
        "#[test]\n"
        "fn false_accept_patterns() {\n"
        "    let _ = calc_mlt(1.0);\n"
        "    let _ = mod_mlt(1.0);\n"
        "    let _ = eval3(eval_mlt_ref);\n"
        "    let _ = other_obj.set_mlt();\n"
        "    let _ = bogus_obj.run_mlt();\n"
        "    let _ = 0x3ff0_0000_0000_0000_u64;\n"
        "}\n",
        encoding="utf-8",
    )
    # Reject Python `jax_twin.set_mlt()` even when `_rust` is imported.
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "from flexmesa import _rust\n"
        "class JaxMock:\n"
        "    def set_mlt(self, x: float) -> float:\n"
        "        return x\n"
        "def test_jax_alongside_rust() -> None:\n"
        "    _ = _rust.other_helper(1.0)\n"
        "    jax_twin = JaxMock()\n"
        "    _ = jax_twin.set_mlt(1.0)\n"
        "    golden_output = 0x3ff0_0000_0000_0000\n"
        "    assert golden_output > 0\n",
        encoding="utf-8",
    )
    tax_strict = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=(
                    "turb/private/mlt.f90::calc_MLT",
                    "turb/private/mlt.f90::set_MLT",
                    "turb/private/mlt_meth.f90::run_MLT",
                    "turb/private/mlt_meth.f90::mod_MLT",
                    "auto_diff/private/o1.f90::fn_ref_op",
                ),
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=("turb/private/mlt.f90::set_MLT",),
            ),
        ),
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(tax_strict), encoding="utf-8"
    )
    strict_errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("calc_MLT" in e and "not called" in e for e in strict_errs)
    assert any("set_MLT" in e and "not called" in e for e in strict_errs)
    assert any("run_MLT" in e and "not called" in e for e in strict_errs)
    assert any("mod_MLT" in e and "not called" in e for e in strict_errs)
    assert not any("fn_ref_op" in e for e in strict_errs)
    assert any(
        e.startswith("tests/test_turb_invariants.py:") and "set_MLT" in e
        for e in strict_errs
    )
    assert not any("must import `flexmesa._rust`" in e for e in strict_errs)

    # 6. Accept symbol alias, module glob + target-type method call, and
    #    direct `from flexmesa._rust import sym as alias` in Python.
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "use flexmesa_turb::mlt::calc_mlt as run_c;\n"
        "use flexmesa_turb::mlt::*;\n"
        "use flexmesa_turb::MltSolver;\n"
        "use flexmesa_turb::set_mlt;\n"
        "#[test]\n"
        "fn valid_aliases_globs_and_methods() {\n"
        "    let got = eval3(eval_mlt_ref);\n"
        "    let _ = run_c(got);\n"
        "    let _ = mod_mlt(got);\n"
        "    let _ = set_mlt(got);\n"
        "    let solver = MltSolver::new();\n"
        "    solver.run_mlt();\n"
        "    let _ = 0x3ff0_0000_0000_0000_u64;\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_turb_invariants.py").write_text(
        "from flexmesa._rust import set_mlt as rust_set_mlt\n"
        "def test_direct_pyo3_alias_call() -> None:\n"
        "    _ = rust_set_mlt(1.0)\n"
        "    golden_output = 0x3ff0_0000_0000_0000\n"
        "    assert golden_output > 0\n",
        encoding="utf-8",
    )
    assert test_taxonomy.verify_test_taxonomy(repo) == []

    # Reject local `fn` shadowing even under valid glob, and wrong glob module.
    (
        repo / "crates" / "flexmesa-turb" / "tests" / "mlt_oracle_test.rs"
    ).write_text(
        "// SPDX-License-Identifier: GPL-3.0-or-later\n"
        "//! @test-kind: oracle\n"
        "use flexmesa_turb::mlt::*;\n"
        "use flexmesa_turb::other::*;\n"
        "fn mod_mlt(x: f64) -> f64 { x }\n"
        "#[test]\n"
        "fn local_shadow_and_wrong_glob() {\n"
        "    let _ = mod_mlt(1.0);\n"
        "    let _ = eval3(set_mlt);\n"
        "    let _ = 0x3ff0_0000_0000_0000_u64;\n"
        "}\n",
        encoding="utf-8",
    )
    tax_glob = test_taxonomy.TestTaxonomy(
        schema_version=1,
        entries=(
            test_taxonomy.TestFileRecord(
                path="crates/flexmesa-turb/tests/mlt_oracle_test.rs",
                language="rust",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=(
                    "turb/private/mlt_meth.f90::mod_MLT",
                    "turb/private/mlt.f90::set_MLT",
                ),
            ),
            test_taxonomy.TestFileRecord(
                path="tests/test_turb_invariants.py",
                language="python",
                module="turb",
                kind="oracle",
                provenance="MESA 26.4.1 turb/private/mlt.f90 IEEE-754 hex",
                routines=("turb/private/mlt.f90::set_MLT",),
            ),
        ),
    )
    tax_path.write_text(
        test_taxonomy.dump_test_taxonomy(tax_glob), encoding="utf-8"
    )
    glob_errs = test_taxonomy.verify_test_taxonomy(repo)
    assert any("mod_MLT" in e and "not called" in e for e in glob_errs)
    assert any("set_MLT" in e and "not called" in e for e in glob_errs)


def test_dashboard_and_cli_report_file_only_taxonomy_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dashboard docs and CLI `--summary` report file-only counts."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    data = dashboard.build_dashboard_data(REPO_ROOT, mesa_dir)
    assert "test_taxonomy" in data
    tax_data = data["test_taxonomy"]
    assert tax_data["total_files"] > 0
    assert tax_data["mapped_routines_total"] > 0
    assert tax_data["listed_oracle_routines_total"] > 0

    docs_md = _dashboard_render.render_docs_page(data)
    assert "Test classification & listed routine oracles (`W4`, `#563`)" in (
        docs_md
    )
    assert (
        "#### Test Classification & Routines with a Listed Oracle Test (W4)"
        in docs_md
    )
    assert "Routines with Listed Oracle Test" in docs_md
    assert "oracle files" in docs_md
    assert "invariant files" in docs_md
    assert "regression files" in docs_md
    assert "test functions" in docs_md
    assert "oracle tests" not in docs_md
    assert "Per-Routine Oracle Coverage (W4)" not in docs_md

    rc = test_taxonomy.main(["--repo-root", str(REPO_ROOT), "--summary"])
    assert rc == 0
    cli_out = capsys.readouterr().out
    assert "oracle files" in cli_out
    assert "test functions" in cli_out
    assert "routines with a listed oracle test=" in cli_out
    assert "oracle tests" not in cli_out
