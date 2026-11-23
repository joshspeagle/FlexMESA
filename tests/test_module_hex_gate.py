# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and Level-2 hex verification gate tests for `module_hex_gate`."""

from __future__ import annotations

import json
import pathlib

import pytest

from flexmesa import _rust, constants
from flexmesa.testing import (
    case_manifest,
    module_hex_gate,
)


def test_live_module_hex_gate_and_summary_match() -> None:
    """Verify live Level-2 0-ULP checks and pinned summary JSON freshness."""
    root = case_manifest.repo_root()
    res = module_hex_gate.verify_module_hex_gate(root)
    assert res.ok, module_hex_gate.render_markdown_summary(res)

    summary = res.to_summary_dict()
    assert summary["total_packages_in_install"] == 26
    assert summary["modules_with_executable_probes"] >= 14
    assert summary["total_values_0_ulp"] >= 3800

    by_mod = {r.module: r for r in res.module_results}
    assert by_mod["const"].status == "pass_0_ulp"
    assert by_mod["const"].zero_ulp_values == 23
    assert by_mod["binary"].status == "pass_0_ulp"
    assert by_mod["binary"].zero_ulp_values == 5
    assert by_mod["math"].status == "pass_0_ulp"
    assert by_mod["math"].checked_values == 99
    assert by_mod["math"].zero_ulp_values == 99
    assert by_mod["auto_diff"].status == "pass_0_ulp"
    assert by_mod["auto_diff"].checked_values == 387
    assert by_mod["auto_diff"].zero_ulp_values == 387
    assert by_mod["turb"].status == "pass_0_ulp"
    assert by_mod["turb"].checked_values == 71
    assert by_mod["turb"].zero_ulp_values == 71
    assert by_mod["ionization"].expected_outputs == 31
    assert by_mod["kap"].status == "pass_0_ulp"
    assert by_mod["kap"].checked_values == 21
    assert by_mod["kap"].zero_ulp_values == 21
    assert by_mod["net"].status == "partial_0_ulp"
    assert by_mod["net"].checked_values == 116
    assert by_mod["net"].zero_ulp_values == 116

    summary_path = root / module_hex_gate.DEFAULT_SUMMARY_JSON_REL
    assert summary_path.is_file()
    rendered = json.dumps(summary, indent=2) + "\n"
    assert summary_path.read_text(encoding="utf-8") == rendered


def test_ieee754_hex_and_ulp_helpers() -> None:
    """Verify exact big-endian hex conversion and signed-magnitude ULP math."""
    pi_hex = module_hex_gate.float64_to_raw_hex(constants.PI)
    assert pi_hex == "0x400921fb54442d18"
    assert module_hex_gate.raw_hex_to_float64(pi_hex) == constants.PI
    assert module_hex_gate.ulp_distance(constants.PI, pi_hex) == 0
    assert module_hex_gate.ulp_distance(-0.0, "0x0000000000000000") == 0
    assert module_hex_gate.ulp_distance(1.0, "0x3ff0000000000001") == 1
    assert module_hex_gate.ulp_distance(float("nan"), "0x7ff8000000000000") == 0


def test_resolve_hex_root_fails_when_missing(tmp_path: pathlib.Path) -> None:
    """Fail loudly if mesa_module_hex bundle directory is absent."""
    with pytest.raises(FileNotFoundError, match="mesa_module_hex"):
        module_hex_gate.resolve_hex_root(tmp_path / "missing_hex")


def _write_const_tsv(
    const_dir: pathlib.Path, const_hex_override: str | None
) -> None:
    lines = [
        "record_idx\tval_idx\tsrc_loc\tlabel\tkind\tdecimal\thex_float\traw_bits_hex"
    ]
    for idx, val in enumerate(_rust.const_probe_values(), start=1):
        h = (
            const_hex_override
            if (idx == 1 and const_hex_override is not None)
            else module_hex_gate.float64_to_raw_hex(val)
        )
        lines.append(
            f"{idx}\t1\ttest.f90:{idx}\tc{idx}\tf64\t{val:.17e}\t0x0\t{h}"
        )
    (const_dir / "exact_tested_values.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_minimal_hex_fixture(
    tmp_path: pathlib.Path, *, const_hex_override: str | None = None
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Create minimal synthetic repo and mesa_module_hex fixture for const."""
    hex_root = tmp_path / "hex"
    (hex_root / "const").mkdir(parents=True)
    summary = {
        "total_packages_in_install": 1,
        "packages_with_test_dir": 1,
        "total_exact_values_captured": 23,
        "modules": [{"module": "const", "exact_values_captured": 23}],
    }
    (hex_root / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    _write_const_tsv(hex_root / "const", const_hex_override)

    repo = tmp_path / "repo"
    (repo / "benchmarks" / "dashboard").mkdir(parents=True)
    (repo / "benchmarks" / "module_hex").mkdir(parents=True)
    (repo / "benchmarks" / "dashboard" / "github_snapshot.json").write_text(
        json.dumps(
            {
                "phases_140": [
                    {
                        "phase": "Phase C",
                        "issues": [
                            {"number": 161, "state": "open"},
                            {"number": 163, "state": "closed"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    gaps_toml = repo / "benchmarks" / "module_hex" / "known_module_gaps.toml"
    gaps_toml.write_text("", encoding="utf-8")
    return repo, hex_root, gaps_toml


def test_synthetic_hex_gate_catches_undeclared_mismatch_and_stale_gap(
    tmp_path: pathlib.Path,
) -> None:
    """Reject undeclared ULP mismatches, stale gaps, and closed issues."""
    repo, hex_root, gaps_toml = _write_minimal_hex_fixture(tmp_path)
    assert module_hex_gate.verify_module_hex_gate(
        repo, hex_dir=hex_root, gaps_toml_path=gaps_toml
    ).ok

    _, bad_hex, _ = _write_minimal_hex_fixture(
        tmp_path / "bad", const_hex_override="0x400921fb54442d19"
    )
    res_bad = module_hex_gate.verify_module_hex_gate(
        repo, hex_dir=bad_hex, gaps_toml_path=gaps_toml
    )
    assert not res_bad.ok
    assert any("const" in msg for msg in res_bad.undeclared_hex_mismatches)

    gaps_toml.write_text(
        "[[gaps]]\n"
        'module = "const"\n'
        'status = "nonzero_ulp"\n'
        'issue = "#161"\n'
        "max_declared_ulp = 5\n"
        "min_zero_ulp_count = 20\n"
        "expect_nonzero_ulp = true\n"
        'reason = "stale"\n',
        encoding="utf-8",
    )
    res_stale = module_hex_gate.verify_module_hex_gate(
        repo, hex_dir=hex_root, gaps_toml_path=gaps_toml
    )
    assert not res_stale.ok
    assert res_stale.stale_module_gaps

    gaps_toml.write_text(
        "[[gaps]]\n"
        'module = "const"\n'
        'status = "nonzero_ulp"\n'
        'issue = "#163"\n'
        "max_declared_ulp = 5\n"
        "expect_nonzero_ulp = true\n"
        'reason = "closed issue"\n',
        encoding="utf-8",
    )
    res_closed = module_hex_gate.verify_module_hex_gate(
        repo, hex_dir=hex_root, gaps_toml_path=gaps_toml
    )
    assert not res_closed.ok
    assert any("#163" in msg for msg in res_closed.closed_or_unknown_issues)


def test_synthetic_hex_gate_rejects_inconsistent_declarations(
    tmp_path: pathlib.Path,
) -> None:
    """Reject contradictory status declarations in known_module_gaps.toml."""
    repo, bad_hex, gaps_toml = _write_minimal_hex_fixture(
        tmp_path, const_hex_override="0x400921fb54442d19"
    )
    gaps_toml.write_text(
        "[[gaps]]\n"
        'module = "const"\n'
        'status = "unported_hex_harness"\n'
        'issue = "#161"\n'
        'reason = "contradicts active probe"\n',
        encoding="utf-8",
    )
    res_unported = module_hex_gate.verify_module_hex_gate(
        repo, hex_dir=bad_hex, gaps_toml_path=gaps_toml
    )
    assert not res_unported.ok
    assert any(
        "unported_hex_harness" in msg
        for msg in res_unported.inconsistent_module_declarations
    )

    gaps_toml.write_text(
        "[[gaps]]\n"
        'module = "const"\n'
        'status = "partial_0_ulp"\n'
        'issue = "#161"\n'
        "max_declared_ulp = 5\n"
        "expect_nonzero_ulp = true\n"
        'reason = "contradicts partial_0_ulp"\n',
        encoding="utf-8",
    )
    res_partial = module_hex_gate.verify_module_hex_gate(
        repo, hex_dir=bad_hex, gaps_toml_path=gaps_toml
    )
    assert not res_partial.ok
    assert any(
        "partial_0_ulp" in msg
        for msg in res_partial.inconsistent_module_declarations
    )
