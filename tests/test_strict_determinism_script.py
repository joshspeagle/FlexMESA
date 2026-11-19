# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for `scripts/check_strict_determinism.py`."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_strict_determinism.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_strict_determinism", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_cases_and_uncapped_trace_commands() -> None:
    """Default determinism check covers uncapped star and binary cases."""
    mod = _load_script_module()
    assert mod._DEFAULT_STEPS is None
    assert mod._DEFAULT_CASES == (
        ("star", "1.3M_ms_high_Z"),
        ("binary", "jdot_gr_check"),
    )

    uncapped_cmd = mod._build_trace_command(
        "star", "1.3M_ms_high_Z", pathlib.Path("out"), None
    )
    assert "--max-steps" not in uncapped_cmd
    assert "--allow-unsupported-controls" in uncapped_cmd
    assert "--extras-factory" in uncapped_cmd
    assert "tests.case_extras:resolve_case_extras" in uncapped_cmd

    capped_cmd = mod._build_trace_command(
        "binary", "jdot_gr_check", pathlib.Path("out"), 5
    )
    assert "--extras-factory" in capped_cmd
    assert "tests.case_extras:resolve_case_extras" in capped_cmd
    assert capped_cmd[-5:] == [
        "--numerics-mode",
        "strict",
        "--allow-unsupported-controls",
        "--max-steps",
        "5",
    ]


def test_collect_history_digests_supports_star_and_binary(
    tmp_path: pathlib.Path,
) -> None:
    """Collect strict history.data and binary_history.data digests cleanly."""
    mod = _load_script_module()

    star_dir = tmp_path / "star_out"
    (star_dir / "LOGS").mkdir(parents=True)
    (star_dir / "LOGS" / "history.data").write_text(
        '1 2\nversion compiler math_backend\n"r26.4.1" "rustc" "strict"\n',
        encoding="utf-8",
    )
    star_files = mod._collect_history_digests(star_dir, "star")
    assert list(star_files) == ["LOGS/history.data"]

    bin_dir = tmp_path / "bin_out"
    (bin_dir / "LOGS1").mkdir(parents=True)
    (bin_dir / "binary_history.data").write_text(
        '1 2\nversion age\n"r26.4.1" 1.0\n', encoding="utf-8"
    )
    (bin_dir / "LOGS1" / "history.data").write_text(
        '1 2\nversion compiler math_backend\n"r26.4.1" "rustc" "strict"\n',
        encoding="utf-8",
    )
    bin_files = mod._collect_history_digests(bin_dir, "binary")
    assert list(bin_files) == ["LOGS1/history.data", "binary_history.data"]

    (star_dir / "LOGS" / "history.data").write_text(
        '1 2\nversion compiler math_backend\n"r26.4.1" "rustc" "fast"\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="strict"):
        mod._collect_history_digests(star_dir, "star")


def _make_cross_arch_record(
    os_name: str, arch: str, bin_sha: str
) -> dict[str, Any]:
    sha_a = "a" * 64
    sha_b = "b" * 64
    return {
        "os": os_name,
        "arch": arch,
        "numerics_mode": "strict",
        "math_vector_count": 474_062,
        "math_vectors_sha256": sha_a,
        "evolution_suite": "star",
        "evolution_case": "1.3M_ms_high_Z",
        "evolution_max_steps": None,
        "history_bytes": 1000,
        "history_1t_sha256": sha_b,
        "history_4t_sha256": sha_b,
        "thread_1_vs_4_byte_identical": True,
        "cases": [
            {
                "suite": "star",
                "case": "1.3M_ms_high_Z",
                "max_steps": None,
                "history_bytes": 1000,
                "history_1t_sha256": sha_b,
                "history_4t_sha256": sha_b,
                "history_files": {
                    "LOGS/history.data": {"bytes": 1000, "sha256": sha_b}
                },
                "thread_1_vs_4_byte_identical": True,
            },
            {
                "suite": "binary",
                "case": "jdot_gr_check",
                "max_steps": None,
                "history_bytes": 500,
                "history_1t_sha256": bin_sha,
                "history_4t_sha256": bin_sha,
                "history_files": {
                    "binary_history.data": {"bytes": 500, "sha256": bin_sha}
                },
                "thread_1_vs_4_byte_identical": True,
            },
        ],
    }


def test_compare_jsons_detects_cross_arch_case_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    """Compare uncapped multi-case JSON records across x86_64 and aarch64."""
    mod = _load_script_module()
    sha_c = "c" * 64
    p1 = tmp_path / "x86.json"
    p2 = tmp_path / "arm.json"
    p3 = tmp_path / "arm_bad.json"
    p1.write_text(
        json.dumps(_make_cross_arch_record("Linux", "x86_64", sha_c)),
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(_make_cross_arch_record("macOS", "arm64", sha_c)),
        encoding="utf-8",
    )
    p3.write_text(
        json.dumps(_make_cross_arch_record("macOS", "arm64", "d" * 64)),
        encoding="utf-8",
    )

    assert mod._compare_jsons([p1, p2]) == 0
    assert mod._compare_jsons([p1, p3]) == 1


def test_trace_summary_parsing_and_step_validation() -> None:
    """Steps and termination are read from the trace summary line."""
    mod = _load_script_module()
    out = (
        "warning: something\n"
        "suite=star case=demo status=ok term=log_L_upper_limit steps=346 "
        "retries=0 wall_s=1.0 peak_rss_gb=1.0 history=/x/history.data\n"
    )
    assert mod._parse_trace_summary(out) == ("log_L_upper_limit", 346)
    with pytest.raises(RuntimeError, match="no trace summary line"):
        mod._parse_trace_summary("nothing useful\n")

    good = {
        "history_bytes": 10,
        "history_1t_sha256": "a" * 64,
        "history_4t_sha256": "a" * 64,
        "thread_1_vs_4_byte_identical": True,
        "steps": 5,
    }
    assert mod._validate_digest_pair(good, "x") == []
    errs = mod._validate_digest_pair({**good, "steps": 0}, "x")
    assert any("steps must be > 0" in e for e in errs)
