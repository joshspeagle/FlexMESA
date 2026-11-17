# SPDX-License-Identifier: LGPL-3.0-or-later
"""Rust (``cargo llvm-cov``) and Python (``pytest --cov``) parsers (`#174`)."""

from __future__ import annotations

import json
import pathlib
from typing import Any


def parse_llvm_cov_summary(
    llvm_cov_json: dict[str, Any],
) -> dict[str, Any]:
    """Extract workspace and per-crate line coverage from ``cargo llvm-cov``.

    Args:
        llvm_cov_json: Parsed JSON payload from
            ``cargo llvm-cov --json --summary-only``.

    Returns:
        Dictionary with ``overall`` and ``by_crate`` line-coverage records.
    """
    data_list = llvm_cov_json.get("data", [])
    if not data_list:
        raise ValueError("Empty data array in cargo llvm-cov JSON")
    entry = data_list[0]

    by_crate_accum: dict[str, list[int]] = {}
    for fitem in entry.get("files", []):
        fname = str(fitem.get("filename", "")).replace("\\", "/")
        marker = "/crates/"
        idx = fname.find(marker)
        if idx < 0:
            continue
        rest = fname[idx + len(marker) :]
        parts = rest.split("/")
        if len(parts) < 2 or parts[1] != "src":
            continue
        crate_name = parts[0]
        lines_summary = fitem.get("summary", {}).get("lines", {})
        count = int(lines_summary.get("count", 0))
        covered = int(lines_summary.get("covered", 0))
        acc = by_crate_accum.setdefault(crate_name, [0, 0])
        acc[0] += count
        acc[1] += covered

    total_lines = sum(v[0] for v in by_crate_accum.values())
    total_covered = sum(v[1] for v in by_crate_accum.values())
    by_crate_rows: list[dict[str, Any]] = []
    for crate_name in sorted(by_crate_accum):
        cnt, cov = by_crate_accum[crate_name]
        pct = round(100.0 * cov / cnt, 1) if cnt else 0.0
        by_crate_rows.append(
            {
                "crate": crate_name,
                "lines": cnt,
                "covered": cov,
                "percent": pct,
            }
        )

    return {
        "tool": "cargo llvm-cov",
        "overall": {
            "lines": total_lines,
            "covered": total_covered,
            "percent": (
                round(100.0 * total_covered / total_lines, 1)
                if total_lines
                else 0.0
            ),
        },
        "by_crate": by_crate_rows,
    }


def parse_pytest_cov_json(
    pytest_cov_json: dict[str, Any],
) -> dict[str, Any]:
    """Extract package and subpackage statement coverage from ``coverage.py``.

    Args:
        pytest_cov_json: Parsed JSON payload from ``pytest --cov-report=json``.

    Returns:
        Dictionary with ``overall`` and ``by_subpackage`` statement coverage.
    """
    by_pkg_accum: dict[str, list[int]] = {}
    for raw_path, fdata in pytest_cov_json.get("files", {}).items():
        norm = str(raw_path).replace("\\", "/")
        marker = "src/flexmesa/"
        idx = norm.find(marker)
        if idx >= 0:
            rest = norm[idx + len(marker) :]
        elif norm.startswith("flexmesa/"):
            rest = norm[len("flexmesa/") :]
        else:
            continue
        parts = rest.split("/")
        subpkg = parts[0] if len(parts) > 1 else "(top-level)"
        summary = fdata.get("summary", {})
        stmts = int(summary.get("num_statements", 0))
        covered = int(summary.get("covered_lines", 0))
        acc = by_pkg_accum.setdefault(subpkg, [0, 0])
        acc[0] += stmts
        acc[1] += covered

    tot_stmts = sum(v[0] for v in by_pkg_accum.values())
    tot_cov = sum(v[1] for v in by_pkg_accum.values())
    rows: list[dict[str, Any]] = []
    for subpkg in sorted(by_pkg_accum):
        cnt, cov = by_pkg_accum[subpkg]
        pct = round(100.0 * cov / cnt, 1) if cnt else 0.0
        rows.append(
            {
                "subpackage": (
                    "flexmesa"
                    if subpkg == "(top-level)"
                    else f"flexmesa.{subpkg}"
                ),
                "statements": cnt,
                "covered": cov,
                "percent": pct,
            }
        )

    return {
        "tool": "pytest --cov",
        "overall": {
            "statements": tot_stmts,
            "covered": tot_cov,
            "percent": (
                round(100.0 * tot_cov / tot_stmts, 1) if tot_stmts else 0.0
            ),
        },
        "by_subpackage": rows,
    }


def write_coverage_summary(
    rust_json_path: pathlib.Path,
    python_json_path: pathlib.Path,
    out_path: pathlib.Path,
) -> dict[str, Any]:
    """Combine Rust and Python JSON reports into ``coverage_summary.json``."""
    rust_raw = json.loads(rust_json_path.read_text(encoding="utf-8"))
    py_raw = json.loads(python_json_path.read_text(encoding="utf-8"))
    summary = {
        "rust": parse_llvm_cov_summary(rust_raw),
        "python": parse_pytest_cov_json(py_raw),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
