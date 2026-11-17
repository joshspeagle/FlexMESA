# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data collectors for the generated FlexMESA progress dashboard (`#174`)."""

from __future__ import annotations

import ast
import json
import pathlib
import re
from typing import Any

from flexmesa.testing import case_manifest, case_report, port_manifest

MODULE_CODE_PATHS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "star": (("crates/flexmesa-star/src",), ("src/flexmesa/star",)),
    "star_data": ((), ()),
    "binary": (("crates/flexmesa-binary/src",), ("src/flexmesa/binary",)),
    "astero": (("crates/flexmesa-astero/src",), ("src/flexmesa/astero",)),
    "eos": (("crates/flexmesa-eos/src",), ("src/flexmesa/eos",)),
    "kap": (("crates/flexmesa-kap/src",), ("src/flexmesa/kap",)),
    "net": (("crates/flexmesa-net/src",), ("src/flexmesa/net",)),
    "rates": (("crates/flexmesa-rates/src",), ("src/flexmesa/rates",)),
    "neu": (("crates/flexmesa-neu/src",), ("src/flexmesa/neu",)),
    "atm": (("crates/flexmesa-atm/src",), ("src/flexmesa/atm",)),
    "turb": (("crates/flexmesa-turb/src",), ("src/flexmesa/turb",)),
    "chem": (("crates/flexmesa-chem/src",), ("src/flexmesa/chem",)),
    "ionization": (
        ("crates/flexmesa-ionization/src",),
        ("src/flexmesa/ionization",),
    ),
    "colors": (("crates/flexmesa-colors/src",), ("src/flexmesa/colors",)),
    "num": (("crates/flexmesa-num/src",), ("src/flexmesa/num",)),
    "mtx": (("crates/flexmesa-mtx/src",), ("src/flexmesa/mtx",)),
    "interp_1d": (
        ("crates/flexmesa-interp1d/src",),
        ("src/flexmesa/interp_1d",),
    ),
    "interp_2d": (
        ("crates/flexmesa-interp2d/src",),
        ("src/flexmesa/interp_2d",),
    ),
    "auto_diff": (
        ("crates/flexmesa-autodiff/src",),
        ("src/flexmesa/auto_diff",),
    ),
    "math": (("crates/flexmesa-math/src",), ("src/flexmesa/math",)),
    "const": (("crates/flexmesa-const/src",), ("src/flexmesa/constants",)),
    "utils": (
        (
            "crates/flexmesa-utils/src",
            "crates/flexmesa-schema/src",
            "crates/flexmesa-data/src",
        ),
        (
            "src/flexmesa/utils",
            "src/flexmesa/schema",
            "src/flexmesa/data",
        ),
    ),
    "gyre": ((), ()),
    "adipls": ((), ()),
    "stella": ((), ()),
    "plotting": ((), ("src/flexmesa/plotting",)),
}

_RUST_COMMENT_RE = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_IGNORE_ATTR_RE = re.compile(
    r"#\s*\[\s*(?:cfg_attr\s*\([^]]*,\s*)?ignore"
    r'(?:\s*=\s*"([^"]*)")?\s*\)?\s*\]'
)
_ISSUE_RE = re.compile(r"#(\d+)")


def _count_dir_lines(
    repo_root: pathlib.Path, rel_dirs: tuple[str, ...], suffix: str
) -> int:
    total = 0
    for rel in rel_dirs:
        base = repo_root / rel
        if not base.is_dir():
            continue
        for fp in sorted(base.rglob(f"*{suffix}")):
            if fp.is_file():
                total += len(fp.read_bytes().splitlines())
    return total


def _collect_plotting_panel_counts(
    manifest: port_manifest.PortManifest, repo_root: pathlib.Path
) -> dict[str, int]:
    plot_fns: list[str] = []
    plot_dir = repo_root / "src" / "flexmesa" / "plotting"
    if plot_dir.is_dir():
        for py_fp in sorted(plot_dir.glob("*.py")):
            if py_fp.name == "__init__.py":
                continue
            plot_fns.extend(
                re.findall(
                    r"^def\s+(plot_[A-Za-z0-9_]+)\b",
                    py_fp.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                )
            )
    pgstar_panels = sum(
        1
        for e in manifest.entries
        if e.module == "plotting"
        and e.upstream_path.endswith(".f90")
        and not e.upstream_path.endswith(
            (
                "ctrls_io.f90",
                "support.f90",
                "stub.f90",
                "pgstar.f90",
                "pgbinary.f90",
                "lib.f90",
            )
        )
    )
    return {
        "flexmesa_plot_builders": len(plot_fns),
        "upstream_panel_modules": pgstar_panels,
    }


def collect_module_coverage(
    manifest: port_manifest.PortManifest, repo_root: pathlib.Path
) -> dict[str, Any]:
    """Aggregate upstream line counts by status plus Rust and JAX/Python LOC."""
    st_enum = port_manifest.PortStatus
    counts: dict[str, dict[str, int]] = {
        mod: {st.value: 0 for st in st_enum}
        for mod in port_manifest.MODULE_ORDER
    }
    file_counts: dict[str, int] = {mod: 0 for mod in port_manifest.MODULE_ORDER}
    for entry in manifest.entries:
        counts[entry.module][entry.status.value] += entry.upstream_lines
        file_counts[entry.module] += 1

    rows: list[dict[str, Any]] = []
    for mod in port_manifest.MODULE_ORDER:
        d = counts[mod]
        up_loc = sum(d.values())
        ported_loc = (
            d[st_enum.PORTED_UNVERIFIED.value]
            + d[st_enum.KERNEL_REFERENCE_VECTORS.value]
            + d[st_enum.CASE_LEVEL_EVIDENCE.value]
        )
        verified_loc = (
            d[st_enum.KERNEL_REFERENCE_VECTORS.value]
            + d[st_enum.CASE_LEVEL_EVIDENCE.value]
        )
        rs_dirs, py_dirs = MODULE_CODE_PATHS[mod]
        rows.append(
            {
                "module": mod,
                "upstream_files": file_counts[mod],
                "upstream_lines": up_loc,
                "absent": d[st_enum.ABSENT.value],
                "stub": d[st_enum.STUB.value],
                "partial": d[st_enum.PARTIAL.value],
                "ported_unverified": d[st_enum.PORTED_UNVERIFIED.value],
                "kernel_reference_vectors": d[
                    st_enum.KERNEL_REFERENCE_VECTORS.value
                ],
                "case_level_evidence": d[st_enum.CASE_LEVEL_EVIDENCE.value],
                "ported_total": ported_loc,
                "verified_total": verified_loc,
                "pct_ported": (
                    round(100.0 * ported_loc / up_loc, 1) if up_loc else 0.0
                ),
                "pct_verified": (
                    round(100.0 * verified_loc / up_loc, 1) if up_loc else 0.0
                ),
                "rust_lines": _count_dir_lines(repo_root, rs_dirs, ".rs"),
                "python_jax_lines": _count_dir_lines(repo_root, py_dirs, ".py"),
            }
        )

    infra_rs = _count_dir_lines(
        repo_root,
        (
            "crates/flexmesa-py/src",
            "crates/flexmesa-testing/src",
            "crates/flexmesa-bench/src",
        ),
        ".rs",
    )
    infra_py = _count_dir_lines(
        repo_root,
        ("src/flexmesa/testing", "src/flexmesa/bench"),
        ".py",
    )
    return {
        "rows": rows,
        "plotting_panels": _collect_plotting_panel_counts(manifest, repo_root),
        "infrastructure_lines": {
            "rust_bridge_testing_bench": infra_rs,
            "python_testing_bench": infra_py,
        },
    }


def collect_numerics_and_module_verification(
    repo_root: pathlib.Path,
) -> dict[str, Any]:
    """Collect numerics-mode verification status and module hex inventory."""
    ci_yml = (repo_root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    strict_report = (
        repo_root / "benchmarks" / "numerics" / "strict_verification.json"
    )
    has_165_ci = "strict-cross-machine-gate:" in ci_yml
    strict_mods: dict[str, str] = {}
    if has_165_ci and strict_report.is_file():
        rep: dict[str, Any] = json.loads(
            strict_report.read_text(encoding="utf-8")
        )
        level1 = str(rep.get("cross_machine_status", "not yet measured"))
        level2 = str(
            rep.get("kernels_bit_identical_status", "not yet measured")
        )
        raw_smods = rep.get("modules", {})
        if isinstance(raw_smods, dict):
            strict_mods = {str(k): str(v) for k, v in raw_smods.items()}
    else:
        level1 = "not yet measured"
        level2 = "not yet measured"

    hex_summary_path = (
        repo_root
        / "tests"
        / "fixtures"
        / "golden"
        / "module_hex"
        / "summary.json"
    )
    hex_data: dict[str, Any] = json.loads(
        hex_summary_path.read_text(encoding="utf-8")
    )
    mod_rows = [
        {
            "module": str(item["module"]),
            "upstream_exact_values": int(item.get("exact_values_captured", 0)),
            "exact_strict_compared": (
                "not yet measured"
                if level2 == "not yet measured"
                else strict_mods.get(str(item["module"]), "0")
            ),
        }
        for item in hex_data.get("modules", [])
        if str(item["module"]) not in {"forum", "sample"}
    ]
    return {
        "has_165_ci": bool(has_165_ci and strict_report.is_file()),
        "strict_cross_machine_identical": level1,
        "strict_kernels_bit_identical": level2,
        "total_upstream_hex_values": int(
            hex_data.get("total_exact_values_captured", 0)
        ),
        "modules": mod_rows,
    }


def collect_harness_summary(repo_root: pathlib.Path) -> dict[str, Any]:
    """Summarize the 106-case history-harness output via `case_report`."""
    hh_dir = repo_root / "benchmarks" / "history_harness"
    _, gaps_by_key, records = case_report.build_manifest_inventory_records(
        manifest_path=hh_dir / "case_manifest.toml",
        known_gaps_path=hh_dir / "known_gaps.toml",
        baseline_json_path=hh_dir / "baseline_run_26_4_1.json",
    )

    by_suite: dict[str, dict[str, int]] = {
        s: {
            "total": 0,
            "mesa_26_4_1_ref": 0,
            "no_reference_yet": 0,
            "reached_termination": 0,
            "within_numeric_tol": 0,
            "passing": 0,
            "declared_gaps": 0,
            "undeclared_failures": 0,
        }
        for s in case_manifest.VALID_SUITES
    }
    within_tol_non_pass: list[dict[str, str]] = []
    for rec in records:
        sr = by_suite[rec.suite]
        sr["total"] += 1
        if rec.reference_status == "mesa_26_4_1":
            sr["mesa_26_4_1_ref"] += 1
        elif rec.reference_status == "no_reference_yet":
            sr["no_reference_yet"] += 1
        within_tol = False
        if rec.executed and rec.termination_matches:
            sr["reached_termination"] += 1
            if (
                rec.comparison is not None
                and rec.comparison.within_final_tolerances
                and rec.comparison.within_track_tolerances
            ):
                sr["within_numeric_tol"] += 1
                within_tol = True
        if rec.verdict == "pass":
            sr["passing"] += 1
        elif rec.verdict.startswith("declared gap"):
            sr["declared_gaps"] += 1
            if within_tol:
                within_tol_non_pass.append(
                    {
                        "suite": rec.suite,
                        "case": rec.case,
                        "issue": rec.known_gap_issue or "",
                    }
                )
        elif rec.is_undeclared_failure:
            sr["undeclared_failures"] += 1

    gaps_by_issue: dict[str, int] = {}
    for gspec in gaps_by_key.values():
        gaps_by_issue[gspec.issue] = gaps_by_issue.get(gspec.issue, 0) + 1

    fig_dir = hh_dir / "figures"
    fig_count = len(list(fig_dir.glob("*.png"))) if fig_dir.is_dir() else 0
    return {
        "by_suite": by_suite,
        "within_tol_non_pass": within_tol_non_pass,
        "gaps_by_issue": dict(
            sorted(
                gaps_by_issue.items(), key=lambda kv: (-kv[1], int(kv[0][1:]))
            )
        ),
        "comparison_figures_count": fig_count,
    }


def collect_controls_summary(
    repo_root: pathlib.Path, mesa_dir: pathlib.Path | None
) -> list[dict[str, Any]]:
    """Parse control and default counts from ``defaults_diff_26_4_1.md``."""
    diff_md = (
        repo_root / "notes" / "audit" / "defaults_diff_26_4_1.md"
    ).read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    in_summary = False
    for line in diff_md.splitlines():
        if line.startswith("## Summary by Namelist"):
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if not in_summary or not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) == 7 and cols[1].isdigit():
            rows.append(
                {
                    "namelist": cols[0],
                    "upstream_scalars": int(cols[1]),
                    "matched_in_flexmesa": int(cols[2]),
                    "defaults_agree": int(cols[3]),
                    "value_mismatch": int(cols[4]),
                    "disabled_limit_none": int(cols[5]),
                    "unimplemented": int(cols[6]),
                }
            )
    if mesa_dir is not None and mesa_dir.is_dir():
        assign_re = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*="
        )
        for label, rel in (
            ("star &pgstar", "star/defaults/pgstar.defaults"),
            ("binary &pgbinary", "binary/defaults/pgbinary.defaults"),
        ):
            p = mesa_dir / rel
            if p.is_file():
                names = {
                    m.group(1).lower()
                    for raw in p.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if (m := assign_re.match(raw.split("!")[0])) is not None
                }
                rows.append(
                    {
                        "namelist": label,
                        "upstream_scalars": len(names),
                        "matched_in_flexmesa": 0,
                        "defaults_agree": 0,
                        "value_mismatch": 0,
                        "disabled_limit_none": 0,
                        "unimplemented": len(names),
                    }
                )
    return rows


def extract_rust_ignores_from_text(
    raw_text: str, rel: str
) -> list[dict[str, str]]:
    """Extract ``#[ignore]`` and ``#[cfg_attr(..., ignore)]`` annotations."""
    if "ignore" not in raw_text:
        return []
    cleaned = _RUST_COMMENT_RE.sub(
        lambda m: "\n" * m.group(0).count("\n"), raw_text
    )
    found: list[dict[str, str]] = []
    for m in _IGNORE_ATTR_RE.finditer(cleaned):
        lno = cleaned.count("\n", 0, m.start()) + 1
        reason = m.group(1) or ""
        issues = sorted({f"#{x}" for x in _ISSUE_RE.findall(reason)})
        found.append(
            {
                "location": f"{rel}:{lno}",
                "reason": reason,
                "issues": ", ".join(issues),
            }
        )
    return found


def extract_py_decorator_skips_from_text(
    source: str, rel: str
) -> list[dict[str, str]]:
    """Extract ``@pytest.mark.{xfail,skip,skipif}`` decorators using ``ast``."""
    if "xfail" not in source and "skip" not in source:
        return []
    tree = ast.parse(source, filename=rel)
    found: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        for dec in node.decorator_list:
            seg = ast.get_source_segment(source, dec) or ""
            if not re.search(r"\bmark\.(?:xfail|skip|skipif)\b", seg):
                continue
            issues = sorted({f"#{x}" for x in _ISSUE_RE.findall(seg)})
            found.append(
                {
                    "location": f"{rel}:{dec.lineno}",
                    "reason": " ".join(seg.split()),
                    "issues": ", ".join(issues),
                }
            )
    return found


def collect_ignored_and_skipped_tests(
    repo_root: pathlib.Path,
) -> dict[str, Any]:
    """Scan Rust ``#[ignore]`` and Python ``xfail``/``skip`` annotations."""
    rust_ignores: list[dict[str, str]] = []
    for rs_path in sorted((repo_root / "crates").rglob("*.rs")):
        rel = rs_path.relative_to(repo_root).as_posix()
        rust_ignores.extend(
            extract_rust_ignores_from_text(
                rs_path.read_text(encoding="utf-8"), rel
            )
        )

    py_skips: list[dict[str, str]] = []
    for root_sub in ("tests", "src/flexmesa"):
        for py_path in sorted((repo_root / root_sub).rglob("*.py")):
            if py_path.name == "conftest.py":
                continue
            rel = py_path.relative_to(repo_root).as_posix()
            py_skips.extend(
                extract_py_decorator_skips_from_text(
                    py_path.read_text(encoding="utf-8"), rel
                )
            )

    unlinked = [
        x["location"] for x in [*rust_ignores, *py_skips] if not x["issues"]
    ]
    return {
        "rust_ignore_count": len(rust_ignores),
        "rust_ignores": rust_ignores,
        "python_xfail_or_skip_count": len(py_skips),
        "python_skips": py_skips,
        "unlinked_locations": unlinked,
    }
