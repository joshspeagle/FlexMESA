# SPDX-License-Identifier: GPL-3.0-or-later
"""Data collectors for the generated FlexMESA progress dashboard (`#174`)."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from flexmesa.testing import (
    case_manifest,
    case_report,
    known_gaps_gate,
    port_manifest,
    test_taxonomy,
)

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
    "math": (
        ("crates/flexmesa-math/src", "crates/flexmesa-crlibm-sys/src"),
        ("src/flexmesa/math",),
    ),
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
            if (
                py_fp.name == "__init__.py"
                or py_fp.name == "cli.py"
                or py_fp.name.startswith("_")
            ):
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


_PLATFORM_INTRINSIC_SECTIONS: frozenset[str] = frozenset(
    {
        "dlog",
        "dlog10",
        "intrinsic_log10",
        "intrinsic_exp",
        "atan2",
        "datan2",
        "intrinsic_pow",
    }
)


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
    fast_vs_strict = repo_root / "notes" / "audit" / "fast_vs_strict.json"
    math_vectors = (
        repo_root
        / "benchmarks"
        / "fortran_mesa"
        / "math_hex_vectors"
        / "mesa_26_4_1_math_vectors.txt"
    )
    has_165_ci = "strict-cross-machine-gate:" in ci_yml

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
    in_scope_hex = [
        item
        for item in hex_data.get("modules", [])
        if str(item["module"]) not in {"forum", "sample"}
    ]

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
    elif has_165_ci and fast_vs_strict.is_file() and math_vectors.is_file():
        fvs: dict[str, Any] = json.loads(
            fast_vs_strict.read_text(encoding="utf-8")
        )
        vec_n = int(fvs.get("total_vector_inputs", 0))
        evo = fvs.get("short_evolution", {})
        evo_case = f"{evo.get('suite')}/{evo.get('case')}"
        evo_steps = int(evo.get("max_steps", 0))
        crmath_n = 0
        plat_n = 0
        for fn_rec in fvs.get("functions", []):
            n_in = int(fn_rec.get("total_inputs", 0))
            if str(fn_rec.get("section", "")) in _PLATFORM_INTRINSIC_SECTIONS:
                plat_n += n_in
            else:
                crmath_n += n_in
        det_script = repo_root / "scripts" / "check_strict_determinism.py"
        det_txt = (
            det_script.read_text(encoding="utf-8")
            if det_script.is_file()
            else ""
        )
        if (
            "_DEFAULT_STEPS: int | None = None" in det_txt
            and "jdot_gr_check" in det_txt
        ):
            evo_summary = "uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check"
        else:
            evo_summary = f"{evo_case} {evo_steps}-step trace"
        level1 = (
            "gated in CI by scripts/check_strict_determinism.py "
            "(x86_64 vs aarch64, 1 vs 4 threads; "
            f"{vec_n:,} math vectors + {evo_summary} "
            "run with unsupported controls ignored; "
            "a bit-identity check, not a comparison with MESA)"
        )
        mhex_path = (
            repo_root / "benchmarks" / "module_hex" / "module_hex_summary.json"
        )
        if mhex_path.is_file():
            mhex: dict[str, Any] = json.loads(
                mhex_path.read_text(encoding="utf-8")
            )
            probed_non_math = 0
            for item in mhex.get("modules", []):
                mname = str(item.get("module", ""))
                if not mname or mname in {"forum", "sample", "math"}:
                    continue
                chk = int(item.get("checked_values", 0))
                exp = int(item.get("expected_outputs", 0))
                z_n = int(item.get("zero_ulp_values", 0))
                m_ulp = int(item.get("max_ulp", 0))
                if chk > 0:
                    probed_non_math += 1
                    if m_ulp == 0:
                        suffix = "(0 ULP)" if chk == exp else "(0 ULP partial)"
                        strict_mods[mname] = f"{z_n:,}/{exp:,} {suffix}"
                    else:
                        strict_mods[mname] = (
                            f"{z_n:,}/{chk:,} 0 ULP "
                            f"(max {m_ulp:,} ULP, {exp:,} total)"
                        )
            chk_tot = int(mhex.get("total_values_checked", 0))
            zero_tot = int(mhex.get("total_values_0_ulp", 0))
            pass_mods = int(mhex.get("modules_passing_0_ulp", 0))
            level2 = (
                f"math: {crmath_n:,}/{crmath_n:,} CRMATH 0 ULP "
                f"({plat_n:,} libm <= 1 ULP); "
                f"{zero_tot:,}/{chk_tot:,} probed module hex 0 ULP "
                f"({pass_mods} complete, {probed_non_math} non-math probed)"
            )
        else:
            level2 = (
                f"math: {crmath_n:,}/{crmath_n:,} CRMATH 0 ULP "
                f"({plat_n:,} libm <= 1 ULP); "
                f"0/{len(in_scope_hex) - 1} other module suites"
            )
        strict_mods["math"] = (
            f"{crmath_n:,} CRMATH (0 ULP) + {plat_n:,} libm (<= 1 ULP)"
        )
    else:
        level1 = "not yet measured"
        level2 = "not yet measured"

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
        for item in in_scope_hex
    ]
    return {
        "has_165_ci": bool(level1 != "not yet measured"),
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
            "strict_validated": 0,
            "diagnostic_runs": 0,
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
        if rec.reference_status in ("mesa_26_4_1", "mesa_26_4_1_artifact"):
            sr["mesa_26_4_1_ref"] += 1
        elif rec.reference_status == "no_reference_yet":
            sr["no_reference_yet"] += 1
        within_tol = False
        if rec.executed:
            if rec.controls_ignored:
                sr["diagnostic_runs"] += 1
            else:
                sr["strict_validated"] += 1
        if rec.executed and rec.termination_matches:
            sr["reached_termination"] += 1
            if rec.is_within_tolerances:
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
        if len(cols) == 8 and cols[1].isdigit():
            rows.append(
                {
                    "namelist": cols[0],
                    "upstream_scalars": int(cols[1]),
                    "matched_in_flexmesa": int(cols[2]),
                    "defaults_agree": int(cols[3]),
                    "value_mismatch": int(cols[4]),
                    "disabled_limit_none": int(cols[5]),
                    "skipped_unparseable_rust": int(cols[6]),
                    "unimplemented": int(cols[7]),
                }
            )
        elif len(cols) == 7 and cols[1].isdigit():
            rows.append(
                {
                    "namelist": cols[0],
                    "upstream_scalars": int(cols[1]),
                    "matched_in_flexmesa": int(cols[2]),
                    "defaults_agree": int(cols[3]),
                    "value_mismatch": int(cols[4]),
                    "disabled_limit_none": int(cols[5]),
                    "skipped_unparseable_rust": 0,
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
                        "skipped_unparseable_rust": 0,
                        "unimplemented": len(names),
                    }
                )
    return rows


def extract_rust_ignores_from_text(
    raw_text: str, rel: str
) -> list[dict[str, str]]:
    """Extract ``#[ignore]`` and ``#[cfg_attr(..., ignore)]`` annotations."""
    return [
        {
            "location": f"{marker.file}:{marker.line}",
            "reason": marker.reason,
            "issues": ", ".join(marker.issues),
        }
        for marker in known_gaps_gate.extract_rust_test_ignores(raw_text, rel)
    ]


def extract_py_decorator_skips_from_text(
    source: str, rel: str
) -> list[dict[str, str]]:
    """Extract ``@pytest.mark.{xfail,skip,skipif}`` decorators using ``ast``."""
    return [
        {
            "location": f"{marker.file}:{marker.line}",
            "reason": marker.reason,
            "issues": ", ".join(marker.issues),
        }
        for marker in known_gaps_gate.extract_python_test_markers(source, rel)
    ]


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


def collect_test_taxonomy(
    repo_root: pathlib.Path,
    manifest: port_manifest.PortManifest | None = None,
) -> dict[str, Any]:
    """Collect W4 file-level taxonomy and listed routine oracle counts."""
    tax_path = repo_root / test_taxonomy.DEFAULT_TAXONOMY_REL
    if not tax_path.is_file():
        return {
            "total_files": 0,
            "total_tests": 0,
            "by_kind": {k: {"files": 0} for k in test_taxonomy.VALID_KINDS},
            "by_language": {
                lang: {"files": 0, "tests": 0}
                for lang in test_taxonomy.VALID_LANGUAGES
            },
            "mislabeled_count": 0,
            "mislabeled_files": [],
            "mapped_routines_total": 0,
            "listed_oracle_routines_total": 0,
            "listed_oracle_routines_pct": 0.0,
            "listed_oracle_routine_ids": [],
            "by_module": [],
        }
    taxonomy = test_taxonomy.load_test_taxonomy(tax_path)
    return test_taxonomy.summarize_test_taxonomy(repo_root, taxonomy, manifest)
