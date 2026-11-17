# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data collectors for the generated FlexMESA progress dashboard (`#174`)."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess
import tomllib
import urllib.request
from typing import Any

from flexmesa.testing import port_manifest

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

_IGNORE_ATTR_RE = re.compile(
    r'^\s*#\s*\[\s*ignore(?:\s*=\s*"([^"]*)")?\s*\]',
    flags=re.MULTILINE,
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
        rs_loc = _count_dir_lines(repo_root, rs_dirs, ".rs")
        py_loc = _count_dir_lines(repo_root, py_dirs, ".py")
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
                "rust_lines": rs_loc,
                "python_jax_lines": py_loc,
            }
        )

    # Plotting panel specification count (#173)
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
        "plotting_panels": {
            "flexmesa_plot_builders": len(plot_fns),
            "upstream_panel_modules": pgstar_panels,
        },
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
    if has_165_ci and strict_report.is_file():
        rep = json.loads(strict_report.read_text(encoding="utf-8"))
        level1 = str(rep.get("cross_machine_status", "not yet measured"))
        level2 = str(
            rep.get("kernels_bit_identical_status", "not yet measured")
        )
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
    mod_rows: list[dict[str, Any]] = []
    for item in hex_data.get("modules", []):
        mname = str(item["module"])
        if mname in {"forum", "sample"}:
            continue
        mod_rows.append(
            {
                "module": mname,
                "upstream_exact_values": int(
                    item.get("exact_values_captured", 0)
                ),
                "exact_strict_compared": (
                    "not yet measured" if level2 == "not yet measured" else "0"
                ),
            }
        )
    return {
        "strict_cross_machine_identical": level1,
        "strict_kernels_bit_identical": level2,
        "total_upstream_hex_values": int(
            hex_data.get("total_exact_values_captured", 0)
        ),
        "modules": mod_rows,
    }


def collect_harness_summary(repo_root: pathlib.Path) -> dict[str, Any]:
    """Summarize the 106-case history-harness output (`#143`)."""
    hh_dir = repo_root / "benchmarks" / "history_harness"
    manifest_raw = tomllib.loads(
        (hh_dir / "case_manifest.toml").read_text(encoding="utf-8")
    )
    gaps_raw = tomllib.loads(
        (hh_dir / "known_gaps.toml").read_text(encoding="utf-8")
    )
    run_raw: dict[str, Any] = json.loads(
        (hh_dir / "baseline_run_26_4_1.json").read_text(encoding="utf-8")
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
        for s in ("star", "binary", "astero")
    }
    for c in manifest_raw.get("cases", []):
        suite = str(c["suite"])
        by_suite[suite]["total"] += 1
        ref_st = str(c["reference_status"])
        if ref_st == "mesa_26_4_1":
            by_suite[suite]["mesa_26_4_1_ref"] += 1
        elif ref_st == "no_reference_yet":
            by_suite[suite]["no_reference_yet"] += 1

    for rec in run_raw.get("cases", []):
        suite = str(rec["suite"])
        verdict = str(rec.get("verdict", ""))
        comp = rec.get("comparison")
        if (
            rec.get("executed") is True
            and rec.get("termination_matches") is True
        ):
            by_suite[suite]["reached_termination"] += 1
            if (
                isinstance(comp, dict)
                and comp.get("within_final_tolerances") is True
                and comp.get("within_track_tolerances") is True
            ):
                by_suite[suite]["within_numeric_tol"] += 1
        if verdict == "pass":
            by_suite[suite]["passing"] += 1
        elif verdict.startswith("declared gap"):
            by_suite[suite]["declared_gaps"] += 1
        elif rec.get("is_undeclared_failure") is True:
            by_suite[suite]["undeclared_failures"] += 1

    gaps_by_issue: dict[str, int] = {}
    for g in gaps_raw.get("gaps", []):
        raw_iss = str(g["issue"]).strip().lstrip("#")
        iss = f"#{int(raw_iss)}"
        gaps_by_issue[iss] = gaps_by_issue.get(iss, 0) + 1

    fig_dir = hh_dir / "figures"
    fig_count = len(list(fig_dir.glob("*.png"))) if fig_dir.is_dir() else 0
    return {
        "by_suite": by_suite,
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


def _extract_py_decorator_skips(
    py_path: pathlib.Path, rel: str
) -> list[dict[str, str]]:
    """Extract ``@pytest.mark.{xfail,skip}`` decorators using ``ast``."""
    source = py_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel)
    found: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        for dec in node.decorator_list:
            seg = ast.get_source_segment(source, dec) or ""
            if not re.search(r"\bmark\.(?:xfail|skip)\b", seg):
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
        raw = rs_path.read_text(encoding="utf-8")
        cleaned = "\n".join(
            "" if ln.lstrip().startswith("//") else ln
            for ln in raw.splitlines()
        )
        for m in _IGNORE_ATTR_RE.finditer(cleaned):
            lno = cleaned.count("\n", 0, m.start()) + 1
            reason = m.group(1) or ""
            issues = sorted({f"#{x}" for x in _ISSUE_RE.findall(reason)})
            rust_ignores.append(
                {
                    "location": f"{rel}:{lno}",
                    "reason": reason,
                    "issues": ", ".join(issues),
                }
            )

    py_skips: list[dict[str, str]] = []
    for root_sub in ("tests", "src/flexmesa"):
        for py_path in sorted((repo_root / root_sub).rglob("*.py")):
            if py_path.name == "conftest.py":
                continue
            rel = py_path.relative_to(repo_root).as_posix()
            py_skips.extend(_extract_py_decorator_skips(py_path, rel))

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


def _gh_api_json(endpoint: str) -> Any:
    """Fetch JSON from GitHub REST API via ``GITHUB_TOKEN`` or ``gh api``."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        url = f"https://api.github.com/{endpoint.lstrip('/')}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "flexmesa-dashboard",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    proc = subprocess.run(
        ["gh", "api", endpoint],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def sync_github_snapshot(
    snapshot_path: pathlib.Path, repo: str = "joshspeagle/FlexMESA"
) -> dict[str, Any]:
    """Query GitHub API for `#140`, `#169`, and open PRs into JSON."""
    issue_140 = _gh_api_json(f"repos/{repo}/issues/140")
    issue_169 = _gh_api_json(f"repos/{repo}/issues/169")

    # Fetch all issues (open + closed, paginated)
    all_issues: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = _gh_api_json(
            f"repos/{repo}/issues?state=all&per_page=100&page={page}"
        )
        if not isinstance(batch, list) or not batch:
            break
        all_issues.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    issues_by_num: dict[int, dict[str, Any]] = {}
    open_prs: list[dict[str, Any]] = []
    for item in all_issues:
        num = int(item["number"])
        if "pull_request" in item:
            if item.get("state") == "open":
                open_prs.append(
                    {
                        "number": num,
                        "title": str(item["title"]),
                        "draft": bool(item.get("draft", False)),
                    }
                )
        else:
            issues_by_num[num] = item

    # Extract #140 sub-issues by phase from #140 body
    body_140 = str(issue_140.get("body", ""))
    phase_order = [
        "Phase 0",
        "Phase A",
        "Phase B",
        "Phase C",
        "Phase D",
        "Phase E",
        "Deferred scope",
    ]
    phase_issues: dict[str, list[int]] = {p: [] for p in phase_order}
    current_phase: str | None = None
    in_work_packages = False
    for line in body_140.splitlines():
        if line.startswith("## Work packages"):
            in_work_packages = True
            continue
        if in_work_packages and line.startswith("## "):
            break
        if not in_work_packages:
            continue
        if line.startswith("### "):
            header = line[4:].strip()
            for p in phase_order[:6]:
                if header.startswith(p):
                    current_phase = p
                    break
            else:
                current_phase = None
            continue
        if current_phase is not None and line.lstrip().startswith("- ["):
            m = re.search(r"#(\d+)\b", line)
            if m is not None:
                inum = int(m.group(1))
                if inum not in phase_issues[current_phase]:
                    phase_issues[current_phase].append(inum)

    # Deferred scope issues (#172 or label deferred-scope)
    seen_140: set[int] = set()
    for nums in phase_issues.values():
        seen_140.update(nums)
    for num, item in sorted(issues_by_num.items()):
        labels = {str(lb["name"]) for lb in item.get("labels", [])}
        if "deferred-scope" in labels and num not in seen_140:
            phase_issues["Deferred scope"].append(num)
            seen_140.add(num)

    phases_payload: list[dict[str, Any]] = []
    for pname in phase_order:
        items_out: list[dict[str, Any]] = []
        for inum in sorted(phase_issues[pname]):
            rec = issues_by_num.get(inum)
            if rec is None:
                continue
            items_out.append(
                {
                    "number": inum,
                    "title": str(rec["title"]),
                    "state": str(rec["state"]),
                }
            )
        phases_payload.append(
            {
                "phase": pname,
                "open": sum(1 for x in items_out if x["state"] == "open"),
                "closed": sum(1 for x in items_out if x["state"] == "closed"),
                "issues": items_out,
            }
        )

    # Extract #169 deferred backlog issues (label 'deferred' or listed in #169)
    body_169 = str(issue_169.get("body", ""))
    nums_in_169 = {
        int(m.group(1))
        for line in body_169.splitlines()
        if line.startswith("| #") and (m := re.match(r"^\|\s*#(\d+)\b", line))
    }
    deferred_items: list[dict[str, Any]] = []
    for num, item in sorted(issues_by_num.items()):
        if num in {140, 169, 172}:
            continue
        label_set = {str(lb["name"]) for lb in item.get("labels", [])}
        if "deferred" in label_set or num in nums_in_169:
            kind = (
                "enhancement" if "enhancement" in label_set else "upstream-bug"
            )
            deferred_items.append(
                {
                    "number": num,
                    "title": str(item["title"]),
                    "state": str(item["state"]),
                    "kind": kind,
                    "parity_pending": "parity-pending" in label_set,
                    "needs_confirmation": "needs-confirmation" in label_set,
                }
            )

    payload = {
        "repo": repo,
        "tracker_issue": 140,
        "backlog_issue": 169,
        "phases_140": phases_payload,
        "open_pull_requests": sorted(open_prs, key=lambda x: int(x["number"])),
        "deferred_169": deferred_items,
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
