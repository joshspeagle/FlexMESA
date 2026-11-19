# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deterministic Markdown and JSON renderers (`#174`)."""

from __future__ import annotations

from typing import Any

from flexmesa.data import fetch
from flexmesa.testing import case_manifest, port_manifest

BEGIN_MARKER = "<!-- BEGIN GENERATED DASHBOARD -->"
END_MARKER = "<!-- END GENERATED DASHBOARD -->"
BEGIN_SUMMARY_MARKER = "<!-- BEGIN GENERATED DASHBOARD SUMMARY -->"
END_SUMMARY_MARKER = "<!-- END GENERATED DASHBOARD SUMMARY -->"

REPO_URL = "https://github.com/joshspeagle/FlexMESA"
_ZENODO_RECORD = fetch.load_upstream_reference().zenodo_record


def _blob(label: str, rel: str) -> str:
    return f"[{label}]({REPO_URL}/blob/main/{rel})"


def _issue_link(num: int | str) -> str:
    clean = str(num).lstrip("#")
    return f"[#{clean}]({REPO_URL}/issues/{clean})"


def _sum_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    int_keys = (
        "upstream_files",
        "upstream_lines",
        "absent",
        "stub",
        "partial",
        "ported_unverified",
        "kernel_reference_vectors",
        "case_level_evidence",
        "ported_total",
        "verified_total",
        "rust_lines",
        "python_jax_lines",
    )
    out: dict[str, Any] = {k: sum(int(r[k]) for r in rows) for k in int_keys}
    up = int(out["upstream_lines"])
    out["pct_ported"] = (
        round(100.0 * int(out["ported_total"]) / up, 1) if up else 0.0
    )
    out["pct_verified"] = (
        round(100.0 * int(out["verified_total"]) / up, 1) if up else 0.0
    )
    return out


def _fmt_mod_row(label: str, r: dict[str, Any], bold: bool = False) -> str:
    tag = f"**{label}**" if bold else f"`{label}`"
    return (
        f"| {tag} | {r['upstream_lines']:,} | {r['absent']:,} | "
        f"{r['stub']:,} | {r['partial']:,} | {r['ported_unverified']:,} | "
        f"{r['kernel_reference_vectors']:,} | {r['case_level_evidence']:,} | "
        f"{r['rust_lines']:,} | {r['python_jax_lines']:,} | "
        f"{r['pct_ported']:.1f}% | {r['pct_verified']:.1f}% |"
    )


def _summarize_harness(hh_suites: dict[str, Any]) -> dict[str, int]:
    keys = (
        "total",
        "mesa_26_4_1_ref",
        "no_reference_yet",
        "strict_validated",
        "diagnostic_runs",
        "reached_termination",
        "within_numeric_tol",
        "passing",
        "declared_gaps",
        "undeclared_failures",
    )
    return {
        k: sum(int(hh_suites[s][k]) for s in case_manifest.VALID_SUITES)
        for k in keys
    }


def _unique_140_counts(gh: dict[str, Any]) -> tuple[int, int]:
    uniq: dict[int, str] = {}
    for ph in gh["phases_140"]:
        for it in ph["issues"]:
            uniq[int(it["number"])] = str(it["state"])
    closed_n = sum(1 for st in uniq.values() if st == "closed")
    open_n = sum(1 for st in uniq.values() if st == "open")
    return closed_n, open_n


def summarize_headline(data: dict[str, Any]) -> dict[str, Any]:
    """Compute canonical headline metrics shared across views and history."""
    rows: list[dict[str, Any]] = data["module_coverage"]["rows"]
    engine_tot = _sum_rows(
        [r for r in rows if r["module"] in port_manifest.CORE_ENGINE_MODULES]
    )
    non_plot_engine = _sum_rows(
        [r for r in rows if r["module"] in {"star", "binary", "astero"}]
    )
    all_tot = _sum_rows(rows)
    hh_tot = _summarize_harness(data["harness"]["by_suite"])
    closed_140, open_140 = _unique_140_counts(data["github"])
    return {
        "engine_tot": engine_tot,
        "non_plot_engine": non_plot_engine,
        "all_tot": all_tot,
        "hh_tot": hh_tot,
        "closed_140": closed_140,
        "open_140": open_140,
        "total_140": closed_140 + open_140,
        "deferred_169_count": len(data["github"]["deferred_169"]),
    }


def _collect_issue_links(items: list[dict[str, str]]) -> str:
    nums: set[int] = set()
    for x in items:
        for tok in str(x.get("issues", "")).split(","):
            clean = tok.strip().lstrip("#")
            if clean.isdigit():
                nums.add(int(clean))
    if not nums:
        return "none"
    return ", ".join(_issue_link(n) for n in sorted(nums))


def _render_headline_and_modules(data: dict[str, Any]) -> list[str]:
    mod_cov = data["module_coverage"]
    rows: list[dict[str, Any]] = mod_cov["rows"]
    head = summarize_headline(data)
    eng, non_plot, all_tot, hh_tot = (
        head["engine_tot"],
        head["non_plot_engine"],
        head["all_tot"],
        head["hh_tot"],
    )
    hh_suites = data["harness"]["by_suite"]
    rs_ov = data["coverage"]["rust"]["overall"]
    py_ov = data["coverage"]["python"]["overall"]
    num = data["numerics"]
    gh = data["github"]
    up_ref = data["upstream_reference"]

    def_items = gh["deferred_169"]
    def_bugs = sum(1 for x in def_items if x["kind"] == "upstream-bug")
    def_enh = sum(1 for x in def_items if x["kind"] == "enhancement")
    enh_lbl = "enhancement" if def_enh == 1 else "enhancements"
    def_owed = sum(1 for x in def_items if x["parity_pending"])
    open_prs = gh["open_pull_requests"]
    suite_split = "/".join(
        f"`{hh_suites[s]['total']}`" for s in case_manifest.VALID_SUITES
    )

    pm_link = _blob(
        "`port_manifest.toml`", "benchmarks/dashboard/port_manifest.toml"
    )
    gh_link = _blob(
        "`github_snapshot.json`", "benchmarks/dashboard/github_snapshot.json"
    )
    cov_link = _blob(
        "`coverage_summary.json`", "benchmarks/dashboard/coverage_summary.json"
    )
    ct_link = _blob("`case_table.md`", "notes/audit/case_table.md")
    hh_link = (
        f"[`benchmarks/history_harness/`]"
        f"({REPO_URL}/tree/main/benchmarks/history_harness)"
    )

    lines = [
        "Generated mechanically by `flexmesa dashboard render` from "
        f"{pm_link}, {hh_link}, `cargo llvm-cov`, `pytest --cov`, and the "
        f"GitHub API ({_issue_link(174)}). Reference: "
        f"**MESA release `{up_ref['version'].lstrip('r')}`** "
        f"(commit `{up_ref['commit'][:7]}`, Zenodo `{_ZENODO_RECORD}`).",
        "",
        "### Headline Status",
        "",
        "| Metric | Value | Source |",
        "| :--- | ---: | :--- |",
        f"| `star` + `star_data` + `binary` + `astero` ported (unverified) | "
        f"{eng['ported_total']:,} / {eng['upstream_lines']:,} "
        f"({eng['pct_ported']:.1f}%; `star`+`binary`+`astero`: "
        f"{non_plot['pct_ported']:.1f}%) | {pm_link} |",
        f"| Upstream lines verified (all {len(rows)} modules) | "
        f"{all_tot['verified_total']:,} / {all_tot['upstream_lines']:,} "
        f"({all_tot['pct_verified']:.1f}%) | {pm_link} |",
        f"| End-to-end MESA `26.4.1` cases passing ({suite_split}) | "
        f"{hh_tot['passing']} / {hh_tot['total']} ({hh_tot['declared_gaps']} "
        f"declared gaps, {hh_tot['no_reference_yet']} no reference yet, "
        f"{hh_tot['undeclared_failures']} undeclared failures) | {ct_link} |",
        f"| Strict mode bit-identical across machines (`#165`) | "
        f"{num['strict_cross_machine_identical']} | {_issue_link(165)} |",
        f"| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | "
        f"{num['strict_kernels_bit_identical']} | {_issue_link(165)} |",
        f"| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | "
        f"{rs_ov['percent']:.1f}% ({rs_ov['covered']:,}/{rs_ov['lines']:,}) / "
        f"{py_ov['percent']:.1f}% ({py_ov['covered']:,}/"
        f"{py_ov['statements']:,}) | {cov_link} |",
        f"| Road-to-`v0.1.0` packages ({_issue_link(140)}) & open PRs | "
        f"{head['closed_140']} closed, {head['open_140']} open; "
        f"{len(open_prs)} open PR(s) | {gh_link} |",
        f"| Post-parity deferred backlog ({_issue_link(169)}) | "
        f"{len(def_items)} open ({def_bugs} upstream bugs, {def_enh} "
        f"{enh_lbl}, {def_owed} `parity-pending`) | {gh_link} |",
        "",
        "### 1. Port Coverage by Module (MESA `26.4.1`)",
        "",
        "Progress is measured by **upstream MESA `26.4.1` source lines by "
        "status**, never by lines of Rust or Python written. No file starts "
        "as verified; raising a status requires a pull request that supplies "
        "the corresponding routine counterparts or generated MESA `26.4.1` "
        "evidence.",
        "",
        "| Module | Upstream LOC | Absent | Stub | Partial | "
        "Ported (unverified) | Kernel ref. vectors | Case-level evidence | "
        "Rust LOC | Python/JAX LOC | % Ported | % Verified |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
        *(_fmt_mod_row(str(r["module"]), r) for r in rows),
        _fmt_mod_row(
            "Subtotal: star + star_data + binary + astero", eng, bold=True
        ),
        _fmt_mod_row(f"Total (all {len(rows)} modules)", all_tot, bold=True),
    ]

    panels = mod_cov["plotting_panels"]
    infra = mod_cov["infrastructure_lines"]
    lines.extend(
        [
            "",
            "*Notes*: `star_data` state structs reside in "
            "`crates/flexmesa-star`. Bridge, testing, and benchmark "
            "infrastructure account for an additional "
            f"`{infra['rust_bridge_testing_bench']:,}` Rust lines "
            "(`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and "
            f"`{infra['python_testing_bench']:,}` Python lines "
            f"(`flexmesa.testing`, `flexmesa.bench`). For plotting "
            f"({_issue_link(173)}), `flexmesa.plotting` currently provides "
            f"`{panels['flexmesa_plot_builders']}` `plot_*` builders against "
            f"`{panels['upstream_panel_modules']}` upstream "
            "`pgstar`/`pgbinary` panel modules.",
        ]
    )
    return lines


def _render_numerics_and_harness(
    data: dict[str, Any], detailed: bool
) -> list[str]:
    num = data["numerics"]
    hh = data["harness"]
    hh_suites = hh["by_suite"]
    hh_tot = _summarize_harness(hh_suites)
    n_hex_mods = len(num["modules"])
    has_165 = bool(num.get("has_165_ci"))
    link_165 = _issue_link(165)

    l1_note = (
        f"Verified by {link_165} CI jobs (`x86_64` vs `aarch64`)"
        if has_165
        else f"Awaiting {link_165} CI jobs (`x86_64` vs `aarch64`)"
    )
    l2_tail = (
        f"measured in {link_165}"
        if has_165
        else f"strict comparison lands in {link_165}"
    )

    lines = [
        "",
        "### 2. Verification & Numerics Status",
        "",
        "| Check | Status | Notes |",
        "| :--- | :--- | :--- |",
        f"| Strict mode bit-identical across machines & thread counts "
        f"(Level 1) | `{num['strict_cross_machine_identical']}` | {l1_note} |",
        f"| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs "
        f"(Level 2) | `{num['strict_kernels_bit_identical']}` | "
        f"`{num['total_upstream_hex_values']:,}` exact IEEE-754 values "
        f"captured across {n_hex_mods} in-scope module test suites "
        f"(`mesa-module-hex-v1`); {l2_tail} |",
    ]
    if detailed:
        lines.extend(
            [
                "",
                "| Module | Upstream `26.4.1` Exact Values "
                "(`test_output` / hex) | Strict Bit-Identical Compared "
                "(`#165`) |",
                "| :--- | ---: | :--- |",
                *(
                    f"| `{m['module']}` | {m['upstream_exact_values']:,} | "
                    f"`{m['exact_strict_compared']}` |"
                    for m in num["modules"]
                ),
            ]
        )

    lines.extend(
        [
            "",
            "### 3. End-to-End Cases (`benchmarks/history_harness`)",
            "",
            "| Suite | `26.4.1` Cases | Stored `26.4.1` Refs | No Ref Yet | "
            "Strict-Validated | Diagnostic Runs | Reach Upstream Term | "
            "Within Numeric Tol | Verified Pass | Declared Gaps | "
            "Undeclared Failures |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: |",
        ]
    )
    for s in case_manifest.VALID_SUITES:
        sr = hh_suites[s]
        lines.append(
            f"| `{s}` | {sr['total']} | {sr['mesa_26_4_1_ref']} | "
            f"{sr['no_reference_yet']} | {sr['strict_validated']} | "
            f"{sr['diagnostic_runs']} | {sr['reached_termination']} | "
            f"{sr['within_numeric_tol']} | {sr['passing']} | "
            f"{sr['declared_gaps']} | {sr['undeclared_failures']} |"
        )
    lines.append(
        f"| **Total** | **{hh_tot['total']}** | "
        f"**{hh_tot['mesa_26_4_1_ref']}** | **{hh_tot['no_reference_yet']}** "
        f"| **{hh_tot['strict_validated']}** | "
        f"**{hh_tot['diagnostic_runs']}** | "
        f"**{hh_tot['reached_termination']}** | "
        f"**{hh_tot['within_numeric_tol']}** | **{hh_tot['passing']}** | "
        f"**{hh_tot['declared_gaps']}** | "
        f"**{hh_tot['undeclared_failures']}** |"
    )
    gap_parts = [
        f"{_issue_link(iss)} ({cnt})"
        for iss, cnt in hh["gaps_by_issue"].items()
    ]
    non_pass_within = hh.get("within_tol_non_pass", [])
    case_table_bullet = (
        f"- **Full per-case table**: "
        f"{_blob('`notes/audit/case_table.md`', 'notes/audit/case_table.md')}."
    )
    if non_pass_within:
        case_names = ", ".join(f"`{x['case']}`" for x in non_pass_within)
        issue_nums = sorted(
            {
                int(x["issue"].lstrip("#"))
                for x in non_pass_within
                if x.get("issue", "").lstrip("#").isdigit()
            }
        )
        iss_links = ", ".join(_issue_link(n) for n in issue_nums)
        case_table_bullet += (
            f" `{len(non_pass_within)}` case(s) within numeric tolerance "
            f"({case_names}) remain governed as declared gaps under "
            f"{iss_links}."
        )
    lines.extend(
        [
            "",
            case_table_bullet,
            f"- **Declared gaps by owning package**: {', '.join(gap_parts)}.",
            f"- **Trajectory comparison figures ({_issue_link(173)})**: "
            f"`{hh['comparison_figures_count']}` committed.",
        ]
    )
    return lines


def _render_controls_health_and_work(
    data: dict[str, Any], detailed: bool
) -> list[str]:
    cov = data["coverage"]
    rs_ov = cov["rust"]["overall"]
    py_ov = cov["python"]["overall"]
    skips = data["skips"]
    gh = data["github"]
    hh_tot = _summarize_harness(data["harness"]["by_suite"])

    lines = [
        "",
        "### 4. Controls & Defaults Coverage (`#147`, `#149`)",
        "",
        "| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | "
        "Defaults Agree | Value Mismatch | Active Limit `None` | "
        "Non-Scalar Rust | Unimplemented |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| `{c['namelist']}` | {c['upstream_scalars']:,} | "
            f"{c['matched_in_flexmesa']:,} | {c['defaults_agree']:,} | "
            f"{c['value_mismatch']:,} | {c['disabled_limit_none']:,} | "
            f"{c.get('skipped_unparseable_rust', 0):,} | "
            f"{c['unimplemented']:,} |"
            for c in data["controls"]
        ),
    ]
    diff_link = _blob(
        "`notes/audit/defaults_diff_26_4_1.md`",
        "notes/audit/defaults_diff_26_4_1.md",
    )
    rs_ignore_issues = _collect_issue_links(skips["rust_ignores"])
    py_skip_issues = _collect_issue_links(skips["python_skips"])
    lines.extend(
        [
            "",
            f"Full scalar diff: {diff_link} (loud errors on unknown controls "
            f"tracked in {_issue_link(147)}; defaults alignment tracked in "
            f"{_issue_link(149)}).",
            "",
            "### 5. Test & Code Health",
            "",
            f"- **Rust workspace line coverage (`cargo llvm-cov`)**: "
            f"**{rs_ov['percent']:.1f}%** (`{rs_ov['covered']:,}` / "
            f"`{rs_ov['lines']:,}` lines across "
            f"`{len(cov['rust']['by_crate'])}` crates).",
            f"- **Python package statement coverage (`pytest --cov`)**: "
            f"**{py_ov['percent']:.1f}%** (`{py_ov['covered']:,}` / "
            f"`{py_ov['statements']:,}` statements across "
            f"`{len(cov['python']['by_subpackage'])}` subpackages).",
            f"- **Ignored / skipped tests**: `{skips['rust_ignore_count']}` "
            f"Rust `#[ignore]` ({rs_ignore_issues}) and "
            f"`{skips['python_xfail_or_skip_count']}` Python `xfail`/`skip` "
            f"({py_skip_issues}; `{len(skips['unlinked_locations'])}` "
            "unlinked).",
            "- **Static & documentation gates**: "
            "`cargo clippy --all-targets -- -D warnings`, "
            "`RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, "
            "`ruff format --check`, `mypy --strict`, `mkdocs build --strict`, "
            "and MESA `26.4.1` `file:line` citation resolution enforced in CI.",
        ]
    )

    if detailed:
        lines.extend(
            [
                "",
                "#### Rust Line Coverage by Crate (`cargo llvm-cov`)",
                "",
                "| Crate | Executable Lines | Covered Lines | Line Coverage |",
                "| :--- | ---: | ---: | ---: |",
                *(
                    f"| `{cr['crate']}` | {cr['lines']:,} | "
                    f"{cr['covered']:,} | {cr['percent']:.1f}% |"
                    for cr in cov["rust"]["by_crate"]
                ),
                "",
                "#### Python Statement Coverage by Subpackage (`pytest --cov`)",
                "",
                "| Subpackage | Statements | Covered | Coverage |",
                "| :--- | ---: | ---: | ---: |",
                *(
                    f"| `{sp['subpackage']}` | {sp['statements']:,} | "
                    f"{sp['covered']:,} | {sp['percent']:.1f}% |"
                    for sp in cov["python"]["by_subpackage"]
                ),
            ]
        )

    lines.extend(
        [
            "",
            "### 6. Work Tracking (`#140` & `#169`)",
            "",
            "| `#140` Phase | Closed | Open | Open Issues |",
            "| :--- | ---: | ---: | :--- |",
        ]
    )
    for ph in gh["phases_140"]:
        open_links = [
            _issue_link(it["number"])
            for it in ph["issues"]
            if it["state"] == "open"
        ]
        lines.append(
            f"| {ph['phase']} | {ph['closed']} | {ph['open']} | "
            f"{', '.join(open_links) if open_links else 'none'} |"
        )

    open_prs = gh["open_pull_requests"]
    def_items = gh["deferred_169"]
    def_bugs = sum(1 for x in def_items if x["kind"] == "upstream-bug")
    def_enh = sum(1 for x in def_items if x["kind"] == "enhancement")
    def_owed = sum(1 for x in def_items if x["parity_pending"])
    pr_links = [
        f"[#{p['number']}]({REPO_URL}/pull/{p['number']})"
        + (" (draft)" if p.get("draft") else "")
        for p in open_prs
    ]
    shown_hist = data["history"] if detailed else data["history"][-5:]
    lines.extend(
        [
            "",
            f"- **Open pull requests ({len(open_prs)})**: "
            f"{', '.join(pr_links) if pr_links else 'none'}.",
            f"- **Post-parity deferred backlog ({_issue_link(169)})**: "
            f"`{len(def_items)}` open issues (`{def_bugs}` `upstream-bug`, "
            f"`{def_enh}` `enhancement`; `{def_owed}` marked `parity-pending` "
            "for upstream restoration before `v0.1.0`).",
            "",
            "### 7. Headline History Across Merges",
            "",
            "| Date (UTC) | Commit | Event | Engine % Ported | "
            "All % Verified | Passing Cases | Rust Cov | Python Cov | "
            "`#140` Closed/Total | `#169` Backlog |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |",
            *(
                f"| `{hr['date_utc']}` | `{hr['commit']}` | {hr['event']} | "
                f"{hr['engine_ported_pct']}% | {hr['all_verified_pct']}% | "
                f"{hr['cases_passing']}/{hh_tot['total']} | "
                f"{hr['rust_cov_pct']}% | {hr['python_cov_pct']}% | "
                f"{hr['closed_140']}/{hr['total_140']} | "
                f"{hr['deferred_169']} |"
                for hr in shown_hist
            ),
        ]
    )
    return lines


def _render_shared_sections(data: dict[str, Any], detailed: bool) -> list[str]:
    return [
        *_render_headline_and_modules(data),
        *_render_numerics_and_harness(data, detailed=detailed),
        *_render_controls_health_and_work(data, detailed=detailed),
    ]


def render_readme_block(data: dict[str, Any]) -> str:
    """Render the marked dashboard block embedded in ``README.md``."""
    body = "\n".join(_render_shared_sections(data, detailed=False))
    return f"{BEGIN_MARKER}\n{body}\n{END_MARKER}"


def render_docs_page(data: dict[str, Any]) -> str:
    """Render the complete ``docs/benchmarks/progress-dashboard.md`` page."""
    body = "\n".join(_render_shared_sections(data, detailed=True))
    return (
        "# Live Progress & Verification Dashboard\n\n"
        f"{BEGIN_MARKER}\n{body}\n{END_MARKER}\n"
    )


def render_docs_index_summary(data: dict[str, Any]) -> str:
    """Render the concise live-status admonition on ``docs/index.md``."""
    head = summarize_headline(data)
    eng = head["engine_tot"]
    all_tot = head["all_tot"]
    hh_tot = head["hh_tot"]
    cov = data["coverage"]
    rs_pct = cov["rust"]["overall"]["percent"]
    py_pct = cov["python"]["overall"]["percent"]
    return (
        f"{BEGIN_SUMMARY_MARKER}\n"
        '!!! info "Development Progress Toward MESA `26.4.1` Parity '
        '(`v0.1.0`)"\n'
        "    `FlexMESA` is in active parity verification against "
        f"**MESA release `26.4.1`** ({_issue_link(140)}). Current "
        f"script-generated status: **{eng['pct_ported']:.1f}%** of "
        "`star`/`star_data`/`binary`/`astero` upstream lines ported "
        f"(`{all_tot['pct_verified']:.1f}%` verified), "
        f"**{hh_tot['passing']} / {hh_tot['total']}** end-to-end test-suite "
        f"cases verified passing (`{hh_tot['declared_gaps']}` declared gaps, "
        f"`{hh_tot['undeclared_failures']}` undeclared failures), "
        f"**{rs_pct:.1f}%** Rust / **{py_pct:.1f}%** Python line coverage, "
        f"and **{head['closed_140']} / {head['total_140']}** `#140` packages "
        "closed. See the [Live Progress & Verification Dashboard]"
        "(benchmarks/progress-dashboard.md) for per-module, per-crate, and "
        "per-phase tables.\n"
        f"{END_SUMMARY_MARKER}"
    )


def replace_marked_block(
    document_text: str, replacement_block: str, begin: str, end: str
) -> str:
    """Replace ``begin ... end`` inside ``document_text``."""
    b_idx = document_text.find(begin)
    e_idx = document_text.find(end)
    if b_idx < 0 or e_idx < 0 or e_idx < b_idx:
        raise ValueError(f"Missing markers {begin!r} .. {end!r}")
    e_end = e_idx + len(end)
    return document_text[:b_idx] + replacement_block + document_text[e_end:]
