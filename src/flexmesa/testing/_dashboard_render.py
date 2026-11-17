# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deterministic Markdown and JSON renderers (`#174`)."""

from __future__ import annotations

from typing import Any

from flexmesa.testing import port_manifest

BEGIN_MARKER = "<!-- BEGIN GENERATED DASHBOARD -->"
END_MARKER = "<!-- END GENERATED DASHBOARD -->"
BEGIN_SUMMARY_MARKER = "<!-- BEGIN GENERATED DASHBOARD SUMMARY -->"
END_SUMMARY_MARKER = "<!-- END GENERATED DASHBOARD SUMMARY -->"

REPO_URL = "https://github.com/joshspeagle/FlexMESA"


def _sum_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    up = sum(int(r["upstream_lines"]) for r in rows)
    ported = sum(int(r["ported_total"]) for r in rows)
    verified = sum(int(r["verified_total"]) for r in rows)
    return {
        "upstream_files": sum(int(r["upstream_files"]) for r in rows),
        "upstream_lines": up,
        "absent": sum(int(r["absent"]) for r in rows),
        "stub": sum(int(r["stub"]) for r in rows),
        "partial": sum(int(r["partial"]) for r in rows),
        "ported_unverified": sum(int(r["ported_unverified"]) for r in rows),
        "kernel_reference_vectors": sum(
            int(r["kernel_reference_vectors"]) for r in rows
        ),
        "case_level_evidence": sum(int(r["case_level_evidence"]) for r in rows),
        "ported_total": ported,
        "verified_total": verified,
        "pct_ported": round(100.0 * ported / up, 1) if up else 0.0,
        "pct_verified": round(100.0 * verified / up, 1) if up else 0.0,
        "rust_lines": sum(int(r["rust_lines"]) for r in rows),
        "python_jax_lines": sum(int(r["python_jax_lines"]) for r in rows),
    }


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
        "reached_termination",
        "within_numeric_tol",
        "passing",
        "declared_gaps",
        "undeclared_failures",
    )
    return {
        k: sum(int(hh_suites[s][k]) for s in ("star", "binary", "astero"))
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


def _render_headline_and_modules(data: dict[str, Any]) -> list[str]:
    mod_cov = data["module_coverage"]
    rows: list[dict[str, Any]] = mod_cov["rows"]
    engine_tot = _sum_rows(
        [r for r in rows if r["module"] in port_manifest.CORE_ENGINE_MODULES]
    )
    non_plot_engine = _sum_rows(
        [r for r in rows if r["module"] in {"star", "binary", "astero"}]
    )
    all_tot = _sum_rows(rows)
    hh_tot = _summarize_harness(data["harness"]["by_suite"])
    rs_ov = data["coverage"]["rust"]["overall"]
    py_ov = data["coverage"]["python"]["overall"]
    num = data["numerics"]
    gh = data["github"]

    closed_140, open_140 = _unique_140_counts(gh)
    def_items = gh["deferred_169"]
    def_bugs = sum(1 for x in def_items if x["kind"] == "upstream-bug")
    def_enh = sum(1 for x in def_items if x["kind"] == "enhancement")
    enh_lbl = "enhancement" if def_enh == 1 else "enhancements"
    def_owed = sum(1 for x in def_items if x["parity_pending"])
    open_prs = gh["open_pull_requests"]

    pm_link = (
        f"[`port_manifest.toml`]"
        f"({REPO_URL}/blob/main/benchmarks/dashboard/port_manifest.toml)"
    )
    gh_link = (
        f"[`github_snapshot.json`]"
        f"({REPO_URL}/blob/main/benchmarks/dashboard/github_snapshot.json)"
    )
    cov_link = (
        f"[`coverage_summary.json`]"
        f"({REPO_URL}/blob/main/benchmarks/dashboard/coverage_summary.json)"
    )
    ct_link = (
        f"[`case_table.md`]({REPO_URL}/blob/main/notes/audit/case_table.md)"
    )
    hh_link = (
        f"[`benchmarks/history_harness/`]"
        f"({REPO_URL}/tree/main/benchmarks/history_harness)"
    )

    lines = [
        "Generated mechanically by `flexmesa dashboard render` from "
        f"{pm_link}, {hh_link}, `cargo llvm-cov`, `pytest --cov`, and the "
        f"GitHub API ([#174]({REPO_URL}/issues/174)). Reference: "
        "**MESA release `26.4.1`** (git tag `26.4.1`, commit `c8c2026`, "
        "Zenodo `19722306`).",
        "",
        "### Headline Status",
        "",
        "| Metric | Value | Source |",
        "| :--- | ---: | :--- |",
        (
            "| `star` + `star_data` + `binary` + `astero` ported (unverified) "
            f"| {engine_tot['ported_total']:,} / "
            f"{engine_tot['upstream_lines']:,} ({engine_tot['pct_ported']:.1f}"
            f"%; `star`+`binary`+`astero`: {non_plot_engine['pct_ported']:.1f}"
            f"%) | {pm_link} |"
        ),
        (
            "| Upstream lines verified (all 26 modules) | "
            f"{all_tot['verified_total']:,} / {all_tot['upstream_lines']:,} "
            f"({all_tot['pct_verified']:.1f}%) | {pm_link} |"
        ),
        (
            "| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | "
            f"{hh_tot['passing']} / {hh_tot['total']} "
            f"({hh_tot['declared_gaps']} declared gaps, "
            f"{hh_tot['no_reference_yet']} no reference yet, "
            f"{hh_tot['undeclared_failures']} undeclared failures) | "
            f"{ct_link} |"
        ),
        (
            "| Strict mode bit-identical across machines (`#165`) | "
            f"{num['strict_cross_machine_identical']} | "
            f"[#165]({REPO_URL}/issues/165) |"
        ),
        (
            "| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | "
            f"{num['strict_kernels_bit_identical']} | "
            f"[#165]({REPO_URL}/issues/165) |"
        ),
        (
            "| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | "
            f"{rs_ov['percent']:.1f}% ({rs_ov['covered']:,}/"
            f"{rs_ov['lines']:,}) / {py_ov['percent']:.1f}% "
            f"({py_ov['covered']:,}/{py_ov['statements']:,}) | {cov_link} |"
        ),
        (
            f"| Road-to-`v0.1.0` packages ([#140]({REPO_URL}/issues/140)) & "
            f"open PRs | {closed_140} closed, {open_140} open; "
            f"{len(open_prs)} open PR(s) | {gh_link} |"
        ),
        (
            f"| Post-parity deferred backlog ([#169]({REPO_URL}/issues/169)) | "
            f"{len(def_items)} total ({def_bugs} upstream bugs, {def_enh} "
            f"{enh_lbl}, {def_owed} `parity-pending`) | {gh_link} |"
        ),
        "",
        "### 1. Port Coverage by Module (MESA `26.4.1`)",
        "",
        "Progress is measured by **upstream MESA `26.4.1` source lines by "
        "status**, never by lines of Rust or Python written. No file starts "
        "as verified; raising a status requires a pull request that supplies "
        "the corresponding routine counterparts or generated MESA `26.4.1` "
        "evidence.",
        "",
        (
            "| Module | Upstream LOC | Absent | Stub | Partial | "
            "Ported (unverified) | Kernel ref. vectors | Case-level evidence "
            "| Rust LOC | Python/JAX LOC | % Ported | % Verified |"
        ),
        (
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for r in rows:
        lines.append(_fmt_mod_row(str(r["module"]), r))
    lines.append(
        _fmt_mod_row(
            "Subtotal: star + star_data + binary + astero",
            engine_tot,
            bold=True,
        )
    )
    lines.append(_fmt_mod_row("Total (all 26 modules)", all_tot, bold=True))

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
            "(`flexmesa.testing`, `flexmesa.bench`). For plotting "
            f"([#173]({REPO_URL}/issues/173)), `flexmesa.plotting` currently "
            f"provides `{panels['flexmesa_plot_builders']}` `plot_*` builders "
            f"against `{panels['upstream_panel_modules']}` upstream "
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

    lines = [
        "",
        "### 2. Verification & Numerics Status",
        "",
        "| Check | Status | Notes |",
        "| :--- | :--- | :--- |",
        (
            "| Strict mode bit-identical across machines & thread counts "
            f"(Level 1) | `{num['strict_cross_machine_identical']}` | "
            f"Awaiting [#165]({REPO_URL}/issues/165) CI jobs "
            "(`x86_64` vs `aarch64`) |"
        ),
        (
            "| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs "
            f"(Level 2) | `{num['strict_kernels_bit_identical']}` | "
            f"`{num['total_upstream_hex_values']:,}` exact IEEE-754 values "
            "captured across 23 in-scope module test suites "
            "(`mesa-module-hex-v1`); strict comparison lands in "
            f"[#165]({REPO_URL}/issues/165) |"
        ),
    ]
    if detailed:
        lines.extend(
            [
                "",
                (
                    "| Module | Upstream `26.4.1` Exact Values "
                    "(`test_output` / hex) | Strict Bit-Identical Compared "
                    "(`#165`) |"
                ),
                "| :--- | ---: | :--- |",
            ]
        )
        for m in num["modules"]:
            lines.append(
                f"| `{m['module']}` | {m['upstream_exact_values']:,} | "
                f"`{m['exact_strict_compared']}` |"
            )

    lines.extend(
        [
            "",
            "### 3. End-to-End Cases (`benchmarks/history_harness`)",
            "",
            (
                "| Suite | `26.4.1` Cases | Stored `26.4.1` Refs | No Ref Yet "
                "| Reach Upstream Term | Within Numeric Tol | Verified Pass "
                "| Declared Gaps | Undeclared Failures |"
            ),
            (
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: |"
            ),
        ]
    )
    for s in ("star", "binary", "astero"):
        sr = hh_suites[s]
        lines.append(
            f"| `{s}` | {sr['total']} | {sr['mesa_26_4_1_ref']} | "
            f"{sr['no_reference_yet']} | {sr['reached_termination']} | "
            f"{sr['within_numeric_tol']} | {sr['passing']} | "
            f"{sr['declared_gaps']} | {sr['undeclared_failures']} |"
        )
    lines.append(
        f"| **Total** | **{hh_tot['total']}** | "
        f"**{hh_tot['mesa_26_4_1_ref']}** | "
        f"**{hh_tot['no_reference_yet']}** | "
        f"**{hh_tot['reached_termination']}** | "
        f"**{hh_tot['within_numeric_tol']}** | **{hh_tot['passing']}** | "
        f"**{hh_tot['declared_gaps']}** | "
        f"**{hh_tot['undeclared_failures']}** |"
    )
    gap_parts = [
        f"[{iss}]({REPO_URL}/issues/{iss[1:]}) ({cnt})"
        for iss, cnt in hh["gaps_by_issue"].items()
    ]
    lines.extend(
        [
            "",
            "- **Full per-case table**: "
            f"[`notes/audit/case_table.md`]"
            f"({REPO_URL}/blob/main/notes/audit/case_table.md). "
            f"The `{hh_tot['within_numeric_tol']}` cases currently within "
            "numeric tolerance (`rsp_BEP`, `rsp_BLAP`, `rsp_Cepheid`, "
            "`rsp_RR_Lyrae`, `rsp_save_and_load_file`) still invoke "
            "library-side `Tier4CaseExtras` and are governed as declared "
            f"gaps under [#148]({REPO_URL}/issues/148).",
            f"- **Declared gaps by owning package**: {', '.join(gap_parts)}.",
            "- **Trajectory comparison figures "
            f"([#173]({REPO_URL}/issues/173))**: "
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

    lines = [
        "",
        "### 4. Controls & Defaults Coverage (`#147`, `#149`)",
        "",
        (
            "| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` "
            "| Defaults Agree | Value Mismatch | Active Limit `None` "
            "| Unimplemented |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for crow in data["controls"]:
        lines.append(
            f"| `{crow['namelist']}` | {crow['upstream_scalars']:,} | "
            f"{crow['matched_in_flexmesa']:,} | {crow['defaults_agree']:,} | "
            f"{crow['value_mismatch']:,} | {crow['disabled_limit_none']:,} | "
            f"{crow['unimplemented']:,} |"
        )
    diff_url = f"{REPO_URL}/blob/main/notes/audit/defaults_diff_26_4_1.md"
    uniq_ignore_nums: set[int] = set()
    for x in skips["rust_ignores"]:
        for tok in str(x["issues"]).split(","):
            clean = tok.strip().lstrip("#")
            if clean.isdigit():
                uniq_ignore_nums.add(int(clean))
    ignore_issues = (
        ", ".join(f"#{n}" for n in sorted(uniq_ignore_nums)) or "none"
    )
    lines.extend(
        [
            "",
            f"Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`]"
            f"({diff_url}) (loud errors on unknown controls tracked in "
            f"[#147]({REPO_URL}/issues/147); defaults alignment tracked in "
            f"[#149]({REPO_URL}/issues/149)).",
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
            f"Rust `#[ignore]` ({ignore_issues}) and "
            f"`{skips['python_xfail_or_skip_count']}` Python "
            f"`xfail`/`skip` (`{len(skips['unlinked_locations'])}` unlinked).",
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
            ]
        )
        for cr in cov["rust"]["by_crate"]:
            lines.append(
                f"| `{cr['crate']}` | {cr['lines']:,} | {cr['covered']:,} | "
                f"{cr['percent']:.1f}% |"
            )
        lines.extend(
            [
                "",
                "#### Python Statement Coverage by Subpackage (`pytest --cov`)",
                "",
                "| Subpackage | Statements | Covered | Coverage |",
                "| :--- | ---: | ---: | ---: |",
            ]
        )
        for sp in cov["python"]["by_subpackage"]:
            lines.append(
                f"| `{sp['subpackage']}` | {sp['statements']:,} | "
                f"{sp['covered']:,} | {sp['percent']:.1f}% |"
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
            f"[#{it['number']}]({REPO_URL}/issues/{it['number']})"
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
    lines.extend(
        [
            "",
            f"- **Open pull requests ({len(open_prs)})**: "
            f"{', '.join(pr_links) if pr_links else 'none'}.",
            f"- **Post-parity deferred backlog "
            f"([#169]({REPO_URL}/issues/169))**: `{len(def_items)}` issues "
            f"(`{def_bugs}` `upstream-bug`, `{def_enh}` `enhancement`; "
            f"`{def_owed}` marked `parity-pending` for upstream restoration "
            "before `v0.1.0`).",
            "",
            "### 7. Headline History Across Merges",
            "",
            (
                "| Date (UTC) | Commit | Event | Engine % Ported "
                "| All % Verified | Passing Cases | Rust Cov | Python Cov "
                "| `#140` Closed/Total | `#169` Backlog |"
            ),
            (
                "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: |"
            ),
        ]
    )
    hist_rows: list[dict[str, str]] = data["history"]
    shown_hist = hist_rows if detailed else hist_rows[-5:]
    for hr in shown_hist:
        lines.append(
            f"| `{hr['date_utc']}` | `{hr['commit']}` | {hr['event']} | "
            f"{hr['engine_ported_pct']}% | {hr['all_verified_pct']}% | "
            f"{hr['cases_passing']}/106 | {hr['rust_cov_pct']}% | "
            f"{hr['python_cov_pct']}% | {hr['closed_140']}/{hr['total_140']} | "
            f"{hr['deferred_169']} |"
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
    mod_cov = data["module_coverage"]["rows"]
    eng = _sum_rows(
        [r for r in mod_cov if r["module"] in port_manifest.CORE_ENGINE_MODULES]
    )
    all_tot = _sum_rows(mod_cov)
    hh_tot = _summarize_harness(data["harness"]["by_suite"])
    cov = data["coverage"]
    rs_pct = cov["rust"]["overall"]["percent"]
    py_pct = cov["python"]["overall"]["percent"]
    closed_140, open_140 = _unique_140_counts(data["github"])
    tot_140 = closed_140 + open_140
    return (
        f"{BEGIN_SUMMARY_MARKER}\n"
        '!!! info "Development Progress Toward MESA `26.4.1` Parity '
        '(`v0.1.0`)"\n'
        "    `FlexMESA` is in active parity verification against "
        f"**MESA release `26.4.1`** ([#140]({REPO_URL}/issues/140)). Current "
        f"script-generated status: **{eng['pct_ported']:.1f}%** of "
        "`star`/`star_data`/`binary`/`astero` upstream lines ported "
        f"(`{all_tot['pct_verified']:.1f}%` verified), "
        f"**{hh_tot['passing']} / 106** end-to-end test-suite cases verified "
        f"passing (`{hh_tot['declared_gaps']}` declared gaps, `0` undeclared "
        f"failures), **{rs_pct:.1f}%** Rust / **{py_pct:.1f}%** Python line "
        f"coverage, and **{closed_140} / {tot_140}** `#140` packages closed. "
        "See the [Live Progress & Verification Dashboard]"
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
