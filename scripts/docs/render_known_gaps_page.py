# SPDX-License-Identifier: LGPL-3.0-or-later
"""Render or verify the public MkDocs declared-known-gaps page (`#142`, `#144`).

Builds `docs/benchmarks/known-gaps.md` deterministically from
`benchmarks/history_harness/case_manifest.toml`,
`benchmarks/history_harness/known_gaps.toml`, and
`benchmarks/history_harness/baseline_run_26_4_1.json`.
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

from flexmesa.testing import (
    _dashboard_collect,
    _dashboard_render,
    case_manifest,
    case_report,
    harness_runner,
)

try:
    import _common
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _common

GENERATED_COMMENT = _common.generated_comment(
    "scripts/docs/render_known_gaps_page.py", "#142, #144"
)
DEFAULT_OUTPUT = pathlib.Path("docs") / "benchmarks" / "known-gaps.md"

_ISSUE_FOCUS: dict[str, str] = {
    "#148": "Remove fabricated per-case extras from library crates",
    "#150": "Port upstream initial-model, `.mod`, and `.photo` restart I/O",
    "#151": "Port multi-step `relax.f90` model relaxation routines",
    "#152": "Align `evolve.rs`, Newton solver control, and retry loops",
    "#153": "Port upstream `mesh_adjust.f90` remeshing planner",
    "#154": "Align `timestep.rs` and `star` mass-change `do_adjust_mass`",
    "#155": "Align convective boundaries, overshooting, CPM, and pred. mix",
    "#156": "Port full Burgers element diffusion and radiative levitation",
    "#157": "Restore MLT/TDC controls and remove non-upstream fallbacks",
    "#158": "Align shellular rotation solver variables and Spruit-Tayler",
    "#159": "Align stellar wind schemes and rotational wind boosts",
    "#160": "Align coupled `BinarySystem` step ordering, tides, and RLOF",
    "#161": "Restore upstream EOS, opacity, network, and atm control paths",
    "#162": "Align `astero` search and chi-squared drivers",
    "#170": "Integrate bundled `GYRE` oscillation eigensolver",
    "#171": "Integrate bundled `ADIPLS` oscillation eigensolver",
}


def _md_cell(text: str) -> str:
    cleaned = re.sub(
        r"\brun_star_extras\.f90:(\d+(?:-\d+)?)\b",
        r"test_suite `run_star_extras.f90` (lines \1)",
        text,
    )
    return cleaned.replace("|", "\\|").replace("\n", " ").strip()


def _format_observed(rec: harness_runner.CaseHarnessRecord) -> str:
    if not rec.executed:
        return "not executed in baseline"
    obs = rec.observed_termination or "error"
    return f"`{obs}` ({rec.steps} steps; {rec.run_mode})"


def _format_within_tol_line(
    within_tol_non_pass: list[harness_runner.CaseHarnessRecord],
) -> str:
    if not within_tol_non_pass:
        return (
            "- **Within Numeric Tolerance While Governed as Gaps**: `0` cases."
        )
    items = ", ".join(
        f"`{r.suite}/{r.case}` ({_common.issue_link(r.known_gap_issue)})"
        if r.known_gap_issue
        else f"`{r.suite}/{r.case}`"
        for r in within_tol_non_pass
    )
    return (
        f"- **Within Numeric Tolerance While Governed as Gaps**: "
        f"`{len(within_tol_non_pass)}` cases ({items}) meet numeric "
        "tolerances while remaining declared gaps under active tracking issues."
    )


def render_known_gaps_markdown(repo_root: pathlib.Path) -> str:
    """Render deterministic Markdown for `docs/benchmarks/known-gaps.md`.

    Args:
        repo_root: Root directory of the `FlexMESA` repository checkout.

    Returns:
        Deterministic UTF-8 Markdown document ending with a newline.
    """
    harness_dir = repo_root / "benchmarks" / "history_harness"
    manifest, gaps, records = case_report.build_manifest_inventory_records(
        manifest_path=harness_dir / "case_manifest.toml",
        known_gaps_path=harness_dir / "known_gaps.toml",
        baseline_json_path=harness_dir / "baseline_run_26_4_1.json",
    )
    summary = case_report.summarize_records(manifest, records, gaps)
    suites = summary["cases_by_suite"]
    hh = _dashboard_collect.collect_harness_summary(repo_root)
    hh_by_suite = hh["by_suite"]
    hh_tot = _dashboard_render._summarize_harness(hh_by_suite)

    no_ref_records = [
        r for r in records if r.reference_status == "no_reference_yet"
    ]
    within_tol_non_pass = [
        r
        for r in records
        if r.comparison is not None
        and r.comparison.within_final_tolerances
        and r.comparison.within_track_tolerances
        and r.termination_matches
        and r.verdict != "pass"
    ]
    gap_records = [r for r in records if r.verdict.startswith("declared gap")]

    issue_counts: collections.Counter[str] = collections.Counter()
    for rec in gap_records:
        if rec.known_gap_issue:
            issue_counts[rec.known_gap_issue] += 1

    lines: list[str] = [
        GENERATED_COMMENT,
        "# Declared Known Gaps & Baseline Trajectory Status",
        "",
        f"`FlexMESA` tracks all **{summary['total_cases']}** end-to-end "
        f"`test_suite` cases of **MESA release `{manifest.mesa_release}`** "
        f"(`{suites['star']}` `star`, `{suites['binary']}` `binary`, "
        f"`{suites['astero']}` `astero`; Zenodo record "
        f"`{manifest.zenodo_record}`). An upstream case counts as passing only "
        "when unmodified release inlists run uncapped in Rust `strict` mode "
        "to upstream's own termination code without in-library case overrides "
        "or pulsation scaffolding and match stored real-MESA history tracks "
        "within manifest tolerances (`notes/SCOPE.md`, `DECISIONS #17`, "
        "`DECISIONS #19`).",
        "",
        "Instead of capping steps or loosening checks to force green runs, "
        "unfinished or diverging cases are declared in "
        "`benchmarks/history_harness/known_gaps.toml` with a linked tracking "
        "issue under `#140` (`#144`). CI fails on any undeclared failure and "
        "alerts if a declared gap starts passing so its entry can be "
        "promoted.",
        "",
        "## Suite Summary",
        "",
        "| Suite | Target Cases | Stored `26.4.1` Refs | `no_reference_yet` | "
        "Strict-Validated | Diagnostic Runs | Reach Upstream Term | "
        "Declared Gaps | Verified Pass |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for suite in case_manifest.VALID_SUITES:
        st = hh_by_suite[suite]
        lines.append(
            f"| `{suite}` | {st['total']} | {st['mesa_26_4_1_ref']} | "
            f"{st['no_reference_yet']} | {st['strict_validated']} | "
            f"{st['diagnostic_runs']} | {st['reached_termination']} | "
            f"{st['declared_gaps']} | {st['passing']} |"
        )

    lines.extend(
        [
            f"| **Total** | **{hh_tot['total']}** | "
            f"**{hh_tot['mesa_26_4_1_ref']}** | "
            f"**{hh_tot['no_reference_yet']}** | "
            f"**{hh_tot['strict_validated']}** | "
            f"**{hh_tot['diagnostic_runs']}** | "
            f"**{hh_tot['reached_termination']}** | "
            f"**{hh_tot['declared_gaps']}** | "
            f"**{hh_tot['passing']}** |",
            "",
            "`Strict-Validated` counts cases whose unmodified inlists pass "
            "strict control validation (`#147`). `Diagnostic Runs` counts "
            "cases that strict validation rejects and that were rerun with "
            "the unsupported controls ignored; their numbers show where the "
            "engine stands and can never count as a pass (`DECISIONS #26`). "
            "`Reach Upstream Term` counts runs of either kind.",
            "",
            f"- **Undeclared Failures**: `{hh_tot['undeclared_failures']}`",
            _format_within_tol_line(within_tol_non_pass),
            "",
            "## Cases Without a Stored `26.4.1` Reference (`no_reference_yet`)",
            "",
            "Cases lacking a committed MESA `26.4.1` baseline report "
            "`no reference yet` and cannot be graded as passing or failing:",
            "",
            "| Suite | Case | Required Termination | Blocking Issue / Reason |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    entry_by_key = {c.key: c for c in manifest.cases}
    for rec in no_ref_records:
        entry = entry_by_key[(rec.suite, rec.case)]
        dep_notes = "; ".join(
            f"{d.issue}: {d.rationale}" for d in entry.departures
        )
        if not dep_notes and entry.scaffolding_issue:
            dep_notes = (
                f"{entry.scaffolding_issue}: awaits bundled pulsation/upstream "
                "support"
            )
        lines.append(
            f"| `{rec.suite}` | `{rec.case}` | "
            f"`{_md_cell(rec.required_termination)}` | "
            f"{_md_cell(dep_notes or 'Awaiting MESA 26.4.1 reference run')} |"
        )

    lines.extend(
        [
            "",
            "## Declared Gaps by Tracking Issue",
            "",
            "| Tracking Issue | Declared Case Count | Primary Focus |",
            "| :--- | ---: | :--- |",
        ]
    )
    for issue, count in sorted(
        issue_counts.items(),
        key=lambda kv: (-kv[1], _common.issue_sort_key(kv[0])),
    ):
        link = _common.issue_link(issue)
        desc = _ISSUE_FOCUS.get(issue, "Open #140 parity sub-issue")
        lines.append(f"| {link} | {count} | {desc} |")

    lines.extend(
        [
            "",
            f"## Complete Known-Gaps Registry ({len(gap_records)} Cases)",
            "",
            "| Suite | Case | Issue | Expected Term | Baseline Outcome | "
            "Mechanism |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for rec in gap_records:
        issue_md = (
            _common.issue_link(rec.known_gap_issue)
            if rec.known_gap_issue
            else "none"
        )
        lines.append(
            f"| `{rec.suite}` | `{rec.case}` | {issue_md} | "
            f"`{_md_cell(rec.required_termination)}` | "
            f"{_format_observed(rec)} | "
            f"{_md_cell(rec.known_gap_reason)} |"
        )

    lines.extend(
        [
            "",
            "## Inspecting Known Gaps Programmatically",
            "",
            "```python",
            "from flexmesa.testing import case_manifest",
            "",
            "manifest = case_manifest.load_case_manifest()",
            "gaps = case_manifest.load_known_gaps()",
            f"assert len(manifest.cases) == {summary['total_cases']}",
            f"assert len(gaps) == {summary['declared_known_gaps']}",
            "no_ref = [",
            "    c.case for c in manifest.cases",
            '    if c.reference_status == "no_reference_yet"',
            "]",
            'print(f"Manifest={len(manifest.cases)}, gaps={len(gaps)}, '
            'no_ref={len(no_ref)}")',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `docs/benchmarks/known-gaps.md`.

    Args:
        argv: Optional argument list overriding `sys.argv[1:]`.

    Returns:
        Exit status code (`0` when rendered or matching under `--check`).
    """
    return _common.run_render_cli(
        argv=argv,
        description="Render or check docs/benchmarks/known-gaps.md.",
        default_rel_output=DEFAULT_OUTPUT,
        render_fn=render_known_gaps_markdown,
        regen_cmd="python scripts/docs/render_known_gaps_page.py",
        check_ok_msg=(
            "OK: {out_path} matches known_gaps.toml and baseline data."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
