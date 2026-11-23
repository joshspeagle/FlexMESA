# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line interface for ``flexmesa trace`` and ``flexmesa harness``."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import pathlib
import sys
import tempfile

from flexmesa import _dispatch
from flexmesa.data import fetch
from flexmesa.testing import (
    baseline_regression,
    case_manifest,
    case_report,
    harness_runner,
    stepdump_compare,
)


def _non_negative_int(raw: str) -> int:
    val = int(raw)
    if val < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {val}")
    return val


def _run_worker(argv: list[str]) -> int:
    """Internal subprocess entry point used by ``run_traced_case`` guards."""
    parser = argparse.ArgumentParser(prog="flexmesa-trace-worker")
    parser.add_argument("suite", choices=case_manifest.VALID_SUITES)
    parser.add_argument("case_dir")
    parser.add_argument("out_target")
    parser.add_argument("meta_json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--is-logs-dir", action="store_true")
    mode.add_argument("--explicit-file", action="store_true")
    parser.add_argument("--max-steps", type=_non_negative_int, default=None)
    parser.add_argument("--full-stages", action="store_true")
    parser.add_argument("--respect-intervals", action="store_true")
    parser.add_argument("--write-profiles", action="store_true")
    parser.add_argument(
        "--numerics-mode",
        choices=_dispatch.VALID_NUMERICS_MODES,
        default="strict",
    )
    parser.add_argument("--allow-unsupported-controls", action="store_true")
    parser.add_argument("--extras-factory", default=None)
    ns = parser.parse_args(argv)

    rep = harness_runner.run_trace_inprocess(
        ns.suite,
        pathlib.Path(ns.case_dir),
        pathlib.Path(ns.out_target),
        max_steps=ns.max_steps,
        skip_optional=not ns.full_stages,
        respect_intervals=ns.respect_intervals,
        write_profiles=ns.write_profiles,
        is_logs_dir=bool(ns.is_logs_dir),
        numerics_mode=ns.numerics_mode,
        allow_unsupported_controls=bool(ns.allow_unsupported_controls),
        extras_factory=ns.extras_factory,
    )
    payload = {
        "termination_code": rep.get("termination_code", ""),
        "total_steps": int(rep.get("total_steps", 0)),
        "total_retries": int(
            rep.get("total_retries", rep.get("binary_retries", 0))
        ),
        "rss_gb": harness_runner.proc_rss_gb(os.getpid()),
    }
    pathlib.Path(ns.meta_json).write_text(
        json.dumps(payload, allow_nan=False), encoding="utf-8"
    )
    return 0


def _build_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexmesa trace",
        description=(
            "Run one upstream star, binary, or astero test-suite case and "
            "stream incremental MESA history logs."
        ),
    )
    parser.add_argument(
        "suite",
        choices=case_manifest.VALID_SUITES,
        help="Upstream test suite ('star', 'binary', or 'astero').",
    )
    parser.add_argument(
        "case",
        help="Case name under $MESA_DIR/<suite>/test_suite or directory path.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory (writes LOGS/history.data or "
            "binary_history.data) or explicit output file when "
            "--explicit-file is passed."
        ),
    )
    parser.add_argument(
        "--explicit-file",
        action="store_true",
        help="Write history directly to --out instead of a LOGS/ directory.",
    )
    parser.add_argument(
        "--mesa-dir",
        default=None,
        help="Optional override for $MESA_DIR test-suite root.",
    )
    parser.add_argument(
        "--max-steps",
        type=_non_negative_int,
        default=None,
        help="Optional per-stage step cap (smoke mode only).",
    )
    parser.add_argument(
        "--full-stages",
        action="store_true",
        help="Execute optional early ./rn stages instead of skipping them.",
    )
    parser.add_argument(
        "--respect-intervals",
        action="store_true",
        help="Honor inlist history_interval instead of recording every step.",
    )
    parser.add_argument(
        "--write-profiles",
        action="store_true",
        help="Enable radial profileN.data output when targeting a directory.",
    )
    parser.add_argument(
        "--numerics-mode",
        choices=_dispatch.VALID_NUMERICS_MODES,
        default=None,
        help="Arithmetic mode ('strict' default, or 'fast'; #165).",
    )
    parser.add_argument(
        "--allow-unsupported-controls",
        action="store_true",
        help=(
            "Opt out of strict inlist validation for unread, unused, or "
            "unimplemented upstream controls (#147)."
        ),
    )
    parser.add_argument(
        "--extras-factory",
        default=None,
        help="Optional 'module:callable' returning case extras handlers.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        default=None,
        help="Optional resident-memory ceiling in GB.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="MESA_STEPDUMP",
        help=(
            "MESA 26.4.1 step dump (benchmarks/fortran_mesa/stepdumps) to "
            "compare with: runs the case with FlexMESA's step dump on for "
            "the same number of models and reports the first record that "
            "differs (#372)."
        ),
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "With --compare: record types (EQU, DM, TRY, ...), column names "
            "(h1, lnd, ...) or 'xa' (every species column) left out, so a "
            "known difference does not hide the next one."
        ),
    )
    return parser


def _stepdump_window(mesa_dump: pathlib.Path) -> tuple[int, int | None]:
    """Returns the models per star and the first model number of a MESA dump.

    The first model number is ``None`` when the dump starts at the first
    model of the run.
    """
    steps: int | None = None
    first: int | None = None
    with mesa_dump.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, val = line[1:].partition(":")
            if key.strip() == "steps_per_star":
                steps = int(val)
            elif key.strip() == "first_model" and val.strip().isdigit():
                first = int(val)
    if steps is None:
        raise ValueError(f"{mesa_dump}: no steps_per_star header line")
    return steps, first


def trace_main(argv: list[str] | None = None) -> int:
    """Execute ``flexmesa trace star|binary|astero <case>``.

    Args:
        argv: CLI arguments following ``flexmesa trace``.

    Returns:
        Exit status (``0`` on success, ``1`` on run failure/timeout/OOM).
    """
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if cli_args and cli_args[0] == "_worker":
        return _run_worker(cli_args[1:])

    ns = _build_trace_parser().parse_args(cli_args)
    eff_mode = harness_runner.resolve_effective_numerics_mode(ns.numerics_mode)
    flex_dump: pathlib.Path | None = None
    if ns.compare is not None:
        steps, first = _stepdump_window(pathlib.Path(ns.compare))
        flex_dump = pathlib.Path(ns.out) / "flexmesa.stepdump"
        flex_dump.parent.mkdir(parents=True, exist_ok=True)
        flex_dump.unlink(missing_ok=True)
        os.environ["FLEXMESA_STEPDUMP"] = str(flex_dump)
        os.environ["FLEXMESA_STEPDUMP_STEPS"] = str(steps)
        if first is not None:
            os.environ["FLEXMESA_STEPDUMP_FIRST"] = str(first)
        elif ns.max_steps is None:
            # A window at the start of the run needs only its own models; a
            # later window runs up to the timeout or --max-steps.
            ns.max_steps = steps
    case_dir = harness_runner.resolve_case_dir(
        ns.suite, ns.case, mesa_dir=ns.mesa_dir
    )
    outcome = harness_runner.run_traced_case(
        ns.suite,
        case_dir,
        pathlib.Path(ns.out),
        max_steps=ns.max_steps,
        skip_optional=not ns.full_stages,
        respect_intervals=ns.respect_intervals,
        write_profiles=ns.write_profiles,
        is_logs_dir=not ns.explicit_file,
        numerics_mode=eff_mode,
        allow_unsupported_controls=bool(ns.allow_unsupported_controls),
        extras_factory=ns.extras_factory,
        timeout_sec=ns.timeout_sec,
        max_rss_gb=ns.max_rss_gb,
    )
    hist_str = str(outcome.history_path) if outcome.history_path else "none"
    print(
        f"suite={outcome.suite} case={outcome.case} status={outcome.status} "
        f"term={outcome.termination_code or 'none'} steps={outcome.steps} "
        f"retries={outcome.retries} wall_s={outcome.wall_time_sec:.2f} "
        f"peak_rss_gb={outcome.peak_rss_gb:.2f} history={hist_str}"
    )
    if outcome.error_message and outcome.status != "ok":
        print(f"error: {outcome.error_message}", file=sys.stderr)
    if flex_dump is not None:
        if not flex_dump.exists():
            print("error: FlexMESA wrote no step dump", file=sys.stderr)
            return 1
        print(
            stepdump_compare.compare_files(
                pathlib.Path(ns.compare), flex_dump, frozenset(ns.ignore)
            ),
            end="",
        )
        return 0
    return 0 if outcome.status == "ok" else 1


def _select_entries(
    manifest: case_manifest.HarnessManifest,
    preset: str,
    cases_filter: list[str] | None,
) -> list[case_manifest.CaseManifestEntry]:
    if cases_filter:
        wanted = set(cases_filter)
        selected = [
            e
            for e in manifest.cases
            if e.case in wanted or f"{e.suite}/{e.case}" in wanted
        ]
        if not selected:
            raise ValueError(f"no manifest entries matched {cases_filter!r}")
        return selected
    if preset == "pr-ci":
        return [e for e in manifest.cases if e.pr_ci]
    if preset == "has-reference":
        return [
            e
            for e in manifest.cases
            if e.reference_status != "no_reference_yet"
        ]
    return list(manifest.cases)


def _build_harness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexmesa harness",
        description=(
            "Generate the MESA 26.4.1 reference-availability table or execute "
            "guarded trajectory comparisons."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_tbl = sub.add_parser(
        "table",
        help="Generate or verify the baseline 106-case availability table.",
    )
    p_tbl.add_argument("--manifest", default=None)
    p_tbl.add_argument("--known-gaps", default=None)
    p_tbl.add_argument("--baseline-json", default=None)
    p_tbl.add_argument(
        "--markdown-out",
        default=str(
            case_manifest.repo_root() / "notes" / "audit" / "case_table.md"
        ),
    )
    p_tbl.add_argument("--json-out", default=None)
    p_tbl.add_argument(
        "--check",
        action="store_true",
        help="Verify that --markdown-out matches the generated table.",
    )

    p_run = sub.add_parser(
        "run",
        help="Execute selected manifest cases and compare stored histories.",
    )
    p_run.add_argument("--manifest", default=None)
    p_run.add_argument("--known-gaps", default=None)
    p_run.add_argument("--mesa-dir", default=None)
    p_run.add_argument(
        "--preset",
        choices=("pr-ci", "has-reference", "all"),
        default="pr-ci",
    )
    p_run.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Explicit case names or '<suite>/<case>' identifiers to execute.",
    )
    p_run.add_argument("--work-dir", default=None)
    p_run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent case workers to launch.",
    )
    p_run.add_argument("--max-steps", type=_non_negative_int, default=None)
    p_run.add_argument(
        "--numerics-mode",
        choices=_dispatch.VALID_NUMERICS_MODES,
        default=None,
        help="Arithmetic mode ('strict' default, or 'fast'; #165).",
    )
    p_run.add_argument(
        "--allow-unsupported-controls",
        action="store_true",
        help=(
            "Skip the strict run and report every case from a diagnostic "
            "run that ignores unsupported upstream controls (#147). By "
            "default each case is run strictly first and rerun this way "
            "only if strict validation rejects it. A diagnostic run never "
            "counts as a pass."
        ),
    )
    p_run.add_argument(
        "--extras-factory",
        default=None,
        help=(
            "Override the manifest 'module:callable' test-side extras factory."
        ),
    )
    p_run.add_argument("--timeout-sec", type=float, default=600.0)
    p_run.add_argument("--max-rss-gb", type=float, default=6.0)
    p_run.add_argument("--markdown-out", default=None)
    p_run.add_argument("--json-out", default=None)
    p_run.add_argument(
        "--figures-dir",
        default=None,
        help=(
            "Optional directory for rendered MESA comparison PNGs linked from "
            "--markdown-out (#173)."
        ),
    )
    p_run.add_argument(
        "--save-baseline",
        default=None,
        help="Write timing-normalized baseline JSON for 'harness table'.",
    )
    p_run.add_argument(
        "--update-baseline",
        default=None,
        help=(
            "Rewrite this baseline JSON with the executed cases replaced by "
            "this run and every other case kept, so a subset run can update "
            "the committed baseline."
        ),
    )
    p_run.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "Baseline run to compare against for regressions (default: the "
            "committed baseline when the default manifest is used)."
        ),
    )
    p_run.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Return exit code 1 on undeclared failures, declared gaps that "
            "pass, or cases that regressed against the baseline run."
        ),
    )
    return parser


def _baseline_regressions(
    ns: argparse.Namespace,
    records: list[harness_runner.CaseHarnessRecord],
) -> list[baseline_regression.BaselineRegression]:
    """Compare a run with the baseline run (``DECISIONS #27``).

    Step-capped runs, runs in fast mode and runs that skip the strict
    attempt are not comparable with the baseline and return no regressions.
    """
    if (
        ns.max_steps is not None
        or (ns.numerics_mode or "strict") != "strict"
        or ns.allow_unsupported_controls
    ):
        return []
    try:
        _, _, base_records = case_report.build_manifest_inventory_records(
            ns.manifest, ns.known_gaps, ns.baseline_json
        )
    except FileNotFoundError:
        if ns.baseline_json is not None:
            raise
        return []
    return baseline_regression.find_baseline_regressions(records, base_records)


def _write_json_file(path_str: str, payload: dict[str, object]) -> None:
    out_p = pathlib.Path(path_str)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _handle_table_subcommand(ns: argparse.Namespace) -> int:
    manifest, gaps, records = case_report.build_manifest_inventory_records(
        ns.manifest, ns.known_gaps, ns.baseline_json
    )
    payload = case_report.build_report_payload(manifest, records, gaps)
    md_text = case_report.render_case_table_markdown(
        manifest, records, summary=payload["summary"], known_gaps=gaps
    )
    md_path = pathlib.Path(ns.markdown_out)
    if ns.check:
        if not md_path.is_file():
            print(f"missing {md_path}", file=sys.stderr)
            return 1
        if md_path.read_text(encoding="utf-8") != md_text:
            print(
                f"{md_path} is out of date; regenerate with "
                "'flexmesa harness table'",
                file=sys.stderr,
            )
            return 1
        return 0

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    if ns.json_out:
        _write_json_file(ns.json_out, payload)
    return 0


def _execute_one_entry(
    entry: case_manifest.CaseManifestEntry,
    gaps: dict[tuple[str, str], case_manifest.KnownGapSpec],
    work_root: pathlib.Path,
    mesa_dir: str | None,
    max_steps: int | None,
    numerics_mode: str,
    allow_unsupported_controls: bool,
    extras_factory: str | None,
    timeout_sec: float | None,
    max_rss_gb: float | None,
    base_dir: pathlib.Path | None,
) -> harness_runner.CaseHarnessRecord:
    case_dir = harness_runner.resolve_case_dir(
        entry.suite, entry.case, mesa_dir=mesa_dir
    )
    out_dir = work_root / entry.suite / entry.case
    eff_extras_factory = extras_factory or entry.extras_factory or None
    outcome = harness_runner.run_case_with_diagnostic(
        entry.suite,
        case_dir,
        out_dir,
        max_steps=max_steps,
        numerics_mode=numerics_mode,
        allow_unsupported_controls=allow_unsupported_controls,
        extras_factory=eff_extras_factory,
        timeout_sec=timeout_sec,
        max_rss_gb=max_rss_gb,
    )
    return harness_runner.evaluate_case_record(
        entry,
        gaps,
        outcome,
        max_steps=max_steps,
        root_dir=base_dir,
    )


def _append_progress(
    work_root: pathlib.Path, rec: harness_runner.CaseHarnessRecord
) -> None:
    """Append one finished case to ``<work_root>/progress.jsonl``.

    The case table is written only when every case has finished. This file
    keeps each result as it arrives, so a run that is cancelled part way
    (for example by a CI time limit) still leaves what it measured.
    """
    line = json.dumps(case_report.record_to_dict(rec), allow_nan=False)
    with (work_root / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _run_selected_entries(
    selected: list[case_manifest.CaseManifestEntry],
    gaps: dict[tuple[str, str], case_manifest.KnownGapSpec],
    work_root: pathlib.Path,
    ns: argparse.Namespace,
    base_dir: pathlib.Path | None,
) -> list[harness_runner.CaseHarnessRecord]:
    jobs = max(1, int(ns.jobs))
    if jobs == 1:
        records: list[harness_runner.CaseHarnessRecord] = []
        for idx, entry in enumerate(selected, start=1):
            rec = _execute_one_entry(
                entry,
                gaps,
                work_root,
                ns.mesa_dir,
                ns.max_steps,
                ns.numerics_mode,
                bool(ns.allow_unsupported_controls),
                ns.extras_factory,
                ns.timeout_sec,
                ns.max_rss_gb,
                base_dir,
            )
            records.append(rec)
            _append_progress(work_root, rec)
            print(
                f"[harness] [{idx}/{len(selected)}] {rec.suite}/{rec.case}: "
                f"verdict={rec.verdict} mode={rec.run_mode} "
                f"term={rec.observed_termination or 'none'} "
                f"steps={rec.steps} wall={rec.wall_time_sec:.1f}s",
                flush=True,
            )
        return records

    indexed_records: dict[int, harness_runner.CaseHarnessRecord] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        fut_to_idx = {
            pool.submit(
                _execute_one_entry,
                entry,
                gaps,
                work_root,
                ns.mesa_dir,
                ns.max_steps,
                ns.numerics_mode,
                bool(ns.allow_unsupported_controls),
                ns.extras_factory,
                ns.timeout_sec,
                ns.max_rss_gb,
                base_dir,
            ): idx
            for idx, entry in enumerate(selected)
        }
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            rec = fut.result()
            indexed_records[idx] = rec
            _append_progress(work_root, rec)
            print(
                f"[harness] [{len(indexed_records)}/{len(selected)}] "
                f"{rec.suite}/{rec.case}: verdict={rec.verdict} "
                f"mode={rec.run_mode} "
                f"term={rec.observed_termination or 'none'} "
                f"steps={rec.steps} wall={rec.wall_time_sec:.1f}s",
                flush=True,
            )
    return [indexed_records[i] for i in range(len(selected))]


def _render_harness_figures(
    manifest: case_manifest.HarnessManifest,
    records: list[harness_runner.CaseHarnessRecord],
    work_root: pathlib.Path,
    figures_dir: str,
    *,
    manifest_path: str | None,
    markdown_out: str | None,
    numerics_mode: str,
    max_steps: int | None,
) -> dict[tuple[str, str], str]:
    import matplotlib.pyplot as plt

    from flexmesa.plotting import compare as plot_compare

    entry_by_key = manifest.by_key()
    fig_root = pathlib.Path(figures_dir)
    md_parent = (
        pathlib.Path(markdown_out).resolve().parent if markdown_out else None
    )
    fig_links: dict[tuple[str, str], str] = {}
    for rec in records:
        entry = entry_by_key[rec.suite, rec.case]
        if (
            not rec.executed
            or entry.reference_status == "no_reference_yet"
            or not entry.reference_path
        ):
            continue
        hist_p = harness_runner.locate_written_history(
            rec.suite, work_root / rec.suite / rec.case, True
        )
        if hist_p is None:
            continue
        fig_out = fig_root / rec.suite / f"{rec.case}.png"
        try:
            fig, _ = plot_compare.plot_case_comparison(
                rec.case,
                suite=rec.suite,
                flexmesa_history=hist_p,
                manifest_path=manifest_path,
                save_path=fig_out,
                termination_code=rec.observed_termination,
                numerics_mode=numerics_mode,
                controls_ignored=rec.controls_ignored,
                n_controls_ignored=rec.n_controls_ignored,
                max_steps=max_steps,
                case_entry=entry,
                comparison=rec.comparison,
            )
        except (OSError, ValueError, KeyError) as exc:
            print(
                f"[harness] skipping figure for {rec.suite}/{rec.case}: {exc}",
                file=sys.stderr,
            )
            continue
        plt.close(fig)
        if md_parent is not None:
            fig_links[(rec.suite, rec.case)] = (
                fig_out.resolve()
                .relative_to(md_parent, walk_up=True)
                .as_posix()
            )
        else:
            fig_links[(rec.suite, rec.case)] = fig_out.as_posix()
    return fig_links


def harness_main(argv: list[str] | None = None) -> int:
    """Execute ``flexmesa harness table|run``.

    Args:
        argv: CLI arguments following ``flexmesa harness``.

    Returns:
        Exit status (``0`` on success, ``1`` on check mismatch or gate failure).
    """
    cli_args = list(sys.argv[1:] if argv is None else argv)
    ns = _build_harness_parser().parse_args(cli_args)
    if ns.subcommand == "table":
        return _handle_table_subcommand(ns)
    eff_mode = harness_runner.resolve_effective_numerics_mode(ns.numerics_mode)
    ns.numerics_mode = eff_mode
    if ns.gate and eff_mode != "strict":
        print(
            "--gate requires strict numerics mode "
            "(#165: only strict mode is judged against MESA)",
            file=sys.stderr,
        )
        return 1

    print(
        f"[harness] upstream reference: "
        f"{fetch.load_upstream_reference().summary_line()}",
        flush=True,
    )
    manifest = case_manifest.load_case_manifest(ns.manifest)
    gaps = case_manifest.load_known_gaps(ns.known_gaps)
    selected = _select_entries(manifest, ns.preset, ns.cases)
    base_dir = pathlib.Path(ns.manifest).parent if ns.manifest else None

    fig_links: dict[tuple[str, str], str] | None = None
    with contextlib.ExitStack() as stack:
        if ns.work_dir is None:
            tmp_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="flexmesa_harness_")
            )
            work_root = pathlib.Path(tmp_dir)
        else:
            work_root = pathlib.Path(ns.work_dir)
            work_root.mkdir(parents=True, exist_ok=True)
        records = _run_selected_entries(selected, gaps, work_root, ns, base_dir)

        if ns.figures_dir is not None:
            fig_links = _render_harness_figures(
                manifest,
                records,
                work_root,
                ns.figures_dir,
                manifest_path=ns.manifest,
                markdown_out=ns.markdown_out,
                numerics_mode=ns.numerics_mode,
                max_steps=ns.max_steps,
            )

    payload = case_report.build_report_payload(
        manifest,
        records,
        gaps,
        numerics_mode=ns.numerics_mode,
    )
    if ns.markdown_out:
        md_text = case_report.render_case_table_markdown(
            manifest,
            records,
            summary=payload["summary"],
            known_gaps=gaps,
            figure_links=fig_links,
        )
        mpath = pathlib.Path(ns.markdown_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(md_text, encoding="utf-8")
    regressions = _baseline_regressions(ns, records)
    payload["summary"]["regressions_vs_baseline"] = [
        r.label for r in regressions
    ]
    if ns.json_out:
        _write_json_file(ns.json_out, payload)
    if ns.update_baseline:
        _, _, base_records = case_report.build_manifest_inventory_records(
            ns.manifest, ns.known_gaps, ns.update_baseline
        )
        merged = baseline_regression.merge_records_into_baseline(
            records, base_records
        )
        _write_json_file(
            ns.update_baseline,
            case_report.build_deterministic_baseline_payload(
                manifest, merged, gaps, numerics_mode=ns.numerics_mode
            ),
        )
    if ns.save_baseline:
        b_payload = case_report.build_deterministic_baseline_payload(
            manifest,
            records,
            gaps,
            numerics_mode=ns.numerics_mode,
        )
        _write_json_file(ns.save_baseline, b_payload)

    summary = payload["summary"]
    if regressions:
        print(
            "regressed against the baseline run: "
            + "; ".join(r.label for r in regressions),
            file=sys.stderr,
        )
    if ns.gate and (
        summary["undeclared_failures"]
        or summary["stale_known_gaps"]
        or regressions
    ):
        if summary["undeclared_failures"]:
            print(
                "undeclared failures: "
                + ", ".join(summary["undeclared_failures"]),
                file=sys.stderr,
            )
        if summary["stale_known_gaps"]:
            print(
                "declared gaps now passing: "
                + ", ".join(summary["stale_known_gaps"]),
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(trace_main())
