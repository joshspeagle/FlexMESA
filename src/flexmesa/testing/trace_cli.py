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

from flexmesa.data import fetch
from flexmesa.testing import case_manifest, case_report, harness_runner


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
        choices=("strict", "fast"),
        default="strict",
    )
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
    )
    payload = {
        "termination_code": rep.get("termination_code", ""),
        "total_steps": int(rep.get("total_steps", 0)),
        "total_retries": int(rep.get("total_retries", 0)),
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
        choices=("strict", "fast"),
        default="strict",
        help="Arithmetic mode ('strict' default, or 'fast'; #165).",
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
    return parser


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
        numerics_mode=ns.numerics_mode,
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
        choices=("strict", "fast"),
        default="strict",
        help="Arithmetic mode ('strict' default, or 'fast'; #165).",
    )
    p_run.add_argument("--timeout-sec", type=float, default=600.0)
    p_run.add_argument("--max-rss-gb", type=float, default=6.0)
    p_run.add_argument("--markdown-out", default=None)
    p_run.add_argument("--json-out", default=None)
    p_run.add_argument(
        "--save-baseline",
        default=None,
        help="Write timing-normalized baseline JSON for 'harness table'.",
    )
    p_run.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Return exit code 1 on undeclared failures or declared gaps that "
            "pass."
        ),
    )
    return parser


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
    timeout_sec: float | None,
    max_rss_gb: float | None,
    base_dir: pathlib.Path | None,
) -> harness_runner.CaseHarnessRecord:
    case_dir = harness_runner.resolve_case_dir(
        entry.suite, entry.case, mesa_dir=mesa_dir
    )
    out_dir = work_root / entry.suite / entry.case
    outcome = harness_runner.run_traced_case(
        entry.suite,
        case_dir,
        out_dir,
        max_steps=max_steps,
        numerics_mode=numerics_mode,
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
                ns.timeout_sec,
                ns.max_rss_gb,
                base_dir,
            )
            records.append(rec)
            print(
                f"[harness] [{idx}/{len(selected)}] {rec.suite}/{rec.case}: "
                f"verdict={rec.verdict} "
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
            print(
                f"[harness] [{len(indexed_records)}/{len(selected)}] "
                f"{rec.suite}/{rec.case}: verdict={rec.verdict} "
                f"term={rec.observed_termination or 'none'} "
                f"steps={rec.steps} wall={rec.wall_time_sec:.1f}s",
                flush=True,
            )
    return [indexed_records[i] for i in range(len(selected))]


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
    if ns.gate and ns.numerics_mode != "strict":
        print(
            "--gate requires --numerics-mode strict "
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
        )
        mpath = pathlib.Path(ns.markdown_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(md_text, encoding="utf-8")
    if ns.json_out:
        _write_json_file(ns.json_out, payload)
    if ns.save_baseline:
        b_payload = case_report.build_deterministic_baseline_payload(
            manifest,
            records,
            gaps,
            numerics_mode=ns.numerics_mode,
        )
        _write_json_file(ns.save_baseline, b_payload)

    summary = payload["summary"]
    if ns.gate and (
        summary["undeclared_failures"] or summary["stale_known_gaps"]
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
