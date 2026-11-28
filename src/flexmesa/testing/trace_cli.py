# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line interface for ``flexmesa trace`` and ``flexmesa harness``."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import sys
import tempfile
from collections.abc import Callable
from typing import Any

from flexmesa import _dispatch
from flexmesa.data import fetch
from flexmesa.testing import (
    _case_verdict,
    artifact_compare,
    baseline_regression,
    case_budgets,
    case_manifest,
    case_report,
    early_stop,
    harness_budgets,
    harness_runner,
    harness_shards,
    stage_steps,
    stepdump_compare,
)


def _positive_int(raw: str) -> int:
    val = int(raw)
    if val < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {val}")
    return val


def _non_negative_int(raw: str) -> int:
    val = int(raw)
    if val < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {val}")
    return val


def _positive_float(raw: str) -> float:
    val = float(raw)
    if not val > 0.0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {raw}")
    return val


def _shard_arg(raw: str) -> tuple[int, int]:
    try:
        return harness_shards.parse_shard(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
        "stage_steps": list(harness_runner.stage_model_numbers(rep)),
        "total_retries": int(
            rep.get("total_retries", rep.get("binary_retries", 0))
        ),
        "rss_gb": harness_runner.proc_rss_gb(os.getpid()),
    }
    for key in ("star1_final_model_number", "star1_retries"):
        if rep.get(key) is not None:
            payload[key] = int(rep[key])
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
        help=(
            "'module:callable' returning case extras handlers; defaults "
            "to the case manifest's extras_factory, as the harness uses."
        ),
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
            "differs (#372). Exit status: 0 when the records are "
            "bit-identical, 2 when they differ, 1 when the run fails."
        ),
    )
    parser.add_argument(
        "--stage",
        type=_positive_int,
        default=None,
        metavar="S",
        help=(
            "With --compare: dump only the S-th stage of a multi-inlist case "
            "that reaches the solver (1-based; one MESA process per ./rn "
            "stage). Earlier stages run without dumping and the run stops "
            "after the window. Default: the stage recorded in the MESA "
            "dump's header, else every stage."
        ),
    )
    parser.add_argument(
        "--occurrence",
        type=_positive_int,
        default=None,
        metavar="K",
        help=(
            "With --compare: the window opens the K-th time the model "
            "number rises to the dump's first model (a relax phase restarts "
            "the numbering). Default: the occurrence in the MESA dump's "
            "header, else 1."
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


def _stepdump_selectors(
    mesa_dump: pathlib.Path,
    stage: int | None,
    occurrence: int | None,
) -> tuple[int | None, int]:
    """Returns the stage and occurrence of the window to dump.

    Args:
        mesa_dump: MESA dump whose ``# stage:`` and ``# occurrence:`` header
            lines (``all`` and ``1`` when absent) give the defaults.
        stage: Explicit ``--stage`` or ``None``.
        occurrence: Explicit ``--occurrence`` or ``None``.

    Returns:
        ``(stage, occurrence)``; ``stage`` is ``None`` for every stage.
    """
    head: dict[str, str] = {}
    with mesa_dump.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, val = line[1:].partition(":")
            head[key.strip()] = val.strip()
    if stage is None and head.get("stage", "all").isdigit():
        stage = int(head["stage"])
    if occurrence is None:
        occ = head.get("occurrence", "1")
        occurrence = int(occ) if occ.isdigit() and int(occ) > 0 else 1
    return stage, occurrence


def trace_main(argv: list[str] | None = None) -> int:
    """Execute ``flexmesa trace star|binary|astero <case>``.

    Args:
        argv: CLI arguments following ``flexmesa trace``.

    Returns:
        Exit status (``0`` on success, ``1`` on run failure/timeout/OOM; with
        ``--compare``, ``2`` when the step dumps differ).
    """
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if cli_args and cli_args[0] == "_worker":
        return _run_worker(cli_args[1:])

    parser = _build_trace_parser()
    ns = parser.parse_args(cli_args)
    if ns.compare is None and (ns.stage or ns.occurrence):
        parser.error("--stage and --occurrence need --compare")
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
        else:
            os.environ.pop("FLEXMESA_STEPDUMP_FIRST", None)
        stage, occurrence = _stepdump_selectors(
            pathlib.Path(ns.compare), ns.stage, ns.occurrence
        )
        os.environ["FLEXMESA_STEPDUMP_OCCURRENCE"] = str(occurrence)
        if stage is None:
            os.environ.pop("FLEXMESA_STEPDUMP_STAGE", None)
        else:
            os.environ["FLEXMESA_STEPDUMP_STAGE"] = str(stage)
        if first is None and stage is None and ns.max_steps is None:
            # A window at the start of the run needs only its own models; a
            # later window runs up to the timeout or --max-steps. A selected
            # stage stops itself after the window, and the earlier stages
            # must run to their own end, so no cap applies.
            ns.max_steps = steps
    case_dir = harness_runner.resolve_case_dir(
        ns.suite, ns.case, mesa_dir=ns.mesa_dir
    )
    # Without --extras-factory, run the case with its manifest extras, as the
    # harness does, so a traced run matches the campaign run of the case.
    extras_factory = ns.extras_factory
    if extras_factory is None:
        entry = (
            case_manifest.load_case_manifest().by_key().get((ns.suite, ns.case))
        )
        if entry is not None and entry.extras_factory:
            extras_factory = entry.extras_factory
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
        extras_factory=extras_factory,
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
        report, div = stepdump_compare.compare_files(
            pathlib.Path(ns.compare), flex_dump, frozenset(ns.ignore)
        )
        print(report, end="")
        if outcome.status != "ok":
            return 1
        return 0 if div is None else 2
    return 0 if outcome.status == "ok" else 1


def _select_entries(
    manifest: case_manifest.HarnessManifest,
    preset: str,
    cases_filter: list[str] | None,
    passing: frozenset[tuple[str, str]] = frozenset(),
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
    if preset == "passes":
        return [e for e in manifest.cases if e.key in passing]
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
        choices=("pr-ci", "has-reference", "passes", "all"),
        default="pr-ci",
        help=(
            "Cases to run: 'pr-ci', 'has-reference', 'passes' (the cases "
            "whose committed baseline verdict is pass), or 'all'. "
            "--cases overrides it."
        ),
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
    p_run.add_argument(
        "--short-budget-sec",
        type=_positive_float,
        default=None,
        help=(
            "Run the cases whose baseline row ended on the time budget with "
            "this shorter wall-clock budget (seconds) and every other case "
            "with --timeout-sec. The gate compares those cases as a prefix "
            "of their baseline row, and --update-baseline keeps their "
            "baseline row when they end on the short budget."
        ),
    )
    p_run.add_argument(
        "--budget-from-baseline",
        action="store_true",
        help=(
            "Give each case a wall-time budget of max(K x its recorded wall "
            "time, floor) capped at --timeout-sec (K=--budget-factor, "
            "floor=--budget-floor-sec). The committed baseline stores wall "
            "times as zero, so the recorded times come from a previous "
            "--json-out report given with --wall-times-json; a case without "
            "one keeps --timeout-sec. A case stopped by its budget is "
            "inconclusive and is compared as a prefix of its baseline row "
            "(DECISIONS #59, #69)."
        ),
    )
    p_run.add_argument(
        "--wall-times-json",
        default=None,
        help=(
            "Harness --json-out report whose wall_time_sec values feed "
            "--budget-from-baseline."
        ),
    )
    p_run.add_argument(
        "--budget-factor",
        type=_positive_float,
        default=2.0,
        help="Multiple K of the recorded wall time (default 2).",
    )
    p_run.add_argument(
        "--budget-floor-sec",
        type=_positive_float,
        default=120.0,
        help="Smallest derived wall-time budget in seconds (default 120).",
    )
    p_run.add_argument(
        "--mesa-step-factor",
        type=_positive_float,
        default=None,
        metavar="M",
        help=(
            "Stop a case once its history reaches M x the step count of its "
            "MESA reference run (off by default). Reported like a budget "
            "cut-off: inconclusive, compared as a prefix of the baseline."
        ),
    )
    p_run.add_argument(
        "--early-stop-on-divergence",
        action="store_true",
        help=(
            "While a star case runs, check its partial history against the "
            "MESA reference with the along-track tolerance test of the final "
            "comparison, and stop it with 'fail (early stop)' as soon as a "
            "tracked column is out of tolerance. It never stops a case the "
            "final comparison would pass. Refused with --update-baseline, "
            "--save-baseline and --max-steps (DECISIONS #69)."
        ),
    )
    p_run.add_argument(
        "--stop-on-excess-steps",
        action="store_true",
        help=(
            "Stop a single-stage star case that runs past MESA's step count "
            "(model number at the end of the stage) plus a margin, and "
            "report it as 'fail (early stop)': 'diverged: exceeded MESA's "
            "step count at stage S (N vs M)'. The margin is "
            "max(--step-margin-abs, --step-margin-frac x MESA steps). "
            "Refused with --update-baseline, --save-baseline and "
            "--max-steps (DECISIONS #69)."
        ),
    )
    p_run.add_argument(
        "--enforce-step-counts",
        action="store_true",
        help=(
            "Apply DECISIONS #70: a case that is otherwise a pass but ends "
            "any stage at another model number than MESA 26.4.1 is "
            "'fail (step count)'. Off by default, when the differences are "
            "only reported (stage_step_deltas, step_count_differs). It "
            "relabels passes, so a gate against a baseline recorded without "
            "it reports them as regressions."
        ),
    )
    p_run.add_argument(
        "--step-margin-abs",
        type=_non_negative_int,
        default=5,
        help="Smallest margin of --stop-on-excess-steps (default 5 steps).",
    )
    p_run.add_argument(
        "--step-margin-frac",
        type=_positive_float,
        default=0.02,
        help="Relative margin of --stop-on-excess-steps (default 0.02).",
    )
    p_run.add_argument(
        "--early-stop-interval-sec",
        type=_positive_float,
        default=harness_runner.MONITOR_INTERVAL_SEC,
        help="Seconds between early-stop and step-bound checks (default 10).",
    )
    p_run.add_argument(
        "--case-budgets",
        default=None,
        metavar="TOML",
        help=(
            "Give each case the wall-clock budget listed for it in this "
            "generated table (benchmarks/history_harness/case_budgets.toml) "
            "instead of --timeout-sec; a case without a row keeps "
            "--timeout-sec. With --shard the cases are packed onto shards by "
            "budget. A case that ends on its budget is inconclusive, as "
            "with any time budget (DECISIONS #59)."
        ),
    )
    p_run.add_argument(
        "--shard",
        type=_shard_arg,
        default=None,
        metavar="I/N",
        help=(
            "Run only shard I of N of the selected cases (1-based). Cases "
            "whose baseline row ended on a time or memory budget are dealt "
            "round-robin first, then the rest; merge the shard reports "
            "with 'flexmesa harness merge'."
        ),
    )
    p_run.set_defaults(
        budgets={}, budgeted_keys=frozenset(), case_budget_table={}
    )
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

    p_merge = sub.add_parser(
        "merge",
        help=(
            "Merge the --json-out reports of 'harness run --shard' into one "
            "case table, baseline and gate verdict."
        ),
    )
    p_merge.add_argument("reports", nargs="+", help="Shard --json-out files.")
    p_merge.add_argument("--manifest", default=None)
    p_merge.add_argument("--known-gaps", default=None)
    p_merge.add_argument("--baseline-json", default=None)
    p_merge.add_argument("--markdown-out", default=None)
    p_merge.add_argument("--json-out", default=None)
    p_merge.add_argument("--save-baseline", default=None)
    p_merge.add_argument("--update-baseline", default=None)
    p_merge.add_argument(
        "--gate",
        action="store_true",
        help=(
            "As for 'harness run', and also fail when the reports do not "
            "form one complete campaign."
        ),
    )
    return parser


def _baseline_regressions(
    ns: argparse.Namespace,
    records: list[harness_runner.CaseHarnessRecord],
    short_budget: frozenset[tuple[str, str]] = frozenset(),
) -> list[baseline_regression.BaselineRegression]:
    """Compare a run with the baseline run (``DECISIONS #27``).

    Step-capped runs, runs in fast mode and runs that skip the strict
    attempt are not comparable with the baseline and return no regressions.
    Cases in ``short_budget`` are compared as prefixes of their baseline row.
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
    return baseline_regression.find_baseline_regressions(
        records, base_records, short_budget=short_budget
    )


def _short_budget_keys(
    ns: argparse.Namespace,
    selected: list[case_manifest.CaseManifestEntry],
) -> frozenset[tuple[str, str]]:
    """Return the selected cases that run with ``--short-budget-sec``.

    Raises:
        FileNotFoundError: If no baseline run is available to choose from.
    """
    if ns.short_budget_sec is None:
        return frozenset()
    _, _, base_records = case_report.build_manifest_inventory_records(
        ns.manifest, ns.known_gaps, ns.baseline_json
    )
    if not any(b.executed for b in base_records):
        raise FileNotFoundError(
            "--short-budget-sec needs a baseline run (--baseline-json)"
        )
    limited = baseline_regression.budget_limited_keys(base_records)
    return frozenset(e.key for e in selected if e.key in limited)


def _costly_keys(ns: argparse.Namespace) -> frozenset[tuple[str, str]]:
    """Return the cases whose baseline run ended on a time or memory budget.

    Returns an empty set when no baseline run is available, in which case
    shards are dealt round-robin in manifest order.
    """
    try:
        _, _, base_records = case_report.build_manifest_inventory_records(
            ns.manifest, ns.known_gaps, ns.baseline_json
        )
    except FileNotFoundError:
        if ns.baseline_json is not None:
            raise
        return frozenset()
    return frozenset(
        (b.suite, b.case)
        for b in base_records
        if b.executed
        and b.run_status in baseline_regression.RESOURCE_LIMITED_STATUSES
    )


def _case_timeout(
    ns: argparse.Namespace, entry: case_manifest.CaseManifestEntry
) -> float | None:
    """Return the wall-clock budget of one case (``--short-budget-sec``)."""
    if entry.key in ns.short_budget_keys:
        return float(ns.short_budget_sec)
    table = ns.case_budget_table.get(entry.key)
    if table is not None:
        return float(table)
    budget = ns.budgets.get(entry.key)
    if budget is not None:
        return float(budget.timeout_sec)
    return float(ns.timeout_sec)


@functools.cache
def _mesa_stage_steps() -> dict[tuple[str, str], tuple[int, ...]]:
    return stage_steps.load_mesa_stage_steps()


def _monitor_factory(
    ns: argparse.Namespace, entry: case_manifest.CaseManifestEntry
) -> Callable[[pathlib.Path], early_stop.RunMonitor | None] | None:
    """Return the builder of the polling hook of one case, if any applies."""
    budget = ns.budgets.get(entry.key)
    step_factor = budget.step_factor if budget is not None else None
    mesa_steps = budget.mesa_steps if budget is not None else None
    if (
        not ns.early_stop_on_divergence
        and not ns.stop_on_excess_steps
        and step_factor is None
    ):
        return None
    mesa_stage = _mesa_stage_steps().get(entry.key, ())
    base_dir = pathlib.Path(ns.manifest).parent if ns.manifest else None

    def build(case_dir: pathlib.Path) -> early_stop.RunMonitor | None:
        return early_stop.build_monitor(
            entry,
            case_dir,
            early_stop=bool(ns.early_stop_on_divergence),
            mesa_steps=mesa_steps,
            step_factor=step_factor,
            root_dir=base_dir,
            stop_on_excess_steps=bool(ns.stop_on_excess_steps),
            mesa_stage_steps=mesa_stage,
            margin_abs=int(ns.step_margin_abs),
            margin_frac=float(ns.step_margin_frac),
        )

    return build


def _early_stop_refusal(ns: argparse.Namespace) -> str:
    """Return why ``--early-stop-on-divergence`` cannot be used, or ``""``.

    An early-stopped case has a truncated history, so it must never become a
    baseline row, and a step-capped smoke run has no tolerance verdict.
    """
    if not (ns.early_stop_on_divergence or ns.stop_on_excess_steps):
        return ""
    flag = (
        "--early-stop-on-divergence"
        if ns.early_stop_on_divergence
        else "--stop-on-excess-steps"
    )
    if ns.update_baseline or ns.save_baseline:
        return (
            f"{flag} cannot be combined with "
            "--update-baseline or --save-baseline: an early-stopped case "
            "has a truncated history (DECISIONS #69)"
        )
    if ns.max_steps is not None:
        return f"{flag} cannot be combined with --max-steps"
    return ""


def _baseline_records(
    ns: argparse.Namespace,
) -> list[harness_runner.CaseHarnessRecord]:
    """Return the baseline records, or ``[]`` when no baseline is present.

    Raises:
        FileNotFoundError: If ``--baseline-json`` names a missing file.
    """
    try:
        _, _, records = case_report.build_manifest_inventory_records(
            ns.manifest, ns.known_gaps, ns.baseline_json
        )
    except FileNotFoundError:
        if ns.baseline_json is not None:
            raise
        return []
    return records


def _case_budgets(
    ns: argparse.Namespace,
    selected: list[case_manifest.CaseManifestEntry],
    base_records: list[harness_runner.CaseHarnessRecord],
) -> dict[tuple[str, str], harness_budgets.CaseBudget]:
    """Return the per-case budgets of this run (empty when none apply).

    Raises:
        ValueError: If ``--budget-from-baseline`` has no recorded wall times
            to work from.
    """
    if not ns.budget_from_baseline and ns.mesa_step_factor is None:
        return {}
    wall_times: dict[tuple[str, str], float] = {}
    if ns.budget_from_baseline:
        if ns.wall_times_json is not None:
            wall_times = harness_budgets.load_wall_times(ns.wall_times_json)
        else:
            wall_times = {
                (b.suite, b.case): b.wall_time_sec
                for b in base_records
                if b.executed
                and b.wall_time_sec > 0.0
                and b.run_status
                not in baseline_regression.RESOURCE_LIMITED_STATUSES
            }
        if not wall_times:
            raise ValueError(
                "the baseline stores wall times as zero (DECISIONS #58); "
                "give --wall-times-json with a previous harness --json-out "
                "report"
            )
    budgets = harness_budgets.derive_budgets(
        selected,
        base_records,
        wall_times,
        timeout_sec=float(ns.timeout_sec),
        factor=float(ns.budget_factor),
        floor_sec=float(ns.budget_floor_sec),
        step_factor=ns.mesa_step_factor,
    )
    if not ns.budget_from_baseline:
        # Only the step bound applies; the time budget stays --timeout-sec.
        budgets = {
            k: dataclasses.replace(b, timeout_sec=float(ns.timeout_sec))
            for k, b in budgets.items()
        }
    n_timed = sum(1 for b in budgets.values() if b.from_baseline)
    print(
        f"[harness] per-case budgets: {n_timed} of {len(budgets)} cases have "
        f"a wall-time budget below {ns.timeout_sec:g} s"
        + (
            f"; step bound {ns.mesa_step_factor:g} x MESA steps"
            if ns.mesa_step_factor is not None
            else ""
        ),
        flush=True,
    )
    return budgets


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
    monitor_factory: (
        Callable[[pathlib.Path], early_stop.RunMonitor | None] | None
    ) = None,
    monitor_interval_sec: float = harness_runner.MONITOR_INTERVAL_SEC,
    enforce_step_counts: bool = False,
) -> harness_runner.CaseHarnessRecord:
    case_dir = harness_runner.resolve_case_dir(
        entry.suite, entry.case, mesa_dir=mesa_dir
    )
    out_dir = work_root / entry.suite / entry.case
    eff_extras_factory = extras_factory or entry.extras_factory or None
    if entry.artifact is not None:
        # A file left by an earlier run must not stand in for this run's.
        (case_dir / entry.artifact.file).unlink(missing_ok=True)
        (out_dir / entry.artifact.file).unlink(missing_ok=True)
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
        monitor=monitor_factory(case_dir) if monitor_factory else None,
        monitor_interval_sec=monitor_interval_sec,
    )
    return harness_runner.evaluate_case_record(
        entry,
        gaps,
        outcome,
        max_steps=max_steps,
        root_dir=base_dir,
        enforce_step_counts=enforce_step_counts,
        artifact_comparison=_compare_run_artifact(
            entry, outcome, [case_dir, out_dir], base_dir
        ),
    )


def _compare_run_artifact(
    entry: case_manifest.CaseManifestEntry,
    outcome: harness_runner.TraceRunOutcome,
    run_dirs: list[pathlib.Path],
    base_dir: pathlib.Path | None,
) -> artifact_compare.ArtifactComparison | None:
    """Compare the run's artifact with its `MESA` reference (``#606``).

    Returns ``None`` when the case declares no artifact. A run that did not
    finish (status other than ``ok``) is compared too, so a partial file is
    reported as differing rather than ignored.
    """
    if entry.artifact is None:
        return None
    roots = [case_manifest.repo_root()]
    if base_dir is not None:
        roots.insert(0, base_dir)
    root = next(
        (r for r in roots if (r / entry.artifact.reference).is_file()),
        roots[-1],
    )
    return artifact_compare.compare_files(entry.artifact, root, run_dirs)


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
                _case_timeout(ns, entry),
                ns.max_rss_gb,
                base_dir,
                _monitor_factory(ns, entry),
                ns.early_stop_interval_sec,
                bool(ns.enforce_step_counts),
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
                _case_timeout(ns, entry),
                ns.max_rss_gb,
                base_dir,
                _monitor_factory(ns, entry),
                ns.early_stop_interval_sec,
                bool(ns.enforce_step_counts),
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
        if not rec.executed or not _case_verdict.has_history_reference(entry):
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


def _label(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


def _short_budget_note(
    short_keys: frozenset[tuple[str, str]], run_info: dict[str, Any]
) -> str:
    """Return the Markdown note naming the short-budget cases of a run."""
    labels = ", ".join(f"`{_label(k)}`" for k in sorted(short_keys))
    return (
        f"\n> Short-budget tier: {len(short_keys)} cases whose baseline run "
        "ended on the time budget ran with a "
        f"{float(run_info['short_budget_sec']):g} s budget instead of "
        f"{float(run_info['timeout_sec']):g} s, so their step counts and "
        f"final values are those of a shorter run: {labels}.\n"
    )


def _report_and_gate(
    ns: argparse.Namespace,
    manifest: case_manifest.HarnessManifest,
    gaps: dict[tuple[str, str], case_manifest.KnownGapSpec],
    records: list[harness_runner.CaseHarnessRecord],
    run_info: dict[str, Any],
    *,
    short_keys: frozenset[tuple[str, str]] = frozenset(),
    fig_links: dict[tuple[str, str], str] | None = None,
    problems: tuple[str, ...] = (),
) -> int:
    """Write the reports of a run or merged campaign and apply the gate.

    Args:
        ns: Parsed ``harness run`` or ``harness merge`` arguments, with the
            run settings (``numerics_mode``, ``max_steps``,
            ``allow_unsupported_controls``) filled in.
        manifest: Loaded case manifest.
        gaps: Loaded known gaps.
        records: Records of the executed cases.
        run_info: Run settings written as the report's ``run`` block.
        short_keys: Cases run with the short budget.
        fig_links: Optional comparison figure links for the Markdown table.
        problems: Reasons the merged reports are not one complete campaign.

    Returns:
        Exit status (``1`` when ``--gate`` fails, else ``0``).
    """
    payload = case_report.build_report_payload(
        manifest,
        records,
        gaps,
        numerics_mode=ns.numerics_mode,
    )
    payload["run"] = run_info
    if ns.markdown_out:
        md_text = case_report.render_case_table_markdown(
            manifest,
            records,
            summary=payload["summary"],
            known_gaps=gaps,
            figure_links=fig_links,
        )
        if short_keys:
            md_text += _short_budget_note(short_keys, run_info)
        mpath = pathlib.Path(ns.markdown_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(md_text, encoding="utf-8")
    regressions = _baseline_regressions(ns, records, short_keys)
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
            records,
            base_records,
            short_budget=short_keys | ns.budgeted_keys,
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
    if problems:
        print(
            "shard reports are not one complete campaign: "
            + "; ".join(problems),
            file=sys.stderr,
        )
    if ns.gate and (
        summary["undeclared_failures"]
        or summary["stale_known_gaps"]
        or regressions
        or problems
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


def _handle_merge_subcommand(ns: argparse.Namespace) -> int:
    """Merge shard reports into one campaign report (``harness merge``)."""
    payloads = [
        json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        for p in ns.reports
    ]
    try:
        merged = harness_shards.merge_shard_payloads(payloads)
    except ValueError as exc:
        print(f"cannot merge shard reports: {exc}", file=sys.stderr)
        return 1
    settings = merged.settings
    if ns.save_baseline and settings["short_budget_sec"] is not None:
        print(
            "--save-baseline records full-budget runs; the shards used "
            "--short-budget-sec",
            file=sys.stderr,
        )
        return 1
    if (
        settings["early_stop_on_divergence"] or settings["stop_on_excess_steps"]
    ) and (ns.save_baseline or ns.update_baseline):
        print(
            "the shards used --early-stop-on-divergence; their truncated "
            "histories cannot become baseline rows (DECISIONS #69)",
            file=sys.stderr,
        )
        return 1
    if settings["budget_from_baseline"] and ns.save_baseline:
        print(
            "--save-baseline records full-budget runs; the shards used "
            "--budget-from-baseline",
            file=sys.stderr,
        )
        return 1
    ns.budgeted_keys = merged.budgeted_cases
    ns.numerics_mode = str(settings["numerics_mode"])
    ns.max_steps = settings["max_steps"]
    ns.allow_unsupported_controls = bool(settings["allow_unsupported_controls"])
    if ns.gate and ns.numerics_mode != "strict":
        print(
            "--gate requires strict numerics mode "
            "(#165: only strict mode is judged against MESA)",
            file=sys.stderr,
        )
        return 1
    manifest = case_manifest.load_case_manifest(ns.manifest)
    gaps = case_manifest.load_known_gaps(ns.known_gaps)
    base_dir = pathlib.Path(ns.manifest).parent if ns.manifest else None
    all_records = case_report.records_from_stored_rows(
        manifest,
        gaps,
        merged.rows,
        base_dir,
        max_steps=ns.max_steps,
        enforce_step_counts=bool(settings["enforce_step_counts"]),
    )
    records = [r for r in all_records if (r.suite, r.case) in merged.rows]
    run_info = dict(settings)
    run_info["shard"] = None
    run_info["merged_shards"] = len(payloads)
    run_info["short_budget_cases"] = [
        _label(k) for k in sorted(merged.short_budget_cases)
    ]
    run_info["budgeted_cases"] = [
        _label(k) for k in sorted(merged.budgeted_cases)
    ]
    run_info["selected_cases"] = [_label((r.suite, r.case)) for r in records]
    return _report_and_gate(
        ns,
        manifest,
        gaps,
        records,
        run_info,
        short_keys=merged.short_budget_cases,
        problems=merged.problems,
    )


def harness_main(argv: list[str] | None = None) -> int:
    """Execute ``flexmesa harness table|run|merge``.

    Args:
        argv: CLI arguments following ``flexmesa harness``.

    Returns:
        Exit status (``0`` on success, ``1`` on check mismatch or gate failure).
    """
    cli_args = list(sys.argv[1:] if argv is None else argv)
    ns = _build_harness_parser().parse_args(cli_args)
    if ns.subcommand == "table":
        return _handle_table_subcommand(ns)
    if ns.subcommand == "merge":
        return _handle_merge_subcommand(ns)
    eff_mode = harness_runner.resolve_effective_numerics_mode(ns.numerics_mode)
    ns.numerics_mode = eff_mode
    if ns.gate and eff_mode != "strict":
        print(
            "--gate requires strict numerics mode "
            "(#165: only strict mode is judged against MESA)",
            file=sys.stderr,
        )
        return 1
    if ns.short_budget_sec is not None:
        if ns.short_budget_sec >= ns.timeout_sec:
            print(
                "--short-budget-sec must be shorter than --timeout-sec",
                file=sys.stderr,
            )
            return 1
        if ns.save_baseline:
            print(
                "--save-baseline records full-budget runs; it cannot be "
                "combined with --short-budget-sec",
                file=sys.stderr,
            )
            return 1

    refusal = _early_stop_refusal(ns)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    if ns.case_budgets is not None and (
        ns.budget_from_baseline or ns.short_budget_sec is not None
    ):
        print(
            "--case-budgets cannot be combined with --budget-from-baseline "
            "or --short-budget-sec",
            file=sys.stderr,
        )
        return 1
    if ns.budget_from_baseline and ns.short_budget_sec is not None:
        print(
            "--budget-from-baseline replaces --short-budget-sec; give one",
            file=sys.stderr,
        )
        return 1
    if ns.budget_from_baseline and ns.save_baseline:
        print(
            "--save-baseline records full-budget runs; it cannot be "
            "combined with --budget-from-baseline",
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
    base_records = _baseline_records(ns)
    selected = _select_entries(
        manifest,
        ns.preset,
        ns.cases,
        harness_budgets.passing_keys(base_records),
    )
    if ns.case_budgets is not None:
        try:
            ns.case_budget_table = case_budgets.load_case_budgets(
                ns.case_budgets
            )
        except (ValueError, OSError) as exc:
            print(f"cannot read --case-budgets: {exc}", file=sys.stderr)
            return 1
    if ns.shard is not None:
        selected = harness_shards.shard_entries(
            selected,
            ns.shard[0],
            ns.shard[1],
            _costly_keys(ns),
            ns.case_budget_table or None,
        )
        print(
            f"[harness] shard {ns.shard[0]}/{ns.shard[1]}: "
            f"{len(selected)} cases",
            flush=True,
        )
    ns.short_budget_keys = _short_budget_keys(ns, selected)
    try:
        ns.budgets = _case_budgets(ns, selected, base_records)
    except (ValueError, OSError) as exc:
        print(f"cannot derive per-case budgets: {exc}", file=sys.stderr)
        return 1
    ns.budgeted_keys = frozenset(
        k for k, b in ns.budgets.items() if b.from_baseline
    )
    if ns.short_budget_keys:
        print(
            f"[harness] {len(ns.short_budget_keys)} cases whose baseline run "
            f"ended on the time budget get {ns.short_budget_sec:g} s: "
            + ", ".join(_label(k) for k in sorted(ns.short_budget_keys)),
            flush=True,
        )
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

    run_info: dict[str, Any] = {
        "numerics_mode": ns.numerics_mode,
        "max_steps": ns.max_steps,
        "allow_unsupported_controls": bool(ns.allow_unsupported_controls),
        "timeout_sec": ns.timeout_sec,
        "short_budget_sec": ns.short_budget_sec,
        "max_rss_gb": ns.max_rss_gb,
        "shard": list(ns.shard) if ns.shard is not None else None,
        "short_budget_cases": [_label(k) for k in sorted(ns.short_budget_keys)],
        "selected_cases": [_label(e.key) for e in selected],
        "budget_from_baseline": (
            {
                "factor": ns.budget_factor,
                "floor_sec": ns.budget_floor_sec,
                "wall_times_json": ns.wall_times_json is not None,
            }
            if ns.budget_from_baseline
            else None
        ),
        "case_budgets": (
            pathlib.Path(ns.case_budgets).name
            if ns.case_budgets is not None
            else None
        ),
        "mesa_step_factor": ns.mesa_step_factor,
        "early_stop_on_divergence": bool(ns.early_stop_on_divergence),
        "enforce_step_counts": bool(ns.enforce_step_counts),
        "stop_on_excess_steps": (
            [ns.step_margin_abs, ns.step_margin_frac]
            if ns.stop_on_excess_steps
            else None
        ),
        "budgeted_cases": [_label(k) for k in sorted(ns.budgeted_keys)],
    }
    return _report_and_gate(
        ns,
        manifest,
        gaps,
        records,
        run_info,
        short_keys=ns.short_budget_keys,
        fig_links=fig_links,
    )


if __name__ == "__main__":
    raise SystemExit(trace_main())
