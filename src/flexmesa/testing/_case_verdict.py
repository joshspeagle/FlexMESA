# SPDX-License-Identifier: LGPL-3.0-or-later
"""Governed verdict evaluation and reference-path resolution for ``#143``.

Evaluates trajectory runs against ``case_manifest.toml`` and
``known_gaps.toml``, enforcing ``r26.4.1`` header provenance, pulsation
scaffolding tags (``#170``, ``#171``), and library-extras tracking (``#148``).
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

from flexmesa.data import fetch
from flexmesa.testing import case_manifest, history_compare


@dataclasses.dataclass(frozen=True)
class TraceRunOutcome:
    """Execution telemetry from a ``star``, ``binary``, or ``astero`` run.

    Attributes:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case: Case directory name.
        status: Execution status (``"ok"``, ``"error"``, ``"timeout"``, or
            ``"oom_killed"``).
        termination_code: Observed termination code (or empty string on crash).
        steps: Total evolutionary steps completed before exit.
        retries: Total step retries recorded.
        wall_time_sec: Elapsed wall-clock time in seconds (``s``).
        peak_rss_gb: Peak resident memory of the worker in gigabytes (``GB``).
        history_path: Path to the emitted ``history.data`` or
            ``binary_history.data`` file if present on disk, else ``None``.
        error_message: Diagnostic error string when ``status != "ok"``.
    """

    suite: str
    case: str
    status: str
    termination_code: str
    steps: int
    retries: int
    wall_time_sec: float
    peak_rss_gb: float
    history_path: pathlib.Path | None
    error_message: str = ""


@dataclasses.dataclass(frozen=True)
class CaseHarnessRecord:
    """Manifest, execution, and trajectory-comparison record for one case."""

    suite: str
    case: str
    reference_status: str
    reference_path: str
    required_termination: str
    verdict: str
    executed: bool
    run_status: str
    observed_termination: str
    termination_matches: bool
    steps: int
    retries: int
    wall_time_sec: float
    peak_rss_gb: float
    known_gap_issue: str
    known_gap_reason: str
    comparison: history_compare.HistoryComparison | None
    error_message: str = ""
    is_undeclared_failure: bool = False
    is_stale_known_gap: bool = False


def resolve_reference_history_path(
    reference_path: str,
    root_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve ``reference_path`` against ``root_dir``, repo root, or cache."""
    if not reference_path:
        raise FileNotFoundError("empty reference_path")
    candidates: list[pathlib.Path] = []
    if root_dir is not None:
        candidates.append(root_dir / reference_path)
    candidates.append(case_manifest.repo_root() / reference_path)
    candidates.append(fetch.resolve_mesa_histories_dir() / reference_path)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"missing reference history {reference_path!r} "
        "(run 'flexmesa data fetch --include-test-inputs')"
    )


def _sanitize_error_message(msg: str) -> str:
    if not msg:
        return ""
    cleaned = msg
    for raw_dir, token in (
        (str(fetch.resolve_mesa_test_inputs_dir()), "$MESA_DIR"),
        (str(fetch.resolve_mesa_histories_dir()), "$FLEXMESA_HISTORIES_DIR"),
        (str(fetch.resolve_data_dir()), "$FLEXMESA_DATA_DIR"),
        (str(case_manifest.repo_root()), "<repo>"),
        (str(pathlib.Path.home()), "~"),
    ):
        if raw_dir:
            cleaned = cleaned.replace(raw_dir, token)
    cleaned = re.sub(r"/(?:tmp|var/folders)/[^\s:'\")\]]+", "<tmpdir>", cleaned)
    cleaned = re.sub(r"/(?:home|Users|mnt)/[^\s:'\")\]]+", "<path>", cleaned)
    return cleaned


def evaluate_case_record(
    spec: case_manifest.CaseManifestEntry,
    known_gaps: dict[tuple[str, str], case_manifest.KnownGapSpec],
    outcome: TraceRunOutcome | None,
    *,
    max_steps: int | None = None,
    root_dir: pathlib.Path | None = None,
    precomputed_comparison: history_compare.HistoryComparison | None = None,
) -> CaseHarnessRecord:
    """Combine manifest metadata, run outcome, and comparison into a verdict."""
    gap = known_gaps.get(spec.key)
    gap_issue = gap.issue if gap is not None else ""
    gap_reason = gap.reason if gap is not None else ""
    base = case_manifest.repo_root() if root_dir is None else root_dir

    if outcome is None:
        if spec.reference_status == "no_reference_yet":
            verdict = "no reference yet"
        elif spec.reference_status == "interim_0b2e6bb":
            verdict = "interim reference (not run)"
        else:
            verdict = "not run"
        return CaseHarnessRecord(
            suite=spec.suite,
            case=spec.case,
            reference_status=spec.reference_status,
            reference_path=spec.reference_path,
            required_termination=spec.required_termination,
            verdict=verdict,
            executed=False,
            run_status="not_run",
            observed_termination="",
            termination_matches=False,
            steps=0,
            retries=0,
            wall_time_sec=0.0,
            peak_rss_gb=0.0,
            known_gap_issue=gap_issue,
            known_gap_reason=gap_reason,
            comparison=None,
        )

    term_ok = case_manifest.termination_matches(
        spec.required_termination, outcome.termination_code
    )
    comp: history_compare.HistoryComparison | None = (
        history_compare.re_evaluate_comparison(precomputed_comparison, spec)
        if precomputed_comparison is not None
        else None
    )
    comp_err = _sanitize_error_message(outcome.error_message)

    if (
        comp is None
        and spec.reference_status != "no_reference_yet"
        and spec.reference_path
    ):
        ref_file = resolve_reference_history_path(
            spec.reference_path, root_dir=base
        )
        ref_tbl = history_compare.load_history_file(ref_file)
        expected_ver = fetch.load_upstream_reference().version_number
        if (
            spec.reference_status == "mesa_26_4_1"
            and not ref_tbl.provenance_comment
            and ref_tbl.mesa_version_number != expected_ver
        ):
            raise ValueError(
                f"{spec.reference_path}: expected MESA version_number "
                f"{expected_ver!r}, got {ref_tbl.mesa_version_number!r}"
            )
        if spec.independent_var not in ref_tbl.columns:
            raise KeyError(
                f"{spec.reference_path}: missing independent_var "
                f"{spec.independent_var!r}"
            )
        if outcome.history_path is not None:
            try:
                run_tbl = history_compare.load_history_file(
                    outcome.history_path
                )
                comp = history_compare.compare_histories(ref_tbl, run_tbl, spec)
            except (OSError, ValueError, KeyError) as err:
                err_s = _sanitize_error_message(str(err))
                comp_err = f"{comp_err}; compare error: {err_s}".strip("; ")

    tol_ok = (
        comp is not None
        and comp.within_final_tolerances
        and comp.within_track_tolerances
    )
    is_undeclared = False
    is_stale_gap = False

    if max_steps is not None:
        if outcome.status == "ok":
            verdict = f"smoke: {outcome.steps} steps"
        else:
            verdict = f"smoke: {outcome.status} ({outcome.steps} steps)"
    elif spec.reference_status == "no_reference_yet":
        verdict = "no reference yet"
    elif spec.reference_status == "interim_0b2e6bb":
        if outcome.status in ("timeout", "oom_killed"):
            verdict = f"interim reference ({outcome.status})"
        elif outcome.status != "ok" or comp is None:
            verdict = "interim reference (run failed)"
        elif not term_ok:
            verdict = "interim reference (wrong termination)"
        elif tol_ok:
            verdict = "interim reference (within tol)"
        else:
            verdict = "interim reference (outside tol)"
        if verdict == "interim reference (within tol)":
            is_stale_gap = gap is not None
        else:
            is_undeclared = gap is None
    else:
        if outcome.status == "ok" and term_ok and tol_ok:
            if spec.uses_pulsation_scaffolding:
                verdict = f"scaffolding ({spec.scaffolding_issue})"
            elif spec.uses_library_extras:
                if gap is not None:
                    verdict = f"declared gap ({gap.issue})"
                else:
                    verdict = "fail (undeclared)"
                    is_undeclared = True
            else:
                verdict = "pass"
                is_stale_gap = gap is not None
        elif gap is not None:
            verdict = f"declared gap ({gap.issue})"
        else:
            verdict = "fail (undeclared)"
            is_undeclared = True

    return CaseHarnessRecord(
        suite=spec.suite,
        case=spec.case,
        reference_status=spec.reference_status,
        reference_path=spec.reference_path,
        required_termination=spec.required_termination,
        verdict=verdict,
        executed=True,
        run_status=outcome.status,
        observed_termination=outcome.termination_code,
        termination_matches=term_ok,
        steps=outcome.steps,
        retries=outcome.retries,
        wall_time_sec=outcome.wall_time_sec,
        peak_rss_gb=outcome.peak_rss_gb,
        known_gap_issue=gap_issue,
        known_gap_reason=gap_reason,
        comparison=comp,
        error_message=comp_err,
        is_undeclared_failure=is_undeclared,
        is_stale_known_gap=is_stale_gap,
    )
