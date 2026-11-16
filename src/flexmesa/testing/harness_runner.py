# SPDX-License-Identifier: LGPL-3.0-or-later
"""Guarded execution and verdict evaluation for upstream MESA trajectories.

Executes ``star``, ``binary``, and ``astero`` test-suite cases with optional
wall-clock and resident-memory limits while preserving incrementally flushed
``LOGS/history.data`` and ``binary_history.data`` files on disk, then evaluates
verdicts against ``case_manifest.toml`` and ``known_gaps.toml``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from flexmesa import _rust
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
    """Manifest, execution, and trajectory-comparison record for one case.

    Attributes:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case: Case directory name.
        reference_status: Reference status from ``case_manifest.toml``.
        reference_path: Stored reference path (or empty string).
        required_termination: Expected upstream termination string.
        verdict: Governed verdict label (for example ``"no reference yet"``,
            ``"smoke: 3 steps"``, ``"interim reference (within tol)"``,
            ``"pass"``, ``"declared gap (#155)"``, or ``"fail (undeclared)"``).
        executed: True when ``FlexMESA`` ran the case in this invocation.
        run_status: Execution status (``"not_run"``, ``"ok"``, ``"error"``,
            ``"timeout"``, or ``"oom_killed"``).
        observed_termination: Observed termination string when executed.
        termination_matches: True when ``observed_termination`` satisfies
            ``required_termination``.
        steps: Steps completed (``0`` if not executed).
        retries: Retries recorded (``0`` if not executed).
        wall_time_sec: Elapsed runtime in seconds (``s``).
        peak_rss_gb: Peak resident memory in gigabytes (``GB``).
        known_gap_issue: Linked ``#<issue>`` from ``known_gaps.toml`` if any.
        known_gap_reason: Declared gap explanation if any.
        comparison: ``HistoryComparison`` when both reference and run histories
            exist, else ``None``.
        error_message: Error detail if execution or log parsing failed.
        is_undeclared_failure: True when an uncapped run fails without an entry
            in ``known_gaps.toml``.
        is_stale_known_gap: True when a case in ``known_gaps.toml`` meets all
            termination and trajectory criteria.
    """

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


def resolve_mesa_test_root(
    mesa_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Locate and validate the unpacked MESA test-suite root directory.

    Args:
        mesa_dir: Optional explicit root directory path.

    Returns:
        Resolved ``pathlib.Path`` containing ``star/test_suite``.

    Raises:
        FileNotFoundError: If the resolved directory does not contain
            ``star/test_suite``.
    """
    cand = fetch.resolve_mesa_test_inputs_dir(mesa_dir)
    if (cand / "star" / "test_suite").is_dir():
        return cand
    if mesa_dir is None:
        fallback = (
            pathlib.Path.home() / ".cache" / "flexmesa" / "mesa_test_inputs"
        )
        if (fallback / "star" / "test_suite").is_dir():
            return fallback.resolve()
    msg = (
        f"MESA test inputs not found at {cand}; set MESA_DIR or run "
        "'flexmesa data fetch --include-test-inputs'"
    )
    raise FileNotFoundError(msg)


def resolve_case_dir(
    suite: str,
    case_or_path: str,
    mesa_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Resolve a case name or filesystem path to an existing directory.

    Args:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case_or_path: Case directory name (such as ``"1.3M_ms_high_Z"``) or
            explicit directory path.
        mesa_dir: Optional MESA test-inputs root override.

    Returns:
        Resolved ``pathlib.Path`` to the case directory.

    Raises:
        FileNotFoundError: If the target case directory does not exist.
    """
    direct = pathlib.Path(case_or_path)
    if direct.is_dir() and (
        (direct / "rn").is_file() or any(direct.glob("inlist*"))
    ):
        return direct.resolve()
    base = resolve_mesa_test_root(mesa_dir)
    target = base / suite / "test_suite" / case_or_path
    if not target.is_dir():
        raise FileNotFoundError(f"missing {suite} case directory: {target}")
    return target.resolve()


def resolve_reference_history_path(
    reference_path: str,
    root_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve ``reference_path`` against the repo root or ``mesa_histories``.

    Args:
        reference_path: Relative path from ``case_manifest.toml`` (either under
            ``benchmarks/history_harness/reference_histories/`` in the
            repository or ``thinned/`` inside ``mesa-histories-v1.tar.zst``).
        root_dir: Optional repository or manifest directory override.

    Returns:
        Resolved ``pathlib.Path`` to the existing reference history file.

    Raises:
        FileNotFoundError: If ``reference_path`` is empty or absent in both the
            repository root and the unpacked ``mesa_histories`` cache.
    """
    if not reference_path:
        raise FileNotFoundError("empty reference_path")
    base = case_manifest.repo_root() if root_dir is None else root_dir
    repo_cand = base / reference_path
    if repo_cand.is_file():
        return repo_cand
    cache_cand = fetch.resolve_mesa_histories_dir() / reference_path
    if cache_cand.is_file():
        return cache_cand
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


def _proc_rss_kb(pid: int) -> int:
    status_file = pathlib.Path(f"/proc/{pid}/status")
    if status_file.is_file():
        try:
            for line in status_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            return 0
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out) if out else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def proc_rss_gb(pid: int) -> float:
    """Return the resident set size of ``pid`` in gigabytes (``GB``).

    Args:
        pid: Operating-system process identifier.

    Returns:
        Resident memory in ``GB`` (``0.0`` if the process has exited).
    """
    return _proc_rss_kb(pid) / 1.0e6


def _clear_stale_history_outputs(
    out_target: pathlib.Path,
    is_logs_dir: bool,
) -> None:
    if not is_logs_dir:
        out_target.unlink(missing_ok=True)
        return
    if not out_target.is_dir():
        return
    for name in ("history.data", "binary_history.data"):
        for stale in out_target.rglob(name):
            if stale.is_file():
                stale.unlink(missing_ok=True)


def locate_written_history(
    suite: str,
    out_target: pathlib.Path,
    is_logs_dir: bool,
) -> pathlib.Path | None:
    """Locate the emitted history file, including partial interrupted tracks.

    Selects the most recently modified history file under ``out_target`` so
    multi-stage cases whose final stage uses a custom ``log_directory`` win
    over preliminary stages.

    Args:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        out_target: Output file or root directory passed to the tracer.
        is_logs_dir: True when ``out_target`` is a directory root.

    Returns:
        Path to the non-empty history file if present, else ``None``.
    """
    if not is_logs_dir:
        if out_target.is_file() and out_target.stat().st_size > 0:
            return out_target
        return None
    if not out_target.is_dir():
        return None

    if suite == "binary":
        primary = out_target / "binary_history.data"
        pattern = "binary_history.data"
    else:
        primary = out_target / "LOGS" / "history.data"
        pattern = "history.data"

    candidates = [
        p
        for p in out_target.rglob(pattern)
        if p.is_file()
        and p.stat().st_size > 0
        and not p.name.startswith("sample_")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: (p.stat().st_mtime_ns, p == primary),
    )


def _count_history_rows(hist: pathlib.Path | None) -> int:
    if hist is None:
        return 0
    try:
        return history_compare.load_history_file(hist).num_rows
    except (ValueError, OSError):
        return 0


def run_trace_inprocess(
    suite: str,
    case_dir: pathlib.Path,
    out_target: pathlib.Path,
    *,
    max_steps: int | None = None,
    skip_optional: bool = True,
    respect_intervals: bool = False,
    write_profiles: bool = False,
    is_logs_dir: bool = True,
) -> dict[str, Any]:
    """Invoke the compiled Safe Rust tracer directly in the current process.

    Args:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case_dir: Path to the upstream case directory.
        out_target: Output directory or explicit history file path.
        max_steps: Optional non-negative per-stage step cap.
        skip_optional: Skip optional multi-stage setup stages when True.
        respect_intervals: Honor ``history_interval`` instead of every step.
        write_profiles: Enable writing radial ``profileN.data`` files.
        is_logs_dir: Treat ``out_target`` as a root directory when True.

    Returns:
        Raw telemetry dictionary returned by ``flexmesa._rust``.

    Raises:
        ValueError: If ``max_steps < 0`` or ``suite`` is unknown.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    if is_logs_dir:
        out_target.mkdir(parents=True, exist_ok=True)
    else:
        out_target.parent.mkdir(parents=True, exist_ok=True)

    c_str = str(case_dir)
    o_str = str(out_target)
    if suite == "star":
        return dict(
            _rust.trace_star_case(
                c_str,
                o_str,
                max_steps_per_stage=max_steps,
                skip_optional=skip_optional,
                respect_intervals=respect_intervals,
                write_profiles=write_profiles,
                is_logs_dir=is_logs_dir,
            )
        )
    if suite == "binary":
        return dict(
            _rust.trace_binary_case(
                c_str,
                o_str,
                max_steps=max_steps,
                respect_intervals=respect_intervals,
                is_logs_dir=is_logs_dir,
            )
        )
    if suite == "astero":
        return dict(
            _rust.trace_astero_case(
                c_str,
                o_str,
                max_steps_per_stage=max_steps,
                skip_optional=skip_optional,
                respect_intervals=respect_intervals,
                write_profiles=write_profiles,
                is_logs_dir=is_logs_dir,
            )
        )
    raise ValueError(f"unsupported suite {suite!r}")


def _build_worker_env(worker_tmp: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    src_root = str(pathlib.Path(__file__).resolve().parents[2])
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_py}" if existing_py else src_root
    )
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning:runpy")
    tmp_str = str(worker_tmp)
    env["TMPDIR"] = tmp_str
    env["TEMP"] = tmp_str
    env["TMP"] = tmp_str
    return env


def _kill_worker_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()
    proc.wait()


def _run_traced_worker(
    suite: str,
    case_dir: pathlib.Path,
    out_target: pathlib.Path,
    *,
    max_steps: int | None,
    skip_optional: bool,
    respect_intervals: bool,
    write_profiles: bool,
    is_logs_dir: bool,
    timeout_sec: float | None,
    max_rss_gb: float | None,
    t0: float,
) -> TraceRunOutcome:
    case_name = case_dir.name
    with tempfile.TemporaryDirectory(prefix="flexmesa_worker_") as meta_dir:
        meta_root = pathlib.Path(meta_dir)
        worker_tmp = meta_root / "tmp"
        worker_tmp.mkdir(parents=True, exist_ok=True)
        meta_json = meta_root / "result.json"
        stderr_path = meta_root / "stderr.log"

        cmd = [
            sys.executable,
            "-m",
            "flexmesa.testing.trace_cli",
            "_worker",
            suite,
            str(case_dir),
            str(out_target),
            str(meta_json),
            "--is-logs-dir" if is_logs_dir else "--explicit-file",
        ]
        if max_steps is not None:
            cmd.extend(["--max-steps", str(max_steps)])
        if not skip_optional:
            cmd.append("--full-stages")
        if respect_intervals:
            cmd.append("--respect-intervals")
        if write_profiles:
            cmd.append("--write-profiles")

        cap_kb = int(max_rss_gb * 1.0e6) if max_rss_gb is not None else None
        peak_kb = 0
        kill_status = ""
        worker_env = _build_worker_env(worker_tmp)

        with (
            stderr_path.open("w", encoding="utf-8") as err_fh,
            subprocess.Popen(
                cmd,
                stderr=err_fh,
                env=worker_env,
                start_new_session=(os.name == "posix"),
            ) as proc,
        ):
            try:
                while proc.poll() is None:
                    peak_kb = max(peak_kb, _proc_rss_kb(proc.pid))
                    elapsed = time.monotonic() - t0
                    if timeout_sec is not None and elapsed > timeout_sec:
                        kill_status = "timeout"
                    elif cap_kb is not None and peak_kb > cap_kb:
                        kill_status = "oom_killed"
                    if kill_status:
                        _kill_worker_group(proc)
                        break
                    try:
                        proc.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                _kill_worker_group(proc)
            retcode = proc.returncode

        wall = time.monotonic() - t0
        hist = locate_written_history(suite, out_target, is_logs_dir)
        err_text = stderr_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()

        if kill_status:
            return TraceRunOutcome(
                suite=suite,
                case=case_name,
                status=kill_status,
                termination_code=kill_status,
                steps=_count_history_rows(hist),
                retries=0,
                wall_time_sec=wall,
                peak_rss_gb=peak_kb / 1.0e6,
                history_path=hist,
                error_message=err_text[-400:] if err_text else kill_status,
            )

        if retcode == 0 and meta_json.is_file():
            payload = json.loads(meta_json.read_text(encoding="utf-8"))
            return TraceRunOutcome(
                suite=suite,
                case=case_name,
                status="ok",
                termination_code=str(payload.get("termination_code", "")),
                steps=int(payload.get("total_steps", 0)),
                retries=int(payload.get("total_retries", 0)),
                wall_time_sec=wall,
                peak_rss_gb=max(
                    peak_kb / 1.0e6, float(payload.get("rss_gb", 0.0))
                ),
                history_path=hist,
            )

        return TraceRunOutcome(
            suite=suite,
            case=case_name,
            status="error",
            termination_code="",
            steps=_count_history_rows(hist),
            retries=0,
            wall_time_sec=wall,
            peak_rss_gb=peak_kb / 1.0e6,
            history_path=hist,
            error_message=err_text[-500:] if err_text else f"exit={retcode}",
        )


def run_traced_case(
    suite: str,
    case_dir: pathlib.Path,
    out_target: pathlib.Path,
    *,
    max_steps: int | None = None,
    skip_optional: bool = True,
    respect_intervals: bool = False,
    write_profiles: bool = False,
    is_logs_dir: bool = True,
    timeout_sec: float | None = None,
    max_rss_gb: float | None = None,
) -> TraceRunOutcome:
    """Execute one upstream case with optional wall-time and RSS guards.

    Args:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case_dir: Path to the upstream case directory.
        out_target: Output directory or explicit history file path.
        max_steps: Optional non-negative per-stage step cap.
        skip_optional: Skip optional setup stages when True.
        respect_intervals: Honor ``history_interval`` instead of every step.
        write_profiles: Enable writing radial ``profileN.data`` files.
        is_logs_dir: Treat ``out_target`` as a root directory when True.
        timeout_sec: Optional wall-clock timeout in seconds (``s``).
        max_rss_gb: Optional resident-memory ceiling in gigabytes (``GB``).

    Returns:
        ``TraceRunOutcome`` summarizing execution status, termination, resource
        usage, and emitted history path.

    Raises:
        ValueError: If ``max_steps < 0``.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    _clear_stale_history_outputs(out_target, is_logs_dir)
    case_name = case_dir.name
    t0 = time.monotonic()

    if timeout_sec is not None or max_rss_gb is not None:
        return _run_traced_worker(
            suite,
            case_dir,
            out_target,
            max_steps=max_steps,
            skip_optional=skip_optional,
            respect_intervals=respect_intervals,
            write_profiles=write_profiles,
            is_logs_dir=is_logs_dir,
            timeout_sec=timeout_sec,
            max_rss_gb=max_rss_gb,
            t0=t0,
        )

    try:
        rep = run_trace_inprocess(
            suite,
            case_dir,
            out_target,
            max_steps=max_steps,
            skip_optional=skip_optional,
            respect_intervals=respect_intervals,
            write_profiles=write_profiles,
            is_logs_dir=is_logs_dir,
        )
        wall = time.monotonic() - t0
        hist = locate_written_history(suite, out_target, is_logs_dir)
        return TraceRunOutcome(
            suite=suite,
            case=case_name,
            status="ok",
            termination_code=str(rep.get("termination_code", "")),
            steps=int(rep.get("total_steps", 0)),
            retries=int(rep.get("total_retries", 0)),
            wall_time_sec=wall,
            peak_rss_gb=proc_rss_gb(os.getpid()),
            history_path=hist,
        )
    except BaseException as err:
        if isinstance(err, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        wall = time.monotonic() - t0
        hist = locate_written_history(suite, out_target, is_logs_dir)
        return TraceRunOutcome(
            suite=suite,
            case=case_name,
            status="error",
            termination_code="",
            steps=_count_history_rows(hist),
            retries=0,
            wall_time_sec=wall,
            peak_rss_gb=proc_rss_gb(os.getpid()),
            history_path=hist,
            error_message=str(err),
        )


def evaluate_case_record(
    spec: case_manifest.CaseManifestEntry,
    known_gaps: dict[tuple[str, str], case_manifest.KnownGapSpec],
    outcome: TraceRunOutcome | None,
    *,
    max_steps: int | None = None,
    root_dir: pathlib.Path | None = None,
    precomputed_comparison: history_compare.HistoryComparison | None = None,
) -> CaseHarnessRecord:
    """Combine manifest metadata, run outcome, and comparison into a verdict.

    Args:
        spec: ``CaseManifestEntry`` for the target case.
        known_gaps: Mapping from ``(suite, case)`` to ``KnownGapSpec``.
        outcome: Optional ``TraceRunOutcome`` if executed, else ``None``.
        max_steps: Step cap if run in smoke mode, else ``None``.
        root_dir: Base directory for resolving relative ``spec.reference_path``.
        precomputed_comparison: Optional deserialized ``HistoryComparison`` when
            re-evaluating stored baseline outcomes without re-running models.

    Returns:
        Evaluated ``CaseHarnessRecord``.

    Raises:
        FileNotFoundError: If ``spec.reference_path`` is configured on an
            executed case and the reference file is missing.
        ValueError: If the stored reference file is malformed or carries a
            non-``r26.4.1`` MESA header version.
        KeyError: If the stored reference file lacks ``spec.independent_var``.
    """
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
    comp: history_compare.HistoryComparison | None = precomputed_comparison
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
        if (
            spec.reference_status == "mesa_26_4_1"
            and not ref_tbl.provenance_comment
            and ref_tbl.mesa_version_number != "r26.4.1"
        ):
            raise ValueError(
                f"{spec.reference_path}: expected MESA version_number "
                f"'r26.4.1', got {ref_tbl.mesa_version_number!r}"
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
