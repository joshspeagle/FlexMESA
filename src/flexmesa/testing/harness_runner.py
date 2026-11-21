# SPDX-License-Identifier: LGPL-3.0-or-later
"""Guarded execution and verdict evaluation for upstream MESA trajectories.

Executes ``star``, ``binary``, and ``astero`` test-suite cases with optional
wall-clock and resident-memory limits while preserving incrementally flushed
``LOGS/history.data`` and ``binary_history.data`` files on disk, then evaluates
verdicts against ``case_manifest.toml`` and ``known_gaps.toml``.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from flexmesa import _dispatch, _rust
from flexmesa.data import fetch
from flexmesa.testing import history_compare
from flexmesa.testing._case_verdict import (
    CaseHarnessRecord,
    TraceRunOutcome,
    _sanitize_error_message,
    count_unsupported_controls,
    evaluate_case_record,
    extract_reference_run_stats,
    is_unsupported_controls_rejection,
    resolve_binary_component_reference_paths,
    resolve_full_reference_history_path,
    resolve_reference_history_path,
)

__all__ = [
    "CaseHarnessRecord",
    "TraceRunOutcome",
    "_sanitize_error_message",
    "count_unsupported_controls",
    "evaluate_case_record",
    "extract_reference_run_stats",
    "is_unsupported_controls_rejection",
    "locate_written_history",
    "proc_rss_gb",
    "resolve_binary_component_reference_paths",
    "resolve_case_dir",
    "resolve_effective_numerics_mode",
    "resolve_full_reference_history_path",
    "resolve_mesa_test_root",
    "resolve_reference_history_path",
    "run_case_with_diagnostic",
    "run_trace_inprocess",
    "run_traced_case",
]


def resolve_effective_numerics_mode(
    numerics_mode: str | None = None,
) -> str:
    """Resolve effective ``"strict"``/``"fast"`` mode from arg, env, or process.

    Args:
        numerics_mode: Optional explicit mode override (``"strict"`` or
            ``"fast"``). When ``None``, inspects ``FLEXMESA_NUMERICS_MODE``
            and falls back to :func:`flexmesa.get_numerics_mode`.

    Returns:
        Canonical lowercase mode string (``"strict"`` or ``"fast"``).

    Raises:
        ValueError: If ``numerics_mode`` or ``FLEXMESA_NUMERICS_MODE`` is not
            ``"strict"`` or ``"fast"``.
    """
    if numerics_mode is not None:
        return _dispatch.validate_numerics_mode(numerics_mode)
    env_raw = os.environ.get("FLEXMESA_NUMERICS_MODE", "").strip()
    if env_raw:
        return _dispatch.validate_numerics_mode(
            env_raw, label="FLEXMESA_NUMERICS_MODE"
        )
    return _dispatch.validate_numerics_mode(
        str(_dispatch.get_numerics_mode()), label="active numerics_mode"
    )


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


def _extract_run_history_telemetry(
    hist: pathlib.Path | None,
    suite: str,
) -> tuple[int, int | None, int | None, float | None]:
    """Return ``(rows, retries, iters_sum, mean_iters)`` from ``hist``."""
    if hist is None or not hist.is_file():
        return 0, None, None, None
    try:
        tbl = history_compare.load_history_file(hist)
    except (ValueError, OSError):
        return 0, None, None, None
    stats = history_compare.extract_history_run_stats(tbl, suite)
    retries = stats.retries
    iters_sum = stats.iters_sum
    mean_iters = stats.mean_iters
    if suite == "binary" and (retries is None or iters_sum is None):
        logs1 = hist.parent / "LOGS1" / "history.data"
        if logs1.is_file():
            try:
                c_tbl = history_compare.load_history_file(logs1)
                c_stats = history_compare.extract_history_run_stats(
                    c_tbl, "star"
                )
                if retries is None:
                    retries = c_stats.retries
                if iters_sum is None:
                    iters_sum = c_stats.iters_sum
                    mean_iters = c_stats.mean_iters
            except (ValueError, OSError):
                pass
    return tbl.num_rows, retries, iters_sum, mean_iters


def load_extras_factory(spec: str | None) -> Any:
    """Resolve an ``'module.path:callable'`` test-side extras factory spec."""
    raw = (spec or "").strip()
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError(
            f"extras_factory must use 'module:callable' syntax, got {spec!r}"
        )
    mod_name, attr_name = raw.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        candidates = [
            pathlib.Path.cwd(),
            pathlib.Path(__file__).resolve().parents[3],
        ]
        for cand in candidates:
            if (cand / "tests").is_dir():
                s_cand = str(cand)
                if s_cand not in sys.path:
                    sys.path.insert(0, s_cand)
        mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


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
    numerics_mode: str | None = None,
    allow_unsupported_controls: bool = False,
    extras_factory: str | None = None,
    extras: Any = None,
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
        numerics_mode: Arithmetic mode (``"strict"`` or ``"fast"``; defaults
            to active ``FLEXMESA_NUMERICS_MODE`` / process mode).
        allow_unsupported_controls: Bypass strict namelist rejection of
            unread, unused, or unimplemented MESA controls when True.
        extras_factory: Optional ``'module:callable'`` specification returning
            test-side extras handlers from ``(suite, case_dir)``.
        extras: Explicit pre-built test-side extras callback object; overrides
            ``extras_factory`` when provided.

    Returns:
        Raw telemetry dictionary returned by ``flexmesa._rust``.

    Raises:
        ValueError: If ``max_steps < 0``, ``numerics_mode`` is invalid,
            ``suite`` is unknown, or custom extras are passed to ``"astero"``.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    eff_mode = resolve_effective_numerics_mode(numerics_mode)
    if is_logs_dir:
        out_target.mkdir(parents=True, exist_ok=True)
    else:
        out_target.parent.mkdir(parents=True, exist_ok=True)

    resolved_extras = extras
    if resolved_extras is None and extras_factory:
        factory = load_extras_factory(extras_factory)
        if factory is not None:
            resolved_extras = factory(suite, case_dir)

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
                numerics_mode=eff_mode,
                allow_unsupported_controls=allow_unsupported_controls,
                extras=resolved_extras,
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
                numerics_mode=eff_mode,
                allow_unsupported_controls=allow_unsupported_controls,
                extras=resolved_extras,
            )
        )
    if suite == "astero":
        if resolved_extras is not None:
            raise ValueError(
                "astero suite does not accept custom star/binary extras"
            )
        return dict(
            _rust.trace_astero_case(
                c_str,
                o_str,
                max_steps_per_stage=max_steps,
                skip_optional=skip_optional,
                respect_intervals=respect_intervals,
                write_profiles=write_profiles,
                is_logs_dir=is_logs_dir,
                numerics_mode=eff_mode,
                allow_unsupported_controls=allow_unsupported_controls,
            )
        )
    raise ValueError(f"unsupported suite {suite!r}")


def _build_worker_env(
    worker_tmp: pathlib.Path,
    numerics_mode: str,
) -> dict[str, str]:
    env = os.environ.copy()
    src_root = str(pathlib.Path(__file__).resolve().parents[2])
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_py}" if existing_py else src_root
    )
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning:runpy")
    env["FLEXMESA_NUMERICS_MODE"] = numerics_mode
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
    numerics_mode: str,
    allow_unsupported_controls: bool,
    extras_factory: str | None,
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
            "--numerics-mode",
            numerics_mode,
        ]
        if max_steps is not None:
            cmd.extend(["--max-steps", str(max_steps)])
        if not skip_optional:
            cmd.append("--full-stages")
        if respect_intervals:
            cmd.append("--respect-intervals")
        if write_profiles:
            cmd.append("--write-profiles")
        if allow_unsupported_controls:
            cmd.append("--allow-unsupported-controls")
        if extras_factory:
            cmd.extend(["--extras-factory", extras_factory])

        cap_kb = int(max_rss_gb * 1.0e6) if max_rss_gb is not None else None
        peak_kb = 0
        kill_status = ""
        worker_env = _build_worker_env(worker_tmp, numerics_mode)

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
        hist_rows, hist_retries, hist_iters, hist_mean_iters = (
            _extract_run_history_telemetry(hist, suite)
        )
        err_text = stderr_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()

        if kill_status:
            return TraceRunOutcome(
                suite=suite,
                case=case_name,
                status=kill_status,
                termination_code=kill_status,
                steps=hist_rows,
                retries=hist_retries if hist_retries is not None else 0,
                wall_time_sec=wall,
                peak_rss_gb=peak_kb / 1.0e6,
                history_path=hist,
                error_message=err_text[-400:] if err_text else kill_status,
                iters=hist_iters,
                mean_iters=hist_mean_iters,
            )

        if retcode == 0 and meta_json.is_file():
            payload = json.loads(meta_json.read_text(encoding="utf-8"))
            rep_retries = int(payload.get("total_retries", 0))
            if (
                suite == "binary"
                and rep_retries == 0
                and hist_retries is not None
            ):
                rep_retries = hist_retries
            return TraceRunOutcome(
                suite=suite,
                case=case_name,
                status="ok",
                termination_code=str(payload.get("termination_code", "")),
                steps=int(payload.get("total_steps", 0)),
                retries=rep_retries,
                wall_time_sec=wall,
                peak_rss_gb=max(
                    peak_kb / 1.0e6, float(payload.get("rss_gb", 0.0))
                ),
                history_path=hist,
                iters=hist_iters,
                mean_iters=hist_mean_iters,
            )

        return TraceRunOutcome(
            suite=suite,
            case=case_name,
            status="error",
            termination_code="",
            steps=hist_rows,
            retries=hist_retries if hist_retries is not None else 0,
            wall_time_sec=wall,
            peak_rss_gb=peak_kb / 1.0e6,
            history_path=hist,
            error_message=(
                _summarize_worker_error(err_text)
                if err_text
                else f"exit={retcode}"
            ),
            n_controls_ignored=(
                count_unsupported_controls(err_text)
                if is_unsupported_controls_rejection(err_text)
                else 0
            ),
            iters=hist_iters,
            mean_iters=hist_mean_iters,
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
    numerics_mode: str | None = None,
    allow_unsupported_controls: bool = False,
    extras_factory: str | None = None,
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
        numerics_mode: Arithmetic mode (``"strict"`` or ``"fast"``; defaults
            to active ``FLEXMESA_NUMERICS_MODE`` / process mode).
        allow_unsupported_controls: Bypass strict namelist rejection of
            unread, unused, or unimplemented MESA controls when True.
        extras_factory: Optional ``'module:callable'`` specification returning
            test-side extras handlers from ``(suite, case_dir)``.
        timeout_sec: Optional wall-clock timeout in seconds (``s``).
        max_rss_gb: Optional resident-memory ceiling in gigabytes (``GB``).

    Returns:
        ``TraceRunOutcome`` summarizing execution status, termination, resource
        usage, and emitted history path.

    Raises:
        ValueError: If ``max_steps < 0`` or ``numerics_mode`` is invalid.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    eff_mode = resolve_effective_numerics_mode(numerics_mode)
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
            numerics_mode=eff_mode,
            allow_unsupported_controls=allow_unsupported_controls,
            extras_factory=extras_factory,
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
            numerics_mode=eff_mode,
            allow_unsupported_controls=allow_unsupported_controls,
            extras_factory=extras_factory,
        )
        wall = time.monotonic() - t0
        hist = locate_written_history(suite, out_target, is_logs_dir)
        _, hist_retries, hist_iters, hist_mean_iters = (
            _extract_run_history_telemetry(hist, suite)
        )
        rep_retries = int(
            rep.get("total_retries", rep.get("binary_retries", 0))
        )
        if suite == "binary" and rep_retries == 0 and hist_retries is not None:
            rep_retries = hist_retries
        return TraceRunOutcome(
            suite=suite,
            case=case_name,
            status="ok",
            termination_code=str(rep.get("termination_code", "")),
            steps=int(rep.get("total_steps", 0)),
            retries=rep_retries,
            wall_time_sec=wall,
            peak_rss_gb=proc_rss_gb(os.getpid()),
            history_path=hist,
            iters=hist_iters,
            mean_iters=hist_mean_iters,
        )
    except BaseException as err:
        if isinstance(err, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        wall = time.monotonic() - t0
        hist = locate_written_history(suite, out_target, is_logs_dir)
        hist_rows, hist_retries, hist_iters, hist_mean_iters = (
            _extract_run_history_telemetry(hist, suite)
        )
        return TraceRunOutcome(
            suite=suite,
            case=case_name,
            status="error",
            termination_code="",
            steps=hist_rows,
            retries=hist_retries if hist_retries is not None else 0,
            wall_time_sec=wall,
            peak_rss_gb=proc_rss_gb(os.getpid()),
            history_path=hist,
            error_message=str(err),
            n_controls_ignored=(
                count_unsupported_controls(str(err))
                if is_unsupported_controls_rejection(str(err))
                else 0
            ),
            iters=hist_iters,
            mean_iters=hist_mean_iters,
        )


def _summarize_worker_error(err_text: str, limit: int = 500) -> str:
    """Return the final error line of a worker's stderr, cut to ``limit``.

    The last non-empty line of a Python traceback is the exception itself,
    which is the part worth keeping. A long line (for example a list of
    unsupported controls) is cut at the end so that its start, which says
    what kind of error it is, survives.

    Args:
        err_text: Full captured stderr of the worker process.
        limit: Maximum number of characters to keep.

    Returns:
        The summarized message, at most ``limit`` characters long.
    """
    lines = [ln for ln in err_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1].strip()
    if len(last) > limit:
        return last[: limit - 3] + "..."
    return last


def run_case_with_diagnostic(
    suite: str,
    case_dir: pathlib.Path,
    out_target: pathlib.Path,
    *,
    max_steps: int | None = None,
    numerics_mode: str | None = None,
    allow_unsupported_controls: bool = False,
    extras_factory: str | None = None,
    timeout_sec: float | None = None,
    max_rss_gb: float | None = None,
) -> TraceRunOutcome:
    """Run one case strictly, then diagnostically if strict rejects it.

    The case is first run under strict ``#147`` control validation. If that
    run stops because the inlists set upstream controls ``FlexMESA`` does not
    support, the case is run a second time with those controls ignored, so
    the table still reports steps, termination and deviations. The returned
    outcome of such a rerun has ``controls_ignored`` set, which bars it from
    a ``pass`` verdict in ``evaluate_case_record``.

    Args:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case_dir: Path to the upstream case directory.
        out_target: Output root directory for the run.
        max_steps: Optional non-negative per-stage step cap.
        numerics_mode: Arithmetic mode (``"strict"`` or ``"fast"``).
        allow_unsupported_controls: Skip the strict run and go straight to
            the diagnostic run when True.
        extras_factory: Optional ``'module:callable'`` specification returning
            test-side extras handlers from ``(suite, case_dir)``.
        timeout_sec: Optional wall-clock timeout in seconds (``s``), applied
            to each of the two runs separately.
        max_rss_gb: Optional resident-memory ceiling in gigabytes (``GB``).

    Returns:
        The strict ``TraceRunOutcome`` when strict validation accepted the
        inlists (whatever happened afterwards), otherwise the outcome of the
        diagnostic rerun.
    """
    strict_message = ""
    n_ignored = 0
    if not allow_unsupported_controls:
        strict = run_traced_case(
            suite,
            case_dir,
            out_target,
            max_steps=max_steps,
            numerics_mode=numerics_mode,
            allow_unsupported_controls=False,
            extras_factory=extras_factory,
            timeout_sec=timeout_sec,
            max_rss_gb=max_rss_gb,
        )
        rejected = strict.status == "error" and (
            strict.n_controls_ignored > 0
            or is_unsupported_controls_rejection(strict.error_message)
        )
        if not rejected:
            return strict
        strict_message = strict.error_message
        n_ignored = strict.n_controls_ignored or count_unsupported_controls(
            strict.error_message
        )
    diag = run_traced_case(
        suite,
        case_dir,
        out_target,
        max_steps=max_steps,
        numerics_mode=numerics_mode,
        allow_unsupported_controls=True,
        extras_factory=extras_factory,
        timeout_sec=timeout_sec,
        max_rss_gb=max_rss_gb,
    )
    return dataclasses.replace(
        diag,
        controls_ignored=True,
        n_controls_ignored=n_ignored,
        strict_rejection=strict_message,
    )
