#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify `NumericsMode::Strict` cross-machine and thread-count bit-identity.

Implements Items 15 and 16 of `#165` and the uncapped campaign extension (#144):
1. Evaluates all `474,062` MESA `26.4.1` reference inputs in
   `benchmarks/fortran_mesa/math_hex_vectors/mesa_26_4_1_math_vectors.txt`
   under `NumericsMode::Strict` and records the canonicalized `SHA-256` digest.
2. Runs uncapped `strict` evolutions for `star/1.3M_ms_high_Z` and
   `binary/jdot_gr_check` in isolated `1`-thread and `4`-thread subprocesses
   (`RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`), verifying `math_backend =
   "strict"` in component history headers and byte-for-byte identity across all
   emitted history logs (`SHA-256`). Diagnostic overrides (`--max-steps`,
   `--cases`, `--suite`/`--case`) remain supported.
3. Supports `--json-out <path>` to emit a machine-portable summary JSON and
   `--compare <json_a> <json_b>` to gate cross-architecture (`x86_64` vs
   `aarch64`) bit-identity in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_VECTORS = (
    _REPO_ROOT
    / "benchmarks"
    / "fortran_mesa"
    / "math_hex_vectors"
    / "mesa_26_4_1_math_vectors.txt"
)
_DEFAULT_SUITE = "star"
_DEFAULT_CASE = "1.3M_ms_high_Z"
_DEFAULT_CASES: tuple[tuple[str, str], ...] = (
    ("star", "1.3M_ms_high_Z"),
    ("binary", "jdot_gr_check"),
)
_DEFAULT_STEPS: int | None = None


def _collect_history_digests(
    out_dir: pathlib.Path, suite: str
) -> dict[str, dict[str, Any]]:
    """Verify ``strict`` header and hash all history logs in ``out_dir``."""
    candidates: list[pathlib.Path] = []
    if suite == "binary":
        bh = out_dir / "binary_history.data"
        if not bh.is_file():
            raise RuntimeError(f"missing binary history output: {bh}")
        candidates.append(bh)
        for logs_name in ("LOGS1", "LOGS2"):
            lh = out_dir / logs_name / "history.data"
            if lh.is_file():
                candidates.append(lh)
        if not any(p.name == "history.data" for p in candidates):
            raise RuntimeError(
                f"missing component history.data under {out_dir}"
            )
    else:
        lh = out_dir / "LOGS" / "history.data"
        if not lh.is_file():
            raise RuntimeError(f"missing history output: {lh}")
        candidates.append(lh)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(
        candidates, key=lambda p: p.relative_to(out_dir).as_posix()
    ):
        rel = path.relative_to(out_dir).as_posix()
        if path.name == "history.data":
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) < 3 or '"strict"' not in lines[2]:
                raise RuntimeError(
                    f'{rel} header missing math_backend="strict": {lines[:4]!r}'
                )
        raw_bytes = path.read_bytes()
        files[rel] = {
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
    return files


def _combined_files_digest(
    files: dict[str, dict[str, Any]],
) -> tuple[str, int]:
    """Compute canonical combined SHA-256 and total byte size across files."""
    h = hashlib.sha256()
    total_bytes = 0
    for rel in sorted(files):
        entry = files[rel]
        sha = str(entry["sha256"])
        size = int(entry["bytes"])
        total_bytes += size
        h.update(f"{rel}:{size}:{sha}\n".encode("ascii"))
    return h.hexdigest(), total_bytes


def _build_trace_command(
    suite: str, case: str, out_dir: pathlib.Path, max_steps: int | None
) -> list[str]:
    """Construct the ``flexmesa trace`` subprocess command line."""
    cmd = [
        sys.executable,
        "-m",
        "flexmesa.cli",
        "trace",
        suite,
        case,
        "--out",
        str(out_dir),
    ]
    if (_REPO_ROOT / "tests" / "case_extras").is_dir():
        cmd.extend(
            ["--extras-factory", "tests.case_extras:resolve_case_extras"]
        )
    cmd.extend(
        [
            "--numerics-mode",
            "strict",
            # This gate compares bit-identity across thread counts and
            # architectures, not agreement with MESA. The upstream case sets
            # controls that are not ported yet (declared gap), so the
            # documented exploratory opt-out is used here.
            "--allow-unsupported-controls",
        ]
    )
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    return cmd


def _run_strict_evolution(
    suite: str,
    case: str,
    max_steps: int | None,
    threads: int,
) -> tuple[str, int, dict[str, dict[str, Any]], str, int]:
    """Run ``flexmesa trace`` in an isolated subprocess with ``threads``.

    Returns:
        ``(combined_sha256, total_bytes, per_file_digests, termination_code,
        steps)``. The last two are parsed from the trace summary line.
    """
    with tempfile.TemporaryDirectory(
        prefix=f"flexmesa_det_{suite}_{threads}t_"
    ) as tmp:
        out_dir = pathlib.Path(tmp) / "run"
        env = dict(os.environ)
        t_str = str(threads)
        env["RAYON_NUM_THREADS"] = t_str
        env["OMP_NUM_THREADS"] = t_str
        env["OPENBLAS_NUM_THREADS"] = t_str
        env["MKL_NUM_THREADS"] = t_str
        env["VECLIB_MAXIMUM_THREADS"] = t_str
        env["FLEXMESA_NUMERICS_MODE"] = "strict"
        cmd = _build_trace_command(suite, case, out_dir, max_steps)
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            code = proc.returncode
            raise RuntimeError(
                f"strict trace ({suite}/{case}, {threads}t) failed ({code}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        files = _collect_history_digests(out_dir, suite)
        combined_sha, total_bytes = _combined_files_digest(files)
        term, steps = _parse_trace_summary(proc.stdout)
        if steps <= 0:
            raise RuntimeError(
                f"strict trace ({suite}/{case}, {threads}t) took no steps "
                f"(term={term!r}); two empty runs would compare equal:\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        return combined_sha, total_bytes, files, term, steps


def _parse_trace_summary(stdout: str) -> tuple[str, int]:
    """Read ``term=<code>`` and ``steps=<n>`` from the trace summary line.

    Raises:
        RuntimeError: If no summary line carrying both fields is present.
    """
    for line in reversed(stdout.splitlines()):
        m_term = re.search(r"\bterm=(\S+)", line)
        m_steps = re.search(r"\bsteps=(\d+)", line)
        if m_term and m_steps:
            return m_term.group(1), int(m_steps.group(1))
    raise RuntimeError(f"no trace summary line in output:\n{stdout}")


def _measure_case(
    suite: str, case: str, max_steps: int | None
) -> dict[str, Any]:
    """Run 1t vs 4t strict trace for one case and assert byte-identity."""
    sha_1t, bytes_1t, files_1t, term_1t, steps_1t = _run_strict_evolution(
        suite, case, max_steps, threads=1
    )
    sha_4t, bytes_4t, files_4t, term_4t, steps_4t = _run_strict_evolution(
        suite, case, max_steps, threads=4
    )
    identical = (
        sha_1t == sha_4t
        and bytes_1t == bytes_4t
        and files_1t == files_4t
        and term_1t == term_4t
        and steps_1t == steps_4t
    )
    if not identical:
        raise RuntimeError(
            f"1-thread vs 4-thread history mismatch on {suite}/{case}: "
            f"1t={sha_1t} ({bytes_1t} B, {steps_1t} steps, {term_1t}) vs "
            f"4t={sha_4t} ({bytes_4t} B, {steps_4t} steps, {term_4t})"
        )
    return {
        "suite": suite,
        "case": case,
        "max_steps": max_steps,
        "steps": steps_1t,
        "termination_code": term_1t,
        "unsupported_controls_ignored": True,
        "history_bytes": bytes_1t,
        "history_1t_sha256": sha_1t,
        "history_4t_sha256": sha_4t,
        "history_files": files_1t,
        "thread_1_vs_4_byte_identical": identical,
    }


def _measure(
    vectors_path: pathlib.Path,
    cases: Sequence[tuple[str, str]],
    max_steps: int | None,
) -> dict[str, Any]:
    """Compute strict reference-vector and 1-vs-4-thread evolution digests."""
    if not cases:
        raise ValueError("at least one (suite, case) pair is required")
    from flexmesa import _rust

    vec_count, vec_sha = _rust.strict_math_vectors_sha256(str(vectors_path))
    case_records = [
        _measure_case(suite, case, max_steps) for suite, case in cases
    ]
    primary = case_records[0]
    all_identical = all(
        bool(rec["thread_1_vs_4_byte_identical"]) for rec in case_records
    )
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "numerics_mode": "strict",
        "unsupported_controls_ignored": True,
        "math_vector_count": vec_count,
        "math_vectors_sha256": vec_sha,
        "evolution_suite": primary["suite"],
        "evolution_case": primary["case"],
        "evolution_max_steps": primary["max_steps"],
        "history_bytes": primary["history_bytes"],
        "history_1t_sha256": primary["history_1t_sha256"],
        "history_4t_sha256": primary["history_4t_sha256"],
        "thread_1_vs_4_byte_identical": all_identical,
        "cases": case_records,
    }


def _normalize_arch(raw: object) -> str:
    """Normalize OS/Python machine architecture strings."""
    if not isinstance(raw, str):
        return ""
    low = raw.strip().lower()
    if low in ("x86_64", "amd64"):
        return "x86_64"
    if low in ("aarch64", "arm64"):
        return "aarch64"
    return low


def _is_sha256_hex(val: object) -> bool:
    return (
        isinstance(val, str)
        and len(val) == 64
        and all(ch in "0123456789abcdef" for ch in val)
    )


def _validate_digest_pair(rec: dict[str, Any], label: str) -> list[str]:
    """Validate byte size and 1t/4t SHA-256 identity on a record."""
    errors: list[str] = []
    hb = rec.get("history_bytes")
    if not isinstance(hb, int) or isinstance(hb, bool) or hb <= 0:
        errors.append(f"{label}: history_bytes must be > 0, got {hb!r}")
    h1 = rec.get("history_1t_sha256")
    h4 = rec.get("history_4t_sha256")
    if not _is_sha256_hex(h1) or h1 != h4:
        errors.append(
            f"{label}: history_1t_sha256 ({h1!r}) != history_4t_sha256 ({h4!r})"
        )
    if rec.get("thread_1_vs_4_byte_identical") is not True:
        errors.append(f"{label}: thread_1_vs_4_byte_identical != True")
    if "steps" in rec:
        steps = rec.get("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            errors.append(f"{label}: steps must be > 0, got {steps!r}")
    return errors


def _validate_record(rec: dict[str, Any], label: str) -> list[str]:
    """Validate schema, strict mode, vector count, and digests on one JSON."""
    errors: list[str] = []
    if rec.get("numerics_mode") != "strict":
        errors.append(
            f"{label}: numerics_mode must be 'strict', got "
            f"{rec.get('numerics_mode')!r}"
        )
    vec_cnt = rec.get("math_vector_count")
    if (
        not isinstance(vec_cnt, int)
        or isinstance(vec_cnt, bool)
        or vec_cnt < 470_000
    ):
        errors.append(
            f"{label}: math_vector_count must be >= 470000, got {vec_cnt!r}"
        )
    if not _is_sha256_hex(rec.get("math_vectors_sha256")):
        errors.append(f"{label}: invalid math_vectors_sha256")

    raw_cases = rec.get("cases")
    if isinstance(raw_cases, list):
        if not raw_cases:
            errors.append(f"{label}: 'cases' must not be empty")
        if rec.get("thread_1_vs_4_byte_identical") is not True:
            errors.append(f"{label}: thread_1_vs_4_byte_identical != True")
        for idx, item in enumerate(raw_cases):
            c_lbl = f"{label}:cases[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{c_lbl}: must be an object")
            else:
                errors.extend(_validate_digest_pair(item, c_lbl))
    else:
        errors.extend(_validate_digest_pair(rec, label))
    return errors


def _check_cross_record_equality(
    paths: Sequence[pathlib.Path], records: Sequence[dict[str, Any]]
) -> bool:
    """Compare all determinism digest fields across validated JSON records."""
    base = records[0]
    keys = [
        "numerics_mode",
        "math_vector_count",
        "math_vectors_sha256",
        "evolution_suite",
        "evolution_case",
        "evolution_max_steps",
        "history_bytes",
        "history_1t_sha256",
        "history_4t_sha256",
        "thread_1_vs_4_byte_identical",
    ]
    if "cases" in base or any("cases" in rec for rec in records[1:]):
        keys.append("cases")
    ok = True
    for idx, rec in enumerate(records[1:], start=1):
        for key in keys:
            if rec.get(key) != base.get(key):
                sys.stderr.write(
                    f"MISMATCH on `{key}` between {paths[0].name} "
                    f"({base.get('os')}/{base.get('arch')}: {base.get(key)!r}) "
                    f"and {paths[idx].name} "
                    f"({rec.get('os')}/{rec.get('arch')}: {rec.get(key)!r})\n"
                )
                ok = False
    return ok


def _compare_jsons(
    paths: list[pathlib.Path],
    *,
    allow_same_arch: bool = False,
) -> int:
    """Compare two or more runner JSON digests and fail on any bit mismatch."""
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        sys.stderr.write(
            "ERROR: duplicate JSON paths passed to --compare: "
            f"{[str(p) for p in paths]}\n"
        )
        return 1

    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    ok = True
    for p, rec in zip(paths, records, strict=True):
        for err in _validate_record(rec, p.name):
            sys.stderr.write(f"INVALID RECORD: {err}\n")
            ok = False

    norm_arches = {_normalize_arch(r.get("arch")) for r in records}
    if not allow_same_arch and not {"x86_64", "aarch64"}.issubset(norm_arches):
        sys.stderr.write(
            "ERROR: cross-architecture check requires both x86_64 and "
            f"aarch64 runner records, got {sorted(norm_arches)!r}\n"
        )
        ok = False

    if not _check_cross_record_equality(paths, records) or not ok:
        return 1
    base = records[0]
    runners = ", ".join(f"{r.get('os')}/{r.get('arch')}" for r in records)
    case_desc = (
        f"{len(base['cases'])} case(s)"
        if isinstance(base.get("cases"), list)
        else f"{base['evolution_suite']}/{base['evolution_case']}"
    )
    sys.stdout.write(
        f"Strict cross-machine bit-identity confirmed across [{runners}]:\n"
        f"  math_vectors ({base['math_vector_count']}): "
        f"{base['math_vectors_sha256']}\n"
        f"  {case_desc} ({base['evolution_max_steps'] or 'uncapped'}, 1t==4t): "
        f"{base['history_1t_sha256']}\n"
        "  (runs ignore unsupported upstream controls; this checks "
        "bit-identity, not agreement with MESA)\n"
    )
    return 0


def _parse_cases_arg(raw_cases: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in raw_cases:
        if "/" not in item:
            raise ValueError(
                f"Expected '<suite>/<case>' format for --cases, got {item!r}"
            )
        suite, case = item.split("/", maxsplit=1)
        parsed.append((suite.strip(), case.strip()))
    return parsed


def _resolve_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.cases is not None:
        return _parse_cases_arg(args.cases)
    if args.suite is not None or args.case is not None:
        return [
            (args.suite or _DEFAULT_SUITE, args.case or _DEFAULT_CASE),
        ]
    return list(_DEFAULT_CASES)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for strict determinism verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vectors",
        type=pathlib.Path,
        default=_DEFAULT_VECTORS,
        help="Path to MESA 26.4.1 hexadecimal math reference vectors.",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Single-case override suite (defaults to star + binary pair).",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Single-case override name.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Explicit '<suite>/<case>' pairs (defaults to star + binary).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_DEFAULT_STEPS,
        help="Optional diagnostic step cap (<= 0 or omitted runs uncapped).",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write JSON determinism summary.",
    )
    parser.add_argument(
        "--compare",
        type=pathlib.Path,
        nargs="+",
        default=None,
        help="Compare two or more JSON determinism summaries and exit.",
    )
    parser.add_argument(
        "--allow-same-arch",
        action="store_true",
        help="Allow --compare on two JSON records from the same architecture.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for cross-machine and thread determinism checks."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.compare is not None:
        if len(args.compare) < 2:
            parser.error("--compare requires at least two JSON files")
        return _compare_jsons(
            list(args.compare),
            allow_same_arch=bool(args.allow_same_arch),
        )

    max_steps = (
        None
        if (args.max_steps is None or int(args.max_steps) <= 0)
        else int(args.max_steps)
    )
    summary = _measure(args.vectors, _resolve_cases(args), max_steps)
    payload = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
