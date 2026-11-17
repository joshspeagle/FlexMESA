#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify `NumericsMode::Strict` cross-machine and thread-count bit-identity.

Implements Items 15 and 16 of `#165`:
1. Evaluates all `474,062` MESA `26.4.1` reference inputs in
   `benchmarks/fortran_mesa/math_hex_vectors/mesa_26_4_1_math_vectors.txt`
   under `NumericsMode::Strict` and records the canonicalized `SHA-256` digest.
2. Runs a short `strict` stellar evolution (`star/1.3M_ms_high_Z`, 5 steps) in
   two isolated subprocesses with `1` thread and `4` threads
   (`RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS`), verifying that `LOGS/history.data` records
   `math_backend = "strict"` and is byte-for-byte identical between `1` and
   `4` threads (`SHA-256`).
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
import subprocess
import sys
import tempfile
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
_DEFAULT_STEPS = 5


def _sha256_file(path: pathlib.Path) -> str:
    """Return lowercase 64-char SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_strict_evolution(
    suite: str,
    case: str,
    max_steps: int,
    threads: int,
) -> tuple[str, int]:
    """Run ``flexmesa trace`` in an isolated subprocess with ``threads``."""
    with tempfile.TemporaryDirectory(prefix=f"flexmesa_det_{threads}t_") as tmp:
        out_dir = pathlib.Path(tmp) / "run"
        env = dict(os.environ)
        t_str = str(threads)
        env["RAYON_NUM_THREADS"] = t_str
        env["OMP_NUM_THREADS"] = t_str
        env["OPENBLAS_NUM_THREADS"] = t_str
        env["MKL_NUM_THREADS"] = t_str
        env["VECLIB_MAXIMUM_THREADS"] = t_str
        env["FLEXMESA_NUMERICS_MODE"] = "strict"
        cmd = [
            sys.executable,
            "-m",
            "flexmesa.cli",
            "trace",
            suite,
            case,
            "--out",
            str(out_dir),
            "--max-steps",
            str(max_steps),
            "--numerics-mode",
            "strict",
        ]
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
                f"strict trace ({threads}t) failed ({code}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        hist_path = out_dir / "LOGS" / "history.data"
        if not hist_path.is_file():
            raise RuntimeError(f"missing history output: {hist_path}")
        text = hist_path.read_text(encoding="utf-8")
        if '"strict"' not in text.splitlines()[2]:
            raise RuntimeError(
                f'history.data header missing math_backend="strict": '
                f"{text.splitlines()[:4]!r}"
            )
        return _sha256_file(hist_path), len(hist_path.read_bytes())


def _measure(
    vectors_path: pathlib.Path,
    suite: str,
    case: str,
    max_steps: int,
) -> dict[str, Any]:
    """Compute strict reference-vector and 1-vs-4-thread evolution digests."""
    from flexmesa import _rust

    vec_count, vec_sha = _rust.strict_math_vectors_sha256(str(vectors_path))
    sha_1t, bytes_1t = _run_strict_evolution(suite, case, max_steps, threads=1)
    sha_4t, bytes_4t = _run_strict_evolution(suite, case, max_steps, threads=4)
    identical = sha_1t == sha_4t and bytes_1t == bytes_4t
    if not identical:
        raise RuntimeError(
            f"1-thread vs 4-thread history.data mismatch on {suite}/{case}: "
            f"1t={sha_1t} ({bytes_1t} B) vs 4t={sha_4t} ({bytes_4t} B)"
        )
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "numerics_mode": "strict",
        "math_vector_count": vec_count,
        "math_vectors_sha256": vec_sha,
        "evolution_suite": suite,
        "evolution_case": case,
        "evolution_max_steps": max_steps,
        "history_bytes": bytes_1t,
        "history_1t_sha256": sha_1t,
        "history_4t_sha256": sha_4t,
        "thread_1_vs_4_byte_identical": identical,
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


def _validate_record(rec: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if rec.get("numerics_mode") != "strict":
        errors.append(
            f"{label}: numerics_mode must be 'strict', got "
            f"{rec.get('numerics_mode')!r}"
        )
    if rec.get("thread_1_vs_4_byte_identical") is not True:
        errors.append(
            f"{label}: thread_1_vs_4_byte_identical must be True, got "
            f"{rec.get('thread_1_vs_4_byte_identical')!r}"
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
    hist_bytes = rec.get("history_bytes")
    if (
        not isinstance(hist_bytes, int)
        or isinstance(hist_bytes, bool)
        or hist_bytes <= 0
    ):
        errors.append(f"{label}: history_bytes must be > 0, got {hist_bytes!r}")
    if not _is_sha256_hex(rec.get("math_vectors_sha256")):
        errors.append(f"{label}: invalid math_vectors_sha256")
    h1 = rec.get("history_1t_sha256")
    h4 = rec.get("history_4t_sha256")
    if not _is_sha256_hex(h1) or h1 != h4:
        errors.append(
            f"{label}: history_1t_sha256 ({h1!r}) != history_4t_sha256 ({h4!r})"
        )
    return errors


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

    base = records[0]
    keys = (
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
    )
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
    if not ok:
        return 1
    runners = ", ".join(f"{r.get('os')}/{r.get('arch')}" for r in records)
    sys.stdout.write(
        f"Strict cross-machine bit-identity confirmed across [{runners}]:\n"
        f"  math_vectors ({base['math_vector_count']}): "
        f"{base['math_vectors_sha256']}\n"
        f"  {base['evolution_suite']}/{base['evolution_case']} "
        f"({base['evolution_max_steps']} steps, 1t==4t): "
        f"{base['history_1t_sha256']}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for cross-machine and thread determinism checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vectors",
        type=pathlib.Path,
        default=_DEFAULT_VECTORS,
        help="Path to MESA 26.4.1 hexadecimal math reference vectors.",
    )
    parser.add_argument("--suite", default=_DEFAULT_SUITE)
    parser.add_argument("--case", default=_DEFAULT_CASE)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_STEPS)
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
    args = parser.parse_args(argv)

    if args.compare is not None:
        if len(args.compare) < 2:
            parser.error("--compare requires at least two JSON files")
        return _compare_jsons(
            list(args.compare),
            allow_same_arch=bool(args.allow_same_arch),
        )

    summary = _measure(args.vectors, args.suite, args.case, args.max_steps)
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
