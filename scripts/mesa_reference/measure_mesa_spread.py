#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Measure MESA-to-MESA numerical spread across compiler/flag configurations.

Compiles or invokes an alternative upstream MESA ``26.4.1`` build (for example
``-O0 -fno-tree-vectorize -ffp-contract=off`` or alternative math slots) using
an installed ``$MESASDK_ROOT`` / ``$MESA_DIR``, compares double-precision
``history.data`` / ``binary_history.data`` outputs in exact IEEE-754 ULPs and
relative differences, and emits structured JSON/Markdown spread reports
matching the ``benchmarks/history_harness/mesa_spread_26_4_1.json`` schema.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
DEFAULT_MESA_VERSION = "26.4.1"
DEFAULT_BASELINE_FLAGS = "-O2 -ftree-vectorize"
DEFAULT_VARIANT_FLAGS = "-O0 -fno-tree-vectorize -ffp-contract=off"
DEFAULT_MATH_SLOT = "crmath"
DEFAULT_CASES: tuple[tuple[str, str], ...] = (
    ("star", "1.3M_ms_high_Z"),
    ("binary", "jdot_gr_check"),
)

_FORTRAN_D_EXP_RE = re.compile(r"([0-9])[dD]([+-]?[0-9]+)")
_FORTRAN_SHORT_EXP_RE = re.compile(r"([0-9])([+-][0-9]{3})\b")
_SIGN_BIT_U64 = np.uint64(1 << 63)
_MAG_MASK_U64 = np.uint64(0x7FFF_FFFF_FFFF_FFFF)


@dataclasses.dataclass(frozen=True)
class CompilerConfigSpec:
    """Compiler flags and MESA math-slot specification for one build."""

    name: str
    fc_flags: str
    math_slot: str = DEFAULT_MATH_SLOT
    omp_num_threads: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration metadata."""
        return {
            "name": self.name,
            "fc_flags": self.fc_flags,
            "math_slot": self.math_slot,
            "omp_num_threads": self.omp_num_threads,
        }


@dataclasses.dataclass(frozen=True)
class ColumnSpread:
    """Spread statistics for one aligned numerical column."""

    column: str
    rows_compared: int
    zero_ulp_rows: int
    max_ulp: int
    max_rel_diff: float
    max_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize column spread record."""
        return {
            "column": self.column,
            "rows_compared": self.rows_compared,
            "zero_ulp_rows": self.zero_ulp_rows,
            "max_ulp": self.max_ulp,
            "max_rel_diff": self.max_rel_diff,
            "max_abs_diff": self.max_abs_diff,
        }


@dataclasses.dataclass(frozen=True)
class CaseSpreadResult:
    """Double-precision history spread summary for one MESA case."""

    suite: str
    case: str
    rows_baseline: int
    rows_variant: int
    rows_compared: int
    columns_compared: int
    total_cells_compared: int
    zero_ulp_cells: int
    max_ulp: int
    mean_ulp: float
    max_rel_diff: float
    max_abs_diff: float
    files_byte_identical: bool
    baseline_sha256: str
    variant_sha256: str
    top_columns_by_ulp: tuple[ColumnSpread, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize case spread metrics."""
        return {
            "suite": self.suite,
            "case": self.case,
            "rows_baseline": self.rows_baseline,
            "rows_variant": self.rows_variant,
            "rows_compared": self.rows_compared,
            "columns_compared": self.columns_compared,
            "total_cells_compared": self.total_cells_compared,
            "zero_ulp_cells": self.zero_ulp_cells,
            "max_ulp": self.max_ulp,
            "mean_ulp": self.mean_ulp,
            "max_rel_diff": self.max_rel_diff,
            "max_abs_diff": self.max_abs_diff,
            "files_byte_identical": self.files_byte_identical,
            "baseline_sha256": self.baseline_sha256,
            "variant_sha256": self.variant_sha256,
            "top_columns_by_ulp": [
                c.to_dict() for c in self.top_columns_by_ulp
            ],
        }


def _to_signed_magnitude(bits: int) -> int:
    sign_bit = 1 << 63
    if bits & sign_bit:
        return -(bits ^ sign_bit)
    return bits


def ulp_distance_f64(val_a: float, val_b: float) -> int:
    """Return IEEE-754 double-precision ULP distance between two floats."""
    fa = float(val_a)
    fb = float(val_b)
    if math.isnan(fa) or math.isnan(fb):
        return 0 if (math.isnan(fa) and math.isnan(fb)) else (1 << 63)
    if math.isinf(fa) or math.isinf(fb):
        return 0 if fa == fb else (1 << 63)
    ba = struct.unpack(">Q", struct.pack(">d", fa))[0]
    bb = struct.unpack(">Q", struct.pack(">d", fb))[0]
    return abs(_to_signed_magnitude(ba) - _to_signed_magnitude(bb))


def _ordered_u64(bits: np.ndarray) -> np.ndarray:
    """Map IEEE-754 float64 bit patterns onto monotonic unsigned order."""
    mag = bits & _MAG_MASK_U64
    neg = (bits & _SIGN_BIT_U64) != 0
    return np.where(neg, _SIGN_BIT_U64 - mag, _SIGN_BIT_U64 + mag)


def _vectorized_ulp_distances(
    arr_a: np.ndarray, arr_b: np.ndarray
) -> np.ndarray:
    """Compute IEEE-754 float64 ULP distances across two 1D float64 arrays."""
    a_f = np.asarray(arr_a, dtype=np.float64)
    b_f = np.asarray(arr_b, dtype=np.float64)
    bits_a = a_f.astype("<f8", copy=False).view(np.uint64)
    bits_b = b_f.astype("<f8", copy=False).view(np.uint64)
    ord_a = _ordered_u64(bits_a)
    ord_b = _ordered_u64(bits_b)
    finite_ulp = np.where(ord_a >= ord_b, ord_a - ord_b, ord_b - ord_a)
    both_finite = np.isfinite(a_f) & np.isfinite(b_f)
    same_nonfinite = (np.isnan(a_f) & np.isnan(b_f)) | (
        np.isinf(a_f) & (a_f == b_f)
    )
    return np.where(
        both_finite,
        finite_ulp,
        np.where(same_nonfinite, np.uint64(0), _SIGN_BIT_U64),
    )


def _parse_fortran_float(token: str) -> float | None:
    """Parse standard or Fortran ``D``/3-digit exponent floating tokens."""
    tok = token.strip()
    if not tok or tok.startswith('"'):
        return None
    norm = _FORTRAN_D_EXP_RE.sub(r"\1e\2", tok)
    norm = _FORTRAN_SHORT_EXP_RE.sub(r"\1e\2", norm)
    try:
        return float(norm)
    except ValueError:
        return None


def parse_mesa_history_table(
    path: pathlib.Path,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Parse numeric columns from a 6-line-header MESA history file.

    Args:
        path: Path to ``LOGS/history.data`` or ``binary_history.data``.

    Returns:
        Tuple ``(column_names, column_arrays)`` containing ``float64`` 1D
        arrays for every purely numeric column.

    Raises:
        ValueError: If ``path`` does not contain a valid 6-line MESA header.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(
            f"Expected at least 7 lines in MESA history file {path}, "
            f"got {len(lines)}"
        )
    header_cols = lines[5].split()
    if not header_cols:
        raise ValueError(f"Empty column header at line 6 of {path}")

    n_cols = len(header_cols)
    raw_rows = [
        toks
        for raw in lines[6:]
        if (stripped := raw.strip()) and len(toks := stripped.split()) == n_cols
    ]
    if not raw_rows:
        raise ValueError(f"No data rows found in MESA history file {path}")

    numeric_names: list[str] = []
    numeric_cols: dict[str, np.ndarray] = {}
    for col_idx, col_name in enumerate(header_cols):
        parsed = [
            val
            for row in raw_rows
            if (val := _parse_fortran_float(row[col_idx])) is not None
        ]
        if len(parsed) == len(raw_rows):
            numeric_names.append(col_name)
            numeric_cols[col_name] = np.asarray(parsed, dtype=np.float64)
    return numeric_names, numeric_cols


def _compare_column_arrays(
    col: str, arr_a: np.ndarray, arr_b: np.ndarray
) -> tuple[ColumnSpread, int]:
    """Return :class:`ColumnSpread` and total ULP sum for one aligned column."""
    ulps = _vectorized_ulp_distances(arr_a, arr_b)
    rows_cmp = int(arr_a.shape[0])
    zero_rows = int(np.count_nonzero(ulps == 0))
    max_ulp = int(ulps.max()) if rows_cmp else 0
    ulp_sum = sum(int(u) for u in ulps.tolist())

    both_finite = np.isfinite(arr_a) & np.isfinite(arr_b)
    mismatch_nonfinite = (~both_finite) & (ulps > 0)
    if np.any(mismatch_nonfinite):
        return (
            ColumnSpread(
                column=col,
                rows_compared=rows_cmp,
                zero_ulp_rows=zero_rows,
                max_ulp=max_ulp,
                max_rel_diff=float("inf"),
                max_abs_diff=float("inf"),
            ),
            ulp_sum,
        )

    safe_a = np.where(both_finite, arr_a, 0.0)
    safe_b = np.where(both_finite, arr_b, 0.0)
    abs_d = np.abs(safe_a - safe_b)
    denom = np.maximum(np.maximum(np.abs(safe_a), np.abs(safe_b)), 1e-300)
    max_abs = float(abs_d.max()) if rows_cmp else 0.0
    max_rel = float((abs_d / denom).max()) if rows_cmp else 0.0
    return (
        ColumnSpread(
            column=col,
            rows_compared=rows_cmp,
            zero_ulp_rows=zero_rows,
            max_ulp=max_ulp,
            max_rel_diff=max_rel,
            max_abs_diff=max_abs,
        ),
        ulp_sum,
    )


def compare_history_tables(
    suite: str,
    case: str,
    baseline_path: pathlib.Path,
    variant_path: pathlib.Path,
    *,
    top_n_columns: int = 10,
) -> CaseSpreadResult:
    """Quantify IEEE-754 ULP and relative spread between two MESA runs."""
    base_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    var_sha = hashlib.sha256(variant_path.read_bytes()).hexdigest()

    base_names, base_cols = parse_mesa_history_table(baseline_path)
    var_names, var_cols = parse_mesa_history_table(variant_path)
    shared_cols = [c for c in base_names if c in set(var_names)]
    if not shared_cols:
        raise ValueError(
            f"No shared numeric columns between {baseline_path} and "
            f"{variant_path}"
        )

    rows_base = len(base_cols[shared_cols[0]])
    rows_var = len(var_cols[shared_cols[0]])
    rows_cmp = min(rows_base, rows_var)

    col_spreads: list[ColumnSpread] = []
    ulp_sum = 0
    for col in shared_cols:
        spread, col_sum = _compare_column_arrays(
            col, base_cols[col][:rows_cmp], var_cols[col][:rows_cmp]
        )
        col_spreads.append(spread)
        ulp_sum += col_sum

    total_cells = rows_cmp * len(shared_cols)
    zero_cells = sum(c.zero_ulp_rows for c in col_spreads)
    max_ulp_all = max((c.max_ulp for c in col_spreads), default=0)
    max_rel_all = max((c.max_rel_diff for c in col_spreads), default=0.0)
    max_abs_all = max((c.max_abs_diff for c in col_spreads), default=0.0)
    ranked = tuple(
        sorted(
            col_spreads,
            key=lambda c: (-c.max_ulp, -c.max_rel_diff, c.column),
        )[:top_n_columns]
    )
    return CaseSpreadResult(
        suite=suite,
        case=case,
        rows_baseline=rows_base,
        rows_variant=rows_var,
        rows_compared=rows_cmp,
        columns_compared=len(shared_cols),
        total_cells_compared=total_cells,
        zero_ulp_cells=zero_cells,
        max_ulp=max_ulp_all,
        mean_ulp=(ulp_sum / total_cells) if total_cells else 0.0,
        max_rel_diff=max_rel_all,
        max_abs_diff=max_abs_all,
        files_byte_identical=(base_sha == var_sha),
        baseline_sha256=base_sha,
        variant_sha256=var_sha,
        top_columns_by_ulp=ranked,
    )


def build_spread_report(
    baseline_cfg: CompilerConfigSpec,
    variant_cfg: CompilerConfigSpec,
    cases: Sequence[CaseSpreadResult],
    *,
    dry_run: bool,
    planned_commands: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble structured MESA-to-MESA spread report JSON."""
    identical_count = sum(1 for c in cases if c.files_byte_identical)
    row_mismatches = sum(1 for c in cases if c.rows_baseline != c.rows_variant)
    max_ulp = max((c.max_ulp for c in cases), default=0)
    max_rel = max((c.max_rel_diff for c in cases), default=0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "mesa_version": DEFAULT_MESA_VERSION,
        "dry_run": dry_run,
        "baseline_config": baseline_cfg.to_dict(),
        "variant_config": variant_cfg.to_dict(),
        "planned_commands": list(planned_commands),
        "cases": [c.to_dict() for c in cases],
        "summary": {
            "cases_compared": len(cases),
            "byte_identical_cases": identical_count,
            "row_count_mismatches": row_mismatches,
            "max_ulp_across_cases": max_ulp,
            "max_rel_diff_across_cases": max_rel,
        },
    }


def render_spread_markdown(report: dict[str, Any]) -> str:
    """Format a human-readable Markdown summary of MESA-to-MESA spread."""
    base = report["baseline_config"]
    var = report["variant_config"]
    summ = report["summary"]
    lines = [
        "# MESA 26.4.1 Compiler-Variant Numerical Spread",
        "",
        f"- **Baseline**: `{base['name']}` (`{base['fc_flags']}`, "
        f"math_slot=`{base['math_slot']}`)",
        f"- **Variant**: `{var['name']}` (`{var['fc_flags']}`, "
        f"math_slot=`{var['math_slot']}`)",
        f"- **Dry run**: `{report['dry_run']}`",
        f"- **Cases compared**: `{summ['cases_compared']}` "
        f"(`{summ['byte_identical_cases']}` byte-identical, "
        f"`{summ.get('row_count_mismatches', 0)}` row-count mismatch)",
        f"- **Max ULP / Max rel diff**: `{summ['max_ulp_across_cases']}` / "
        f"`{summ['max_rel_diff_across_cases']:.3e}`",
    ]
    if report["cases"]:
        lines.extend(
            [
                "",
                "| Case | Rows | Cells | 0 ULP | Max ULP | Max Rel Diff "
                "| Identical |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        for c in report["cases"]:
            rows_str = (
                str(c["rows_compared"])
                if c["rows_baseline"] == c["rows_variant"]
                else f"{c['rows_compared']} (base={c['rows_baseline']}, "
                f"var={c['rows_variant']})"
            )
            lines.append(
                f"| `{c['suite']}/{c['case']}` | {rows_str} | "
                f"{c['total_cells_compared']} | {c['zero_ulp_cells']} | "
                f"{c['max_ulp']} | {c['max_rel_diff']:.3e} | "
                f"`{c['files_byte_identical']}` |"
            )
    if report["planned_commands"]:
        lines.extend(["", "## Planned Commands", ""])
        lines.extend(f"- `{cmd}`" for cmd in report["planned_commands"])
    return "\n".join(lines) + "\n"


def _patch_makefile_header(
    header_path: pathlib.Path, variant_fc_flags: str
) -> None:
    """Update ``FCopt`` and disable PGSTAR inside a copied MESA worktree."""
    lines = header_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for ln in lines:
        if ln.startswith("USE_PGSTAR ="):
            out.append("USE_PGSTAR = NO")
        elif ln.startswith("FCopt ="):
            out.append(f"FCopt = {variant_fc_flags}")
        else:
            out.append(ln)
    header_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def plan_alt_mesa_commands(
    mesa_dir: pathlib.Path,
    mesasdk_root: pathlib.Path,
    alt_mesa_dir: pathlib.Path,
    variant: CompilerConfigSpec,
    cases: Sequence[tuple[str, str]],
) -> list[str]:
    """Return ordered shell steps to build and run the variant MESA tree."""
    cmds = [
        f"copy {mesa_dir} -> {alt_mesa_dir} && patch utils/makefile_header "
        f"(USE_PGSTAR=NO, FCopt={variant.fc_flags!r})",
        f"source {mesasdk_root}/bin/mesasdk_init.sh && "
        f"MESASDK_MATH_SLOT={variant.math_slot} MESA_DIR={alt_mesa_dir} "
        f"OMP_NUM_THREADS={variant.omp_num_threads} ./touch && ./install",
    ]
    for suite, case in cases:
        cmds.append(
            f"run {suite}/test_suite/{case} in baseline ({mesa_dir}) and "
            f"variant ({alt_mesa_dir}) with OMP_NUM_THREADS="
            f"{variant.omp_num_threads}"
        )
    return cmds


def _mesa_subprocess_env(
    mesa_root: pathlib.Path,
    mesasdk_root: pathlib.Path,
    math_slot: str,
    omp_threads: int,
) -> dict[str, str]:
    """Build sanitized headless environment for MESA build/run steps."""
    env = os.environ.copy()
    sysdeps_raw = env.get("MESA_SYSDEPS_DIR", "").strip()
    sysdeps = (
        pathlib.Path(sysdeps_raw)
        if sysdeps_raw
        else pathlib.Path.home() / "mesasdk-sysdeps"
    )
    if sysdeps.is_dir():
        cur_path = env.get("PATH", "")
        env["PATH"] = f"{sysdeps}/bin:{sysdeps}/usr/bin" + (
            f":{cur_path}" if cur_path else ""
        )
        cur_lib = env.get("LIBRARY_PATH", "")
        env["LIBRARY_PATH"] = f"{sysdeps}/usr/lib/x86_64-linux-gnu" + (
            f":{cur_lib}" if cur_lib else ""
        )
        cur_cpath = env.get("CPATH", "")
        env["CPATH"] = (
            f"{sysdeps}/usr/include:{sysdeps}/usr/include/x86_64-linux-gnu"
            + (f":{cur_cpath}" if cur_cpath else "")
        )
    env["MESA_DIR"] = str(mesa_root)
    env["MESASDK_ROOT"] = str(mesasdk_root)
    env["MESASDK_MATH_SLOT"] = math_slot
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    env["MESA_SKIP_OPTIONAL"] = "t"
    env.pop("DISPLAY", None)
    return env


def _install_variant_mesa(
    mesa_dir: pathlib.Path,
    alt_dir: pathlib.Path,
    sdk_root: pathlib.Path,
    var_cfg: CompilerConfigSpec,
) -> None:
    """Clone baseline MESA tree, patch flags, and rebuild cleanly."""
    if not mesa_dir.is_dir():
        raise FileNotFoundError(f"MESA_DIR not found: {mesa_dir}")
    if not sdk_root.is_dir():
        raise FileNotFoundError(f"MESASDK_ROOT not found: {sdk_root}")
    if alt_dir.exists():
        return
    shutil.copytree(mesa_dir, alt_dir, symlinks=True)
    _patch_makefile_header(
        alt_dir / "utils" / "makefile_header", var_cfg.fc_flags
    )
    install_script = (
        "set -eu; set +u; "
        'source "$MESASDK_ROOT/bin/mesasdk_init.sh"; set -u; '
        "./touch && ./install"
    )
    env = _mesa_subprocess_env(
        alt_dir, sdk_root, var_cfg.math_slot, var_cfg.omp_num_threads
    )
    env["NPROCS"] = "4"
    subprocess.run(
        ["bash", "-c", install_script], cwd=str(alt_dir), env=env, check=True
    )


def _run_mesa_case_in_tree(
    mesa_root: pathlib.Path,
    mesasdk_root: pathlib.Path,
    suite: str,
    case: str,
    work_dir: pathlib.Path,
    cfg: CompilerConfigSpec,
) -> pathlib.Path:
    """Compile and execute one upstream MESA test_suite case outside repo."""
    case_src = mesa_root / suite / "test_suite" / case
    if not case_src.is_dir():
        raise FileNotFoundError(f"MESA test_suite case missing: {case_src}")
    dest = work_dir / f"{suite}_{case}"
    shutil.copytree(case_src, dest)
    env = _mesa_subprocess_env(
        mesa_root, mesasdk_root, cfg.math_slot, cfg.omp_num_threads
    )
    script = (
        'set -eu; set +u; source "$MESASDK_ROOT/bin/mesasdk_init.sh"; '
        "set -u; "
        "if [ -f Makefile ] || [ -f makefile ]; then "
        "make clean || true; "
        'make MESA_DIR="$MESA_DIR" WITH_PGSTAR=no; '
        "else make -C make clean || true; "
        'make -C make MESA_DIR="$MESA_DIR"; fi; '
        "./rn"
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(dest),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"MESA run failed for {suite}/{case} under {mesa_root}:\n"
            f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
        )
    hist_name = (
        "binary_history.data" if suite == "binary" else "LOGS/history.data"
    )
    hist_path = dest / hist_name
    if not hist_path.is_file():
        raise FileNotFoundError(f"Expected MESA output missing: {hist_path}")
    return hist_path


def _parse_suite_case(raw: str) -> tuple[str, str]:
    if "/" not in raw:
        raise ValueError(f"Invalid case spec {raw!r}; expected suite/case")
    suite, case = raw.split("/", maxsplit=1)
    return (suite.strip(), case.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify MESA-to-MESA double-precision spread."
    )
    parser.add_argument("--mesa-dir", type=pathlib.Path, default=None)
    parser.add_argument("--mesasdk-root", type=pathlib.Path, default=None)
    parser.add_argument("--alt-mesa-dir", type=pathlib.Path, default=None)
    parser.add_argument("--baseline-fc-flags", default=DEFAULT_BASELINE_FLAGS)
    parser.add_argument("--variant-fc-flags", default=DEFAULT_VARIANT_FLAGS)
    parser.add_argument("--baseline-math-slot", default=DEFAULT_MATH_SLOT)
    parser.add_argument("--variant-math-slot", default=DEFAULT_MATH_SLOT)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[f"{s}/{c}" for s, c in DEFAULT_CASES],
        help="Cases in '<suite>/<case>' format.",
    )
    parser.add_argument(
        "--compare-histories",
        nargs=3,
        action="append",
        metavar=("SUITE_CASE", "BASE_HIST", "VARIANT_HIST"),
        default=[],
        help="Compare two pre-existing history files directly (no MESA build).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan build/comparison commands and emit valid schema JSON only.",
    )
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    parser.add_argument("--markdown-out", type=pathlib.Path, default=None)
    return parser


def _execute_live_spread(
    mesa_dir: pathlib.Path,
    sdk_root: pathlib.Path,
    alt_dir: pathlib.Path,
    base_cfg: CompilerConfigSpec,
    var_cfg: CompilerConfigSpec,
    cases: Sequence[tuple[str, str]],
) -> list[CaseSpreadResult]:
    _install_variant_mesa(mesa_dir, alt_dir, sdk_root, var_cfg)
    work_base = alt_dir.parent / "mesa-spread-runs-base"
    work_var = alt_dir.parent / "mesa-spread-runs-alt"
    work_base.mkdir(parents=True, exist_ok=True)
    work_var.mkdir(parents=True, exist_ok=True)
    spreads: list[CaseSpreadResult] = []
    for suite, case in cases:
        b_hist = _run_mesa_case_in_tree(
            mesa_dir, sdk_root, suite, case, work_base, base_cfg
        )
        v_hist = _run_mesa_case_in_tree(
            alt_dir, sdk_root, suite, case, work_var, var_cfg
        )
        spreads.append(compare_history_tables(suite, case, b_hist, v_hist))
    return spreads


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning ``0`` on clean execution."""
    ns = _build_arg_parser().parse_args(argv)
    base_cfg = CompilerConfigSpec(
        name="baseline_o2",
        fc_flags=ns.baseline_fc_flags,
        math_slot=ns.baseline_math_slot,
        omp_num_threads=int(ns.omp_threads),
    )
    var_cfg = CompilerConfigSpec(
        name="variant_alt",
        fc_flags=ns.variant_fc_flags,
        math_slot=ns.variant_math_slot,
        omp_num_threads=int(ns.omp_threads),
    )
    cases = [_parse_suite_case(raw) for raw in ns.cases]
    mesa_dir = (
        ns.mesa_dir.resolve()
        if ns.mesa_dir
        else pathlib.Path(os.environ.get("MESA_DIR", "mesa")).expanduser()
    )
    sdk_root = (
        ns.mesasdk_root.resolve()
        if ns.mesasdk_root
        else pathlib.Path(
            os.environ.get("MESASDK_ROOT", "mesasdk")
        ).expanduser()
    )
    alt_dir = (
        ns.alt_mesa_dir.resolve()
        if ns.alt_mesa_dir
        else mesa_dir.with_name(f"{mesa_dir.name}-alt-spread")
    )

    spreads: list[CaseSpreadResult] = []
    planned: list[str] = []
    if ns.compare_histories:
        for suite_case, b_hist, v_hist in ns.compare_histories:
            suite, case = _parse_suite_case(suite_case)
            spreads.append(
                compare_history_tables(
                    suite,
                    case,
                    pathlib.Path(b_hist).resolve(),
                    pathlib.Path(v_hist).resolve(),
                )
            )
    elif ns.dry_run:
        planned = plan_alt_mesa_commands(
            mesa_dir, sdk_root, alt_dir, var_cfg, cases
        )
    else:
        spreads = _execute_live_spread(
            mesa_dir, sdk_root, alt_dir, base_cfg, var_cfg, cases
        )

    report = build_spread_report(
        base_cfg,
        var_cfg,
        spreads,
        dry_run=bool(ns.dry_run),
        planned_commands=planned,
    )
    md_text = render_spread_markdown(report)
    if ns.json_out is not None:
        ns.json_out.parent.mkdir(parents=True, exist_ok=True)
        ns.json_out.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if ns.markdown_out is not None:
        ns.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        ns.markdown_out.write_text(md_text, encoding="utf-8")
    sys.stdout.write(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
