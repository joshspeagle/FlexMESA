#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the `Fast`-vs-`Strict` numerical divergence report (`#165` Item 17).

Compares `NumericsMode::Fast` exclusively against `NumericsMode::Strict` (never
against MESA):
1. Across all `474,068` reference inputs (`47` sections) in
   `benchmarks/fortran_mesa/math_hex_vectors/mesa_26_4_1_math_vectors.txt`,
   recording per-section finite input count, maximum ULP distance, maximum
   absolute difference, and maximum relative difference.
2. Across a short stellar evolution (`star/1.3M_ms_high_Z`, 5 steps) run in
   isolated subprocesses under `--numerics-mode strict` and `--numerics-mode
   fast`, recording per-column maximum absolute and relative differences on
   `LOGS/history.data`.

Note that `NumericsMode::Fast` invokes platform `f64` intrinsics and hardware
FMA (`f64::mul_add`), so the measured numbers depend on the host CPU and C math
library; the report records the host architecture and OS explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt

from flexmesa import _rust

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_VECTORS = (
    _REPO_ROOT
    / "benchmarks"
    / "fortran_mesa"
    / "math_hex_vectors"
    / "mesa_26_4_1_math_vectors.txt"
)
_DEFAULT_MD = _REPO_ROOT / "notes" / "audit" / "fast_vs_strict.md"
_DEFAULT_JSON = _REPO_ROOT / "notes" / "audit" / "fast_vs_strict.json"

type _ArrF64 = npt.NDArray[np.float64]


def _parse_mesa_history(path: pathlib.Path) -> tuple[list[str], _ArrF64]:
    """Parse column names and numeric table from a MESA `history.data` file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    col_names = lines[5].split()
    rows = [
        [float(tok.replace("D", "E").replace("d", "e")) for tok in ln.split()]
        for ln in lines[6:]
        if ln.strip()
    ]
    return col_names, np.asarray(rows, dtype=np.float64)


def _run_mode_history(
    suite: str,
    case: str,
    max_steps: int,
    mode: str,
) -> tuple[list[str], _ArrF64]:
    """Run ``flexmesa trace`` in ``mode`` and parse ``LOGS/history.data``."""
    with tempfile.TemporaryDirectory(prefix=f"flexmesa_{mode}_") as tmp:
        out_dir = pathlib.Path(tmp) / "run"
        env = dict(os.environ)
        env["RAYON_NUM_THREADS"] = "2"
        env["OMP_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["FLEXMESA_NUMERICS_MODE"] = mode
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
            mode,
        ]
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{mode} trace failed ({proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        return _parse_mesa_history(out_dir / "LOGS" / "history.data")


def _build_report_data(
    vectors_path: pathlib.Path,
    suite: str,
    case: str,
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate `Fast`-vs-`Strict` across reference vectors and evolution."""
    rows = _rust.compare_fast_vs_strict_vectors(str(vectors_path))
    fn_records: list[dict[str, Any]] = [
        {
            "section": name,
            "total_inputs": int(tot),
            "finite_pairs": int(fin),
            "max_ulp_diff": int(mulp),
            "max_abs_diff": float(mabs),
            "max_rel_diff": float(mrel),
        }
        for (name, tot, fin, mulp, mabs, mrel) in rows
    ]
    total_vec_inputs = sum(int(tot) for (_, tot, _, _, _, _) in rows)

    cols_s, arr_s = _run_mode_history(suite, case, max_steps, "strict")
    cols_f, arr_f = _run_mode_history(suite, case, max_steps, "fast")
    if cols_s != cols_f or arr_s.shape != arr_f.shape:
        raise RuntimeError(
            f"history shape/column mismatch between strict {arr_s.shape} and "
            f"fast {arr_f.shape}"
        )

    col_records: list[dict[str, Any]] = []
    max_rel_overall = 0.0
    for idx, cname in enumerate(cols_s):
        vs = arr_s[:, idx]
        vf = arr_f[:, idx]
        abs_d = np.abs(vf - vs)
        denom = np.maximum(np.abs(vs), np.abs(vf))
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        rel_d = np.where(denom > 0.0, abs_d / safe_denom, 0.0)
        m_abs = float(np.max(abs_d))
        m_rel = float(np.max(rel_d))
        max_rel_overall = max(max_rel_overall, m_rel)
        col_records.append(
            {
                "column": cname,
                "max_abs_diff": m_abs,
                "max_rel_diff": m_rel,
            }
        )

    return {
        "comparison": "fast_vs_strict_only",
        "host_os": platform.system(),
        "host_arch": platform.machine(),
        "vector_section_count": len(fn_records),
        "total_vector_inputs": total_vec_inputs,
        "functions": fn_records,
        "short_evolution": {
            "suite": suite,
            "case": case,
            "max_steps": max_steps,
            "row_count": int(arr_s.shape[0]),
            "column_count": len(cols_s),
            "max_rel_diff_overall": max_rel_overall,
            "columns": col_records,
        },
    }


def _render_markdown(data: dict[str, Any]) -> str:
    """Format the `Fast`-vs-`Strict` report as GitHub-flavored Markdown."""
    evo = data["short_evolution"]
    host = f"{data['host_os']} / {data['host_arch']}"
    hdr = (
        "| Section | Inputs | Finite Pairs "
        "| Max ULP | Max Abs Diff | Max Rel Diff |"
    )
    lines = [
        "# Fast vs. Strict Numerics Mode Report (`#165`)",
        "",
        "Generated by `scripts/generate_fast_vs_strict_report.py`. Compares",
        "`NumericsMode::Fast` (`std` `f64` intrinsics with hardware `mul_add`)",
        "exclusively against `NumericsMode::Strict` (`crlibm` without FMA).",
        "`Fast` mode is hardware-dependent and is never compared against MESA.",
        "",
        f"- **Host platform**: `{host}`",
        f"- **Reference input sections**: `{data['vector_section_count']}` "
        f"(`{data['total_vector_inputs']:,}` total inputs)",
        f"- **Short evolution check**: `{evo['suite']}/{evo['case']}` "
        f"(`{evo['max_steps']}` steps, `{evo['row_count']}` logged rows, "
        f"`{evo['column_count']}` columns, overall max relative difference "
        f"`{evo['max_rel_diff_overall']:.3e}`)",
        "",
        "## 1. Per-Function `Fast` vs. `Strict` Differences",
        "",
        hdr,
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fn in data["functions"]:
        lines.append(
            f"| `{fn['section']}` | {fn['total_inputs']:,} | "
            f"{fn['finite_pairs']:,} | {fn['max_ulp_diff']:,} | "
            f"`{fn['max_abs_diff']:.3e}` | `{fn['max_rel_diff']:.3e}` |"
        )

    lines.extend(
        [
            "",
            f"## 2. Short Evolution (`{evo['suite']}/{evo['case']}`, "
            f"`{evo['max_steps']}` Steps): `Fast` vs. `Strict`",
            "",
            "| History Column | Max Abs Diff | Max Rel Diff |",
            "| :--- | ---: | ---: |",
        ]
    )
    for col in evo["columns"]:
        lines.append(
            f"| `{col['column']}` | `{col['max_abs_diff']:.3e}` | "
            f"`{col['max_rel_diff']:.3e}` |"
        )
    lines.extend(
        [
            "",
            "## 3. What Was Not Verified",
            "",
            "- `NumericsMode::Fast` is platform-dependent; values above",
            f"  reflect `{host}` only and can differ on other CPUs or `libm`",
            "  implementations.",
            "- Full-lifetime stellar evolutions (`10^3` to `10^5` steps)",
            "  across the 106-case suite were not run in `Fast` mode here;",
            f"  only the 5-step `{evo['suite']}/{evo['case']}` run above was",
            "  measured.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating `notes/audit/fast_vs_strict.{md,json}`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vectors", type=pathlib.Path, default=_DEFAULT_VECTORS
    )
    parser.add_argument("--suite", default="star")
    parser.add_argument("--case", default="1.3M_ms_high_Z")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--md-out", type=pathlib.Path, default=_DEFAULT_MD)
    parser.add_argument("--json-out", type=pathlib.Path, default=_DEFAULT_JSON)
    args = parser.parse_args(argv)

    data = _build_report_data(
        args.vectors, args.suite, args.case, args.max_steps
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.md_out.write_text(_render_markdown(data), encoding="utf-8")
    evo_rel = data["short_evolution"]["max_rel_diff_overall"]
    sys.stdout.write(
        f"Wrote {args.md_out} and {args.json_out} "
        f"({data['total_vector_inputs']:,} inputs, max_rel={evo_rel:.3e})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
