#!/usr/bin/env python3
"""Reproduce the MESA 26.4.1 binary RLOF test-fixture check.

Copies ``$MESA_DIR/binary/work`` into a temporary directory, stages the
committed inlists and ``run_binary_extras.f90`` for both the original
``(1.5 Msun, 1.0 Msun, P = 0.45 d)`` setup and the replacement
``(2.0 Msun, 1.5 Msun, P = 0.5045 d)`` setup, builds and runs MESA 26.4.1,
and writes a structured JSON summary parsed from ``run.log`` and
``binary_history.data``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any


def parse_mesa_run(
    run_dir: pathlib.Path,
    label: str,
    m1_msun: float,
    m2_msun: float,
    period_days: float,
) -> dict[str, Any]:
    """Parse MESA binary run logs and history headers into a summary dict.

    Args:
        run_dir: Working directory containing ``run.log`` and
            ``binary_history.data``.
        label: Short case identifier.
        m1_msun: Initial donor mass in solar masses.
        m2_msun: Initial point-mass companion mass in solar masses.
        period_days: Initial orbital period in days.

    Returns:
        Dictionary of version headers, retry counts, and step-1 observables.
    """
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    bh_lines = (
        (run_dir / "binary_history.data")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    hdr_names = bh_lines[1].split()
    hdr_vals = [val.strip('"') for val in bh_lines[2].split()]
    header = dict(zip(hdr_names, hdr_vals, strict=False))

    col_names = bh_lines[5].split()
    rows: list[dict[str, float | int]] = []
    for line in bh_lines[6:]:
        parts = line.split()
        if len(parts) == len(col_names):
            row: dict[str, float | int] = {}
            for key, val in zip(col_names, parts, strict=True):
                if key in ("model_number", "donor_index", "point_mass_index"):
                    row[key] = int(val)
                else:
                    row[key] = float(val)
            rows.append(row)

    term_match = re.search(r"termination code:\s*(\S+)", log_text)
    solver_ret_match = re.search(r"solver retries\s+(\d+)", log_text)
    step1_retries = len(re.findall(r"retry:\s+.*?\s+1\s*$", log_text, re.M))
    fr_hard_retries = len(
        re.findall(r"retry because of fr_hard limit", log_text)
    )
    give_up_retries = len(re.findall(r"give up in solver", log_text))
    bound_retries = len(re.findall(r"retry:\s+log(?:T|Rho)\s+[<>]", log_text))
    dt_vals = [
        float(val.replace("D", "e"))
        for val in re.findall(r"\bdt\s+([0-9.]+D[+-]\d+)", log_text)
    ]

    return {
        "case": label,
        "m1_msun": m1_msun,
        "m2_msun": m2_msun,
        "initial_period_days": period_days,
        "mesa_version": header.get("version_number", "unknown"),
        "compiler": (
            f"{header.get('compiler', 'unknown')} "
            f"{header.get('build', 'unknown')}"
        ),
        "mesa_sdk_version": header.get("MESA_SDK_version", "unknown"),
        "run_date": header.get("date", ""),
        "termination_code": term_match.group(1) if term_match else "unknown",
        "models_completed": max(
            (int(r["model_number"]) for r in rows), default=0
        ),
        "step1_age_yr": float(rows[0]["age"]) if rows else None,
        "step1_rl_relative_overflow_1": (
            float(rows[0]["rl_relative_overflow_1"]) if rows else None
        ),
        "step1_lg_mtransfer_rate": (
            float(rows[0]["lg_mtransfer_rate"]) if rows else None
        ),
        "step1_retries_logged": step1_retries,
        "solver_retries_reported": (
            int(solver_ret_match.group(1)) if solver_ret_match else 0
        ),
        "give_up_in_solver_count": give_up_retries,
        "hydro_mtx_bound_retry_count": bound_retries,
        "fr_hard_retry_count": fr_hard_retries,
        "min_reported_dt_sec": min(dt_vals) if dt_vals else None,
    }


def run_case(
    fixture_dir: pathlib.Path,
    mesa_dir: pathlib.Path,
    label: str,
    m1_msun: float,
    m2_msun: float,
    period_days: float,
    max_retries: int | None,
    timeout_sec: int,
) -> dict[str, Any]:
    """Build and execute one MESA 26.4.1 binary case in a temporary workdir.

    Args:
        fixture_dir: Directory holding the committed template inlists.
        mesa_dir: Root of a compiled MESA 26.4.1 checkout.
        label: Case name written to the summary JSON.
        m1_msun: Initial donor mass in solar masses.
        m2_msun: Initial companion point mass in solar masses.
        period_days: Initial orbital period in days.
        max_retries: Optional ``max_number_retries`` cap for ``inlist1``.
        timeout_sec: Subprocess timeout in seconds.

    Returns:
        Parsed summary dictionary from :func:`parse_mesa_run`.
    """
    with tempfile.TemporaryDirectory(prefix=f"mesa_rlof_{label}_") as tmp:
        work = pathlib.Path(tmp) / "work"
        shutil.copytree(mesa_dir / "binary" / "work", work)
        for fname in ("inlist", "inlist_project", "inlist1", "inlist2"):
            shutil.copy2(fixture_dir / fname, work / fname)
        shutil.copy2(
            fixture_dir / "run_binary_extras.f90",
            work / "src" / "run_binary_extras.f90",
        )

        proj_txt = (work / "inlist_project").read_text(encoding="utf-8")
        proj_txt = proj_txt.replace("m1 = 1.5d0", f"m1 = {m1_msun}d0")
        proj_txt = proj_txt.replace("m2 = 1.0d0", f"m2 = {m2_msun}d0")
        proj_txt = proj_txt.replace(
            "initial_period_in_days = 0.45d0",
            f"initial_period_in_days = {period_days}d0",
        )
        (work / "inlist_project").write_text(proj_txt, encoding="utf-8")

        in1_txt = (work / "inlist1").read_text(encoding="utf-8")
        in1_txt = in1_txt.replace(
            "initial_mass = 1.5", f"initial_mass = {m1_msun}"
        )
        if max_retries is None:
            in1_txt = in1_txt.replace("   max_number_retries = 40\n", "")
        else:
            in1_txt = in1_txt.replace(
                "max_number_retries = 40",
                f"max_number_retries = {max_retries}",
            )
        (work / "inlist1").write_text(in1_txt, encoding="utf-8")

        env = os.environ.copy()
        env["MESA_DIR"] = str(mesa_dir)
        env.setdefault("OMP_NUM_THREADS", "4")
        subprocess.run(["./mk"], cwd=work, env=env, check=True)
        with (work / "run.log").open("w", encoding="utf-8") as log_fh:
            subprocess.run(
                ["./rn"],
                cwd=work,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        return parse_mesa_run(work, label, m1_msun, m2_msun, period_days)


def main() -> None:
    """CLI entry point for regenerating ``results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("MESA_DIR", "")),
        help="Path to a compiled MESA 26.4.1 tree.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results.json",
        help="Output path for the JSON summary.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=240,
        help="Per-case wall-time timeout in seconds.",
    )
    args = parser.parse_args()
    if not (args.mesa_dir / "binary" / "work").is_dir():
        raise SystemExit(
            "Error: --mesa-dir must point to a compiled MESA 26.4.1 checkout."
        )

    fixture_dir = pathlib.Path(__file__).resolve().parent
    res_1p5 = run_case(
        fixture_dir,
        args.mesa_dir,
        "original_1p5_1p0_0p45d",
        1.5,
        1.0,
        0.45,
        40,
        args.timeout_sec,
    )
    res_2p0 = run_case(
        fixture_dir,
        args.mesa_dir,
        "replacement_2p0_1p5_0p5045d",
        2.0,
        1.5,
        0.5045,
        None,
        args.timeout_sec,
    )
    payload: dict[str, Any] = {
        "reference": {
            "mesa_release": "26.4.1",
            "mesa_version_header": res_1p5["mesa_version"],
            "compiler": res_1p5["compiler"],
            "mesa_sdk_version": res_1p5["mesa_sdk_version"],
            "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "4")),
        },
        "mesa_26_4_1_runs": [res_1p5, res_2p0],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
