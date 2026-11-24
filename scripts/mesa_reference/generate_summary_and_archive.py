# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate test-suite summary tables in ``MESA_GROUND_TRUTH_ROOT``.

Reads ``runs/<suite>/<case>/case_result.json`` across ``star``, ``binary``, and
``astero`` in exact ``do1_test_source`` order and writes ``SUMMARY_TABLE.md``,
``summary_table.tsv``, and ``summary_table.json`` programmatically before
``scripts/data/build_mesa_histories.py`` packages the ``data-v1`` release
archives.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FLEXMESA_DIR = pathlib.Path(
    os.environ.get("FLEXMESA_DIR", str(SCRIPT_DIR.parents[1]))
)
THIN_SCRIPT = FLEXMESA_DIR / "notes/audit/tools/thin_mesa_history.py"
ROOT_DIR = pathlib.Path(
    os.environ.get(
        "MESA_GROUND_TRUTH_ROOT",
        str(pathlib.Path.home() / "mesa-ground-truth-26.4.1"),
    )
)
MESA_DIR = pathlib.Path(
    os.environ.get("MESA_DIR", str(pathlib.Path.home() / "mesa-26.04.1"))
)


def load_suite_order(suite: str) -> list[str]:
    """Return case names in exact ``do1_test_source`` order."""
    src = (MESA_DIR / suite / "test_suite/do1_test_source").read_text(
        encoding="utf-8"
    )
    cases: list[str] = []
    for ln in src.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r'^do_one\s+(\S+)\s+"([^"]*)"\s+(\S+)\s+(\S+)', s)
        if m:
            cases.append(m.group(1))
    return cases


def refresh_case_result_if_needed(
    case_dir: pathlib.Path, r: dict[str, Any]
) -> dict[str, Any]:
    """Backfill termination strings and skip/timeout details from ``out.txt``."""
    out_path = case_dir / "out.txt"
    if not out_path.exists():
        return r
    changed = False
    out_text = out_path.read_text(encoding="utf-8", errors="replace")
    if r.get("termination_code") is None:
        for pat, prefix in (
            (r"^\s*termination code:\s*(.+?)\s*$", ""),
            (r"^\s*terminated evolution:\s*(.+?)\s*$", ""),
            (r"^\s*(Terminate\b.+?)\s*$", ""),
            (r"^\s*(terminate due to\b.+?)\s*$", ""),
            (r"^\s*(stop because\b.+?)\s*$", ""),
            (r"^\s*(save_sample_results_to_file\s+\S+)\s*$", ""),
            (r"^\s*(have reached\s+\S+)\s*$", "extras: "),
            (
                r"^\s*(crossed blue edge to start 3rd crossing)\b.*$",
                "extras_finish_step: ",
            ),
            (r"^\s*(good match for period)\b.*$", "extras_finish_step: "),
            (
                r"^\s*(done rsp_check_2nd_crossing)\b.*$",
                "other_rsp_build_model: ",
            ),
        ):
            found = [
                f"{prefix}{m.group(1).strip()}"
                for m in re.finditer(pat, out_text, re.MULTILINE)
            ]
            if found:
                r["termination_code"] = found[-1]
                r["all_termination_codes"] = found
                changed = True
                break
    if r.get("case") == "rsp_check_2nd_crossing" and r.get("steps") is None:
        r["steps"] = 0
        r["retries"] = 0
        r["total_steps_all_stages"] = 0
        r["total_retries_all_stages"] = 0
        r["reason"] = (
            "Linear nonadiabatic LINA stability scan "
            "(951 models across Teff; 0 time-evolution steps)"
        )
        changed = True
    if r.get("status") == "timeout":
        hist_p = case_dir / "LOGS/history.data"
        thin_dst = (
            ROOT_DIR / "thinned" / r["suite"] / f"{r['case']}.partial.tsv"
        )
        if hist_p.exists() and not thin_dst.exists() and THIN_SCRIPT.is_file():
            thin_dst.parent.mkdir(parents=True, exist_ok=True)
            note = (
                "MESA 26.4.1 (c8c2026), MESA SDK 26.6.1 (gfortran 15.2.0), "
                "OMP_NUM_THREADS=4, MESA_SKIP_OPTIONAL=t, partial track up to "
                f"{r.get('wall_minutes', 240.0):.1f} min wall timeout "
                f"(model_number={r.get('steps')}); "
                f"{r['suite']}/test_suite/{r['case']}"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(THIN_SCRIPT),
                    str(hist_p),
                    str(thin_dst),
                    note,
                ],
                check=True,
            )
            r["thinned_files"] = [str(thin_dst.relative_to(ROOT_DIR))]
            changed = True
    if changed:
        (case_dir / "case_result.json").write_text(
            json.dumps(r, indent=2) + "\n", encoding="utf-8"
        )
    return r


def build_tables() -> dict[str, Any]:
    """Write ``summary_table.{json,tsv}`` and ``SUMMARY_TABLE.md``."""
    rows: list[dict[str, Any]] = []
    for suite in ("star", "binary", "astero"):
        for idx, case in enumerate(load_suite_order(suite), 1):
            cdir = ROOT_DIR / "runs" / suite / case
            rpath = cdir / "case_result.json"
            if rpath.exists():
                r = json.loads(rpath.read_text(encoding="utf-8"))
                r = refresh_case_result_if_needed(cdir, r)
            else:
                r = {
                    "suite": suite,
                    "case": case,
                    "order_in_suite": idx,
                    "status": "not_run",
                    "reason": "Not yet executed",
                    "termination_code": None,
                    "steps": None,
                    "retries": None,
                    "total_steps_all_stages": None,
                    "total_retries_all_stages": None,
                    "num_stages_reported": 0,
                    "wall_minutes": None,
                    "pulse_modes_count": 0,
                    "pulse_models_count": 0,
                }
            rows.append(r)

    totals = {
        "total_cases": len(rows),
        "by_suite": {
            s: sum(1 for r in rows if r["suite"] == s)
            for s in ("star", "binary", "astero")
        },
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "skipped_upstream": sum(
            1 for r in rows if r["status"] == "skipped_upstream"
        ),
        "failed": sum(
            1
            for r in rows
            if r["status"]
            in ("failed", "compile_failed", "timeout", "rss_exceeded")
        ),
        "not_run": sum(1 for r in rows if r["status"] == "not_run"),
        "sum_wall_minutes": round(
            sum(float(r["wall_minutes"] or 0.0) for r in rows), 2
        ),
        "sum_final_stage_steps": sum(int(r["steps"] or 0) for r in rows),
        "sum_all_stage_steps": sum(
            int(r.get("total_steps_all_stages") or r.get("steps") or 0)
            for r in rows
        ),
        "sum_final_stage_retries": sum(int(r["retries"] or 0) for r in rows),
        "sum_all_stage_retries": sum(
            int(r.get("total_retries_all_stages") or r.get("retries") or 0)
            for r in rows
        ),
        "pulsation_cases_with_modes": sum(
            1 for r in rows if (r.get("pulse_modes_count") or 0) > 0
        ),
        "total_pulsation_modes_recorded": sum(
            int(r.get("pulse_modes_count") or 0) for r in rows
        ),
        "total_pulsation_models_saved": sum(
            int(r.get("pulse_models_count") or 0) for r in rows
        ),
    }

    mod_summary_path = ROOT_DIR / "module_tests/summary.json"
    mod_summary = (
        json.loads(mod_summary_path.read_text(encoding="utf-8"))
        if mod_summary_path.exists()
        else {}
    )
    payload = {
        "totals": totals,
        "module_tests_summary": mod_summary,
        "cases": rows,
    }
    (ROOT_DIR / "summary_table.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    tsv_lines = [
        "suite\tcase\tstatus\ttermination_code\tsteps\tretries\t"
        "all_stage_steps\tall_stage_retries\twall_minutes\tpulse_modes\t"
        "pulse_models\treason_or_notes"
    ]
    md = [
        "# MESA 26.4.1 Reference Histories Summary (`mesa-histories-v1`)",
        "",
        (
            "Generated programmatically by "
            "`scripts/mesa_reference/generate_summary_and_archive.py`."
        ),
        "",
        "| Suite | Case | Status | Termination Code | Steps | Retries | Wall (min) | Notes |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for r in rows:
        notes_parts: list[str] = []
        if r.get("reason"):
            notes_parts.append(str(r["reason"]))
        if (r.get("pulse_modes_count") or 0) > 0:
            notes_parts.append(
                f"{r['pulse_modes_count']} modes / {r['pulse_models_count']} models"
            )
        note_s = "; ".join(notes_parts) if notes_parts else "-"
        wall_s = (
            f"{r['wall_minutes']:.2f}"
            if r.get("wall_minutes") is not None
            else "-"
        )
        tsv_lines.append(
            "\t".join(
                [
                    r["suite"],
                    r["case"],
                    r["status"],
                    str(r["termination_code"] or "-"),
                    str(r["steps"] if r["steps"] is not None else "-"),
                    str(r["retries"] if r["retries"] is not None else "-"),
                    str(
                        r.get("total_steps_all_stages")
                        if r.get("total_steps_all_stages") is not None
                        else "-"
                    ),
                    str(
                        r.get("total_retries_all_stages")
                        if r.get("total_retries_all_stages") is not None
                        else "-"
                    ),
                    wall_s,
                    str(r.get("pulse_modes_count") or 0),
                    str(r.get("pulse_models_count") or 0),
                    note_s,
                ]
            )
        )
        term_md = (
            f"`{r['termination_code']}`" if r.get("termination_code") else "-"
        )
        steps_md = str(r["steps"]) if r["steps"] is not None else "-"
        ret_md = str(r["retries"]) if r["retries"] is not None else "-"
        md.append(
            f"| `{r['suite']}` | `{r['case']}` | `{r['status']}` | "
            f"{term_md} | {steps_md} | {ret_md} | {wall_s} | "
            f"{note_s.replace('|', '\\|')} |"
        )

    (ROOT_DIR / "summary_table.tsv").write_text(
        "\n".join(tsv_lines) + "\n", encoding="utf-8"
    )
    (ROOT_DIR / "SUMMARY_TABLE.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    """Generate summary tables in ``ROOT_DIR``."""
    payload = build_tables()
    print(
        f"Generated summary tables ({payload['totals']['total_cases']} cases)"
    )


if __name__ == "__main__":
    main()
