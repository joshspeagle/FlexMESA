#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""First-divergence table: FlexMESA strict runs against MESA 26.4.1 step dumps.

For every ``<suite>/<case>.stepdump`` under ``--dumps`` (written by
``run_stepdumps.py``), runs ``flexmesa trace <suite> <case> --compare`` with
this checkout's build and records the first record that differs from MESA:
the model, the record type, the zone and the variable. Writes a Markdown
table and a JSON file (#372).

Usage::

    python first_divergence_table.py --dumps <dir> --work <scratch dir> \\
        --markdown-out first_divergence.md --json-out first_divergence.json \\
        [--cases star/1.3M_ms_high_Z ...] [--timeout-sec 600]

Runs use ``--allow-unsupported-controls`` so that every case reaches its
first solver call; a case whose run ignores controls is marked as such.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

from flexmesa.testing import stepdump_compare


def run_case(
    suite: str, case: str, dump: pathlib.Path, work: pathlib.Path, timeout: int
) -> dict[str, object]:
    """Runs one case with the dump on and compares it with MESA's dump.

    Args:
        suite: ``star``, ``binary`` or ``astero``.
        case: Case name.
        dump: MESA ``.stepdump`` file.
        work: Scratch directory for this case.
        timeout: Wall-clock limit in seconds.

    Returns:
        One table row.
    """
    out = work / suite / case
    cmd = [
        sys.executable,
        "-m",
        "flexmesa.cli",
        "trace",
        suite,
        case,
        "--allow-unsupported-controls",
        "--out",
        str(out),
        "--compare",
        str(dump),
        "--timeout-sec",
        str(timeout),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=os.environ.copy()
    )
    row: dict[str, object] = {"case": f"{suite}/{case}"}
    status = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("suite=")), ""
    )
    row["run"] = status.split(" status=")[-1].split()[0] if status else "?"
    flex = out / "flexmesa.stepdump"
    if not flex.exists():
        errs = [
            ln
            for ln in proc.stderr.splitlines()
            if ln.startswith("error:") and "wrote no step dump" not in ln
        ]
        row["first_divergence"] = "no FlexMESA dump (run stopped first)"
        row["detail"] = errs[0][len("error: ") :][:160] if errs else ""
        return row
    mesa = stepdump_compare.parse_dump(dump.read_text(encoding="utf-8"))
    mine = stepdump_compare.parse_dump(flex.read_text(encoding="utf-8"))
    div = stepdump_compare.first_divergence(mesa, mine)
    if div is None:
        row["first_divergence"] = "none (bit-identical)"
        row["detail"] = ""
        return row
    row["first_divergence"] = (
        f"model #{div.model_index + 1} {div.kind} {div.record}"
        + (f" zone {div.zone}" if div.zone is not None else "")
        + (f" record {div.position}" if div.position is not None else "")
    )
    row["detail"] = f"{div.field}" + (
        f", {div.ulps} ULP" if div.ulps is not None else ""
    )
    row["divergence"] = dataclasses.asdict(div)
    return row


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    parser.add_argument("--markdown-out", type=pathlib.Path, required=True)
    parser.add_argument("--json-out", type=pathlib.Path, required=True)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()
    dumps = sorted(args.dumps.glob("*/*.stepdump"))
    if args.cases:
        wanted = set(args.cases)
        dumps = [d for d in dumps if f"{d.parent.name}/{d.stem}" in wanted]
    rows = []
    for d in dumps:
        row = run_case(d.parent.name, d.stem, d, args.work, args.timeout_sec)
        print(f"{row['case']}: {row['first_divergence']}", flush=True)
        rows.append(row)
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n", "utf-8")
    lines = [
        "| Case | Run | First divergence from MESA 26.4.1 | Field |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['case']}` | {r['run']} | {r['first_divergence']} "
            f"| {r['detail']} |"
        )
    args.markdown_out.write_text("\n".join(lines) + "\n", "utf-8")


if __name__ == "__main__":
    main()
