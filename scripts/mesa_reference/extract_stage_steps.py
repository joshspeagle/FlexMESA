# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract MESA 26.4.1 per-stage step counts from the stored real-MESA runs.

Each stage of a MESA ``test_suite`` case ends with a ``termination code:``
line followed by ``runtime (minutes), retries, steps <min> <retries> <steps>``
(``star/job/run_star_support.f90``). ``steps`` there is the model number at
the end of the stage, so for a stage that starts at model 0 it is the number
of steps taken, and for a stage that continues a loaded model it includes
the models of the earlier stage. The harness compares FlexMESA's final model
number of each stage with this value.

A completed run that prints no such line (``star/custom_colors``) uses the
model number of the last row of its single ``history.data``, and the row
says so in ``stage_steps_source``: MESA saves the history of the final step.
A skipped, timed-out or failed run has no step count (empty list), so no
comparison is made for it.

The counts are cross-checked against the ``stage_runs`` of each run's
``case_result.json`` and written, with the SHA-256 of every ``out.txt``, to
``benchmarks/history_harness/mesa_stage_steps.json``.

Usage:
    python scripts/mesa_reference/extract_stage_steps.py RUNS_DIR OUT_JSON
where ``RUNS_DIR`` is the ``runs`` directory of the ``mesa-histories``
bundle (``<suite>/<case>/out.txt``); the release, tag, commit, SDK and
compiler recorded in the bundle's ``PROVENANCE.json`` next to it are copied
into the output.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

_STEPS_RE = re.compile(
    r"runtime \(minutes\), retries, steps\s+\S+\s+(\d+)\s+(\d+)"
)
_TERM_RE = re.compile(r"^termination code:\s*(\S+)", re.MULTILINE)
_VERSION_RE = re.compile(r"version_number\s+(r\S+)")


def _last_model_number(history: pathlib.Path) -> int:
    """Return ``model_number`` of the last row of a MESA ``history.data``."""
    header: list[str] | None = None
    last = None
    for line in history.read_text(encoding="utf-8").splitlines():
        cols = line.split()
        if header is None:
            if cols and cols[0] == "model_number":
                header = cols
            continue
        if cols:
            last = int(cols[0])
    if header is None or last is None:
        raise ValueError(f"{history}: no model_number rows")
    return last


def extract(runs_dir: pathlib.Path) -> dict[str, object]:
    """Return the stage-step table of every case under ``runs_dir``."""
    cases: dict[str, object] = {}
    versions: set[str] = set()
    for out in sorted(runs_dir.glob("*/*/out.txt")):
        suite, case = out.parent.parent.name, out.parent.name
        raw = out.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        versions.update(_VERSION_RE.findall(text))
        steps = [int(m.group(2)) for m in _STEPS_RE.finditer(text)]
        retries = [int(m.group(1)) for m in _STEPS_RE.finditer(text)]
        result = json.loads(
            (out.parent / "case_result.json").read_text(encoding="utf-8")
        )
        recorded = [int(s["steps"]) for s in result.get("stage_runs") or []]
        if recorded and recorded != steps:
            raise ValueError(
                f"{suite}/{case}: out.txt stage steps {steps} differ from "
                f"case_result.json {recorded}"
            )
        source = "out.txt"
        if not steps:
            files = result.get("history_files") or []
            if result.get("status") == "ok" and len(files) == 1:
                steps = [_last_model_number(out.parent / files[0])]
                source = "history.data last row"
            else:
                source = "none (run not completed or skipped)"
            retries = []
        cases[f"{suite}/{case}"] = {
            "stage_steps_source": source,
            "stage_steps": steps,
            "stage_retries": retries,
            "termination_codes": _TERM_RE.findall(text),
            "out_txt_sha256": hashlib.sha256(raw).hexdigest(),
        }
    bundle = runs_dir.parent / "PROVENANCE.json"
    meta = (
        json.loads(bundle.read_text(encoding="utf-8"))
        if bundle.is_file()
        else {}
    )
    mesa = meta.get("mesa", {})
    sdk = meta.get("sdk_and_compiler", {})
    return {
        "provenance": {
            "bundle": meta.get("bundle_name"),
            "mesa_release": mesa.get("release"),
            "mesa_git_tag": mesa.get("git_tag"),
            "mesa_git_commit": mesa.get("git_commit"),
            "mesasdk_version": sdk.get("mesasdk_version"),
            "compiler": sdk.get("compiler"),
            "mesa_version_number": sorted(versions),
            "source": (
                "runs/<suite>/<case>/out.txt of the mesa-histories bundle"
            ),
            "rule": (
                "steps of the 'runtime (minutes), retries, steps' line that "
                "ends each stage: the model number at the end of the stage"
            ),
            "script": "scripts/mesa_reference/extract_stage_steps.py",
        },
        "cases": cases,
    }


def main(argv: list[str]) -> int:
    """Write the stage-step table (see module docstring)."""
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    table = extract(pathlib.Path(argv[0]))
    pathlib.Path(argv[1]).write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
