#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""First-divergence table: FlexMESA strict runs against MESA 26.4.1 step dumps.

For every ``<suite>/<case>.stepdump`` under ``--dumps`` (written by
``run_stepdumps.py``), runs ``flexmesa trace <suite> <case> --compare`` with
this checkout's build and records the first record that differs from MESA:
the model, the record type, the zone and the variable. Writes a Markdown
table and a JSON file (#372).

Usage::

    python first_divergence_table.py --dumps <dir> --work <scratch dir> \\
        --markdown-out first_divergence.md --json-out first_divergence.json \\
        [--cases star/1.3M_ms_high_Z ...] [--timeout-sec 600] \\
        [--baseline <baseline_run.json>] [--flexmesa-commit <sha>] \\
        [--command-text "<command recorded in the header>"]

With ``--baseline`` and ``--flexmesa-commit`` the Markdown page carries a
header (FlexMESA commit, the MESA version, compiler and SDK read from the
dump headers, the command) and a ranked list of the rows that are not
bit-identical, each with the harness verdict of its case. Output is
deterministic: no timings and no machine paths.

A window that opens at a later model is compared with ``--max-steps`` set a
little past its end, so the run stops once it has the window. A window whose
header names a stage or an occurrence after the first is compared with
``--stage`` and ``--occurrence``: earlier stages run uncapped without
dumping and the run stops after the window. Only the ``star`` runner marks
stage boundaries, so a later stage of another suite is listed as skipped.

Runs use ``--allow-unsupported-controls`` so that every case reaches its
first solver call. A run that ignores controls does not do what its inlist
asks (``DECISIONS #26``); the table does not mark such runs, so check a
case's ignored controls before reading its row as a faithful run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys

from flexmesa.testing import stepdump_compare


def parse_header(text: str) -> dict[str, str]:
    """Reads the ``# key: value`` header lines of a dump.

    Args:
        text: Dump file contents (only the leading ``#`` lines are read).

    Returns:
        Mapping from header key to its stripped value.
    """
    head: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        key, _, val = line[1:].partition(":")
        head[key.strip()] = val.strip()
    return head


def window_skip_reason(head: dict[str, str], suite: str = "star") -> str:
    """Returns why a dump window cannot be compared, or ``""``.

    Args:
        head: Header from `parse_header`.
        suite: ``star``, ``binary`` or ``astero``.

    Returns:
        A reason when the window names a later stage of a suite whose
        FlexMESA runner does not mark stage boundaries for the step dump
        (only the ``star`` runner does), else the empty string.
    """
    stage = head.get("stage", "all")
    if suite != "star" and stage not in ("all", "1"):
        return f"stage {stage} (no stage selector for the {suite} runner)"
    return ""


def window_selector_args(head: dict[str, str]) -> list[str]:
    """Returns the ``trace`` options that select the window's stage and run.

    Args:
        head: Header from `parse_header`.

    Returns:
        ``--stage S`` and ``--occurrence K`` for the values the header
        records (``all`` and ``1`` when absent select nothing).
    """
    args: list[str] = []
    stage = head.get("stage", "all")
    if stage.isdigit() and int(stage) > 0:
        args += ["--stage", stage]
    occ = head.get("occurrence", "1")
    if occ.isdigit() and int(occ) > 1:
        args += ["--occurrence", occ]
    return args


def window_max_steps(head: dict[str, str]) -> int | None:
    """Returns a ``--max-steps`` value that stops a run after its window.

    Args:
        head: Header from `parse_header`.

    Returns:
        ``None`` for a window at the start of the run (``trace`` already caps
        it) or in a selected stage (the run stops itself after the window and
        earlier stages must reach their own end), else a step count
        comfortably past the end of the window.
    """
    if head.get("stage", "all").isdigit():
        return None
    first = head.get("first_model", "")
    steps = head.get("steps_per_star", "")
    if not (first.isdigit() and steps.isdigit()):
        return None
    return int(1.5 * (int(first) + int(steps))) + 20


def run_case(
    suite: str,
    case: str,
    dump: pathlib.Path,
    work: pathlib.Path,
    timeout: int,
    label: str | None = None,
) -> dict[str, object]:
    """Runs one case with the dump on and compares it with MESA's dump.

    Args:
        suite: ``star``, ``binary`` or ``astero``.
        case: Case name.
        dump: MESA ``.stepdump`` file.
        work: Scratch directory for this case.
        timeout: Wall-clock limit in seconds.
        label: Row label, ``<suite>/<case>`` when omitted (a window id may
            be appended so that two windows of one case stay distinct).

    Returns:
        One table row.
    """
    row: dict[str, object] = {"case": label or f"{suite}/{case}"}
    head = parse_header(dump.read_text(encoding="utf-8"))
    row["window"] = (
        f"models {head['first_model']}..+{head['steps_per_star']}"
        if head.get("first_model", "").isdigit()
        else "start of run"
    )
    skip = window_skip_reason(head, suite)
    if skip:
        row["run"] = "skipped"
        row["first_divergence"] = "not compared"
        row["detail"] = skip
        return row
    out = work / re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["case"]))
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
    cmd += window_selector_args(head)
    max_steps = window_max_steps(head)
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=os.environ.copy()
    )
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


def _is_identical(row: dict[str, object]) -> bool:
    return str(row["first_divergence"]).startswith("none")


def _rank_key(row: dict[str, object]) -> tuple[int, str]:
    div = row.get("divergence")
    model = div["mesa_model"] if isinstance(div, dict) else 10**9
    return (int(model), str(row["case"]))


def _verdicts(path: pathlib.Path | None) -> dict[str, str]:
    """Maps ``suite/case`` to the harness verdict and known-gap issue."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for c in data["cases"]:
        out[f"{c['suite']}/{c['case']}"] = str(c["verdict"])
    return out


def _provenance(dumps: list[pathlib.Path]) -> str:
    """One line naming the MESA version, compiler and SDK of the dumps."""
    seen: set[tuple[str, str, str]] = set()
    for d in dumps:
        h = parse_header(d.read_text(encoding="utf-8"))
        seen.add(
            (
                h.get("mesa_version", "?"),
                h.get("compiler", "?"),
                h.get("mesa_sdk_version", h.get("sdk", "?")),
            )
        )
    return "; ".join(f"MESA {v}, {c}, SDK {k}" for v, c, k in sorted(seen))


def render_markdown(
    rows: list[dict[str, object]],
    *,
    commit: str,
    provenance: str,
    command: str,
    verdicts: dict[str, str],
) -> str:
    """Renders the page: header, counts, ranked list, full table.

    Args:
        rows: Table rows from `run_case`.
        commit: FlexMESA commit the runs used.
        provenance: MESA version, compiler and SDK of the dumps.
        command: Command recorded in the header.
        verdicts: ``suite/case`` to harness verdict text.

    Returns:
        The Markdown text (deterministic for fixed input).
    """
    compared = [r for r in rows if r["run"] != "skipped"]
    ident = [r for r in compared if _is_identical(r)]
    nodump = [
        r for r in compared if "no FlexMESA dump" in str(r["first_divergence"])
    ]
    differ = [r for r in compared if r not in ident and r not in nodump]
    skipped = [r for r in rows if r["run"] == "skipped"]
    lines = [
        "# First divergence of FlexMESA from MESA step dumps",
        "",
        "Generated by `first_divergence_table.py`; do not edit by hand.",
        "",
        "Every run uses `--allow-unsupported-controls`, so a row may come "
        "from a run that ignored controls (`DECISIONS #26`); check the "
        "case's ignored controls before reading a bit-identical row as a "
        "faithful run. `ULP` is the distance between the IEEE-754 bit "
        "patterns, so a sign or exponent mismatch reads as a very large "
        "number.",
        "",
        f"- FlexMESA commit: `{commit}`",
        f"- MESA dumps: {provenance}",
        f"- Command: `{command}`",
        f"- Rows: {len(rows)}; compared {len(compared)}; "
        f"bit-identical over the window {len(ident)}; "
        f"first difference found {len(differ)}; "
        f"FlexMESA wrote no dump {len(nodump)}; skipped {len(skipped)}.",
        "",
        "## Rows that differ, earliest MESA model first",
        "",
        "| Rank | Row | First divergence | Field | Harness verdict |",
        "| ---: | :--- | :--- | :--- | :--- |",
    ]
    for n, r in enumerate(sorted(differ, key=_rank_key), 1):
        v = verdicts.get(str(r["case"]).split(" (")[0], "")
        lines.append(
            f"| {n} | `{r['case']}` | {r['first_divergence']} "
            f"| {r['detail']} | {v} |"
        )
    if nodump:
        lines += [
            "",
            "## Rows where FlexMESA wrote no dump",
            "",
            "| Row | Reason | Harness verdict |",
            "| :--- | :--- | :--- |",
        ]
        for r in sorted(nodump, key=lambda r: str(r["case"])):
            v = verdicts.get(str(r["case"]).split(" (")[0], "")
            lines.append(f"| `{r['case']}` | {r['detail']} | {v} |")
    if skipped:
        lines += [
            "",
            "## Rows not compared",
            "",
            "| Row | Reason |",
            "| :--- | :--- |",
        ]
        for r in sorted(skipped, key=lambda r: str(r["case"])):
            lines.append(f"| `{r['case']}` | {r['detail']} |")
    lines += [
        "",
        "## All rows",
        "",
        "| Row | Window | Run | First divergence from MESA | Field |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in sorted(rows, key=lambda r: str(r["case"])):
        lines.append(
            f"| `{r['case']}` | {r['window']} | {r['run']} "
            f"| {r['first_divergence']} | {r['detail']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dumps",
        type=pathlib.Path,
        nargs="*",
        default=[],
        help="Directories laid out as <suite>/<case>.stepdump.",
    )
    parser.add_argument(
        "--windows",
        type=pathlib.Path,
        default=None,
        help="Mid-run windows root, <window id>/<suite>/<case>.stepdump.",
    )
    parser.add_argument("--work", type=pathlib.Path, required=True)
    parser.add_argument(
        "--render-from",
        type=pathlib.Path,
        nargs="*",
        default=[],
        help="Skip the runs; load rows from these JSON files (earlier "
        "outputs) and render one page. --dumps/--windows still give the "
        "header's MESA provenance.",
    )
    parser.add_argument("--markdown-out", type=pathlib.Path, required=True)
    parser.add_argument("--json-out", type=pathlib.Path, required=True)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--baseline", type=pathlib.Path, default=None)
    parser.add_argument("--flexmesa-commit", default="unknown")
    parser.add_argument(
        "--command-text",
        default=None,
        help="Command text for the page header (default: this argv).",
    )
    args = parser.parse_args()
    items: list[tuple[pathlib.Path, str | None]] = []
    for root in args.dumps:
        items += [(d, None) for d in sorted(root.glob("*/*.stepdump"))]
    if args.windows is not None:
        items += [
            (d, "window") for d in sorted(args.windows.glob("*/*/*.stepdump"))
        ]
    if args.cases:
        wanted = set(args.cases)
        items = [
            (d, k) for d, k in items if f"{d.parent.name}/{d.stem}" in wanted
        ]
    rows: list[dict[str, object]] = []
    for js in args.render_from:
        rows += json.loads(js.read_text(encoding="utf-8"))
    for d, kind in [] if args.render_from else items:
        label = None
        if kind == "window":
            h = parse_header(d.read_text(encoding="utf-8"))
            lo = (
                int(h["first_model"])
                if h.get("first_model", "").isdigit()
                else 1
            )
            n = int(h.get("steps_per_star", "3"))
            label = (
                f"{d.parent.name}/{d.stem} "
                f"(stage {h.get('stage', 'all')}, occurrence "
                f"{h.get('occurrence', '1')}, models {lo}..{lo + n - 1})"
            )
        row = run_case(
            d.parent.name, d.stem, d, args.work, args.timeout_sec, label
        )
        print(f"{row['case']}: {row['first_divergence']}", flush=True)
        rows.append(row)
    rows.sort(key=lambda r: str(r["case"]))
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n", "utf-8")
    command = args.command_text or "first_divergence_table.py " + " ".join(
        sys.argv[1:]
    )
    args.markdown_out.write_text(
        render_markdown(
            rows,
            commit=args.flexmesa_commit,
            provenance=_provenance([d for d, _ in items]),
            command=command,
            verdicts=_verdicts(args.baseline),
        ),
        "utf-8",
    )


if __name__ == "__main__":
    main()
