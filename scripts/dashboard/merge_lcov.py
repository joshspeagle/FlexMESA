#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Merge per-shard LCOV reports into a ``cargo llvm-cov`` summary JSON (`#345`).

The dashboard's Rust coverage runs as one job per test shard. Each shard writes
an LCOV report with ``cargo llvm-cov report --lcov``. This script takes the
union of those reports, line by line: a line is instrumented if any shard
reports it, and covered if any shard hit it. It writes the result in the shape
of ``cargo llvm-cov --json --summary-only`` (``data[0].files[].filename`` and
``.summary.lines.{count,covered}``), which is what
``flexmesa.testing._dashboard_coverage.parse_llvm_cov_summary`` reads.

Usage::

    python scripts/dashboard/merge_lcov.py --out rust_cov.json shard*.lcov
"""

from __future__ import annotations

import argparse
import json
import pathlib


def read_lcov(path: pathlib.Path, lines: dict[str, dict[int, int]]) -> None:
    """Add the ``DA`` records of one LCOV file to ``lines``.

    Args:
        path: LCOV file.
        lines: Mapping of source file to line number to hit count, updated in
            place with the maximum hit count seen for each line.
    """
    current: dict[int, int] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("SF:"):
            current = lines.setdefault(raw[3:], {})
        elif raw.startswith("DA:") and current is not None:
            fields = raw[3:].split(",")
            line_no, hits = int(fields[0]), int(fields[1])
            current[line_no] = max(current.get(line_no, 0), hits)
        elif raw == "end_of_record":
            current = None


def summary_json(lines: dict[str, dict[int, int]]) -> dict[str, object]:
    """Return the merged report in ``cargo llvm-cov`` summary shape."""
    files = []
    for filename in sorted(lines):
        hits = lines[filename]
        files.append(
            {
                "filename": filename,
                "summary": {
                    "lines": {
                        "count": len(hits),
                        "covered": sum(1 for h in hits.values() if h > 0),
                    }
                },
            }
        )
    return {"type": "llvm.coverage.json.export", "data": [{"files": files}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("lcov", type=pathlib.Path, nargs="+")
    ns = parser.parse_args(argv)
    lines: dict[str, dict[int, int]] = {}
    for path in ns.lcov:
        read_lcov(path, lines)
    if not lines:
        parser.error("no DA records found in the LCOV inputs")
    ns.out.write_text(json.dumps(summary_json(lines)) + "\n", encoding="utf-8")
    total = sum(len(v) for v in lines.values())
    covered = sum(1 for v in lines.values() for h in v.values() if h > 0)
    print(f"merged {len(ns.lcov)} reports: {covered}/{total} lines covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
