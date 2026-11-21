#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Merge a campaign's baseline candidate into the committed baseline.

The scheduled campaign uploads `baseline_candidate.json`, a regression
snapshot of every harness case on a GitHub runner. A runner is slower than
the machines some committed rows came from, so a case that reached its
termination condition in the committed baseline can end on the time budget
in the candidate. Such a row is kept; every other row is taken from the
candidate (`notes/DECISIONS.md` #51).

Usage:
    python scripts/ci/merge_baseline_candidate.py CANDIDATE BASELINE
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


def keeps_committed_row(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Return whether the committed row carries more than the candidate row.

    Args:
        old: The committed baseline row of a case.
        new: The candidate row of the same case.

    Returns:
        True when the committed run ended on a termination condition other
        than the time budget and the candidate run ended on the time budget.
    """
    old_ended = old.get("run_status") == "ok" and old.get(
        "observed_termination"
    ) not in ("", "timeout", None)
    return old_ended and new.get("run_status") == "timeout"


def merge(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Merge candidate rows into the baseline.

    Args:
        candidate: Parsed `baseline_candidate.json`.
        baseline: Parsed committed baseline.

    Returns:
        The merged payload, the cases whose committed row was kept, and the
        cases whose row was replaced by a different candidate row.

    Raises:
        ValueError: If the two files do not list the same cases.
    """
    old_rows = {(c["suite"], c["case"]): c for c in baseline["cases"]}
    new_keys = {(c["suite"], c["case"]) for c in candidate["cases"]}
    if set(old_rows) != new_keys:
        raise ValueError("candidate and baseline list different cases")
    kept: list[str] = []
    replaced: list[str] = []
    rows: list[dict[str, Any]] = []
    for new in candidate["cases"]:
        key = (new["suite"], new["case"])
        old = old_rows[key]
        label = f"{key[0]}/{key[1]}"
        if keeps_committed_row(old, new):
            rows.append(old)
            kept.append(label)
        else:
            rows.append(new)
            if new != old:
                replaced.append(label)
    merged = {"summary": candidate["summary"], "cases": rows}
    return merged, kept, replaced


def main(argv: list[str]) -> int:
    """Run the merge and rewrite the baseline file in place."""
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    cand_path, base_path = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    candidate = json.loads(cand_path.read_text(encoding="utf-8"))
    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    merged, kept, replaced = merge(candidate, baseline)
    base_path.write_text(
        json.dumps(merged, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Rebuild the payload with the harness's own writer so that the summary
    # counts and verdicts are derived from the merged rows.
    from flexmesa.testing import case_report

    harness_dir = base_path.resolve().parent
    manifest, gaps, records = case_report.build_manifest_inventory_records(
        harness_dir / "case_manifest.toml",
        harness_dir / "known_gaps.toml",
        base_path,
    )
    payload = case_report.build_deterministic_baseline_payload(
        manifest, records, gaps, numerics_mode="strict"
    )
    base_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"kept {len(kept)} committed rows: {', '.join(kept)}")
    print(f"replaced {len(replaced)} rows: {', '.join(replaced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
