# SPDX-License-Identifier: GPL-3.0-or-later
"""Failing-case attribution for the unmapped-routine backlog (`#611`).

The backlog lists routines of core-engine files with status ``stub``,
``partial``, or ``absent`` that no gate-checked ``routine_map`` entry names.
This module attaches to each routine the non-passing harness cases that
plausibly need it, using one mechanical signal only: the free-text ``reason``
of each declared gap in ``benchmarks/history_harness/known_gaps.toml``.

A routine is attributed to a case when the case's reason

* names the routine (an identifier equal to the Fortran name, compared
  without regard to case, of at least eight characters or containing an
  underscore, so that short common words never match), basis ``routine``; or
* names the routine's upstream file, by full path or by file name of at least
  six characters, basis ``file`` (every unmapped routine of the file is then
  attributed, because the reason does not say which one is needed).

A routine name that occurs in several backlog files is attributed only when
the same reason also names one of those files; otherwise it is left
unattributed. Neither the unsupported-controls report nor
``extras_status.toml`` names a routine, so they cannot attribute anything and
are not used. Routines that no reason names stay unattributed, which does not
mean that no case needs them.
"""

from __future__ import annotations

import pathlib
import re
import tomllib
from typing import Any

GAPS_RELPATH = "benchmarks/history_harness/known_gaps.toml"
BACKLOG_STATUSES = ("stub", "partial", "absent")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILE = re.compile(r"[A-Za-z0-9_]+\.(?:f90|F90|f|inc|dek)\b")


def load_gap_cases(repo_root: pathlib.Path) -> list[tuple[str, str]]:
    """Return ``(case_id, reason)`` for every declared harness gap.

    Args:
        repo_root: Repository root.

    Returns:
        Pairs of ``"<suite>/<case>"`` and the gap's ``reason`` text.
    """
    raw = tomllib.loads((repo_root / GAPS_RELPATH).read_text(encoding="utf-8"))
    return [
        (f"{g['suite']}/{g['case']}", str(g.get("reason", "")))
        for g in raw.get("gaps", [])
    ]


def _name_ok(name: str) -> bool:
    return "_" in name or len(name) >= 8


def attribute(
    backlog: list[dict[str, Any]], gaps: list[tuple[str, str]]
) -> None:
    """Fill ``cases`` and ``basis`` on each backlog row, in place.

    Args:
        backlog: Rows with ``upstream_path`` and ``routine`` keys.
        gaps: Output of :func:`load_gap_cases`.
    """
    by_name: dict[str, set[str]] = {}
    for b in backlog:
        by_name.setdefault(b["routine"].lower(), set()).add(b["upstream_path"])
    paths = {b["upstream_path"] for b in backlog}
    for b in backlog:
        b["cases"] = []
        b["basis"] = ""
    for case_id, reason in gaps:
        low = reason.lower()
        idents = {t.lower() for t in _IDENT.findall(reason)}
        files = {
            p
            for p in paths
            if p.lower() in low
            or (
                len(p.rsplit("/", 1)[-1]) >= 6
                and p.rsplit("/", 1)[-1].lower() in _file_tokens(reason)
            )
        }
        for b in backlog:
            name = b["routine"].lower()
            if name in idents and _name_ok(name):
                homes = by_name[name]
                if len(homes) == 1 or b["upstream_path"] in files:
                    _add(b, case_id, "routine")
            elif b["upstream_path"] in files:
                _add(b, case_id, "file")


def _file_tokens(reason: str) -> set[str]:
    return {t.lower() for t in _FILE.findall(reason)}


def _add(row: dict[str, Any], case_id: str, basis: str) -> None:
    if case_id not in row["cases"]:
        row["cases"].append(case_id)
    if basis == "routine" or not row["basis"]:
        row["basis"] = basis
