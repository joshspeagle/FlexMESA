# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream routines declared not applicable to a Rust port (`#611`).

``benchmarks/dashboard/port_status_na.toml`` lists upstream routines that have
no counterpart by design, such as Fortran memory management, module-global
callback adapters, debug helpers, empty stubs, and ``pgstar`` plotting. Each
``[[na]]`` entry carries ``upstream_path``, ``routine``, ``category`` and
``reason``, and may carry ``parent``, ``first`` and ``last`` to pin the
inventory row. The routine table reports these routines as their own level.
They are never counted as mapped and never removed from the denominators.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from typing import Any

from flexmesa.testing import routine_gate

NA_RELPATH = "benchmarks/dashboard/port_status_na.toml"


@dataclasses.dataclass(frozen=True)
class NotApplicable:
    """One routine declared not applicable to the Rust port.

    Attributes:
        upstream_path: Manifest path of the upstream file.
        routine: Fortran routine name.
        category: Reason category, for example ``fortran_mem`` or ``stub``.
        reason: Free-text justification.
        parent: Enclosing routine name, empty for a top-level routine.
        first: First line of the routine in the inventory, or ``None``.
        last: Last line of the routine in the inventory, or ``None``.
    """

    upstream_path: str
    routine: str
    category: str
    reason: str
    parent: str = ""
    first: int | None = None
    last: int | None = None

    def key(self) -> str:
        """Return the ``name`` or ``parent::name`` key of the routine."""
        return f"{self.parent}::{self.routine}" if self.parent else self.routine


def load_na(path: pathlib.Path) -> list[NotApplicable]:
    """Load the N/A declarations (empty when the file does not exist).

    Raises:
        ValueError: If an entry lacks a required field.
    """
    if not path.is_file():
        return []
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    out: list[NotApplicable] = []
    for item in raw.get("na", []):
        for field in ("upstream_path", "routine", "category", "reason"):
            if not item.get(field):
                raise ValueError(f"{path.name}: entry lacks {field!r}: {item}")
        out.append(
            NotApplicable(
                upstream_path=str(item["upstream_path"]),
                routine=str(item["routine"]),
                category=str(item["category"]),
                reason=str(item["reason"]),
                parent=str(item.get("parent", "")),
                first=int(item["first"]) if "first" in item else None,
                last=int(item["last"]) if "last" in item else None,
            )
        )
    return out


def na_row_ids(
    entries: list[NotApplicable], path: str, routines: list[list[Any]]
) -> set[int]:
    """Return ``id()`` of the inventory rows of ``path`` declared N/A."""
    out: set[int] = set()
    for na in entries:
        if na.upstream_path == path:
            rows = routine_gate.na_rows(na, routines)
            if len(rows) == 1:
                out.add(id(rows[0]))
    return out
