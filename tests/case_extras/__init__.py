# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream MESA `run_star_extras.f90` and `run_binary_extras.f90` dispatch
for test suites and the history harness (`#148`).

Dispatch inspects normalized Fortran source text inside `<case_dir>/src/` so
relocating or renaming a case directory never changes which extras handler is
attached.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from typing import Any

from tests.case_extras import binary_extras, rsp_extras, star_extras

_INCLUDE_RE = re.compile(r'include\s+[\'"]([^\'"]+)[\'"]', re.IGNORECASE)


def load_fortran_source(case_dir: pathlib.Path, rel_name: str) -> str:
    """Read `<case_dir>/src/<rel_name>` and inline local `src/*.inc` files.

    Args:
        case_dir: Upstream MESA case directory root.
        rel_name: Relative filename inside `<case_dir>/src/`.

    Returns:
        Concatenated Fortran source text, or `""` when the file is absent.
    """
    src_dir = case_dir / "src"
    target = src_dir / rel_name
    if not target.is_file():
        return ""
    raw = target.read_text(encoding="utf-8", errors="replace")
    parts = [raw]
    for inc_name in _INCLUDE_RE.findall(raw):
        inc_path = src_dir / inc_name
        if inc_path.is_file():
            parts.append(inc_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def normalize_fortran(text: str) -> str:
    """Strip Fortran comments and normalize whitespace and casing.

    Args:
        text: Raw Fortran source string.

    Returns:
        Lowercased, whitespace-collapsed string without `!` comments.
    """
    lines: list[str] = []
    for line in text.splitlines():
        code = line.split("!", 1)[0].strip()
        if code:
            lines.append(code.lower())
    return re.sub(r"\s+", " ", " ".join(lines))


def source_sha256(normalized_text: str) -> str:
    """Return the SHA-256 hex digest of normalized Fortran text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def resolve_case_extras(suite: str, case_dir: pathlib.Path | str) -> Any:
    """Resolve test-side extras handler from upstream Fortran source files.

    Args:
        suite: Suite identifier (`"star"`, `"binary"`, or `"astero"`).
        case_dir: Path to the case directory on disk.

    Returns:
        Python extras object implementing `StarExtras` or `BinaryExtras`
        callbacks, or `None` when the upstream source uses default no-op hooks.
    """
    root = pathlib.Path(case_dir)
    if suite == "star":
        raw_star = load_fortran_source(root, "run_star_extras.f90")
        if not raw_star:
            return None
        norm_star = normalize_fortran(raw_star)
        rsp_handler = rsp_extras.match_rsp_extras(norm_star)
        if rsp_handler is not None:
            return rsp_handler
        return star_extras.match_star_extras(norm_star)
    if suite == "binary":
        raw_bin = load_fortran_source(root, "run_binary_extras.f90")
        if not raw_bin:
            return None
        raw_star = load_fortran_source(root, "run_star_extras.f90")
        norm_bin = normalize_fortran(raw_bin)
        norm_star = normalize_fortran(raw_star) if raw_star else ""
        return binary_extras.match_binary_extras(norm_bin, norm_star)
    return None
