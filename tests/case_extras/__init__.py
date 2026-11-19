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


def extras_source_digest(case_dir: pathlib.Path | str) -> str:
    """Return the SHA-256 of a case's normalized upstream extras sources.

    Args:
        case_dir: Path to the case directory on disk.

    Returns:
        Hex digest over normalized `run_star_extras.f90` and
        `run_binary_extras.f90` (an absent file contributes an empty string).
    """
    root = pathlib.Path(case_dir)
    texts = []
    for rel_name in ("run_star_extras.f90", "run_binary_extras.f90"):
        raw = load_fortran_source(root, rel_name)
        texts.append(normalize_fortran(raw) if raw else "")
    return source_sha256("\x00".join(texts))


# Digests of extras sources that were read and contain no case logic: every
# routine returns `keep_going`, reports zero extra columns, or calls only the
# shared `test_suite_startup` / `test_suite_after_evolve` timing helpers.
STOCK_EXTRAS_DIGESTS: frozenset[str] = frozenset(
    {
        # star/test_suite/1.3M_ms_high_Z and 12 cases with identical text.
        "4b7b570f3a144678fa67d368b2c5b485ad89e03f9b43abe8575c1635939c1a5a",
        # binary/test_suite/evolve_both_stars and 2 cases with identical text.
        "f4e8cb4d5f84304947b6a817879063dc65c04cacca90bba857f07fd50020e440",
    }
)


def classify_case_extras(suite: str, case_dir: pathlib.Path | str) -> str:
    """Classify a case's upstream extras sources.

    Args:
        suite: Suite identifier (`"star"`, `"binary"`, or `"astero"`).
        case_dir: Path to the case directory on disk.

    Returns:
        `"ported"` when `resolve_case_extras` attaches a handler, `"stock"`
        when the normalized sources equal a reviewed no-logic template, and
        `"unported"` otherwise. An unported case runs with default no-op
        hooks, so its upstream extras logic is not executed.
    """
    if resolve_case_extras(suite, case_dir) is not None:
        return "ported"
    if extras_source_digest(case_dir) in STOCK_EXTRAS_DIGESTS:
        return "stock"
    return "unported"


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
