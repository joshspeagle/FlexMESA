# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream Fortran and FlexMESA symbol parsers (`#174`)."""

from __future__ import annotations

import pathlib
import re

CORE_UPSTREAM_MODULES: tuple[str, ...] = (
    "star",
    "star_data",
    "binary",
    "astero",
    "eos",
    "kap",
    "net",
    "rates",
    "neu",
    "atm",
    "turb",
    "chem",
    "ionization",
    "colors",
    "num",
    "mtx",
    "interp_1d",
    "interp_2d",
    "auto_diff",
    "math",
    "const",
    "utils",
)

CORE_SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {".f", ".f90", ".inc", ".dek", ".c", ".h"}
)
ADIPLS_SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {".f", ".f90", ".inc", ".incl", ".dek"}
)
STELLA_SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {".f", ".f90", ".inc", ".trf"}
)
GYRE_SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {".fypp", ".f90", ".f", ".inc", ".fpp", ".dek"}
)

_PROC_DECL_RE = re.compile(
    r"^(?:(?:pure|elemental|recursive|impure|module)\s+)*"
    r"(?:(?:real|integer|logical|character|complex|double\s+precision|type)"
    r"\s*(?:\([^)]*\))?\s+)*"
    r"(?:subroutine|function)\s+([A-Za-z0-9_]+)\b",
    flags=re.IGNORECASE,
)
_RUST_SYM_RE = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:fn|struct|enum|trait|const|type|static)\s+([A-Za-z0-9_]+)\b"
)
_PY_SYM_RE = re.compile(
    r"^(?:def|class)\s+([A-Za-z0-9_]+)\b|^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=",
    flags=re.MULTILINE,
)


def classify_upstream_relpath(relpath: str) -> str | None:
    """Map an upstream MESA 26.4.1 relative path to its dashboard module ID.

    Args:
        relpath: POSIX path relative to the MESA 26.4.1 root directory.

    Returns:
        Canonical module name (one of the 26 dashboard modules), or ``None``
        when the path is a test program, ``test_suite`` directory, build
        script, or non-source asset excluded from ``notes/SCOPE.md``.
    """
    parts = relpath.split("/")
    if not parts:
        return None
    top = parts[0]
    ext = pathlib.PurePosixPath(relpath).suffix.lower()

    if top in CORE_UPSTREAM_MODULES:
        if len(parts) >= 2 and (
            parts[1] in {"test", "test_suite", "examples"}
            or parts[1].startswith("dev_cases")
        ):
            return None
        if ext not in CORE_SOURCE_SUFFIXES:
            return None
        if (
            "/pgstar" in relpath
            or "/pgbinary" in relpath
            or relpath.endswith(
                (
                    "star_pgstar.f90",
                    "binary_pgbinary.f90",
                    "pgstar_controls.inc",
                    "pgbinary_controls.inc",
                )
            )
        ):
            return "plotting"
        return top

    if top == "gyre":
        if len(parts) >= 2 and parts[1] == "test":
            return None
        if relpath.startswith("gyre/gyre/"):
            return (
                "gyre"
                if ("src" in parts and ext in GYRE_SOURCE_SUFFIXES)
                else None
            )
        return "gyre" if ext in CORE_SOURCE_SUFFIXES else None

    if top == "adipls":
        if "/test" in relpath:
            return None
        return "adipls" if ext in ADIPLS_SOURCE_SUFFIXES else None

    if top == "stella":
        return "stella" if ext in STELLA_SOURCE_SUFFIXES else None

    return None


def _logical_fortran_lines(source_text: str, fixed_form: bool) -> list[str]:
    """Join Fortran line continuations and strip trailing ``!`` comments."""
    logical: list[str] = []
    buf = ""
    for raw in source_text.splitlines():
        if fixed_form and raw[:1] in {"c", "C", "*"}:
            continue
        code = raw.split("!")[0].rstrip()
        if not code.strip():
            continue
        if (
            fixed_form
            and len(raw) >= 6
            and raw[:5].strip() == ""
            and raw[5] not in {" ", "0"}
            and "\t" not in raw[:6]
        ):
            cont = code[6:].lstrip()
            buf = f"{buf} {cont}".strip() if buf else cont
            continue
        if buf:
            stripped = code.lstrip()
            if stripped.startswith("&"):
                stripped = stripped[1:]
            buf = f"{buf} {stripped}"
        else:
            buf = code
        if buf.endswith("&"):
            buf = buf[:-1].rstrip()
        else:
            logical.append(buf)
            buf = ""
    if buf:
        logical.append(buf)
    return logical


def extract_fortran_public_routines(
    source_text: str, suffix: str = ".f90"
) -> list[str]:
    """Extract module-level public procedure names from a Fortran source file.

    Args:
        source_text: Decoded upstream Fortran file text.
        suffix: File extension (used to detect fixed-form ``.f`` sources).

    Returns:
        Ordered list of public module-level subroutine and function names.
    """
    ext = suffix.lower()
    if ext not in {".f90", ".f"}:
        return []
    logical = _logical_fortran_lines(source_text, fixed_form=(ext == ".f"))

    has_module = False
    has_default_private = False
    explicit_public: list[str] = []
    explicit_private: set[str] = set()
    module_procs: list[str] = []
    in_module_contains = False
    in_interface = False
    proc_depth = 0

    for line in logical:
        stmt = line.strip()
        low = stmt.lower()
        if re.match(r"^module\s+(?!procedure\b)[a-z0-9_]+\b", low):
            has_module = True
            continue
        if re.match(r"^(?:abstract\s+)?interface\b", low):
            in_interface = True
            continue
        if re.match(r"^end\s*interface\b", low):
            in_interface = False
            continue
        if in_interface:
            continue

        if has_module and not in_module_contains:
            if low == "contains":
                in_module_contains = True
                continue
            if low == "private":
                has_default_private = True
            elif re.match(r"^public\s*(?:::)?\s+", low):
                rest = re.sub(
                    r"^public\s*(?:::)?\s*", "", stmt, flags=re.IGNORECASE
                )
                for tok in rest.split(","):
                    clean = tok.strip()
                    if clean:
                        explicit_public.append(clean)
            elif re.match(r"^private\s*(?:::)?\s+", low):
                rest = re.sub(
                    r"^private\s*(?:::)?\s*", "", stmt, flags=re.IGNORECASE
                )
                for tok in rest.split(","):
                    clean = tok.strip()
                    if clean:
                        explicit_private.add(clean.lower())
        else:
            if re.match(r"^end\s*(?:subroutine|function)\b", low):
                proc_depth = max(0, proc_depth - 1)
                continue
            match = _PROC_DECL_RE.match(stmt)
            if match is not None:
                if proc_depth == 0:
                    name = match.group(1)
                    if name.lower() not in {p.lower() for p in module_procs}:
                        module_procs.append(name)
                proc_depth += 1

    by_lower = {p.lower(): p for p in module_procs}
    if has_default_private:
        seen: set[str] = set()
        out: list[str] = []
        for pub in explicit_public:
            key = pub.lower()
            if key in by_lower and key not in seen:
                seen.add(key)
                out.append(by_lower[key])
        return out
    return [p for p in module_procs if p.lower() not in explicit_private]


def extract_flexmesa_symbols(path: pathlib.Path) -> set[str]:
    """Extract top-level and impl item identifiers from a Rust or Python file.

    Args:
        path: Path to a ``.rs`` or ``.py`` source file in the repository.

    Returns:
        Set of defined symbol names in ``path``.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rs":
        return set(_RUST_SYM_RE.findall(text))
    if path.suffix == ".py":
        out: set[str] = set()
        for m in _PY_SYM_RE.finditer(text):
            sym = m.group(1) or m.group(2)
            if sym:
                out.add(sym)
        return out
    return set()
