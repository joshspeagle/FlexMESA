# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream Fortran and FlexMESA symbol parsers (`#174`)."""

from __future__ import annotations

import collections.abc
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
FORTRAN_PARSE_SUFFIXES: frozenset[str] = frozenset(
    {".f90", ".f", ".dek", ".inc", ".incl", ".fpp", ".fypp"}
)

_PROC_DECL_RE = re.compile(
    r"^(?:(?:pure|elemental|recursive|impure|module)\s+)*"
    r"(?:(?:real|integer|logical|character|complex|double\s+precision|type)"
    r"\s*(?:\*\s*(?:\([^)]*\)|\d+)|\([^)]*\))?\s+)*"
    r"(?:subroutine|function)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    flags=re.IGNORECASE,
)
_NAMED_INTERFACE_RE = re.compile(
    r"^interface\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
    flags=re.IGNORECASE,
)
_INCLUDE_STMT_RE = re.compile(
    r'^include\s+["\']([^"\']+)["\']\s*$',
    flags=re.IGNORECASE,
)
_RUST_STRIP_RE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/|\bb?r(#*)\"[\s\S]*?\"\1"
    r"|b?\"(?:\\[\s\S]|[^\"\\])*\""
    r"|(?:\bb)?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|.)|[^'\n\\])'"
)
_RUST_SYM_RE = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async)\s+)*"
    r"(?:fn|struct|enum|trait|const|type|static)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_RUST_MACRO_METH_RE = re.compile(
    r"\b(?:unary_method|unary3_method|pown3_method)!"
    r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
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


def strip_rust_comments_and_strings(text: str) -> str:
    """Replace Rust comments, string, and char literals with whitespace.

    Preserves all newline characters so line numbers remain invariant.

    Args:
        text: Raw Rust source text.

    Returns:
        Sanitized Rust source text with line numbers preserved.
    """
    return _RUST_STRIP_RE.sub(
        lambda m: "\n" * m.group(0).count("\n") + " ", text
    )


def _logical_fortran_lines(source_text: str, fixed_form: bool) -> list[str]:
    """Join Fortran line continuations and strip comments/directives."""
    logical: list[str] = []
    buf = ""
    if fixed_form:
        for raw in source_text.splitlines():
            if not raw or raw[:1] in {"c", "C", "*", "!"}:
                continue
            code = raw.split("!")[0].rstrip()
            if not code.strip():
                continue
            is_cont = (
                len(raw) >= 6
                and raw[:5].strip() == ""
                and raw[5] not in {" ", "0"}
                and "\t" not in raw[:6]
            )
            if is_cont:
                cont = code[6:].lstrip()
                buf = f"{buf} {cont}".strip() if buf else cont
                continue
            if buf:
                logical.append(buf)
            stmt_body = code
            if len(raw) >= 6 and raw[:5].strip().isdigit():
                stmt_body = code[6:]
            buf = stmt_body.strip()
        if buf:
            logical.append(buf)
        return logical

    for raw in source_text.splitlines():
        if raw.lstrip().startswith(("#", "$:")):
            continue
        code = raw.split("!")[0].rstrip()
        if not code.strip():
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
    source_text: str,
    suffix: str = ".f90",
    include_resolver: (
        collections.abc.Callable[[str], tuple[str, str] | None] | None
    ) = None,
) -> list[str]:
    """Extract public procedure and generic-interface names from Fortran text.

    Args:
        source_text: Decoded upstream Fortran file text.
        suffix: File extension (used to detect fixed-form ``.f`` sources).
        include_resolver: Optional callback returning ``(text, suffix)`` for a
            ``contains``-level ``include "<file>"`` statement.

    Returns:
        Ordered list of public subroutine, function, and generic-interface
        names.
    """
    ext = suffix.lower()
    if ext not in FORTRAN_PARSE_SUFFIXES:
        return []
    logical = _logical_fortran_lines(source_text, fixed_form=(ext == ".f"))

    has_module = False
    has_default_private = False
    explicit_public: list[str] = []
    explicit_private: set[str] = set()
    module_procs: list[str] = []
    seen_procs_lower: set[str] = set()
    in_module_contains = False
    in_interface = False
    generic_iface_name: str | None = None
    proc_depth = 0
    seen_includes: set[str] = set()

    def _add_proc(name: str) -> None:
        low_n = name.lower()
        if low_n not in seen_procs_lower:
            seen_procs_lower.add(low_n)
            module_procs.append(name)

    def _scan_lines(lines: list[str]) -> None:
        nonlocal has_module, has_default_private, in_module_contains
        nonlocal in_interface, generic_iface_name, proc_depth
        for line in lines:
            stmt = line.strip()
            low = stmt.lower()
            if re.match(r"^module\s+(?!procedure\b)[a-z_][a-z0-9_]*\b", low):
                has_module = True
                continue
            if re.match(r"^(?:abstract\s+)?interface\b", low):
                in_interface = True
                m_gen = _NAMED_INTERFACE_RE.match(stmt)
                generic_iface_name = m_gen.group(1) if m_gen else None
                continue
            if re.match(r"^end\s*interface\b", low):
                in_interface = False
                generic_iface_name = None
                continue
            if in_interface:
                if (
                    generic_iface_name is not None
                    and re.match(r"^(?:module\s+)?procedure\b", low) is not None
                ):
                    _add_proc(generic_iface_name)
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
                        if clean and "(" not in clean:
                            explicit_public.append(clean)
                elif re.match(r"^private\s*(?:::)?\s+", low):
                    rest = re.sub(
                        r"^private\s*(?:::)?\s*", "", stmt, flags=re.IGNORECASE
                    )
                    for tok in rest.split(","):
                        clean = tok.strip()
                        if clean and "(" not in clean:
                            explicit_private.add(clean.lower())
            else:
                if re.match(r"^end\s*(?:subroutine|function)\b", low) or (
                    low == "end" and proc_depth > 0
                ):
                    proc_depth = max(0, proc_depth - 1)
                    continue
                if proc_depth == 0 and include_resolver is not None:
                    m_inc = _INCLUDE_STMT_RE.match(stmt)
                    if m_inc is not None:
                        inc_name = m_inc.group(1).strip()
                        inc_ext = pathlib.PurePosixPath(inc_name).suffix.lower()
                        if (
                            inc_ext in FORTRAN_PARSE_SUFFIXES
                            and inc_name not in seen_includes
                        ):
                            seen_includes.add(inc_name)
                            resolved = include_resolver(inc_name)
                            if resolved is not None:
                                inc_txt, inc_sfx = resolved
                                _scan_lines(
                                    _logical_fortran_lines(
                                        inc_txt,
                                        fixed_form=(inc_sfx.lower() == ".f"),
                                    )
                                )
                        continue
                match = _PROC_DECL_RE.match(stmt)
                if match is not None:
                    if proc_depth == 0:
                        _add_proc(match.group(1))
                    proc_depth += 1

    _scan_lines(logical)

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
        stripped = strip_rust_comments_and_strings(text)
        syms = set(_RUST_SYM_RE.findall(stripped))
        syms.update(_RUST_MACRO_METH_RE.findall(stripped))
        return syms
    if path.suffix == ".py":
        out: set[str] = set()
        for m in _PY_SYM_RE.finditer(text):
            sym = m.group(1) or m.group(2)
            if sym:
                out.add(sym)
        return out
    return set()


_RUST_IMPL_RE = re.compile(r"\bimpl\b")
_RUST_FN_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_RUST_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_target_symbols(syms_raw: str) -> list[tuple[str | None, str]]:
    """Split the symbol part of a ``routine_map`` target.

    The symbol part is ``sym[+sym...]``, where each ``sym`` is a function
    name or a type-qualified method ``Type::method``.

    Args:
        syms_raw: Text after the first ``::`` of ``<path>::<symbols>``.

    Returns:
        ``(type, name)`` pairs in order; ``type`` is ``None`` for an
        unqualified symbol. Empty items are dropped.
    """
    out: list[tuple[str | None, str]] = []
    for item in syms_raw.split("+"):
        item = item.strip()
        if not item:
            continue
        owner, sep, name = item.rpartition("::")
        owner = owner.split("::")[-1].strip()
        out.append((owner if sep and owner else None, name.strip()))
    return out


def _skip_angle(text: str, i: int) -> int:
    """Return the index after the ``<...>`` group starting at ``text[i]``."""
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">" and text[i - 1] != "-":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _impl_self_type(header: str) -> str | None:
    """Return the self type name of an ``impl`` header (after ``impl``)."""
    header = header.strip()
    if header.startswith("<"):
        header = header[_skip_angle(header, 0) :].strip()
    header = re.split(r"\bwhere\b", header, maxsplit=1)[0]
    # A trait impl is `Trait<..> for Type<..>`; take what follows `for`.
    depth = 0
    for_at = -1
    for m in re.finditer(r"[<>]|\bfor\b", header):
        tok = m.group(0)
        if tok == "<":
            depth += 1
        elif tok == ">" and header[m.start() - 1] != "-":
            depth -= 1
        elif depth == 0:
            for_at = m.end()
    if for_at >= 0:
        header = header[for_at:]
    header = re.sub(r"^[\s&*]*(?:mut\b|const\b|dyn\b)?\s*", "", header.strip())
    idents = _RUST_IDENT_RE.findall(header.split("<", 1)[0])
    return idents[-1] if idents else None


def rust_impl_methods(stripped: str) -> dict[str, set[str]]:
    """Map each ``impl`` self type to the ``fn`` names of its impl blocks.

    Inherent impls (``impl Type``), generic impls (``impl<T> Type<T>``), and
    trait impls (``impl Trait for Type``) all contribute to ``Type``. Only
    items directly in the impl body count, not functions nested in method
    bodies.

    Args:
        stripped: Rust source from ``strip_rust_comments_and_strings``.

    Returns:
        Mapping from type name to the set of method names.
    """
    out: dict[str, set[str]] = {}
    for m in _RUST_IMPL_RE.finditer(stripped):
        open_at = stripped.find("{", m.end())
        semi_at = stripped.find(";", m.end())
        if open_at < 0 or 0 <= semi_at < open_at:
            continue
        tname = _impl_self_type(stripped[m.end() : open_at])
        if tname is None:
            continue
        depth = 0
        body_start = open_at + 1
        pos = open_at
        methods = out.setdefault(tname, set())
        item_start = body_start
        while pos < len(stripped):
            ch = stripped[pos]
            if ch == "{":
                if depth == 1:
                    methods.update(
                        _RUST_FN_RE.findall(stripped[item_start:pos])[:1]
                    )
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                if depth == 1:
                    item_start = pos + 1
            elif ch == ";" and depth == 1:
                methods.update(
                    _RUST_FN_RE.findall(stripped[item_start:pos])[:1]
                )
                item_start = pos + 1
            pos += 1
    return out


def rust_fn_counts(stripped: str) -> dict[str, int]:
    """Count ``fn name`` occurrences (free functions and methods together)."""
    counts: dict[str, int] = {}
    for name in _RUST_FN_RE.findall(stripped):
        counts[name] = counts.get(name, 0) + 1
    return counts
