# SPDX-License-Identifier: GPL-3.0-or-later
"""Routine-level parser for upstream MESA Fortran sources (`#611`).

The parser reads one Fortran source text and returns every ``subroutine``,
``function``, ``program``, ``block data``, and submodule ``module procedure``
definition with its first and last physical line and its enclosing routine.
It is a statement-level scanner, not a full Fortran front end. It understands
the forms that occur in MESA 26.4.1:

* free-form and fixed-form sources, including continuation lines whose
  routine names fall on a later line;
* ``end subroutine`` / ``end function`` / ``endsubroutine`` forms and a bare
  ``end`` that closes a routine or program unit;
* type-prefixed and attribute-prefixed functions (``real(dp) function f``,
  ``pure``, ``elemental``, ``recursive``, ``impure``, ``module``);
* ``interface`` blocks, whose body declarations are not routine definitions,
  and ``module procedure`` name lists inside them;
* contained (internal) procedures, recorded with their parent routine;
* ``fypp`` templates (``#:`` directives and ``${...}$`` expressions), which
  are scanned as the free-form text they contain and are not expanded.

Preprocessor conditionals (``#ifdef``) are not evaluated, so a routine that
appears in two alternative branches is listed twice.
"""

from __future__ import annotations

import dataclasses
import re

FIXED_FORM_SUFFIXES = frozenset({".f", ".for", ".incl"})

_COND_RE = re.compile(
    r"^#\s*:?\s*(if|ifdef|ifndef|elif|else|endif)\b", re.IGNORECASE
)
_NAME = r"(?:[a-z_]|\$\{[^}]*\}\$)(?:[a-z0-9_]|\$\{[^}]*\}\$)*"
_PAREN = r"\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"
_ATTR = r"(?:pure|elemental|impure|recursive|non_recursive|module)"
_ROUTINE_END = (
    r"subroutine|function|program|module|submodule|interface|type|"
    r"procedure|enum|block\s*data"
)


@dataclasses.dataclass(frozen=True)
class _Patterns:
    """Statement patterns for one source form.

    Fixed-form text has all blanks removed before matching (blanks are not
    significant, and ratfor-generated sources omit them), so its patterns
    use ``\\s*`` between keywords where free form needs ``\\s+``.
    """

    sub: re.Pattern[str]
    func: re.Pattern[str]
    program: re.Pattern[str]
    block_data: re.Pattern[str]
    module_proc: re.Pattern[str]
    module: re.Pattern[str]
    submodule: re.Pattern[str]
    interface: re.Pattern[str]
    type_def: re.Pattern[str]
    enum: re.Pattern[str]
    end: re.Pattern[str]
    bare_end: re.Pattern[str]


def _build_patterns(sp: str) -> _Patterns:
    """Build the statement patterns with ``sp`` between keywords."""
    type_spec = (
        r"(?:(?:real|integer|logical|complex|character|double\s*precision|"
        r"type|class)\s*(?:\*\s*(?:\d+|" + _PAREN + r")|" + _PAREN + r")?)"
    )
    prefix = r"(?:(?:" + _ATTR + r"|" + type_spec + r")" + sp + r")*"
    flags = re.IGNORECASE
    bound = r"\b" if sp == r"\s+" else ""
    return _Patterns(
        sub=re.compile(
            r"^(?:" + _ATTR + sp + r")*subroutine" + sp + r"(" + _NAME + r")",
            flags,
        ),
        func=re.compile(
            r"^" + prefix + r"function" + sp + r"(" + _NAME + r")\s*\(",
            flags,
        ),
        program=re.compile(r"^program" + sp + r"(" + _NAME + r")$", flags),
        block_data=re.compile(
            r"^block\s*data(?:" + sp + r"(" + _NAME + r"))?$", flags
        ),
        module_proc=re.compile(
            r"^module" + sp + r"procedure" + sp + r"(" + _NAME + r")$", flags
        ),
        module=re.compile(r"^module" + sp + r"(" + _NAME + r")$", flags),
        submodule=re.compile(
            r"^submodule\s*\([^)]*\)\s*(" + _NAME + r")$", flags
        ),
        interface=re.compile(
            r"^(?:abstract" + sp + r")?interface" + bound, flags
        ),
        type_def=re.compile(
            r"^type\s*(?:,[^:]*)?::\s*" + _NAME + r"|"
            r"^type"
            + sp
            + r"(?!is\b)(?!guard\b)"
            + _NAME
            + r"\s*(?:"
            + _PAREN
            + r")?$",
            flags,
        ),
        enum=re.compile(r"^enum\b\s*(?:,.*)?$", flags),
        end=re.compile(r"^end\s*(" + _ROUTINE_END + r")" + bound, flags),
        bare_end=re.compile(r"^end$", flags),
    )


_FREE = _build_patterns(r"\s+")
_FIXED = _build_patterns(r"\s*")


@dataclasses.dataclass(frozen=True)
class Routine:
    """One routine definition found in a Fortran source file.

    Attributes:
        name: Routine name as written in the source (case preserved).
        kind: ``subroutine``, ``function``, ``program``, ``blockdata``, or
            ``procedure`` (a submodule ``module procedure`` definition).
        parent: Name of the enclosing routine for a contained procedure, or
            an empty string at module or file level.
        first: First physical line of the defining statement (1-based).
        last: Last physical line of the closing statement (1-based).
    """

    name: str
    kind: str
    parent: str
    first: int
    last: int

    @property
    def lines(self) -> int:
        """Return the number of physical lines the routine spans."""
        return self.last - self.first + 1


@dataclasses.dataclass(frozen=True)
class ParseResult:
    """Routines and diagnostics for one parsed Fortran source file.

    Attributes:
        routines: Routine definitions sorted by ``(first, last)``.
        total_lines: Number of physical lines in the file.
        outside_lines: Physical lines not inside any top-level routine.
        warnings: Parser diagnostics (unbalanced or unterminated scopes).
    """

    routines: tuple[Routine, ...]
    total_lines: int
    outside_lines: int
    warnings: tuple[str, ...]


@dataclasses.dataclass
class _Scope:
    """One entry of the parser's scope stack."""

    kind: str
    name: str
    first: int
    record: bool


def _strip_comment(text: str) -> str:
    """Remove a trailing ``!`` comment that is not inside a string."""
    quote = ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return text[:i]
    return text


def _split_semicolons(text: str) -> list[str]:
    """Split ``text`` at ``;`` characters that are outside strings."""
    parts: list[str] = []
    quote = ""
    start = 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == ";":
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def detect_fixed_form(suffix: str, lines: list[str]) -> bool:
    """Decide whether a source file is fixed-form Fortran.

    A fixed-form suffix is treated as fixed form unless the file uses
    free-form continuation (``&`` at line end) and has no column-1 comment
    markers. Other suffixes are free form unless they carry column-1 ``c``,
    ``C``, or ``*`` comment lines and no free-form continuation.

    Args:
        suffix: Lowercase file suffix including the dot.
        lines: Physical source lines.

    Returns:
        ``True`` for fixed-form source.
    """
    col1 = sum(
        1
        for ln in lines
        if ln[:1] in {"c", "C", "*"} and (len(ln) == 1 or not ln[1].isalnum())
    )
    amp = sum(1 for ln in lines if _strip_comment(ln).rstrip().endswith("&"))
    if suffix in FIXED_FORM_SUFFIXES:
        return not (amp > 0 and col1 == 0)
    return col1 > 0 and amp == 0


def _logical_statements_free(
    lines: list[str], is_fypp: bool
) -> list[tuple[int, int, str]]:
    """Join free-form continuation lines into ``(first, last, text)``."""
    out: list[tuple[int, int, str]] = []
    buf = ""
    first = 0
    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("#") or (
            is_fypp and stripped.startswith(("$:", "@:"))
        ):
            cond = _COND_RE.match(stripped)
            if cond is not None and not buf:
                word = cond.group(1).lower()
                kind = {"endif": "#endif", "else": "#else", "elif": "#else"}
                out.append((idx, idx, kind.get(word, "#if")))
            if not buf:
                continue
        text = _strip_comment(raw).strip()
        if not text:
            continue
        if buf:
            if text.startswith("&"):
                text = text[1:].lstrip()
            buf += " " + text
        else:
            buf = text
            first = idx
        if buf.endswith("&"):
            buf = buf[:-1].rstrip()
            continue
        out.append((first, idx, buf))
        buf = ""
    if buf:
        out.append((first, len(lines), buf))
    return out


def _logical_statements_fixed(
    lines: list[str],
) -> list[tuple[int, int, str]]:
    """Join fixed-form continuation lines into ``(first, last, text)``."""
    out: list[tuple[int, int, str]] = []
    first = last = 0
    text = ""
    open_stmt = False
    for idx, raw in enumerate(lines, start=1):
        if not raw.strip() or raw[0] in "cC*!dD#":
            continue
        if raw[0] == "\t":
            # DEC tab format: the statement follows the tab; a digit 1-9
            # first after the tab marks a continuation line.
            is_cont = raw[1:2] in set("123456789")
            body = _strip_comment(raw[2:] if is_cont else raw[1:]).strip()
        else:
            body = _strip_comment(raw[6:]).strip() if len(raw) > 6 else ""
            is_cont = (
                len(raw) > 5 and raw[5] not in " 0" and not raw[:5].strip()
            )
        if is_cont and open_stmt:
            last = idx
            text = f"{text} {body}"
            continue
        if not body:
            continue
        if open_stmt:
            out.append((first, last, text))
        first = last = idx
        text = body
        open_stmt = True
    if open_stmt:
        out.append((first, last, text))
    return out


def _match_start(
    pat: _Patterns, stmt: str, in_interface: bool
) -> tuple[str, str, bool] | None:
    """Classify a statement that opens a routine, interface, or unit.

    Returns:
        ``(scope_kind, name, is_routine)`` or ``None``. ``scope_kind`` is a
        routine kind, ``interface``, ``type``, ``enum``, or ``module``.
    """
    if (m := pat.sub.match(stmt)) is not None:
        return "subroutine", m.group(1), True
    if (m := pat.func.match(stmt)) is not None:
        return "function", m.group(1), True
    if (m := pat.module_proc.match(stmt)) is not None:
        # Inside an interface this is a name list, not a definition.
        return None if in_interface else ("procedure", m.group(1), True)
    if pat.interface.match(stmt):
        return "interface", "", False
    if (m := pat.program.match(stmt)) is not None:
        return "program", m.group(1), True
    if (m := pat.block_data.match(stmt)) is not None:
        return "blockdata", m.group(1) or "", True
    if (m := pat.submodule.match(stmt)) is not None:
        return "module", m.group(1), False
    if (m := pat.module.match(stmt)) is not None:
        return "module", m.group(1), False
    if pat.enum.match(stmt):
        return "enum", "", False
    if pat.type_def.match(stmt):
        return "type", "", False
    return None


_END_KIND = {
    "subroutine": "subroutine",
    "function": "function",
    "program": "program",
    "module": "module",
    "submodule": "module",
    "interface": "interface",
    "type": "type",
    "procedure": "procedure",
    "enum": "enum",
    "blockdata": "blockdata",
}
_ROUTINE_KINDS = frozenset(
    {"subroutine", "function", "program", "blockdata", "procedure"}
)


def parse_fortran(text_lines: list[str], suffix: str) -> ParseResult:
    """Parse routine definitions out of Fortran source lines.

    Args:
        text_lines: Physical source lines (without line terminators).
        suffix: Lowercase file suffix including the dot, used to pick the
            fixed-form or free-form reader and to enable ``fypp`` handling.

    Returns:
        Routines with spans, the number of lines outside routines, and any
        parser warnings.
    """
    suffix = suffix.lower()
    fixed = detect_fixed_form(suffix, text_lines)
    pat = _FIXED if fixed else _FREE
    if fixed:
        stmts = _logical_statements_fixed(text_lines)
    else:
        stmts = _logical_statements_free(text_lines, suffix == ".fypp")

    stack: list[_Scope] = []
    found: list[Routine] = []
    warnings: list[str] = []

    def in_interface() -> bool:
        return any(s.kind == "interface" for s in stack)

    def enclosing_routine() -> str:
        for s in reversed(stack):
            if s.record and s.kind in _ROUTINE_KINDS:
                return s.name
        return ""

    unit_first = [0]

    def close(kind: str | None, last: int, line: int) -> None:
        if not stack:
            if kind is None and unit_first[0]:
                # A main program with no ``program`` statement.
                found.append(Routine("", "program", "", unit_first[0], last))
                unit_first[0] = 0
                return
            warnings.append(f"line {line}: end with no open scope")
            return
        if kind is not None:
            for j in range(len(stack) - 1, -1, -1):
                if stack[j].kind == kind:
                    if j != len(stack) - 1:
                        warnings.append(
                            f"line {line}: end {kind} closes "
                            f"{len(stack) - 1 - j} unclosed inner scope(s)"
                        )
                    del stack[j + 1 :]
                    break
            else:
                warnings.append(f"line {line}: end {kind} with no open {kind}")
                return
        elif stack[-1].kind in {"interface", "type", "enum"}:
            warnings.append(f"line {line}: bare end inside {stack[-1].kind}")
            return
        scope = stack.pop()
        if not stack:
            unit_first[0] = 0
        if scope.record:
            found.append(
                Routine(
                    scope.name,
                    scope.kind,
                    enclosing_routine(),
                    scope.first,
                    last,
                )
            )

    # Preprocessor conditionals: (stack depth at ``#if``, first line of the
    # discarded branch's outermost scope, or 0).
    conds: list[list[int]] = []

    for first, last, raw_text in stmts:
        if raw_text == "#if":
            conds.append([len(stack), 0])
            continue
        if raw_text == "#endif":
            if conds:
                conds.pop()
            continue
        if raw_text == "#else":
            # Alternative branches open the same scope, so drop scopes the
            # previous branch left open and let the new one inherit its start.
            if conds and len(stack) > conds[-1][0]:
                dropped = stack[conds[-1][0]]
                conds[-1][1] = (
                    min(conds[-1][1], dropped.first)
                    if conds[-1][1]
                    else dropped.first
                )
                del stack[conds[-1][0] :]
            continue
        if not stack and not unit_first[0]:
            unit_first[0] = first
        for stmt in _split_semicolons(raw_text):
            norm = "".join(stmt.split()) if fixed else " ".join(stmt.split())
            if (m := pat.end.match(norm)) is not None:
                kw = re.sub(r"\s+", "", m.group(1).lower())
                close(_END_KIND[kw], last, last)
                continue
            if pat.bare_end.match(norm):
                close(None, last, last)
                continue
            start = _match_start(pat, norm, in_interface())
            if start is None:
                continue
            kind, name, is_routine = start
            record = is_routine and not in_interface()
            depth = len(stack)
            if conds and conds[-1][1] and depth == conds[-1][0]:
                first = min(first, conds[-1][1])
                conds[-1][1] = 0
            stack.append(_Scope(kind, name, first, record))

    for scope in stack:
        warnings.append(
            f"unterminated {scope.kind} {scope.name!r} opened at line "
            f"{scope.first}"
        )

    found.sort(key=lambda r: (r.first, r.last, r.name))
    covered = covered_lines(found, len(text_lines))
    return ParseResult(
        routines=tuple(found),
        total_lines=len(text_lines),
        outside_lines=len(text_lines) - covered,
        warnings=tuple(warnings),
    )


def covered_lines(routines: list[Routine], total: int) -> int:
    """Return the number of lines inside the union of routine spans."""
    covered = 0
    reach = 0
    for r in routines:
        lo = max(r.first, reach + 1)
        hi = min(r.last, total)
        if hi >= lo:
            covered += hi - lo + 1
        reach = max(reach, r.last)
    return covered
