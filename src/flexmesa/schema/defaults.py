# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parser for MESA ``<pkg>/defaults/*.defaults`` and ``*_columns.list`` files.

Extracts parameter names, Fortran control types (``LOGICAL``, ``INTEGER``,
``FLOAT``, ``STRING``), scalar and array default values, indexed element
overrides, section headings, and inline documentation from MESA ``.defaults``
and ``*_columns.list`` files by delegating to the canonical Safe Rust
``flexmesa_schema`` implementation in ``flexmesa._rust``.
"""

from __future__ import annotations

import dataclasses
import enum
import pathlib
import re

from flexmesa import _rust

type ControlValue = bool | int | float | str

_ASSIGNMENT_RE: re.Pattern[str] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(\s*([^()\s][^()]*?)\s*\))?\s*=\s*(.+)$"
)


class ControlType(enum.StrEnum):
    """Fortran scalar type for a MESA ``.defaults`` control parameter."""

    LOGICAL = "logical"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"


@dataclasses.dataclass(frozen=True, slots=True)
class ControlDefault:
    """Single control parameter definition parsed from a ``.defaults`` file.

    Attributes:
        name: Canonical control identifier (for example, ``use_PC`` or
            ``initial_mass``).
        control_type: Parsed ``ControlType`` (``LOGICAL``, ``INTEGER``,
            ``FLOAT``, or ``STRING``).
        default_value: Scalar or broadcast default value (in the physical CGS
            or solar units documented by the MESA control).
        is_array: Whether the control is an array parameter (declared with
            ``(:)`` or an explicit subscript in ``.defaults``).
        element_overrides: Tuple of ``(subscript, value)`` pairs for per-index
            defaults (such as ``("13", "_CO")`` on ``suffix_for_FreeEOS_Z``).
        section: Active underlined section header preceding this control.
        doc: Cleaned documentation comment block preceding the control.
        line_number: 1-based line number of the initial declaration.
    """

    name: str
    control_type: ControlType
    default_value: ControlValue
    is_array: bool
    element_overrides: tuple[tuple[str, ControlValue], ...]
    section: str
    doc: str
    line_number: int


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnEntry:
    """Single active column or directive from a MESA ``*_columns.list`` file.

    Attributes:
        name: Column identifier or directive (for example, ``model_number``,
            ``star_age``, or ``add_abundances``).
        section: Active ``!#`` or ``!##`` section heading preceding this column.
        comment: Inline trailing comment describing units or meaning.
        line_number: 1-based line number in the ``*_columns.list`` file.
    """

    name: str
    section: str
    comment: str
    line_number: int


def find_unquoted_char(text: str, target: str) -> int | None:
    """Return index of first ``target`` character outside quotes, or ``None``.

    Args:
        text: Input Fortran source or configuration line.
        target: Single-character delimiter (such as ``'!'`` or ``'/'``).

    Returns:
        0-based character index of the first unquoted ``target``, or ``None``.
    """
    if target not in text:
        return None
    if "'" not in text and '"' not in text:
        return text.find(target)
    in_single = False
    in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == target and not in_single and not in_double:
            return idx
    return None


def strip_fortran_comment(line: str) -> tuple[str, str]:
    """Split a Fortran line into ``(code_part, comment_part)`` outside quotes.

    Args:
        line: Raw single line of Fortran source or ``.defaults`` text.

    Returns:
        Tuple ``(code, comment)`` with outer whitespace stripped.
    """
    idx = find_unquoted_char(line, "!")
    if idx is None:
        return (line.strip(), "")
    return (line[:idx].strip(), line[idx + 1 :].strip())


def parse_control_value(raw: str) -> tuple[ControlType, ControlValue]:
    """Parse a Fortran ``.defaults`` or namelist literal into type and value.

    Delegates to ``flexmesa_schema::parse_control_value`` via
    ``flexmesa._rust``.

    Args:
        raw: Stripped right-hand-side literal string (for example, ``.false.``,
            ``1d-4``, ``'undefined'``, or ``170``).

    Returns:
        Tuple ``(control_type, parsed_value)``.

    Raises:
        ValueError: If ``raw`` is not a valid Fortran logical, string, 64-bit
            integer, or finite floating-point literal.

    Examples:
        >>> from flexmesa.schema import defaults
        >>> defaults.parse_control_value(".true.")
        (<ControlType.LOGICAL: 'logical'>, True)
        >>> defaults.parse_control_value("1d-4")
        (<ControlType.FLOAT: 'float'>, 0.0001)
    """
    type_str, val = _rust.parse_control_value(raw)
    return (ControlType(type_str), val)


def parse_assignment_clause(
    clause: str, line_idx: int
) -> tuple[str, str | None, ControlType, ControlValue]:
    """Parse a ``name[(subscript)] = value`` clause into its typed components.

    Args:
        clause: Single assignment expression without trailing comments.
        line_idx: 1-based line number for error reporting.

    Returns:
        Tuple ``(name, subscript, control_type, value)``.

    Raises:
        ValueError: If ``clause`` does not match ``name[(subscript)] = value``.
    """
    match = _ASSIGNMENT_RE.match(clause)
    if match is None:
        raise ValueError(
            f"Malformed control assignment at line {line_idx}: {clause!r}."
        )
    name, subscript, raw_val = match.group(1), match.group(2), match.group(3)
    ctrl_type, val = parse_control_value(raw_val)
    return (name, subscript, ctrl_type, val)


def parse_defaults_text(text: str) -> dict[str, ControlDefault]:
    """Parse the text of a MESA ``<pkg>/defaults/*.defaults`` file.

    Delegates to ``flexmesa_schema::parse_defaults_text`` via
    ``flexmesa._rust``.

    Args:
        text: Full UTF-8 contents of a MESA ``.defaults`` file.

    Returns:
        Ordered dictionary mapping lowercase control names to ``ControlDefault``
        entries.

    Raises:
        ValueError: If any non-comment line contains a malformed assignment or
            invalid literal.

    Examples:
        >>> from flexmesa.schema import defaults
        >>> sample = "! EOS\\n! ---\\n  use_PC = .false.\\n  logT1 = 7.6d0\\n"
        >>> schema = defaults.parse_defaults_text(sample)
        >>> schema["use_pc"].default_value, schema["logt1"].section
        (False, 'EOS')
    """
    try:
        raw = _rust.parse_defaults_text(text)
    except ValueError as exc:
        raise ValueError(
            f"Malformed control assignment or literal: {exc}"
        ) from exc
    return {
        name.lower(): ControlDefault(
            name=name,
            control_type=ControlType(t_str),
            default_value=def_val,
            is_array=is_arr,
            element_overrides=tuple(overrides),
            section=section,
            doc=doc,
            line_number=line_no,
        )
        for (
            name,
            t_str,
            def_val,
            is_arr,
            overrides,
            section,
            doc,
            line_no,
        ) in raw
    }


def parse_defaults_file(path: str | pathlib.Path) -> dict[str, ControlDefault]:
    """Read and parse a MESA ``<pkg>/defaults/*.defaults`` file from disk.

    Args:
        path: Path to the ``.defaults`` file.

    Returns:
        Ordered dictionary mapping lowercase control names to ``ControlDefault``
        objects.
    """
    return parse_defaults_text(pathlib.Path(path).read_text(encoding="utf-8"))


def parse_columns_text(text: str) -> list[ColumnEntry]:
    """Parse active column declarations from a MESA ``*_columns.list`` file.

    Delegates to ``flexmesa_schema::parse_columns_text`` via
    ``flexmesa._rust``.

    Args:
        text: Full UTF-8 contents of a ``history_columns.list`` or
            ``profile_columns.list`` file.

    Returns:
        Ordered list of active ``ColumnEntry`` declarations (preserving first
        occurrence order when duplicates appear, matching MESA's rule).

    Examples:
        >>> from flexmesa.schema import defaults
        >>> sample = "!# general\\n  model_number ! count\\n  !star_age_sec\\n"
        >>> cols = defaults.parse_columns_text(sample)
        >>> len(cols), cols[0].name, cols[0].section
        (1, 'model_number', 'general')
    """
    return [
        ColumnEntry(
            name=name,
            section=section,
            comment=comment,
            line_number=line_no,
        )
        for name, section, comment, line_no in _rust.parse_columns_text(text)
    ]


def parse_columns_file(path: str | pathlib.Path) -> list[ColumnEntry]:
    """Read and parse a MESA ``*_columns.list`` file from disk.

    Args:
        path: Path to the ``*_columns.list`` file.

    Returns:
        Ordered list of active ``ColumnEntry`` declarations.
    """
    return parse_columns_text(pathlib.Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ColumnEntry",
    "ControlDefault",
    "ControlType",
    "ControlValue",
    "find_unquoted_char",
    "parse_assignment_clause",
    "parse_columns_file",
    "parse_columns_text",
    "parse_control_value",
    "parse_defaults_file",
    "parse_defaults_text",
    "strip_fortran_comment",
]
