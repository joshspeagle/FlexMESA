# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parser and schema validator for legacy MESA Fortran `&namelist` (`inlist`).

Parses Fortran `&namelist ... /` blocks (`&eos`, `&kap`, `&star_job`,
`&controls`) into structured `NamelistBlock` objects and validates every
control assignment against a parsed `ControlDefault` schema.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Mapping

from flexmesa.schema import defaults

_NAMELIST_HEADER_RE: re.Pattern[str] = re.compile(
    r"^&([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$"
)
_ASSIGNMENT_LHS_RE: re.Pattern[str] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(\s*([^)]*?)\s*\))?\s*=\s*(.+)$"
)


@dataclasses.dataclass(frozen=True, slots=True)
class NamelistAssignment:
    """Single `name[(subscript)] = value` assignment inside a namelist block.

    Attributes:
        name: Control parameter name as written in the `inlist`.
        subscript: Optional array subscript string (for example, `"1"` or
            `"13"`), or `None` for scalar assignments.
        value_type: Parsed `ControlType` of the assigned literal.
        value: Parsed Python scalar value (`bool | int | float | str`).
        line_number: 1-based line number in the `inlist` source.
    """

    name: str
    subscript: str | None
    value_type: defaults.ControlType
    value: defaults.ControlValue
    line_number: int


@dataclasses.dataclass(frozen=True, slots=True)
class NamelistBlock:
    """Parsed Fortran `&namelist ... /` block from a MESA `inlist` file.

    Attributes:
        name: Namelist group name without leading `&` (for example, `"eos"`,
            `"controls"`, or `"star_job"`).
        assignments: Ordered tuple of `NamelistAssignment` entries.
    """

    name: str
    assignments: tuple[NamelistAssignment, ...]


def _split_top_level_commas(code: str) -> list[str]:
    """Split a namelist line on commas outside quotes and parentheses."""
    parts: list[str] = []
    start = 0
    in_single = False
    in_double = False
    paren_depth = 0

    for idx, ch in enumerate(code):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "," and paren_depth == 0:
                segment = code[start:idx].strip()
                if segment:
                    parts.append(segment)
                start = idx + 1

    tail = code[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _find_slash_terminator(code: str) -> int | None:
    """Return index of namelist terminator `/` outside quotes, or `None`."""
    in_single = False
    in_double = False
    for idx, ch in enumerate(code):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "/" and not in_single and not in_double:
            return idx
    return None


def _parse_assignments_in_segment(
    segment: str,
    line_idx: int,
    out: list[NamelistAssignment],
) -> None:
    """Parse comma-separated assignments from `segment` into `out`."""
    for item in _split_top_level_commas(segment):
        match = _ASSIGNMENT_LHS_RE.match(item)
        if match is None:
            raise ValueError(
                f"Malformed namelist assignment at line {line_idx}: {item!r}."
            )
        name, sub, raw_val = match.group(1), match.group(2), match.group(3)
        val_type, val = defaults.parse_control_value(raw_val)
        out.append(
            NamelistAssignment(
                name=name,
                subscript=sub,
                value_type=val_type,
                value=val,
                line_number=line_idx,
            )
        )


def parse_namelist_text(text: str) -> list[NamelistBlock]:
    """Parse one or more Fortran `&namelist ... /` blocks from `inlist` text.

    Args:
        text: Full UTF-8 contents of a MESA `inlist` file.

    Returns:
        Ordered list of `NamelistBlock` objects.

    Raises:
        ValueError: If namelist headers, assignments, or terminators (`/`) are
            malformed or unclosed.

    Examples:
        >>> from flexmesa.schema import namelist
        >>> src = "&eos\\n  use_PC = .true.\\n/\\n"
        >>> blocks = namelist.parse_namelist_text(src)
        >>> blocks[0].name, blocks[0].assignments[0].value
        ('eos', True)
    """
    blocks: list[NamelistBlock] = []
    active_name: str | None = None
    active_assignments: list[NamelistAssignment] = []

    for line_idx, raw_line in enumerate(text.splitlines(), start=1):
        code, _ = defaults.strip_fortran_comment(raw_line)
        if not code:
            continue

        if active_name is None:
            hdr = _NAMELIST_HEADER_RE.match(code)
            if hdr is None:
                raise ValueError(
                    f"Expected '&namelist' header at line {line_idx}: {code!r}."
                )
            active_name = hdr.group(1)
            code = hdr.group(2).strip()
            if not code:
                continue

        slash_idx = _find_slash_terminator(code)
        body = code if slash_idx is None else code[:slash_idx].strip()
        if body:
            _parse_assignments_in_segment(body, line_idx, active_assignments)
        if slash_idx is not None:
            blocks.append(
                NamelistBlock(
                    name=active_name,
                    assignments=tuple(active_assignments),
                )
            )
            active_name = None
            active_assignments = []

    if active_name is not None:
        raise ValueError(
            f"Unclosed namelist block '&{active_name}' (missing '/')."
        )
    return blocks


def parse_namelist_file(path: str | pathlib.Path) -> list[NamelistBlock]:
    """Read and parse a MESA Fortran `inlist` file from disk.

    Args:
        path: Path to the `inlist` file.

    Returns:
        Ordered list of `NamelistBlock` objects.
    """
    return parse_namelist_text(pathlib.Path(path).read_text(encoding="utf-8"))


def _is_type_compatible(
    expected_type: defaults.ControlType,
    actual_type: defaults.ControlType,
    *,
    strict_numeric_types: bool,
) -> bool:
    """Return whether `actual_type` is compatible with `expected_type`."""
    if expected_type == actual_type:
        return True
    if (
        expected_type == defaults.ControlType.FLOAT
        and actual_type == defaults.ControlType.INTEGER
    ):
        return True
    return (
        not strict_numeric_types
        and expected_type == defaults.ControlType.INTEGER
        and actual_type == defaults.ControlType.FLOAT
    )


def _validate_single_assignment(
    block_name: str,
    assign: NamelistAssignment,
    schema: Mapping[str, defaults.ControlDefault],
    *,
    strict_numeric_types: bool,
) -> tuple[defaults.ControlDefault, defaults.ControlValue]:
    """Validate one `NamelistAssignment` and return `(spec, coerced_value)`."""
    key = assign.name.lower()
    if key not in schema:
        raise KeyError(
            f"Unknown control {assign.name!r} in '&{block_name}' at line "
            f"{assign.line_number}."
        )
    spec = schema[key]
    if assign.subscript is not None and not spec.is_array:
        raise TypeError(
            f"Scalar control {spec.name!r} cannot be indexed with "
            f"({assign.subscript}) at line {assign.line_number}."
        )
    if not _is_type_compatible(
        spec.control_type,
        assign.value_type,
        strict_numeric_types=strict_numeric_types,
    ):
        raise TypeError(
            f"Type mismatch for control {spec.name!r} at line "
            f"{assign.line_number}: expected {spec.control_type}, "
            f"got {assign.value_type} ({assign.value!r})."
        )
    value = assign.value
    if (
        spec.control_type == defaults.ControlType.FLOAT
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        value = float(value)
    return (spec, value)


def validate_namelist_block(
    block: NamelistBlock,
    schema: Mapping[str, defaults.ControlDefault],
    *,
    strict_numeric_types: bool = False,
) -> dict[str, defaults.ControlValue | dict[str, defaults.ControlValue]]:
    """Validate a `NamelistBlock` against a `.defaults` schema.

    Args:
        block: Parsed `NamelistBlock` to validate.
        schema: Mapping of lowercase control names to `ControlDefault` entries
            (as returned by `defaults.parse_defaults_text`).
        strict_numeric_types: If `True`, reject `FLOAT` literals on controls
            whose default literal in `.defaults` was parsed as `INTEGER`.
            Defaults to `False` because MESA `.defaults` files frequently write
            `-1` or `1` for `real(dp)` controls (`Zbase = -1`,
            `initial_mass = 1`).

    Returns:
        Dictionary of validated control overrides keyed by canonical control
        name (mapping scalar controls to `ControlValue` and subscripted array
        controls to `{subscript: ControlValue}`).

    Raises:
        KeyError: If an assigned control name does not exist in `schema`.
        TypeError: If an assigned value's type or subscript is incompatible
            with the schema definition.
    """
    overrides: dict[
        str, defaults.ControlValue | dict[str, defaults.ControlValue]
    ] = {}
    for assign in block.assignments:
        spec, value = _validate_single_assignment(
            block.name,
            assign,
            schema,
            strict_numeric_types=strict_numeric_types,
        )
        if assign.subscript is not None:
            existing = overrides.get(spec.name)
            sub_map = dict(existing) if isinstance(existing, dict) else {}
            sub_map[assign.subscript] = value
            overrides[spec.name] = sub_map
        else:
            overrides[spec.name] = value
    return overrides
