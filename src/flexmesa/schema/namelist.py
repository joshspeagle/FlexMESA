# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parser and schema validator for legacy MESA Fortran ``&namelist`` blocks.

Parses Fortran ``&namelist ... /`` blocks (``&eos``, ``&kap``, ``&star_job``,
``&controls``) into structured ``NamelistBlock`` objects by delegating to the
canonical Safe Rust ``flexmesa_schema::parse_namelist_text`` implementation in
``flexmesa._rust``, and validates every control assignment against a parsed
``ControlDefault`` schema.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping

from flexmesa import _rust
from flexmesa.schema import defaults


@dataclasses.dataclass(frozen=True, slots=True)
class NamelistAssignment:
    """Single ``name[(subscript)] = value`` assignment inside a namelist block.

    Attributes:
        name: Control parameter name as written in the ``inlist``.
        subscript: Optional array subscript string (for example, ``"1"`` or
            ``"13"``), or ``None`` for scalar assignments.
        value_type: Parsed ``ControlType`` of the assigned literal.
        value: Parsed Python scalar value (``bool | int | float | str``).
        line_number: 1-based line number in the ``inlist`` source.
    """

    name: str
    subscript: str | None
    value_type: defaults.ControlType
    value: defaults.ControlValue
    line_number: int


@dataclasses.dataclass(frozen=True, slots=True)
class NamelistBlock:
    """Parsed Fortran ``&namelist ... /`` block from a MESA ``inlist`` file.

    Attributes:
        name: Namelist group name without leading ``&`` (for example, ``"eos"``,
            ``"controls"``, or ``"star_job"``).
        assignments: Ordered tuple of ``NamelistAssignment`` entries.
    """

    name: str
    assignments: tuple[NamelistAssignment, ...]


def parse_namelist_text(text: str) -> list[NamelistBlock]:
    """Parse one or more Fortran ``&namelist ... /`` blocks from text.

    Delegates directly to ``flexmesa_schema::parse_namelist_text`` via
    ``flexmesa._rust``.

    Args:
        text: Full UTF-8 contents of a MESA ``inlist`` file.

    Returns:
        Ordered list of ``NamelistBlock`` objects.

    Raises:
        ValueError: If namelist headers, assignments, or terminators (``/``) are
            malformed or unclosed.

    Examples:
        >>> from flexmesa.schema import namelist
        >>> src = "&eos\\n  use_PC = .true.\\n/\\n"
        >>> blocks = namelist.parse_namelist_text(src)
        >>> blocks[0].name, blocks[0].assignments[0].value
        ('eos', True)
    """
    try:
        raw = _rust.parse_namelist_text(text)
    except ValueError as exc:
        msg = str(exc)
        if ": " in msg:
            prefix, rest = msg.split(": ", 1)
            msg = f"{prefix}: {rest[:1].upper()}{rest[1:]}"
        else:
            msg = f"{msg[:1].upper()}{msg[1:]}"
        raise ValueError(msg) from exc
    return [
        NamelistBlock(
            name=block_name,
            assignments=tuple(
                NamelistAssignment(
                    name=a_name,
                    subscript=subscript,
                    value_type=defaults.ControlType(t_str),
                    value=val,
                    line_number=line_no,
                )
                for a_name, subscript, t_str, val, line_no in assigns
            ),
        )
        for block_name, assigns in raw
    ]


def parse_namelist_file(path: str | pathlib.Path) -> list[NamelistBlock]:
    """Read and parse a MESA Fortran ``inlist`` file from disk.

    Args:
        path: Path to the ``inlist`` file.

    Returns:
        Ordered list of ``NamelistBlock`` objects.
    """
    return parse_namelist_text(pathlib.Path(path).read_text(encoding="utf-8"))


def _is_type_compatible(
    expected_type: defaults.ControlType,
    actual_type: defaults.ControlType,
    *,
    strict_numeric_types: bool,
) -> bool:
    """Return whether ``actual_type`` is compatible with ``expected_type``."""
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
    """Validate one ``NamelistAssignment`` into ``(spec, coerced_value)``."""
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
    """Validate a ``NamelistBlock`` against a ``.defaults`` schema.

    Args:
        block: Parsed ``NamelistBlock`` to validate.
        schema: Mapping of lowercase control names to ``ControlDefault`` entries
            (as returned by ``defaults.parse_defaults_text``).
        strict_numeric_types: If ``True``, reject ``FLOAT`` literals on controls
            whose default literal in ``.defaults`` was parsed as ``INTEGER``.
            Defaults to ``False`` because MESA ``.defaults`` files frequently
            write ``-1`` or ``1`` for ``real(dp)`` controls (``Zbase = -1``,
            ``initial_mass = 1``).

    Returns:
        Dictionary of validated control overrides keyed by canonical control
        name (mapping scalar controls to ``ControlValue`` and subscripted array
        controls to ``{subscript: ControlValue}``).

    Raises:
        KeyError: If an assigned control name does not exist in ``schema``.
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
        if assign.subscript is None:
            overrides[spec.name] = value
        else:
            existing = overrides.get(spec.name)
            if not isinstance(existing, dict):
                existing = {}
                overrides[spec.name] = existing
            existing[assign.subscript] = value

    return overrides


__all__ = [
    "NamelistAssignment",
    "NamelistBlock",
    "parse_namelist_file",
    "parse_namelist_text",
    "validate_namelist_block",
]
