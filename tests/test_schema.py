# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and property tests for `flexmesa.schema.defaults` and `namelist`.

Tests cover:
1. Fortran `.defaults` literal parsing (`LOGICAL`, `INTEGER`, `FLOAT`,
   `STRING`, escaped quotes, and inline `!` comments).
2. Full schema extraction (`parse_defaults_text`, `parse_defaults_file`) with
   underlined section headers, docstrings, and array element overrides.
3. Active column extraction from `*_columns.list` (`parse_columns_text`) with
   duplicate suppression and `!#` section headings.
4. Legacy Fortran `&namelist` (`inlist`) parsing and schema validation
   (`parse_namelist_text`, `validate_namelist_block`), including rejection of
   unknown controls, scalar indexing, and type mismatches.
5. Property-based round-trip invariants across arbitrary control literals via
   `hypothesis`.
"""

from __future__ import annotations

import pathlib

import hypothesis
import pytest
from hypothesis import strategies as st

from flexmesa.schema import defaults, namelist

_SAMPLE_EOS_DEFAULTS: str = """\
! eos module controls
! ===================

! PC EOS controls
! ---------------

   ! ``use_PC``
   ! Select whether to use the PC eos.
   ! ::

      use_PC = .false.
      mass_fraction_limit_for_PC = 1d-4
      logRho1_PC_limit = 2.999d0 ! keep < logRho2

! FreeEOS controls
! ----------------
      Zbase = -1
      Skye_min_gamma_for_solid = 170
      suffix_for_FreeEOS_Z(:) = ''
      suffix_for_FreeEOS_Z(13) = '_CO' ! Z = 0.8
      suffix_for_FreeEOS_Z(14) = '_CO'
"""

_SAMPLE_HISTORY_COLUMNS: str = """\
! history_columns.list
!# general info about the model
   model_number ! counting from the start of the run
   num_zones ! number of zones in the model
   !## age
   star_age ! elapsed simulated time in years
   !star_age_sec ! commented out column
   model_number ! duplicate should be ignored
"""

_SAMPLE_INLIST: str = """\
&eos
   use_PC = .true.
   mass_fraction_limit_for_PC = 2.5d-4, logRho1_PC_limit = 3
   Zbase = 0.02d0
   suffix_for_FreeEOS_Z(13) = '_O_NE'
/
"""


def test_parse_defaults_text_extracts_types_sections_and_arrays(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `.defaults` parsing across logicals, floats, ints, and arrays.

    Checks that section titles (`PC EOS controls`, `FreeEOS controls`),
    docstrings, broadcast array declarations (`(:)`), and element-specific
    overrides (`(13)`, `(14)`) are extracted accurately.
    """
    defaults_path = tmp_path / "eos.defaults"
    defaults_path.write_text(_SAMPLE_EOS_DEFAULTS, encoding="utf-8")
    schema = defaults.parse_defaults_file(defaults_path)

    assert len(schema) == 6
    use_pc = schema["use_pc"]
    assert use_pc.name == "use_PC"
    assert use_pc.control_type == defaults.ControlType.LOGICAL
    assert use_pc.default_value is False
    assert use_pc.section == "PC EOS controls"
    assert "Select whether to use the PC eos." in use_pc.doc

    mf_limit = schema["mass_fraction_limit_for_pc"]
    assert mf_limit.control_type == defaults.ControlType.FLOAT
    assert mf_limit.default_value == pytest.approx(1e-4, rel=1e-15)

    suffix = schema["suffix_for_freeeos_z"]
    assert suffix.control_type == defaults.ControlType.STRING
    assert suffix.is_array is True
    assert not suffix.default_value
    assert suffix.element_overrides == (("13", "_CO"), ("14", "_CO"))


def test_parse_columns_text_extracts_active_columns_and_deduplicates(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `*_columns.list` parsing, `!#` sections, and deduplication.

    Checks that commented-out columns (`!star_age_sec`) and duplicate column
    names (`model_number`) are skipped while preserving section metadata.
    """
    cols_path = tmp_path / "history_columns.list"
    cols_path.write_text(_SAMPLE_HISTORY_COLUMNS, encoding="utf-8")
    cols = defaults.parse_columns_file(cols_path)

    assert [c.name for c in cols] == ["model_number", "num_zones", "star_age"]
    assert cols[0].section == "general info about the model"
    assert cols[0].comment == "counting from the start of the run"
    assert cols[2].section == "age"


def test_parse_and_validate_namelist_against_schema(
    tmp_path: pathlib.Path,
) -> None:
    """Verify Fortran `&namelist` parsing and schema validation rules.

    Confirms valid overrides (including `int -> float` widening, `Zbase = -1`
    real override, and array subscripting) pass, while unknown controls,
    indexed scalar controls, and type mismatches raise errors.
    """
    schema = defaults.parse_defaults_text(_SAMPLE_EOS_DEFAULTS)
    inlist_path = tmp_path / "inlist_eos"
    inlist_path.write_text(_SAMPLE_INLIST, encoding="utf-8")
    blocks = namelist.parse_namelist_file(inlist_path)

    assert len(blocks) == 1
    assert blocks[0].name == "eos"
    overrides = namelist.validate_namelist_block(blocks[0], schema)
    assert overrides["use_PC"] is True
    assert overrides["mass_fraction_limit_for_PC"] == pytest.approx(2.5e-4)
    assert overrides["logRho1_PC_limit"] == pytest.approx(3.0)
    assert overrides["Zbase"] == pytest.approx(0.02)
    assert overrides["suffix_for_FreeEOS_Z"] == {"13": "_O_NE"}

    with pytest.raises(TypeError, match="Type mismatch for control 'Zbase'"):
        namelist.validate_namelist_block(
            blocks[0], schema, strict_numeric_types=True
        )

    bad_unknown = namelist.parse_namelist_text("&eos\n  no_such_ctrl = 1\n/\n")
    with pytest.raises(KeyError, match="Unknown control 'no_such_ctrl'"):
        namelist.validate_namelist_block(bad_unknown[0], schema)

    bad_index = namelist.parse_namelist_text("&eos\n  use_PC(1) = .true.\n/\n")
    with pytest.raises(TypeError, match="Scalar control 'use_PC' cannot"):
        namelist.validate_namelist_block(bad_index[0], schema)

    bad_type = namelist.parse_namelist_text("&eos\n  use_PC = 1.0d0\n/\n")
    with pytest.raises(TypeError, match="Type mismatch for control 'use_PC'"):
        namelist.validate_namelist_block(bad_type[0], schema)


def test_namelist_and_control_value_edge_cases() -> None:
    """Verify Stage B edge cases: 64-bit bounds, identifiers, and shorthands."""
    schema = defaults.parse_defaults_text(_SAMPLE_EOS_DEFAULTS)
    with pytest.raises(ValueError, match="64-bit integer overflow"):
        defaults.parse_control_value("9223372036854775808")
    with pytest.raises(ValueError, match="Unexpected trailing tokens"):
        namelist.parse_namelist_text("&eos\n  use_PC = .true. / extra\n")
    with pytest.raises(ValueError, match="Malformed control assignment"):
        defaults.parse_defaults_text("  foo() = 1\n")

    shorthand = namelist.parse_namelist_text(
        "&eos\n  use_PC = .true\n  suffix_for_FreeEOS_Z = '_A', '_B'\n/\n"
    )
    validated_short = namelist.validate_namelist_block(shorthand[0], schema)
    assert validated_short["use_PC"] is True
    assert validated_short["suffix_for_FreeEOS_Z"] == {"1": "_A", "2": "_B"}


@hypothesis.given(
    int_val=st.integers(min_value=-10_000_000, max_value=10_000_000),
    bool_val=st.booleans(),
    text_val=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-./'",
        ),
        max_size=24,
    ),
)
@hypothesis.settings(max_examples=80)
def test_control_value_parsing_roundtrips_scalars(
    int_val: int,
    bool_val: bool,
    text_val: str,
) -> None:
    """Property test: Fortran integer, logical, and string literals round-trip.

    Generates signed integers, booleans, and strings containing single quotes,
    formats them as Fortran literals, and verifies `parse_control_value`.
    """
    t_int, v_int = defaults.parse_control_value(str(int_val))
    assert t_int == defaults.ControlType.INTEGER
    assert v_int == int_val

    t_bool, v_bool = defaults.parse_control_value(
        ".true." if bool_val else ".false."
    )
    assert t_bool == defaults.ControlType.LOGICAL
    assert v_bool is bool_val

    escaped = text_val.replace("'", "''")
    t_str, v_str = defaults.parse_control_value(f"'{escaped}'")
    assert t_str == defaults.ControlType.STRING
    assert v_str == text_val
