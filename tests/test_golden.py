# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and property tests for `flexmesa.testing.golden`.

Tests cover:
1. Fortran double-precision (`D`/`E` and bare 3-digit exponent) float parsing.
2. Section-aware `GoldenRecord` extraction across scalar, indexed, and dense
   tabular MESA `<pkg>/test/test_output` formats.
3. Dense 2D `float64` table stacking (`extract_golden_table`) and file I/O.
4. Vectorized numerical parity verification (`assert_values_match` and
   `assert_golden_records_match`) at `rtol=1e-12, atol=1e-15`.
5. Property-based round-trip invariants across arbitrary mantissas and
   exponents via `hypothesis`.
"""

from __future__ import annotations

import dataclasses
import pathlib

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st

from flexmesa.testing import golden

_SAMPLE_CONST_OUTPUT: str = (
    "  pi     3.1415926535897931E+00\n"
    "  ln10   2.3025850929940455E+00\n"
    "  sigma  5.6703744191844259E-05\n"
    "  Msun   1.9884098706980504E+33\n"
)

_SAMPLE_MATH_INTERP_OUTPUT: str = (
    " CRLIBM\n\n"
    "   Function Value (x=   0.1000000000000000E+00 )\n"
    "        log  -0.2302585092994045E+01\n"
    "      log10  -0.9999999999999999E+00\n\n"
    " test2 T\n"
    "  x val exact  1  0.0000000000000000D+00  0.0000000000000000D+00\n"
    "  x val exact  2  7.8539816339744828D-01  7.0710667730056076D-01\n"
    "  test2D_db  2  3  2.5132741228718345D+00  1.8849555921538759D+00"
    "  -2.4581494853809029D-02\n"
)

_SAMPLE_NEU_TABLE_OUTPUT: str = (
    "  6.0000000000000000E+00  6.0000000000000000E+00"
    "  1.0000000000000000E+00  1.0000000000000000E+00\n"
    "  6.0000000000000000E+00  6.0000000000000000E+00"
    "  1.0000000000000000E+00  3.5000000000000000E+00\n"
)

_SAMPLE_TURB_AUTODIFF_OUTPUT: str = (
    " ----------------------------------------------------------------\n"
    " Test MLT\n"
    " ----------------------------------------------------------------\n"
    "  vc at 1d5 Lsun    3.9654203785866620D+06\n"
    "  vc at 1d8 Lsun    3.9765811706780650D+07\n"
    "  Ratio:            1.0028145293628064D+01\n"
    "Expected ~10 because in the efficient limit vc ~ L^{1/3}\n"
    " ----------------------------------------------------------------\n"
    " Testing assignment and comparison\n"
    " Should print  3.1000000000000001D+00  0.0000000000000000D+00 :"
    "  3.1000000000000001D+00  0.0000000000000000D+00\n"
)


def test_parse_fortran_float_supports_standard_d_and_bare_exponents() -> None:
    """Verify parsing of E, D, and 3-digit bare Fortran exponents.

    Checks that standard scientific notation (`E+00`), Fortran double-precision
    notation (`D-04`), and bare 3-digit exponents (`-102`, `+105`) parse to
    IEEE-754 `float64` within `1e-15` relative accuracy, and that empty or
    overflowing (`1.0e500`) strings raise `ValueError`.
    """
    pi_str = "3.1415926535897931E+00"
    assert golden.parse_fortran_float(pi_str) == pytest.approx(np.pi, rel=1e-15)
    assert golden.parse_fortran_float(
        "-8.2630874248383526D-04"
    ) == pytest.approx(-8.2630874248383526e-4, rel=1e-15)
    bare_neg = "1.2500000000000000-102"
    assert golden.parse_fortran_float(bare_neg) == pytest.approx(
        1.25e-102, rel=1e-15
    )
    bare_pos = "-4.5000000000000000+105"
    assert golden.parse_fortran_float(bare_pos) == pytest.approx(
        -4.5e105, rel=1e-15
    )

    with pytest.raises(ValueError, match="Cannot parse empty token"):
        golden.parse_fortran_float("   ")
    with pytest.raises(ValueError, match="Non-finite"):
        golden.parse_fortran_float("1.0e500")


def test_parse_golden_text_extracts_sections_labels_and_values() -> None:
    """Verify structured record extraction across MESA output patterns.

    Checks key-value scalar lines (`const`), section headers with embedded
    numbers (`math`), indexed multi-column spline lines (`interp_1d`,
    `interp_2d`), and separator-delimited diagnostics (`turb`, `auto_diff`).
    """
    const_records = golden.parse_golden_text(_SAMPLE_CONST_OUTPUT)
    assert len(const_records) == 4
    assert const_records[0].label == "pi"
    assert not const_records[0].indices
    assert const_records[0].values[0] == pytest.approx(np.pi, rel=1e-15)

    mixed_records = golden.parse_golden_text(_SAMPLE_MATH_INTERP_OUTPUT)
    assert len(mixed_records) == 5
    expected_sec = "Function Value (x=   0.1000000000000000E+00 )"
    assert mixed_records[0].section == expected_sec
    assert mixed_records[0].label == "log"
    assert mixed_records[2].section == "test2 T"
    assert mixed_records[2].label == "x val exact"
    assert mixed_records[2].indices == (1,)
    assert mixed_records[4].label == "test2D_db"
    assert mixed_records[4].indices == (2, 3)
    assert mixed_records[4].values.shape == (3,)

    turb_records = golden.parse_golden_text(_SAMPLE_TURB_AUTODIFF_OUTPUT)
    assert len(turb_records) == 4
    assert turb_records[0].section == "Test MLT"
    assert turb_records[0].label == "vc at 1d5 Lsun"
    assert turb_records[3].values.shape == (2,)
    assert turb_records[3].values[0] == pytest.approx(3.1, rel=1e-15)


def test_extract_golden_table_and_file_roundtrip(
    tmp_path: pathlib.Path,
) -> None:
    """Verify dense table stacking and file reading from disk.

    Confirms that `parse_golden_file` and `extract_golden_table` produce a
    C-contiguous 2D `float64` array of shape `(n_records, n_cols)` and reject
    empty record sequences.
    """
    out_file = tmp_path / "test_output"
    out_file.write_text(_SAMPLE_NEU_TABLE_OUTPUT, encoding="utf-8")
    records = golden.parse_golden_file(out_file)
    assert len(records) == 2
    table = golden.extract_golden_table(records)
    assert table.shape == (2, 4)
    assert table.dtype == np.float64
    assert table[1, 3] == 3.5

    with pytest.raises(ValueError, match="Cannot stack an empty sequence"):
        golden.extract_golden_table([])


def test_assert_golden_records_match_checks_parity_and_failures() -> None:
    """Verify that 1e-14 perturbations pass and 1e-6 perturbations fail.

    Tests `assert_golden_records_match` under a Multiplicative `(1 + 1e-14)`
    perturbation (which must satisfy `rtol=1e-12`) and an additive `1e-6`
    perturbation (which must raise `AssertionError` with record context).
    """
    expected = golden.parse_golden_text(_SAMPLE_MATH_INTERP_OUTPUT)
    perturbed = [
        dataclasses.replace(rec, values=rec.values * (1.0 + 1e-14))
        for rec in expected
    ]

    summary = golden.assert_golden_records_match(
        perturbed, expected, rtol=1e-12, atol=1e-15
    )
    assert summary.num_records == 5
    assert summary.num_values == 9
    assert summary.max_rel_error <= 1e-12

    bad_records = list(expected)
    bad_vals = bad_records[1].values.copy()
    bad_vals[0] += 1e-6
    bad_records[1] = dataclasses.replace(bad_records[1], values=bad_vals)
    with pytest.raises(AssertionError, match="Golden parity check failed"):
        golden.assert_golden_records_match(bad_records, expected, rtol=1e-12)


@hypothesis.given(
    mantissa=st.floats(
        min_value=-9.99999999999999,
        max_value=9.99999999999999,
        allow_nan=False,
        allow_infinity=False,
    ),
    exponent=st.integers(min_value=-200, max_value=200),
    use_d_char=st.booleans(),
)
@hypothesis.settings(max_examples=100)
def test_fortran_float_parsing_roundtrips_arbitrary_exponents(
    mantissa: float,
    exponent: int,
    use_d_char: bool,
) -> None:
    """Property test: arbitrary Fortran D/E floats round-trip to 1e-14.

    Generates finite mantissas in `[-9.99999999999999, 9.99999999999999]` and
    base-10 exponents in `[-200, 200]` formatted with `D` or `E` and checks
    `assert_values_match` against IEEE-754 `float(...)`.
    """
    exp_char = "D" if use_d_char else "E"
    formatted = f"{mantissa:.15f}{exp_char}{exponent:+03d}"
    expected = float(f"{mantissa:.15f}e{exponent:+03d}")
    parsed = golden.parse_fortran_float(formatted)
    golden.assert_values_match(parsed, expected, rtol=1e-14, atol=1e-300)
