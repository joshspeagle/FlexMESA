# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parser and numerical parity comparator for MESA golden ``test_output`` files.

MESA's standalone module unit-test suites (``<pkg>/test/test_output*``) emit
16-digit formatted floating-point reference values (``1pe24.16``) organized into
section headers, key-value scalar lines, indexed multi-column records, and
dense numeric tables. This module routes golden text and Fortran float parsing
through the canonical Safe Rust ``flexmesa_testing`` implementation in
``flexmesa._rust`` and verifies elementwise relative and absolute numerical
parity (``<= 1e-12`` relative error by default).
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from flexmesa import _rust


@dataclasses.dataclass(frozen=True, slots=True)
class GoldenRecord:
    """Single numeric record parsed from a MESA ``test_output`` file.

    Attributes:
        line_number: 1-based line number in the source golden file.
        section: Active section header preceding this record (empty string if
            none).
        label: Text label preceding the integer indices and floating-point
            values on the line (empty string for unlabeled table rows).
        indices: Tuple of integer indices preceding the floating-point values
            on the line (for example, ``(1, 2)`` in ``test2D_db 1 2 ...``).
        values: 1D ``float64`` NumPy array of shape ``(n_values,)`` containing
            the parsed IEEE-754 double-precision reference values (dimensionless
            or in the physical CGS/solar units of the corresponding MESA test).
    """

    line_number: int
    section: str
    label: str
    indices: tuple[int, ...]
    values: npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class GoldenComparisonSummary:
    """Summary metrics from comparing two sequences of golden records.

    Attributes:
        num_records: Total number of ``GoldenRecord`` entries compared.
        num_values: Total number of scalar ``float64`` values compared across
            all records.
        max_abs_error: Maximum elementwise absolute error
            ``|actual - expected|`` across all compared values.
        max_rel_error: Maximum elementwise relative error
            ``|actual - expected| / max(|expected|, atol)`` across all values.
    """

    num_records: int
    num_values: int
    max_abs_error: float
    max_rel_error: float


def is_fortran_float_token(token: str) -> bool:
    """Return whether ``token`` is a Fortran floating-point literal.

    Delegates to ``flexmesa_testing::is_fortran_float_token`` via
    ``flexmesa._rust``.

    Args:
        token: Whitespace-delimited string token from a MESA output line.

    Returns:
        ``True`` if ``token`` contains a decimal point or exponent in standard
        or Fortran (``D``/``d`` or bare 3-digit signed exponent) notation.

    Examples:
        >>> from flexmesa.testing import golden
        >>> golden.is_fortran_float_token("8.2630874248383526D-04")
        True
        >>> golden.is_fortran_float_token("42")
        False
    """
    return _rust.is_fortran_float_token(token)


def parse_fortran_float(token: str) -> float:
    """Parse a Fortran or IEEE-754 floating-point string into a ``float64``.

    Delegates to ``flexmesa_testing::parse_fortran_float`` via
    ``flexmesa._rust``.

    Args:
        token: String representation of a floating-point number.

    Returns:
        Parsed Python ``float`` (IEEE-754 binary64).

    Raises:
        ValueError: If ``token`` cannot be parsed as a finite floating-point
            number.

    Examples:
        >>> from flexmesa.testing import golden
        >>> golden.parse_fortran_float("2.3025850929940455D+00")
        2.3025850929940455
        >>> golden.parse_fortran_float("-1.5000000000000000-100")
        -1.5e-100
    """
    if not token.strip():
        raise ValueError("Cannot parse empty token as a Fortran float.")
    try:
        return _rust.parse_fortran_float(token)
    except ValueError as exc:
        raise ValueError(
            f"Non-finite or invalid Fortran float in {token!r}: {exc}"
        ) from exc


def parse_golden_text(text: str) -> list[GoldenRecord]:
    """Parse the text of a MESA ``<pkg>/test/test_output`` file into records.

    Delegates to ``flexmesa_testing::parse_golden_text`` via
    ``flexmesa._rust``.

    Args:
        text: Full UTF-8 contents of a MESA ``test_output`` file.

    Returns:
        Ordered list of ``GoldenRecord`` entries extracted from ``text``.

    Raises:
        ValueError: If a trailing numeric token is non-finite (``NaN``, ``Inf``,
            or exponent overflow).

    Examples:
        >>> from flexmesa.testing import golden
        >>> sample = " test1 T\\n  f(2, 1)  8.2630874248383526D-04\\n"
        >>> records = golden.parse_golden_text(sample)
        >>> len(records)
        1
        >>> records[0].section, records[0].label
        ('test1 T', 'f(2, 1)')
    """
    try:
        raw = _rust.parse_golden_text(text)
    except ValueError as exc:
        raise ValueError(f"Non-finite or invalid golden text: {exc}") from exc
    return [
        GoldenRecord(
            line_number=line_no,
            section=section,
            label=label,
            indices=tuple(indices),
            values=values,
        )
        for line_no, section, label, indices, values in raw
    ]


def parse_golden_file(path: str | pathlib.Path) -> list[GoldenRecord]:
    """Read and parse a MESA ``<pkg>/test/test_output`` file from disk.

    Args:
        path: Path to the golden ``test_output`` file.

    Returns:
        Ordered list of ``GoldenRecord`` entries extracted from ``path``.
    """
    return parse_golden_text(pathlib.Path(path).read_text(encoding="utf-8"))


def extract_golden_table(
    records: Sequence[GoldenRecord],
) -> npt.NDArray[np.float64]:
    """Stack uniform-width ``GoldenRecord`` entries into a 2D ``float64`` array.

    Args:
        records: Non-empty sequence of ``GoldenRecord`` objects whose ``values``
            vectors all have the same length ``n_cols``.

    Returns:
        2D C-contiguous ``float64`` array of shape ``(n_records, n_cols)``.

    Raises:
        ValueError: If ``records`` is empty or records have differing column
            counts.
    """
    if not records:
        raise ValueError(
            "Cannot stack an empty sequence of GoldenRecord items."
        )
    first_len = records[0].values.size
    for rec in records:
        if rec.values.size != first_len:
            raise ValueError(
                f"Inconsistent column count at line {rec.line_number}: "
                f"expected {first_len}, got {rec.values.size}."
            )
    return np.ascontiguousarray(
        np.vstack([rec.values for rec in records]), dtype=np.float64
    )


def _validate_match_inputs(
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    *,
    rtol: float,
    atol: float,
    context: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate tolerances, array shapes, and finiteness for parity checks."""
    if (
        not (math.isfinite(rtol) and math.isfinite(atol))
        or rtol < 0.0
        or atol <= 0.0
    ):
        raise ValueError(
            f"Tolerances must be finite with rtol >= 0 and atol > 0; "
            f"got rtol={rtol}, atol={atol}."
        )
    act_arr = np.asarray(actual, dtype=np.float64)
    exp_arr = np.asarray(expected, dtype=np.float64)
    ctx_suffix = f" in {context}" if context else ""
    if act_arr.shape != exp_arr.shape:
        raise ValueError(
            f"Shape mismatch{ctx_suffix}: "
            f"actual {act_arr.shape} vs expected {exp_arr.shape}."
        )
    if not (np.all(np.isfinite(act_arr)) and np.all(np.isfinite(exp_arr))):
        raise AssertionError(f"Non-finite values encountered{ctx_suffix}.")
    return (act_arr, exp_arr)


def assert_values_match(
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-15,
    context: str = "",
) -> tuple[float, float]:
    """Assert elementwise parity between two ``float64`` arrays or scalars.

    For each element pair ``(a, e)``, checks ``|a - e| <= atol + rtol * |e|``
    and reports the maximum absolute and relative errors.

    Args:
        actual: Computed array or scalar of shape ``(*shape,)``.
        expected: Reference golden array or scalar of shape ``(*shape,)``.
        rtol: Maximum permitted relative error (default ``1e-12``).
        atol: Absolute error floor for near-zero values (default ``1e-15``).
        context: Optional diagnostic label included in failure messages.

    Returns:
        Tuple ``(max_abs_error, max_rel_error)`` in ``float64``.

    Raises:
        ValueError: If ``rtol < 0``, ``atol <= 0``, non-finite tolerances, or
            shape mismatch.
        AssertionError: If any element contains ``NaN``/``Inf`` or exceeds the
            tolerance bound.
    """
    act_arr, exp_arr = _validate_match_inputs(
        actual, expected, rtol=rtol, atol=atol, context=context
    )
    if act_arr.size == 0:
        return (0.0, 0.0)

    abs_exp = np.abs(exp_arr)
    abs_err = np.abs(act_arr - exp_arr)
    rel_err = abs_err / np.maximum(abs_exp, atol)
    allowed = atol + rtol * abs_exp

    if np.any(abs_err > allowed):
        bad_idx = int(np.argmax(abs_err - allowed))
        prefix = f"[{context}] " if context else ""
        raise AssertionError(
            f"{prefix}Golden parity check failed at flat index {bad_idx}: "
            f"actual={float(act_arr.flat[bad_idx]):.16e}, "
            f"expected={float(exp_arr.flat[bad_idx]):.16e}, "
            f"abs_err={float(abs_err.flat[bad_idx]):.3e}, "
            f"rel_err={float(rel_err.flat[bad_idx]):.3e} "
            f"(rtol={rtol:.1e}, atol={atol:.1e})."
        )

    return (float(np.max(abs_err)), float(np.max(rel_err)))


def _verify_record_pair_metadata(
    idx: int, act_rec: GoldenRecord, exp_rec: GoldenRecord
) -> None:
    """Verify metadata and value lengths for one ``GoldenRecord`` pair."""
    if (
        act_rec.label != exp_rec.label
        or act_rec.indices != exp_rec.indices
        or act_rec.values.shape != exp_rec.values.shape
    ):
        raise AssertionError(
            f"Record {idx} metadata mismatch at line {exp_rec.line_number} "
            f"({exp_rec.section!r}): "
            f"actual=({act_rec.label!r}, {act_rec.indices}, "
            f"len={act_rec.values.size}) vs "
            f"expected=({exp_rec.label!r}, {exp_rec.indices}, "
            f"len={exp_rec.values.size})."
        )


def assert_golden_records_match(
    actual: Sequence[GoldenRecord],
    expected: Sequence[GoldenRecord],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> GoldenComparisonSummary:
    """Compare two sequences of ``GoldenRecord`` objects for numerical parity.

    Args:
        actual: Computed sequence of ``GoldenRecord`` entries.
        expected: Reference sequence of ``GoldenRecord`` entries.
        rtol: Maximum permitted relative error (default ``1e-12``).
        atol: Absolute error floor for near-zero values (default ``1e-15``).

    Returns:
        ``GoldenComparisonSummary`` reporting record/value counts and peak
        errors.

    Raises:
        AssertionError: If record counts, labels, indices, value vector lengths,
            or numeric tolerances fail to match.
    """
    if len(actual) != len(expected):
        raise AssertionError(
            f"Golden record count mismatch: actual {len(actual)} vs "
            f"expected {len(expected)}."
        )

    total_values = 0
    max_abs = 0.0
    max_rel = 0.0

    for idx, (act_rec, exp_rec) in enumerate(
        zip(actual, expected, strict=True)
    ):
        _verify_record_pair_metadata(idx, act_rec, exp_rec)
        ctx = (
            f"line {exp_rec.line_number} "
            f"({exp_rec.section or 'root'}: {exp_rec.label or 'row'} "
            f"{exp_rec.indices})"
        )
        rec_abs, rec_rel = assert_values_match(
            act_rec.values, exp_rec.values, rtol=rtol, atol=atol, context=ctx
        )
        total_values += int(exp_rec.values.size)
        max_abs = max(max_abs, rec_abs)
        max_rel = max(max_rel, rec_rel)

    return GoldenComparisonSummary(
        num_records=len(expected),
        num_values=total_values,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )


compare_golden_records = assert_golden_records_match


__all__ = [
    "GoldenComparisonSummary",
    "GoldenRecord",
    "assert_golden_records_match",
    "assert_values_match",
    "compare_golden_records",
    "extract_golden_table",
    "is_fortran_float_token",
    "parse_fortran_float",
    "parse_golden_file",
    "parse_golden_text",
]
