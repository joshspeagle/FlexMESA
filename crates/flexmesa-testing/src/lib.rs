// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden baseline parser and numerical comparison harness for `FlexMESA`.
//!
//! MESA's standalone module unit-test suites (`<pkg>/test/test_output*`) emit
//! 16-digit formatted floating-point reference values (`1pe24.16`) organized
//! into section headers, key-value scalar lines, indexed multi-column records,
//! and dense numeric tables. This crate parses those files into structured
//! [`GoldenRecord`] entries and verifies elementwise relative and absolute
//! numerical parity (`<= 1e-12` relative error by default) in 100% Safe Rust.

use std::fmt;
use std::fs;
use std::path::Path;

/// Single numeric record parsed from a MESA `test_output` file.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenRecord {
    /// 1-based line number in the source golden file.
    pub line_number: usize,
    /// Active section header preceding this record (empty if none).
    pub section: String,
    /// Text label preceding the integer indices and floating-point values.
    pub label: String,
    /// Integer indices preceding the floating-point values on the line.
    pub indices: Vec<i64>,
    /// Parsed IEEE-754 binary64 (`f64`) reference values.
    pub values: Vec<f64>,
}

/// Summary metrics from comparing two slices of [`GoldenRecord`] entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenComparisonSummary {
    /// Total number of [`GoldenRecord`] entries compared.
    pub num_records: usize,
    /// Total number of scalar `f64` values compared across all records.
    pub num_values: usize,
    /// Maximum elementwise absolute error `|actual - expected|`.
    pub max_abs_error: f64,
    /// Maximum elementwise relative error `|actual - expected| / max(|expected|, atol)`.
    pub max_rel_error: f64,
}

/// Error returned when parsing or comparing golden baseline records.
#[derive(Clone, Debug, PartialEq)]
pub enum GoldenError {
    /// A token could not be parsed as a finite Fortran `f64` value.
    InvalidFloatToken {
        /// Raw string token that failed parsing.
        token: String,
    },
    /// An I/O error occurred while reading a golden file from disk.
    IoError {
        /// Human-readable error description.
        message: String,
    },
    /// Two value slices or record lists failed the numerical parity check.
    ParityMismatch {
        /// Detailed diagnostic message pinpointing the failing record and value.
        message: String,
    },
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFloatToken { token } => {
                write!(f, "invalid or non-finite Fortran float token: {token:?}")
            }
            Self::IoError { message } => write!(f, "I/O error: {message}"),
            Self::ParityMismatch { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for GoldenError {}

/// Return whether `token` is an integer token (`^[+-]?\d+$`).
#[must_use]
pub fn is_integer_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let start = usize::from(bytes[0] == b'+' || bytes[0] == b'-');
    if start >= bytes.len() {
        return false;
    }
    bytes[start..].iter().all(u8::is_ascii_digit)
}

/// Return whether `token` is a Fortran floating-point literal (not an integer).
///
/// # Examples
/// ```
/// use flexmesa_testing::is_fortran_float_token;
///
/// assert!(is_fortran_float_token("8.2630874248383526D-04"));
/// assert!(is_fortran_float_token("-1.5000000000000000-100"));
/// assert!(!is_fortran_float_token("42"));
/// ```
#[must_use]
pub fn is_fortran_float_token(token: &str) -> bool {
    if is_integer_token(token) {
        return false;
    }
    if !(token.contains('.')
        || token.contains('e')
        || token.contains('E')
        || token.contains('d')
        || token.contains('D'))
    {
        return false;
    }
    parse_fortran_float(token).is_ok()
}

/// Normalize bare 3-digit Fortran exponents (such as `1.25-100`) by inserting `'e'`.
fn normalize_fortran_token(trimmed: &str) -> String {
    let mut normalized = String::with_capacity(trimmed.len() + 2);
    for ch in trimmed.chars() {
        if ch == 'd' || ch == 'D' {
            normalized.push('e');
        } else {
            normalized.push(ch);
        }
    }

    if !normalized.contains('e') && !normalized.contains('E') {
        let bytes = normalized.as_bytes();
        for idx in (1..bytes.len()).rev() {
            if bytes[idx] == b'+' || bytes[idx] == b'-' {
                let exp_part = &bytes[idx + 1..];
                if (2..=3).contains(&exp_part.len()) && exp_part.iter().all(u8::is_ascii_digit) {
                    let mut with_e = String::with_capacity(normalized.len() + 1);
                    with_e.push_str(&normalized[..idx]);
                    with_e.push('e');
                    with_e.push_str(&normalized[idx..]);
                    return with_e;
                }
                break;
            }
        }
    }
    normalized
}

/// Parse a Fortran or IEEE-754 floating-point string into a finite `f64`.
///
/// Supports standard `e`/`E` scientific notation, Fortran `d`/`D` double
/// exponents, and bare 2- or 3-digit signed Fortran exponents.
///
/// # Errors
/// Returns [`GoldenError::InvalidFloatToken`] if `token` cannot be parsed as a
/// finite `f64`.
///
/// # Examples
/// ```
/// use flexmesa_testing::parse_fortran_float;
///
/// let val = parse_fortran_float("2.3025850929940455D+00").unwrap();
/// assert!((val - std::f64::consts::LN_10).abs() < 1e-15);
/// ```
pub fn parse_fortran_float(token: &str) -> Result<f64, GoldenError> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Err(GoldenError::InvalidFloatToken {
            token: token.to_owned(),
        });
    }
    if trimmed.eq_ignore_ascii_case("nan")
        || trimmed.eq_ignore_ascii_case("inf")
        || trimmed.eq_ignore_ascii_case("-inf")
        || trimmed.eq_ignore_ascii_case("+inf")
        || trimmed.eq_ignore_ascii_case("infinity")
    {
        return Err(GoldenError::InvalidFloatToken {
            token: token.to_owned(),
        });
    }

    let normalized = normalize_fortran_token(trimmed);
    let value: f64 = normalized
        .parse()
        .map_err(|_| GoldenError::InvalidFloatToken {
            token: token.to_owned(),
        })?;
    if !value.is_finite() {
        return Err(GoldenError::InvalidFloatToken {
            token: token.to_owned(),
        });
    }
    Ok(value)
}

/// Return whether a trimmed line consists solely of separator characters (`-`, `=`, `#`, `*`).
fn is_separator_line(trimmed: &str) -> bool {
    !trimmed.is_empty()
        && trimmed
            .bytes()
            .all(|b| matches!(b, b' ' | b'\t' | b'-' | b'=' | b'#' | b'*'))
}

/// Parse the text of a MESA `<pkg>/test/test_output` file into [`GoldenRecord`] entries.
///
/// # Errors
/// Returns [`GoldenError`] if a floating-point token is malformed.
///
/// # Examples
/// ```
/// use flexmesa_testing::parse_golden_text;
///
/// let sample = " test1 T\n  f(2, 1)  8.2630874248383526D-04\n";
/// let records = parse_golden_text(sample).unwrap();
/// assert_eq!(records.len(), 1);
/// assert_eq!(records[0].section, "test1 T");
/// assert_eq!(records[0].label, "f(2, 1)");
/// ```
pub fn parse_golden_text(text: &str) -> Result<Vec<GoldenRecord>, GoldenError> {
    let mut records = Vec::new();
    let mut current_section = String::new();

    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || is_separator_line(trimmed) {
            continue;
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let mut float_values = Vec::new();
        let mut cursor = tokens.len();
        while cursor > 0 && is_fortran_float_token(tokens[cursor - 1]) {
            float_values.push(parse_fortran_float(tokens[cursor - 1])?);
            cursor -= 1;
        }

        if float_values.is_empty() {
            trimmed.clone_into(&mut current_section);
            continue;
        }
        float_values.reverse();

        let mut int_indices = Vec::new();
        while cursor > 0 && is_integer_token(tokens[cursor - 1]) {
            if let Ok(parsed_int) = tokens[cursor - 1].parse::<i64>() {
                int_indices.push(parsed_int);
                cursor -= 1;
            } else {
                break;
            }
        }
        int_indices.reverse();

        let label = tokens[..cursor].join(" ");
        records.push(GoldenRecord {
            line_number,
            section: current_section.clone(),
            label,
            indices: int_indices,
            values: float_values,
        });
    }

    Ok(records)
}

/// Read and parse a MESA `<pkg>/test/test_output` file from disk.
///
/// # Errors
/// Returns [`GoldenError::IoError`] if the file cannot be read, or
/// [`GoldenError::InvalidFloatToken`] on malformed numeric tokens.
pub fn parse_golden_file(path: &Path) -> Result<Vec<GoldenRecord>, GoldenError> {
    let content = fs::read_to_string(path).map_err(|err| GoldenError::IoError {
        message: format!("{}: {err}", path.display()),
    })?;
    parse_golden_text(&content)
}

/// Compare two `f64` slices for elementwise numerical parity within `rtol` and `atol`.
///
/// # Errors
/// Returns [`GoldenError::ParityMismatch`] if lengths differ, any element is
/// non-finite, or `|a - e| > atol + rtol * |e|`.
pub fn assert_values_match(
    actual: &[f64],
    expected: &[f64],
    rtol: f64,
    atol: f64,
    context: &str,
) -> Result<(f64, f64), GoldenError> {
    if rtol < 0.0 || atol <= 0.0 {
        return Err(GoldenError::ParityMismatch {
            message: format!("invalid tolerances: rtol={rtol}, atol={atol}"),
        });
    }
    if actual.len() != expected.len() {
        return Err(GoldenError::ParityMismatch {
            message: format!(
                "[{context}] length mismatch: actual {} vs expected {}",
                actual.len(),
                expected.len()
            ),
        });
    }

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;

    for (idx, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        if !act.is_finite() || !exp.is_finite() {
            return Err(GoldenError::ParityMismatch {
                message: format!(
                    "[{context}] non-finite value at index {idx}: actual={act}, expected={exp}"
                ),
            });
        }
        let abs_err = (act - exp).abs();
        let denom = exp.abs().max(atol);
        let rel_err = abs_err / denom;
        let allowed = atol + rtol * exp.abs();

        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);

        if abs_err > allowed {
            return Err(GoldenError::ParityMismatch {
                message: format!(
                    "[{context}] golden parity check failed at index {idx}: \
                     actual={act:.16e}, expected={exp:.16e}, abs_err={abs_err:.3e}, \
                     rel_err={rel_err:.3e} (rtol={rtol:.1e}, atol={atol:.1e})"
                ),
            });
        }
    }

    Ok((max_abs, max_rel))
}

/// Compare two slices of [`GoldenRecord`] entries for metadata and numerical parity.
///
/// # Errors
/// Returns [`GoldenError::ParityMismatch`] if record counts, labels, indices,
/// or floating-point values diverge beyond `rtol` and `atol`.
pub fn assert_golden_records_match(
    actual: &[GoldenRecord],
    expected: &[GoldenRecord],
    rtol: f64,
    atol: f64,
) -> Result<GoldenComparisonSummary, GoldenError> {
    if actual.len() != expected.len() {
        return Err(GoldenError::ParityMismatch {
            message: format!(
                "record count mismatch: actual {} vs expected {}",
                actual.len(),
                expected.len()
            ),
        });
    }

    let mut num_values = 0_usize;
    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;

    for (idx, (act_rec, exp_rec)) in actual.iter().zip(expected.iter()).enumerate() {
        if act_rec.label != exp_rec.label || act_rec.indices != exp_rec.indices {
            return Err(GoldenError::ParityMismatch {
                message: format!(
                    "record {idx} metadata mismatch at line {}: \
                     actual=({:?}, {:?}) vs expected=({:?}, {:?})",
                    exp_rec.line_number,
                    act_rec.label,
                    act_rec.indices,
                    exp_rec.label,
                    exp_rec.indices
                ),
            });
        }
        let ctx = format!(
            "line {} ({}: {})",
            exp_rec.line_number, exp_rec.section, exp_rec.label
        );
        let (rec_abs, rec_rel) =
            assert_values_match(&act_rec.values, &exp_rec.values, rtol, atol, &ctx)?;
        num_values += exp_rec.values.len();
        max_abs_error = max_abs_error.max(rec_abs);
        max_rel_error = max_rel_error.max(rec_rel);
    }

    Ok(GoldenComparisonSummary {
        num_records: expected.len(),
        num_values,
        max_abs_error,
        max_rel_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const SAMPLE_MIXED: &str = "\
 CRLIBM
   Function Value (x=   0.1000000000000000E+00 )
        log  -0.2302585092994045E+01
 test2 T
                                            x val exact           1    0.0000000000000000D+00    0.0000000000000000D+00
                                              test2D_db           2           3    2.5132741228718345D+00   -2.4581494853809029D-02
";

    #[test]
    fn test_parse_fortran_float_handles_d_e_and_bare_exponents() {
        let pi = parse_fortran_float("3.1415926535897931E+00").unwrap();
        assert!((pi - std::f64::consts::PI).abs() < 1e-15);

        let d_val = parse_fortran_float("-8.2630874248383526D-04").unwrap();
        assert!((d_val - (-8.263_087_424_838_353e-4)).abs() < 1e-19);

        let bare = parse_fortran_float("1.2500000000000000-102").unwrap();
        assert!((bare - 1.25e-102).abs() < 1e-117);

        assert!(parse_fortran_float("NaN").is_err());
        assert!(parse_fortran_float("").is_err());
    }

    #[test]
    fn test_parse_and_compare_golden_records() {
        let expected = parse_golden_text(SAMPLE_MIXED).unwrap();
        assert_eq!(expected.len(), 3);
        assert_eq!(expected[0].label, "log");
        assert_eq!(expected[1].indices, vec![1]);
        assert_eq!(expected[2].indices, vec![2, 3]);

        let mut perturbed = expected.clone();
        for rec in &mut perturbed {
            for val in &mut rec.values {
                *val *= 1.0 + 1e-14;
            }
        }
        let summary = assert_golden_records_match(&perturbed, &expected, 1e-12, 1e-15).unwrap();
        assert_eq!(summary.num_records, 3);
        assert_eq!(summary.num_values, 5);
        assert!(summary.max_rel_error <= 1e-12);

        perturbed[0].values[0] += 1e-5;
        assert!(assert_golden_records_match(&perturbed, &expected, 1e-12, 1e-15).is_err());
    }

    proptest! {
        #[test]
        fn prop_fortran_d_exponent_roundtrips(
            mantissa in -9.999_999_999_f64..9.999_999_999_f64,
            exponent in -200_i32..200_i32,
        ) {
            let formatted = format!("{mantissa:.14}D{exponent:+03}");
            let expected: f64 = format!("{mantissa:.14}e{exponent:+03}").parse().unwrap();
            let parsed = parse_fortran_float(&formatted).unwrap();
            let (_, rel_err) = assert_values_match(&[parsed], &[expected], 1e-13, 1e-300, "prop").unwrap();
            prop_assert!(rel_err <= 1e-13);
        }
    }
}
