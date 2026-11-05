// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden baseline parser and numerical comparison harness for `FlexMESA`.
//!
//! MESA's standalone module unit-test suites (`<pkg>/test/test_output*`) emit
//! 16-digit formatted floating-point reference values (`1pe24.16`) organized
//! into section headers, key-value scalar lines, indexed multi-column records,
//! and dense numeric tables. This crate parses those files into structured
//! [`GoldenRecord`] entries and verifies elementwise relative and absolute
//! numerical parity (`<= 1e-12` relative error by default) in 100% Safe Rust.

use std::borrow::Cow;
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
    /// Maximum elementwise relative error
    /// `|actual - expected| / max(|expected|, atol)`.
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
        /// Detailed diagnostic message pinpointing the failing record/value.
        message: String,
    },
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFloatToken { token } => {
                write!(f, "invalid or non-finite Fortran float: {token:?}")
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
    start < bytes.len() && bytes[start..].iter().all(u8::is_ascii_digit)
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
    try_parse_fortran_float(token).is_some()
}

/// Normalize Fortran `d`/`D` or bare 2- to 3-digit exponents into IEEE syntax.
fn normalize_fortran_token(trimmed: &str) -> Cow<'_, str> {
    let base: Cow<'_, str> = if trimmed.contains(['d', 'D']) {
        Cow::Owned(trimmed.replace(['d', 'D'], "e"))
    } else {
        Cow::Borrowed(trimmed)
    };

    if !base.contains(['e', 'E']) {
        if let Some(idx) = base.rfind(['+', '-']).filter(|&i| i > 0) {
            let exp = &base.as_bytes()[idx + 1..];
            if (2..=3).contains(&exp.len())
                && exp.iter().all(u8::is_ascii_digit)
            {
                return Cow::Owned(format!(
                    "{}e{}",
                    &base[..idx],
                    &base[idx..]
                ));
            }
        }
    }
    base
}

/// Try to parse `token` as a Fortran float literal (rejecting plain integers).
fn try_parse_fortran_float(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    let normalized = normalize_fortran_token(trimmed);
    if !normalized.contains(['.', 'e', 'E']) {
        return None;
    }
    normalized.parse::<f64>().ok().filter(|v| v.is_finite())
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
    let normalized = normalize_fortran_token(token.trim());
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

/// Return whether a trimmed line consists solely of separator characters.
fn is_separator_line(trimmed: &str) -> bool {
    !trimmed.is_empty()
        && trimmed
            .bytes()
            .all(|b| matches!(b, b' ' | b'\t' | b'-' | b'=' | b'#' | b'*'))
}

/// Parse the text of a MESA `<pkg>/test/test_output` file into records.
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
        while cursor > 0 {
            let Some(val) = try_parse_fortran_float(tokens[cursor - 1]) else {
                break;
            };
            float_values.push(val);
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

/// Stack uniform-width [`GoldenRecord`] entries into a row-major `Vec<Vec<f64>>`.
///
/// # Errors
/// Returns [`GoldenError::ParityMismatch`] if `records` is empty or records
/// have differing column counts.
pub fn extract_golden_table(
    records: &[GoldenRecord],
) -> Result<Vec<Vec<f64>>, GoldenError> {
    let Some(first) = records.first() else {
        return Err(GoldenError::ParityMismatch {
            message: "cannot stack an empty GoldenRecord slice".to_owned(),
        });
    };
    let n_cols = first.values.len();
    let mut table = Vec::with_capacity(records.len());
    for rec in records {
        if rec.values.len() != n_cols {
            return Err(GoldenError::ParityMismatch {
                message: format!(
                    "inconsistent column count at line {}: expected {n_cols}, got {}",
                    rec.line_number,
                    rec.values.len()
                ),
            });
        }
        table.push(rec.values.clone());
    }
    Ok(table)
}

/// Compare two `f64` slices for elementwise numerical parity.
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
                "[{context}] length mismatch: {} vs {}",
                actual.len(),
                expected.len()
            ),
        });
    }
    let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
    for (idx, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        if !act.is_finite() || !exp.is_finite() {
            return Err(GoldenError::ParityMismatch {
                message: format!(
                    "[{context}] non-finite at index {idx}: actual={act}, expected={exp}"
                ),
            });
        }
        let abs_exp = exp.abs();
        let abs_err = (act - exp).abs();
        let rel_err = abs_err / abs_exp.max(atol);
        let allowed = atol + rtol * abs_exp;
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

/// Compare two slices of [`GoldenRecord`] entries for metadata and parity.
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
    let (mut num_values, mut max_abs_error, mut max_rel_error) =
        (0_usize, 0.0_f64, 0.0_f64);
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
        let (rec_abs, rec_rel) =
            assert_values_match(&act_rec.values, &exp_rec.values, rtol, atol, "")
                .map_err(|err| GoldenError::ParityMismatch {
                    message: format!(
                        "line {} ({}: {}): {err}",
                        exp_rec.line_number, exp_rec.section, exp_rec.label
                    ),
                })?;
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
