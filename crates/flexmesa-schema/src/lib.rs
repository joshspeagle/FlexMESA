// SPDX-License-Identifier: LGPL-3.0-or-later
//! Declarative MESA `.defaults` and `*_columns.list` schema parser and
//! Fortran `&namelist` (`inlist`) validator for `FlexMESA`.
//!
//! Parses upstream MESA `<pkg>/defaults/*.defaults` and `*_columns.list`
//! files into typed [`ControlDefault`] and [`ColumnEntry`] structures, and
//! validates legacy Fortran `&namelist ... /` configurations against those
//! schemas in 100% Safe Rust.

pub mod namelist;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;

use flexmesa_testing::{is_integer_token, parse_fortran_float};

pub use namelist::{
    parse_namelist_file, parse_namelist_text, validate_namelist_block,
    NamelistAssignment, NamelistBlock,
};

/// Fortran scalar control type in a MESA `.defaults` or `inlist` file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlType {
    /// Fortran `logical` (`.true.` or `.false.`).
    Logical,
    /// Fortran `integer` (`i64`).
    Integer,
    /// Fortran `real(dp)` (`f64`).
    Float,
    /// Fortran `character(len=*)` (`String`).
    String,
}

/// Parsed scalar control value from a MESA `.defaults` or `inlist` file.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlValue {
    /// Boolean control value.
    Logical(bool),
    /// 64-bit signed integer control value.
    Integer(i64),
    /// IEEE-754 double-precision (`f64`) control value.
    Float(f64),
    /// String control value.
    String(String),
}

impl ControlValue {
    /// Return the [`ControlType`] corresponding to this [`ControlValue`].
    #[must_use]
    pub const fn control_type(&self) -> ControlType {
        match self {
            Self::Logical(_) => ControlType::Logical,
            Self::Integer(_) => ControlType::Integer,
            Self::Float(_) => ControlType::Float,
            Self::String(_) => ControlType::String,
        }
    }
}

/// Single control parameter definition parsed from a MESA `.defaults` file.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlDefault {
    /// Canonical control identifier (for example, `use_PC`).
    pub name: String,
    /// Parsed [`ControlType`] of the default literal.
    pub control_type: ControlType,
    /// Default scalar or broadcast value.
    pub default_value: ControlValue,
    /// Whether the control is declared as an array (`(:)` or `(idx)`).
    pub is_array: bool,
    /// Per-index element overrides `(subscript, value)`.
    pub element_overrides: Vec<(String, ControlValue)>,
    /// Active underlined section heading preceding this control.
    pub section: String,
    /// Documentation comment block preceding this control.
    pub doc: String,
    /// 1-based line number in the `.defaults` file.
    pub line_number: usize,
}

/// Single active column or directive from a MESA `*_columns.list` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnEntry {
    /// Column name or directive (for example, `model_number` or `star_age`).
    pub name: String,
    /// Active `!#` or `!##` section heading preceding this column.
    pub section: String,
    /// Inline trailing comment describing units or meaning.
    pub comment: String,
    /// 1-based line number in the `*_columns.list` file.
    pub line_number: usize,
}

/// Error returned when parsing or validating `.defaults` or `inlist` files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaError {
    /// An I/O error occurred while reading a file from disk.
    IoError {
        /// Human-readable I/O error message.
        message: String,
    },
    /// A `.defaults` or `inlist` line could not be parsed.
    ParseError {
        /// 1-based line number where parsing failed.
        line_number: usize,
        /// Description of the syntax or literal error.
        message: String,
    },
    /// A namelist assignment referenced an unknown control name.
    UnknownControl {
        /// Namelist block name.
        namelist: String,
        /// Unrecognized control name.
        control: String,
        /// 1-based line number of the assignment.
        line_number: usize,
    },
    /// A namelist assignment violated the control's type or dimensionality.
    ValidationError {
        /// Control parameter name.
        control: String,
        /// 1-based line number of the assignment.
        line_number: usize,
        /// Detailed reason for the type or subscript failure.
        message: String,
    },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError { message } => write!(f, "I/O error: {message}"),
            Self::ParseError {
                line_number,
                message,
            } => write!(f, "line {line_number}: {message}"),
            Self::UnknownControl {
                namelist,
                control,
                line_number,
            } => write!(
                f,
                "unknown control {control:?} in '&{namelist}' at line \
                 {line_number}"
            ),
            Self::ValidationError {
                control,
                line_number,
                message,
            } => {
                write!(
                    f,
                    "control {control:?} at line {line_number}: {message}"
                )
            }
        }
    }
}

impl std::error::Error for SchemaError {}

/// Split a Fortran line into `(code_part, comment_part)` outside quotes.
#[must_use]
pub fn strip_fortran_comment(line: &str) -> (&str, &str) {
    let mut in_single = false;
    let mut in_double = false;
    for (idx, ch) in line.char_indices() {
        if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == '!' && !in_single && !in_double {
            return (line[..idx].trim(), line[idx + 1..].trim());
        }
    }
    (line.trim(), "")
}

/// Parse a Fortran `.defaults` or namelist literal into a [`ControlValue`].
///
/// # Errors
/// Returns [`SchemaError::ParseError`] if `raw` is not a valid logical, string,
/// integer, or finite floating-point literal.
///
/// # Examples
/// ```
/// use flexmesa_schema::{parse_control_value, ControlValue};
///
/// assert_eq!(
///     parse_control_value(".true.", 1).unwrap(),
///     ControlValue::Logical(true)
/// );
/// ```
pub fn parse_control_value(
    raw: &str,
    line_number: usize,
) -> Result<ControlValue, SchemaError> {
    let tok = raw.trim();
    if tok.is_empty() {
        return Err(SchemaError::ParseError {
            line_number,
            message: "empty control value literal".to_owned(),
        });
    }
    if tok.eq_ignore_ascii_case(".true.") || tok.eq_ignore_ascii_case("t") {
        return Ok(ControlValue::Logical(true));
    }
    if tok.eq_ignore_ascii_case(".false.") || tok.eq_ignore_ascii_case("f") {
        return Ok(ControlValue::Logical(false));
    }
    let bytes = tok.as_bytes();
    if bytes.len() >= 2
        && bytes[0] == bytes[bytes.len() - 1]
        && (bytes[0] == b'\'' || bytes[0] == b'"')
    {
        let q = char::from(bytes[0]);
        let doubled = format!("{q}{q}");
        let inner = &tok[1..tok.len() - 1];
        if inner.replace(&doubled, "").contains(q) {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("unescaped quote in string literal: {tok}"),
            });
        }
        return Ok(ControlValue::String(
            inner.replace(&doubled, &q.to_string()),
        ));
    }
    if is_integer_token(tok) {
        if let Ok(int_val) = tok.parse::<i64>() {
            return Ok(ControlValue::Integer(int_val));
        }
    }
    let float_val =
        parse_fortran_float(tok).map_err(|_| SchemaError::ParseError {
            line_number,
            message: format!("invalid Fortran control literal: {tok:?}"),
        })?;
    Ok(ControlValue::Float(float_val))
}

/// Split `lhs = rhs` and parse an optional `(subscript)` on `lhs`.
pub(crate) fn split_assignment(
    item: &str,
    line_number: usize,
) -> Result<(&str, Option<&str>, &str), SchemaError> {
    let Some((lhs_raw, rhs_raw)) = item.split_once('=') else {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!("expected '=' in assignment: {item:?}"),
        });
    };
    let lhs = lhs_raw.trim();
    let rhs = rhs_raw.trim();
    let (name, sub) = if let Some(open_idx) = lhs.find('(') {
        if !lhs.ends_with(')') {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("unclosed '(' in {lhs:?}"),
            });
        }
        (
            lhs[..open_idx].trim(),
            Some(lhs[open_idx + 1..lhs.len() - 1].trim()),
        )
    } else {
        (lhs, None)
    };
    if name.is_empty()
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!("invalid control identifier: {name:?}"),
        });
    }
    Ok((name, sub, rhs))
}

/// Return whether `comment` is a section/parameter underline (`===`, `---`, `~~~`).
fn is_underline(comment: &str) -> bool {
    comment.len() >= 3
        && comment
            .bytes()
            .all(|b| matches!(b, b'=' | b'-' | b'~' | b'^'))
}

/// Parse the text of a MESA `<pkg>/defaults/*.defaults` file.
///
/// # Errors
/// Returns [`SchemaError::ParseError`] if any assignment line is malformed.
pub fn parse_defaults_text(
    text: &str,
) -> Result<Vec<ControlDefault>, SchemaError> {
    let mut controls: Vec<ControlDefault> = Vec::new();
    let mut index_by_key: HashMap<String, usize> = HashMap::new();
    let mut current_section = String::new();
    let mut pending_docs: Vec<&str> = Vec::new();

    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let (code, comment) = strip_fortran_comment(raw_line);
        if code.is_empty() {
            if !comment.is_empty() {
                if is_underline(comment) {
                    if comment.starts_with(['=', '-', '^']) {
                        if let Some(title) = pending_docs.pop() {
                            title.trim().clone_into(&mut current_section);
                            pending_docs.clear();
                        }
                    }
                } else if comment != "::" {
                    pending_docs.push(comment);
                }
            }
            continue;
        }
        let (name, sub, rhs) = split_assignment(code, line_number)?;
        let val = parse_control_value(rhs, line_number)?;
        let key = name.to_ascii_lowercase();
        let is_specific = sub.is_some_and(|s| s != ":");
        if let (Some(&pos), Some(sub_str)) =
            (index_by_key.get(&key), sub.filter(|_| is_specific))
        {
            controls[pos].is_array = true;
            controls[pos]
                .element_overrides
                .push((sub_str.to_owned(), val));
        } else {
            let overrides = match sub.filter(|_| is_specific) {
                Some(s) => vec![(s.to_owned(), val.clone())],
                None => Vec::new(),
            };
            index_by_key.insert(key, controls.len());
            controls.push(ControlDefault {
                name: name.to_owned(),
                control_type: val.control_type(),
                default_value: val,
                is_array: sub.is_some(),
                element_overrides: overrides,
                section: current_section.clone(),
                doc: pending_docs.join("\n"),
                line_number,
            });
        }
        pending_docs.clear();
    }
    Ok(controls)
}

/// Read and parse a MESA `<pkg>/defaults/*.defaults` file from disk.
///
/// # Errors
/// Returns [`SchemaError`] on I/O failure or malformed `.defaults` syntax.
pub fn parse_defaults_file(
    path: &Path,
) -> Result<Vec<ControlDefault>, SchemaError> {
    let content =
        fs::read_to_string(path).map_err(|err| SchemaError::IoError {
            message: format!("{}: {err}", path.display()),
        })?;
    parse_defaults_text(&content)
}

/// Parse active column declarations from a MESA `*_columns.list` file.
#[must_use]
pub fn parse_columns_text(text: &str) -> Vec<ColumnEntry> {
    let mut entries = Vec::new();
    let mut seen = HashSet::new();
    let mut current_section = String::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let (code, comment) = strip_fortran_comment(raw_line);
        if code.is_empty() {
            if comment.starts_with('#') {
                comment
                    .trim_start_matches('#')
                    .trim()
                    .clone_into(&mut current_section);
            }
            continue;
        }
        if seen.insert(code.to_ascii_lowercase()) {
            entries.push(ColumnEntry {
                name: code.to_owned(),
                section: current_section.clone(),
                comment: comment.to_owned(),
                line_number: idx + 1,
            });
        }
    }
    entries
}
