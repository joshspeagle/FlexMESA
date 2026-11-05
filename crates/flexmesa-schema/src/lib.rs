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

use flexmesa_testing::{
    is_fortran_float_token, is_integer_token, parse_fortran_float,
};

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

/// Return whether `ident` is a valid Fortran identifier (`^[A-Za-z_][A-Za-z0-9_]*$`).
pub(crate) fn is_valid_identifier(ident: &str) -> bool {
    let bytes = ident.as_bytes();
    bytes
        .first()
        .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
        && bytes[1..]
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

/// Return the byte index of the first `target` char outside quotes.
pub(crate) fn find_unquoted_char(line: &str, target: char) -> Option<usize> {
    if !line.contains(target) {
        return None;
    }
    if !line.contains(['\'', '"']) {
        return line.find(target);
    }
    let (mut in_single, mut in_double) = (false, false);
    for (idx, ch) in line.char_indices() {
        if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == target && !in_single && !in_double {
            return Some(idx);
        }
    }
    None
}

/// Read UTF-8 text from `path`, mapping I/O errors to [`SchemaError::IoError`].
pub(crate) fn read_file_text(path: &Path) -> Result<String, SchemaError> {
    fs::read_to_string(path).map_err(|err| SchemaError::IoError {
        message: format!("{}: {err}", path.display()),
    })
}

/// Split a Fortran line into `(code_part, comment_part)` outside quotes.
#[must_use]
pub fn strip_fortran_comment(line: &str) -> (&str, &str) {
    match find_unquoted_char(line, '!') {
        Some(idx) => (line[..idx].trim(), line[idx + 1..].trim()),
        None => (line.trim(), ""),
    }
}

/// Parse a Fortran `.defaults` or namelist literal into a [`ControlValue`].
///
/// # Errors
/// Returns [`SchemaError::ParseError`] if `raw` is not a valid logical, string,
/// 64-bit integer, or finite floating-point literal.
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
    if tok.eq_ignore_ascii_case(".true.")
        || tok.eq_ignore_ascii_case(".true")
        || tok.eq_ignore_ascii_case(".t.")
        || tok.eq_ignore_ascii_case("t")
    {
        return Ok(ControlValue::Logical(true));
    }
    if tok.eq_ignore_ascii_case(".false.")
        || tok.eq_ignore_ascii_case(".false")
        || tok.eq_ignore_ascii_case(".f.")
        || tok.eq_ignore_ascii_case("f")
    {
        return Ok(ControlValue::Logical(false));
    }
    let bytes = tok.as_bytes();
    if bytes.len() >= 2
        && bytes[0] == bytes[bytes.len() - 1]
        && (bytes[0] == b'\'' || bytes[0] == b'"')
    {
        let (q, doubled, single) = if bytes[0] == b'\'' {
            ('\'', "''", "'")
        } else {
            ('"', "\"\"", "\"")
        };
        let inner = &tok[1..tok.len() - 1];
        if !inner.contains(q) {
            return Ok(ControlValue::String(inner.to_owned()));
        }
        if inner.replace(doubled, "").contains(q) {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("unescaped quote in string literal: {tok}"),
            });
        }
        return Ok(ControlValue::String(inner.replace(doubled, single)));
    }
    if is_integer_token(tok) {
        let int_val =
            tok.parse::<i64>().map_err(|_| SchemaError::ParseError {
                line_number,
                message: format!("64-bit integer overflow in literal: {tok:?}"),
            })?;
        return Ok(ControlValue::Integer(int_val));
    }
    if !is_fortran_float_token(tok) {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!("invalid Fortran control literal: {tok:?}"),
        });
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
    let (lhs, rhs) = (lhs_raw.trim(), rhs_raw.trim());
    let (name, sub) = if let Some(open_idx) = lhs.find('(') {
        let sub_str = lhs[open_idx + 1..].strip_suffix(')').map(str::trim);
        let Some(s) =
            sub_str.filter(|s| !s.is_empty() && !s.contains(['(', ')']))
        else {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("malformed subscript in {lhs:?}"),
            });
        };
        (lhs[..open_idx].trim(), Some(s))
    } else {
        (lhs, None)
    };
    if !is_valid_identifier(name) {
        return Err(SchemaError::ParseError {
            line_number,
            message: format!("invalid control identifier: {name:?}"),
        });
    }
    Ok((name, sub, rhs))
}

/// Update `current_section` and `pending_docs` from a `.defaults` comment line.
fn update_comment_state<'a>(
    comment: &'a str,
    pending_docs: &mut Vec<&'a str>,
    current_section: &mut String,
) {
    let is_under = comment.len() >= 3
        && comment
            .bytes()
            .all(|b| matches!(b, b'=' | b'-' | b'~' | b'^'));
    if is_under {
        if comment.starts_with(['=', '-', '^']) {
            if let Some(title) = pending_docs.pop() {
                title.trim().clone_into(current_section);
                pending_docs.clear();
            }
        }
    } else if !comment.is_empty() && comment != "::" {
        pending_docs.push(comment);
    }
}

/// Merge an indexed element override into an existing [`ControlDefault`].
fn merge_element_override(
    ctrl: &mut ControlDefault,
    sub_str: &str,
    val: ControlValue,
    line_number: usize,
) -> Result<(), SchemaError> {
    ctrl.is_array = true;
    let val_t = val.control_type();
    if ctrl.control_type == val_t {
        ctrl.element_overrides.push((sub_str.to_owned(), val));
        return Ok(());
    }
    if ctrl.control_type == ControlType::Float {
        if let ControlValue::Integer(i) = val {
            #[allow(clippy::cast_precision_loss)]
            ctrl.element_overrides
                .push((sub_str.to_owned(), ControlValue::Float(i as f64)));
            return Ok(());
        }
    }
    if ctrl.control_type == ControlType::Integer && val_t == ControlType::Float
    {
        if let ControlValue::Integer(def_i) = ctrl.default_value {
            ctrl.control_type = ControlType::Float;
            #[allow(clippy::cast_precision_loss)]
            {
                ctrl.default_value = ControlValue::Float(def_i as f64);
                for (_, v) in &mut ctrl.element_overrides {
                    if let ControlValue::Integer(elem_i) = *v {
                        *v = ControlValue::Float(elem_i as f64);
                    }
                }
            }
            ctrl.element_overrides.push((sub_str.to_owned(), val));
            return Ok(());
        }
    }
    Err(SchemaError::ParseError {
        line_number,
        message: format!(
            "incompatible override type for {:?}: expected {:?}, got {val_t:?}",
            ctrl.name, ctrl.control_type
        ),
    })
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
            update_comment_state(
                comment,
                &mut pending_docs,
                &mut current_section,
            );
            continue;
        }
        let (name, sub, rhs) = split_assignment(code, line_number)?;
        let val = parse_control_value(rhs, line_number)?;
        let key = name.to_ascii_lowercase();
        let is_specific = sub.is_some_and(|s| s != ":");
        if let (Some(&pos), Some(sub_str)) =
            (index_by_key.get(&key), sub.filter(|_| is_specific))
        {
            merge_element_override(
                &mut controls[pos],
                sub_str,
                val,
                line_number,
            )?;
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
    parse_defaults_text(&read_file_text(path)?)
}

/// Parse active column declarations from a MESA `*_columns.list` file.
#[must_use]
pub fn parse_columns_text(text: &str) -> Vec<ColumnEntry> {
    let (mut entries, mut seen, mut section) =
        (Vec::new(), HashSet::new(), String::new());
    for (idx, raw_line) in text.lines().enumerate() {
        let (code, comment) = strip_fortran_comment(raw_line);
        if code.is_empty() {
            if comment.starts_with('#') {
                comment
                    .trim_start_matches('#')
                    .trim()
                    .clone_into(&mut section);
            }
            continue;
        }
        if seen.insert(code.to_ascii_lowercase()) {
            entries.push(ColumnEntry {
                name: code.to_owned(),
                section: section.clone(),
                comment: comment.to_owned(),
                line_number: idx + 1,
            });
        }
    }
    entries
}

/// Read and parse a MESA `*_columns.list` file from disk.
///
/// # Errors
/// Returns [`SchemaError::IoError`] if `path` cannot be read.
pub fn parse_columns_file(
    path: &Path,
) -> Result<Vec<ColumnEntry>, SchemaError> {
    Ok(parse_columns_text(&read_file_text(path)?))
}
