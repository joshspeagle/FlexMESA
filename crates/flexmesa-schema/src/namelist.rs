// SPDX-License-Identifier: LGPL-3.0-or-later
//! Parser and schema validator for legacy MESA Fortran `&namelist` (`inlist`)
//! files.
//!
//! Extracts [`NamelistBlock`] and [`NamelistAssignment`] entries from Fortran
//! `&namelist ... /` text and validates every assigned parameter against a
//! [`ControlDefault`] schema.

use std::collections::HashMap;
use std::path::Path;

use crate::{
    find_unquoted_char, is_valid_identifier, parse_control_value,
    read_file_text, split_assignment, strip_fortran_comment, ControlDefault,
    ControlType, ControlValue, SchemaError,
};

/// Single `name[(subscript)] = value` assignment inside a Fortran namelist.
#[derive(Clone, Debug, PartialEq)]
pub struct NamelistAssignment {
    /// Control parameter name as written in the `inlist`.
    pub name: String,
    /// Optional subscript expression inside `(...)`.
    pub subscript: Option<String>,
    /// Parsed [`ControlValue`] on the right-hand side.
    pub value: ControlValue,
    /// 1-based line number in the `inlist` source.
    pub line_number: usize,
}

/// Parsed Fortran `&namelist ... /` block from a MESA `inlist` file.
#[derive(Clone, Debug, PartialEq)]
pub struct NamelistBlock {
    /// Namelist group name without leading `&` (for example, `"eos"`).
    pub name: String,
    /// Ordered list of assignments inside the block.
    pub assignments: Vec<NamelistAssignment>,
}

/// Split a namelist body on commas outside quotes and parentheses.
fn split_top_level_commas(code: &str) -> Vec<&str> {
    if !code.contains(',') {
        let trimmed = code.trim();
        return if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed]
        };
    }
    let (mut parts, mut start, mut in_s, mut in_d, mut depth) =
        (Vec::new(), 0, false, false, 0_usize);
    for (idx, ch) in code.char_indices() {
        if ch == '\'' && !in_d {
            in_s = !in_s;
        } else if ch == '"' && !in_s {
            in_d = !in_d;
        } else if !in_s && !in_d {
            if ch == '(' {
                depth += 1;
            } else if ch == ')' {
                depth = depth.saturating_sub(1);
            } else if ch == ',' && depth == 0 {
                let seg = code[start..idx].trim();
                if !seg.is_empty() {
                    parts.push(seg);
                }
                start = idx + 1;
            }
        }
    }
    let tail = code[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts
}

/// Parse comma-separated assignments or array value lists on one line.
fn parse_assignments_in_segment(
    body: &str,
    line_number: usize,
    out: &mut Vec<NamelistAssignment>,
) -> Result<(), SchemaError> {
    let mut seq_name: Option<String> = None;
    let mut seq_next_idx = 1_usize;

    for item in split_top_level_commas(body) {
        if find_unquoted_char(item, '=').is_some() {
            let (name, sub, rhs) = split_assignment(item, line_number)?;
            let value = parse_control_value(rhs, line_number)?;
            seq_name = Some(name.to_owned());
            seq_next_idx = sub
                .and_then(|s| s.parse::<usize>().ok())
                .map_or(2, |n| n + 1);
            out.push(NamelistAssignment {
                name: name.to_owned(),
                subscript: sub.map(str::to_owned),
                value,
                line_number,
            });
        } else if let Some(ref arr_name) = seq_name {
            if let Some(prev) = out.last_mut() {
                if prev.name == *arr_name && prev.subscript.is_none() {
                    prev.subscript = Some("1".to_owned());
                }
            }
            let value = parse_control_value(item, line_number)?;
            out.push(NamelistAssignment {
                name: arr_name.clone(),
                subscript: Some(seq_next_idx.to_string()),
                value,
                line_number,
            });
            seq_next_idx += 1;
        } else {
            return Err(SchemaError::ParseError {
                line_number,
                message: format!("expected '=' in assignment: {item:?}"),
            });
        }
    }
    Ok(())
}

/// Parse one or more Fortran `&namelist ... /` blocks from `inlist` text.
///
/// # Errors
/// Returns [`SchemaError::ParseError`] on malformed namelist syntax.
///
/// # Examples
/// ```
/// use flexmesa_schema::parse_namelist_text;
///
/// let blocks = parse_namelist_text("&eos\n  use_PC = .true.\n/\n").unwrap();
/// assert_eq!(blocks.len(), 1);
/// assert_eq!(blocks[0].name, "eos");
/// ```
pub fn parse_namelist_text(
    text: &str,
) -> Result<Vec<NamelistBlock>, SchemaError> {
    let mut blocks = Vec::new();
    let mut active_name: Option<String> = None;
    let mut active_assigns: Vec<NamelistAssignment> = Vec::new();

    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let (mut code, _) = strip_fortran_comment(raw_line);
        if code.is_empty() {
            continue;
        }
        if active_name.is_none() {
            let Some(rest) = code.strip_prefix('&') else {
                return Err(SchemaError::ParseError {
                    line_number,
                    message: format!("expected '&namelist' header: {code:?}"),
                });
            };
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '/')
                .unwrap_or(rest.len());
            let group = &rest[..end];
            if !is_valid_identifier(group) {
                return Err(SchemaError::ParseError {
                    line_number,
                    message: format!("invalid namelist name: {group:?}"),
                });
            }
            active_name = Some(group.to_owned());
            code = rest[end..].trim();
            if code.is_empty() {
                continue;
            }
        }
        let slash_pos = find_unquoted_char(code, '/');
        if let Some(p) = slash_pos {
            let trailing = code[p + 1..].trim();
            if !trailing.is_empty() {
                return Err(SchemaError::ParseError {
                    line_number,
                    message: format!(
                        "unexpected trailing tokens after '/': {trailing:?}"
                    ),
                });
            }
        }
        let body = slash_pos.map_or(code, |p| code[..p].trim());
        parse_assignments_in_segment(body, line_number, &mut active_assigns)?;
        if slash_pos.is_some() {
            blocks.push(NamelistBlock {
                name: active_name.take().unwrap_or_default(),
                assignments: std::mem::take(&mut active_assigns),
            });
        }
    }
    if let Some(unclosed) = active_name {
        return Err(SchemaError::ParseError {
            line_number: text.lines().count().max(1),
            message: format!("unclosed namelist '&{unclosed}'"),
        });
    }
    Ok(blocks)
}

/// Read and parse a MESA Fortran `inlist` file from disk.
///
/// # Errors
/// Returns [`SchemaError`] on I/O failure or malformed namelist syntax.
pub fn parse_namelist_file(
    path: &Path,
) -> Result<Vec<NamelistBlock>, SchemaError> {
    parse_namelist_text(&read_file_text(path)?)
}

/// Validate a [`NamelistBlock`] against a [`ControlDefault`] slice.
///
/// # Errors
/// Returns [`SchemaError::UnknownControl`] or [`SchemaError::ValidationError`]
/// if an assignment references an unknown control, indexes a scalar, or
/// supplies an incompatible value type.
///
/// # Examples
/// ```
/// use flexmesa_schema::{
///     parse_defaults_text, parse_namelist_text, validate_namelist_block,
/// };
///
/// let schema = parse_defaults_text("  use_PC = .false.\n").unwrap();
/// let blocks = parse_namelist_text("&eos\n  use_PC = .true.\n/\n").unwrap();
/// assert!(validate_namelist_block(&blocks[0], &schema, false).is_ok());
/// ```
pub fn validate_namelist_block(
    block: &NamelistBlock,
    schema: &[ControlDefault],
    strict_numeric_types: bool,
) -> Result<(), SchemaError> {
    let by_name: HashMap<String, &ControlDefault> = schema
        .iter()
        .map(|c| (c.name.to_ascii_lowercase(), c))
        .collect();
    for assign in &block.assignments {
        let Some(&spec) = by_name.get(&assign.name.to_ascii_lowercase()) else {
            return Err(SchemaError::UnknownControl {
                namelist: block.name.clone(),
                control: assign.name.clone(),
                line_number: assign.line_number,
            });
        };
        if assign.subscript.is_some() && !spec.is_array {
            return Err(SchemaError::ValidationError {
                control: spec.name.clone(),
                line_number: assign.line_number,
                message: "scalar control cannot be indexed".to_owned(),
            });
        }
        let actual_t = assign.value.control_type();
        let expected_t = spec.control_type;
        let ok = actual_t == expected_t
            || (expected_t == ControlType::Float
                && actual_t == ControlType::Integer)
            || (!strict_numeric_types
                && expected_t == ControlType::Integer
                && actual_t == ControlType::Float);
        if !ok {
            return Err(SchemaError::ValidationError {
                control: spec.name.clone(),
                line_number: assign.line_number,
                message: format!("expected {expected_t:?}, got {actual_t:?}"),
            });
        }
    }
    Ok(())
}
