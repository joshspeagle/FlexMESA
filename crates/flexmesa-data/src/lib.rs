// SPDX-License-Identifier: LGPL-3.0-or-later
//! Physics table `SHA-256` manifest, `.safetensors` I/O, and MESA converters.
//!
//! Provides:
//! 1. [`TableArtifact`] and [`parse_manifest_text`], [`parse_manifest_file`],
//!    and [`load_default_manifest`] for parsing `manifest.toml` table
//!    integrity declarations.
//! 2. [`sha256`] for pure Safe Rust FIPS 180-4 `SHA-256` hashing and file
//!    verification.
//! 3. [`safetensors`] for reading and writing `F64` (`f64`) and `I64` (`i64`)
//!    `.safetensors` bundles byte-for-byte compatible with Python
//!    `safetensors.numpy`.
//! 4. [`converters`] for converting MESA `lodders03.data`, `isotopes.data`, and
//!    `helm_table.dat` ASCII tables into row-major `.safetensors` bundles.

pub mod converters;
mod json_cursor;
pub mod safetensors;
pub mod sha256;

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use sha256::is_valid_sha256_hex;

const DEFAULT_MANIFEST_TOML: &str =
    include_str!("../../../src/flexmesa/data/manifest.toml");

/// Metadata and integrity record for a `.safetensors` table in `manifest.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableArtifact {
    /// Logical table identifier (`"lodders03"`, `"isotopes"`, `"helm_table"`).
    pub name: String,
    /// Canonical `.safetensors` filename (e.g., `"lodders03.safetensors"`).
    pub filename: String,
    /// Lowercase 64-character hexadecimal `SHA-256` digest.
    pub sha256: String,
    /// Uncompressed `.safetensors` file size in bytes.
    pub size_bytes: u64,
    /// Ordered list of tensor names required in the `.safetensors` bundle.
    pub tensors: Vec<String>,
    /// Human-readable description of the physical table and source.
    pub description: String,
}

/// Error returned by manifest parsing, `SHA-256` checks, or `.safetensors` I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// File could not be read or written on disk.
    IoError {
        /// File path string.
        path: String,
        /// Underlying OS I/O error message.
        message: String,
    },
    /// Syntax error while parsing `manifest.toml`, JSON, or ASCII tables.
    ParseError(String),
    /// Schema, dtype, shape, or finiteness validation failure.
    ValidationError(String),
    /// Computed `SHA-256` digest did not match the expected manifest digest.
    ChecksumMismatch {
        /// File path string.
        path: String,
        /// Expected 64-character lowercase hex digest.
        expected: String,
        /// Actual computed 64-character lowercase hex digest.
        actual: String,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError { path, message } => {
                write!(f, "failed to access {path:?}: {message}")
            }
            Self::ParseError(msg) => write!(f, "data parse error: {msg}"),
            Self::ValidationError(msg) => {
                write!(f, "data validation error: {msg}")
            }
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "SHA-256 mismatch for {path:?}: \
                 expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for DataError {}

/// Extract a double-quoted TOML string value.
fn parse_toml_quoted(raw: &str) -> Result<String, DataError> {
    let trimmed = raw.trim();
    trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .map(str::to_owned)
        .ok_or_else(|| {
            DataError::ParseError(format!(
                "expected double-quoted TOML string, got {trimmed:?}"
            ))
        })
}

/// Parse comma-separated quoted strings from `segment` into `tensors`.
fn push_quoted_items(
    segment: &str,
    tensors: &mut Vec<String>,
) -> Result<(), DataError> {
    for item in segment.split(',') {
        if !item.trim().is_empty() {
            tensors.push(parse_toml_quoted(item)?);
        }
    }
    Ok(())
}

/// Remove and validate a required non-empty string field from `fields`.
fn take_required_field(
    fields: &mut BTreeMap<String, String>,
    table: &str,
    key: &str,
) -> Result<String, DataError> {
    fields
        .remove(key)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            DataError::ValidationError(format!(
                "manifest table {table:?} missing {key:?}"
            ))
        })
}

/// Finalize and insert a single `[[tables]]` stanza into `artifacts`.
fn flush_artifact(
    fields: &mut BTreeMap<String, String>,
    tensors: &mut Vec<String>,
    artifacts: &mut BTreeMap<String, TableArtifact>,
) -> Result<(), DataError> {
    let name = take_required_field(fields, "<unnamed>", "name")?;
    let filename = take_required_field(fields, &name, "filename")?;
    let sha256 = take_required_field(fields, &name, "sha256")?;
    if !is_valid_sha256_hex(&sha256) {
        return Err(DataError::ValidationError(format!(
            "manifest table {name:?} has invalid 'sha256': {sha256:?}"
        )));
    }
    let size_str = take_required_field(fields, &name, "size_bytes")?;
    let size_bytes = size_str.parse::<u64>().map_err(|_| {
        DataError::ValidationError(format!(
            "manifest table {name:?} has invalid 'size_bytes': {size_str:?}"
        ))
    })?;
    if tensors.is_empty() || tensors.iter().any(|t| t.trim().is_empty()) {
        return Err(DataError::ValidationError(format!(
            "manifest table {name:?} must declare a non-empty 'tensors' list"
        )));
    }
    let artifact = TableArtifact {
        name: name.clone(),
        filename,
        sha256: sha256.to_ascii_lowercase(),
        size_bytes,
        tensors: std::mem::take(tensors),
        description: fields.remove("description").unwrap_or_default(),
    };
    if artifacts.insert(name, artifact).is_some() {
        return Err(DataError::ValidationError(
            "duplicate table name in manifest".to_owned(),
        ));
    }
    Ok(())
}

/// Parse `manifest.toml` text (`[[tables]]` stanzas) into [`TableArtifact`]s.
///
/// # Errors
/// Returns [`DataError::ParseError`] or [`DataError::ValidationError`] if the
/// manifest syntax is malformed, `[[tables]]` is empty, duplicate table names
/// occur, or any required field fails validation.
///
/// # Examples
/// ```
/// use flexmesa_data::load_default_manifest;
///
/// let tables = load_default_manifest().unwrap();
/// assert!(tables.contains_key("lodders03"));
/// assert!(tables.contains_key("isotopes"));
/// ```
pub fn parse_manifest_text(
    text: &str,
) -> Result<BTreeMap<String, TableArtifact>, DataError> {
    let mut artifacts = BTreeMap::new();
    let mut fields: BTreeMap<String, String> = BTreeMap::new();
    let mut tensors: Vec<String> = Vec::new();
    let (mut in_table, mut in_tensors_array) = (false, false);

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[tables]]" {
            if in_table {
                flush_artifact(&mut fields, &mut tensors, &mut artifacts)?;
            }
            in_table = true;
            in_tensors_array = false;
            continue;
        }
        if !in_table {
            return Err(DataError::ParseError(format!(
                "unexpected line before [[tables]]: {line:?}"
            )));
        }
        if in_tensors_array {
            let body = line.strip_suffix(']').unwrap_or(line);
            push_quoted_items(body, &mut tensors)?;
            in_tensors_array = !line.ends_with(']');
            continue;
        }
        let Some((lhs, rhs)) = line.split_once('=') else {
            return Err(DataError::ParseError(format!(
                "malformed manifest line: {line:?}"
            )));
        };
        let (key, val) = (lhs.trim(), rhs.trim());
        if key == "tensors" {
            let Some(after_open) = val.strip_prefix('[') else {
                return Err(DataError::ParseError(
                    "tensors must be a bracketed array".to_owned(),
                ));
            };
            let body = after_open.strip_suffix(']').unwrap_or(after_open);
            push_quoted_items(body, &mut tensors)?;
            in_tensors_array = !after_open.ends_with(']');
        } else if key == "size_bytes" {
            fields.insert(key.to_owned(), val.to_owned());
        } else {
            fields.insert(key.to_owned(), parse_toml_quoted(val)?);
        }
    }
    if in_table {
        flush_artifact(&mut fields, &mut tensors, &mut artifacts)?;
    }
    if artifacts.is_empty() {
        return Err(DataError::ValidationError(
            "manifest must define at least one [[tables]] entry".to_owned(),
        ));
    }
    Ok(artifacts)
}

/// Read and parse a `manifest.toml` file from disk.
///
/// # Errors
/// Returns [`DataError`] on I/O failure or invalid manifest content.
pub fn parse_manifest_file(
    path: &Path,
) -> Result<BTreeMap<String, TableArtifact>, DataError> {
    let text = fs::read_to_string(path).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    parse_manifest_text(&text)
}

/// Load the bundled `src/flexmesa/data/manifest.toml` physics table manifest.
///
/// # Errors
/// Returns [`DataError`] if the embedded manifest fails validation.
pub fn load_default_manifest(
) -> Result<BTreeMap<String, TableArtifact>, DataError> {
    parse_manifest_text(DEFAULT_MANIFEST_TOML)
}
