// SPDX-License-Identifier: LGPL-3.0-or-later
//! Physics table `SHA-256` manifest, `.safetensors` I/O, and MESA converters.
//!
//! Provides:
//! 1. [`TableArtifact`] and [`parse_manifest_text`] / [`parse_manifest_file`]
//!    for parsing `manifest.toml` table integrity declarations.
//! 2. [`sha256`] for pure Safe Rust FIPS 180-4 `SHA-256` hashing and file
//!    verification.
//! 3. [`safetensors`] for reading and writing `F64` (`f64`) and `I64` (`i64`)
//!    `.safetensors` bundles byte-for-byte compatible with Python
//!    `safetensors.numpy`.
//! 4. [`converters`] for converting MESA `lodders03.data`, `isotopes.data`, and
//!    `helm_table.dat` ASCII tables into row-major `.safetensors` bundles.

pub mod converters;
pub mod safetensors;
pub mod sha256;

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use sha256::is_valid_sha256_hex;

/// Metadata and integrity record for a `.safetensors` physics table in `manifest.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableArtifact {
    /// Logical table identifier (e.g., `"lodders03"`, `"isotopes"`, `"helm_table"`).
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

/// Error returned during manifest parsing, `SHA-256` verification, or `.safetensors` I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// File could not be read or written on disk.
    IoError {
        /// File path string.
        path: String,
        /// Underlying OS I/O error message.
        message: String,
    },
    /// Syntax error while parsing `manifest.toml`, `.safetensors` header, or ASCII tables.
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
                "SHA-256 checksum mismatch for {path:?}: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for DataError {}

/// Extract a double-quoted TOML string value.
fn parse_toml_quoted(raw: &str) -> Result<String, DataError> {
    let trimmed = raw.trim();
    let Some(inner) =
        trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
    else {
        return Err(DataError::ParseError(format!(
            "expected double-quoted TOML string, got {trimmed:?}"
        )));
    };
    Ok(inner.to_owned())
}

/// Finalize and validate a single `[[tables]]` stanza into a [`TableArtifact`].
fn build_artifact(
    fields: &mut BTreeMap<String, String>,
    tensors: &mut Vec<String>,
) -> Result<TableArtifact, DataError> {
    let name = fields
        .remove("name")
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            DataError::ValidationError(
                "manifest table missing 'name'".to_owned(),
            )
        })?;
    let filename = fields
        .remove("filename")
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
        DataError::ValidationError(format!(
            "manifest table {name:?} missing 'filename'"
        ))
    })?;
    let sha256 = fields.remove("sha256").ok_or_else(|| {
        DataError::ValidationError(format!(
            "manifest table {name:?} missing 'sha256'"
        ))
    })?;
    if !is_valid_sha256_hex(&sha256) {
        return Err(DataError::ValidationError(format!(
            "manifest table {name:?} has invalid 'sha256': {sha256:?}"
        )));
    }
    let size_str = fields.remove("size_bytes").ok_or_else(|| {
        DataError::ValidationError(format!(
            "manifest table {name:?} missing 'size_bytes'"
        ))
    })?;
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
    let description = fields.remove("description").unwrap_or_default();
    Ok(TableArtifact {
        name,
        filename,
        sha256: sha256.to_ascii_lowercase(),
        size_bytes,
        tensors: std::mem::take(tensors),
        description,
    })
}

/// Parse `manifest.toml` text (`[[tables]]` stanzas) into a map of [`TableArtifact`]s.
///
/// # Errors
/// Returns [`DataError::ParseError`] or [`DataError::ValidationError`] if the
/// manifest syntax is malformed, `[[tables]]` is empty, duplicate table names
/// occur, or any required field fails validation.
pub fn parse_manifest_text(
    text: &str,
) -> Result<BTreeMap<String, TableArtifact>, DataError> {
    let mut artifacts = BTreeMap::new();
    let mut fields: BTreeMap<String, String> = BTreeMap::new();
    let mut tensors: Vec<String> = Vec::new();
    let mut in_table = false;
    let mut in_tensors_array = false;

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[tables]]" {
            if in_table {
                let art = build_artifact(&mut fields, &mut tensors)?;
                if artifacts.insert(art.name.clone(), art).is_some() {
                    return Err(DataError::ValidationError(
                        "duplicate table name in manifest".to_owned(),
                    ));
                }
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
            if let Some(before_close) = line.strip_suffix(']') {
                for item in before_close.split(',') {
                    if !item.trim().is_empty() {
                        tensors.push(parse_toml_quoted(item)?);
                    }
                }
                in_tensors_array = false;
            } else {
                for item in line.split(',') {
                    if !item.trim().is_empty() {
                        tensors.push(parse_toml_quoted(item)?);
                    }
                }
            }
            continue;
        }
        let Some((lhs, rhs)) = line.split_once('=') else {
            return Err(DataError::ParseError(format!(
                "malformed manifest line: {line:?}"
            )));
        };
        let key = lhs.trim();
        let val = rhs.trim();
        if key == "tensors" {
            let Some(after_open) = val.strip_prefix('[') else {
                return Err(DataError::ParseError(
                    "tensors must be a bracketed array".to_owned(),
                ));
            };
            if let Some(inner) = after_open.strip_suffix(']') {
                for item in inner.split(',') {
                    if !item.trim().is_empty() {
                        tensors.push(parse_toml_quoted(item)?);
                    }
                }
            } else {
                for item in after_open.split(',') {
                    if !item.trim().is_empty() {
                        tensors.push(parse_toml_quoted(item)?);
                    }
                }
                in_tensors_array = true;
            }
        } else if key == "size_bytes" {
            fields.insert(key.to_owned(), val.to_owned());
        } else {
            fields.insert(key.to_owned(), parse_toml_quoted(val)?);
        }
    }

    if in_table {
        let art = build_artifact(&mut fields, &mut tensors)?;
        if artifacts.insert(art.name.clone(), art).is_some() {
            return Err(DataError::ValidationError(
                "duplicate table name in manifest".to_owned(),
            ));
        }
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
