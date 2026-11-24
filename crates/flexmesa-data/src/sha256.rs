// SPDX-License-Identifier: GPL-3.0-or-later
//! `SHA-256` file verification wrappers over [`flexmesa_utils::sha256`].

use std::fs::File;
use std::path::Path;

pub use flexmesa_utils::sha256::{
    compute_sha256_bytes, compute_sha256_reader, is_valid_sha256_hex,
    Sha256Hasher,
};

use crate::DataError;

/// Compute the lowercase 64-character `SHA-256` hex digest of a file on disk.
///
/// # Errors
/// Returns [`DataError::IoError`] if `path` cannot be opened or read.
pub fn compute_sha256_file(path: &Path) -> Result<String, DataError> {
    let mut file = File::open(path).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    compute_sha256_reader(&mut file).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })
}

/// Verify that `path` matches `expected_sha256`.
///
/// # Errors
/// Returns [`DataError::ValidationError`] if `expected_sha256` is not a valid
/// 64-character hex string, [`DataError::IoError`] on file read failure, or
/// [`DataError::ChecksumMismatch`] if the computed digest differs.
pub fn verify_file_sha256(
    path: &Path,
    expected_sha256: &str,
) -> Result<String, DataError> {
    if !is_valid_sha256_hex(expected_sha256) {
        return Err(DataError::ValidationError(format!(
            "invalid expected SHA-256 hex digest: {expected_sha256:?}"
        )));
    }
    let expected_norm = expected_sha256.trim().to_ascii_lowercase();
    let actual = compute_sha256_file(path)?;
    if actual != expected_norm {
        return Err(DataError::ChecksumMismatch {
            path: path.display().to_string(),
            expected: expected_norm,
            actual,
        });
    }
    Ok(actual)
}
