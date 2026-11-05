// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure Safe Rust `.safetensors` (`F64` and `I64`) encoder and decoder.
//!
//! Reads and writes standard `.safetensors` binary buffers (`8`-byte
//! little-endian header size + UTF-8 JSON header padded with ASCII spaces to an
//! 8-byte boundary + contiguous little-endian raw tensor bytes), interoperating
//! byte-for-byte with Python `safetensors.numpy`.
//!
//! Enforces:
//! - Checked dimension products (`shape.iter().try_fold(...)`) to prevent
//!   integer overflow on malformed tensor shapes.
//! - Exact byte-length matching between `data_offsets` and `8 * elem_count`.
//! - Optional IEEE-754 finiteness verification (`require_finite`) on all `F64`
//!   tensors.
//!
//! # Examples
//! ```
//! use std::collections::BTreeMap;
//! use flexmesa_data::safetensors::{
//!     decode_safetensors_bytes, encode_safetensors_bytes,
//!     SafeTensorBundle, TensorData,
//! };
//!
//! let t = TensorData::f64(vec![2], vec![1.0, 2.0]);
//! let bundle = SafeTensorBundle {
//!     tensors: BTreeMap::from([("x".into(), t)]),
//!     metadata: BTreeMap::new(),
//! };
//! let bytes = encode_safetensors_bytes(&bundle, true).unwrap();
//! assert_eq!(decode_safetensors_bytes(&bytes, true).unwrap(), bundle);
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::json_cursor::{
    escape_json_str, parse_safetensors_header, split_header_and_payload,
};
use crate::sha256::{compute_sha256_bytes, is_valid_sha256_hex};
use crate::DataError;

/// Typed multi-dimensional row-major tensor stored in a `.safetensors` bundle.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// IEEE-754 double-precision (`F64`) tensor with shape and row-major
    /// values.
    Float64 {
        /// Dimension sizes along each axis (`(d_0, d_1, ...)`).
        shape: Vec<usize>,
        /// Row-major flattened `f64` elements (`len == shape.product()`).
        values: Vec<f64>,
    },
    /// Signed 64-bit integer (`I64`) tensor with shape and row-major values.
    Int64 {
        /// Dimension sizes along each axis (`(d_0, d_1, ...)`).
        shape: Vec<usize>,
        /// Row-major flattened `i64` elements (`len == shape.product()`).
        values: Vec<i64>,
    },
}

impl TensorData {
    /// Construct a [`TensorData::Float64`] tensor from `shape` and `values`.
    #[must_use]
    pub fn f64(shape: Vec<usize>, values: Vec<f64>) -> Self {
        Self::Float64 { shape, values }
    }

    /// Construct a [`TensorData::Int64`] tensor from `shape` and `values`.
    #[must_use]
    pub fn i64(shape: Vec<usize>, values: Vec<i64>) -> Self {
        Self::Int64 { shape, values }
    }

    /// Return the tensor shape slice (`&[usize]`).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Float64 { shape, .. } | Self::Int64 { shape, .. } => shape,
        }
    }

    /// Return the number of scalar elements stored in the tensor.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Float64 { values, .. } => values.len(),
            Self::Int64 { values, .. } => values.len(),
        }
    }

    /// Return `true` if the tensor has zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the `f64` element slice if this is a [`TensorData::Float64`].
    #[must_use]
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            Self::Float64 { values, .. } => Some(values),
            Self::Int64 { .. } => None,
        }
    }

    /// Return the `i64` element slice if this is a [`TensorData::Int64`].
    #[must_use]
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match self {
            Self::Int64 { values, .. } => Some(values),
            Self::Float64 { .. } => None,
        }
    }
}

/// Collection of named [`TensorData`] arrays and string metadata.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SafeTensorBundle {
    /// Named tensors in deterministic lexicographical key order.
    pub tensors: BTreeMap<String, TensorData>,
    /// Optional string-to-string header metadata (`__metadata__`).
    pub metadata: BTreeMap<String, String>,
}

/// Checked product of `shape` dimensions multiplied by 8 bytes per element.
///
/// Prevents integer wrap-around when untrusted `.safetensors` headers contain
/// large axis dimensions.
fn checked_byte_len(name: &str, shape: &[usize]) -> Result<usize, DataError> {
    let elems = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            DataError::ValidationError(format!(
                "shape product overflow for {name:?}"
            ))
        })?;
    elems.checked_mul(8).ok_or_else(|| {
        DataError::ValidationError(format!("byte length overflow for {name:?}"))
    })
}

/// Validate a [`TensorData`] entry and return `(header_entry_json, byte_len)`.
///
/// Checks that `name` is non-empty and not `"__metadata__"`, verifies that the
/// flattened element count matches `shape`, and checks `f64` finiteness when
/// `require_finite` is `true`.
fn build_tensor_header_entry(
    name: &str,
    tensor: &TensorData,
    offset: usize,
    require_finite: bool,
) -> Result<(String, usize), DataError> {
    if name.trim().is_empty() || name == "__metadata__" {
        return Err(DataError::ValidationError(format!(
            "invalid tensor name: {name:?}"
        )));
    }
    let byte_len = checked_byte_len(name, tensor.shape())?;
    if tensor.len().checked_mul(8) != Some(byte_len) {
        return Err(DataError::ValidationError(format!(
            "tensor {name:?} element count {} does not match shape {:?}",
            tensor.len(),
            tensor.shape()
        )));
    }
    let dtype_str = match tensor {
        TensorData::Float64 { values, .. } => {
            if require_finite && !values.iter().all(|v| v.is_finite()) {
                return Err(DataError::ValidationError(format!(
                    "tensor {name:?} contains non-finite float64 values"
                )));
            }
            "F64"
        }
        TensorData::Int64 { .. } => "I64",
    };
    let end = offset + byte_len;
    let shape_str = tensor
        .shape()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let entry = format!(
        "{}:{{\"dtype\":\"{dtype_str}\",\"shape\":[{shape_str}],\
         \"data_offsets\":[{offset},{end}]}}",
        escape_json_str(name)
    );
    Ok((entry, byte_len))
}

/// Serialize a [`SafeTensorBundle`] into `.safetensors` binary bytes.
///
/// # Errors
/// Returns [`DataError::ValidationError`] if `bundle.tensors` is empty, if any
/// tensor's element count mismatches its shape, or if non-finite `f64` values
/// are present when `require_finite` is `true`.
pub fn encode_safetensors_bytes(
    bundle: &SafeTensorBundle,
    require_finite: bool,
) -> Result<Vec<u8>, DataError> {
    if bundle.tensors.is_empty() {
        return Err(DataError::ValidationError(
            "cannot encode an empty SafeTensorBundle".to_owned(),
        ));
    }
    let mut header_parts: Vec<String> = Vec::new();
    if !bundle.metadata.is_empty() {
        let meta: Vec<String> = bundle
            .metadata
            .iter()
            .map(|(k, v)| {
                format!("{}:{}", escape_json_str(k), escape_json_str(v))
            })
            .collect();
        header_parts.push(format!("\"__metadata__\":{{{}}}", meta.join(",")));
    }
    let mut total_payload = 0usize;
    for (name, tensor) in &bundle.tensors {
        let (entry, n_bytes) = build_tensor_header_entry(
            name,
            tensor,
            total_payload,
            require_finite,
        )?;
        header_parts.push(entry);
        total_payload += n_bytes;
    }
    let mut header_json = format!("{{{}}}", header_parts.join(","));
    while !header_json.len().is_multiple_of(8) {
        header_json.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header_json.len() + total_payload);
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    for tensor in bundle.tensors.values() {
        match tensor {
            TensorData::Float64 { values, .. } => {
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            TensorData::Int64 { values, .. } => {
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    Ok(out)
}

/// Decode a single tensor's raw little-endian slice into [`TensorData`].
fn decode_tensor_slice(
    name: &str,
    dtype: &str,
    shape: Vec<usize>,
    slice: &[u8],
    require_finite: bool,
) -> Result<TensorData, DataError> {
    let expected = checked_byte_len(name, &shape)?;
    if slice.len() != expected {
        return Err(DataError::ValidationError(format!(
            "tensor {name:?} byte length {} != expected {expected}",
            slice.len()
        )));
    }
    match dtype {
        "F64" => {
            let mut values = Vec::with_capacity(slice.len() / 8);
            for chunk in slice.as_chunks::<8>().0 {
                let val = f64::from_le_bytes(*chunk);
                if require_finite && !val.is_finite() {
                    return Err(DataError::ValidationError(format!(
                        "tensor {name:?} contains non-finite float64 {val}"
                    )));
                }
                values.push(val);
            }
            Ok(TensorData::Float64 { shape, values })
        }
        "I64" => {
            let values = slice
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            Ok(TensorData::Int64 { shape, values })
        }
        other => Err(DataError::ValidationError(format!(
            "unsupported dtype {other:?} for {name:?}; expected F64 or I64"
        ))),
    }
}

/// Deserialize a `.safetensors` byte buffer into a [`SafeTensorBundle`].
///
/// # Errors
/// Returns [`DataError::ParseError`] on malformed header syntax or
/// [`DataError::ValidationError`] on invalid offsets, unsupported dtypes, or
/// non-finite `f64` values when `require_finite` is `true`.
pub fn decode_safetensors_bytes(
    data: &[u8],
    require_finite: bool,
) -> Result<SafeTensorBundle, DataError> {
    let (header_str, payload) = split_header_and_payload(data)?;
    let (metadata, raw_descs) = parse_safetensors_header(header_str)?;
    let mut tensors = BTreeMap::new();
    for (name, (dtype, shape, begin, end)) in raw_descs {
        if begin > end || end > payload.len() {
            return Err(DataError::ValidationError(format!(
                "invalid data_offsets [{begin}, {end}] for {name:?}"
            )));
        }
        let tensor = decode_tensor_slice(
            &name,
            &dtype,
            shape,
            &payload[begin..end],
            require_finite,
        )?;
        tensors.insert(name, tensor);
    }
    Ok(SafeTensorBundle { tensors, metadata })
}

/// Write a [`SafeTensorBundle`] to `path` and return its `SHA-256` hex digest.
///
/// # Errors
/// Returns [`DataError`] if validation or file writing fails.
pub fn save_safetensors_file(
    path: &Path,
    bundle: &SafeTensorBundle,
    require_finite: bool,
) -> Result<String, DataError> {
    let bytes = encode_safetensors_bytes(bundle, require_finite)?;
    let digest = compute_sha256_bytes(&bytes);
    fs::write(path, &bytes).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    Ok(digest)
}

/// Verify (if `expected_sha256` is given) and load a `.safetensors` file.
///
/// # Errors
/// Returns [`DataError`] on I/O failure, checksum mismatch, or invalid data.
pub fn load_safetensors_file(
    path: &Path,
    expected_sha256: Option<&str>,
    require_finite: bool,
) -> Result<SafeTensorBundle, DataError> {
    if let Some(expected) = expected_sha256 {
        if !is_valid_sha256_hex(expected) {
            return Err(DataError::ValidationError(format!(
                "invalid expected SHA-256 hex digest: {expected:?}"
            )));
        }
    }
    let bytes = fs::read(path).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    if let Some(expected) = expected_sha256 {
        let expected_norm = expected.trim().to_ascii_lowercase();
        let actual = compute_sha256_bytes(&bytes);
        if actual != expected_norm {
            return Err(DataError::ChecksumMismatch {
                path: path.display().to_string(),
                expected: expected_norm,
                actual,
            });
        }
    }
    decode_safetensors_bytes(&bytes, require_finite)
}
