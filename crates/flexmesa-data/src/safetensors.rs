// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure Safe Rust `.safetensors` (`F64` and `I64`) encoder and decoder.
//!
//! Reads and writes standard `.safetensors` binary buffers (`8`-byte
//! little-endian header size + UTF-8 JSON header padded with ASCII spaces to an
//! 8-byte boundary + contiguous little-endian raw tensor bytes), interoperating
//! byte-for-byte with Python `safetensors.numpy`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::sha256::{compute_sha256_bytes, verify_file_sha256};
use crate::DataError;

/// Typed multi-dimensional row-major tensor stored in a `.safetensors` bundle.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// IEEE-754 double-precision (`F64`) tensor with shape and row-major values.
    Float64 {
        /// Dimension sizes along each axis.
        shape: Vec<usize>,
        /// Row-major flattened `f64` elements (`len == shape.iter().product()`).
        values: Vec<f64>,
    },
    /// Signed 64-bit integer (`I64`) tensor with shape and row-major values.
    Int64 {
        /// Dimension sizes along each axis.
        shape: Vec<usize>,
        /// Row-major flattened `i64` elements (`len == shape.iter().product()`).
        values: Vec<i64>,
    },
}

impl TensorData {
    /// Return the tensor shape slice.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Float64 { shape, .. } | Self::Int64 { shape, .. } => shape,
        }
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
fn checked_byte_len(name: &str, shape: &[usize]) -> Result<usize, DataError> {
    let elem_count = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            DataError::ValidationError(format!(
                "shape product overflow for tensor {name:?}"
            ))
        })?;
    elem_count.checked_mul(8).ok_or_else(|| {
        DataError::ValidationError(format!(
            "byte length overflow for tensor {name:?}"
        ))
    })
}

/// Escape a string for JSON serialization.
fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Validate a single [`TensorData`] entry and append its raw little-endian bytes.
fn encode_tensor_entry(
    name: &str,
    tensor: &TensorData,
    require_finite: bool,
    payload: &mut Vec<u8>,
) -> Result<String, DataError> {
    if name.trim().is_empty() || name == "__metadata__" {
        return Err(DataError::ValidationError(format!(
            "invalid tensor name: {name:?}"
        )));
    }
    let expected_bytes = checked_byte_len(name, tensor.shape())?;
    let begin = payload.len();
    let dtype_str = match tensor {
        TensorData::Float64 { values, .. } => {
            if values.len() * 8 != expected_bytes {
                return Err(DataError::ValidationError(format!(
                    "tensor {name:?} value count {} does not match shape {:?}",
                    values.len(),
                    tensor.shape()
                )));
            }
            if require_finite && !values.iter().all(|v| v.is_finite()) {
                return Err(DataError::ValidationError(format!(
                    "tensor {name:?} contains non-finite (NaN or Inf) float64 values"
                )));
            }
            for &v in values {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            "F64"
        }
        TensorData::Int64 { values, .. } => {
            if values.len() * 8 != expected_bytes {
                return Err(DataError::ValidationError(format!(
                    "tensor {name:?} value count {} does not match shape {:?}",
                    values.len(),
                    tensor.shape()
                )));
            }
            for &v in values {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            "I64"
        }
    };
    let end = payload.len();
    let shape_str = tensor
        .shape()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{}:{{\"dtype\":\"{dtype_str}\",\"shape\":[{shape_str}],\"data_offsets\":[{begin},{end}]}}",
        escape_json_str(name)
    ))
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
        let meta_entries = bundle
            .metadata
            .iter()
            .map(|(k, v)| {
                format!("{}:{}", escape_json_str(k), escape_json_str(v))
            })
            .collect::<Vec<_>>()
            .join(",");
        header_parts.push(format!("\"__metadata__\":{{{meta_entries}}}"));
    }
    let mut payload: Vec<u8> = Vec::new();
    for (name, tensor) in &bundle.tensors {
        header_parts.push(encode_tensor_entry(
            name,
            tensor,
            require_finite,
            &mut payload,
        )?);
    }
    let mut header_json = format!("{{{}}}", header_parts.join(","));
    while !header_json.len().is_multiple_of(8) {
        header_json.push(' ');
    }
    let header_len = header_json.len() as u64;
    let mut out = Vec::with_capacity(8 + header_json.len() + payload.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Minimal cursor for parsing `.safetensors` JSON headers.
struct JsonCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonCursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && self.bytes[self.pos].is_ascii_whitespace()
        {
            self.pos += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> Result<(), DataError> {
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(DataError::ParseError(format!(
                "expected {:?} at header byte {}",
                expected as char, self.pos
            )))
        }
    }

    fn parse_string(&mut self) -> Result<String, DataError> {
        self.consume(b'"')?;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                let esc = *self.bytes.get(self.pos).ok_or_else(|| {
                    DataError::ParseError(
                        "unterminated escape in JSON header".to_owned(),
                    )
                })?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => out.push(other as char),
                }
            } else {
                out.push(b as char);
            }
        }
        Err(DataError::ParseError(
            "unterminated string in JSON header".to_owned(),
        ))
    }

    fn parse_usize_list(&mut self) -> Result<Vec<usize>, DataError> {
        self.consume(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(items);
        }
        loop {
            self.skip_ws();
            let start = self.pos;
            while self.pos < self.bytes.len()
                && self.bytes[self.pos].is_ascii_digit()
            {
                self.pos += 1;
            }
            let num_str = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|e| DataError::ParseError(e.to_string()))?;
            let val = num_str.parse::<usize>().map_err(|_| {
                DataError::ParseError(format!(
                    "invalid integer {num_str:?} in JSON array"
                ))
            })?;
            items.push(val);
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b',') {
                self.pos += 1;
            } else {
                self.consume(b']')?;
                break;
            }
        }
        Ok(items)
    }
}

/// Raw parsed tensor header descriptor (`dtype`, `shape`, `[begin, end]`).
type RawTensorDesc = (String, Vec<usize>, usize, usize);

/// Parse the `"__metadata__"` object inside a `.safetensors` JSON header.
fn parse_metadata_object(
    cur: &mut JsonCursor<'_>,
) -> Result<BTreeMap<String, String>, DataError> {
    cur.consume(b'{')?;
    let mut meta = BTreeMap::new();
    cur.skip_ws();
    if cur.bytes.get(cur.pos) == Some(&b'}') {
        cur.pos += 1;
        return Ok(meta);
    }
    loop {
        let k = cur.parse_string()?;
        cur.consume(b':')?;
        let v = cur.parse_string()?;
        meta.insert(k, v);
        cur.skip_ws();
        if cur.bytes.get(cur.pos) == Some(&b',') {
            cur.pos += 1;
        } else {
            cur.consume(b'}')?;
            break;
        }
    }
    Ok(meta)
}

/// Parse one tensor descriptor object `{"dtype": ..., "shape": [...], "data_offsets": [b, e]}`.
fn parse_tensor_desc(
    cur: &mut JsonCursor<'_>,
) -> Result<RawTensorDesc, DataError> {
    cur.consume(b'{')?;
    let mut dtype: Option<String> = None;
    let mut shape: Option<Vec<usize>> = None;
    let mut offsets: Option<(usize, usize)> = None;
    loop {
        let field = cur.parse_string()?;
        cur.consume(b':')?;
        match field.as_str() {
            "dtype" => dtype = Some(cur.parse_string()?),
            "shape" => shape = Some(cur.parse_usize_list()?),
            "data_offsets" => {
                let list = cur.parse_usize_list()?;
                if list.len() != 2 {
                    return Err(DataError::ParseError(
                        "data_offsets must have 2 elements".to_owned(),
                    ));
                }
                offsets = Some((list[0], list[1]));
            }
            other => {
                return Err(DataError::ParseError(format!(
                    "unknown tensor header field {other:?}"
                )));
            }
        }
        cur.skip_ws();
        if cur.bytes.get(cur.pos) == Some(&b',') {
            cur.pos += 1;
        } else {
            cur.consume(b'}')?;
            break;
        }
    }
    match (dtype, shape, offsets) {
        (Some(d), Some(s), Some((b, e))) => Ok((d, s, b, e)),
        _ => Err(DataError::ParseError(
            "incomplete tensor descriptor in header".to_owned(),
        )),
    }
}

/// Decode a single tensor's raw little-endian slice into [`TensorData`].
fn decode_tensor_slice(
    name: &str,
    dtype: &str,
    shape: Vec<usize>,
    slice: &[u8],
    require_finite: bool,
) -> Result<TensorData, DataError> {
    let expected_bytes = checked_byte_len(name, &shape)?;
    if slice.len() != expected_bytes {
        return Err(DataError::ValidationError(format!(
            "tensor {name:?} byte length {} does not match shape {shape:?} (expected {expected_bytes})",
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
                        "tensor {name:?} contains non-finite float64 value {val}"
                    )));
                }
                values.push(val);
            }
            Ok(TensorData::Float64 { shape, values })
        }
        "I64" => {
            let mut values = Vec::with_capacity(slice.len() / 8);
            for chunk in slice.as_chunks::<8>().0 {
                values.push(i64::from_le_bytes(*chunk));
            }
            Ok(TensorData::Int64 { shape, values })
        }
        other => Err(DataError::ValidationError(format!(
            "unsupported tensor dtype {other:?} for {name:?}; expected F64 or I64"
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
    if data.len() < 8 {
        return Err(DataError::ParseError(
            "buffer shorter than 8-byte SafeTensors header prefix".to_owned(),
        ));
    }
    let header_len_u64 = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]);
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        DataError::ParseError("SafeTensors header length overflow".to_owned())
    })?;
    let payload_start = 8usize.checked_add(header_len).ok_or_else(|| {
        DataError::ParseError("SafeTensors header offset overflow".to_owned())
    })?;
    if payload_start > data.len() {
        return Err(DataError::ParseError(format!(
            "SafeTensors header length {header_len} exceeds buffer size {}",
            data.len()
        )));
    }
    let header_str = std::str::from_utf8(&data[8..payload_start])
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    let payload = &data[payload_start..];

    let mut cur = JsonCursor::new(header_str);
    cur.consume(b'{')?;
    let mut metadata = BTreeMap::new();
    let mut raw_descs: Vec<(String, RawTensorDesc)> = Vec::new();
    cur.skip_ws();
    while cur.bytes.get(cur.pos) != Some(&b'}') {
        let key = cur.parse_string()?;
        cur.consume(b':')?;
        if key == "__metadata__" {
            metadata = parse_metadata_object(&mut cur)?;
        } else {
            raw_descs.push((key, parse_tensor_desc(&mut cur)?));
        }
        cur.skip_ws();
        if cur.bytes.get(cur.pos) == Some(&b',') {
            cur.pos += 1;
        } else {
            cur.consume(b'}')?;
            break;
        }
    }
    let mut tensors = BTreeMap::new();
    for (name, (dtype, shape, begin, end)) in raw_descs {
        if begin > end || end > payload.len() {
            return Err(DataError::ValidationError(format!(
                "invalid data_offsets [{begin}, {end}] for tensor {name:?}"
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

/// Verify (if `expected_sha256` is given) and load a `.safetensors` file from disk.
///
/// # Errors
/// Returns [`DataError`] on I/O failure, checksum mismatch, or invalid tensor data.
pub fn load_safetensors_file(
    path: &Path,
    expected_sha256: Option<&str>,
    require_finite: bool,
) -> Result<SafeTensorBundle, DataError> {
    if let Some(expected) = expected_sha256 {
        verify_file_sha256(path, expected)?;
    }
    let bytes = fs::read(path).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    decode_safetensors_bytes(&bytes, require_finite)
}
