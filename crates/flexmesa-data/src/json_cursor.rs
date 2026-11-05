// SPDX-License-Identifier: LGPL-3.0-or-later
//! Minimal Safe Rust JSON header parser for `.safetensors` buffers.
//!
//! Parses the UTF-8 JSON header block of a `.safetensors` file (`__metadata__`
//! string maps and per-tensor `{"dtype", "shape", "data_offsets"}` objects)
//! without external parser dependencies or `unsafe` code.
//!
//! Design invariants:
//! - Rejects unknown tensor descriptor fields and malformed `data_offsets`.
//! - Rejects unterminated strings, invalid integer tokens, and missing keys.
//! - Preserves exact tensor insertion order for offset validation.

use std::collections::BTreeMap;

use crate::DataError;

/// Raw parsed tensor header descriptor (`dtype`, `shape`, `begin`, `end`).
///
/// Holds the validated dtype string, shape axis vector, and `[begin, end]`
/// byte offset range within the `.safetensors` payload buffer.
pub(crate) type RawTensorDesc = (String, Vec<usize>, usize, usize);

/// Parsed `.safetensors` JSON header (`metadata` and tensor descriptors).
pub(crate) type ParsedHeader =
    (BTreeMap<String, String>, Vec<(String, RawTensorDesc)>);

/// Minimal byte cursor for parsing `.safetensors` JSON headers.
///
/// Operates directly on the validated UTF-8 header string and tracks the
/// current byte offset `pos`.
pub(crate) struct JsonCursor<'a> {
    /// Underlying UTF-8 header string slice.
    text: &'a str,
    /// Underlying UTF-8 header byte slice.
    bytes: &'a [u8],
    /// Current read position within `bytes`.
    pos: usize,
}

impl<'a> JsonCursor<'a> {
    /// Create a new [`JsonCursor`] positioned at byte 0 of `text`.
    pub(crate) fn new(text: &'a str) -> Self {
        Self {
            text,
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    /// Advance `pos` past any ASCII whitespace characters (`' '`, `'\t'`,
    /// `'\n'`, `'\r'`).
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && self.bytes[self.pos].is_ascii_whitespace()
        {
            self.pos += 1;
        }
    }

    /// Skip whitespace and consume a single expected ASCII byte `expected`.
    ///
    /// Returns [`DataError::ParseError`] if the next non-whitespace byte does
    /// not equal `expected`.
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

    /// Parse a double-quoted JSON string, unescaping `\"`, `\\`, `\n`, `\r`,
    /// and `\t` while preserving multi-byte UTF-8 characters.
    ///
    /// Returns [`DataError::ParseError`] on unterminated strings or escapes.
    pub(crate) fn parse_string(&mut self) -> Result<String, DataError> {
        self.consume(b'"')?;
        let mut out = String::new();
        while let Some(ch) = self.text[self.pos..].chars().next() {
            self.pos += ch.len_utf8();
            if ch == '"' {
                return Ok(out);
            }
            if ch == '\\' {
                let esc =
                    self.text[self.pos..].chars().next().ok_or_else(|| {
                        DataError::ParseError("unterminated escape".to_owned())
                    })?;
                self.pos += esc.len_utf8();
                out.push(match esc {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                });
            } else {
                out.push(ch);
            }
        }
        Err(DataError::ParseError(
            "unterminated string in JSON header".to_owned(),
        ))
    }

    /// Parse a bracketed JSON array of non-negative integers (`[d0, d1, ...]`).
    ///
    /// Used for both `"shape"` arrays and 2-element `"data_offsets"` pairs.
    pub(crate) fn parse_usize_list(&mut self) -> Result<Vec<usize>, DataError> {
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
            let s = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|e| DataError::ParseError(e.to_string()))?;
            items.push(s.parse::<usize>().map_err(|_| {
                DataError::ParseError(format!("invalid integer {s:?}"))
            })?);
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

    /// Parse a brace-delimited JSON object `{ "key": value, ... }`, invoking
    /// `on_entry` after each `"key":` colon.
    pub(crate) fn parse_object<F>(
        &mut self,
        mut on_entry: F,
    ) -> Result<(), DataError>
    where
        F: FnMut(&mut Self, String) -> Result<(), DataError>,
    {
        self.consume(b'{')?;
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            let key = self.parse_string()?;
            self.consume(b':')?;
            on_entry(self, key)?;
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b',') {
                self.pos += 1;
            } else {
                self.consume(b'}')?;
                break;
            }
        }
        Ok(())
    }
}

/// Escape a string for JSON serialization (`"`, `\`, `\n`, `\r`, `\t`).
///
/// Wraps `s` in ASCII double quotes and escapes control characters and quotes
/// so the returned string is a valid JSON string literal.
pub(crate) fn escape_json_str(s: &str) -> String {
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

/// Format a `"__metadata__":{...}` JSON entry from a string map.
///
/// Serializes each `(key, value)` pair in deterministic `BTreeMap` order using
/// [`escape_json_str`].
pub(crate) fn format_metadata_entry(meta: &BTreeMap<String, String>) -> String {
    let pairs: Vec<String> = meta
        .iter()
        .map(|(k, v)| format!("{}:{}", escape_json_str(k), escape_json_str(v)))
        .collect();
    format!("\"__metadata__\":{{{}}}", pairs.join(","))
}

/// Format a per-tensor JSON header entry with `"dtype"`, `"shape"`, and
/// `"data_offsets": [offset, end]`.
///
/// Produces a compact JSON key-value fragment suitable for joining into the
/// top-level `.safetensors` header object.
pub(crate) fn format_tensor_entry(
    name: &str,
    dtype: &str,
    shape: &[usize],
    offset: usize,
    end: usize,
) -> String {
    let shape_str = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}:{{\"dtype\":\"{dtype}\",\"shape\":[{shape_str}],\
         \"data_offsets\":[{offset},{end}]}}",
        escape_json_str(name)
    )
}

/// Parse one tensor descriptor object with `"dtype"`, `"shape"`, and
/// `"data_offsets": [begin, end]`.
///
/// Rejects unknown or duplicate JSON keys and verifies that `"data_offsets"`
/// contains exactly two non-negative byte offsets `[begin, end]`.
pub(crate) fn parse_tensor_desc(
    cur: &mut JsonCursor<'_>,
) -> Result<RawTensorDesc, DataError> {
    let (mut dtype, mut shape, mut offsets) = (None, None, None);
    cur.parse_object(|c, field| {
        match field.as_str() {
            "dtype" if dtype.is_none() => dtype = Some(c.parse_string()?),
            "shape" if shape.is_none() => shape = Some(c.parse_usize_list()?),
            "data_offsets" if offsets.is_none() => {
                let list = c.parse_usize_list()?;
                let [b, e] = list.as_slice() else {
                    return Err(DataError::ParseError(
                        "data_offsets must have 2 elements".to_owned(),
                    ));
                };
                offsets = Some((*b, *e));
            }
            other => {
                return Err(DataError::ParseError(format!(
                    "unknown or duplicate tensor header field {other:?}"
                )));
            }
        }
        Ok(())
    })?;
    match (dtype, shape, offsets) {
        (Some(d), Some(s), Some((b, e))) => Ok((d, s, b, e)),
        _ => Err(DataError::ParseError(
            "incomplete tensor descriptor in header".to_owned(),
        )),
    }
}

/// Split a `.safetensors` byte buffer into `(header_json_str, payload_bytes)`.
///
/// Reads the 8-byte little-endian `u64` header length prefix, validates that
/// `8 + header_len` does not overflow or exceed `data.len()`, and decodes the
/// JSON header slice as UTF-8.
pub(crate) fn split_header_and_payload(
    data: &[u8],
) -> Result<(&str, &[u8]), DataError> {
    let Some(prefix) = data.first_chunk::<8>() else {
        return Err(DataError::ParseError(
            "buffer shorter than 8-byte SafeTensors header prefix".to_owned(),
        ));
    };
    let header_len =
        usize::try_from(u64::from_le_bytes(*prefix)).map_err(|_| {
            DataError::ParseError(
                "SafeTensors header length overflow".to_owned(),
            )
        })?;
    let payload_start = 8usize.checked_add(header_len).ok_or_else(|| {
        DataError::ParseError("SafeTensors header offset overflow".to_owned())
    })?;
    if payload_start > data.len() {
        return Err(DataError::ParseError(format!(
            "header length {header_len} exceeds buffer size {}",
            data.len()
        )));
    }
    let header_str = std::str::from_utf8(&data[8..payload_start])
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    Ok((header_str, &data[payload_start..]))
}

/// Parse the complete `.safetensors` JSON header string into metadata and
/// tensor descriptors.
///
/// Extracts the optional `"__metadata__"` string-to-string dictionary and all
/// top-level tensor descriptor objects in appearance order, rejecting trailing
/// non-whitespace characters or duplicate metadata keys.
///
/// # Errors
/// Returns [`DataError::ParseError`] if the JSON syntax is invalid or if any
/// tensor descriptor is missing `"dtype"`, `"shape"`, or `"data_offsets"`.
pub(crate) fn parse_safetensors_header(
    header_str: &str,
) -> Result<ParsedHeader, DataError> {
    let mut cur = JsonCursor::new(header_str);
    let mut metadata = BTreeMap::new();
    let mut seen_meta = false;
    let mut raw_descs: Vec<(String, RawTensorDesc)> = Vec::new();
    cur.parse_object(|c, key| {
        if key == "__metadata__" {
            if seen_meta {
                return Err(DataError::ParseError(
                    "duplicate __metadata__ in header".to_owned(),
                ));
            }
            seen_meta = true;
            c.parse_object(|mc, mk| {
                let mv = mc.parse_string()?;
                if metadata.insert(mk, mv).is_some() {
                    return Err(DataError::ParseError(
                        "duplicate metadata key in header".to_owned(),
                    ));
                }
                Ok(())
            })
        } else {
            raw_descs.push((key, parse_tensor_desc(c)?));
            Ok(())
        }
    })?;
    cur.skip_ws();
    if cur.pos != cur.bytes.len() {
        return Err(DataError::ParseError(
            "unexpected trailing characters after JSON header".to_owned(),
        ));
    }
    Ok((metadata, raw_descs))
}
