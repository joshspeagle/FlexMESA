// SPDX-License-Identifier: LGPL-3.0-or-later
//! Lossless ASCII readers for upstream MESA `26.4.1` `history.data`,
//! `profileN.data`, `binary_history.data`, and `profiles.index` files (`#163`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::controls::StarError;
use crate::mesa_profile::{ProfilesIndex, ProfilesIndexEntry};

/// Parsed scalar header value on line 3 of a MESA `history.data`, `profileN.data`,
/// or `binary_history.data` file.
#[derive(Debug, Clone, PartialEq)]
pub enum MesaHeaderValue {
    /// Quoted string literal (stored without surrounding quotes, e.g. `"r26.4.1"`).
    Text(String),
    /// 64-bit signed integer literal.
    Int(i64),
    /// IEEE-754 double-precision (`f64`) literal.
    Float(f64),
}

/// Parsed MESA 6-line-header ASCII table (`history.data`, `profileN.data`, or
/// `binary_history.data`).
#[derive(Debug, Clone)]
pub struct MesaLogTable {
    /// Ordered `(name, value)` header items from lines 2 and 3.
    pub header: Vec<(String, MesaHeaderValue)>,
    /// Ordered column names from line 6.
    pub column_names: Vec<String>,
    /// Positional `f64` column vectors aligned 1-to-1 with `column_names`
    /// (lossless even when `append_to_star_history` appends duplicate names).
    pub column_values: Vec<Vec<f64>>,
    /// Columnar `f64` data vectors keyed by first-occurrence column name.
    pub columns: HashMap<String, Vec<f64>>,
    /// Number of data rows (`lines 7+`).
    pub num_rows: usize,
}

impl MesaLogTable {
    /// Looks up a header item by name.
    #[must_use]
    pub fn header_value(&self, name: &str) -> Option<&MesaHeaderValue> {
        self.header
            .iter()
            .find_map(|(k, v)| (k == name).then_some(v))
    }

    /// Looks up a data column slice by name.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.columns.get(name).map(Vec::as_slice)
    }
}

fn parse_mesa_float_token(tok: &str) -> Result<f64, std::num::ParseFloatError> {
    if let Ok(fv) = tok.parse::<f64>() {
        return Ok(fv);
    }
    let normalized = tok.replace(['d', 'D'], "E");
    if let Ok(fv) = normalized.parse::<f64>() {
        return Ok(fv);
    }
    let bytes = normalized.as_bytes();
    for idx in (1..bytes.len()).rev() {
        let ch = bytes[idx];
        let prev = bytes[idx - 1];
        if (ch == b'+' || ch == b'-') && prev != b'E' && prev != b'e' {
            let candidate =
                format!("{}E{}", &normalized[..idx], &normalized[idx..]);
            return candidate.parse::<f64>();
        }
    }
    normalized.parse::<f64>()
}

fn tokenize_header_values(line: &str) -> Result<Vec<MesaHeaderValue>, StarError> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0_usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] == b'"' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(StarError::IoError(String::from(
                    "unterminated quoted string in MESA log header",
                )));
            }
            let s = &line[start..i];
            out.push(MesaHeaderValue::Text(s.to_owned()));
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let tok = &line[start..i];
            if let Ok(iv) = tok.parse::<i64>() {
                out.push(MesaHeaderValue::Int(iv));
            } else {
                let fv = parse_mesa_float_token(tok).map_err(|e| {
                    StarError::IoError(format!(
                        "invalid numeric token {tok:?} in MESA log header: {e}"
                    ))
                })?;
                out.push(MesaHeaderValue::Float(fv));
            }
        }
    }
    Ok(out)
}

/// Parses a MESA 6-line-header ASCII log (`history.data`, `profileN.data`, or
/// `binary_history.data`) from string `text`.
///
/// # Errors
/// Returns [`StarError::IoError`] if the 6-line header or any data row is
/// malformed or has mismatched column counts.
pub fn read_mesa_log_str(text: &str) -> Result<MesaLogTable, StarError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 6 {
        return Err(StarError::IoError(format!(
            "MESA log has {} lines; expected at least 6 header lines",
            lines.len()
        )));
    }
    let hdr_idx: Vec<usize> = lines[0]
        .split_whitespace()
        .map(|s| {
            s.parse::<usize>().map_err(|e| {
                StarError::IoError(format!("bad header index {s:?}: {e}"))
            })
        })
        .collect::<Result<_, _>>()?;
    let hdr_names: Vec<String> = lines[1]
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let hdr_vals = tokenize_header_values(lines[2])?;
    if hdr_idx.len() != hdr_names.len() || hdr_names.len() != hdr_vals.len() {
        return Err(StarError::IoError(format!(
            "MESA log header count mismatch: idx={}, names={}, vals={}",
            hdr_idx.len(),
            hdr_names.len(),
            hdr_vals.len()
        )));
    }
    let header: Vec<(String, MesaHeaderValue)> =
        hdr_names.into_iter().zip(hdr_vals).collect();

    let col_idx: Vec<usize> = lines[4]
        .split_whitespace()
        .map(|s| {
            s.parse::<usize>().map_err(|e| {
                StarError::IoError(format!("bad column index {s:?}: {e}"))
            })
        })
        .collect::<Result<_, _>>()?;
    let column_names: Vec<String> = lines[5]
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    if col_idx.len() != column_names.len() || column_names.is_empty() {
        return Err(StarError::IoError(format!(
            "MESA log column header mismatch: idx={}, names={}",
            col_idx.len(),
            column_names.len()
        )));
    }

    let ncols = column_names.len();
    let mut col_vecs: Vec<Vec<f64>> =
        vec![Vec::with_capacity(lines.len().saturating_sub(6)); ncols];
    let mut num_rows = 0_usize;
    for (offset, raw) in lines[6..].iter().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut seen = 0_usize;
        for (j, tok) in trimmed.split_whitespace().enumerate() {
            if j >= ncols {
                seen = j + 1 + trimmed.split_whitespace().skip(j + 1).count();
                break;
            }
            let v = parse_mesa_float_token(tok).map_err(|e| {
                StarError::IoError(format!(
                    "MESA log line {} col {}: invalid float {tok:?}: {e}",
                    offset + 7,
                    j + 1
                ))
            })?;
            col_vecs[j].push(v);
            seen = j + 1;
        }
        if seen != ncols {
            return Err(StarError::IoError(format!(
                "MESA log line {}: expected {ncols} columns, got {seen}",
                offset + 7
            )));
        }
        num_rows += 1;
    }

    let mut columns = HashMap::with_capacity(ncols);
    for (name, vec) in column_names.iter().zip(&col_vecs) {
        columns.entry(name.clone()).or_insert_with(|| vec.clone());
    }
    Ok(MesaLogTable {
        header,
        column_names,
        column_values: col_vecs,
        columns,
        num_rows,
    })
}

/// Reads and parses a MESA 6-line-header ASCII log file from `path`.
pub fn read_mesa_log_file(path: &Path) -> Result<MesaLogTable, StarError> {
    let text = fs::read_to_string(path).map_err(|e| {
        StarError::IoError(format!("failed to read {}: {e}", path.display()))
    })?;
    read_mesa_log_str(&text)
}

/// Parses a MESA `LOGS/profiles.index` string into [`ProfilesIndex`]
/// (`read_profiles_info` in `star/private/profile.f90:1138-1160`).
pub fn read_profiles_index_str(text: &str) -> Result<ProfilesIndex, StarError> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(first) = lines.next() else {
        return Ok(ProfilesIndex::default());
    };
    let Some(first_tok) = first.split_whitespace().next() else {
        return Err(StarError::IoError(String::from(
            "empty header line in profiles.index",
        )));
    };
    let declared: usize = first_tok.parse().map_err(|e| {
        StarError::IoError(format!(
            "invalid model count {first_tok:?} in profiles.index: {e}"
        ))
    })?;
    let mut entries = Vec::with_capacity(declared);
    for (idx, line) in lines.enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 3 {
            return Err(StarError::IoError(format!(
                "profiles.index entry line {} expected 3 integers, got {:?}",
                idx + 2,
                line
            )));
        }
        let model_number: usize = toks[0].parse().map_err(|e| {
            StarError::IoError(format!("bad model_number {:?}: {e}", toks[0]))
        })?;
        let priority: i32 = toks[1].parse().map_err(|e| {
            StarError::IoError(format!("bad priority {:?}: {e}", toks[1]))
        })?;
        let profile_number: usize = toks[2].parse().map_err(|e| {
            StarError::IoError(format!("bad profile_number {:?}: {e}", toks[2]))
        })?;
        entries.push(ProfilesIndexEntry {
            model_number,
            priority,
            profile_number,
        });
    }
    if entries.len() != declared {
        return Err(StarError::IoError(format!(
            "profiles.index declared {declared} models, found {}",
            entries.len()
        )));
    }
    Ok(ProfilesIndex { entries })
}

/// Reads and parses a MESA `LOGS/profiles.index` file from `path`.
pub fn read_profiles_index_file(path: &Path) -> Result<ProfilesIndex, StarError> {
    let text = fs::read_to_string(path).map_err(|e| {
        StarError::IoError(format!("failed to read {}: {e}", path.display()))
    })?;
    read_profiles_index_str(&text)
}
