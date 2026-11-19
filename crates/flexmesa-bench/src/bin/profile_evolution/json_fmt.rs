// SPDX-License-Identifier: LGPL-3.0-or-later
//! Minimal JSON serializer and Linux `/proc/self/status` RSS helpers for
//! `profile_evolution` (`#168`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_SEQ: AtomicU64 = AtomicU64::new(0);

/// RAII temporary copy of a read-only `test_suite` case directory.
pub struct StagedCaseGuard {
    tmp_root: PathBuf,
    pub staged_dir: PathBuf,
}

impl StagedCaseGuard {
    /// Copies `orig_case_dir` into a unique temporary directory under
    /// `std::env::temp_dir()`.
    pub fn new(orig_case_dir: &Path) -> Result<Self, String> {
        let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let case_name = orig_case_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("case");
        let tmp_root = std::env::temp_dir()
            .join(format!("flexmesa_prof_{pid}_{seq}_{case_name}"));
        let staged_dir = tmp_root.join(case_name);
        copy_case_tree(orig_case_dir, &staged_dir, 0)?;
        Ok(Self {
            tmp_root,
            staged_dir,
        })
    }
}

impl Drop for StagedCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.tmp_root);
    }
}

fn copy_case_tree(src: &Path, dst: &Path, depth: usize) -> Result<(), String> {
    if depth > 12 {
        return Err(String::from("exceeded directory nesting limit"));
    }
    fs::create_dir_all(dst).map_err(|e| format!("mkdir failed: {e}"))?;
    let entries =
        fs::read_dir(src).map_err(|e| format!("read_dir failed: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name == "docs" || name == "LOGS" || name == "photos" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        let meta = fs::metadata(&src_path).map_err(|e| e.to_string())?;
        if meta.is_dir() {
            copy_case_tree(&src_path, &dst_path, depth + 1)?;
        } else if meta.is_file() {
            fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("copy failed: {e}"))?;
        }
    }
    Ok(())
}

/// Appends a JSON-escaped string literal (including surrounding quotes) to `out`.
pub fn push_json_str(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Formats a finite `f64` or `"null"` when non-finite.
#[must_use]
pub fn fmt_json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.9e}")
    } else {
        String::from("null")
    }
}

/// Structured builder for indented JSON objects.
pub struct JsonObjectBuilder {
    buf: String,
    indent: usize,
    first: bool,
}

impl JsonObjectBuilder {
    /// Creates a new JSON object builder at `indent` spaces.
    #[must_use]
    pub fn new(indent: usize) -> Self {
        let mut buf = String::with_capacity(1024);
        buf.push_str("{\n");
        Self {
            buf,
            indent,
            first: true,
        }
    }

    fn write_key(&mut self, key: &str) {
        if !self.first {
            self.buf.push_str(",\n");
        }
        self.first = false;
        for _ in 0..(self.indent + 2) {
            self.buf.push(' ');
        }
        push_json_str(&mut self.buf, key);
        self.buf.push_str(": ");
    }

    /// Appends a string field.
    pub fn field_str(&mut self, key: &str, value: &str) {
        self.write_key(key);
        push_json_str(&mut self.buf, value);
    }

    /// Appends a boolean field.
    pub fn field_bool(&mut self, key: &str, value: bool) {
        self.write_key(key);
        self.buf.push_str(if value { "true" } else { "false" });
    }

    /// Appends a `usize` integer field.
    pub fn field_usize(&mut self, key: &str, value: usize) {
        self.write_key(key);
        let _ = write!(self.buf, "{value}");
    }

    /// Appends a `u64` hex string field (`"0x..."`) so JSON float parsers do not
    /// truncate 64-bit hashes.
    pub fn field_hex_u64(&mut self, key: &str, value: u64) {
        self.write_key(key);
        let _ = write!(self.buf, "\"0x{value:016x}\"");
    }

    /// Appends an `f64` field (`null` if non-finite).
    pub fn field_f64(&mut self, key: &str, value: f64) {
        self.write_key(key);
        self.buf.push_str(&fmt_json_f64(value));
    }

    /// Appends a pre-formatted raw JSON value (`{...}` or `[...]`).
    pub fn field_raw(&mut self, key: &str, raw_json: &str) {
        self.write_key(key);
        self.buf.push_str(raw_json);
    }

    /// Closes and returns the serialized JSON object string.
    #[must_use]
    pub fn finish(mut self) -> String {
        self.buf.push('\n');
        for _ in 0..self.indent {
            self.buf.push(' ');
        }
        self.buf.push('}');
        self.buf
    }
}

fn parse_proc_status_kib(field_prefix: &str) -> usize {
    let Ok(text) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field_prefix) {
            if let Some(num_tok) = rest.split_whitespace().next() {
                if let Ok(val) = num_tok.parse::<usize>() {
                    return val;
                }
            }
        }
    }
    0
}

/// Reads peak resident set size (`VmHWM`) in KiB on Linux (`0` elsewhere).
#[must_use]
pub fn peak_rss_kib() -> usize {
    parse_proc_status_kib("VmHWM:")
}

/// Reads current resident set size (`VmRSS`) in KiB on Linux (`0` elsewhere).
#[must_use]
pub fn current_rss_kib() -> usize {
    parse_proc_status_kib("VmRSS:")
}

#[cfg(test)]
mod tests {
    use super::{fmt_json_f64, JsonObjectBuilder};

    #[test]
    fn test_json_object_builder_escapes_and_formats_cleanly() {
        let mut obj = JsonObjectBuilder::new(0);
        obj.field_str("name", "star/\"test\"");
        obj.field_bool("ok", true);
        obj.field_usize("steps", 42);
        obj.field_hex_u64("hash", 0x1234_abcd);
        obj.field_f64("dt", 1.25);
        obj.field_f64("nan_val", f64::NAN);
        let out = obj.finish();
        assert!(out.contains("\"name\": \"star/\\\"test\\\"\""));
        assert!(out.contains("\"ok\": true"));
        assert!(out.contains("\"steps\": 42"));
        assert!(out.contains("\"hash\": \"0x000000001234abcd\""));
        assert!(out.contains("\"nan_val\": null"));
        assert_eq!(fmt_json_f64(f64::INFINITY), "null");
    }
}
