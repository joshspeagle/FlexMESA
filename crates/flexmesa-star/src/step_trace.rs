// SPDX-License-Identifier: GPL-3.0-or-later
//! Opt-in structured JSONL per-attempt step telemetry for diagnosing retry
//! cascades and timestep collapses in evolutionary campaigns (`#137`, `#150`).
//!
//! Telemetry is disabled unless the environment variable
//! `FLEXMESA_STEP_TRACE` is set. Its value is the first 1-based `step_number`
//! at which JSON lines are emitted (`FLEXMESA_STEP_TRACE=780` starts tracing
//! at step 780; an empty or non-numeric value traces from step 0). Each event
//! writes one JSON object per line to `FLEXMESA_STEP_TRACE_FILE` (appended if
//! set) or standard error (`stderr`).

use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

static TRACE_FROM: OnceLock<Option<usize>> = OnceLock::new();
static TRACE_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

/// Returns `true` when step telemetry is enabled for 1-based `step_number`.
#[must_use]
pub fn enabled(step_number: usize) -> bool {
    TRACE_FROM
        .get_or_init(|| {
            std::env::var("FLEXMESA_STEP_TRACE")
                .ok()
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        })
        .is_some_and(|from| step_number >= from)
}

/// Appends one serialized JSON line to `FLEXMESA_STEP_TRACE_FILE` if set, or
/// prints it to standard error.
pub fn emit_raw_json_line(line: &str) {
    let target = TRACE_FILE.get_or_init(|| {
        std::env::var("FLEXMESA_STEP_TRACE_FILE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .and_then(|path| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path.trim())
                    .ok()
                    .map(Mutex::new)
            })
    });
    if let Some(mtx) = target {
        if let Ok(mut f) = mtx.lock() {
            let _ = writeln!(f, "{line}");
            return;
        }
    }
    eprintln!("{line}");
}

fn push_json_escaped(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
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

/// Compact builder emitting single-line RFC 8259 JSON objects (`#150`).
#[derive(Debug, Clone)]
pub struct JsonLine {
    buf: String,
}

impl JsonLine {
    /// Starts a new JSON object with `"event":"<event>"`.
    #[must_use]
    pub fn new(event: &str) -> Self {
        let mut buf = String::with_capacity(256);
        buf.push_str("{\"event\":");
        push_json_escaped(&mut buf, event);
        Self { buf }
    }

    fn push_key(&mut self, key: &str) {
        self.buf.push(',');
        push_json_escaped(&mut self.buf, key);
        self.buf.push(':');
    }

    /// Appends a string field, JSON-escaping special characters.
    pub fn field_str(&mut self, key: &str, value: &str) {
        self.push_key(key);
        push_json_escaped(&mut self.buf, value);
    }

    /// Appends an integer field.
    pub fn field_usize(&mut self, key: &str, value: usize) {
        self.push_key(key);
        let _ = write!(self.buf, "{value}");
    }

    /// Appends a boolean field.
    pub fn field_bool(&mut self, key: &str, value: bool) {
        self.push_key(key);
        self.buf.push_str(if value { "true" } else { "false" });
    }

    /// Appends a floating-point field (`null` if non-finite).
    pub fn field_f64(&mut self, key: &str, value: f64) {
        self.push_key(key);
        if value.is_finite() {
            let _ = write!(self.buf, "{value:.6e}");
        } else {
            self.buf.push_str("null");
        }
    }

    /// Finalizes and returns the single-line JSON string.
    #[must_use]
    pub fn finish(mut self) -> String {
        self.buf.push('}');
        self.buf
    }

    /// Finalizes and emits the line via [`emit_raw_json_line`].
    pub fn emit(self) {
        let line = self.finish();
        emit_raw_json_line(&line);
    }
}

/// Locates the largest absolute entry of a residual vector laid out as
/// `equ[k * nvar + i]`, returning `(zone_k, equation_i, value)`.
#[must_use]
pub fn argmax_abs(equ: &[f64], nvar: usize) -> (usize, usize, f64) {
    let mut best = (0_usize, 0_usize, 0.0_f64);
    if nvar == 0 {
        return best;
    }
    for (idx, &v) in equ.iter().enumerate() {
        if v.abs() > best.2.abs() || v.is_nan() {
            best = (idx / nvar, idx % nvar, v);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::JsonLine;

    #[test]
    fn test_json_line_escapes_strings_and_handles_nonfinite_floats() {
        let mut line = JsonLine::new("test_event");
        line.field_usize("model", 12);
        line.field_bool("converged", false);
        line.field_f64("dt", 1.5e3);
        line.field_f64("bad", f64::NAN);
        line.field_str("msg", "quote\"slash\\nl\nend");
        let s = line.finish();
        assert!(!s.contains('\n'));
        assert!(s.starts_with("{\"event\":\"test_event\","));
        assert!(s.ends_with('}'));
        assert!(s.contains("\"bad\":null"));
        assert!(s.contains("\"msg\":\"quote\\\"slash\\\\nl\\nend\""));
    }
}
