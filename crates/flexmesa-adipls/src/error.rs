// SPDX-License-Identifier: GPL-3.0-or-later
//! Error types for MESA 26.4.1 ADIPLS parameter parsing, FGONG/AMDL
//! conversion, control setup, and mode output processing.

use std::fmt;

/// Errors returned by `flexmesa-adipls` operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AdiplsError {
    /// Filesystem I/O failure while reading an ADIPLS or redistribution file.
    Io {
        /// Optional path associated with the failed I/O operation.
        path: Option<String>,
        /// Human-readable description of the underlying I/O error.
        message: String,
    },
    /// Parse failure on `adipls.c.pruned.in` or `redistrb.c.pruned.in`.
    Parse {
        /// Kind of file being parsed (e.g., `"adipls.c.pruned.in"`).
        kind: &'static str,
        /// 1-based source line number where parsing failed (0 if global/EOF).
        line: usize,
        /// Description of the syntax or type mismatch.
        message: String,
    },
    /// Invalid FGONG or AMDL model geometry, dimensions, or unsupported
    /// upstream out-of-bounds state (`DECISIONS #43`).
    InvalidModel {
        /// Description of the invalid model state.
        message: String,
    },
    /// Attempted to read an explicitly unset `aa` row (`#479`, `DECISIONS #43`).
    ///
    /// Upstream `store_amdl` leaves rows `6..=9` (1-based) carrying
    /// uninitialized memory because `fgong_amdl` only populates rows `1..=6`
    /// and `store_amdl` shifts `aa(i, n) = aa1(i + 1, n1)` for `i = 1..=6`.
    /// This port refuses to fabricate or copy uninitialized values and returns
    /// this error on any read of an unset row.
    UnsetAmdlRow {
        /// 1-based row index (`1..=10`) that was never populated.
        row: usize,
    },
    /// Out-of-bounds 1-based row or 0-based zone index into AMDL `aa`.
    InvalidAmdlIndex {
        /// 1-based row index requested.
        row: usize,
        /// 0-based zone index requested.
        zone: usize,
        /// Number of zones `nn` in the AMDL model.
        nn: usize,
    },
    /// Out-of-bounds 1-based index into `csummm(1:50)`.
    InvalidCsummmIndex {
        /// 1-based index requested (valid range `1..=50`).
        index: usize,
    },
    /// Non-finite or out-of-range floating-point value during conversion to
    /// an integer mode output field.
    ConversionOverflow {
        /// Name of the field being converted.
        field: &'static str,
        /// Offending `f64` value.
        value: f64,
    },
    /// Attempted to execute an unported or unsupported ADIPLS control path
    /// (`#147`, `CLAUDE.md` loud failure on unsupported inputs).
    UnsupportedControl {
        /// Name of the control parameter or option.
        control: &'static str,
        /// Human-readable description of the unsupported setting.
        message: String,
    },
    /// Runtime error encountered inside the ADIPLS scan or eigenvalue solver.
    Solver {
        /// Upstream ADIPLS routine name where the condition occurred.
        routine: &'static str,
        /// Description of the solver error condition.
        message: String,
    },
}

impl fmt::Display for AdiplsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path: Some(p), message } => {
                write!(f, "I/O error reading '{p}': {message}")
            }
            Self::Io { path: None, message } => {
                write!(f, "I/O error: {message}")
            }
            Self::Parse { kind, line, message } => {
                if *line == 0 {
                    write!(f, "failed parsing {kind}: {message}")
                } else {
                    write!(f, "failed parsing {kind} at line {line}: {message}")
                }
            }
            Self::InvalidModel { message } => {
                write!(f, "invalid FGONG/AMDL model: {message}")
            }
            Self::UnsetAmdlRow { row } => write!(
                f,
                "attempted to read unset AMDL aa row {row} (#479, DECISIONS #43)"
            ),
            Self::InvalidAmdlIndex { row, zone, nn } => write!(
                f,
                "AMDL aa index out of bounds: 1-based row {row} (expected 1..=10), zone {zone} (nn = {nn})"
            ),
            Self::InvalidCsummmIndex { index } => write!(
                f,
                "csummm index {index} out of bounds (expected 1..=50)"
            ),
            Self::ConversionOverflow { field, value } => write!(
                f,
                "cannot convert non-finite or out-of-range {field} value {value} to i32"
            ),
            Self::UnsupportedControl { control, message } => write!(
                f,
                "unsupported ADIPLS control '{control}': {message}"
            ),
            Self::Solver { routine, message } => {
                write!(f, "ADIPLS solver error in {routine}: {message}")
            }
        }
    }
}

impl std::error::Error for AdiplsError {}
