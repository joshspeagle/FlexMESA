// SPDX-License-Identifier: GPL-3.0-or-later
//! Text writers for pulsation data: `write_gyre_data`
//! (`star/private/pulse_gyre.f90:419-474`), `write_fgong_data`
//! (`star/private/pulse_fgong.f90:633-708`) and `write_osc_data`
//! (`star/private/pulse_osc.f90:555-606`), with the gfortran edit
//! descriptors they use reproduced byte for byte (#491).
//!
//! `global_data` and `point_data` are as the `get_*_data` producers return
//! them: `point_data[j][c]` holds upstream `point_data(c+1, j+1)`.

use std::fmt::Write as _;

/// Errors of the pulsation-data writers and producers.
#[derive(Debug, Clone, PartialEq)]
pub enum PulseWriteError {
    /// An upstream `ierr /= 0` return or `mesa_error` stop.
    Upstream(String),
    /// A control value this port does not support.
    Unsupported(String),
}

impl std::fmt::Display for PulseWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upstream(m) => write!(f, "pulse data: {m}"),
            Self::Unsupported(m) => write!(f, "pulse data not supported: {m}"),
        }
    }
}

impl std::error::Error for PulseWriteError {}

/// gfortran output of `v` with the edit descriptor `1PEw.d` (`e = None`)
/// or `1PEw.dEe` (`e = Some(e)`): one digit before the decimal point, `d`
/// after it, and a signed exponent. Without `Ee` an exponent up to 99 is
/// written `E±dd` and one up to 999 as `±ddd`. A field that does not fit
/// in `w` characters is `w` asterisks. Digits are the correctly rounded
/// decimal expansion of the binary value, as gfortran writes them.
#[must_use]
pub fn fortran_1pe(v: f64, w: usize, d: usize, e: Option<usize>) -> String {
    let body = if v.is_nan() {
        String::from("NaN")
    } else if v.is_infinite() {
        let s = if v > 0.0 { "Infinity" } else { "-Infinity" };
        if s.len() <= w {
            String::from(s)
        } else if v > 0.0 {
            String::from("Inf")
        } else {
            String::from("-Inf")
        }
    } else {
        let s = format!("{v:.d$e}");
        let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
        let exp: i64 = exp.parse().unwrap_or(0);
        let mag = exp.unsigned_abs();
        let sign = if exp < 0 { '-' } else { '+' };
        let exp_field = match e {
            Some(ed) => {
                let digits = mag.to_string();
                if digits.len() > ed {
                    return "*".repeat(w);
                }
                format!("E{sign}{mag:0ed$}")
            }
            None if mag <= 99 => format!("E{sign}{mag:02}"),
            None if mag <= 999 => format!("{sign}{mag:03}"),
            None => return "*".repeat(w),
        };
        format!("{mant}{exp_field}")
    };
    if body.len() > w {
        "*".repeat(w)
    } else {
        format!("{body:>w$}")
    }
}

/// gfortran list-directed `write(unit, *) str`: a leading blank, then the
/// characters of `str` (with any trailing blanks it holds).
fn list_directed_line(out: &mut String, items: &[&str]) {
    out.push(' ');
    for item in items {
        out.push_str(item);
    }
    out.push('\n');
}

/// Writes `vals` with a format `(1P5Ew.d...)`-style repeat of five items
/// per record, each item `sep` followed by `fmt(v)`; a trailing `x` at the
/// end of the format writes nothing (gfortran does not emit trailing
/// positioning blanks).
fn write_records(
    out: &mut String,
    vals: &[f64],
    sep: &str,
    fmt: impl Fn(f64) -> String,
) {
    for chunk in vals.chunks(5) {
        for &v in chunk {
            out.push_str(sep);
            out.push_str(&fmt(v));
        }
        out.push('\n');
    }
}

/// `write_gyre_data(id, filename, global_data, point_data, ierr)`
/// (`star/private/pulse_gyre.f90:419-474`), returning the file text.
///
/// # Errors
///
/// [`PulseWriteError::Upstream`] for a schema other than 101 or 120
/// (`invalid gyre_data_schema`, `:440-446`).
pub fn write_gyre_data(
    global_data: &[f64],
    point_data: &[Vec<f64>],
    gyre_data_schema: i32,
) -> Result<String, PulseWriteError> {
    match gyre_data_schema {
        101 | 120 => {}
        _ => {
            return Err(PulseWriteError::Upstream(String::from(
                "invalid gyre_data_schema",
            )))
        }
    }
    let nn = point_data.len();
    let mut out = String::with_capacity(64 + nn * 27 * 19);
    // 100 format(I6, 3(1X,1PE26.16), 1X, I6)
    let _ = write!(out, "{nn:6}");
    for &v in global_data.iter().take(3) {
        out.push(' ');
        out.push_str(&fortran_1pe(v, 26, 16, None));
    }
    let _ = writeln!(out, " {gyre_data_schema:6}");
    // 110 format(I6, 99(1X,1PE26.16))
    for (j, row) in point_data.iter().enumerate() {
        let _ = write!(out, "{:6}", j + 1);
        for &v in row {
            out.push(' ');
            out.push_str(&fortran_1pe(v, 26, 16, None));
        }
        out.push('\n');
    }
    Ok(out)
}

/// `ICONST` and `IVAR` of FGONG versions 300 and 1300
/// (`star/private/pulse_fgong.f90:37-38`).
pub const FGONG_ICONST: usize = 15;
/// See [`FGONG_ICONST`].
pub const FGONG_IVAR: usize = 40;

/// `write_fgong_data(id, filename, global_data, point_data, ierr)`
/// (`star/private/pulse_fgong.f90:633-708`), returning the file text.
///
/// # Errors
///
/// [`PulseWriteError::Upstream`] for `fgong_ivers` other than 300 or 1300
/// (`:658-668`).
pub fn write_fgong_data(
    global_data: &[f64],
    point_data: &[Vec<f64>],
    fgong_header: &[String; 4],
    fgong_ivers: i32,
) -> Result<String, PulseWriteError> {
    let (sep, w, d, e) = match fgong_ivers {
        // '(1P5E16.9,x)'
        300 => ("", 16, 9, None),
        // '(1P,5(X,E26.18E3))'
        1300 => (" ", 26, 18, Some(3)),
        _ => {
            return Err(PulseWriteError::Upstream(format!(
                "bad fgong_ivers: must be 300 or 1300, not {fgong_ivers:4}"
            )))
        }
    };
    let nn = point_data.len();
    let mut out = String::with_capacity(256 + nn * 27 * FGONG_IVAR);
    for h in fgong_header {
        list_directed_line(&mut out, &[h.trim_end()]);
    }
    let _ = writeln!(
        out,
        "{nn:10}{FGONG_ICONST:10}{FGONG_IVAR:10}{fgong_ivers:10}"
    );
    let fmt = |v: f64| fortran_1pe(v, w, d, e);
    write_records(&mut out, global_data, sep, fmt);
    for row in point_data {
        write_records(&mut out, row, sep, fmt);
    }
    Ok(out)
}

/// `ICONST`, `IVAR` and `IABUND` of OSC version 2K
/// (`star/private/pulse_osc.f90:37-39`).
pub const OSC_ICONST: usize = 15;
/// See [`OSC_ICONST`].
pub const OSC_IVAR: usize = 22;
/// See [`OSC_ICONST`].
pub const OSC_IABUND: usize = 14;

/// `format_for_OSC_data` default (`star/defaults/controls.defaults`).
pub const DEFAULT_FORMAT_FOR_OSC_DATA: &str = "(1P5E19.12,x)";

/// `write_osc_data(id, filename, global_data, point_data, ierr)`
/// (`star/private/pulse_osc.f90:555-606`), returning the file text.
/// `version_number` is MESA's `character(len=24)` version string, written
/// with its trailing blanks by the list-directed header line.
///
/// # Errors
///
/// [`PulseWriteError::Unsupported`] for a `format_for_osc_data` other than
/// `(1P5E19.12,x)` or `(1P5E16.9,x)`: the record format is a run-time
/// Fortran format string, and only these are reproduced.
pub fn write_osc_data(
    global_data: &[f64],
    point_data: &[Vec<f64>],
    version_number: &str,
    format_for_osc_data: &str,
) -> Result<String, PulseWriteError> {
    let compact: String = format_for_osc_data
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_uppercase();
    let (w, d) = match compact.as_str() {
        "(1P5E19.12,X)" => (19, 12),
        "(1P5E16.9,X)" => (16, 9),
        _ => {
            return Err(PulseWriteError::Unsupported(format!(
                "format_for_osc_data = '{format_for_osc_data}' (only \
                 '(1P5E19.12,x)' and '(1P5E16.9,x)' are reproduced)"
            )))
        }
    };
    let nn = point_data.len();
    let mut out = String::with_capacity(256 + nn * w * 36);
    list_directed_line(&mut out, &["OSC file"]);
    let version = format!("{version_number:<24}");
    list_directed_line(&mut out, &["Created by MESAstar version", &version]);
    // `write(iounit, *)` with no items writes an empty record.
    out.push('\n');
    out.push('\n');
    out.push_str("14 H1 H2 He3 He4 Li7 Be7 C12 C13 N14 N15 O16 O17 Be9 Si28\n");
    let _ = writeln!(
        out,
        "{nn:10}{OSC_ICONST:10}{OSC_IVAR:10}{OSC_IABUND:10}{:10}",
        2000
    );
    let fmt = |v: f64| fortran_1pe(v, w, d, None);
    write_records(&mut out, global_data, "", fmt);
    for row in point_data {
        write_records(&mut out, row, "", fmt);
    }
    Ok(out)
}
