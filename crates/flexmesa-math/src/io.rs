// SPDX-License-Identifier: LGPL-3.0-or-later
//! Sign, Heaviside step, and Fortran `math_io` string conversion helpers
//! (`math/private/math_io.f90` and `auto_diff/private/support_functions.f90`).

use core::fmt::Write as _;
use core::num::ParseFloatError;

/// Evaluate the three-valued signum function $\operatorname{sgn}(x)$ matching
/// `auto_diff/private/support_functions.f90`.
#[must_use]
#[inline]
pub fn sgn(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Evaluate the symmetric Heaviside step function $H(x)$ matching
/// `auto_diff/private/support_functions.f90`.
#[must_use]
#[inline]
pub fn heaviside(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x < 0.0 {
        0.0
    } else if x > 0.0 {
        1.0
    } else {
        0.5
    }
}

/// Parse a single Fortran or IEEE-754 floating-point token into `f64`
/// (`math/private/math_io.f90:84-92`), including `D`/`d` exponents and
/// 3-digit bare Fortran exponents such as `9.9999999999999997-106`.
///
/// # Errors
/// Returns [`ParseFloatError`] if the string cannot be parsed as a number.
pub fn str_to_double(token: &str) -> Result<f64, ParseFloatError> {
    let trimmed = token.trim();
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() || !trimmed.is_ascii() {
        return trimmed.parse::<f64>();
    }

    let mut has_exp_letter = false;
    let mut has_d_exp = false;
    let mut bare_sign_idx = None;
    for (idx, &b) in bytes.iter().enumerate() {
        match b {
            b'e' | b'E' => has_exp_letter = true,
            b'd' | b'D' => {
                has_exp_letter = true;
                has_d_exp = true;
            }
            b'+' | b'-' if idx > 0 => {
                let prev = bytes[idx - 1];
                if prev.is_ascii_digit() || prev == b'.' {
                    bare_sign_idx = Some(idx);
                }
            }
            _ => {}
        }
    }

    if !has_d_exp && (has_exp_letter || bare_sign_idx.is_none()) {
        return trimmed.parse::<f64>();
    }

    if bytes.len() < 63 {
        let mut buf = [0_u8; 64];
        let mut out_len = 0_usize;
        let insert_e_at = if has_exp_letter { None } else { bare_sign_idx };
        for (idx, &b) in bytes.iter().enumerate() {
            if insert_e_at == Some(idx) {
                buf[out_len] = b'e';
                out_len += 1;
            }
            buf[out_len] = if b == b'd' || b == b'D' { b'e' } else { b };
            out_len += 1;
        }
        let s = core::str::from_utf8(&buf[..out_len]).unwrap_or(trimmed);
        s.parse::<f64>()
    } else {
        let mut norm = String::with_capacity(bytes.len() + 1);
        let insert_e_at = if has_exp_letter { None } else { bare_sign_idx };
        for (idx, ch) in trimmed.char_indices() {
            if insert_e_at == Some(idx) {
                norm.push('e');
            }
            norm.push(if ch == 'd' || ch == 'D' { 'e' } else { ch });
        }
        norm.parse::<f64>()
    }
}

/// Parse a whitespace-delimited string of Fortran/IEEE-754 numbers into a
/// `Vec<f64>` (`math/private/math_io.f90:34-81`).
///
/// # Errors
/// Returns [`ParseFloatError`] if any whitespace-separated token is invalid.
pub fn str_to_vector(input: &str) -> Result<Vec<f64>, ParseFloatError> {
    input.split_whitespace().map(str_to_double).collect()
}

/// Format an `f64` into MESA's 26-character `1PD26.16` string representation
/// (`math/private/math_io.f90:95-105`).
#[must_use]
pub fn double_to_str(x: f64) -> String {
    if x.is_nan() {
        return format!("{:>26}", "NaN");
    }
    if x.is_infinite() {
        let tok = if x.is_sign_negative() {
            "-Infinity"
        } else {
            "Infinity"
        };
        return format!("{tok:>26}");
    }
    let sign_prefix = if x.is_sign_negative() { '-' } else { ' ' };
    let abs_x = x.abs();
    if abs_x == 0.0 {
        return format!("   {sign_prefix}0.0000000000000000D+00");
    }
    let sci = format!("{abs_x:.16e}");
    let (mant, exp_str) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp_val: i32 = exp_str.parse().unwrap_or(0);
    let exp_sign = if exp_val < 0 { '-' } else { '+' };
    let exp_abs = exp_val.unsigned_abs();
    let mut out = String::with_capacity(26);
    if exp_abs >= 100 {
        let _ = write!(out, "   {sign_prefix}{mant}{exp_sign}{exp_abs:03}");
    } else {
        let _ = write!(out, "   {sign_prefix}{mant}D{exp_sign}{exp_abs:02}");
    }
    if out.len() == 26 {
        out
    } else {
        format!("{:>26}", out.trim_start())
    }
}
