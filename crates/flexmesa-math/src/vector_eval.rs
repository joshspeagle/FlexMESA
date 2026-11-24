// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared named-section evaluators and dispatch-overhead microbenchmarks for
//! MESA `26.4.1` hexadecimal reference vectors (`#165` Items 2, 14, 15, 17).

use std::hint::black_box;
use std::time::Instant;

use crate::{
    acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atanh, atanpi, cos,
    cosh, cospi, exp, exp10, expm1, fma, hypot, intrinsic_exp, intrinsic_log,
    intrinsic_log10, intrinsic_pow, log, log10, log1p, log2, pow, pow2, pow3,
    pow4, pow5, pow6, pow7, pow8, powi, powi_f90, powm1, safe_log, safe_log10,
    safe_sqrt, sin, sinh, sinpi, tan, tanh, tanpi, with_numerics_mode,
    NumericsMode,
};

/// Canonicalize `NaN` bit patterns to IEEE-754 positive quiet `NaN`
/// (`0x7ff8_0000_0000_0000`) while preserving all 64 bits (including `-0.0` vs
/// `+0.0`) on every non-`NaN` output.
#[must_use]
#[inline]
pub fn canonical_strict_bits(val: f64) -> u64 {
    if val.is_nan() {
        0x7ff8_0000_0000_0000_u64
    } else {
        val.to_bits()
    }
}

/// Evaluate a named unary `[unary:<section>]` reference vector under the
/// currently active [`NumericsMode`].
///
/// # Errors
/// Returns an error string if `section` is not a recognized unary section.
pub fn eval_unary_section(section: &str, x: f64) -> Result<f64, String> {
    let val = match section {
        "safe_sqrt" => safe_sqrt(x),
        "log" => log(x),
        "safe_log" => safe_log(x),
        "log10" => log10(x),
        "safe_log10" => safe_log10(x),
        "log1p" => log1p(x),
        "log2" => log2(x),
        "exp" => exp(x),
        "exp10" => exp10(x),
        "expm1" => expm1(x),
        "powm1" => powm1(x),
        "pow2" => pow2(x),
        "pow3" => pow3(x),
        "pow4" => pow4(x),
        "pow5" => pow5(x),
        "pow6" => pow6(x),
        "pow7" => pow7(x),
        "pow8" => pow8(x),
        "cos" => cos(x),
        "sin" => sin(x),
        "tan" => tan(x),
        "cospi" => cospi(x),
        "sinpi" => sinpi(x),
        "tanpi" => tanpi(x),
        "acos" => acos(x),
        "asin" => asin(x),
        "atan" => atan(x),
        "acospi" => acospi(x),
        "asinpi" => asinpi(x),
        "atanpi" => atanpi(x),
        "cosh" => cosh(x),
        "sinh" => sinh(x),
        "tanh" => tanh(x),
        "acosh" => acosh(x),
        "asinh" => asinh(x),
        "atanh" => atanh(x),
        "dlog" => intrinsic_log(x),
        "dlog10" | "intrinsic_log10" => intrinsic_log10(x),
        "intrinsic_exp" => intrinsic_exp(x),
        other => return Err(format!("unknown unary section: {other}")),
    };
    Ok(val)
}

/// Evaluate a named `[binary_f64:<section>]` reference vector under the
/// currently active [`NumericsMode`].
///
/// # Errors
/// Returns an error string if `section` is not a recognized `binary_f64`
/// section.
pub fn eval_binary_f64_section(
    section: &str,
    a: f64,
    b: f64,
) -> Result<f64, String> {
    let val = match section {
        "pow" => pow(a, b),
        "intrinsic_pow" => intrinsic_pow(a, b),
        "atan2" | "datan2" => atan2(a, b),
        "hypot" => hypot(a, b),
        other => return Err(format!("unknown binary_f64 section: {other}")),
    };
    Ok(val)
}

/// Evaluate a named `[binary_i32:<section>]` reference vector under the
/// currently active [`NumericsMode`].
///
/// # Errors
/// Returns an error string if `section` is not a recognized `binary_i32`
/// section.
pub fn eval_binary_i32_section(
    section: &str,
    a: f64,
    exp_i: i32,
) -> Result<f64, String> {
    let val = match section {
        "powi" => powi(a, exp_i),
        "powi_f90" => powi_f90(a, exp_i),
        other => return Err(format!("unknown binary_i32 section: {other}")),
    };
    Ok(val)
}

/// Measure `(strict_exp_ns, fast_exp_ns, strict_fma_ns, fast_fma_ns)` over
/// `n_iters` black-boxed calls (`#165` Item 2).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn measure_dispatch_overhead_ns(n_iters: usize) -> (f64, f64, f64, f64) {
    let iters = n_iters.max(1);
    let denom = iters as f64;

    let run_exp = |mode: NumericsMode| -> f64 {
        with_numerics_mode(mode, || {
            let mut acc = 0.0_f64;
            let t0 = Instant::now();
            for idx in 0..iters {
                let arg = black_box(0.5 + (idx & 255) as f64 * 0.001_953_125);
                acc += black_box(exp(arg));
            }
            let dt = t0.elapsed().as_nanos() as f64 / denom;
            black_box(acc);
            dt
        })
    };

    let run_fma = |mode: NumericsMode| -> f64 {
        with_numerics_mode(mode, || {
            let mut acc = 1.0_f64;
            let t0 = Instant::now();
            for idx in 0..iters {
                let arg = black_box(1.0 + (idx & 255) as f64 * 1.0e-7);
                acc = black_box(fma(acc, 0.999_999_9, arg * 1.0e-7));
            }
            let dt = t0.elapsed().as_nanos() as f64 / denom;
            black_box(acc);
            dt
        })
    };

    (
        run_exp(NumericsMode::Strict),
        run_exp(NumericsMode::Fast),
        run_fma(NumericsMode::Strict),
        run_fma(NumericsMode::Fast),
    )
}
