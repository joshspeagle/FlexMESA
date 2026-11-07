// SPDX-License-Identifier: LGPL-3.0-or-later
//! IEEE-754 `f64` elementary and domain-safe mathematical intrinsics (`math`).
//!
//! This crate ports MESA's `math` module (`math/public/math_lib_intrinsic.f90`,
//! `math/private/math_pown.f90`, and `math/private/math_io.f90`) to Safe Rust.
//! It provides domain-safe logarithm and square-root functions (`safe_log`,
//! `safe_log10`, `safe_sqrt`), base-10 and shifted exponentials (`exp10`,
//! `expm1`, `log1p`, `log2`), fixed integer-power helpers (`powm1`, `pow2`
//! through `pow8`), half-turn trigonometric functions (`sinpi`, `cospi`,
//! `tanpi`, `asinpi`, `acospi`, `atanpi`), Euclidean norm (`hypot`), and
//! sign/step utilities (`sgn`, `heaviside`).

#![forbid(unsafe_code)]

pub mod trig;

use flexmesa_const::{LN10, LN2};
use flexmesa_utils::is_bad;

pub use trig::{
    acos, acosh, acospi, asin, asinh, asinpi, atan, atanh, atanpi, cos, cosh,
    cospi, heaviside, sgn, sin, sinh, sinpi, str_to_double, str_to_vector, tan,
    tanh, tanpi,
};

/// Minimum positive floor applied by [`safe_log`] and [`safe_log10`].
pub const SAFE_LOG_FLOOR: f64 = 1.0e-99;

/// Fallback value returned by [`safe_log`] and [`safe_log10`] on non-finite
/// input.
pub const SAFE_LOG_BAD_RETURN: f64 = -99.0;

/// Compute the domain-safe square root $\sqrt{\max(x, 0)}$.
///
/// Negative inputs are clamped to `0.0` before taking the square root, whereas
/// `NaN` propagates unchanged.
///
/// # Examples
/// ```
/// use flexmesa_math::safe_sqrt;
///
/// assert_eq!(safe_sqrt(4.0), 2.0);
/// assert_eq!(safe_sqrt(-3.5), 0.0);
/// ```
#[must_use]
#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x.max(0.0).sqrt()
}

/// Compute the domain-safe natural logarithm $\ln(\max(10^{-99}, x))$.
///
/// Returns `-99.0` when `x` is non-finite (`NaN` or `±Inf`), matching MESA's
/// `safe_log` safeguard in `math_lib_intrinsic.f90`.
///
/// # Examples
/// ```
/// use flexmesa_math::safe_log;
///
/// assert_eq!(safe_log(1.0), 0.0);
/// assert_eq!(safe_log(f64::NAN), -99.0);
/// assert!((safe_log(-1.0) - 1.0e-99_f64.ln()).abs() < 1.0e-12);
/// ```
#[must_use]
#[inline]
pub fn safe_log(x: f64) -> f64 {
    if is_bad(x) {
        return SAFE_LOG_BAD_RETURN;
    }
    x.max(SAFE_LOG_FLOOR).ln()
}

/// Compute the domain-safe base-10 logarithm $\log_{10}(\max(10^{-99}, x))$.
///
/// Returns `-99.0` when `x` is non-finite (`NaN` or `±Inf`) or when
/// `x <= 10^{-99}`.
///
/// # Examples
/// ```
/// use flexmesa_math::safe_log10;
///
/// assert_eq!(safe_log10(10.0), 1.0);
/// assert_eq!(safe_log10(0.0), -99.0);
/// assert_eq!(safe_log10(f64::NAN), -99.0);
/// ```
#[must_use]
#[inline]
pub fn safe_log10(x: f64) -> f64 {
    if is_bad(x) {
        return SAFE_LOG_BAD_RETURN;
    }
    x.max(SAFE_LOG_FLOOR).log10()
}

/// Compute the natural logarithm $\ln(x)$.
#[must_use]
#[inline]
pub fn log(x: f64) -> f64 {
    x.ln()
}

/// Compute the base-10 logarithm $\log_{10}(x)$.
#[must_use]
#[inline]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Compute the shifted natural logarithm $\ln(1 + x)$ matching MESA's `log1p`.
#[must_use]
#[inline]
pub fn log1p(x: f64) -> f64 {
    (1.0 + x).ln()
}

/// Compute the base-2 logarithm $\log_{2}(x) = \ln(x) / \ln(2)$.
#[must_use]
#[inline]
pub fn log2(x: f64) -> f64 {
    x.ln() / LN2
}

/// Compute the natural exponential $e^{x}$.
#[must_use]
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Compute the shifted exponential $e^{x} - 1$ matching MESA's `expm1`.
#[must_use]
#[inline]
pub fn expm1(x: f64) -> f64 {
    x.exp() - 1.0
}

/// Compute the base-10 exponential $10^{x}$ matching MESA's `exp10`.
///
/// When `x` is a finite integer in `[-308, 308]`, `exp10` evaluates integer
/// exponentiation so integer decades match exact decimal powers; otherwise it
/// evaluates $\exp(x \ln 10)$.
///
/// # Examples
/// ```
/// use flexmesa_math::exp10;
///
/// assert_eq!(exp10(0.0), 1.0);
/// assert_eq!(exp10(3.0), 1000.0);
/// assert!((exp10(-2.0) - 0.01).abs() < 1.0e-16);
/// ```
#[must_use]
#[inline]
#[allow(clippy::float_cmp)]
pub fn exp10(x: f64) -> f64 {
    if x.is_finite() && x == x.floor() && (-308.0..=308.0).contains(&x) {
        #[allow(clippy::cast_possible_truncation)]
        let ix = x as i32;
        let mag = 10.0_f64.powi(ix.abs());
        if ix < 0 {
            1.0_f64 / mag
        } else {
            mag
        }
    } else {
        (x * LN10).exp()
    }
}

/// Compute the Euclidean hypotenuse $\sqrt{x^{2} + y^{2}}$ without overflow.
#[must_use]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Raise `x` to a floating-point power $x^{y}$.
#[must_use]
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Raise `x` to an integer power $x^{n}$.
#[must_use]
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// Compute the reciprocal $x^{-1} = 1 / x$.
#[must_use]
#[inline]
pub fn powm1(x: f64) -> f64 {
    1.0_f64 / x
}

/// Compute $x^{2}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow2(x: f64) -> f64 {
    x * x
}

/// Compute $x^{3}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow3(x: f64) -> f64 {
    x * x * x
}

/// Compute $x^{4}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow4(x: f64) -> f64 {
    x * x * x * x
}

/// Compute $x^{5}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow5(x: f64) -> f64 {
    x * x * x * x * x
}

/// Compute $x^{6}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow6(x: f64) -> f64 {
    x * x * x * x * x * x
}

/// Compute $x^{7}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow7(x: f64) -> f64 {
    x * x * x * x * x * x * x
}

/// Compute $x^{8}$ via repeated multiplication.
#[must_use]
#[inline]
pub fn pow8(x: f64) -> f64 {
    x * x * x * x * x * x * x * x
}
