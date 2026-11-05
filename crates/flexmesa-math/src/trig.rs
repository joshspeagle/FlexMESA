// SPDX-License-Identifier: LGPL-3.0-or-later
//! Trigonometric, half-turn (`pi`-scaled), hyperbolic, sign, and string
//! parsing functions for `flexmesa-math`.

use core::num::ParseFloatError;

use flexmesa_const::PI;

/// Compute $\cos(x)$ for $x$ in radians.
#[must_use]
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Compute $\sin(x)$ for $x$ in radians.
#[must_use]
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Compute $\tan(x)$ for $x$ in radians.
#[must_use]
#[inline]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Compute $\cos(\pi x)$ for half-turn argument $x$.
#[must_use]
#[inline]
pub fn cospi(x: f64) -> f64 {
    (x * PI).cos()
}

/// Compute $\sin(\pi x)$ for half-turn argument $x$.
#[must_use]
#[inline]
pub fn sinpi(x: f64) -> f64 {
    (x * PI).sin()
}

/// Compute $\tan(\pi x)$ for half-turn argument $x$.
#[must_use]
#[inline]
pub fn tanpi(x: f64) -> f64 {
    (x * PI).tan()
}

/// Compute $\arccos(x)$ in radians.
#[must_use]
#[inline]
pub fn acos(x: f64) -> f64 {
    x.acos()
}

/// Compute $\arcsin(x)$ in radians.
#[must_use]
#[inline]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// Compute $\arctan(x)$ in radians.
#[must_use]
#[inline]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Compute $\arccos(x) / \pi$ in half-turns.
#[must_use]
#[inline]
pub fn acospi(x: f64) -> f64 {
    x.acos() / PI
}

/// Compute $\arcsin(x) / \pi$ in half-turns.
#[must_use]
#[inline]
pub fn asinpi(x: f64) -> f64 {
    x.asin() / PI
}

/// Compute $\arctan(x) / \pi$ in half-turns.
#[must_use]
#[inline]
pub fn atanpi(x: f64) -> f64 {
    x.atan() / PI
}

/// Compute $\cosh(x)$.
#[must_use]
#[inline]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

/// Compute $\sinh(x)$.
#[must_use]
#[inline]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// Compute $\tanh(x)$.
#[must_use]
#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Compute $\operatorname{arcosh}(x)$.
#[must_use]
#[inline]
pub fn acosh(x: f64) -> f64 {
    x.acosh()
}

/// Compute $\operatorname{arsinh}(x)$.
#[must_use]
#[inline]
pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

/// Compute $\operatorname{artanh}(x)$.
#[must_use]
#[inline]
pub fn atanh(x: f64) -> f64 {
    x.atanh()
}

/// Evaluate the three-valued signum function $\operatorname{sgn}(x)$.
///
/// Returns `-1.0` when `x < 0.0`, `0.0` when `x == 0.0`, `+1.0` when `x > 0.0`,
/// and `NaN` when `x` is `NaN`, matching MESA's `sgn` helper in
/// `auto_diff/private/support_functions.f90`.
///
/// # Examples
/// ```
/// use flexmesa_math::sgn;
///
/// assert_eq!(sgn(-2.5), -1.0);
/// assert_eq!(sgn(0.0), 0.0);
/// assert_eq!(sgn(2.5), 1.0);
/// ```
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

/// Evaluate the symmetric Heaviside step function $H(x)$.
///
/// Returns `0.0` when `x < 0.0`, `0.5` when `x == 0.0`, `1.0` when `x > 0.0`,
/// and `NaN` when `x` is `NaN`, matching MESA's `Heaviside` helper in
/// `auto_diff/private/support_functions.f90`.
///
/// # Examples
/// ```
/// use flexmesa_math::heaviside;
///
/// assert_eq!(heaviside(-1.0), 0.0);
/// assert_eq!(heaviside(0.0), 0.5);
/// assert_eq!(heaviside(1.0), 1.0);
/// ```
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

/// Parse a single Fortran or IEEE-754 floating-point token into `f64`.
///
/// Normalizes Fortran `D`/`d` exponent markers to `e` before parsing.
///
/// # Errors
/// Returns [`ParseFloatError`] if the string cannot be parsed as a number.
pub fn str_to_double(token: &str) -> Result<f64, ParseFloatError> {
    let trimmed = token.trim();
    if trimmed.contains(['d', 'D']) {
        trimmed.replace(['d', 'D'], "e").parse::<f64>()
    } else {
        trimmed.parse::<f64>()
    }
}

/// Parse a whitespace-delimited string of Fortran/IEEE-754 numbers into a
/// `Vec<f64>`.
///
/// # Errors
/// Returns [`ParseFloatError`] if any whitespace-separated token is invalid.
///
/// # Examples
/// ```
/// use flexmesa_math::str_to_vector;
///
/// let vals = str_to_vector("1.0D0  2.5d-1  -3.0").unwrap();
/// assert_eq!(vals, vec![1.0, 0.25, -3.0]);
/// ```
pub fn str_to_vector(input: &str) -> Result<Vec<f64>, ParseFloatError> {
    input.split_whitespace().map(str_to_double).collect()
}
