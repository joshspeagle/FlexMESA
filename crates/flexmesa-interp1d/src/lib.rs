//! One-dimensional piecewise-monotonic and monotonicity-preserving cubic
//! spline interpolation for `FlexMESA`.
//!
//! This crate ports upstream MESA `interp_1d` (`interp_1d_def.f90`,
//! `interp_1d_lib.f90`, `interp_1d_pm.f90`, `interp_1d_mp.f90`, and
//! `interp_1d_misc.f90`) into Safe Rust in `f64` precision.
//!
//! # Coefficient Convention
//!
//! Given $N$ strictly monotonic knot abscissas $x_0, x_1, \dots, x_{N-1}$ and
//! values $y_0, y_1, \dots, y_{N-1}$, each spline builder returns a vector of
//! $N$ four-element coefficient arrays `[c0, c1, c2, c3]`. On interval
//! $[x_i, x_{i+1}]$ (or $[x_{i+1}, x_i]$ when $x$ is strictly decreasing), the
//! cubic polynomial for displacement $\Delta x = x - x_i$ is:
//! $$p_i(x) = c_{0,i} + \Delta x\,(c_{1,i} + \Delta x\,(
//!   c_{2,i} + \Delta x\,c_{3,i}))$$
//! where $c_{0,i} = y_i$, $c_{1,i} = p_i'(x_i)$,
//! $c_{2,i} = \tfrac{1}{2} p_i''(x_i)$, and
//! $c_{3,i} = \tfrac{1}{6} p_i'''(x_i)$.

#![forbid(unsafe_code)]
#![allow(
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::if_not_else,
    clippy::many_single_char_names,
    clippy::similar_names
)]

pub mod eval;
pub mod mp;
pub mod pm;
pub mod remesh;

use flexmesa_utils::is_bad;
use std::error::Error;
use std::fmt;

pub use eval::{
    integrate_values, interp_value, interp_value_and_slope, interp_values,
    interp_values_and_slopes,
};
pub use mp::{
    interp_m3, interp_m3a, interp_m3b, interp_m3q, median, minmod, minmod4,
};
pub use pm::{
    interp_4pt_pm, interp_pm, interp_pm_autodiff, interp_pm_slopes_only,
};
pub use remesh::{
    interp_3_to_1, interp_3_to_2, interp_4_to_1, interp_values_autodiff,
    interpolate_vector, interpolate_vector_pm, interpolate_vector_pm_autodiff,
};

/// Work array multiplier for Steffen (1990) piecewise-monotonic splines in
/// upstream MESA `interp_1d_def`.
pub const PM_WORK_SIZE: usize = 3;

/// Work array multiplier for Huynh (1993) monotonicity-preserving splines in
/// upstream MESA `interp_1d_def`.
pub const MP_WORK_SIZE: usize = 14;

/// Four cubic polynomial coefficients `[c0, c1, c2, c3]` at a single knot.
pub type CubicCoeffs = [f64; 4];

/// Slope-limiter variant for Huynh (1993) monotonicity-preserving cubic
/// interpolation (`interp_m3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MpLimiter {
    /// Average slope limiter (`interp_m3a`, upstream `average = 1`).
    Average = 1,
    /// Quartic median slope limiter (`interp_m3q`, upstream `quartic = 2`).
    Quartic = 2,
    /// Superbee slope limiter (`interp_m3b`, upstream `super_bee = 3`).
    SuperBee = 3,
}

/// Error returned by 1D spline construction, evaluation, or integration.
#[derive(Debug, Clone, PartialEq)]
pub enum Interp1dError {
    /// Input arrays `x` and `y` (or `coeffs`) have different lengths.
    LengthMismatch {
        /// Length of the knot abscissa slice.
        x_len: usize,
        /// Length of the data or coefficient slice.
        y_len: usize,
    },
    /// Fewer knots were provided than required by the interpolation routine.
    InsufficientPoints {
        /// Minimum required number of points.
        min_points: usize,
        /// Provided number of points.
        actual: usize,
    },
    /// Input array contains `NaN` or $\pm\infty$.
    NonFiniteInput,
    /// Knot abscissas are not strictly monotonic (duplicate or reversed step).
    NonMonotonicGrid {
        /// Zero-based interval index where monotonicity was violated.
        index: usize,
    },
    /// Integration query point lies outside the knot domain `[x[0], x[n - 1]]`
    /// or violates the grid ordering.
    IntegrationDomainError,
}

impl fmt::Display for Interp1dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { x_len, y_len } => write!(
                f,
                "x length ({x_len}) does not match y/coeffs length ({y_len})"
            ),
            Self::InsufficientPoints { min_points, actual } => write!(
                f,
                "interpolation requires at least {min_points} points, \
                 got {actual}"
            ),
            Self::NonFiniteInput => {
                write!(f, "interpolation input contains NaN or infinity")
            }
            Self::NonMonotonicGrid { index } => write!(
                f,
                "knot grid is not strictly monotonic at interval index {index}"
            ),
            Self::IntegrationDomainError => write!(
                f,
                "integration points must lie within knot range and follow \
                 knot order"
            ),
        }
    }
}

impl Error for Interp1dError {}

/// Validates that `x` and `y` have matching length `>= min_points`, contain
/// only finite values, and that `x` is strictly monotonic (increasing or
/// decreasing). Returns the interval widths $h_i = x_{i+1} - x_i$.
///
/// # Errors
///
/// Returns [`Interp1dError`] if lengths mismatch, `x.len() < min_points`, any
/// value is non-finite, or `x` is not strictly monotonic.
pub fn validate_grid_and_widths(
    x: &[f64],
    y: &[f64],
    min_points: usize,
) -> Result<Vec<f64>, Interp1dError> {
    if x.len() != y.len() {
        return Err(Interp1dError::LengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    let n = x.len();
    if n < min_points {
        return Err(Interp1dError::InsufficientPoints {
            min_points,
            actual: n,
        });
    }
    if x.iter().any(|&v| is_bad(v)) || y.iter().any(|&v| is_bad(v)) {
        return Err(Interp1dError::NonFiniteInput);
    }
    let mut h = Vec::with_capacity(n - 1);
    let h0 = x[1] - x[0];
    if h0 == 0.0 || is_bad(h0) {
        return Err(Interp1dError::NonMonotonicGrid { index: 0 });
    }
    let increasing = h0 > 0.0;
    h.push(h0);
    for i in 1..(n - 1) {
        let hi = x[i + 1] - x[i];
        if hi == 0.0 || is_bad(hi) || (hi > 0.0) != increasing {
            return Err(Interp1dError::NonMonotonicGrid { index: i });
        }
        h.push(hi);
    }
    Ok(h)
}
