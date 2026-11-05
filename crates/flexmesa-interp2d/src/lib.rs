//! Two-dimensional bicubic spline, piecewise-monotonic cubic, and Akima (1996)
//! bivariate surface interpolation for `FlexMESA`.
//!
//! This crate ports upstream MESA `interp_2d` (`interp_2d_lib_db.f90`,
//! `bicub_db.f`, `bipm_db.f90`, and `akima760_db.f`) into Safe Rust in `f64`
//! precision.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::if_not_else,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]

pub mod akima;
pub mod akima_pd;
pub mod bicub;
pub mod bicub_eval;
pub mod bipm;
pub mod spline1d;

use std::error::Error;
use std::fmt;

pub use akima::{interp_rgbi3p_db, interp_rgsf3p_db, rgpd3p_db};
pub use bicub::{
    herm2xy_db, interp_evbicub_db, interp_evbicub_mask_db, interp_mkbicub_db,
};
pub use bipm::{interp_evbipm_db, interp_mkbipm_db};
pub use spline1d::{mkspline_db, splinck_db};

/// Boundary condition selector for 1D and 2D cubic splines (`mkspline_db` and
/// `interp_mkbicub_db`), matching upstream PSPLINE / MESA codes `-1..=7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BicubicBc {
    /// Periodic boundary condition (`-1`).
    Periodic = -1,
    /// Not-a-knot boundary condition (`0`, default in MESA).
    NotAKnot = 0,
    /// Match specified first derivative (`1`).
    FirstDeriv = 1,
    /// Match specified second derivative (`2`).
    SecondDeriv = 2,
    /// Zero first derivative (`3`).
    ZeroFirstDeriv = 3,
    /// Zero second derivative / natural spline (`4`).
    ZeroSecondDeriv = 4,
    /// Match first divided difference (`5`).
    DividedDiff1 = 5,
    /// Match second divided difference (`6`).
    DividedDiff2 = 6,
    /// Match third divided difference (`7`).
    DividedDiff3 = 7,
}

impl BicubicBc {
    /// Converts an integer code in `-1..=7` to [`BicubicBc`].
    ///
    /// # Errors
    ///
    /// Returns [`Interp2dError::InvalidBoundaryCondition`] if `code` is outside
    /// `-1..=7`.
    pub fn from_code(code: i32) -> Result<Self, Interp2dError> {
        match code {
            -1 => Ok(Self::Periodic),
            0 => Ok(Self::NotAKnot),
            1 => Ok(Self::FirstDeriv),
            2 => Ok(Self::SecondDeriv),
            3 => Ok(Self::ZeroFirstDeriv),
            4 => Ok(Self::ZeroSecondDeriv),
            5 => Ok(Self::DividedDiff1),
            6 => Ok(Self::DividedDiff2),
            7 => Ok(Self::DividedDiff3),
            _ => Err(Interp2dError::InvalidBoundaryCondition(code)),
        }
    }
}

/// Complete set of value and first/second partial derivatives evaluated from a
/// 2D bicubic spline at $(x, y)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BicubicEval {
    /// Interpolated value $f(x, y)$.
    pub f: f64,
    /// First partial derivative $\partial f / \partial x$.
    pub df_dx: f64,
    /// First partial derivative $\partial f / \partial y$.
    pub df_dy: f64,
    /// Second partial derivative $\partial^2 f / \partial x^2$.
    pub d2f_dx2: f64,
    /// Second partial derivative $\partial^2 f / \partial y^2$.
    pub d2f_dy2: f64,
    /// Mixed second partial derivative $\partial^2 f / \partial x\,\partial y$.
    pub d2f_dxdy: f64,
}

/// Error returned by 2D interpolation setup and evaluation routines.
#[derive(Debug, Clone, PartialEq)]
pub enum Interp2dError {
    /// Fewer grid points were provided than required by the interpolant.
    InsufficientPoints {
        /// Axis name (`"x"`, `"y"`, or `"query"`).
        axis: &'static str,
        /// Minimum required points.
        min_points: usize,
        /// Provided number of points.
        actual: usize,
    },
    /// Array dimensions do not match `(nx, ny)`.
    DimensionMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
    /// Input contains `NaN` or $\pm\infty$.
    NonFiniteInput,
    /// Grid axis is not strictly ascending.
    NonAscendingGrid {
        /// Axis name (`"x"` or `"y"`).
        axis: &'static str,
    },
    /// Invalid boundary condition flag.
    InvalidBoundaryCondition(i32),
    /// Target evaluation point $(x, y)$ lies outside the grid domain beyond
    /// the clamping tolerance.
    OutOfRange {
        /// Axis name (`"x"` or `"y"`).
        axis: &'static str,
        /// Requested coordinate value.
        value: f64,
    },
}

impl fmt::Display for Interp2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientPoints {
                axis,
                min_points,
                actual,
            } => write!(
                f,
                "{axis} axis requires at least {min_points} points, got {actual}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "grid array length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteInput => {
                write!(f, "2D interpolation input contains NaN or infinity")
            }
            Self::NonAscendingGrid { axis } => {
                write!(f, "{axis} grid is not strictly ascending")
            }
            Self::InvalidBoundaryCondition(code) => {
                write!(f, "invalid boundary condition code {code}")
            }
            Self::OutOfRange { axis, value } => {
                write!(f, "query coordinate {axis} = {value} is out of range")
            }
        }
    }
}

impl Error for Interp2dError {}
