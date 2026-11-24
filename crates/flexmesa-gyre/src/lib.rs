// SPDX-License-Identifier: GPL-3.0-only
//! Safe Rust port of the GYRE 8.1 stellar oscillation code as bundled with
//! MESA release `26.4.1` (`#170`).
//!
//! This first package (G1) holds the numerics substrate GYRE's reached code
//! is built on, each piece checked bit for bit against vectors produced by
//! MESA's own GYRE build and by code compiled with GYRE's flags
//! (`tests/g1_vectors.rs`, `benchmarks/fortran_mesa/gyre_g1_vectors/`):
//!
//! - [`fcomplex`]: complex arithmetic as gfortran 15.2.0 expands it;
//! - [`powi`]: constant-exponent power chains and `__powidf2`;
//! - [`crmath`]: the CRMATH 1.2 composites (`atan2`, `abs`, `sqrt`, `exp`
//!   and `log` of complex numbers) over `flexmesa-math`;
//! - [`math_m`]: GYRE `math_m` (constants and `pow`);
//! - [`matmul`]: gfortran's inlined `MATMUL` loop order;
//! - [`ext_real`], [`ext_complex`]: GYRE's extended-exponent arithmetic;
//! - [`integ`]: GYRE `integ_m` trapezoidal quadrature;
//! - [`forum`]: the ForUM sort and search helpers GYRE calls.
//!
//! Elementary functions go only through `flexmesa-math`. The crate is
//! licensed GPL-3.0-only because GYRE, ForUM and CRMATH are.

#![forbid(unsafe_code)]
// `manual_midpoint`: GYRE writes `0.5_RD*(a + b)`; `f64::midpoint` is a
// different operation sequence, so the upstream form is kept verbatim.
#![allow(
    clippy::manual_midpoint,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

pub mod crmath;
pub mod ext_complex;
pub mod ext_real;
pub mod fcomplex;
pub mod forum;
pub mod integ;
pub mod lapack;
pub mod math_m;
pub mod matmul;
pub mod powi;

/// Errors raised where upstream has no defined result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GyreNumError {
    /// CRMATH `atan2(y, x)` with `x = y = 0` (or NaN) falls through every
    /// branch of `crmath_r_dp.f90:301-309` and leaves the result unset.
    Atan2Undefined {
        /// `y` argument.
        y: f64,
        /// `x` argument.
        x: f64,
    },
    /// `MATMUL` sizes beyond gfortran's inline limit (`m*k*n > 27000`):
    /// upstream would call `_gfortran_matmul_*`, whose blocked summation
    /// order is not ported.
    MatmulLibraryPath {
        /// Rows of `A`.
        m: usize,
        /// Columns of `A` / rows of `B`.
        k: usize,
        /// Columns of `B`.
        n: usize,
    },
    /// `x**n` with a constant exponent for which GCC's multiplication
    /// chain has not been verified (`powi::powi_const`).
    UnverifiedPowiChain {
        /// The exponent.
        n: i32,
    },
}

impl std::fmt::Display for GyreNumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Atan2Undefined { y, x } => {
                write!(
                    f,
                    "CRMATH atan2 is undefined for (y, x) = ({y:e}, {x:e})"
                )
            }
            Self::MatmulLibraryPath { m, k, n } => write!(
                f,
                "MATMUL of {m}x{k} by {k}x{n} exceeds gfortran's inline limit; \
                 the library summation order is not ported"
            ),
            Self::UnverifiedPowiChain { n } => {
                write!(
                    f,
                    "no verified GCC powi chain for constant exponent {n}"
                )
            }
        }
    }
}

impl std::error::Error for GyreNumError {}
