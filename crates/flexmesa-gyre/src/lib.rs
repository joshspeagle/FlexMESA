// SPDX-License-Identifier: GPL-3.0-only
//! Safe Rust port of the GYRE 8.1 stellar oscillation code as bundled with
//! MESA release `26.4.1` (`#170`).
//!
//! **Numerics substrate (G1)**, each piece checked bit for bit against
//! vectors produced by MESA's own GYRE build and by code compiled with
//! GYRE's flags (`tests/g1_vectors.rs`, `benchmarks/fortran_mesa/gyre_g1_vectors/`):
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
//! **Real LAPACK/BLAS subset (G2)**: [`lapack`], checked against the MESA
//! SDK libraries (`tests/g2_vectors.rs`).
//!
//! **Radial adiabatic path (G3)**, the GYRE code the `astero_gyre` case
//! reaches, checked against MESA's GYRE on that case's calls
//! (`tests/g3_astero_gyre.rs`):
//!
//! - [`mesa`]: the `gyre_mesa_m` interface (`init`, `set_constant`,
//!   `set_model`, `get_modes`);
//! - [`namelist`], [`par`]: input parameters; [`constants`];
//! - [`model`]: evolutionary models from MESA pulsation data;
//! - [`grid`], [`grid_refine`]: grids and their refinement;
//! - [`context`]: context, null rotation and frequency scales;
//! - [`rad`]: radial equations, boundaries, collocation and the BVP;
//! - [`sysmtx`]: the block system matrix;
//! - [`search`]: scans, discriminants, Brent and the bracket search;
//! - [`mode`]: waves and modes.
//!
//! Branches the reached cases do not take return [`GyreError::Unported`]
//! with the upstream location instead of computing something different.
//!
//! Elementary functions go only through `flexmesa-math`. The crate is
//! licensed GPL-3.0-only because GYRE, ForUM and CRMATH are.

#![forbid(unsafe_code)]
// `manual_midpoint`: GYRE writes `0.5_RD*(a + b)`; `f64::midpoint` is a
// different operation sequence, so the upstream form is kept verbatim.
// The transcription keeps upstream's local names (`pt_a`/`pt_b`, `c_i`/
// `c_1`), its `if (a /= b) then ... else` order, index loops and
// routine lengths, and its parameter types' many logical flags.
#![allow(
    clippy::manual_midpoint,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::if_not_else,
    clippy::nonminimal_bool,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::struct_excessive_bools,
    clippy::semicolon_if_nothing_returned
)]

pub mod ad;
pub mod constants;
pub mod context;
pub mod crmath;
pub mod ext_complex;
pub mod ext_real;
pub mod fcomplex;
pub mod forum;
pub mod grid;
pub mod grid_refine;
pub mod integ;
pub mod lapack;
pub mod magnus;
pub mod math_m;
pub mod matmul;
pub mod mesa;
pub mod mode;
pub mod model;
pub mod nad;
pub mod namelist;
pub mod par;
pub mod powi;
pub mod prox;
pub mod rad;
pub mod scalar;
pub mod search;
pub mod sysmtx;

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

/// Errors of the GYRE port: upstream `ASSERT`/`ABORT` stops, unported
/// paths, and errors of the numerics substrate.
#[derive(Debug, Clone, PartialEq)]
pub enum GyreError {
    /// An upstream `@:ASSERT` that would stop the program.
    Assert(&'static str),
    /// An upstream `@:ABORT` (invalid option and similar).
    Abort(&'static str),
    /// A branch upstream takes that is not ported yet.
    Unported {
        /// What the branch does.
        what: &'static str,
        /// Upstream location (`<dir>/<file>.fypp:<lines>`, GYRE 8.1).
        at: &'static str,
    },
    /// A namelist input error.
    Namelist(namelist::NmlError),
    /// An error of the numerics substrate.
    Num(GyreNumError),
    /// A LAPACK error.
    Lapack(lapack::LapackError),
    /// A non-zero `INFO` that upstream asserts against.
    LapackInfo {
        /// The GYRE wrapper.
        routine: &'static str,
        /// The returned `INFO`.
        info: i64,
    },
}

impl std::fmt::Display for GyreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Assert(m) => write!(f, "GYRE assertion failed: {m}"),
            Self::Abort(m) => write!(f, "GYRE abort: {m}"),
            Self::Unported { what, at } => {
                write!(f, "GYRE path not ported: {what} ({at})")
            }
            Self::Namelist(e) => write!(f, "{e}"),
            Self::Num(e) => write!(f, "{e}"),
            Self::Lapack(e) => write!(f, "{e}"),
            Self::LapackInfo { routine, info } => {
                write!(f, "non-zero return {info} from {routine}")
            }
        }
    }
}

impl std::error::Error for GyreError {}

impl From<namelist::NmlError> for GyreError {
    fn from(e: namelist::NmlError) -> Self {
        Self::Namelist(e)
    }
}

impl From<GyreNumError> for GyreError {
    fn from(e: GyreNumError) -> Self {
        Self::Num(e)
    }
}

impl From<lapack::LapackError> for GyreError {
    fn from(e: lapack::LapackError) -> Self {
        Self::Lapack(e)
    }
}
