//! Root-finding, optimization, quadrature, ODE/DAE, and Newton solvers (`num`).
//!
//! Ports upstream MESA `num` (`num_def.f90`, `num_lib.f90`, `mod_brent.f90`,
//! `mod_root.f90`, `mod_simplex.f90`, `mod_random.f90`, `mod_integrate.f90`,
//! `mod_dopri5.f`, `mod_dop853.f`, `mod_rosenbrock.f`, and `mod_newton.f90`)
//! into Safe Rust in `f64` precision. Includes implicit forward-mode
//! differentiation (`safe_root_dual`) and fixes the upstream
//! `find_max_quadratic` concavity check bug (`y2 < max(y1, y3)`).
//!
//! # Module Organization
//!
//! - [`search`]: Knuth `two_sum`, Neumaier compensated summation, monotonic
//!   `binary_search`, index permutation `qsort_index`, and quadratic
//!   interpolation/extrema solvers (`find0`, `find0_quadratic`,
//!   `find_max_quadratic`).
//! - [`quadrature`]: Ridders Neville-tableau numerical differentiation
//!   (`dfridr`) and adaptive Simpson 3/8 quadrature (`integrate`).
//! - [`brent`]: Brent (1973) derivative-free root-finding (`brent_safe_zero`)
//!   and univariate local/global minimization (`brent_local_min`,
//!   `brent_global_min`).
//! - [`root`]: Safeguarded Newton-Raphson + secant/bisection root-finding
//!   (`safe_root_with_brackets`, `safe_root_with_guess`) and implicit dual
//!   differentiation (`safe_root_dual`).
//! - [`simplex`]: Nelder-Mead (`1965`) downhill simplex optimization
//!   (`nm_simplex`) and Park-Miller `ran1_zah` uniform PRNG.
//! - [`rk`] and [`rk_tableau`]: Explicit Runge-Kutta integrators (`rk2`, `rk4`,
//!   `dopri5`, `dop853`) with Shampine 4th-order dense output (`dense_eval`).
//! - [`rosenbrock`] and [`rosenbrock_coeffs`]: L-stable Rosenbrock-Wanner stiff
//!   ODE integrators across all 7 MESA stage families (`Ros2`, `Rose2`,
//!   `Ros3p`, `Ros3pl`, `Rodas3`, `Rodas4`, `Rodasp`).
//! - [`newton`]: Damped Newton-Raphson solver for dense, banded, and
//!   block-tridiagonal Jacobians (`newton_solve_dense`, `newton_solve_banded`,
//!   `newton_solve_block_tridiagonal`).

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
    clippy::too_many_lines
)]

pub mod brent;
pub mod newton;
pub mod quadrature;
pub mod rk;
pub mod rk_tableau;
pub mod root;
pub mod rosenbrock;
pub mod rosenbrock_coeffs;
pub mod search;
pub mod simplex;

pub use brent::{brent_global_min, brent_local_min, brent_safe_zero};
pub use newton::{
    newton_solve_banded, newton_solve_block_tridiagonal, newton_solve_dense,
    NewtonConfig, NewtonResult,
};
pub use quadrature::{dfridr, integrate};
pub use rk::{dop853, dopri5, rk2, rk4, DopriResult};
pub use root::{
    look_for_brackets, safe_root_dual, safe_root_with_brackets,
    safe_root_with_guess, safe_root_without_brackets,
};
pub use rosenbrock::{rosenbrock_banded, rosenbrock_dense, RosenbrockResult};
pub use rosenbrock_coeffs::RosenbrockVariant;
pub use search::{
    binary_search, binary_search_1based, find0, find0_quadratic,
    find_max_quadratic, linear_interp, neumaier_sum, qsort_index,
    two_piece_linear_coeffs, two_sum,
};
pub use simplex::{
    i4_uniform, nm_simplex, perm_uniform, r8_uniform_01, SimplexConfig,
    SimplexResult,
};

use std::fmt;

/// Error conditions returned by `flexmesa-num` numerical solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumError {
    /// Input slices or bounds do not satisfy required preconditions.
    InvalidParameter(&'static str),
    /// An input argument or function evaluation produced `NaN` or $\pm\infty$.
    NonFiniteValue(&'static str),
    /// Root bracketing failed on the given interval or search range.
    BracketingFailed(&'static str),
    /// Iterative solver exceeded its maximum iteration or step budget.
    ConvergenceFailed(&'static str),
    /// Underlying linear algebra factorization or solve failed.
    LinearSolveFailed(&'static str),
}

impl fmt::Display for NumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => {
                write!(f, "invalid parameter: {msg}")
            }
            Self::NonFiniteValue(msg) => write!(f, "non-finite value: {msg}"),
            Self::BracketingFailed(msg) => {
                write!(f, "bracketing failed: {msg}")
            }
            Self::ConvergenceFailed(msg) => {
                write!(f, "convergence failed: {msg}")
            }
            Self::LinearSolveFailed(msg) => {
                write!(f, "linear solve failed: {msg}")
            }
        }
    }
}

impl std::error::Error for NumError {}
