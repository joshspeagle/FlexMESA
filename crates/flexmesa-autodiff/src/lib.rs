// SPDX-License-Identifier: LGPL-3.0-or-later
//! Forward-mode automatic differentiation (`auto_diff`) for `FlexMESA`.
//!
//! This crate ports MESA's `auto_diff` module (`auto_diff/public/auto_diff.f90`
//! and `auto_diff/private/auto_diff_real_*_module.f90`) to Safe Rust using
//! const-generic first-order dual numbers ([`DualOrder1`]) and two-variable
//! second-order hyperdual numbers ([`AutoDiffReal2VarOrder2`]).
//!
//! # Provided Types
//! - [`DualOrder1<N>`]: First-order forward-mode dual number storing a scalar
//!   value `val` and `N` partial derivatives `d1: [f64; N]`.
//! - [`AutoDiffReal1VarOrder1`]: 1-variable first-order dual number (`N = 1`).
//! - [`AutoDiffReal2VarOrder1`]: 2-variable first-order dual number (`N = 2`).
//! - [`AutoDiffReal4VarOrder1`] / [`AutoDiffReal4Coeff`]: 4-variable
//!   first-order dual number (`N = 4`).
//! - [`AutoDiffRealStarOrder1`]: 33-variable stellar structure first-order dual
//!   number (`N = 33`).
//! - [`AutoDiffReal2VarOrder2`] / [`AutoDiffReal2ndOrder`]: 2-variable
//!   second-order hyperdual number storing `(val, d1val1, d1val2, d2val1,
//!   d1val1_d1val2, d2val2)`.

#![forbid(unsafe_code)]

pub mod kernels;
pub mod order1;
mod order1_ops;
pub mod order2;
mod order2_ops;

pub use order1::DualOrder1;
pub use order2::AutoDiffReal2VarOrder2;

/// Number of independent stellar structure variables in
/// [`AutoDiffRealStarOrder1`].
pub const STAR_ORDER1_VARS: usize = 33;

/// One-variable first-order dual number (`auto_diff_real_1var_order1`).
pub type AutoDiffReal1VarOrder1 = DualOrder1<1>;

/// Two-variable first-order dual number (`auto_diff_real_2var_order1`).
pub type AutoDiffReal2VarOrder1 = DualOrder1<2>;

/// Four-variable first-order dual number (`auto_diff_real_4var_order1`).
pub type AutoDiffReal4VarOrder1 = DualOrder1<4>;

/// Alias for [`AutoDiffReal4VarOrder1`] matching `auto_diff_real_4coeff`.
pub type AutoDiffReal4Coeff = DualOrder1<4>;

/// Stellar 33-variable first-order dual number (`auto_diff_real_star_order1`).
pub type AutoDiffRealStarOrder1 = DualOrder1<STAR_ORDER1_VARS>;

/// Alias for [`AutoDiffReal2VarOrder2`] matching `auto_diff_real_2nd_order`.
pub type AutoDiffReal2ndOrder = AutoDiffReal2VarOrder2;
