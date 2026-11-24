// SPDX-License-Identifier: GPL-3.0-only
//! Fortran integer powers `x**n` as gfortran 15.2.0 compiles them at `-O2`.
//!
//! - **Constant exponent** (`x**5` written in the source): GCC expands
//!   `__builtin_powi(x, n)` into the multiplication chain of
//!   `powi_as_mults` (`tree-ssa-math-opts.cc`, `powi_table`), and negative
//!   exponents into `1/x**|n|`. These chains differ from binary squaring for
//!   some `n` (for `n = 5`: `(x*x) * (x*(x*x))`; squaring gives
//!   `x * ((x*x)*(x*x))`), so they are spelled out per exponent below,
//!   transcribed from the `-fdump-tree-optimized` output recorded in
//!   `benchmarks/fortran_mesa/gyre_g1_vectors/README.md`.
//! - **Run-time exponent** (`x**n` with `n` a variable): GCC calls libgcc's
//!   `__powidf2` (binary squaring, `1/y` for negative `n`), which is
//!   [`flexmesa_math::powi_f90`].

use crate::GyreNumError;

/// `x**n` with a run-time integer exponent (`__powidf2`).
#[must_use]
#[inline]
pub fn powi_var(x: f64, n: i32) -> f64 {
    flexmesa_math::powi_f90(x, n)
}

/// `x**n` for a compile-time constant exponent `n`, following GCC's
/// `powi_as_mults` chain. Supported: `-3 ..= 8` (the exponents written in
/// GYRE 8.1's reached code are 2 to 6).
///
/// # Errors
///
/// [`GyreNumError::UnverifiedPowiChain`] for an exponent outside `-3 ..= 8`:
/// no chain has been verified for it.
pub fn powi_const(x: f64, n: i32) -> Result<f64, GyreNumError> {
    Ok(match n {
        0 => 1.0,
        1 => x,
        2 => x * x,
        3 => {
            let x2 = x * x;
            x * x2
        }
        4 => {
            let x2 = x * x;
            x2 * x2
        }
        5 => {
            let x2 = x * x;
            let x3 = x * x2;
            x2 * x3
        }
        6 => {
            let x2 = x * x;
            let x3 = x * x2;
            x3 * x3
        }
        7 => {
            let x2 = x * x;
            let x3 = x * x2;
            let x4 = x2 * x2;
            x3 * x4
        }
        8 => {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x4
        }
        -1 => 1.0 / x,
        -2 => 1.0 / (x * x),
        -3 => {
            let x2 = x * x;
            1.0 / (x * x2)
        }
        _ => return Err(GyreNumError::UnverifiedPowiChain { n }),
    })
}
