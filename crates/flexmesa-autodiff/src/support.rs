// SPDX-License-Identifier: GPL-3.0-or-later
//! Scalar helpers used by the translated `auto_diff` routines.
//!
//! `sgn` and `heaviside` follow `auto_diff/private/support_functions.f90:48-68`
//! at MESA `26.4.1`. `f_abs`, `f_sqrt`, `f_max` and `f_min` stand for the
//! Fortran intrinsics `Abs`, `sqrt`, `Max` and `Min`, which are exact or
//! correctly rounded and need no `CRMATH` routing.

/// `sgn` (`support_functions.f90:48-57`): `-1`, `0` or `1`; a `NaN` argument
/// takes the last branch.
#[inline]
#[must_use]
pub(crate) fn sgn(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        1.0
    }
}

/// `Heaviside` (`support_functions.f90:59-68`): `0`, `0.5` at zero, `1`; a
/// `NaN` argument takes the last branch.
#[inline]
#[must_use]
pub(crate) fn heaviside(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x == 0.0 {
        0.5
    } else {
        1.0
    }
}

/// Fortran `Abs` of a real.
#[inline]
#[must_use]
pub(crate) fn f_abs(x: f64) -> f64 {
    x.abs()
}

/// Fortran `sqrt` of a real (IEEE correctly rounded).
#[inline]
#[must_use]
pub(crate) fn f_sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Fortran `Max` of two reals.
#[inline]
#[must_use]
pub(crate) fn f_max(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Fortran `Min` of two reals.
#[inline]
#[must_use]
pub(crate) fn f_min(a: f64, b: f64) -> f64 {
    a.min(b)
}
