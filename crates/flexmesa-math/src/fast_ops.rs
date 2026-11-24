// SPDX-License-Identifier: GPL-3.0-or-later
//! Isolated `std` `f64` platform math operations used exclusively by
//! [`crate::NumericsMode::Fast`].
//!
//! This private module is the **only** source file in `flexmesa-math` permitted
//! to suppress `clippy::disallowed_methods`.

#![allow(clippy::disallowed_methods)]

#[inline]
pub(crate) fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[inline]
pub(crate) fn ln(x: f64) -> f64 {
    x.ln()
}

#[inline]
pub(crate) fn log10(x: f64) -> f64 {
    x.log10()
}

#[inline]
pub(crate) fn ln_1p(x: f64) -> f64 {
    x.ln_1p()
}

#[inline]
pub(crate) fn log2(x: f64) -> f64 {
    x.log2()
}

#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    x.exp()
}

#[inline]
pub(crate) fn exp_m1(x: f64) -> f64 {
    x.exp_m1()
}

#[inline]
pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

#[inline]
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    x.powf(y)
}

#[inline]
pub(crate) fn powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    x.cos()
}

#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    x.sin()
}

#[inline]
pub(crate) fn sin_cos(x: f64) -> (f64, f64) {
    x.sin_cos()
}

#[inline]
pub(crate) fn tan(x: f64) -> f64 {
    x.tan()
}

#[inline]
pub(crate) fn acos(x: f64) -> f64 {
    x.acos()
}

#[inline]
pub(crate) fn asin(x: f64) -> f64 {
    x.asin()
}

#[inline]
pub(crate) fn atan(x: f64) -> f64 {
    x.atan()
}

#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

#[inline]
pub(crate) fn cosh(x: f64) -> f64 {
    x.cosh()
}

#[inline]
pub(crate) fn sinh(x: f64) -> f64 {
    x.sinh()
}

#[inline]
pub(crate) fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline]
pub(crate) fn acosh(x: f64) -> f64 {
    x.acosh()
}

#[inline]
pub(crate) fn asinh(x: f64) -> f64 {
    x.asinh()
}

#[inline]
pub(crate) fn atanh(x: f64) -> f64 {
    x.atanh()
}
