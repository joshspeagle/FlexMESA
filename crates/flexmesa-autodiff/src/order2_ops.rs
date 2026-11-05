// SPDX-License-Identifier: LGPL-3.0-or-later
//! Arithmetic and comparison operator trait implementations for
//! [`AutoDiffReal2VarOrder2`].
//!
//! Relational operators (`PartialEq`, `PartialOrd`) compare the primal value
//! `val`. Linear scaling (`Neg`, `Mul<f64>`, `Div<f64>`) scales all six
//! components directly so large first derivatives never evaluate `0.0 * Inf`
//! via `pow2(d1val)`.

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::order2::AutoDiffReal2VarOrder2;

/// Convert a scalar `f64` constant into a hyperdual number with zero partials.
impl From<f64> for AutoDiffReal2VarOrder2 {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

/// Primal-value equality comparison between two hyperdual numbers.
impl PartialEq for AutoDiffReal2VarOrder2 {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

/// Primal-value equality comparison against an `f64` scalar.
impl PartialEq<f64> for AutoDiffReal2VarOrder2 {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &f64) -> bool {
        self.val == *other
    }
}

/// Primal-value ordering comparison between two hyperdual numbers.
impl PartialOrd for AutoDiffReal2VarOrder2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

/// Primal-value ordering comparison against an `f64` scalar.
impl PartialOrd<f64> for AutoDiffReal2VarOrder2 {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

/// Unary negation $-x$ negating all six components directly without squaring
/// first derivatives.
impl Neg for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(
            -self.val,
            -self.d1val1,
            -self.d1val2,
            -self.d2val1,
            -self.d1val1_d1val2,
            -self.d2val2,
        )
    }
}

/// Hyperdual addition $x + y$ summing all six components elementwise.
impl Add for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.val + rhs.val,
            self.d1val1 + rhs.d1val1,
            self.d1val2 + rhs.d1val2,
            self.d2val1 + rhs.d2val1,
            self.d1val1_d1val2 + rhs.d1val1_d1val2,
            self.d2val2 + rhs.d2val2,
        )
    }
}

/// Hyperdual-scalar addition $x + c$ shifting `val` while preserving partials.
impl Add<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self { val: self.val + rhs, ..self }
    }
}

/// Scalar-hyperdual addition $c + x$ shifting `val` while preserving partials.
impl Add<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn add(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        rhs + self
    }
}

/// Hyperdual subtraction $x - y$ subtracting all six components elementwise.
impl Sub for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.val - rhs.val,
            self.d1val1 - rhs.d1val1,
            self.d1val2 - rhs.d1val2,
            self.d2val1 - rhs.d2val1,
            self.d1val1_d1val2 - rhs.d1val1_d1val2,
            self.d2val2 - rhs.d2val2,
        )
    }
}

/// Hyperdual-scalar subtraction $x - c$ shifting `val` while preserving
/// partials.
impl Sub<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self { val: self.val - rhs, ..self }
    }
}

/// Scalar-hyperdual subtraction $c - x$ negating all partial derivatives.
impl Sub<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        (-rhs) + self
    }
}

/// Hyperdual multiplication $x y$ via the second-order product rule.
impl Mul for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(
            rhs,
            self.val * rhs.val,
            rhs.val,
            self.val,
            0.0,
            1.0,
            0.0,
        )
    }
}

/// Hyperdual-scalar multiplication $c x$ scaling all six components directly.
impl Mul<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            self.val * rhs,
            self.d1val1 * rhs,
            self.d1val2 * rhs,
            self.d2val1 * rhs,
            self.d1val1_d1val2 * rhs,
            self.d2val2 * rhs,
        )
    }
}

/// Scalar-hyperdual multiplication $c x$ scaling all six components directly.
impl Mul<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn mul(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        rhs * self
    }
}

/// Hyperdual division $x / y$ via the second-order quotient rule.
impl Div for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        let inv_y = 1.0 / rhs.val;
        let inv_y2 = inv_y * inv_y;
        self.make_binary_operator(
            rhs,
            self.val * inv_y,
            inv_y,
            -self.val * inv_y2,
            0.0,
            -inv_y2,
            2.0 * self.val * inv_y2 * inv_y,
        )
    }
}

/// Hyperdual-scalar division $x / c$ scaling all six components by $1 / c$.
impl Div<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

/// Scalar-hyperdual division $c / y = c y^{-1}$.
impl Div<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        rhs.powm1() * self
    }
}
