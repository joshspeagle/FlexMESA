// SPDX-License-Identifier: LGPL-3.0-or-later
//! Arithmetic and comparison operator trait implementations for
//! [`AutoDiffReal2VarOrder2`].
//!
//! Relational operators (`PartialEq`, `PartialOrd`) compare the primal value
//! `val`. Arithmetic evaluates the expression of the upstream routine of the same
//! name (`order2_gen`, `#349`).

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

/// `unary_minus_self`.
impl Neg for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        crate::order2_gen::unary_minus_self(self)
    }
}

/// `add_self`.
impl Add for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        crate::order2_gen::add_self(self, rhs)
    }
}

/// `add_self_real`.
impl Add<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        crate::order2_gen::add_self_real(self, rhs)
    }
}

/// `add_real_self`.
impl Add<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn add(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        crate::order2_gen::add_real_self(self, rhs)
    }
}

/// `sub_self`.
impl Sub for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        crate::order2_gen::sub_self(self, rhs)
    }
}

/// `sub_self_real`.
impl Sub<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        crate::order2_gen::sub_self_real(self, rhs)
    }
}

/// `sub_real_self`.
impl Sub<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn sub(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        crate::order2_gen::sub_real_self(self, rhs)
    }
}

/// `mul_self`.
impl Mul for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        crate::order2_gen::mul_self(self, rhs)
    }
}

/// `mul_self_real`.
impl Mul<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        crate::order2_gen::mul_self_real(self, rhs)
    }
}

/// `mul_real_self`.
impl Mul<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn mul(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        crate::order2_gen::mul_real_self(self, rhs)
    }
}

/// `div_self`.
impl Div for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        crate::order2_gen::div_self(self, rhs)
    }
}

/// `div_self_real`.
impl Div<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        crate::order2_gen::div_self_real(self, rhs)
    }
}

/// `div_real_self`.
impl Div<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn div(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        crate::order2_gen::div_real_self(self, rhs)
    }
}

impl AutoDiffReal2VarOrder2 {
    /// `pow_self`: $x^{y}$ for a dual exponent.
    #[must_use]
    #[inline]
    pub fn pow(self, other: Self) -> Self {
        crate::order2_gen::pow_self(self, other)
    }

    /// `pow_self_real`: $x^{p}$ for a constant exponent.
    #[must_use]
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        crate::order2_gen::pow_self_real(self, p)
    }

    /// `pow_self_int`: the exponent is converted to `real(dp)` and the
    /// expression is that of `pow_self_real`.
    #[must_use]
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        self.powf(f64::from(n))
    }

    /// `pow_real_self`: $b^{x}$ for a constant base.
    #[must_use]
    #[inline]
    pub fn rpow(self, base: f64) -> Self {
        crate::order2_gen::pow_real_self(base, self)
    }

    /// `pow2_self`.
    #[must_use]
    #[inline]
    pub fn pow2(self) -> Self {
        crate::order2_gen::pow2_self(self)
    }

    /// `pow3_self`.
    #[must_use]
    #[inline]
    pub fn pow3(self) -> Self {
        crate::order2_gen::pow3_self(self)
    }

    /// `pow4_self`.
    #[must_use]
    #[inline]
    pub fn pow4(self) -> Self {
        crate::order2_gen::pow4_self(self)
    }

    /// `pow5_self`.
    #[must_use]
    #[inline]
    pub fn pow5(self) -> Self {
        crate::order2_gen::pow5_self(self)
    }

    /// `pow6_self`.
    #[must_use]
    #[inline]
    pub fn pow6(self) -> Self {
        crate::order2_gen::pow6_self(self)
    }

    /// `pow7_self`.
    #[must_use]
    #[inline]
    pub fn pow7(self) -> Self {
        crate::order2_gen::pow7_self(self)
    }

    /// `pow8_self`.
    #[must_use]
    #[inline]
    pub fn pow8(self) -> Self {
        crate::order2_gen::pow8_self(self)
    }

    /// `max_self`.
    #[must_use]
    #[inline]
    pub fn max(self, other: Self) -> Self {
        crate::order2_gen::max_self(self, other)
    }

    /// `max_self_real`.
    #[must_use]
    #[inline]
    pub fn max_scalar(self, c: f64) -> Self {
        crate::order2_gen::max_self_real(self, c)
    }

    /// `min_self`.
    #[must_use]
    #[inline]
    pub fn min(self, other: Self) -> Self {
        crate::order2_gen::min_self(self, other)
    }

    /// `min_self_real`.
    #[must_use]
    #[inline]
    pub fn min_scalar(self, c: f64) -> Self {
        crate::order2_gen::min_self_real(self, c)
    }

    /// `dim_self`.
    #[must_use]
    #[inline]
    pub fn dim(self, other: Self) -> Self {
        crate::order2_gen::dim_self(self, other)
    }

    /// `dim_self_real`.
    #[must_use]
    #[inline]
    pub fn dim_scalar(self, c: f64) -> Self {
        crate::order2_gen::dim_self_real(self, c)
    }
}
