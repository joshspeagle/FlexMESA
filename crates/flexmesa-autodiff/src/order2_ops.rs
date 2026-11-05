// SPDX-License-Identifier: LGPL-3.0-or-later
//! Arithmetic and comparison operator trait implementations for
//! [`AutoDiffReal2VarOrder2`].

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::order2::AutoDiffReal2VarOrder2;

impl From<f64> for AutoDiffReal2VarOrder2 {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

impl PartialEq for AutoDiffReal2VarOrder2 {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialEq<f64> for AutoDiffReal2VarOrder2 {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &f64) -> bool {
        self.val == *other
    }
}

impl PartialOrd for AutoDiffReal2VarOrder2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl PartialOrd<f64> for AutoDiffReal2VarOrder2 {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

impl Neg for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.make_unary_operator(-self.val, -1.0, 0.0)
    }
}

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

impl Add<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self { val: self.val + rhs, ..self }
    }
}

impl Add<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn add(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        rhs + self
    }
}

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

impl Sub<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self { val: self.val - rhs, ..self }
    }
}

impl Sub<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        (-rhs) + self
    }
}

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

impl Mul<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self.make_unary_operator(self.val * rhs, rhs, 0.0)
    }
}

impl Mul<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    fn mul(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        rhs * self
    }
}

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

impl Div<f64> for AutoDiffReal2VarOrder2 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Div<AutoDiffReal2VarOrder2> for f64 {
    type Output = AutoDiffReal2VarOrder2;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: AutoDiffReal2VarOrder2) -> Self::Output {
        rhs.powm1() * self
    }
}
