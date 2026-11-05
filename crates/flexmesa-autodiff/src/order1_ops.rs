// SPDX-License-Identifier: LGPL-3.0-or-later
//! Arithmetic and comparison operator trait implementations for [`DualOrder1`].
//!
//! Relational comparisons (`PartialEq`, `PartialOrd`) compare the primal value
//! `val`, matching MESA's `auto_diff` operator overloads where control-flow
//! branches depend solely on the primal physical state. Arithmetic operators
//! (`Neg`, `Add`, `Sub`, `Mul`, `Div`) propagate exact first-order partial
//! derivatives for both dual-dual and dual-scalar operands.

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::order1::DualOrder1;

/// Default initialization produces the zero constant with zero partials.
impl<const N: usize> Default for DualOrder1<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

/// Convert a scalar `f64` constant into a [`DualOrder1`] with zero partials.
impl<const N: usize> From<f64> for DualOrder1<N> {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

/// Primal-value equality comparison between two [`DualOrder1`] instances.
impl<const N: usize> PartialEq for DualOrder1<N> {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

/// Primal-value equality comparison against an `f64` scalar.
impl<const N: usize> PartialEq<f64> for DualOrder1<N> {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &f64) -> bool {
        self.val == *other
    }
}

/// Primal-value ordering comparison between two [`DualOrder1`] instances.
impl<const N: usize> PartialOrd for DualOrder1<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

/// Primal-value ordering comparison against an `f64` scalar.
impl<const N: usize> PartialOrd<f64> for DualOrder1<N> {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

/// Unary negation $-x$ negating both `val` and all partial derivatives `d1`.
impl<const N: usize> Neg for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.make_unary_operator(-self.val, -1.0)
    }
}

/// Dual-dual addition $x + y$ summing primal values and gradients.
impl<const N: usize> Add for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(rhs, self.val + rhs.val, 1.0, 1.0)
    }
}

/// Dual-scalar addition $x + c$ shifting `val` while preserving `d1`.
impl<const N: usize> Add<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self { val: self.val + rhs, d1: self.d1 }
    }
}

/// Scalar-dual addition $c + x$ shifting `val` while preserving `d1`.
impl<const N: usize> Add<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    fn add(self, rhs: DualOrder1<N>) -> Self::Output {
        rhs + self
    }
}

/// Dual-dual subtraction $x - y$ subtracting primal values and gradients.
impl<const N: usize> Sub for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(rhs, self.val - rhs.val, 1.0, -1.0)
    }
}

/// Dual-scalar subtraction $x - c$ shifting `val` while preserving `d1`.
impl<const N: usize> Sub<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self { val: self.val - rhs, d1: self.d1 }
    }
}

/// Scalar-dual subtraction $c - x$ negating `d1`.
impl<const N: usize> Sub<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: DualOrder1<N>) -> Self::Output {
        (-rhs) + self
    }
}

/// Dual-dual multiplication $x y$ via the product rule
/// $y \nabla x + x \nabla y$.
impl<const N: usize> Mul for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(rhs, self.val * rhs.val, rhs.val, self.val)
    }
}

/// Dual-scalar multiplication $c x$ scaling both `val` and `d1` by `c`.
impl<const N: usize> Mul<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self.make_unary_operator(self.val * rhs, rhs)
    }
}

/// Scalar-dual multiplication $c x$ scaling both `val` and `d1` by `c`.
impl<const N: usize> Mul<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    fn mul(self, rhs: DualOrder1<N>) -> Self::Output {
        rhs * self
    }
}

/// Dual-dual division $x / y$ via the quotient rule.
impl<const N: usize> Div for DualOrder1<N> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        let inv_y = 1.0 / rhs.val;
        self.make_binary_operator(
            rhs,
            self.val * inv_y,
            inv_y,
            -self.val * inv_y * inv_y,
        )
    }
}

/// Dual-scalar division $x / c$ scaling `val` and `d1` by $1 / c$.
impl<const N: usize> Div<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

/// Scalar-dual division $c / y = c y^{-1}$.
impl<const N: usize> Div<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: DualOrder1<N>) -> Self::Output {
        rhs.powm1() * self
    }
}
