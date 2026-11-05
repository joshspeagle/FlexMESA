// SPDX-License-Identifier: LGPL-3.0-or-later
//! Arithmetic and comparison operator trait implementations for [`DualOrder1`].

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::order1::DualOrder1;

impl<const N: usize> Default for DualOrder1<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> From<f64> for DualOrder1<N> {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

impl<const N: usize> PartialEq for DualOrder1<N> {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<const N: usize> PartialEq<f64> for DualOrder1<N> {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &f64) -> bool {
        self.val == *other
    }
}

impl<const N: usize> PartialOrd for DualOrder1<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<const N: usize> PartialOrd<f64> for DualOrder1<N> {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

impl<const N: usize> Neg for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.make_unary_operator(-self.val, -1.0)
    }
}

impl<const N: usize> Add for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(rhs, self.val + rhs.val, 1.0, 1.0)
    }
}

impl<const N: usize> Add<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self { val: self.val + rhs, d1: self.d1 }
    }
}

impl<const N: usize> Add<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    fn add(self, rhs: DualOrder1<N>) -> Self::Output {
        rhs + self
    }
}

impl<const N: usize> Sub for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(rhs, self.val - rhs.val, 1.0, -1.0)
    }
}

impl<const N: usize> Sub<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self { val: self.val - rhs, d1: self.d1 }
    }
}

impl<const N: usize> Sub<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: DualOrder1<N>) -> Self::Output {
        (-rhs) + self
    }
}

impl<const N: usize> Mul for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.make_binary_operator(rhs, self.val * rhs.val, rhs.val, self.val)
    }
}

impl<const N: usize> Mul<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self.make_unary_operator(self.val * rhs, rhs)
    }
}

impl<const N: usize> Mul<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    fn mul(self, rhs: DualOrder1<N>) -> Self::Output {
        rhs * self
    }
}

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

impl<const N: usize> Div<f64> for DualOrder1<N> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<const N: usize> Div<DualOrder1<N>> for f64 {
    type Output = DualOrder1<N>;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: DualOrder1<N>) -> Self::Output {
        rhs.powm1() * self
    }
}
