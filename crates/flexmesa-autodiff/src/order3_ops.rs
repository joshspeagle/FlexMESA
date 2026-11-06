// SPDX-License-Identifier: LGPL-3.0-or-later
//! Arithmetic and comparison operator trait implementations for
//! [`AutoDiffReal2VarOrder3`].
//!
//! Relational operators (`PartialEq`, `PartialOrd`) compare the primal value
//! `val`. Linear scaling (`Neg`, `Mul<f64>`, `Div<f64>`) scales all ten
//! components directly so large first derivatives never evaluate `0.0 * Inf`.

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use flexmesa_math::heaviside;

use crate::kernels3;
use crate::order3::AutoDiffReal2VarOrder3;

/// Convert a scalar `f64` constant into a third-order hyperdual number.
impl From<f64> for AutoDiffReal2VarOrder3 {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

/// Primal-value equality comparison between two third-order hyperdual numbers.
impl PartialEq for AutoDiffReal2VarOrder3 {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

/// Primal-value equality comparison against an `f64` scalar.
impl PartialEq<f64> for AutoDiffReal2VarOrder3 {
    #[allow(clippy::float_cmp)]
    fn eq(&self, other: &f64) -> bool {
        self.val == *other
    }
}

/// Primal-value ordering comparison between two third-order hyperdual numbers.
impl PartialOrd for AutoDiffReal2VarOrder3 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

/// Primal-value ordering comparison against an `f64` scalar.
impl PartialOrd<f64> for AutoDiffReal2VarOrder3 {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

/// Unary negation $-x$ negating all ten components directly.
impl Neg for AutoDiffReal2VarOrder3 {
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
            -self.d3val1,
            -self.d2val1_d1val2,
            -self.d1val1_d2val2,
            -self.d3val2,
        )
    }
}

/// Hyperdual addition $x + y$ summing all ten components elementwise.
impl Add for AutoDiffReal2VarOrder3 {
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
            self.d3val1 + rhs.d3val1,
            self.d2val1_d1val2 + rhs.d2val1_d1val2,
            self.d1val1_d2val2 + rhs.d1val1_d2val2,
            self.d3val2 + rhs.d3val2,
        )
    }
}

/// Hyperdual-scalar addition $x + c$ shifting `val` while preserving partials.
impl Add<f64> for AutoDiffReal2VarOrder3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        Self {
            val: self.val + rhs,
            ..self
        }
    }
}

/// Scalar-hyperdual addition $c + x$ shifting `val` while preserving partials.
impl Add<AutoDiffReal2VarOrder3> for f64 {
    type Output = AutoDiffReal2VarOrder3;
    #[inline]
    fn add(self, rhs: AutoDiffReal2VarOrder3) -> Self::Output {
        rhs + self
    }
}

/// Hyperdual subtraction $x - y$ subtracting all ten components elementwise.
impl Sub for AutoDiffReal2VarOrder3 {
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
            self.d3val1 - rhs.d3val1,
            self.d2val1_d1val2 - rhs.d2val1_d1val2,
            self.d1val1_d2val2 - rhs.d1val1_d2val2,
            self.d3val2 - rhs.d3val2,
        )
    }
}

/// Hyperdual-scalar subtraction $x - c$ shifting `val` while preserving
/// partials.
impl Sub<f64> for AutoDiffReal2VarOrder3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        Self {
            val: self.val - rhs,
            ..self
        }
    }
}

/// Scalar-hyperdual subtraction $c - x$ negating all partial derivatives.
impl Sub<AutoDiffReal2VarOrder3> for f64 {
    type Output = AutoDiffReal2VarOrder3;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: AutoDiffReal2VarOrder3) -> Self::Output {
        (-rhs) + self
    }
}

/// Hyperdual multiplication $x y$ via the third-order product rule.
impl Mul for AutoDiffReal2VarOrder3 {
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
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }
}

/// Hyperdual-scalar multiplication $c x$ scaling all ten components directly.
impl Mul<f64> for AutoDiffReal2VarOrder3 {
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
            self.d3val1 * rhs,
            self.d2val1_d1val2 * rhs,
            self.d1val1_d2val2 * rhs,
            self.d3val2 * rhs,
        )
    }
}

/// Scalar-hyperdual multiplication $c x$ scaling all ten components directly.
impl Mul<AutoDiffReal2VarOrder3> for f64 {
    type Output = AutoDiffReal2VarOrder3;
    #[inline]
    fn mul(self, rhs: AutoDiffReal2VarOrder3) -> Self::Output {
        rhs * self
    }
}

/// Hyperdual division $x / y$ via the third-order quotient rule.
impl Div for AutoDiffReal2VarOrder3 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        let inv_y = 1.0 / rhs.val;
        let inv_y2 = inv_y * inv_y;
        let inv_y3 = inv_y2 * inv_y;
        self.make_binary_operator(
            rhs,
            self.val * inv_y,
            inv_y,
            -self.val * inv_y2,
            0.0,
            -inv_y2,
            2.0 * self.val * inv_y3,
            0.0,
            0.0,
            2.0 * inv_y3,
            -6.0 * self.val * inv_y2 * inv_y2,
        )
    }
}

/// Hyperdual-scalar division $x / c$ scaling all ten components by $1 / c$.
impl Div<f64> for AutoDiffReal2VarOrder3 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

/// Scalar-hyperdual division $c / y = c y^{-1}$.
impl Div<AutoDiffReal2VarOrder3> for f64 {
    type Output = AutoDiffReal2VarOrder3;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: AutoDiffReal2VarOrder3) -> Self::Output {
        rhs.powm1() * self
    }
}

macro_rules! pown3_method {
    ($name:ident, $n:expr, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        #[inline]
        pub fn $name(self) -> Self {
            let (v, d1, d2, d3) = kernels3::unary3_pown(self.val, $n);
            self.make_unary_operator(v, d1, d2, d3)
        }
    };
}

impl AutoDiffReal2VarOrder3 {
    /// Raise `self` to a third-order hyperdual exponent `other` ($x^{y}$).
    #[must_use]
    #[inline]
    pub fn pow(self, other: Self) -> Self {
        (other * self.log()).exp()
    }

    /// Raise `self` to a constant `f64` exponent $p$.
    #[must_use]
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        let (v, d1, d2, d3) = kernels3::unary3_pow_scalar(self.val, p);
        self.make_unary_operator(v, d1, d2, d3)
    }

    /// Raise `self` to a constant integer exponent $n$.
    #[must_use]
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        self.powf(f64::from(n))
    }

    /// Raise a constant positive scalar `base` to the hyperdual exponent `self`
    /// ($b^{x}$).
    #[must_use]
    #[inline]
    pub fn rpow(self, base: f64) -> Self {
        let (v, d1, d2, d3) = kernels3::unary3_rpow_scalar(base, self.val);
        self.make_unary_operator(v, d1, d2, d3)
    }

    pown3_method!(pow2, 2, "Compute $x^{2}$.");
    pown3_method!(pow3, 3, "Compute $x^{3}$.");
    pown3_method!(pow4, 4, "Compute $x^{4}$.");
    pown3_method!(pow5, 5, "Compute $x^{5}$.");
    pown3_method!(pow6, 6, "Compute $x^{6}$.");
    pown3_method!(pow7, 7, "Compute $x^{7}$.");
    pown3_method!(pow8, 8, "Compute $x^{8}$.");

    /// Compute $\max(x, y)$ using MESA's symmetric Heaviside subgradient
    /// (`heaviside(0.0) == 0.0`, so ties have zero derivative).
    #[must_use]
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let q0 = heaviside(self.val - other.val);
        let q1 = heaviside(other.val - self.val);
        self.make_binary_operator(
            other,
            self.val.max(other.val),
            q0,
            q1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Compute $\max(x, c)$ with a scalar constant $c$.
    #[must_use]
    #[inline]
    pub fn max_scalar(self, c: f64) -> Self {
        self.make_unary_operator(
            self.val.max(c),
            heaviside(self.val - c),
            0.0,
            0.0,
        )
    }

    /// Compute $\min(x, y)$ using MESA's symmetric Heaviside subgradient
    /// (`heaviside(0.0) == 0.0`, so ties have zero derivative).
    #[must_use]
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let q0 = heaviside(other.val - self.val);
        let q1 = heaviside(self.val - other.val);
        self.make_binary_operator(
            other,
            self.val.min(other.val),
            q0,
            q1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Compute $\min(x, c)$ with a scalar constant $c$.
    #[must_use]
    #[inline]
    pub fn min_scalar(self, c: f64) -> Self {
        self.make_unary_operator(
            self.val.min(c),
            heaviside(c - self.val),
            0.0,
            0.0,
        )
    }

    /// Compute positive difference $\operatorname{dim}(x, y) = \max(x - y, 0)$.
    #[must_use]
    #[inline]
    pub fn dim(self, other: Self) -> Self {
        (self - other).max_scalar(0.0)
    }

    /// Compute positive difference $\operatorname{dim}(x, c) = \max(x - c, 0)$.
    #[must_use]
    #[inline]
    pub fn dim_scalar(self, c: f64) -> Self {
        (self - c).max_scalar(0.0)
    }
}
