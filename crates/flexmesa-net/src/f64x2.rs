// SPDX-License-Identifier: LGPL-3.0-or-later
//! Error-free double-double (`F64x2`) extended-precision floating-point
//! arithmetic (~106-bit significand) matching upstream Fortran `real(qp)`
//! accumulations in `net/private/net_derivs.f90` and `net_approx21.f90`.
//!
//! In stiff nuclear reaction networks near Nuclear Statistical Equilibrium
//! (NSE), forward and photodisintegration rates cancel to 10-14 digits.
//! Accumulating $\dot{Y}_i$ and $\epsilon_{\mathrm{nuc}}$ in [`F64x2`] prevents
//! catastrophic cancellation while remaining 100% pure Safe Rust.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Error-free Knuth (1969) `two_sum` returning `(s, e)` with $s + e = a + b$
/// exactly in IEEE-754 arithmetic.
#[inline]
#[must_use]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Fast `quick_two_sum` assuming $|a| \ge |b|$.
#[inline]
#[must_use]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Error-free Dekker (1971) `two_prod` using hardware fused multiply-add
/// (`f64::mul_add`), returning `(p, e)` with $p + e = a \cdot b$ exactly.
#[inline]
#[must_use]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Unevaluated sum of two IEEE-754 `f64` numbers `hi + lo` with
/// $|\mathrm{lo}| \le \tfrac{1}{2}\,\mathrm{ulp}(\mathrm{hi})$, providing ~106
/// bits of significand precision ($\approx 31$ decimal digits).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct F64x2 {
    /// Leading IEEE-754 double-precision component.
    pub hi: f64,
    /// Trailing roundoff compensation component.
    pub lo: f64,
}

impl F64x2 {
    /// Additive identity `0.0`.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// Multiplicative identity `1.0`.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    /// Creates a new [`F64x2`] from an exact `f64` scalar.
    #[inline]
    #[must_use]
    pub const fn new(val: f64) -> Self {
        Self { hi: val, lo: 0.0 }
    }

    /// Alias for [`Self::new`].
    #[inline]
    #[must_use]
    pub const fn from_f64(val: f64) -> Self {
        Self::new(val)
    }

    /// Adds an `f64` scalar in place.
    #[inline]
    pub fn add_assign_f64(&mut self, rhs: f64) {
        *self += rhs;
    }

    /// Subtracts an `f64` scalar in place.
    #[inline]
    pub fn sub_assign_f64(&mut self, rhs: f64) {
        *self -= rhs;
    }

    /// Creates a normalized [`F64x2`] from `(hi, lo)`.
    #[inline]
    #[must_use]
    pub fn from_parts(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        Self { hi: s, lo: e }
    }

    /// Exact product of two `f64` values lifted into [`F64x2`].
    #[inline]
    #[must_use]
    pub fn mul_f64(a: f64, b: f64) -> Self {
        let (p, e) = two_prod(a, b);
        Self { hi: p, lo: e }
    }

    /// Rounds the extended-precision value back to a single IEEE-754 `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Absolute value $|x|$.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) {
            -self
        } else {
            self
        }
    }
}

impl From<f64> for F64x2 {
    #[inline]
    fn from(val: f64) -> Self {
        Self::new(val)
    }
}

impl Neg for F64x2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Add for F64x2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (s1, s2) = two_sum(self.hi, rhs.hi);
        let (t1, t2) = two_sum(self.lo, rhs.lo);
        let (s1, s2) = quick_two_sum(s1, s2 + t1);
        let (hi, lo) = quick_two_sum(s1, s2 + t2);
        Self { hi, lo }
    }
}

impl Add<f64> for F64x2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        let (s1, s2) = two_sum(self.hi, rhs);
        let (hi, lo) = quick_two_sum(s1, self.lo + s2);
        Self { hi, lo }
    }
}

impl AddAssign for F64x2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<f64> for F64x2 {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}

impl Sub for F64x2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<f64> for F64x2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for F64x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<f64> for F64x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        *self = *self - rhs;
    }
}

impl Mul for F64x2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (p1, p2) = two_prod(self.hi, rhs.hi);
        let p2 = self.lo.mul_add(rhs.hi, self.hi.mul_add(rhs.lo, p2));
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }
}

impl Mul<f64> for F64x2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        let (p1, p2) = two_prod(self.hi, rhs);
        let (hi, lo) = quick_two_sum(p1, self.lo.mul_add(rhs, p2));
        Self { hi, lo }
    }
}

impl MulAssign for F64x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for F64x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl Div for F64x2 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let q1 = self.hi / rhs.hi;
        let r = self - rhs * q1;
        let q2 = r.hi / rhs.hi;
        let r = r - rhs * q2;
        let q3 = r.hi / rhs.hi;
        let (s1, s2) = quick_two_sum(q1, q2);
        let (hi, lo) = quick_two_sum(s1, s2 + q3);
        Self { hi, lo }
    }
}

impl Div<f64> for F64x2 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self / Self::new(rhs)
    }
}
