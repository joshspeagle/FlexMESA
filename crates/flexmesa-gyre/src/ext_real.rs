// SPDX-License-Identifier: GPL-3.0-only
//! Extended-exponent reals: GYRE `ext_real_m`
//! (`gyre/gyre/src/ext/ext_real_m.fypp`).
//!
//! An [`ExtR`] holds a fraction `f` (`0.5 <= |f| < 1`, or `0`) and an
//! integer exponent `e`, representing `f * 2**e`. GYRE uses it for
//! discriminants and determinants whose magnitudes leave the range of
//! `real(RD)`. Each routine follows the upstream operation order; Fortran
//! intrinsics map to [`fraction`], [`exponent`] (gfortran's
//! `__builtin_frexp`), `SIGN`, `FLOOR` and `MOD`.

use std::cmp::Ordering;

use crate::math_m::pow_rr;
use crate::powi::powi_var;

/// `RADIX(1._RD)` as a real.
pub const RADIX_RD: f64 = 2.0;
/// `MINEXPONENT(0._RD)`.
pub const MINEXPONENT: i32 = -1021;
/// `DIGITS(0._RD)`.
pub const DIGITS: i32 = 53;

/// Fortran `FRACTION(x)` (`frexp` fraction; `FRACTION(+-0) = +-0`).
#[must_use]
pub fn fraction(x: f64) -> f64 {
    frexp(x).0
}

/// Fortran `EXPONENT(x)` (`frexp` exponent; `EXPONENT(0) = 0`).
#[must_use]
pub fn exponent(x: f64) -> i32 {
    frexp(x).1
}

/// Exact `frexp` on IEEE-754 bits: `x = f * 2**e` with `0.5 <= |f| < 1`.
#[must_use]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let biased = i32::try_from((bits >> 52) & 0x7FF).unwrap_or(0);
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    if biased == 0 {
        // Subnormal: normalise the mantissa
        let shift = i32::try_from(mant.leading_zeros()).unwrap_or(0) - 11;
        let m = (mant << shift) & 0x000F_FFFF_FFFF_FFFF;
        let e = -1022 - shift + 1;
        let f = f64::from_bits(sign | (1022_u64 << 52) | m);
        (f, e)
    } else {
        let f = f64::from_bits(sign | (1022_u64 << 52) | mant);
        (f, biased - 1022)
    }
}

/// GYRE `ext_rt`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtR {
    /// Fractional part.
    pub f: f64,
    /// Exponent.
    pub e: i32,
}

impl ExtR {
    /// Structure constructor `ext_rt(f, e)` (no normalisation), used
    /// internally by upstream.
    #[must_use]
    #[inline]
    pub const fn raw(f: f64, e: i32) -> Self {
        Self { f, e }
    }

    /// `ext_rt(r)` (`ext_rt_r_`): split `r` into fraction and exponent.
    #[must_use]
    #[inline]
    pub fn from_real(r: f64) -> Self {
        let (f, e) = split(r);
        Self { f, e }
    }

    /// `scale(rx, de)` (`scale_`): exponent shift; a zero fraction resets
    /// the exponent to 0.
    #[must_use]
    #[inline]
    pub fn scale(self, de: i32) -> Self {
        if self.f == 0.0 {
            Self { f: self.f, e: 0 }
        } else {
            Self {
                f: self.f,
                e: self.e + de,
            }
        }
    }

    /// `real(rx)` (`real_`).
    #[must_use]
    pub fn to_real(self) -> f64 {
        if self.f == 0.0 {
            0.0
        } else if self.e >= MINEXPONENT {
            self.f * powi_var(RADIX_RD, self.e)
        } else {
            (self.f
                * powi_var(RADIX_RD, (self.e - MINEXPONENT).max(-DIGITS - 1)))
                * powi_var(RADIX_RD, MINEXPONENT)
        }
    }

    /// `valid(rx)` (`valid_`).
    #[must_use]
    pub fn valid(self) -> bool {
        self.f.abs() >= powi_var(RADIX_RD, -1) && self.f.abs() < 1.0
    }

    /// `abs(rx)` (`abs_`).
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            f: self.f.abs(),
            e: self.e,
        }
    }

    /// Unary minus (`op_minus_rx_`).
    #[must_use]
    pub fn negate(self) -> Self {
        Self {
            f: -self.f,
            e: self.e,
        }
    }

    /// `this + that` (`op_plus_rx_rx_`).
    #[must_use]
    pub fn plus(self, that: Self) -> Self {
        if self.f == 0.0 {
            that
        } else if that.f == 0.0 {
            self
        } else {
            let (f, e) = if self.e > that.e {
                (
                    self.f + Self::raw(that.f, that.e - self.e).to_real(),
                    self.e,
                )
            } else {
                (
                    Self::raw(self.f, self.e - that.e).to_real() + that.f,
                    that.e,
                )
            };
            Self::from_real(f).scale(e)
        }
    }

    /// `this - that` (`op_minus_rx_rx_`).
    #[must_use]
    pub fn minus(self, that: Self) -> Self {
        if self.f == 0.0 {
            Self {
                f: -that.f,
                e: that.e,
            }
        } else if that.f == 0.0 {
            self
        } else {
            let (f, e) = if self.e > that.e {
                (
                    self.f - Self::raw(that.f, that.e - self.e).to_real(),
                    self.e,
                )
            } else {
                (
                    Self::raw(self.f, self.e - that.e).to_real() - that.f,
                    that.e,
                )
            };
            Self::from_real(f).scale(e)
        }
    }

    /// `this * that` (`op_times_rx_rx_`).
    #[must_use]
    pub fn times(self, that: Self) -> Self {
        if self.f == 0.0 || that.f == 0.0 {
            Self::from_real(0.0)
        } else {
            Self::from_real(self.f * that.f).scale(self.e + that.e)
        }
    }

    /// `this / that` (`op_divide_rx_rx_`).
    #[must_use]
    pub fn divide(self, that: Self) -> Self {
        if self.f == 0.0 && that.f != 0.0 {
            Self::from_real(0.0)
        } else {
            Self::from_real(self.f / that.f).scale(self.e - that.e)
        }
    }

    /// `this == that` (`op_eq_rx_rx_`).
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn ext_eq(self, that: Self) -> bool {
        self.e == that.e && self.f == that.f
    }

    /// `this < that` (`op_lt_rx_rx_`).
    #[must_use]
    pub fn lt(self, that: Self) -> bool {
        if self.f == 0.0 || that.f == 0.0 {
            self.f < that.f
        } else {
            let this_s = 1.0_f64.copysign(self.f);
            let that_s = 1.0_f64.copysign(that.f);
            if this_s == that_s {
                if this_s > 0.0 {
                    self.e < that.e || (self.e == that.e && self.f < that.f)
                } else {
                    self.e > that.e || (self.e == that.e && self.f < that.f)
                }
            } else {
                this_s < 0.0
            }
        }
    }

    /// `this > that` (`op_gt_rx_rx_`: `that < this`).
    #[must_use]
    pub fn gt(self, that: Self) -> bool {
        that.lt(self)
    }

    /// `this <= that` (`op_le_rx_rx_`: `.NOT. this > that`).
    #[must_use]
    pub fn le(self, that: Self) -> bool {
        !self.gt(that)
    }

    /// `this >= that` (`op_ge_rx_rx_`: `.NOT. this < that`).
    #[must_use]
    pub fn ge(self, that: Self) -> bool {
        !self.lt(that)
    }

    /// `exp(rx)` (`exp_`): `g = real(rx)/log(2)`, `e = FLOOR(g)`,
    /// `split(pow(2, g-e))` scaled by `e`.
    #[must_use]
    pub fn exp(self) -> Self {
        let g = self.to_real() / flexmesa_math::log(RADIX_RD);
        let e = floor_i32(g);
        Self::from_real(pow_rr(RADIX_RD, g - f64::from(e))).scale(e)
    }

    /// `sqrt(rx)` (`sqrt_`).
    #[must_use]
    pub fn sqrt(self) -> Self {
        let s = Self::from_real(self.f.sqrt());
        if self.e % 2 == 0 {
            s.scale(self.e / 2)
        } else {
            s.scale((self.e - 1) / 2)
                .times(Self::from_real(2.0_f64.sqrt()))
        }
    }

    /// `max(a, b)` (`max_`).
    #[must_use]
    pub fn max(a: Self, b: Self) -> Self {
        let a_s = 1.0_f64.copysign(a.f);
        let b_s = 1.0_f64.copysign(b.f);
        if a_s > 0.0 && b_s > 0.0 {
            match a.e.cmp(&b.e) {
                Ordering::Greater => a,
                Ordering::Less => b,
                Ordering::Equal => Self {
                    f: fmax(a.f, b.f),
                    e: a.e,
                },
            }
        } else if a_s < 0.0 && b_s < 0.0 {
            match a.e.cmp(&b.e) {
                Ordering::Greater => b,
                Ordering::Less => a,
                Ordering::Equal => Self {
                    f: fmax(a.f, b.f),
                    e: a.e,
                },
            }
        } else if a_s > 0.0 {
            a
        } else {
            b
        }
    }

    /// `min(a, b)` (`min_`: `-MAX(-a, -b)`).
    #[must_use]
    pub fn min(a: Self, b: Self) -> Self {
        Self::max(a.negate(), b.negate()).negate()
    }

    /// `ext_rt` op real: `this op ext_rt(that)`.
    #[must_use]
    pub fn add_r(self, that: f64) -> Self {
        self.plus(Self::from_real(that))
    }

    /// `this - ext_rt(that)`.
    #[must_use]
    pub fn sub_r(self, that: f64) -> Self {
        self.minus(Self::from_real(that))
    }

    /// `this * ext_rt(that)`.
    #[must_use]
    pub fn mul_r(self, that: f64) -> Self {
        self.times(Self::from_real(that))
    }

    /// `this / ext_rt(that)`.
    #[must_use]
    pub fn div_r(self, that: f64) -> Self {
        self.divide(Self::from_real(that))
    }
}

/// `product(rx)` (`product_`).
#[must_use]
pub fn product(rx: &[ExtR]) -> ExtR {
    let mut p = ExtR {
        f: 1.0,
        e: rx.iter().map(|x| x.e).sum(),
    };
    for x in rx {
        let g = p.f * x.f;
        p.f = fraction(g);
        p = p.scale(exponent(g));
    }
    p
}

/// `split_`: `f = FRACTION(x)`, `e = EXPONENT(x)`.
#[must_use]
#[inline]
pub fn split(x: f64) -> (f64, i32) {
    frexp(x)
}

/// Fortran `FLOOR(g)` to default integer.
#[allow(clippy::cast_possible_truncation)]
fn floor_i32(g: f64) -> i32 {
    g.floor() as i32
}

/// gfortran `MAX(a, b)` for reals (`MAX_EXPR`, `maxsd`: the second operand
/// unless the first is greater).
#[inline]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

impl std::ops::Add for ExtR {
    type Output = Self;
    fn add(self, that: Self) -> Self {
        self.plus(that)
    }
}

impl std::ops::Sub for ExtR {
    type Output = Self;
    fn sub(self, that: Self) -> Self {
        self.minus(that)
    }
}

impl std::ops::Mul for ExtR {
    type Output = Self;
    fn mul(self, that: Self) -> Self {
        self.times(that)
    }
}

impl std::ops::Div for ExtR {
    type Output = Self;
    fn div(self, that: Self) -> Self {
        self.divide(that)
    }
}

impl std::ops::Neg for ExtR {
    type Output = Self;
    fn neg(self) -> Self {
        self.negate()
    }
}
