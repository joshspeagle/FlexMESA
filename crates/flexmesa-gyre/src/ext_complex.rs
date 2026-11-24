// SPDX-License-Identifier: GPL-3.0-only
//! Extended-exponent complex numbers: GYRE `ext_complex_m`
//! (`gyre/gyre/src/ext/ext_complex_m.fypp`).
//!
//! An [`ExtC`] holds a complex fraction `f` (the larger-exponent component
//! normalised by `frexp`, the other scaled to the same exponent) and an
//! integer exponent `e`. The operations follow upstream's order, including
//! one asymmetry: `op_plus_cx_cx_` adds the exponent shift directly
//! (`cx%e = cx%e + e`) instead of calling `scale`, so a sum that cancels to
//! zero keeps a non-zero exponent, while `op_minus_cx_cx_` calls `scale`
//! and resets it.

use crate::crmath::{abs_c, exp_c, sqrt_c};
use crate::ext_real::{exponent, fraction, ExtR, MINEXPONENT, RADIX_RD};
use crate::fcomplex::Cplx;
use crate::powi::powi_var;

/// `DIGITS(0._RD)`.
const DIGITS: i32 = 53;

/// GYRE `ext_ct`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtC {
    /// Fractional part.
    pub f: Cplx,
    /// Exponent.
    pub e: i32,
}

/// `split_` (`ext_complex_m.fypp:860-919`).
#[must_use]
pub fn split(c: Cplx) -> (Cplx, i32) {
    let c_r = c.re;
    let c_i = c.im;
    let mut f_r = fraction(c_r);
    let mut f_i = fraction(c_i);
    if f_r == 0.0 && f_i == 0.0 {
        return (Cplx::new(0.0, 0.0), 0);
    }
    let e_r = exponent(c_r);
    let e_i = exponent(c_i);
    // MAXVAL([e_r, e_i], MASK=[f_r, f_i] /= 0)
    let e_ref = match (f_r != 0.0, f_i != 0.0) {
        (true, true) => e_r.max(e_i),
        (true, false) => e_r,
        _ => e_i,
    };
    if f_r != 0.0 {
        if e_r - e_ref < MINEXPONENT {
            f_r = 0.0;
        } else {
            f_r *= powi_var(RADIX_RD, e_r - e_ref);
        }
    }
    if f_i != 0.0 {
        if e_i - e_ref < MINEXPONENT {
            f_i = 0.0;
        } else {
            f_i *= powi_var(RADIX_RD, e_i - e_ref);
        }
    }
    (Cplx::new(f_r, f_i), e_ref)
}

impl ExtC {
    /// Structure constructor `ext_ct(f, e)` (no normalisation).
    #[must_use]
    #[inline]
    pub const fn raw(f: Cplx, e: i32) -> Self {
        Self { f, e }
    }

    /// `ext_ct(c)` (`ext_ct_c_`).
    #[must_use]
    #[inline]
    pub fn from_cplx(c: Cplx) -> Self {
        let (f, e) = split(c);
        Self { f, e }
    }

    /// `ext_ct(r)` (`ext_ct_r_`: `split_(CMPLX(r, KIND=RD))`).
    #[must_use]
    #[inline]
    pub fn from_real(r: f64) -> Self {
        Self::from_cplx(Cplx::from_real(r))
    }

    /// `ext_ct(r, i)` (`ext_ct_r_i_`).
    #[must_use]
    #[inline]
    pub fn from_parts(r: f64, i: f64) -> Self {
        Self::from_cplx(Cplx::new(r, i))
    }

    /// `ext_ct(rx)` (`ext_ct_rx_`).
    #[must_use]
    pub fn from_extr(rx: ExtR) -> Self {
        Self::from_cplx(Cplx::from_real(rx.f)).scale(rx.e)
    }

    /// `ext_ct(rx, ix)` (`ext_ct_rx_ix_`: `ext_ct(rx) + ext_ct(0,1)*ext_ct(ix)`).
    #[must_use]
    pub fn from_extr_parts(rx: ExtR, ix: ExtR) -> Self {
        Self::from_extr(rx)
            .plus(Self::from_parts(0.0, 1.0).times(Self::from_extr(ix)))
    }

    /// `scale(cx, de)` (`scale_`).
    #[must_use]
    #[inline]
    pub fn scale(self, de: i32) -> Self {
        if self.f.is_zero() {
            Self { f: self.f, e: 0 }
        } else {
            Self {
                f: self.f,
                e: self.e + de,
            }
        }
    }

    /// `cmplx(cx)` (`cmplx_`).
    #[must_use]
    pub fn to_cplx(self) -> Cplx {
        if self.f.is_zero() {
            Cplx::new(0.0, 0.0)
        } else if self.e >= MINEXPONENT {
            self.f.mul_real(powi_var(RADIX_RD, self.e))
        } else {
            self.f
                .mul_real(powi_var(
                    RADIX_RD,
                    (self.e - MINEXPONENT).max(-DIGITS - 1),
                ))
                .mul_real(powi_var(RADIX_RD, MINEXPONENT))
        }
    }

    /// `real(cx)` (`real_`: `REAL(cmplx(cx))`).
    #[must_use]
    pub fn to_real(self) -> f64 {
        self.to_cplx().re
    }

    /// `ext_rt(cx)` (`ext_rt_cx_`).
    #[must_use]
    pub fn to_extr(self) -> ExtR {
        ExtR::from_real(self.f.re).scale(self.e)
    }

    /// `real_part(cx)`.
    #[must_use]
    pub fn real_part(self) -> ExtR {
        ExtR::from_real(self.f.re).scale(self.e)
    }

    /// `imag_part(cx)`.
    #[must_use]
    pub fn imag_part(self) -> ExtR {
        ExtR::from_real(self.f.im).scale(self.e)
    }

    /// `valid(cx)` (`valid_`: tests the real part of the fraction only).
    #[must_use]
    pub fn valid(self) -> bool {
        self.f.re.abs() >= powi_var(RADIX_RD, -1) && self.f.re.abs() < 1.0
    }

    /// `conjg(cx)`.
    #[must_use]
    pub fn conj(self) -> Self {
        Self {
            f: self.f.conj(),
            e: self.e,
        }
    }

    /// Unary minus (`op_minus_cx_`).
    #[must_use]
    pub fn negate(self) -> Self {
        Self {
            f: -self.f,
            e: self.e,
        }
    }

    /// `this + that` (`op_plus_cx_cx_`; exponent shift added without `scale`).
    #[must_use]
    pub fn plus(self, that: Self) -> Self {
        if self.f.is_zero() {
            that
        } else if that.f.is_zero() {
            self
        } else {
            let (f, e) = if self.e > that.e {
                (
                    self.f + Self::raw(that.f, that.e - self.e).to_cplx(),
                    self.e,
                )
            } else {
                (
                    Self::raw(self.f, self.e - that.e).to_cplx() + that.f,
                    that.e,
                )
            };
            let (sf, se) = split(f);
            Self { f: sf, e: se + e }
        }
    }

    /// `this - that` (`op_minus_cx_cx_`).
    #[must_use]
    pub fn minus(self, that: Self) -> Self {
        if self.f.is_zero() {
            Self {
                f: -that.f,
                e: that.e,
            }
        } else if that.f.is_zero() {
            self
        } else {
            let (f, e) = if self.e > that.e {
                (
                    self.f - Self::raw(that.f, that.e - self.e).to_cplx(),
                    self.e,
                )
            } else {
                (
                    Self::raw(self.f, self.e - that.e).to_cplx() - that.f,
                    that.e,
                )
            };
            Self::from_cplx(f).scale(e)
        }
    }

    /// `this * that` (`op_times_cx_cx_`).
    #[must_use]
    pub fn times(self, that: Self) -> Self {
        if self.f.is_zero() || that.f.is_zero() {
            Self::from_real(0.0)
        } else {
            Self::from_cplx(self.f * that.f).scale(self.e + that.e)
        }
    }

    /// `this / that` (`op_divide_cx_cx_`).
    #[must_use]
    pub fn divide(self, that: Self) -> Self {
        if self.f.is_zero() && !that.f.is_zero() {
            Self::from_real(0.0)
        } else {
            Self::from_cplx(self.f / that.f).scale(self.e - that.e)
        }
    }

    /// `this == that` (`op_eq_cx_cx_`).
    #[must_use]
    pub fn ext_eq(self, that: Self) -> bool {
        self.e == that.e && self.f.feq(that.f)
    }

    /// `abs(cx)` (`abs_`: `scale(ext_rt(abs(f)), e)`).
    #[must_use]
    pub fn abs(self) -> ExtR {
        ExtR::from_real(abs_c(self.f)).scale(self.e)
    }

    /// `exp(cx)` (`exp_`).
    #[must_use]
    pub fn exp(self) -> Self {
        let exp_ex = self.to_extr().exp();
        let phase = exp_c(Cplx::new(0.0, 1.0).mul_real(self.to_cplx().im));
        Self {
            f: Cplx::real_mul(exp_ex.f, phase),
            e: exp_ex.e,
        }
    }

    /// `sqrt(cx)` (`sqrt_`).
    #[must_use]
    pub fn sqrt(self) -> Self {
        let s = Self::from_cplx(sqrt_c(self.f));
        if self.e % 2 == 0 {
            s.scale(self.e / 2)
        } else {
            s.scale((self.e - 1) / 2)
                .times(Self::from_real(2.0_f64.sqrt()))
        }
    }
}

/// `product(cx)` (`product_`).
#[must_use]
pub fn product(cx: &[ExtC]) -> ExtC {
    let mut p = ExtC {
        f: Cplx::new(1.0, 0.0),
        e: cx.iter().map(|x| x.e).sum(),
    };
    for x in cx {
        let (f, e) = split(p.f * x.f);
        p.f = f;
        p = p.scale(e);
    }
    p
}

impl std::ops::Add for ExtC {
    type Output = Self;
    fn add(self, that: Self) -> Self {
        self.plus(that)
    }
}

impl std::ops::Sub for ExtC {
    type Output = Self;
    fn sub(self, that: Self) -> Self {
        self.minus(that)
    }
}

impl std::ops::Mul for ExtC {
    type Output = Self;
    fn mul(self, that: Self) -> Self {
        self.times(that)
    }
}

impl std::ops::Div for ExtC {
    type Output = Self;
    fn div(self, that: Self) -> Self {
        self.divide(that)
    }
}

impl std::ops::Neg for ExtC {
    type Output = Self;
    fn neg(self) -> Self {
        self.negate()
    }
}
