// SPDX-License-Identifier: GPL-3.0-only
//! Double-precision complex arithmetic exactly as gfortran 15.2.0 expands it
//! for GYRE 8.1 and the MESA SDK `26.6.1` LAPACK.
//!
//! gfortran compiles Fortran with `-fcx-fortran-rules`, so the middle end
//! lowers complex arithmetic inline (`tree-complex.cc`), with no NaN or
//! infinity recovery and no library calls. The expansions below transcribe
//! the GIMPLE that GCC 15.2.0 emits at `-O2 -ffp-contract=off` (GYRE's
//! flags) for each operand pattern. `benchmarks/fortran_mesa/gyre_g1_vectors/`
//! records the GIMPLE and the generator checks every function here bit for
//! bit, including signed zeros:
//!
//! - `a*b` (both complex): `(ar*br - ai*bi, ar*bi + ai*br)`.
//! - `a*r` and `r*a` (one operand real): the real operand is widened to
//!   `(r, 0)` and the full product is kept, so the zero terms `ai*0`,
//!   `ar*0` stay in the expansion.
//! - `a/b`: Smith's method (`expand_complex_div_wide`), branching on
//!   `|br| < |bi|`.
//! - `a/r`: the same Smith expansion with `bi = 0`; GCC drops the branch
//!   (`|r| < 0` is false) but keeps the `0/r` ratio terms.
//! - `r/b`: Smith's method with `ai = 0`.
//! - `z**2`: `(zr*zr - zi*zi, (zr*zi)*2)`; `z**(-1)` is `1/z`.
//!
//! Every routine here is an IEEE-754 operation sequence; Rust never
//! contracts `a*b + c` into a fused multiply-add, which matches
//! `-ffp-contract=off` and the FMA-free reference objects.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A double-precision complex number with gfortran's arithmetic semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cplx {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Cplx {
    /// `CMPLX(re, im, RD)`.
    #[must_use]
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `CMPLX(re, KIND=RD)`: imaginary part `+0`.
    #[must_use]
    #[inline]
    pub const fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// `CONJG(z)`.
    #[must_use]
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Fortran `z == w`: componentwise IEEE equality (`-0 == +0`).
    #[must_use]
    #[inline]
    #[allow(clippy::float_cmp)]
    pub fn feq(self, other: Self) -> bool {
        self.re == other.re && self.im == other.im
    }

    /// Fortran `z == 0._RD`: both parts compare equal to zero.
    #[must_use]
    #[inline]
    pub fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    /// `z*r` with `r` real (`(zr*r ... )`, GIMPLE: `r*zr - zi*0`, `zr*0 + r*zi`).
    #[must_use]
    #[inline]
    pub fn mul_real(self, r: f64) -> Self {
        Self {
            re: r * self.re - self.im * 0.0,
            im: self.re * 0.0 + r * self.im,
        }
    }

    /// `r*z` with `r` real; same expansion as [`Cplx::mul_real`].
    #[must_use]
    #[inline]
    pub fn real_mul(r: f64, z: Self) -> Self {
        z.mul_real(r)
    }

    /// `z/r` with `r` real: Smith's bottom branch with `bi = 0`
    /// (`ratio = 0/r`, `div = r + 0*ratio`).
    #[must_use]
    #[inline]
    pub fn div_real(self, r: f64) -> Self {
        let ratio = 0.0 / r;
        let div = r + ratio * 0.0;
        let tr = self.re + self.im * ratio;
        let ti = self.im - self.re * ratio;
        Self {
            re: tr / div,
            im: ti / div,
        }
    }

    /// `r/z` with `r` real: Smith's method with `ai = 0`.
    #[must_use]
    #[inline]
    pub fn real_div(r: f64, b: Self) -> Self {
        if b.re.abs() < b.im.abs() {
            let ratio = b.re / b.im;
            let div = b.im + b.re * ratio;
            let tr = r * ratio + 0.0;
            let ti = ratio * 0.0 - r;
            Self {
                re: tr / div,
                im: ti / div,
            }
        } else {
            let ratio = b.im / b.re;
            let div = b.re + b.im * ratio;
            let tr = r + ratio * 0.0;
            let ti = 0.0 - r * ratio;
            Self {
                re: tr / div,
                im: ti / div,
            }
        }
    }

    /// `z**2` (`tree-ssa-math-opts` complex square).
    #[must_use]
    #[inline]
    pub fn powi2(self) -> Self {
        Self {
            re: self.re * self.re - self.im * self.im,
            im: (self.re * self.im) * 2.0,
        }
    }

    /// `z**(-1)`, expanded as `1/z`.
    #[must_use]
    #[inline]
    pub fn powi_m1(self) -> Self {
        Self::real_div(1.0, self)
    }
}

impl Add for Cplx {
    type Output = Self;
    #[inline]
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}

impl Sub for Cplx {
    type Output = Self;
    #[inline]
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
}

impl Neg for Cplx {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul for Cplx {
    type Output = Self;
    /// `(ar*br - ai*bi, ar*bi + ai*br)`.
    #[inline]
    fn mul(self, b: Self) -> Self {
        let t1 = self.re * b.re;
        let t2 = self.im * b.im;
        let t3 = self.re * b.im;
        let t4 = self.im * b.re;
        Self {
            re: t1 - t2,
            im: t3 + t4,
        }
    }
}

impl Div for Cplx {
    type Output = Self;
    /// Smith's method as `expand_complex_div_wide` emits it.
    #[inline]
    fn div(self, b: Self) -> Self {
        if b.re.abs() < b.im.abs() {
            let ratio = b.re / b.im;
            let div = b.im + b.re * ratio;
            let tr = self.im + self.re * ratio;
            let ti = self.im * ratio - self.re;
            Self {
                re: tr / div,
                im: ti / div,
            }
        } else {
            let ratio = b.im / b.re;
            let div = b.re + b.im * ratio;
            let tr = self.re + self.im * ratio;
            let ti = self.im - self.re * ratio;
            Self {
                re: tr / div,
                im: ti / div,
            }
        }
    }
}
