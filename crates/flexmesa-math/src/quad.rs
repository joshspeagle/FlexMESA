// SPDX-License-Identifier: GPL-3.0-or-later
//! IEEE-754 `binary128` arithmetic in software, for upstream `real(qp)`.
//!
//! MESA `26.4.1` declares `qp = selected_real_kind(33, 4931)`
//! (`const/public/const_def.f90`), which `gfortran` on `x86_64` maps to
//! `REAL(16)`: IEEE-754 `binary128` with a 113-bit significand and a 15-bit
//! exponent, evaluated in software by `libgcc`'s `soft-fp` routines
//! (`__addtf3`, `__subtf3`, `__multf3`, `__divtf3`, `__extenddftf2`,
//! `__trunctfdf2`, `__floatsitf`, `__powitf2`) in round-to-nearest, ties to
//! even. Upstream uses the kind for sums and differences that cancel in
//! `real(dp)`, for example `cell_specific_KE_qp` and `cell_specific_PE_qp`
//! (`star/private/star_utils.f90`), and only with `+`, `-`, `*`, `/`, integer
//! powers, `abs`, comparisons, and conversions to and from `real(dp)` and
//! integers.
//!
//! [`Quad`] implements exactly those operations with the same rounding, so a
//! translated `real(qp)` expression gives the same bits as upstream when it
//! performs the same operations in the same order. It is not a double-double:
//! a double-double carries about 106 significand bits with a rounding rule
//! that differs from `binary128`, so its results can differ from upstream's in
//! the last place after conversion back to `f64`.
//!
//! The type holds no platform-dependent state and uses integer arithmetic
//! only, so results are identical in `strict` and `fast` numerics modes and on
//! every target.

// Exponents here are bounded by the binary128 and binary64 formats (at most
// 15 bits plus small shifts) and significands are masked before narrowing, so
// the integer casts below cannot wrap, change sign or drop set bits.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use std::cmp::Ordering;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Number of fraction bits stored in a `binary128` value.
const FRAC_BITS: u32 = 112;
/// Exponent bias of `binary128`.
const BIAS: i32 = 16383;
/// Largest biased exponent field (infinities and NaNs).
const EXP_MAX: i32 = 0x7fff;
/// Mask of the 112 stored fraction bits.
const FRAC_MASK: u128 = (1_u128 << FRAC_BITS) - 1;
/// The implicit leading significand bit of a normal number.
const IMPLICIT: u128 = 1_u128 << FRAC_BITS;
/// Sign bit.
const SIGN_MASK: u128 = 1_u128 << 127;
/// Bits of positive infinity.
const INF_BITS: u128 = (EXP_MAX as u128) << FRAC_BITS;
/// Bits of the default quiet NaN that `libgcc` returns for invalid operations
/// on `x86_64` (sign set, quiet bit set, zero payload).
const DEFAULT_NAN_BITS: u128 = SIGN_MASK | INF_BITS | (1_u128 << (FRAC_BITS - 1));
/// Position of the leading significand bit in the working format used by
/// [`round_pack`]: 113 significand bits followed by 3 guard bits.
const WORK_TOP: u32 = FRAC_BITS + 3;

/// An IEEE-754 `binary128` floating-point number (Fortran `real(qp)` under
/// `gfortran` on `x86_64`).
///
/// Arithmetic operators round to nearest, ties to even. Equality and ordering
/// follow IEEE-754: `+0 == -0`, and a NaN is unordered and unequal to
/// everything including itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct Quad(u128);

/// A finite non-zero value split into sign, exponent and significand.
///
/// The value is `(-1)^sign * sig * 2^(exp - 112)` with `sig` in
/// `[2^112, 2^113)`.
#[derive(Clone, Copy)]
struct Parts {
    sign: bool,
    exp: i32,
    sig: u128,
}

/// IEEE-754 class of a value, carrying [`Parts`] when finite and non-zero.
#[derive(Clone, Copy)]
enum Class {
    Nan,
    Inf(bool),
    Zero(bool),
    Finite(Parts),
}

/// Shifts `x` right by `n` bits, setting bit 0 of the result if any non-zero
/// bit was shifted out ("sticky" shift).
#[inline]
const fn shr_sticky(x: u128, n: u32) -> u128 {
    if n == 0 {
        x
    } else if n >= 128 {
        (x != 0) as u128
    } else {
        (x >> n) | ((x << (128 - n)) != 0) as u128
    }
}

/// Rounds and packs `(-1)^sign * sig * 2^(exp - 115)` into `binary128`.
///
/// `sig` may be any non-zero `u128`; bit 0 is treated as a sticky bit that
/// already summarises all lower-order bits of the exact value. Rounding is to
/// nearest, ties to even, with gradual underflow and overflow to infinity.
fn round_pack(sign: bool, mut exp: i32, mut sig: u128) -> Quad {
    debug_assert!(sig != 0);
    let sign_bits = if sign { SIGN_MASK } else { 0 };
    // Normalise so that the leading bit sits at `WORK_TOP`.
    let top = sig.ilog2();
    if top > WORK_TOP {
        let n = top - WORK_TOP;
        sig = shr_sticky(sig, n);
        exp += n as i32;
    } else if top < WORK_TOP {
        let n = WORK_TOP - top;
        sig <<= n;
        exp -= n as i32;
    }
    let mut biased = exp + BIAS;
    if biased >= EXP_MAX {
        return Quad(sign_bits | INF_BITS);
    }
    if biased <= 0 {
        // Gradual underflow: denormalise to the minimum exponent.
        let n = (1 - biased) as u32;
        sig = shr_sticky(sig, n);
        biased = 0;
    }
    let round_bits = sig & 0b111;
    let mut mant = sig >> 3;
    if round_bits > 0b100 || (round_bits == 0b100 && (mant & 1) == 1) {
        mant += 1;
    }
    // For a normal number `mant` holds the implicit bit, which is removed by
    // subtracting `IMPLICIT`; a carry to `2^113` then increments the exponent
    // field, including the step to infinity. For a subnormal number the
    // exponent field is zero and a carry to `2^112` produces the smallest
    // normal number.
    let bits = if biased == 0 {
        mant
    } else {
        ((biased as u128) << FRAC_BITS) + (mant - IMPLICIT)
    };
    Quad(sign_bits | bits)
}

impl Quad {
    /// Positive zero.
    pub const ZERO: Self = Self(0);
    /// One.
    pub const ONE: Self = Self((BIAS as u128) << FRAC_BITS);
    /// Positive infinity.
    pub const INFINITY: Self = Self(INF_BITS);
    /// The quiet NaN returned for invalid operations.
    pub const NAN: Self = Self(DEFAULT_NAN_BITS);

    /// Builds a value from its IEEE-754 `binary128` bit pattern.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    /// Returns the IEEE-754 `binary128` bit pattern.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u128 {
        self.0
    }

    /// Returns `true` for a NaN.
    #[inline]
    #[must_use]
    pub const fn is_nan(self) -> bool {
        (self.0 & !SIGN_MASK) > INF_BITS
    }

    /// Returns `true` for positive or negative infinity.
    #[inline]
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        (self.0 & !SIGN_MASK) == INF_BITS
    }

    /// Returns `true` unless the value is infinite or a NaN.
    #[inline]
    #[must_use]
    pub const fn is_finite(self) -> bool {
        (self.0 & !SIGN_MASK) < INF_BITS
    }

    /// Returns `true` when the sign bit is set, including for `-0` and NaNs
    /// with the sign bit set.
    #[inline]
    #[must_use]
    pub const fn is_sign_negative(self) -> bool {
        (self.0 & SIGN_MASK) != 0
    }

    /// Absolute value (Fortran `abs`): clears the sign bit.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0 & !SIGN_MASK)
    }

    fn classify(self) -> Class {
        let sign = self.is_sign_negative();
        let biased = ((self.0 >> FRAC_BITS) & 0x7fff) as i32;
        let frac = self.0 & FRAC_MASK;
        if biased == EXP_MAX {
            if frac == 0 {
                Class::Inf(sign)
            } else {
                Class::Nan
            }
        } else if biased == 0 {
            if frac == 0 {
                Class::Zero(sign)
            } else {
                // Subnormal: normalise the significand.
                let shift = frac.leading_zeros() - (127 - FRAC_BITS);
                Class::Finite(Parts {
                    sign,
                    exp: 1 - BIAS - shift as i32,
                    sig: frac << shift,
                })
            }
        } else {
            Class::Finite(Parts {
                sign,
                exp: biased - BIAS,
                sig: frac | IMPLICIT,
            })
        }
    }

    /// Converts an `f64` exactly (Fortran `real(x, kind=qp)`, `libgcc`
    /// `__extenddftf2`). A NaN converts to [`Quad::NAN`].
    #[must_use]
    pub fn from_f64(x: f64) -> Self {
        let bits = x.to_bits();
        let sign_bits = if (bits >> 63) != 0 { SIGN_MASK } else { 0 };
        let biased = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1_u64 << 52) - 1);
        if biased == 0x7ff {
            return if frac == 0 {
                Self(sign_bits | INF_BITS)
            } else {
                Self::NAN
            };
        }
        if biased == 0 {
            if frac == 0 {
                return Self(sign_bits);
            }
            // Subnormal `f64`: value is `frac * 2^-1074`.
            let top = frac.ilog2();
            let sig = u128::from(frac) << (FRAC_BITS - top);
            let exp = top as i32 - 1074;
            return Self(
                sign_bits
                    | (((exp + BIAS) as u128) << FRAC_BITS)
                    | (sig & FRAC_MASK),
            );
        }
        let exp = biased - 1023;
        Self(
            sign_bits
                | (((exp + BIAS) as u128) << FRAC_BITS)
                | (u128::from(frac) << (FRAC_BITS - 52)),
        )
    }

    /// Converts an `i64` exactly (Fortran `real(i, kind=qp)`).
    #[must_use]
    pub fn from_i64(i: i64) -> Self {
        if i == 0 {
            return Self::ZERO;
        }
        let mag = u128::from(i.unsigned_abs());
        let top = mag.ilog2();
        let sign_bits = if i < 0 { SIGN_MASK } else { 0 };
        Self(
            sign_bits
                | (((top as i32 + BIAS) as u128) << FRAC_BITS)
                | ((mag << (FRAC_BITS - top)) & FRAC_MASK),
        )
    }

    /// Converts to `f64`, rounding to nearest, ties to even (Fortran
    /// `real(x, kind=dp)`, `libgcc` `__trunctfdf2`).
    ///
    /// Values too large for `f64` become infinite, and values below the `f64`
    /// subnormal range round to zero with the sign kept.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        let p = match self.classify() {
            Class::Nan => return f64::NAN,
            Class::Inf(s) => {
                return if s { f64::NEG_INFINITY } else { f64::INFINITY }
            }
            Class::Zero(s) => return if s { -0.0 } else { 0.0 },
            Class::Finite(p) => p,
        };
        let sign_bits = u64::from(p.sign) << 63;
        let mut biased = p.exp + 1023;
        if biased >= 0x7ff {
            return f64::from_bits(sign_bits | (0x7ff_u64 << 52));
        }
        // Keep 53 significand bits and 3 guard bits (leading bit at 55).
        let mut sig = shr_sticky(p.sig, FRAC_BITS - 55);
        if biased <= 0 {
            sig = shr_sticky(sig, (1 - biased) as u32);
            biased = 0;
        }
        let round_bits = sig & 0b111;
        let mut mant = (sig >> 3) as u64;
        if round_bits > 0b100 || (round_bits == 0b100 && (mant & 1) == 1) {
            mant += 1;
        }
        let bits = if biased == 0 {
            mant
        } else {
            ((biased as u64) << 52) + (mant - (1_u64 << 52))
        };
        f64::from_bits(sign_bits | bits)
    }

    /// Integer power by repeated multiplication, following `libgcc`
    /// `__powitf2`, which `gfortran` calls for `x**n` with a `real(qp)` base
    /// and an integer exponent that is not a small compile-time constant.
    ///
    /// The products are formed in the order of binary exponentiation from the
    /// least significant bit of `|n|`, and a negative exponent takes the
    /// reciprocal of the result at the end. For `n = 2` and `n = 3` this is
    /// `x * x` and `x * (x * x)`, the same values `gfortran` produces when it
    /// expands a constant exponent in line.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        let mut m = n.unsigned_abs();
        let mut x = self;
        let mut y = if m % 2 == 1 { x } else { Self::ONE };
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            x *= x;
            if m % 2 == 1 {
                y *= x;
            }
        }
        if n < 0 {
            Self::ONE / y
        } else {
            y
        }
    }
}

impl From<f64> for Quad {
    #[inline]
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl From<i32> for Quad {
    #[inline]
    fn from(i: i32) -> Self {
        Self::from_i64(i64::from(i))
    }
}

impl From<i64> for Quad {
    #[inline]
    fn from(i: i64) -> Self {
        Self::from_i64(i)
    }
}

impl Neg for Quad {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0 ^ SIGN_MASK)
    }
}

/// Adds two finite non-zero magnitudes with the given result sign handling.
fn add_finite(a: Parts, b: Parts) -> Quad {
    // Order by magnitude so that `hi` has the larger exponent (and the larger
    // significand when the exponents are equal).
    let (hi, lo) = if (a.exp, a.sig) >= (b.exp, b.sig) {
        (a, b)
    } else {
        (b, a)
    };
    let d = (hi.exp - lo.exp) as u32;
    let hi_sig = hi.sig << 3;
    let lo_sig = shr_sticky(lo.sig << 3, d);
    if hi.sign == lo.sign {
        round_pack(hi.sign, hi.exp, hi_sig + lo_sig)
    } else {
        let diff = hi_sig - lo_sig;
        if diff == 0 {
            // Exact cancellation gives `+0` in round-to-nearest.
            Quad::ZERO
        } else {
            round_pack(hi.sign, hi.exp, diff)
        }
    }
}

impl Add for Quad {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (self.classify(), rhs.classify()) {
            (Class::Nan, _) | (_, Class::Nan) => Self::NAN,
            (Class::Inf(sa), Class::Inf(sb)) => {
                if sa == sb {
                    self
                } else {
                    Self::NAN
                }
            }
            (Class::Zero(sa), Class::Zero(sb)) => {
                if sa && sb {
                    self
                } else {
                    Self::ZERO
                }
            }
            (Class::Inf(_), _) | (_, Class::Zero(_)) => self,
            (_, Class::Inf(_)) | (Class::Zero(_), _) => rhs,
            (Class::Finite(a), Class::Finite(b)) => add_finite(a, b),
        }
    }
}

impl Sub for Quad {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        if rhs.is_nan() {
            Self::NAN
        } else {
            self + (-rhs)
        }
    }
}

/// Full 256-bit product of two `u128` values as `(high, low)`.
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const M: u128 = (1_u128 << 64) - 1;
    let (a1, a0) = (a >> 64, a & M);
    let (b1, b0) = (b >> 64, b & M);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & M) + (p10 & M);
    let lo = (p00 & M) | (mid << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

impl Mul for Quad {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let sign = self.is_sign_negative() != rhs.is_sign_negative();
        let sign_bits = if sign { SIGN_MASK } else { 0 };
        match (self.classify(), rhs.classify()) {
            (Class::Nan, _) | (_, Class::Nan) => Self::NAN,
            (Class::Inf(_), Class::Zero(_)) | (Class::Zero(_), Class::Inf(_)) => {
                Self::NAN
            }
            (Class::Inf(_), _) | (_, Class::Inf(_)) => {
                Self(sign_bits | INF_BITS)
            }
            (Class::Zero(_), _) | (_, Class::Zero(_)) => Self(sign_bits),
            (Class::Finite(a), Class::Finite(b)) => {
                // The product of two significands in `[2^112, 2^113)` lies in
                // `[2^224, 2^226)`. Keep its top bits with the bit of weight
                // `2^224` at `WORK_TOP` and fold the rest into the sticky bit.
                let (hi, lo) = mul_wide(a.sig, b.sig);
                let drop = 224 - WORK_TOP;
                let sig = (hi << (128 - drop))
                    | (lo >> drop)
                    | u128::from((lo << (128 - drop)) != 0);
                round_pack(sign, a.exp + b.exp, sig)
            }
        }
    }
}

/// Exact quotient `floor(a * 2^115 / b)` of two significands in `[2^112, 2^113)`
/// with bit 0 set if the division is inexact (the input `round_pack` expects).
///
/// Knuth's Algorithm D (`TAOCP` vol. 2, 4.3.1) on `32`-bit limbs: `b` is
/// normalised by a left shift of `15` so that its top limb has its high bit
/// set, which bounds the quotient-digit estimate error by two. The result is
/// bit-identical to the bit-serial restoring division it replaces
/// (`div_sig_bitserial` in the unit tests), because both compute the exact
/// integer quotient and the exact "remainder is non-zero" flag.
#[inline]
fn div_sig(a: u128, b: u128) -> u128 {
    const BASE: u64 = 1 << 32;
    debug_assert!(a >> FRAC_BITS == 1 && b >> FRAC_BITS == 1);
    let v = b << 15;
    let vn = [v as u32, (v >> 32) as u32, (v >> 64) as u32, (v >> 96) as u32];
    // Numerator `a * 2^115 * 2^15 = (a << 2) * 2^128`: the low four limbs
    // are zero and the high four hold `a << 2` (at most 115 bits, below `v`).
    let hi = a << 2;
    let mut un = [
        0_u32,
        0,
        0,
        0,
        hi as u32,
        (hi >> 32) as u32,
        (hi >> 64) as u32,
        (hi >> 96) as u32,
    ];
    let mut quo = 0_u128;
    for j in (0..4).rev() {
        let num = (u64::from(un[j + 4]) << 32) | u64::from(un[j + 3]);
        let mut qhat = num / u64::from(vn[3]);
        let mut rhat = num % u64::from(vn[3]);
        while qhat >= BASE
            || qhat * u64::from(vn[2]) > ((rhat << 32) | u64::from(un[j + 2]))
        {
            qhat -= 1;
            rhat += u64::from(vn[3]);
            if rhat >= BASE {
                break;
            }
        }
        // Multiply and subtract `qhat * v` from `un[j..=j + 4]`.
        let mut borrow = 0_i64;
        for i in 0..4 {
            let prod = qhat * u64::from(vn[i]);
            let diff =
                i64::from(un[i + j]) - borrow - (prod & 0xffff_ffff) as i64;
            un[i + j] = diff as u32;
            borrow = (prod >> 32) as i64 - (diff >> 32);
        }
        let diff = i64::from(un[j + 4]) - borrow;
        un[j + 4] = diff as u32;
        if diff < 0 {
            // `qhat` was one too large: add `v` back.
            qhat -= 1;
            let mut carry = 0_u64;
            for i in 0..4 {
                let sum = u64::from(un[i + j]) + u64::from(vn[i]) + carry;
                un[i + j] = sum as u32;
                carry = sum >> 32;
            }
            un[j + 4] = un[j + 4].wrapping_add(carry as u32);
        }
        quo |= u128::from(qhat as u32) << (32 * j);
    }
    let inexact = (un[0] | un[1] | un[2] | un[3]) != 0;
    quo | u128::from(inexact)
}

impl Div for Quad {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let sign = self.is_sign_negative() != rhs.is_sign_negative();
        let sign_bits = if sign { SIGN_MASK } else { 0 };
        match (self.classify(), rhs.classify()) {
            (Class::Nan, _) | (_, Class::Nan) => Self::NAN,
            (Class::Inf(_), Class::Inf(_)) | (Class::Zero(_), Class::Zero(_)) => {
                Self::NAN
            }
            (Class::Inf(_), _) | (_, Class::Zero(_)) => {
                Self(sign_bits | INF_BITS)
            }
            (Class::Zero(_), _) | (_, Class::Inf(_)) => Self(sign_bits),
            (Class::Finite(a), Class::Finite(b)) => {
                // `floor(a.sig * 2^115 / b.sig)`, which lies in
                // `(2^114, 2^116)`, with the remainder folded into the sticky
                // bit (`div_sig`).
                round_pack(sign, a.exp - b.exp, div_sig(a.sig, b.sig))
            }
        }
    }
}

impl AddAssign for Quad {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Quad {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Quad {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for Quad {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl PartialEq for Quad {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for Quad {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        let (ma, mb) = (self.0 & !SIGN_MASK, other.0 & !SIGN_MASK);
        if ma == 0 && mb == 0 {
            return Some(Ordering::Equal);
        }
        Some(match (self.is_sign_negative(), other.is_sign_negative()) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => ma.cmp(&mb),
            (true, true) => mb.cmp(&ma),
        })
    }
}

// Mixed `real(qp)` and `real(dp)` arithmetic. Fortran converts the `real(dp)`
// operand to `real(qp)`, which is exact, and then applies the `real(qp)`
// operation; these impls do the same.
macro_rules! impl_quad_f64_ops {
    ($($trait:ident $method:ident $assign_trait:ident $assign_method:ident;)*) => {$(
        impl $trait<f64> for Quad {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: f64) -> Self {
                $trait::$method(self, Self::from_f64(rhs))
            }
        }

        impl $assign_trait<f64> for Quad {
            #[inline]
            fn $assign_method(&mut self, rhs: f64) {
                *self = $trait::$method(*self, Self::from_f64(rhs));
            }
        }
    )*};
}

impl_quad_f64_ops! {
    Add add AddAssign add_assign;
    Sub sub SubAssign sub_assign;
    Mul mul MulAssign mul_assign;
    Div div DivAssign div_assign;
}

#[cfg(test)]
mod div_sig_tests {
    use super::{div_sig, FRAC_BITS, WORK_TOP};

    /// The restoring long division that `div_sig` replaces.
    fn div_sig_bitserial(a: u128, b: u128) -> u128 {
        let mut rem = a;
        let mut quo = 0_u128;
        for _ in 0..=WORK_TOP {
            quo <<= 1;
            if rem >= b {
                rem -= b;
                quo |= 1;
            }
            rem <<= 1;
        }
        quo | u128::from(rem != 0)
    }

    /// `SplitMix64` step, for a dependency-free deterministic sequence.
    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn sig(s: &mut u64, mode: u64) -> u128 {
        let frac_mask = (1_u128 << FRAC_BITS) - 1;
        let r = (u128::from(next(s)) << 64) | u128::from(next(s));
        let f = match mode % 4 {
            0 => r & frac_mask,
            1 => frac_mask - (r & 0xff),     // just below 2^113
            2 => r & 0xff,                   // just above 2^112
            _ => (r & frac_mask) & !0xffff_ffff, // many trailing zeros
        };
        (1_u128 << FRAC_BITS) | f
    }

    #[test]
    fn test_div_sig_matches_bit_serial_division_on_random_and_edge_inputs() {
        let mut s = 0x1234_5678_9abc_def0_u64;
        for i in 0..2_000_000_u64 {
            let ma = next(&mut s) % 4;
            let a = sig(&mut s, ma);
            let mb = next(&mut s) % 4;
            let b = sig(&mut s, mb);
            assert_eq!(div_sig(a, b), div_sig_bitserial(a, b), "iter {i}");
        }
        let one = 1_u128 << FRAC_BITS;
        for (a, b) in [(one, one), (one, 2 * one - 1), (2 * one - 1, one)] {
            assert_eq!(div_sig(a, b), div_sig_bitserial(a, b));
        }
    }
}
