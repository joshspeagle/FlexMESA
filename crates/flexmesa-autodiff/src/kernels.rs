// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared scalar value, first-derivative, and second-derivative kernels for
//! unary and binary elementary operations in `flexmesa-autodiff`.
//!
//! Every unary kernel evaluates `(f(x), f'(x), f''(x))` in IEEE-754 `f64` so
//! that first-order dual numbers ([`crate::DualOrder1`]) and second-order
//! hyperdual numbers ([`crate::AutoDiffReal2VarOrder2`]) share identical
//! analytical derivative formulas and domain guards.

use flexmesa_const::{ILN10, LN10, LN2, PI, PI2};
use flexmesa_math::{
    SAFE_LOG_FLOOR, acos, acosh, asin, asinh, atan, atanh, cos, cosh, cospi,
    exp, exp10, log, log10, log1p, log2, pow, pow2, pow3, pow4, pow5, pow6,
    powm1, safe_log, safe_log10, sgn, sin, sinh, sinpi, tan, tanh, tanpi,
};

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{sign}(x)$ with zero
/// derivatives everywhere.
#[must_use]
#[inline]
pub fn unary_sign(x: f64) -> (f64, f64, f64) {
    (sgn(x), 0.0, 0.0)
}

/// Return `(f(x), f'(x), f''(x))` for $|x|$ with subgradient
/// $\operatorname{sgn}(x)$.
#[must_use]
#[inline]
pub fn unary_abs(x: f64) -> (f64, f64, f64) {
    (x.abs(), sgn(x), 0.0)
}

/// Return `(f(x), f'(x), f''(x))` for $\sqrt{x}$ on $x > 0$.
#[must_use]
#[inline]
pub fn unary_sqrt(x: f64) -> (f64, f64, f64) {
    let q0 = x.sqrt();
    let d1 = 0.5 / q0;
    let d2 = -0.25 / (x * q0);
    (q0, d1, d2)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{safe\_sqrt}(x) =
/// \sqrt{\max(x, 0)}$.
///
/// For $x \le 0$, both the primal value and derivatives evaluate to `0.0`
/// without computing `0.0 * Inf`.
#[must_use]
#[inline]
pub fn unary_safe_sqrt(x: f64) -> (f64, f64, f64) {
    if x.is_nan() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else if x <= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        unary_sqrt(x)
    }
}

/// Return `(f(x), f'(x), f''(x))` for $\exp(x) = e^{x}$.
#[must_use]
#[inline]
pub fn unary_exp(x: f64) -> (f64, f64, f64) {
    let q0 = exp(x);
    (q0, q0, q0)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{expm1}(x) = e^{x} - 1$.
#[must_use]
#[inline]
pub fn unary_expm1(x: f64) -> (f64, f64, f64) {
    let q0 = exp(x);
    (q0 - 1.0, q0, q0)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{exp10}(x) = 10^{x}$ with
/// derivatives scaled by $\ln(10)$ and $\ln(10)^{2}$.
#[must_use]
#[inline]
pub fn unary_exp10(x: f64) -> (f64, f64, f64) {
    let q0 = exp10(x);
    let d1 = q0 * LN10;
    let d2 = d1 * LN10;
    (q0, d1, d2)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{powm1}(x) = x^{-1}$.
#[must_use]
#[inline]
pub fn unary_powm1(x: f64) -> (f64, f64, f64) {
    let inv = powm1(x);
    let inv2 = pow2(inv);
    (inv, -inv2, 2.0 * inv2 * inv)
}

/// Return `(f(x), f'(x), f''(x))` for the natural logarithm $\ln(x)$.
#[must_use]
#[inline]
pub fn unary_log(x: f64) -> (f64, f64, f64) {
    let inv = powm1(x);
    (log(x), inv, -pow2(inv))
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{safe\_log}(x) =
/// \ln(\max(10^{-99}, x))$.
///
/// Outside the active domain ($x \le 10^{-99}$ or non-finite $x$), derivatives
/// evaluate to `0.0` so inactive partials never compute `0.0 * Inf`.
#[must_use]
#[inline]
pub fn unary_safe_log(x: f64) -> (f64, f64, f64) {
    let val = safe_log(x);
    if x.is_finite() && x > SAFE_LOG_FLOOR {
        let inv = powm1(x);
        (val, inv, -pow2(inv))
    } else {
        (val, 0.0, 0.0)
    }
}

/// Return `(f(x), f'(x), f''(x))` for the shifted logarithm $\ln(1 + x)$.
#[must_use]
#[inline]
pub fn unary_log1p(x: f64) -> (f64, f64, f64) {
    let inv = powm1(1.0 + x);
    (log1p(x), inv, -pow2(inv))
}

/// Return `(f(x), f'(x), f''(x))` for the base-10 logarithm $\log_{10}(x)$.
#[must_use]
#[inline]
pub fn unary_log10(x: f64) -> (f64, f64, f64) {
    let inv_x = powm1(x);
    (log10(x), ILN10 * inv_x, -ILN10 * pow2(inv_x))
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{safe\_log10}(x)$ with
/// zero derivatives on the clamped or non-finite branch.
#[must_use]
#[inline]
pub fn unary_safe_log10(x: f64) -> (f64, f64, f64) {
    let val = safe_log10(x);
    if x.is_finite() && x > SAFE_LOG_FLOOR {
        let inv_x = powm1(x);
        (val, ILN10 * inv_x, -ILN10 * pow2(inv_x))
    } else {
        (val, 0.0, 0.0)
    }
}

/// Return `(f(x), f'(x), f''(x))` for the base-2 logarithm $\log_{2}(x)$.
#[must_use]
#[inline]
pub fn unary_log2(x: f64) -> (f64, f64, f64) {
    let inv_ln2 = powm1(LN2);
    let inv_x = powm1(x);
    (log2(x), inv_ln2 * inv_x, -inv_ln2 * pow2(inv_x))
}

/// Return `(f(x), f'(x), f''(x))` for $\sin(x)$ with $x$ in radians.
#[must_use]
#[inline]
pub fn unary_sin(x: f64) -> (f64, f64, f64) {
    let s = sin(x);
    let c = cos(x);
    (s, c, -s)
}

/// Return `(f(x), f'(x), f''(x))` for $\cos(x)$ with $x$ in radians.
#[must_use]
#[inline]
pub fn unary_cos(x: f64) -> (f64, f64, f64) {
    let s = sin(x);
    let c = cos(x);
    (c, -s, -c)
}

/// Return `(f(x), f'(x), f''(x))` for $\tan(x)$ with $\sec^{2}(x) = 1 +
/// \tan^{2}(x)$.
#[must_use]
#[inline]
pub fn unary_tan(x: f64) -> (f64, f64, f64) {
    let t = tan(x);
    let sec2 = pow2(t) + 1.0;
    (t, sec2, 2.0 * t * sec2)
}

/// Return `(f(x), f'(x), f''(x))` for half-turn sine $\sin(\pi x)$.
#[must_use]
#[inline]
pub fn unary_sinpi(x: f64) -> (f64, f64, f64) {
    let s = sinpi(x);
    let c = cospi(x);
    (s, PI * c, -PI2 * s)
}

/// Return `(f(x), f'(x), f''(x))` for half-turn cosine $\cos(\pi x)$.
#[must_use]
#[inline]
pub fn unary_cospi(x: f64) -> (f64, f64, f64) {
    let s = sinpi(x);
    let c = cospi(x);
    (c, -PI * s, -PI2 * c)
}

/// Return `(f(x), f'(x), f''(x))` for half-turn tangent $\tan(\pi x)$.
#[must_use]
#[inline]
pub fn unary_tanpi(x: f64) -> (f64, f64, f64) {
    let t = tanpi(x);
    let sec2 = pow2(t) + 1.0;
    (t, PI * sec2, 2.0 * PI2 * t * sec2)
}

/// Return `(f(x), f'(x), f''(x))` for hyperbolic sine $\sinh(x)$.
#[must_use]
#[inline]
pub fn unary_sinh(x: f64) -> (f64, f64, f64) {
    let sh = sinh(x);
    let ch = cosh(x);
    (sh, ch, sh)
}

/// Return `(f(x), f'(x), f''(x))` for hyperbolic cosine $\cosh(x)$.
#[must_use]
#[inline]
pub fn unary_cosh(x: f64) -> (f64, f64, f64) {
    let sh = sinh(x);
    let ch = cosh(x);
    (ch, sh, ch)
}

/// Return `(f(x), f'(x), f''(x))` for hyperbolic tangent $\tanh(x)$.
#[must_use]
#[inline]
pub fn unary_tanh(x: f64) -> (f64, f64, f64) {
    let th = tanh(x);
    let sech2 = 1.0 - pow2(th);
    (th, sech2, -2.0 * th * sech2)
}

/// Return `(f(x), f'(x), f''(x))` for inverse sine $\arcsin(x)$ in radians.
#[must_use]
#[inline]
pub fn unary_asin(x: f64) -> (f64, f64, f64) {
    let q1 = 1.0 - pow2(x);
    let sq = q1.sqrt();
    (asin(x), 1.0 / sq, x / (q1 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for inverse cosine $\arccos(x)$ in radians.
#[must_use]
#[inline]
pub fn unary_acos(x: f64) -> (f64, f64, f64) {
    let q1 = 1.0 - pow2(x);
    let sq = q1.sqrt();
    (acos(x), -1.0 / sq, -x / (q1 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for inverse tangent $\arctan(x)$ in radians.
#[must_use]
#[inline]
pub fn unary_atan(x: f64) -> (f64, f64, f64) {
    let q0 = 1.0 + pow2(x);
    (atan(x), 1.0 / q0, -2.0 * x / pow2(q0))
}

/// Return `(f(x), f'(x), f''(x))` for half-turn inverse sine $\arcsin(x) /
/// \pi$.
#[must_use]
#[inline]
pub fn unary_asinpi(x: f64) -> (f64, f64, f64) {
    let (v, d1, d2) = unary_asin(x);
    let inv_pi = powm1(PI);
    (v / PI, d1 * inv_pi, d2 * inv_pi)
}

/// Return `(f(x), f'(x), f''(x))` for half-turn inverse cosine $\arccos(x) /
/// \pi$.
#[must_use]
#[inline]
pub fn unary_acospi(x: f64) -> (f64, f64, f64) {
    let (v, d1, d2) = unary_acos(x);
    let inv_pi = powm1(PI);
    (v / PI, d1 * inv_pi, d2 * inv_pi)
}

/// Return `(f(x), f'(x), f''(x))` for half-turn inverse tangent $\arctan(x) /
/// \pi$.
#[must_use]
#[inline]
pub fn unary_atanpi(x: f64) -> (f64, f64, f64) {
    let (v, d1, d2) = unary_atan(x);
    let inv_pi = powm1(PI);
    (v / PI, d1 * inv_pi, d2 * inv_pi)
}

/// Return `(f(x), f'(x), f''(x))` for inverse hyperbolic sine
/// $\operatorname{arsinh}(x)$.
#[must_use]
#[inline]
pub fn unary_asinh(x: f64) -> (f64, f64, f64) {
    let q0 = pow2(x) + 1.0;
    let sq = q0.sqrt();
    (asinh(x), 1.0 / sq, -x / (q0 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for inverse hyperbolic cosine
/// $\operatorname{arcosh}(x)$.
#[must_use]
#[inline]
pub fn unary_acosh(x: f64) -> (f64, f64, f64) {
    let q0 = pow2(x) - 1.0;
    let sq = q0.sqrt();
    (acosh(x), 1.0 / sq, -x / (q0 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for inverse hyperbolic tangent
/// $\operatorname{artanh}(x)$.
#[must_use]
#[inline]
pub fn unary_atanh(x: f64) -> (f64, f64, f64) {
    let q0 = 1.0 - pow2(x);
    (atanh(x), 1.0 / q0, 2.0 * x / pow2(q0))
}

/// Return `(f(x), f'(x), f''(x))` for integer powers $x^{2}$ through $x^{8}$.
///
/// Reuses $x^{n-2}$ and $x^{n-1} = x^{n-2} x$ to avoid redundant multiplications
/// while preserving left-associative IEEE-754 rounding.
#[must_use]
#[inline]
pub fn unary_pown(x: f64, n: u32) -> (f64, f64, f64) {
    match n {
        2 => (pow2(x), 2.0 * x, 2.0),
        3 => {
            let p2 = pow2(x);
            (p2 * x, 3.0 * p2, 6.0 * x)
        }
        4 => {
            let p2 = pow2(x);
            let p3 = p2 * x;
            (p3 * x, 4.0 * p3, 12.0 * p2)
        }
        5 => {
            let p3 = pow3(x);
            let p4 = p3 * x;
            (p4 * x, 5.0 * p4, 20.0 * p3)
        }
        6 => {
            let p4 = pow4(x);
            let p5 = p4 * x;
            (p5 * x, 6.0 * p5, 30.0 * p4)
        }
        7 => {
            let p5 = pow5(x);
            let p6 = p5 * x;
            (p6 * x, 7.0 * p6, 42.0 * p5)
        }
        8 => {
            let p6 = pow6(x);
            let p7 = p6 * x;
            (p7 * x, 8.0 * p7, 56.0 * p6)
        }
        _ => unary_pow_scalar(x, f64::from(n)),
    }
}

/// Return `(f(x), f'(x), f''(x))` for $x^{p}$ with constant exponent $p$.
#[must_use]
#[inline]
pub fn unary_pow_scalar(x: f64, p: f64) -> (f64, f64, f64) {
    let val = pow(x, p);
    let d1 = p * pow(x, p - 1.0);
    let d2 = p * (p - 1.0) * pow(x, p - 2.0);
    (val, d1, d2)
}

/// Return `(f(x), f'(x), f''(x))` for $b^{x}$ with constant base $b > 0$.
#[must_use]
#[inline]
pub fn unary_rpow_scalar(base: f64, x: f64) -> (f64, f64, f64) {
    let val = pow(base, x);
    let lnb = log(base);
    let d1 = val * lnb;
    let d2 = d1 * lnb;
    (val, d1, d2)
}

/// Return `(z, dz/dx, dz/dy, d2z/dx2, d2z/dxdy, d2z/dy2)` for $x^{y}$ with
/// $x > 0$.
#[must_use]
#[inline]
pub fn binary_pow(x: f64, y: f64) -> (f64, f64, f64, f64, f64, f64) {
    let val = pow(x, y);
    let lnx = log(x);
    let xm1 = pow(x, y - 1.0);
    let xm2 = pow(x, y - 2.0);
    let d1x = y * xm1;
    let d1y = val * lnx;
    let d2x = y * (y - 1.0) * xm2;
    let d1x_d1y = xm1 * (1.0 + y * lnx);
    let d2y = val * pow2(lnx);
    (val, d1x, d1y, d2x, d1x_d1y, d2y)
}
