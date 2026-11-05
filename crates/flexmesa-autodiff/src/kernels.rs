// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared scalar value, first-derivative, and second-derivative kernels for
//! unary and binary elementary operations in `flexmesa-autodiff`.

use flexmesa_const::{LN10, PI};
use flexmesa_math::{
    acos, acosh, acospi, asin, asinh, asinpi, atan, atanh, atanpi, cos, cosh,
    cospi, exp, exp10, expm1, heaviside, log, log10, log1p, log2, pow, pow2,
    pow3, pow4, pow5, pow6, pow7, pow8, powm1, safe_log, safe_log10, sgn, sin,
    sinh, sinpi, tan, tanh, tanpi,
};

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{sign}(x)$.
#[must_use]
#[inline]
pub fn unary_sign(x: f64) -> (f64, f64, f64) {
    (sgn(x), 0.0, 0.0)
}

/// Return `(f(x), f'(x), f''(x))` for $|x|$.
#[must_use]
#[inline]
pub fn unary_abs(x: f64) -> (f64, f64, f64) {
    (x.abs(), sgn(x), 0.0)
}

/// Return `(f(x), f'(x), f''(x))` for $\sqrt{x}$.
#[must_use]
#[inline]
pub fn unary_sqrt(x: f64) -> (f64, f64, f64) {
    let q0 = x.sqrt();
    let d1 = 0.5 / q0;
    let d2 = -0.25 / (x * q0);
    (q0, d1, d2)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{safe\_sqrt}(x) = \sqrt{x H(x)}$.
#[must_use]
#[inline]
pub fn unary_safe_sqrt(x: f64) -> (f64, f64, f64) {
    let q0 = (x * heaviside(x)).sqrt();
    let d1 = 0.5 * q0 * powm1(x);
    let d2 = -0.25 * q0 * powm1(pow2(x));
    (q0, d1, d2)
}

/// Return `(f(x), f'(x), f''(x))` for $\exp(x)$.
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
    (expm1(x), q0, q0)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{exp10}(x) = 10^{x}$.
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

/// Return `(f(x), f'(x), f''(x))` for $\ln(x)$.
#[must_use]
#[inline]
pub fn unary_log(x: f64) -> (f64, f64, f64) {
    let inv = powm1(x);
    (log(x), inv, -pow2(inv))
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{safe\_log}(x)$.
#[must_use]
#[inline]
pub fn unary_safe_log(x: f64) -> (f64, f64, f64) {
    let inv = powm1(x);
    (safe_log(x), inv, -pow2(inv))
}

/// Return `(f(x), f'(x), f''(x))` for $\ln(1 + x)$.
#[must_use]
#[inline]
pub fn unary_log1p(x: f64) -> (f64, f64, f64) {
    let inv = powm1(1.0 + x);
    (log1p(x), inv, -pow2(inv))
}

/// Return `(f(x), f'(x), f''(x))` for $\log_{10}(x)$.
#[must_use]
#[inline]
pub fn unary_log10(x: f64) -> (f64, f64, f64) {
    let inv_ln10 = powm1(LN10);
    let inv_x = powm1(x);
    (log10(x), inv_ln10 * inv_x, -inv_ln10 * pow2(inv_x))
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{safe\_log10}(x)$.
#[must_use]
#[inline]
pub fn unary_safe_log10(x: f64) -> (f64, f64, f64) {
    let inv_ln10 = powm1(LN10);
    let inv_x = powm1(x);
    (safe_log10(x), inv_ln10 * inv_x, -inv_ln10 * pow2(inv_x))
}

/// Return `(f(x), f'(x), f''(x))` for $\log_{2}(x)$.
#[must_use]
#[inline]
pub fn unary_log2(x: f64) -> (f64, f64, f64) {
    let inv_ln2 = powm1(2.0_f64.ln());
    let inv_x = powm1(x);
    (log2(x), inv_ln2 * inv_x, -inv_ln2 * pow2(inv_x))
}

/// Return `(f(x), f'(x), f''(x))` for $\sin(x)$.
#[must_use]
#[inline]
pub fn unary_sin(x: f64) -> (f64, f64, f64) {
    let s = sin(x);
    let c = cos(x);
    (s, c, -s)
}

/// Return `(f(x), f'(x), f''(x))` for $\cos(x)$.
#[must_use]
#[inline]
pub fn unary_cos(x: f64) -> (f64, f64, f64) {
    let s = sin(x);
    let c = cos(x);
    (c, -s, -c)
}

/// Return `(f(x), f'(x), f''(x))` for $\tan(x)$.
#[must_use]
#[inline]
pub fn unary_tan(x: f64) -> (f64, f64, f64) {
    let t = tan(x);
    let sec2 = pow2(t) + 1.0;
    (t, sec2, 2.0 * t * sec2)
}

/// Return `(f(x), f'(x), f''(x))` for $\sin(\pi x)$.
#[must_use]
#[inline]
pub fn unary_sinpi(x: f64) -> (f64, f64, f64) {
    let s = sinpi(x);
    let c = cospi(x);
    (s, PI * c, -PI * PI * s)
}

/// Return `(f(x), f'(x), f''(x))` for $\cos(\pi x)$.
#[must_use]
#[inline]
pub fn unary_cospi(x: f64) -> (f64, f64, f64) {
    let s = sinpi(x);
    let c = cospi(x);
    (c, -PI * s, -PI * PI * c)
}

/// Return `(f(x), f'(x), f''(x))` for $\tan(\pi x)$.
#[must_use]
#[inline]
pub fn unary_tanpi(x: f64) -> (f64, f64, f64) {
    let t = tanpi(x);
    let sec2 = pow2(t) + 1.0;
    (t, PI * sec2, 2.0 * PI * PI * t * sec2)
}

/// Return `(f(x), f'(x), f''(x))` for $\sinh(x)$.
#[must_use]
#[inline]
pub fn unary_sinh(x: f64) -> (f64, f64, f64) {
    let sh = sinh(x);
    let ch = cosh(x);
    (sh, ch, sh)
}

/// Return `(f(x), f'(x), f''(x))` for $\cosh(x)$.
#[must_use]
#[inline]
pub fn unary_cosh(x: f64) -> (f64, f64, f64) {
    let sh = sinh(x);
    let ch = cosh(x);
    (ch, sh, ch)
}

/// Return `(f(x), f'(x), f''(x))` for $\tanh(x)$.
#[must_use]
#[inline]
pub fn unary_tanh(x: f64) -> (f64, f64, f64) {
    let th = tanh(x);
    let sech2 = 1.0 - pow2(th);
    (th, sech2, -2.0 * th * sech2)
}

/// Return `(f(x), f'(x), f''(x))` for $\arcsin(x)$.
#[must_use]
#[inline]
pub fn unary_asin(x: f64) -> (f64, f64, f64) {
    let q1 = 1.0 - pow2(x);
    let sq = q1.sqrt();
    (asin(x), 1.0 / sq, x / (q1 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for $\arccos(x)$.
#[must_use]
#[inline]
pub fn unary_acos(x: f64) -> (f64, f64, f64) {
    let q1 = 1.0 - pow2(x);
    let sq = q1.sqrt();
    (acos(x), -1.0 / sq, -x / (q1 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for $\arctan(x)$.
#[must_use]
#[inline]
pub fn unary_atan(x: f64) -> (f64, f64, f64) {
    let q0 = 1.0 + pow2(x);
    (atan(x), 1.0 / q0, -2.0 * x / pow2(q0))
}

/// Return `(f(x), f'(x), f''(x))` for $\arcsin(x) / \pi$.
#[must_use]
#[inline]
pub fn unary_asinpi(x: f64) -> (f64, f64, f64) {
    let (_, d1, d2) = unary_asin(x);
    let inv_pi = powm1(PI);
    (asinpi(x), d1 * inv_pi, d2 * inv_pi)
}

/// Return `(f(x), f'(x), f''(x))` for $\arccos(x) / \pi$.
#[must_use]
#[inline]
pub fn unary_acospi(x: f64) -> (f64, f64, f64) {
    let (_, d1, d2) = unary_acos(x);
    let inv_pi = powm1(PI);
    (acospi(x), d1 * inv_pi, d2 * inv_pi)
}

/// Return `(f(x), f'(x), f''(x))` for $\arctan(x) / \pi$.
#[must_use]
#[inline]
pub fn unary_atanpi(x: f64) -> (f64, f64, f64) {
    let (_, d1, d2) = unary_atan(x);
    let inv_pi = powm1(PI);
    (atanpi(x), d1 * inv_pi, d2 * inv_pi)
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{arsinh}(x)$.
#[must_use]
#[inline]
pub fn unary_asinh(x: f64) -> (f64, f64, f64) {
    let q0 = pow2(x) + 1.0;
    let sq = q0.sqrt();
    (asinh(x), 1.0 / sq, -x / (q0 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{arcosh}(x)$.
#[must_use]
#[inline]
pub fn unary_acosh(x: f64) -> (f64, f64, f64) {
    let q0 = pow2(x) - 1.0;
    let sq = q0.sqrt();
    (acosh(x), 1.0 / sq, -x / (q0 * sq))
}

/// Return `(f(x), f'(x), f''(x))` for $\operatorname{artanh}(x)$.
#[must_use]
#[inline]
pub fn unary_atanh(x: f64) -> (f64, f64, f64) {
    let q0 = 1.0 - pow2(x);
    (atanh(x), 1.0 / q0, 2.0 * x / pow2(q0))
}

/// Return `(f(x), f'(x), f''(x))` for $x^{2}$ through $x^{8}$.
#[must_use]
#[inline]
pub fn unary_pown(x: f64, n: u32) -> (f64, f64, f64) {
    match n {
        2 => (pow2(x), 2.0 * x, 2.0),
        3 => (pow3(x), 3.0 * pow2(x), 6.0 * x),
        4 => (pow4(x), 4.0 * pow3(x), 12.0 * pow2(x)),
        5 => (pow5(x), 5.0 * pow4(x), 20.0 * pow3(x)),
        6 => (pow6(x), 6.0 * pow5(x), 30.0 * pow4(x)),
        7 => (pow7(x), 7.0 * pow6(x), 42.0 * pow5(x)),
        8 => (pow8(x), 8.0 * pow7(x), 56.0 * pow6(x)),
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

/// Return `(z, dz/dx, dz/dy, d2z/dx2, d2z/dxdy, d2z/dy2)` for $x^{y}$.
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
