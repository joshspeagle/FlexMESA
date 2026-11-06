// SPDX-License-Identifier: LGPL-3.0-or-later
//! Third-order scalar derivative kernels for [`crate::AutoDiffReal2VarOrder3`].
//!
//! Every unary kernel returns `(f(x), f'(x), f''(x), f'''(x))` in IEEE-754
//! `f64` precision matching upstream MESA's
//! `auto_diff_real_2var_order3_module`.

use flexmesa_const::{ILN10, LN10, LN2, PI, PI2};
use flexmesa_math::{
    acos, acosh, asin, asinh, atan, atanh, cos, cosh, cospi, exp, exp10, log,
    log10, log1p, log2, pow, pow2, pow3, pow4, pow5, powm1, safe_log,
    safe_log10, sgn, sin, sinh, sinpi, tan, tanh, tanpi, SAFE_LOG_FLOOR,
};

/// Return `(f, f', f'', f''')` for $\operatorname{sign}(x)$.
#[must_use]
#[inline]
pub fn unary3_sign(x: f64) -> (f64, f64, f64, f64) {
    (sgn(x), 0.0, 0.0, 0.0)
}

/// Return `(f, f', f'', f''')` for $|x|$ with subgradient
/// $\operatorname{sgn}(x)$.
#[must_use]
#[inline]
pub fn unary3_abs(x: f64) -> (f64, f64, f64, f64) {
    (x.abs(), sgn(x), 0.0, 0.0)
}

/// Return `(f, f', f'', f''')` for $\sqrt{x}$ on $x > 0$.
#[must_use]
#[inline]
pub fn unary3_sqrt(x: f64) -> (f64, f64, f64, f64) {
    let q0 = x.sqrt();
    let d1 = 0.5 / q0;
    let d2 = -0.25 / (x * q0);
    let d3 = 0.375 / (pow2(x) * q0);
    (q0, d1, d2, d3)
}

/// Return `(f, f', f'', f''')` for $\sqrt{\max(x, 0)}$.
#[must_use]
#[inline]
pub fn unary3_safe_sqrt(x: f64) -> (f64, f64, f64, f64) {
    if x.is_nan() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else if x <= 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        unary3_sqrt(x)
    }
}

/// Return `(f, f', f'', f''')` for $\exp(x) = e^{x}$.
#[must_use]
#[inline]
pub fn unary3_exp(x: f64) -> (f64, f64, f64, f64) {
    let q0 = exp(x);
    (q0, q0, q0, q0)
}

/// Return `(f, f', f'', f''')` for $e^{x} - 1$.
#[must_use]
#[inline]
pub fn unary3_expm1(x: f64) -> (f64, f64, f64, f64) {
    let q0 = exp(x);
    (q0 - 1.0, q0, q0, q0)
}

/// Return `(f, f', f'', f''')` for $10^{x}$.
#[must_use]
#[inline]
pub fn unary3_exp10(x: f64) -> (f64, f64, f64, f64) {
    let q0 = exp10(x);
    let d1 = q0 * LN10;
    let d2 = d1 * LN10;
    let d3 = d2 * LN10;
    (q0, d1, d2, d3)
}

/// Return `(f, f', f'', f''')` for $x^{-1}$.
#[must_use]
#[inline]
pub fn unary3_powm1(x: f64) -> (f64, f64, f64, f64) {
    let inv = powm1(x);
    let inv2 = pow2(inv);
    (inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2)
}

/// Return `(f, f', f'', f''')` for $\ln(x)$.
#[must_use]
#[inline]
pub fn unary3_log(x: f64) -> (f64, f64, f64, f64) {
    let inv = powm1(x);
    let inv2 = pow2(inv);
    (log(x), inv, -inv2, 2.0 * inv2 * inv)
}

/// Return `(f, f', f'', f''')` for $\operatorname{safe\_log}(x)$.
#[must_use]
#[inline]
pub fn unary3_safe_log(x: f64) -> (f64, f64, f64, f64) {
    let val = safe_log(x);
    if x.is_finite() && x > SAFE_LOG_FLOOR {
        let inv = powm1(x);
        let inv2 = pow2(inv);
        (val, inv, -inv2, 2.0 * inv2 * inv)
    } else {
        (val, 0.0, 0.0, 0.0)
    }
}

/// Return `(f, f', f'', f''')` for $\ln(1 + x)$.
#[must_use]
#[inline]
pub fn unary3_log1p(x: f64) -> (f64, f64, f64, f64) {
    let inv = powm1(1.0 + x);
    let inv2 = pow2(inv);
    (log1p(x), inv, -inv2, 2.0 * inv2 * inv)
}

/// Return `(f, f', f'', f''')` for $\log_{10}(x)$.
#[must_use]
#[inline]
pub fn unary3_log10(x: f64) -> (f64, f64, f64, f64) {
    let inv_x = powm1(x);
    let inv2 = pow2(inv_x);
    let d3 = 2.0 * ILN10 * inv2 * inv_x;
    (log10(x), ILN10 * inv_x, -ILN10 * inv2, d3)
}

/// Return `(f, f', f'', f''')` for $\operatorname{safe\_log10}(x)$.
#[must_use]
#[inline]
pub fn unary3_safe_log10(x: f64) -> (f64, f64, f64, f64) {
    let val = safe_log10(x);
    if x.is_finite() && x > SAFE_LOG_FLOOR {
        let inv_x = powm1(x);
        let inv2 = pow2(inv_x);
        let d3 = 2.0 * ILN10 * inv2 * inv_x;
        (val, ILN10 * inv_x, -ILN10 * inv2, d3)
    } else {
        (val, 0.0, 0.0, 0.0)
    }
}

/// Return `(f, f', f'', f''')` for $\log_{2}(x)$.
#[must_use]
#[inline]
pub fn unary3_log2(x: f64) -> (f64, f64, f64, f64) {
    let inv_ln2 = powm1(LN2);
    let inv_x = powm1(x);
    let inv2 = pow2(inv_x);
    let d3 = 2.0 * inv_ln2 * inv2 * inv_x;
    (log2(x), inv_ln2 * inv_x, -inv_ln2 * inv2, d3)
}

/// Return `(f, f', f'', f''')` for $\sin(x)$.
#[must_use]
#[inline]
pub fn unary3_sin(x: f64) -> (f64, f64, f64, f64) {
    let s = sin(x);
    let c = cos(x);
    (s, c, -s, -c)
}

/// Return `(f, f', f'', f''')` for $\cos(x)$.
#[must_use]
#[inline]
pub fn unary3_cos(x: f64) -> (f64, f64, f64, f64) {
    let s = sin(x);
    let c = cos(x);
    (c, -s, -c, s)
}

/// Return `(f, f', f'', f''')` for $\tan(x)$.
#[must_use]
#[inline]
pub fn unary3_tan(x: f64) -> (f64, f64, f64, f64) {
    let t = tan(x);
    let t2 = pow2(t);
    let sec2 = t2 + 1.0;
    (t, sec2, 2.0 * t * sec2, 2.0 * sec2 * (3.0 * t2 + 1.0))
}

/// Return `(f, f', f'', f''')` for $\sin(\pi x)$.
#[must_use]
#[inline]
pub fn unary3_sinpi(x: f64) -> (f64, f64, f64, f64) {
    let s = sinpi(x);
    let c = cospi(x);
    (s, PI * c, -PI2 * s, -PI2 * PI * c)
}

/// Return `(f, f', f'', f''')` for $\cos(\pi x)$.
#[must_use]
#[inline]
pub fn unary3_cospi(x: f64) -> (f64, f64, f64, f64) {
    let s = sinpi(x);
    let c = cospi(x);
    (c, -PI * s, -PI2 * c, PI2 * PI * s)
}

/// Return `(f, f', f'', f''')` for $\tan(\pi x)$.
#[must_use]
#[inline]
pub fn unary3_tanpi(x: f64) -> (f64, f64, f64, f64) {
    let t = tanpi(x);
    let t2 = pow2(t);
    let sec2 = t2 + 1.0;
    let d3 = 2.0 * PI2 * PI * sec2 * (3.0 * t2 + 1.0);
    (t, PI * sec2, 2.0 * PI2 * t * sec2, d3)
}

/// Return `(f, f', f'', f''')` for $\sinh(x)$.
#[must_use]
#[inline]
pub fn unary3_sinh(x: f64) -> (f64, f64, f64, f64) {
    let sh = sinh(x);
    let ch = cosh(x);
    (sh, ch, sh, ch)
}

/// Return `(f, f', f'', f''')` for $\cosh(x)$.
#[must_use]
#[inline]
pub fn unary3_cosh(x: f64) -> (f64, f64, f64, f64) {
    let sh = sinh(x);
    let ch = cosh(x);
    (ch, sh, ch, sh)
}

/// Return `(f, f', f'', f''')` for $\tanh(x)$.
#[must_use]
#[inline]
pub fn unary3_tanh(x: f64) -> (f64, f64, f64, f64) {
    let th = tanh(x);
    let th2 = pow2(th);
    let sech2 = 1.0 - th2;
    let d3 = -2.0 * sech2 * (1.0 - 3.0 * th2);
    (th, sech2, -2.0 * th * sech2, d3)
}

/// Return `(f, f', f'', f''')` for $\arcsin(x)$.
#[must_use]
#[inline]
pub fn unary3_asin(x: f64) -> (f64, f64, f64, f64) {
    let x2 = pow2(x);
    let q1 = 1.0 - x2;
    let sq = q1.sqrt();
    let d3 = (1.0 + 2.0 * x2) / (pow2(q1) * sq);
    (asin(x), 1.0 / sq, x / (q1 * sq), d3)
}

/// Return `(f, f', f'', f''')` for $\arccos(x)$.
#[must_use]
#[inline]
pub fn unary3_acos(x: f64) -> (f64, f64, f64, f64) {
    let (_, d1, d2, d3) = unary3_asin(x);
    (acos(x), -d1, -d2, -d3)
}

/// Return `(f, f', f'', f''')` for $\arctan(x)$.
#[must_use]
#[inline]
pub fn unary3_atan(x: f64) -> (f64, f64, f64, f64) {
    let x2 = pow2(x);
    let q0 = 1.0 + x2;
    let q0_2 = pow2(q0);
    let d3 = (6.0 * x2 - 2.0) / (q0_2 * q0);
    (atan(x), 1.0 / q0, -2.0 * x / q0_2, d3)
}

/// Return `(f, f', f'', f''')` for $\arcsin(x) / \pi$.
#[must_use]
#[inline]
pub fn unary3_asinpi(x: f64) -> (f64, f64, f64, f64) {
    let (v, d1, d2, d3) = unary3_asin(x);
    let inv_pi = powm1(PI);
    (v / PI, d1 * inv_pi, d2 * inv_pi, d3 * inv_pi)
}

/// Return `(f, f', f'', f''')` for $\arccos(x) / \pi$.
#[must_use]
#[inline]
pub fn unary3_acospi(x: f64) -> (f64, f64, f64, f64) {
    let (v, d1, d2, d3) = unary3_acos(x);
    let inv_pi = powm1(PI);
    (v / PI, d1 * inv_pi, d2 * inv_pi, d3 * inv_pi)
}

/// Return `(f, f', f'', f''')` for $\arctan(x) / \pi$.
#[must_use]
#[inline]
pub fn unary3_atanpi(x: f64) -> (f64, f64, f64, f64) {
    let (v, d1, d2, d3) = unary3_atan(x);
    let inv_pi = powm1(PI);
    (v / PI, d1 * inv_pi, d2 * inv_pi, d3 * inv_pi)
}

/// Return `(f, f', f'', f''')` for $\operatorname{arsinh}(x)$.
#[must_use]
#[inline]
pub fn unary3_asinh(x: f64) -> (f64, f64, f64, f64) {
    let x2 = pow2(x);
    let q0 = x2 + 1.0;
    let sq = q0.sqrt();
    let d3 = (2.0 * x2 - 1.0) / (pow2(q0) * sq);
    (asinh(x), 1.0 / sq, -x / (q0 * sq), d3)
}

/// Return `(f, f', f'', f''')` for $\operatorname{arcosh}(x)$.
#[must_use]
#[inline]
pub fn unary3_acosh(x: f64) -> (f64, f64, f64, f64) {
    let x2 = pow2(x);
    let q0 = x2 - 1.0;
    let sq = q0.sqrt();
    let d3 = (2.0 * x2 + 1.0) / (pow2(q0) * sq);
    (acosh(x), 1.0 / sq, -x / (q0 * sq), d3)
}

/// Return `(f, f', f'', f''')` for $\operatorname{artanh}(x)$.
#[must_use]
#[inline]
pub fn unary3_atanh(x: f64) -> (f64, f64, f64, f64) {
    let x2 = pow2(x);
    let q0 = 1.0 - x2;
    let q0_2 = pow2(q0);
    let d3 = (6.0 * x2 + 2.0) / (q0_2 * q0);
    (atanh(x), 1.0 / q0, 2.0 * x / q0_2, d3)
}

/// Return `(f, f', f'', f''')` for integer powers $x^{2}$ through $x^{8}$.
#[must_use]
#[inline]
pub fn unary3_pown(x: f64, n: u32) -> (f64, f64, f64, f64) {
    match n {
        2 => (pow2(x), 2.0 * x, 2.0, 0.0),
        3 => (pow3(x), 3.0 * pow2(x), 6.0 * x, 6.0),
        4..=8 => {
            let p_nm3 = match n {
                4 => x,
                5 => pow2(x),
                6 => pow3(x),
                7 => pow4(x),
                _ => pow5(x),
            };
            let p_nm2 = p_nm3 * x;
            let p_nm1 = p_nm2 * x;
            let nf = f64::from(n);
            let d2 = nf * (nf - 1.0) * p_nm2;
            let d3 = nf * (nf - 1.0) * (nf - 2.0) * p_nm3;
            (p_nm1 * x, nf * p_nm1, d2, d3)
        }
        _ => unary3_pow_scalar(x, f64::from(n)),
    }
}

/// Return `(f, f', f'', f''')` for $x^{p}$ with constant exponent $p$.
#[must_use]
#[inline]
pub fn unary3_pow_scalar(x: f64, p: f64) -> (f64, f64, f64, f64) {
    let d2 = p * (p - 1.0) * pow(x, p - 2.0);
    let d3 = p * (p - 1.0) * (p - 2.0) * pow(x, p - 3.0);
    (pow(x, p), p * pow(x, p - 1.0), d2, d3)
}

/// Return `(f, f', f'', f''')` for $b^{x}$ with constant base $b > 0$.
#[must_use]
#[inline]
pub fn unary3_rpow_scalar(base: f64, x: f64) -> (f64, f64, f64, f64) {
    let val = pow(base, x);
    let lnb = log(base);
    let d1 = val * lnb;
    let d2 = d1 * lnb;
    (val, d1, d2, d2 * lnb)
}
