// SPDX-License-Identifier: GPL-3.0-or-later
//! Generated from MESA `26.4.1` `auto_diff/private/auto_diff_real_2var_order2_module.f90`
//! by `scripts/autodiff/translate_mesa_dual.py`. Do not edit by hand:
//! rerun the script against a MESA checkout instead (`#349`).
//!
//! Each function keeps the upstream routine's name, its temporaries,
//! their order and every parenthesis, so the result rounds as MESA's.

#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::neg_multiply,
    clippy::field_reassign_with_default,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::similar_names,
    clippy::manual_midpoint,
    clippy::precedence,
    clippy::unnecessary_cast,
    unused_parens,
    dead_code
)]

use flexmesa_const::{LN10, PI};
use flexmesa_math::{
    acos, acosh, asin, asinh, atan, atanh, cos, cosh, exp, expm1, log,
    log1p, pow, pow2, pow3, pow4, pow5, pow6, pow7, pow8, powm1, safe_log,
    sin, sinh, tan, tanh,
};

use crate::support::{f_abs, f_max, f_min, f_sqrt, heaviside, sgn};
use crate::AutoDiffReal2VarOrder2;

/// `make_unary_operator` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn make_unary_operator(x: AutoDiffReal2VarOrder2, z_val: f64, z_d1x: f64, z_d2x: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = z_val;
    unary.d1val1 = x.d1val1*z_d1x;
    unary.d1val2 = x.d1val2*z_d1x;
    unary.d2val1 = x.d2val1*z_d1x + z_d2x*pow2(x.d1val1);
    unary.d1val1_d1val2 = x.d1val1*x.d1val2*z_d2x + x.d1val1_d1val2*z_d1x;
    unary.d2val2 = x.d2val2*z_d1x + z_d2x*pow2(x.d1val2);
    unary
}

/// `make_binary_operator` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn make_binary_operator(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2, z_val: f64, z_d1x: f64, z_d1y: f64, z_d2x: f64, z_d1x_d1y: f64, z_d2y: f64) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    let q0 = x.d1val1*z_d1x_d1y;
    let q1 = x.d1val2*z_d1x_d1y;
    binary.val = z_val;
    binary.d1val1 = x.d1val1*z_d1x + y.d1val1*z_d1y;
    binary.d1val2 = x.d1val2*z_d1x + y.d1val2*z_d1y;
    binary.d2val1 = 2.0*q0*y.d1val1 + x.d2val1*z_d1x + y.d2val1*z_d1y + z_d2x*pow2(x.d1val1) + z_d2y*pow2(y.d1val1);
    binary.d1val1_d1val2 = q0*y.d1val2 + q1*y.d1val1 + x.d1val1*x.d1val2*z_d2x  + x.d1val1_d1val2*z_d1x + y.d1val1*y.d1val2*z_d2y + y.d1val1_d1val2*z_d1y;
    binary.d2val2 = 2.0*q1*y.d1val2 + x.d2val2*z_d1x + y.d2val2*z_d1y + z_d2x*pow2(x.d1val2) + z_d2y*pow2(y.d1val2);
    binary
}

/// `sign_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sign_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = sgn(x.val);
    unary.d1val1 = 0.0;
    unary.d1val2 = 0.0;
    unary.d2val1 = 0.0;
    unary.d1val1_d1val2 = 0.0;
    unary.d2val2 = 0.0;
    unary
}

/// `safe_sqrt_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn safe_sqrt_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = f_sqrt(x.val*heaviside(x.val));
    let q1 = 0.5*q0*powm1(x.val);
    let q2 = 2.0*x.val;
    let q3 = 0.25*q0*powm1(pow2(x.val));
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q3*(q2*x.d2val1 - pow2(x.d1val1));
    unary.d1val1_d1val2 = q3*(q2*x.d1val1_d1val2 - x.d1val1*x.d1val2);
    unary.d2val2 = q3*(q2*x.d2val2 - pow2(x.d1val2));
    unary
}

/// `unary_minus_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn unary_minus_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = -x.val;
    unary.d1val1 = -x.d1val1;
    unary.d1val2 = -x.d1val2;
    unary.d2val1 = -x.d2val1;
    unary.d1val1_d1val2 = -x.d1val1_d1val2;
    unary.d2val2 = -x.d2val2;
    unary
}

/// `exp_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn exp_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = exp(x.val);
    unary.val = q0;
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*(x.d2val1 + pow2(x.d1val1));
    unary.d1val1_d1val2 = q0*(x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q0*(x.d2val2 + pow2(x.d1val2));
    unary
}

/// `expm1_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn expm1_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = exp(x.val);
    unary.val = expm1(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*(x.d2val1 + pow2(x.d1val1));
    unary.d1val1_d1val2 = q0*(x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q0*(x.d2val2 + pow2(x.d1val2));
    unary
}

/// `exp10_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn exp10_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow(10.0, x.val);
    let q1 = LN10;
    let q2 = q0*q1;
    unary.val = q0;
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*(q1*pow2(x.d1val1) + x.d2val1);
    unary.d1val1_d1val2 = q2*(q1*x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q2*(q1*pow2(x.d1val2) + x.d2val2);
    unary
}

/// `powm1_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn powm1_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(pow2(x.val));
    let q1 = powm1(pow3(x.val));
    unary.val = powm1(x.val);
    unary.d1val1 = -q0*x.d1val1;
    unary.d1val2 = -q0*x.d1val2;
    unary.d2val1 = q1*(2.0*pow2(x.d1val1) - x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(2.0*x.d1val1*x.d1val2 - x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(2.0*pow2(x.d1val2) - x.d2val2*x.val);
    unary
}

/// `log_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(x.val);
    let q1 = powm1(pow2(x.val));
    unary.val = log(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(x.d2val1*x.val - pow2(x.d1val1));
    unary.d1val1_d1val2 = q1*(-x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(x.d2val2*x.val - pow2(x.d1val2));
    unary
}

/// `log1p_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log1p_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = x.val + 1.0;
    let q1 = powm1(q0);
    let q2 = powm1(pow2(q0));
    unary.val = log1p(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(q0*x.d2val1 - pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(q0*x.d1val1_d1val2 - x.d1val1*x.d1val2);
    unary.d2val2 = q2*(q0*x.d2val2 - pow2(x.d1val2));
    unary
}

/// `safe_log_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn safe_log_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(x.val);
    let q1 = powm1(pow2(x.val));
    unary.val = safe_log(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(x.d2val1*x.val - pow2(x.d1val1));
    unary.d1val1_d1val2 = q1*(-x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(x.d2val2*x.val - pow2(x.d1val2));
    unary
}

/// `log10_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log10_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(LN10);
    let q1 = q0*powm1(x.val);
    let q2 = q0*powm1(pow2(x.val));
    unary.val = q0*log(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(x.d2val1*x.val - pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(-x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q2*(x.d2val2*x.val - pow2(x.d1val2));
    unary
}

/// `safe_log10_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn safe_log10_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(LN10);
    let q1 = q0*powm1(x.val);
    let q2 = q0*powm1(pow2(x.val));
    unary.val = q0*safe_log(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(x.d2val1*x.val - pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(-x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q2*(x.d2val2*x.val - pow2(x.d1val2));
    unary
}

/// `log2_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log2_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(log(2.0));
    let q1 = q0*powm1(x.val);
    let q2 = q0*powm1(pow2(x.val));
    unary.val = q0*log(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(x.d2val1*x.val - pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(-x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q2*(x.d2val2*x.val - pow2(x.d1val2));
    unary
}

/// `sin_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sin_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = sin(x.val);
    let q1 = cos(x.val);
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = -q0*pow2(x.d1val1) + q1*x.d2val1;
    unary.d1val1_d1val2 = -q0*x.d1val1*x.d1val2 + q1*x.d1val1_d1val2;
    unary.d2val2 = -q0*pow2(x.d1val2) + q1*x.d2val2;
    unary
}

/// `cos_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn cos_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = cos(x.val);
    let q1 = sin(x.val);
    unary.val = q0;
    unary.d1val1 = -q1*x.d1val1;
    unary.d1val2 = -q1*x.d1val2;
    unary.d2val1 = -q0*pow2(x.d1val1) - q1*x.d2val1;
    unary.d1val1_d1val2 = -q0*x.d1val1*x.d1val2 - q1*x.d1val1_d1val2;
    unary.d2val2 = -q0*pow2(x.d1val2) - q1*x.d2val2;
    unary
}

/// `tan_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn tan_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = tan(x.val);
    let q1 = pow2(q0) + 1.0;
    let q2 = 2.0*q0;
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q1*(q2*pow2(x.d1val1) + x.d2val1);
    unary.d1val1_d1val2 = (q2*x.d1val1*x.d1val2 + x.d1val1_d1val2)*powm1(pow2(cos(x.val)));
    unary.d2val2 = q1*(q2*pow2(x.d1val2) + x.d2val2);
    unary
}

/// `sinpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sinpi_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = PI*x.val;
    let q1 = sin(q0);
    let q2 = cos(q0);
    let q3 = PI*q2;
    let q4 = PI*q1;
    unary.val = q1;
    unary.d1val1 = q3*x.d1val1;
    unary.d1val2 = q3*x.d1val2;
    unary.d2val1 = PI*(q2*x.d2val1 - q4*pow2(x.d1val1));
    unary.d1val1_d1val2 = PI*(q2*x.d1val1_d1val2 - q4*x.d1val1*x.d1val2);
    unary.d2val2 = PI*(q2*x.d2val2 - q4*pow2(x.d1val2));
    unary
}

/// `cospi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn cospi_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = PI*x.val;
    let q1 = cos(q0);
    let q2 = sin(q0);
    let q3 = PI*q2;
    let q4 = PI*q1;
    unary.val = q1;
    unary.d1val1 = -q3*x.d1val1;
    unary.d1val2 = -q3*x.d1val2;
    unary.d2val1 = -PI*(q2*x.d2val1 + q4*pow2(x.d1val1));
    unary.d1val1_d1val2 = -PI*(q2*x.d1val1_d1val2 + q4*x.d1val1*x.d1val2);
    unary.d2val2 = -PI*(q2*x.d2val2 + q4*pow2(x.d1val2));
    unary
}

/// `tanpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn tanpi_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = PI*x.val;
    let q1 = tan(q0);
    let q2 = PI*(pow2(q1) + 1.0);
    let q3 = 2.0*PI*q1;
    unary.val = q1;
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*(q3*pow2(x.d1val1) + x.d2val1);
    unary.d1val1_d1val2 = PI*(q3*x.d1val1*x.d1val2 + x.d1val1_d1val2)*powm1(pow2(cos(q0)));
    unary.d2val2 = q2*(q3*pow2(x.d1val2) + x.d2val2);
    unary
}

/// `sinh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sinh_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = sinh(x.val);
    let q1 = cosh(x.val);
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q0*pow2(x.d1val1) + q1*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1*x.d1val2 + q1*x.d1val1_d1val2;
    unary.d2val2 = q0*pow2(x.d1val2) + q1*x.d2val2;
    unary
}

/// `cosh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn cosh_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = cosh(x.val);
    let q1 = sinh(x.val);
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q0*pow2(x.d1val1) + q1*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1*x.d1val2 + q1*x.d1val1_d1val2;
    unary.d2val2 = q0*pow2(x.d1val2) + q1*x.d2val2;
    unary
}

/// `tanh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn tanh_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = tanh(x.val);
    let q1 = pow2(q0) - 1.0;
    let q2 = 2.0*q0;
    unary.val = q0;
    unary.d1val1 = -q1*x.d1val1;
    unary.d1val2 = -q1*x.d1val2;
    unary.d2val1 = q1*(q2*pow2(x.d1val1) - x.d2val1);
    unary.d1val1_d1val2 = -(q2*x.d1val1*x.d1val2 - x.d1val1_d1val2)*powm1(pow2(cosh(x.val)));
    unary.d2val2 = q1*(q2*pow2(x.d1val2) - x.d2val2);
    unary
}

/// `asin_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn asin_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val);
    let q1 = 1.0 - q0;
    let q2 = powm1(f_sqrt(q1));
    let q3 = powm1(pow3(f_sqrt(q1)));
    let q4 = q0 - 1.0;
    unary.val = asin(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q3*(-q4*x.d2val1 + x.val*pow2(x.d1val1));
    unary.d1val1_d1val2 = q3*(q1*x.d1val1_d1val2 + x.d1val1*x.d1val2*x.val);
    unary.d2val2 = q3*(-q4*x.d2val2 + x.val*pow2(x.d1val2));
    unary
}

/// `acos_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn acos_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val);
    let q1 = 1.0 - q0;
    let q2 = powm1(f_sqrt(q1));
    let q3 = powm1(pow3(f_sqrt(q1)));
    let q4 = q0 - 1.0;
    unary.val = acos(x.val);
    unary.d1val1 = -q2*x.d1val1;
    unary.d1val2 = -q2*x.d1val2;
    unary.d2val1 = q3*(q4*x.d2val1 - x.val*pow2(x.d1val1));
    unary.d1val1_d1val2 = -q3*(-q4*x.d1val1_d1val2 + x.d1val1*x.d1val2*x.val);
    unary.d2val2 = q3*(q4*x.d2val2 - x.val*pow2(x.d1val2));
    unary
}

/// `atan_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn atan_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val) + 1.0;
    let q1 = powm1(q0);
    let q2 = powm1(pow2(q0));
    let q3 = 2.0*x.val;
    unary.val = atan(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(q0*x.d2val1 - q3*pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(q0*x.d1val1_d1val2 - q3*x.d1val1*x.d1val2);
    unary.d2val2 = q2*(q0*x.d2val2 - q3*pow2(x.d1val2));
    unary
}

/// `asinpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn asinpi_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(PI);
    let q1 = pow2(x.val);
    let q2 = 1.0 - q1;
    let q3 = q0*powm1(f_sqrt(q2));
    let q4 = q1 - 1.0;
    let q5 = q0*powm1(pow3(f_sqrt(q2)));
    unary.val = q0*asin(x.val);
    unary.d1val1 = q3*x.d1val1;
    unary.d1val2 = q3*x.d1val2;
    unary.d2val1 = q5*(-q4*x.d2val1 + x.val*pow2(x.d1val1));
    unary.d1val1_d1val2 = q5*(q2*x.d1val1_d1val2 + x.d1val1*x.d1val2*x.val);
    unary.d2val2 = q5*(-q4*x.d2val2 + x.val*pow2(x.d1val2));
    unary
}

/// `acospi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn acospi_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(PI);
    let q1 = pow2(x.val);
    let q2 = 1.0 - q1;
    let q3 = q0*powm1(f_sqrt(q2));
    let q4 = q1 - 1.0;
    let q5 = q0*powm1(pow3(f_sqrt(q2)));
    unary.val = q0*acos(x.val);
    unary.d1val1 = -q3*x.d1val1;
    unary.d1val2 = -q3*x.d1val2;
    unary.d2val1 = q5*(q4*x.d2val1 - x.val*pow2(x.d1val1));
    unary.d1val1_d1val2 = -q5*(-q4*x.d1val1_d1val2 + x.d1val1*x.d1val2*x.val);
    unary.d2val2 = q5*(q4*x.d2val2 - x.val*pow2(x.d1val2));
    unary
}

/// `atanpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn atanpi_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val);
    let q1 = PI*q0;
    let q2 = powm1(PI + q1);
    let q3 = 2.0*x.val;
    let q4 = powm1(2.0*q1 + PI*pow4(x.val) + PI);
    unary.val = powm1(PI)*atan(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q4*(q0*x.d2val1 - q3*pow2(x.d1val1) + x.d2val1);
    unary.d1val1_d1val2 = q4*(q0*x.d1val1_d1val2 - q3*x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q4*(q0*x.d2val2 - q3*pow2(x.d1val2) + x.d2val2);
    unary
}

/// `asinh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn asinh_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val) + 1.0;
    let q1 = powm1(f_sqrt(q0));
    let q2 = powm1(pow3(f_sqrt(q0)));
    unary.val = asinh(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(q0*x.d2val1 - x.val*pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(q0*x.d1val1_d1val2 - x.d1val1*x.d1val2*x.val);
    unary.d2val2 = q2*(q0*x.d2val2 - x.val*pow2(x.d1val2));
    unary
}

/// `acosh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn acosh_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val) - 1.0;
    let q1 = powm1(f_sqrt(q0));
    let q2 = powm1(pow3(f_sqrt(q0)));
    unary.val = acosh(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q2*(q0*x.d2val1 - x.val*pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(q0*x.d1val1_d1val2 - x.d1val1*x.d1val2*x.val);
    unary.d2val2 = q2*(q0*x.d2val2 - x.val*pow2(x.d1val2));
    unary
}

/// `atanh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn atanh_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(x.val) - 1.0;
    let q1 = powm1(q0);
    let q2 = powm1(pow2(q0));
    let q3 = 2.0*x.val;
    unary.val = atanh(x.val);
    unary.d1val1 = -q1*x.d1val1;
    unary.d1val2 = -q1*x.d1val2;
    unary.d2val1 = q2*(-q0*x.d2val1 + q3*pow2(x.d1val1));
    unary.d1val1_d1val2 = q2*(-q0*x.d1val1_d1val2 + q3*x.d1val1*x.d1val2);
    unary.d2val2 = q2*(-q0*x.d2val2 + q3*pow2(x.d1val2));
    unary
}

/// `sqrt_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sqrt_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = f_sqrt(x.val);
    let q1 = 0.5*powm1(q0);
    let q2 = 2.0*x.val;
    let q3 = 0.25*powm1(pow3(f_sqrt(x.val)));
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q3*(q2*x.d2val1 - pow2(x.d1val1));
    unary.d1val1_d1val2 = q3*(q2*x.d1val1_d1val2 - x.d1val1*x.d1val2);
    unary.d2val2 = q3*(q2*x.d2val2 - pow2(x.d1val2));
    unary
}

/// `pow2_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow2_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 2.0*x.val;
    unary.val = pow2(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = 2.0*pow2(x.d1val1) + q0*x.d2val1;
    unary.d1val1_d1val2 = 2.0*x.d1val1*x.d1val2 + q0*x.d1val1_d1val2;
    unary.d2val2 = 2.0*pow2(x.d1val2) + q0*x.d2val2;
    unary
}

/// `pow3_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow3_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 3.0*pow2(x.val);
    let q1 = 3.0*x.val;
    unary.val = pow3(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(2.0*pow2(x.d1val1) + x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(2.0*x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(2.0*pow2(x.d1val2) + x.d2val2*x.val);
    unary
}

/// `pow4_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow4_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 4.0*pow3(x.val);
    let q1 = 4.0*pow2(x.val);
    unary.val = pow4(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(3.0*pow2(x.d1val1) + x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(3.0*x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(3.0*pow2(x.d1val2) + x.d2val2*x.val);
    unary
}

/// `pow5_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow5_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 5.0*pow4(x.val);
    let q1 = 5.0*pow3(x.val);
    unary.val = pow5(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(4.0*pow2(x.d1val1) + x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(4.0*x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(4.0*pow2(x.d1val2) + x.d2val2*x.val);
    unary
}

/// `pow6_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow6_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 6.0*pow5(x.val);
    let q1 = 6.0*pow4(x.val);
    unary.val = pow6(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(5.0*pow2(x.d1val1) + x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(5.0*x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(5.0*pow2(x.d1val2) + x.d2val2*x.val);
    unary
}

/// `pow7_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow7_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 7.0*pow6(x.val);
    let q1 = 7.0*pow5(x.val);
    unary.val = pow7(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(6.0*pow2(x.d1val1) + x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(6.0*x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(6.0*pow2(x.d1val2) + x.d2val2*x.val);
    unary
}

/// `pow8_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow8_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = 8.0*pow7(x.val);
    let q1 = 8.0*pow6(x.val);
    unary.val = pow8(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q1*(7.0*pow2(x.d1val1) + x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(7.0*x.d1val1*x.d1val2 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(7.0*pow2(x.d1val2) + x.d2val2*x.val);
    unary
}

/// `abs_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn abs_self(x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = sgn(x.val);
    unary.val = f_abs(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary
}

/// `add_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn add_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    binary.val = x.val + y.val;
    binary.d1val1 = x.d1val1 + y.d1val1;
    binary.d1val2 = x.d1val2 + y.d1val2;
    binary.d2val1 = x.d2val1 + y.d2val1;
    binary.d1val1_d1val2 = x.d1val1_d1val2 + y.d1val1_d1val2;
    binary.d2val2 = x.d2val2 + y.d2val2;
    binary
}

/// `add_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn add_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = x.val + y;
    unary.d1val1 = x.d1val1;
    unary.d1val2 = x.d1val2;
    unary.d2val1 = x.d2val1;
    unary.d1val1_d1val2 = x.d1val1_d1val2;
    unary.d2val2 = x.d2val2;
    unary
}

/// `add_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn add_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = x.val + z;
    unary.d1val1 = x.d1val1;
    unary.d1val2 = x.d1val2;
    unary.d2val1 = x.d2val1;
    unary.d1val1_d1val2 = x.d1val1_d1val2;
    unary.d2val2 = x.d2val2;
    unary
}

/// `sub_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sub_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    binary.val = x.val - y.val;
    binary.d1val1 = x.d1val1 - y.d1val1;
    binary.d1val2 = x.d1val2 - y.d1val2;
    binary.d2val1 = x.d2val1 - y.d2val1;
    binary.d1val1_d1val2 = x.d1val1_d1val2 - y.d1val1_d1val2;
    binary.d2val2 = x.d2val2 - y.d2val2;
    binary
}

/// `sub_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sub_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = x.val - y;
    unary.d1val1 = x.d1val1;
    unary.d1val2 = x.d1val2;
    unary.d2val1 = x.d2val1;
    unary.d1val1_d1val2 = x.d1val1_d1val2;
    unary.d2val2 = x.d2val2;
    unary
}

/// `sub_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sub_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = -x.val + z;
    unary.d1val1 = -x.d1val1;
    unary.d1val2 = -x.d1val2;
    unary.d2val1 = -x.d2val1;
    unary.d1val1_d1val2 = -x.d1val1_d1val2;
    unary.d2val2 = -x.d2val2;
    unary
}

/// `mul_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn mul_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    binary.val = x.val*y.val;
    binary.d1val1 = x.d1val1*y.val + x.val*y.d1val1;
    binary.d1val2 = x.d1val2*y.val + x.val*y.d1val2;
    binary.d2val1 = 2.0*x.d1val1*y.d1val1 + x.d2val1*y.val + x.val*y.d2val1;
    binary.d1val1_d1val2 = x.d1val1*y.d1val2 + x.d1val1_d1val2*y.val + x.d1val2*y.d1val1 + x.val*y.d1val1_d1val2;
    binary.d2val2 = 2.0*x.d1val2*y.d1val2 + x.d2val2*y.val + x.val*y.d2val2;
    binary
}

/// `mul_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn mul_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = x.val*y;
    unary.d1val1 = x.d1val1*y;
    unary.d1val2 = x.d1val2*y;
    unary.d2val1 = x.d2val1*y;
    unary.d1val1_d1val2 = x.d1val1_d1val2*y;
    unary.d2val2 = x.d2val2*y;
    unary
}

/// `mul_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn mul_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    unary.val = x.val*z;
    unary.d1val1 = x.d1val1*z;
    unary.d1val2 = x.d1val2*z;
    unary.d2val1 = x.d2val1*z;
    unary.d1val1_d1val2 = x.d1val1_d1val2*z;
    unary.d2val2 = x.d2val2*z;
    unary
}

/// `div_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn div_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    let q0 = pow2(y.val);
    let q1 = powm1(q0);
    let q2 = x.d1val1*y.val;
    let q3 = x.val*y.d1val1;
    let q4 = x.d1val2*y.val;
    let q5 = powm1(pow3(y.val));
    let q6 = 2.0*y.d1val2;
    binary.val = x.val*powm1(y.val);
    binary.d1val1 = q1*(q2 - q3);
    binary.d1val2 = q1*(q4 - x.val*y.d1val2);
    binary.d2val1 = q5*(-2.0*q2*y.d1val1 + q0*x.d2val1 + x.val*(2.0*pow2(y.d1val1) - y.d2val1*y.val));
    binary.d1val1_d1val2 = q5*(q0*x.d1val1_d1val2 + q3*q6 - y.val*(x.d1val1*y.d1val2 + x.d1val2*y.d1val1 + x.val*y.d1val1_d1val2));
    binary.d2val2 = q5*(q0*x.d2val2 - q4*q6 + x.val*(2.0*pow2(y.d1val2) - y.d2val2*y.val));
    binary
}

/// `div_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn div_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = powm1(y);
    unary.val = q0*x.val;
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary
}

/// `div_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn div_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = z*powm1(pow2(x.val));
    let q1 = z*powm1(pow3(x.val));
    unary.val = z*powm1(x.val);
    unary.d1val1 = -q0*x.d1val1;
    unary.d1val2 = -q0*x.d1val2;
    unary.d2val1 = q1*(2.0*pow2(x.d1val1) - x.d2val1*x.val);
    unary.d1val1_d1val2 = q1*(2.0*x.d1val1*x.d1val2 - x.d1val1_d1val2*x.val);
    unary.d2val2 = q1*(2.0*pow2(x.d1val2) - x.d2val2*x.val);
    unary
}

/// `pow_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    let q0 = pow(x.val, y.val - 1.0);
    let q1 = x.d1val1*y.val;
    let q2 = log(x.val);
    let q3 = q2*x.val;
    let q4 = q1 + q3*y.d1val1;
    let q5 = q3*y.d1val2 + x.d1val2*y.val;
    let q6 = pow(x.val, -2.0 + y.val);
    let q7 = q2*pow2(x.val);
    binary.val = pow(x.val, y.val);
    binary.d1val1 = q0*q4;
    binary.d1val2 = q0*q5;
    binary.d2val1 = q6*(q7*y.d2val1 + x.val*(2.0*x.d1val1*y.d1val1 + x.d2val1*y.val) - y.val*pow2(x.d1val1) + pow2(q4));
    binary.d1val1_d1val2 = (-q1*x.d1val2 + q4*q5 + q7*y.d1val1_d1val2 + x.val*(x.d1val1*y.d1val2 + x.d1val1_d1val2*y.val  + x.d1val2*y.d1val1))*pow(x.val, 3.0 + y.val)*powm1(pow5(x.val));
    binary.d2val2 = q6*(q7*y.d2val2 + x.val*(2.0*x.d1val2*y.d1val2 + x.d2val2*y.val) - y.val*pow2(x.d1val2) + pow2(q5));
    binary
}

/// `pow_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = y*pow(x.val, y - 1.0);
    let q1 = pow2(x.d1val1);
    let q2 = y*pow(x.val, -2.0 + y);
    let q3 = x.d1val1*x.d1val2;
    let q4 = pow2(x.d1val2);
    unary.val = pow(x.val, y);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*(q1*y - q1 + x.d2val1*x.val);
    unary.d1val1_d1val2 = q2*(q3*y - q3 + x.d1val1_d1val2*x.val);
    unary.d2val2 = q2*(q4*y - q4 + x.d2val2*x.val);
    unary
}

/// `pow_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = pow(z, x.val);
    let q1 = log(z);
    let q2 = q0*q1;
    unary.val = q0;
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*(q1*pow2(x.d1val1) + x.d2val1);
    unary.d1val1_d1val2 = q2*(q1*x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q2*(q1*pow2(x.d1val2) + x.d2val2);
    unary
}

/// `max_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn max_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    let q0 = heaviside(x.val - y.val);
    let q1 = heaviside(-x.val + y.val);
    binary.val = f_max(x.val, y.val);
    binary.d1val1 = q0*x.d1val1 + q1*y.d1val1;
    binary.d1val2 = q0*x.d1val2 + q1*y.d1val2;
    binary.d2val1 = q0*x.d2val1 + q1*y.d2val1;
    binary.d1val1_d1val2 = q0*x.d1val1_d1val2 + q1*y.d1val1_d1val2;
    binary.d2val2 = q0*x.d2val2 + q1*y.d2val2;
    binary
}

/// `max_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn max_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = heaviside(x.val - y);
    unary.val = f_max(x.val, y);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary
}

/// `max_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn max_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = heaviside(x.val - z);
    unary.val = f_max(x.val, z);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary
}

/// `min_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn min_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    let q0 = heaviside(-x.val + y.val);
    let q1 = heaviside(x.val - y.val);
    binary.val = f_min(x.val, y.val);
    binary.d1val1 = q0*x.d1val1 + q1*y.d1val1;
    binary.d1val2 = q0*x.d1val2 + q1*y.d1val2;
    binary.d2val1 = q0*x.d2val1 + q1*y.d2val1;
    binary.d1val1_d1val2 = q0*x.d1val1_d1val2 + q1*y.d1val1_d1val2;
    binary.d2val2 = q0*x.d2val2 + q1*y.d2val2;
    binary
}

/// `min_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn min_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = heaviside(-x.val + y);
    unary.val = f_min(x.val, y);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary
}

/// `min_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn min_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = heaviside(-x.val + z);
    unary.val = f_min(x.val, z);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary
}

/// `dim_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn dim_self(x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut binary = AutoDiffReal2VarOrder2::default();
    let q0 = x.val - y.val;
    let q1 = 0.5*sgn(q0);
    binary.val = -0.5*y.val + 0.5*x.val + 0.5*f_abs(q0);
    binary.d1val1 = -0.5*y.d1val1 + 0.5*x.d1val1 + q1*(x.d1val1 - y.d1val1);
    binary.d1val2 = -0.5*y.d1val2 + 0.5*x.d1val2 + q1*(x.d1val2 - y.d1val2);
    binary.d2val1 = -0.5*y.d2val1 + 0.5*x.d2val1 + q1*(x.d2val1 - y.d2val1);
    binary.d1val1_d1val2 = -0.5*y.d1val1_d1val2 + 0.5*x.d1val1_d1val2 + q1*(x.d1val1_d1val2 - y.d1val1_d1val2);
    binary.d2val2 = -0.5*y.d2val2 + 0.5*x.d2val2 + q1*(x.d2val2 - y.d2val2);
    binary
}

/// `dim_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn dim_self_real(x: AutoDiffReal2VarOrder2, y: f64) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = x.val - y;
    let q1 = sgn(q0);
    let q2 = 0.5*q1 + 0.5;
    let q3 = 0.5*x.d1val1_d1val2;
    unary.val = -0.5*y + 0.5*x.val + 0.5*f_abs(q0);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*x.d2val1;
    unary.d1val1_d1val2 = q1*q3 + q3;
    unary.d2val2 = q2*x.d2val2;
    unary
}

/// `dim_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn dim_real_self(z: f64, x: AutoDiffReal2VarOrder2) -> AutoDiffReal2VarOrder2 {
    let mut unary = AutoDiffReal2VarOrder2::default();
    let q0 = x.val - z;
    let q1 = sgn(q0);
    let q2 = -0.5 + 0.5*q1;
    let q3 = 0.5*x.d1val1_d1val2;
    unary.val = -0.5*x.val + 0.5*z + 0.5*f_abs(q0);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*x.d2val1;
    unary.d1val1_d1val2 = q1*q3 - q3;
    unary.d2val2 = q2*x.d2val2;
    unary
}
