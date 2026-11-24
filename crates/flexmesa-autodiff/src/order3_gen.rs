// SPDX-License-Identifier: GPL-3.0-or-later
//! Generated from MESA `26.4.1` `auto_diff/private/auto_diff_real_2var_order3_module.f90`
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
use crate::AutoDiffReal2VarOrder3;

/// `make_unary_operator` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn make_unary_operator(x: AutoDiffReal2VarOrder3, z_val: f64, z_d1x: f64, z_d2x: f64, z_d3x: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.d1val1);
    let q1 = x.d1val2*z_d2x;
    let q2 = pow2(x.d1val2);
    let q3 = x.d1val1*z_d2x;
    let q4 = 2.0*x.d1val1_d1val2;
    unary.val = z_val;
    unary.d1val1 = x.d1val1*z_d1x;
    unary.d1val2 = x.d1val2*z_d1x;
    unary.d2val1 = q0*z_d2x + x.d2val1*z_d1x;
    unary.d1val1_d1val2 = q1*x.d1val1 + x.d1val1_d1val2*z_d1x;
    unary.d2val2 = q2*z_d2x + x.d2val2*z_d1x;
    unary.d3val1 = 3.0*q3*x.d2val1 + x.d3val1*z_d1x + z_d3x*pow3(x.d1val1);
    unary.d2val1_d1val2 = q0*x.d1val2*z_d3x + q1*x.d2val1 + q3*q4 + x.d2val1_d1val2*z_d1x;
    unary.d1val1_d2val2 = q1*q4 + q2*x.d1val1*z_d3x + q3*x.d2val2 + x.d1val1_d2val2*z_d1x;
    unary.d3val2 = 3.0*q1*x.d2val2 + x.d3val2*z_d1x + z_d3x*pow3(x.d1val2);
    unary
}

/// `make_binary_operator` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn make_binary_operator(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3, z_val: f64, z_d1x: f64, z_d1y: f64, z_d2x: f64, z_d1x_d1y: f64, z_d2y: f64, z_d3x: f64, z_d2x_d1y: f64, z_d1x_d2y: f64, z_d3y: f64) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.d1val1);
    let q1 = pow2(y.d1val1);
    let q2 = x.d1val1*z_d1x_d1y;
    let q3 = 2.0*q2;
    let q4 = x.d1val2*z_d2x;
    let q5 = y.d1val2*z_d1x_d1y;
    let q6 = x.d1val2*z_d1x_d1y;
    let q7 = y.d1val2*z_d2y;
    let q8 = pow2(x.d1val2);
    let q9 = pow2(y.d1val2);
    let q10 = 2.0*q5;
    let q11 = x.d1val1*z_d2x;
    let q12 = 3.0*x.d2val1;
    let q13 = 3.0*y.d2val1;
    let q14 = y.d1val1*z_d1x_d1y;
    let q15 = y.d1val1*z_d2y;
    let q16 = q1*z_d1x_d2y;
    let q17 = q0*z_d2x_d1y;
    let q18 = 2.0*x.d1val1_d1val2;
    let q19 = 2.0*y.d1val1_d1val2;
    let q20 = 2.0*x.d1val1*y.d1val1;
    let q21 = x.d1val2*z_d2x_d1y;
    let q22 = y.d1val2*z_d1x_d2y;
    let q23 = q9*z_d1x_d2y;
    let q24 = q8*z_d2x_d1y;
    let q25 = 3.0*x.d2val2;
    let q26 = 3.0*y.d2val2;
    binary.val = z_val;
    binary.d1val1 = x.d1val1*z_d1x + y.d1val1*z_d1y;
    binary.d1val2 = x.d1val2*z_d1x + y.d1val2*z_d1y;
    binary.d2val1 = q0*z_d2x + q1*z_d2y + q3*y.d1val1 + x.d2val1*z_d1x + y.d2val1*z_d1y;
    binary.d1val1_d1val2 = q4*x.d1val1 + q5*x.d1val1 + q6*y.d1val1 + q7*y.d1val1  + x.d1val1_d1val2*z_d1x + y.d1val1_d1val2*z_d1y;
    binary.d2val2 = q10*x.d1val2 + q8*z_d2x + q9*z_d2y + x.d2val2*z_d1x + y.d2val2*z_d1y;
    binary.d3val1 = 3.0*q16*x.d1val1 + 3.0*q17*y.d1val1 + q11*q12 + q12*q14 + q13*q15  + q13*q2 + x.d3val1*z_d1x + y.d3val1*z_d1y + z_d3x*pow3(x.d1val1)  + z_d3y*pow3(y.d1val1);
    binary.d2val1_d1val2 = q0*x.d1val2*z_d3x + q1*y.d1val2*z_d3y + q11*q18 + q14*q18 + q15*q19  + q16*x.d1val2 + q17*y.d1val2 + q20*q21 + q20*q22 + q3*y.d1val1_d1val2  + q4*x.d2val1 + q5*x.d2val1 + q6*y.d2val1 + q7*y.d2val1  + x.d2val1_d1val2*z_d1x + y.d2val1_d1val2*z_d1y;
    binary.d1val1_d2val2 = 2.0*q21*x.d1val1*y.d1val2 + 2.0*q22*x.d1val2*y.d1val1  + q10*x.d1val1_d1val2 + q11*x.d2val2 + q14*x.d2val2 + q15*y.d2val2  + q18*q4 + q19*q6 + q19*q7 + q2*y.d2val2 + q23*x.d1val1  + q24*y.d1val1 + q8*x.d1val1*z_d3x + q9*y.d1val1*z_d3y  + x.d1val1_d2val2*z_d1x + y.d1val1_d2val2*z_d1y;
    binary.d3val2 = 3.0*q23*x.d1val2 + 3.0*q24*y.d1val2 + q25*q4 + q25*q5 + q26*q6  + q26*q7 + x.d3val2*z_d1x + y.d3val2*z_d1y + z_d3x*pow3(x.d1val2)  + z_d3y*pow3(y.d1val2);
    binary
}

/// `sign_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sign_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = sgn(x.val);
    unary.d1val1 = 0.0;
    unary.d1val2 = 0.0;
    unary.d2val1 = 0.0;
    unary.d1val1_d1val2 = 0.0;
    unary.d2val2 = 0.0;
    unary.d3val1 = 0.0;
    unary.d2val1_d1val2 = 0.0;
    unary.d1val1_d2val2 = 0.0;
    unary.d3val2 = 0.0;
    unary
}

/// `safe_sqrt_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn safe_sqrt_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(x.val);
    let q1 = q0*x.val;
    let q2 = f_sqrt(q1);
    let q3 = 0.5*q2*powm1(x.val);
    let q4 = 2.0*x.val;
    let q5 = q4*x.d2val1;
    let q6 = pow2(x.d1val1);
    let q7 = pow2(x.val);
    let q8 = 0.25*q2*powm1(q7);
    let q9 = q4*x.d2val2;
    let q10 = pow2(x.d1val2);
    let q11 = x.d1val1*x.val;
    let q12 = 4.0*q7;
    let q13 = 0.125*q2*powm1(pow3(x.val));
    let q14 = 4.0*x.d1val1_d1val2;
    let q15 = x.d1val2*x.val;
    unary.val = q2;
    unary.d1val1 = q3*x.d1val1;
    unary.d1val2 = q3*x.d1val2;
    unary.d2val1 = q8*(q5 - q6);
    unary.d1val1_d1val2 = q8*(q4*x.d1val1_d1val2 - x.d1val1*x.d1val2);
    unary.d2val2 = q8*(-q10 + q9);
    unary.d3val1 = q13*(-6.0*q11*x.d2val1 + 3.0*pow3(x.d1val1) + q12*x.d3val1);
    unary.d2val1_d1val2 = 0.125*(3.0*q6*x.d1val2 - q11*q14 + q12*x.d2val1_d1val2 - q5*x.d1val2)  * pow3(f_sqrt(q1))*powm1(q0)*powm1(pow4(x.val));
    unary.d1val1_d2val2 = q13*(3.0*q10*x.d1val1 + q12*x.d1val1_d2val2 - q14*q15 - q9*x.d1val1);
    unary.d3val2 = q13*(-6.0*q15*x.d2val2 + 3.0*pow3(x.d1val2) + q12*x.d3val2);
    unary
}

/// `unary_minus_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn unary_minus_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = -x.val;
    unary.d1val1 = -x.d1val1;
    unary.d1val2 = -x.d1val2;
    unary.d2val1 = -x.d2val1;
    unary.d1val1_d1val2 = -x.d1val1_d1val2;
    unary.d2val2 = -x.d2val2;
    unary.d3val1 = -x.d3val1;
    unary.d2val1_d1val2 = -x.d2val1_d1val2;
    unary.d1val1_d2val2 = -x.d1val1_d2val2;
    unary.d3val2 = -x.d3val2;
    unary
}

/// `exp_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn exp_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = exp(x.val);
    let q1 = x.d2val1 + pow2(x.d1val1);
    let q2 = x.d2val2 + pow2(x.d1val2);
    let q3 = 2.0*x.d1val1_d1val2;
    unary.val = q0;
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*q1;
    unary.d1val1_d1val2 = q0*(x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q0*q2;
    unary.d3val1 = q0*(3.0*x.d1val1*x.d2val1 + x.d3val1 + pow3(x.d1val1));
    unary.d2val1_d1val2 = q0*(q1*x.d1val2 + q3*x.d1val1 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q0*(q2*x.d1val1 + q3*x.d1val2 + x.d1val1_d2val2);
    unary.d3val2 = q0*(3.0*x.d1val2*x.d2val2 + x.d3val2 + pow3(x.d1val2));
    unary
}

/// `expm1_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn expm1_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = exp(x.val);
    let q1 = x.d2val1 + pow2(x.d1val1);
    let q2 = x.d2val2 + pow2(x.d1val2);
    let q3 = 2.0*x.d1val1_d1val2;
    unary.val = expm1(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*q1;
    unary.d1val1_d1val2 = q0*(x.d1val1*x.d1val2 + x.d1val1_d1val2);
    unary.d2val2 = q0*q2;
    unary.d3val1 = q0*(3.0*x.d1val1*x.d2val1 + x.d3val1 + pow3(x.d1val1));
    unary.d2val1_d1val2 = q0*(q1*x.d1val2 + q3*x.d1val1 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q0*(q2*x.d1val1 + q3*x.d1val2 + x.d1val1_d2val2);
    unary.d3val2 = q0*(3.0*x.d1val2*x.d2val2 + x.d3val2 + pow3(x.d1val2));
    unary
}

/// `exp10_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn exp10_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow(10.0, x.val);
    let q1 = LN10;
    let q2 = q0*q1;
    let q3 = q1*pow2(x.d1val1) + x.d2val1;
    let q4 = q1*x.d1val2;
    let q5 = q1*pow2(x.d1val2) + x.d2val2;
    let q6 = q1*x.d1val1;
    let q7 = pow2(q1);
    let q8 = 2.0*x.d1val1_d1val2;
    unary.val = q0;
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q2*(q4*x.d1val1 + x.d1val1_d1val2);
    unary.d2val2 = q2*q5;
    unary.d3val1 = q2*(3.0*q6*x.d2val1 + q7*pow3(x.d1val1) + x.d3val1);
    unary.d2val1_d1val2 = q2*(q3*q4 + q6*q8 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q2*(q4*q8 + q5*q6 + x.d1val1_d2val2);
    unary.d3val2 = q2*(3.0*q4*x.d2val2 + q7*pow3(x.d1val2) + x.d3val2);
    unary
}

/// `powm1_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn powm1_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = powm1(q0);
    let q2 = powm1(pow3(x.val));
    let q3 = x.d2val1*x.val;
    let q4 = pow2(x.d1val1);
    let q5 = 2.0*x.d1val1;
    let q6 = x.d1val1_d1val2*x.val;
    let q7 = x.d2val2*x.val;
    let q8 = -q7;
    let q9 = pow2(x.d1val2);
    let q10 = powm1(pow4(x.val));
    let q11 = 4.0*q6;
    let q12 = 6.0*x.d1val2;
    unary.val = powm1(x.val);
    unary.d1val1 = -q1*x.d1val1;
    unary.d1val2 = -q1*x.d1val2;
    unary.d2val1 = q2*(2.0*q4 - q3);
    unary.d1val1_d1val2 = q2*(q5*x.d1val2 - q6);
    unary.d2val2 = q2*(2.0*q9 + q8);
    unary.d3val1 = q10*(-6.0*pow3(x.d1val1) + 6.0*q3*x.d1val1 - q0*x.d3val1);
    unary.d2val1_d1val2 = q10*(2.0*q3*x.d1val2 - q0*x.d2val1_d1val2 + q11*x.d1val1 - q12*q4);
    unary.d1val1_d2val2 = q10*(-q0*x.d1val1_d2val2 + q11*x.d1val2 - q5*(3.0*q9 + q8));
    unary.d3val2 = q10*(-6.0*pow3(x.d1val2) - q0*x.d3val2 + q12*q7);
    unary
}

/// `log_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(x.val);
    let q1 = pow2(x.val);
    let q2 = powm1(q1);
    let q3 = x.d2val1*x.val;
    let q4 = pow2(x.d1val1);
    let q5 = x.d1val1_d1val2*x.val;
    let q6 = x.d2val2*x.val;
    let q7 = pow2(x.d1val2);
    let q8 = powm1(pow3(x.val));
    let q9 = 2.0*x.d1val2;
    unary.val = log(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*(q3 - q4);
    unary.d1val1_d1val2 = q2*(q5 - x.d1val1*x.d1val2);
    unary.d2val2 = q2*(q6 - q7);
    unary.d3val1 = q8*(-3.0*q3*x.d1val1 + 2.0*pow3(x.d1val1) + q1*x.d3val1);
    unary.d2val1_d1val2 = q8*(-2.0*q5*x.d1val1 + q1*x.d2val1_d1val2 - q3*x.d1val2 + q4*q9);
    unary.d1val1_d2val2 = q8*(q1*x.d1val1_d2val2 - q5*q9 + x.d1val1*(2.0*q7 - q6));
    unary.d3val2 = q8*(-3.0*q6*x.d1val2 + 2.0*pow3(x.d1val2) + q1*x.d3val2);
    unary
}

/// `log1p_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log1p_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = x.val + 1.0;
    let q1 = powm1(q0);
    let q2 = pow2(q0);
    let q3 = powm1(q2);
    let q4 = pow2(x.d1val1);
    let q5 = q0*x.d2val1;
    let q6 = q0*x.d1val1_d1val2;
    let q7 = pow2(x.d1val2);
    let q8 = q0*x.d2val2;
    let q9 = powm1(pow3(q0));
    let q10 = 2.0*q6;
    unary.val = log1p(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q3*(-q4 + q5);
    unary.d1val1_d1val2 = q3*(q6 - x.d1val1*x.d1val2);
    unary.d2val2 = q3*(-q7 + q8);
    unary.d3val1 = q9*(-3.0*q5*x.d1val1 + 2.0*pow3(x.d1val1) + q2*x.d3val1);
    unary.d2val1_d1val2 = q9*(-q10*x.d1val1 + q2*x.d2val1_d1val2 + q4*x.d1val2 + x.d1val2*(q4 - q5));
    unary.d1val1_d2val2 = q9*(-q10*x.d1val2 + q2*x.d1val1_d2val2 + x.d1val1*(2.0*q7 - q8));
    unary.d3val2 = q9*(-3.0*q8*x.d1val2 + 2.0*pow3(x.d1val2) + q2*x.d3val2);
    unary
}

/// `safe_log_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn safe_log_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(x.val);
    let q1 = pow2(x.val);
    let q2 = powm1(q1);
    let q3 = x.d2val1*x.val;
    let q4 = pow2(x.d1val1);
    let q5 = x.d1val1_d1val2*x.val;
    let q6 = x.d2val2*x.val;
    let q7 = pow2(x.d1val2);
    let q8 = powm1(pow3(x.val));
    let q9 = 2.0*x.d1val2;
    unary.val = safe_log(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*(q3 - q4);
    unary.d1val1_d1val2 = q2*(q5 - x.d1val1*x.d1val2);
    unary.d2val2 = q2*(q6 - q7);
    unary.d3val1 = q8*(-3.0*q3*x.d1val1 + 2.0*pow3(x.d1val1) + q1*x.d3val1);
    unary.d2val1_d1val2 = q8*(-2.0*q5*x.d1val1 + q1*x.d2val1_d1val2 - q3*x.d1val2 + q4*q9);
    unary.d1val1_d2val2 = q8*(q1*x.d1val1_d2val2 - q5*q9 + x.d1val1*(2.0*q7 - q6));
    unary.d3val2 = q8*(-3.0*q6*x.d1val2 + 2.0*pow3(x.d1val2) + q1*x.d3val2);
    unary
}

/// `log10_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log10_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(LN10);
    let q1 = q0*powm1(x.val);
    let q2 = x.d2val1*x.val;
    let q3 = pow2(x.d1val1);
    let q4 = pow2(x.val);
    let q5 = q0*powm1(q4);
    let q6 = x.d1val1_d1val2*x.val;
    let q7 = x.d2val2*x.val;
    let q8 = pow2(x.d1val2);
    let q9 = q0*powm1(pow3(x.val));
    let q10 = 2.0*x.d1val2;
    unary.val = q0*log(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q5*(q2 - q3);
    unary.d1val1_d1val2 = q5*(q6 - x.d1val1*x.d1val2);
    unary.d2val2 = q5*(q7 - q8);
    unary.d3val1 = q9*(-3.0*q2*x.d1val1 + 2.0*pow3(x.d1val1) + q4*x.d3val1);
    unary.d2val1_d1val2 = q9*(-2.0*q6*x.d1val1 + q10*q3 - q2*x.d1val2 + q4*x.d2val1_d1val2);
    unary.d1val1_d2val2 = q9*(-q10*q6 + q4*x.d1val1_d2val2 + x.d1val1*(2.0*q8 - q7));
    unary.d3val2 = q9*(-3.0*q7*x.d1val2 + 2.0*pow3(x.d1val2) + q4*x.d3val2);
    unary
}

/// `safe_log10_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn safe_log10_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(LN10);
    let q1 = q0*powm1(x.val);
    let q2 = x.d2val1*x.val;
    let q3 = pow2(x.d1val1);
    let q4 = pow2(x.val);
    let q5 = q0*powm1(q4);
    let q6 = x.d1val1_d1val2*x.val;
    let q7 = x.d2val2*x.val;
    let q8 = pow2(x.d1val2);
    let q9 = q0*powm1(pow3(x.val));
    let q10 = 2.0*x.d1val2;
    unary.val = q0*safe_log(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q5*(q2 - q3);
    unary.d1val1_d1val2 = q5*(q6 - x.d1val1*x.d1val2);
    unary.d2val2 = q5*(q7 - q8);
    unary.d3val1 = q9*(-3.0*q2*x.d1val1 + 2.0*pow3(x.d1val1) + q4*x.d3val1);
    unary.d2val1_d1val2 = q9*(-2.0*q6*x.d1val1 + q10*q3 - q2*x.d1val2 + q4*x.d2val1_d1val2);
    unary.d1val1_d2val2 = q9*(-q10*q6 + q4*x.d1val1_d2val2 + x.d1val1*(2.0*q8 - q7));
    unary.d3val2 = q9*(-3.0*q7*x.d1val2 + 2.0*pow3(x.d1val2) + q4*x.d3val2);
    unary
}

/// `log2_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn log2_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(log(2.0));
    let q1 = q0*powm1(x.val);
    let q2 = x.d2val1*x.val;
    let q3 = pow2(x.d1val1);
    let q4 = pow2(x.val);
    let q5 = q0*powm1(q4);
    let q6 = x.d1val1_d1val2*x.val;
    let q7 = x.d2val2*x.val;
    let q8 = pow2(x.d1val2);
    let q9 = q0*powm1(pow3(x.val));
    let q10 = 2.0*x.d1val2;
    unary.val = q0*log(x.val);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q5*(q2 - q3);
    unary.d1val1_d1val2 = q5*(q6 - x.d1val1*x.d1val2);
    unary.d2val2 = q5*(q7 - q8);
    unary.d3val1 = q9*(-3.0*q2*x.d1val1 + 2.0*pow3(x.d1val1) + q4*x.d3val1);
    unary.d2val1_d1val2 = q9*(-2.0*q6*x.d1val1 + q10*q3 - q2*x.d1val2 + q4*x.d2val1_d1val2);
    unary.d1val1_d2val2 = q9*(-q10*q6 + q4*x.d1val1_d2val2 + x.d1val1*(2.0*q8 - q7));
    unary.d3val2 = q9*(-3.0*q7*x.d1val2 + 2.0*pow3(x.d1val2) + q4*x.d3val2);
    unary
}

/// `sin_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sin_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = sin(x.val);
    let q1 = cos(x.val);
    let q2 = q1*x.d1val2;
    let q3 = pow2(x.d1val1);
    let q4 = q0*x.d1val2;
    let q5 = pow2(x.d1val2);
    let q6 = q0*x.d1val1;
    let q7 = 2.0*x.d1val1_d1val2;
    let q8 = q0*x.d2val2;
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q2;
    unary.d2val1 = -q0*q3 + q1*x.d2val1;
    unary.d1val1_d1val2 = q1*x.d1val1_d1val2 - q4*x.d1val1;
    unary.d2val2 = -q0*q5 + q1*x.d2val2;
    unary.d3val1 = -3.0*q6*x.d2val1 + q1*x.d3val1 - q1*pow3(x.d1val1);
    unary.d2val1_d1val2 = q1*x.d2val1_d1val2 - q2*q3 - q4*x.d2val1 - q6*q7;
    unary.d1val1_d2val2 = q1*x.d1val1_d2val2 - q4*q7 - x.d1val1*(q1*q5 + q8);
    unary.d3val2 = -3.0*q8*x.d1val2 + q1*x.d3val2 - q1*pow3(x.d1val2);
    unary
}

/// `cos_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn cos_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = cos(x.val);
    let q1 = sin(x.val);
    let q2 = q1*x.d1val2;
    let q3 = pow2(x.d1val1);
    let q4 = q0*x.d1val2;
    let q5 = pow2(x.d1val2);
    let q6 = q0*x.d1val1;
    let q7 = 2.0*x.d1val1_d1val2;
    let q8 = q0*x.d2val2;
    unary.val = q0;
    unary.d1val1 = -q1*x.d1val1;
    unary.d1val2 = -q2;
    unary.d2val1 = -q0*q3 - q1*x.d2val1;
    unary.d1val1_d1val2 = -q1*x.d1val1_d1val2 - q4*x.d1val1;
    unary.d2val2 = -q0*q5 - q1*x.d2val2;
    unary.d3val1 = -3.0*q6*x.d2val1 - q1*x.d3val1 + q1*pow3(x.d1val1);
    unary.d2val1_d1val2 = -q1*x.d2val1_d1val2 + q2*q3 - q4*x.d2val1 - q6*q7;
    unary.d1val1_d2val2 = -q1*x.d1val1_d2val2 - q4*q7 + x.d1val1*(q1*q5 - q8);
    unary.d3val2 = -3.0*q8*x.d1val2 - q1*x.d3val2 + q1*pow3(x.d1val2);
    unary
}

/// `tan_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn tan_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = tan(x.val);
    let q1 = pow2(q0);
    let q2 = q1 + 1.0;
    let q3 = powm1(pow2(cos(x.val)));
    let q4 = pow2(x.d1val1);
    let q5 = 2.0*q0;
    let q6 = q4*q5 + x.d2val1;
    let q7 = q5*x.d1val2;
    let q8 = pow2(x.d1val2);
    let q9 = q0*x.d1val1;
    let q10 = pow3(x.d1val1);
    let q11 = 2.0*q3;
    let q12 = 4.0*q1;
    let q13 = 4.0*x.d1val1_d1val2;
    let q14 = q0*x.d2val2;
    let q15 = pow3(x.d1val2);
    unary.val = q0;
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q3*q6;
    unary.d1val1_d1val2 = q3*(q7*x.d1val1 + x.d1val1_d1val2);
    unary.d2val2 = q3*(q5*q8 + x.d2val2);
    unary.d3val1 = q3*(6.0*q9*x.d2val1 + q10*q11 + q10*q12 + x.d3val1);
    unary.d2val1_d1val2 = q3*(q11*q4*x.d1val2 + q13*q9 + q6*q7 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q3*(2.0*x.d1val1*(2.0*q1*q8 + q14 + q3*q8) + q0*q13*x.d1val2 + x.d1val1_d2val2);
    unary.d3val2 = q3*(6.0*q14*x.d1val2 + q11*q15 + q12*q15 + x.d3val2);
    unary
}

/// `sinpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sinpi_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = PI*x.val;
    let q1 = sin(q0);
    let q2 = cos(q0);
    let q3 = PI*q2;
    let q4 = pow2(x.d1val1);
    let q5 = PI*q1;
    let q6 = q5*x.d1val2;
    let q7 = pow2(x.d1val2);
    let q8 = q5*x.d1val1;
    let q9 = q2*pow2(PI);
    let q10 = 2.0*x.d1val1_d1val2;
    let q11 = q1*x.d2val2;
    unary.val = q1;
    unary.d1val1 = q3*x.d1val1;
    unary.d1val2 = q3*x.d1val2;
    unary.d2val1 = PI*(q2*x.d2val1 - q4*q5);
    unary.d1val1_d1val2 = PI*(q2*x.d1val1_d1val2 - q6*x.d1val1);
    unary.d2val2 = PI*(q2*x.d2val2 - q5*q7);
    unary.d3val1 = PI*(-3.0*q8*x.d2val1 + q2*x.d3val1 - q9*pow3(x.d1val1));
    unary.d2val1_d1val2 = PI*(-q10*q8 + q2*x.d2val1_d1val2 - q4*q9*x.d1val2 - q6*x.d2val1);
    unary.d1val1_d2val2 = -PI*(PI*x.d1val1*(q11 + q3*q7) + q10*q6 - q2*x.d1val1_d2val2);
    unary.d3val2 = PI*(-3.0*PI*q11*x.d1val2 + q2*x.d3val2 - q9*pow3(x.d1val2));
    unary
}

/// `cospi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn cospi_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = PI*x.val;
    let q1 = cos(q0);
    let q2 = sin(q0);
    let q3 = PI*q2;
    let q4 = pow2(x.d1val1);
    let q5 = PI*q1;
    let q6 = q5*x.d1val2;
    let q7 = pow2(x.d1val2);
    let q8 = q5*x.d1val1;
    let q9 = q2*pow2(PI);
    let q10 = 2.0*x.d1val1_d1val2;
    let q11 = q1*x.d2val2;
    unary.val = q1;
    unary.d1val1 = -q3*x.d1val1;
    unary.d1val2 = -q3*x.d1val2;
    unary.d2val1 = -PI*(q2*x.d2val1 + q4*q5);
    unary.d1val1_d1val2 = -PI*(q2*x.d1val1_d1val2 + q6*x.d1val1);
    unary.d2val2 = -PI*(q2*x.d2val2 + q5*q7);
    unary.d3val1 = PI*(-3.0*q8*x.d2val1 - q2*x.d3val1 + q9*pow3(x.d1val1));
    unary.d2val1_d1val2 = PI*(-q10*q8 - q2*x.d2val1_d1val2 + q4*q9*x.d1val2 - q6*x.d2val1);
    unary.d1val1_d2val2 = -PI*(-PI*x.d1val1*(-q11 + q3*q7) + q10*q6 + q2*x.d1val1_d2val2);
    unary.d3val2 = PI*(-3.0*PI*q11*x.d1val2 - q2*x.d3val2 + q9*pow3(x.d1val2));
    unary
}

/// `tanpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn tanpi_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = PI*x.val;
    let q1 = tan(q0);
    let q2 = pow2(q1);
    let q3 = q2 + 1.0;
    let q4 = PI*q3;
    let q5 = PI*q1;
    let q6 = 2.0*pow2(x.d1val1);
    let q7 = q5*q6 + x.d2val1;
    let q8 = 2.0*q5;
    let q9 = q8*x.d1val2;
    let q10 = powm1(pow2(cos(q0)));
    let q11 = PI*q10;
    let q12 = pow2(x.d1val2);
    let q13 = 6.0*q5;
    let q14 = pow2(PI);
    let q15 = q14*pow3(x.d1val1);
    let q16 = 4.0*q2;
    let q17 = 2.0*q3;
    let q18 = PI*x.d1val1;
    let q19 = 4.0*q1;
    let q20 = q19*x.d1val1_d1val2;
    let q21 = q14*pow3(x.d1val2);
    unary.val = q1;
    unary.d1val1 = q4*x.d1val1;
    unary.d1val2 = q4*x.d1val2;
    unary.d2val1 = q4*q7;
    unary.d1val1_d1val2 = q11*(q9*x.d1val1 + x.d1val1_d1val2);
    unary.d2val2 = q4*(q12*q8 + x.d2val2);
    unary.d3val1 = q4*(q13*x.d1val1*x.d2val1 + q15*q16 + q15*q17 + x.d3val1);
    unary.d2val1_d1val2 = q11*(q10*q14*q6*x.d1val2 + q18*q20 + q7*q9 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q4*(0.5*q18*(4.0*q12*q4 + 8.0*PI*q12*q2 + q19*x.d2val2) + PI*q20*x.d1val2 + x.d1val1_d2val2);
    unary.d3val2 = q4*(q13*x.d1val2*x.d2val2 + q16*q21 + q17*q21 + x.d3val2);
    unary
}

/// `sinh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sinh_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = sinh(x.val);
    let q1 = cosh(x.val);
    let q2 = q1*x.d1val2;
    let q3 = pow2(x.d1val1);
    let q4 = q0*x.d1val2;
    let q5 = pow2(x.d1val2);
    let q6 = q0*x.d1val1;
    let q7 = 2.0*x.d1val1_d1val2;
    let q8 = q0*x.d2val2;
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q2;
    unary.d2val1 = q0*q3 + q1*x.d2val1;
    unary.d1val1_d1val2 = q1*x.d1val1_d1val2 + q4*x.d1val1;
    unary.d2val2 = q0*q5 + q1*x.d2val2;
    unary.d3val1 = 3.0*q6*x.d2val1 + q1*x.d3val1 + q1*pow3(x.d1val1);
    unary.d2val1_d1val2 = q1*x.d2val1_d1val2 + q2*q3 + q4*x.d2val1 + q6*q7;
    unary.d1val1_d2val2 = q1*x.d1val1_d2val2 + q4*q7 + x.d1val1*(q1*q5 + q8);
    unary.d3val2 = 3.0*q8*x.d1val2 + q1*x.d3val2 + q1*pow3(x.d1val2);
    unary
}

/// `cosh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn cosh_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = cosh(x.val);
    let q1 = sinh(x.val);
    let q2 = q1*x.d1val2;
    let q3 = pow2(x.d1val1);
    let q4 = q0*x.d1val2;
    let q5 = pow2(x.d1val2);
    let q6 = q0*x.d1val1;
    let q7 = 2.0*x.d1val1_d1val2;
    let q8 = q0*x.d2val2;
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q2;
    unary.d2val1 = q0*q3 + q1*x.d2val1;
    unary.d1val1_d1val2 = q1*x.d1val1_d1val2 + q4*x.d1val1;
    unary.d2val2 = q0*q5 + q1*x.d2val2;
    unary.d3val1 = 3.0*q6*x.d2val1 + q1*x.d3val1 + q1*pow3(x.d1val1);
    unary.d2val1_d1val2 = q1*x.d2val1_d1val2 + q2*q3 + q4*x.d2val1 + q6*q7;
    unary.d1val1_d2val2 = q1*x.d1val1_d2val2 + q4*q7 + x.d1val1*(q1*q5 + q8);
    unary.d3val2 = 3.0*q8*x.d1val2 + q1*x.d3val2 + q1*pow3(x.d1val2);
    unary
}

/// `tanh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn tanh_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = tanh(x.val);
    let q1 = pow2(q0);
    let q2 = q1 - 1.0;
    let q3 = powm1(pow2(cosh(x.val)));
    let q4 = pow2(x.d1val1);
    let q5 = 2.0*q0;
    let q6 = q4*q5 - x.d2val1;
    let q7 = q5*x.d1val2;
    let q8 = pow2(x.d1val2);
    let q9 = pow3(x.d1val1);
    let q10 = q0*x.d1val1;
    let q11 = 6.0*q1;
    let q12 = 4.0*x.d1val1_d1val2;
    let q13 = q8*x.d1val1;
    let q14 = q0*x.d1val2;
    let q15 = pow3(x.d1val2);
    unary.val = q0;
    unary.d1val1 = -q2*x.d1val1;
    unary.d1val2 = -q2*x.d1val2;
    unary.d2val1 = -q3*q6;
    unary.d1val1_d1val2 = -q3*(q7*x.d1val1 - x.d1val1_d1val2);
    unary.d2val2 = -q3*(q5*q8 - x.d2val2);
    unary.d3val1 = q3*(-2.0*q9 - 6.0*q10*x.d2val1 + q11*q9 + x.d3val1);
    unary.d2val1_d1val2 = -q3*(2.0*q3*q4*x.d1val2 + q10*q12 - q6*q7 - x.d2val1_d1val2);
    unary.d1val1_d2val2 = -q3*(2.0*q13 - q11*q13 + q12*q14 + q5*x.d1val1*x.d2val2 - x.d1val1_d2val2);
    unary.d3val2 = q3*(-2.0*q15 - 6.0*q14*x.d2val2 + q11*q15 + x.d3val2);
    unary
}

/// `asin_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn asin_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = 1.0 - q0;
    let q2 = powm1(f_sqrt(q1));
    let q3 = powm1(pow3(f_sqrt(q1)));
    let q4 = pow2(x.d1val1);
    let q5 = q0 - 1.0;
    let q6 = q4*x.val - q5*x.d2val1;
    let q7 = x.d1val1*x.d1val2;
    let q8 = pow2(x.d1val2);
    let q9 = powm1(pow5(f_sqrt(q1)));
    let q10 = 3.0*q0;
    let q11 = pow2(q5);
    let q12 = q5*x.d1val1;
    let q13 = 2.0*x.d1val1_d1val2*x.val;
    let q14 = q5*x.d1val2;
    let q15 = x.d2val2*x.val;
    unary.val = asin(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q3*q6;
    unary.d1val1_d1val2 = q3*(q1*x.d1val1_d1val2 + q7*x.val);
    unary.d2val2 = q3*(-q5*x.d2val2 + q8*x.val);
    unary.d3val1 = q9*(q10*pow3(x.d1val1) + q11*x.d3val1 - q12*(3.0*x.d2val1*x.val + q4));
    unary.d2val1_d1val2 = (q1*q6*x.d1val2*x.val  + q5*(-2.0*q0*q4*x.d1val2 - q11*x.d2val1_d1val2 + q12*(q13 + q7)))*powm1(pow7(f_sqrt(q1)));
    unary.d1val1_d2val2 = q9*(q11*x.d1val1_d2val2 - q13*q14 - x.d1val1*(-q10*q8 + q5*(q15 + q8)));
    unary.d3val2 = q9*(q10*pow3(x.d1val2) + q11*x.d3val2 - q14*(3.0*q15 + q8));
    unary
}

/// `acos_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn acos_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = 1.0 - q0;
    let q2 = powm1(f_sqrt(q1));
    let q3 = powm1(pow3(f_sqrt(q1)));
    let q4 = pow2(x.d1val1);
    let q5 = q4*x.val;
    let q6 = q0 - 1.0;
    let q7 = q6*x.d2val1;
    let q8 = x.d1val1*x.d1val2;
    let q9 = q6*x.d1val1_d1val2;
    let q10 = pow2(x.d1val2);
    let q11 = powm1(pow7(f_sqrt(q1)));
    let q12 = pow3(q6);
    let q13 = 3.0*q0;
    let q14 = q1*q13;
    let q15 = pow2(q6);
    let q16 = 2.0*x.val;
    let q17 = x.d2val2*x.val;
    unary.val = acos(x.val);
    unary.d1val1 = -q2*x.d1val1;
    unary.d1val2 = -q2*x.d1val2;
    unary.d2val1 = q3*(-q5 + q7);
    unary.d1val1_d1val2 = -q3*(q8*x.val - q9);
    unary.d2val2 = q3*(-q10*x.val + q6*x.d2val2);
    unary.d3val1 = q11*(q12*x.d3val1 - q14*pow3(x.d1val1) - q15*x.d1val1*(3.0*x.d2val1*x.val + q4));
    unary.d2val1_d1val2 = q11*(-q1*x.d1val2*x.val*(q5 - q7)  + q6*(2.0*q0*q4*x.d1val2 + q15*x.d2val1_d1val2 - q6*x.d1val1*(q16*x.d1val1_d1val2 + q8)));
    unary.d1val1_d2val2 = (-q15*x.d1val1_d2val2 + q16*q9*x.d1val2 + x.d1val1*(-q10*q13 + q6*(q10 + q17)))*powm1(pow5(f_sqrt(q1)));
    unary.d3val2 = q11*(q12*x.d3val2 - q14*pow3(x.d1val2) - q15*x.d1val2*(3.0*q17 + q10));
    unary
}

/// `atan_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn atan_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = q0 + 1.0;
    let q2 = powm1(q1);
    let q3 = pow2(q1);
    let q4 = powm1(q3);
    let q5 = pow2(x.d1val1);
    let q6 = 2.0*x.val;
    let q7 = q5*q6;
    let q8 = q1*x.d2val1;
    let q9 = x.d1val1*x.d1val2;
    let q10 = q1*x.d1val1_d1val2;
    let q11 = pow2(x.d1val2);
    let q12 = powm1(pow3(q1));
    let q13 = 8.0*q0;
    let q14 = 2.0*x.d1val1;
    let q15 = q1*q14;
    let q16 = 4.0*q0;
    let q17 = x.d2val2*x.val;
    unary.val = atan(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q4*(-q7 + q8);
    unary.d1val1_d1val2 = q4*(q10 - q6*q9);
    unary.d2val2 = q4*(q1*x.d2val2 - q11*q6);
    unary.d3val1 = q12*(q13*pow3(x.d1val1) - q15*(3.0*x.d2val1*x.val + q5) + q3*x.d3val1);
    unary.d2val1_d1val2 = q12*(-q15*(q6*x.d1val1_d1val2 + q9) + q16*q5*x.d1val2 + q3*x.d2val1_d1val2 + q6*x.d1val2*(q7 - q8));
    unary.d1val1_d2val2 = q12*(-4.0*q10*x.d1val2*x.val - q14*(q1*(q11 + q17) - q11*q16) + q3*x.d1val1_d2val2);
    unary.d3val2 = q12*(-2.0*q1*x.d1val2*(3.0*q17 + q11) + q13*pow3(x.d1val2) + q3*x.d3val2);
    unary
}

/// `asinpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn asinpi_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(PI);
    let q1 = pow2(x.val);
    let q2 = 1.0 - q1;
    let q3 = q0*powm1(f_sqrt(q2));
    let q4 = pow2(x.d1val1);
    let q5 = q1 - 1.0;
    let q6 = q0*powm1(pow3(f_sqrt(q2)));
    let q7 = x.d1val2*x.val;
    let q8 = pow2(x.d1val2);
    let q9 = 3.0*q1;
    let q10 = pow2(q5);
    let q11 = q0*powm1(pow5(f_sqrt(q2)));
    let q12 = pow4(x.val);
    let q13 = 2.0*q1;
    let q14 = q4*x.d1val2;
    let q15 = 2.0*x.d1val1_d1val2;
    let q16 = q15*x.d1val1;
    let q17 = pow3(x.val);
    let q18 = x.d2val2*x.val;
    unary.val = q0*asin(x.val);
    unary.d1val1 = q3*x.d1val1;
    unary.d1val2 = q3*x.d1val2;
    unary.d2val1 = q6*(q4*x.val - q5*x.d2val1);
    unary.d1val1_d1val2 = q6*(q2*x.d1val1_d1val2 + q7*x.d1val1);
    unary.d2val2 = q6*(-q5*x.d2val2 + q8*x.val);
    unary.d3val1 = q11*(q10*x.d3val1 - q5*x.d1val1*(3.0*x.d2val1*x.val + q4) + q9*pow3(x.d1val1));
    unary.d2val1_d1val2 = q3*(q12*x.d2val1_d1val2 + q13*q14 - q13*x.d2val1_d1val2 + q14  - q16*q17 + q16*x.val - q17*x.d1val2*x.d2val1 + q7*x.d2val1 + x.d2val1_d1val2)*powm1(q12 - q13 + 1.0);
    unary.d1val1_d2val2 = q11*(q10*x.d1val1_d2val2 - q15*q5*q7 - x.d1val1*(q5*(q18 + q8) - q8*q9));
    unary.d3val2 = q11*(q10*x.d3val2 - q5*x.d1val2*(3.0*q18 + q8) + q9*pow3(x.d1val2));
    unary
}

/// `acospi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn acospi_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(PI);
    let q1 = pow2(x.val);
    let q2 = 1.0 - q1;
    let q3 = q0*powm1(f_sqrt(q2));
    let q4 = pow2(x.d1val1);
    let q5 = q1 - 1.0;
    let q6 = q0*powm1(pow3(f_sqrt(q2)));
    let q7 = x.d1val2*x.val;
    let q8 = q5*x.d1val1_d1val2;
    let q9 = pow2(x.d1val2);
    let q10 = pow3(q5);
    let q11 = 3.0*q1;
    let q12 = q1*(3.0 - q11);
    let q13 = pow2(q5);
    let q14 = q0*powm1(pow7(f_sqrt(q2)));
    let q15 = pow4(x.val);
    let q16 = 2.0*q1;
    let q17 = 2.0*x.d1val1*x.d1val1_d1val2;
    let q18 = q4*x.d1val2;
    let q19 = pow3(x.val);
    let q20 = x.d2val2*x.val;
    unary.val = q0*acos(x.val);
    unary.d1val1 = -q3*x.d1val1;
    unary.d1val2 = -q3*x.d1val2;
    unary.d2val1 = q6*(-q4*x.val + q5*x.d2val1);
    unary.d1val1_d1val2 = -q6*(q7*x.d1val1 - q8);
    unary.d2val2 = q6*(q5*x.d2val2 - q9*x.val);
    unary.d3val1 = q14*(q10*x.d3val1 - q12*pow3(x.d1val1) - q13*x.d1val1*(3.0*x.d2val1*x.val + q4));
    unary.d2val1_d1val2 = q3*(-q15*x.d2val1_d1val2 - q16*q18 + q16*x.d2val1_d1val2 + q17*q19 - q17*x.val  - q18 + q19*x.d1val2*x.d2val1 - q7*x.d2val1 - x.d2val1_d1val2)*powm1(q15 - q16 + 1.0);
    unary.d1val1_d2val2 = q0*(2.0*q7*q8 - q13*x.d1val1_d2val2 + x.d1val1*(-q11*q9 + q5*(q20 + q9)))*powm1(pow5(f_sqrt(q2)));
    unary.d3val2 = q14*(q10*x.d3val2 - q12*pow3(x.d1val2) - q13*x.d1val2*(3.0*q20 + q9));
    unary
}

/// `atanpi_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn atanpi_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = PI*q0;
    let q2 = powm1(PI + q1);
    let q3 = pow2(x.d1val1);
    let q4 = 2.0*x.val;
    let q5 = pow4(x.val);
    let q6 = PI*q5;
    let q7 = powm1(2.0*q1 + PI + q6);
    let q8 = q4*x.d1val2;
    let q9 = pow2(x.d1val2);
    let q10 = powm1(3.0*q1 + 3.0*q6 + PI*pow6(x.val) + PI);
    let q11 = pow3(x.d1val1);
    let q12 = x.d1val1*x.val;
    let q13 = 6.0*x.d2val1;
    let q14 = 2.0*q0;
    let q15 = pow3(x.val);
    let q16 = q15*x.d1val1;
    let q17 = 6.0*q0;
    let q18 = 4.0*x.d1val1_d1val2;
    let q19 = 2.0*x.d1val2;
    let q20 = q18*x.d1val2;
    let q21 = 2.0*x.d1val1;
    let q22 = q15*x.d2val2;
    let q23 = pow3(x.d1val2);
    let q24 = 6.0*x.d1val2;
    unary.val = powm1(PI)*atan(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q7*(q0*x.d2val1 - q3*q4 + x.d2val1);
    unary.d1val1_d1val2 = q7*(q0*x.d1val1_d1val2 - q8*x.d1val1 + x.d1val1_d1val2);
    unary.d2val2 = q7*(q0*x.d2val2 - q4*q9 + x.d2val2);
    unary.d3val1 = q10*(-2.0*q11 + q11*q17 - q12*q13 - q13*q16 + q14*x.d3val1 + q5*x.d3val1 + x.d3val1);
    unary.d2val1_d1val2 = q10 *  (-q12*q18 + q14*x.d2val1_d1val2 - q15*q19*x.d2val1 - q16*q18  + q17*q3*x.d1val2 - q19*q3 + q5*x.d2val1_d1val2 - q8*x.d2val1 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q10 *  (q14*x.d1val1_d2val2 - q15*q20 + q17*q9*x.d1val1 - q20*x.val  - q21*q22 - q21*q9 - q4*x.d1val1*x.d2val2 + q5*x.d1val1_d2val2 + x.d1val1_d2val2);
    unary.d3val2 = q10*(-2.0*q23 + q14*x.d3val2 + q17*q23 - q22*q24 - q24*x.d2val2*x.val + q5*x.d3val2 + x.d3val2);
    unary
}

/// `asinh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn asinh_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = q0 + 1.0;
    let q2 = powm1(f_sqrt(q1));
    let q3 = powm1(pow3(f_sqrt(q1)));
    let q4 = pow2(x.d1val1);
    let q5 = q4*x.val;
    let q6 = q1*x.d2val1;
    let q7 = x.d1val1*x.d1val2;
    let q8 = q1*x.d1val1_d1val2;
    let q9 = pow2(x.d1val2);
    let q10 = powm1(pow5(f_sqrt(q1)));
    let q11 = 3.0*q0;
    let q12 = pow2(q1);
    let q13 = q1*x.d1val1;
    let q14 = x.d1val2*x.val;
    let q15 = x.d2val2*x.val;
    unary.val = asinh(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q3*(-q5 + q6);
    unary.d1val1_d1val2 = q3*(-q7*x.val + q8);
    unary.d2val2 = q3*(q1*x.d2val2 - q9*x.val);
    unary.d3val1 = q10*(q11*pow3(x.d1val1) + q12*x.d3val1 - q13*(3.0*x.d2val1*x.val + q4));
    unary.d2val1_d1val2 = q10*(2.0*q0*q4*x.d1val2  + q12*x.d2val1_d1val2 - q13*(2.0*x.d1val1_d1val2*x.val + q7) + q14*(q5 - q6));
    unary.d1val1_d2val2 = q10*(-2.0*q14*q8 + q12*x.d1val1_d2val2 - x.d1val1*(q1*(q15 + q9) - q11*q9));
    unary.d3val2 = q10*(-q1*x.d1val2*(3.0*q15 + q9) + q11*pow3(x.d1val2) + q12*x.d3val2);
    unary
}

/// `acosh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn acosh_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = q0 - 1.0;
    let q2 = powm1(f_sqrt(q1));
    let q3 = powm1(pow3(f_sqrt(q1)));
    let q4 = pow2(x.d1val1);
    let q5 = q4*x.val;
    let q6 = q1*x.d2val1;
    let q7 = x.d1val1*x.d1val2;
    let q8 = q1*x.d1val1_d1val2;
    let q9 = pow2(x.d1val2);
    let q10 = powm1(pow5(f_sqrt(q1)));
    let q11 = 3.0*q0;
    let q12 = pow2(q1);
    let q13 = q1*x.d1val1;
    let q14 = x.d1val2*x.val;
    let q15 = x.d2val2*x.val;
    unary.val = acosh(x.val);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q3*(-q5 + q6);
    unary.d1val1_d1val2 = q3*(-q7*x.val + q8);
    unary.d2val2 = q3*(q1*x.d2val2 - q9*x.val);
    unary.d3val1 = q10*(q11*pow3(x.d1val1) + q12*x.d3val1 - q13*(3.0*x.d2val1*x.val + q4));
    unary.d2val1_d1val2 = q10*(2.0*q0*q4*x.d1val2  + q12*x.d2val1_d1val2 - q13*(2.0*x.d1val1_d1val2*x.val + q7) + q14*(q5 - q6));
    unary.d1val1_d2val2 = q10*(-2.0*q14*q8 + q12*x.d1val1_d2val2 - x.d1val1*(q1*(q15 + q9) - q11*q9));
    unary.d3val2 = q10*(-q1*x.d1val2*(3.0*q15 + q9) + q11*pow3(x.d1val2) + q12*x.d3val2);
    unary
}

/// `atanh_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn atanh_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = q0 - 1.0;
    let q2 = powm1(q1);
    let q3 = pow2(q1);
    let q4 = powm1(q3);
    let q5 = pow2(x.d1val1);
    let q6 = 2.0*x.val;
    let q7 = -q1*x.d2val1 + q5*q6;
    let q8 = x.d1val1*x.d1val2;
    let q9 = q1*x.d1val1_d1val2;
    let q10 = pow2(x.d1val2);
    let q11 = powm1(pow4(q1));
    let q12 = pow3(q1);
    let q13 = 8.0*q0*(1.0 - q0);
    let q14 = 2.0*x.d1val1;
    let q15 = powm1(q12);
    let q16 = 4.0*q0;
    let q17 = x.d2val2*x.val;
    unary.val = atanh(x.val);
    unary.d1val1 = -q2*x.d1val1;
    unary.d1val2 = -q2*x.d1val2;
    unary.d2val1 = q4*q7;
    unary.d1val1_d1val2 = q4*(q6*q8 - q9);
    unary.d2val2 = q4*(-q1*x.d2val2 + q10*q6);
    unary.d3val1 = q11*(-q12*x.d3val1 + q13*pow3(x.d1val1) + q14*q3*(3.0*x.d2val1*x.val + q5));
    unary.d2val1_d1val2 = q15*(q1*q14*(q6*x.d1val1_d1val2 + q8) - q16*q5*x.d1val2 - q3*x.d2val1_d1val2 - q6*q7*x.d1val2);
    unary.d1val1_d2val2 = q15*(4.0*q9*x.d1val2*x.val + q14*(q1*(q10 + q17) - q10*q16) - q3*x.d1val1_d2val2);
    unary.d3val2 = q11*(2.0*q3*x.d1val2*(3.0*q17 + q10) - q12*x.d3val2 + q13*pow3(x.d1val2));
    unary
}

/// `sqrt_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sqrt_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = f_sqrt(x.val);
    let q1 = 0.5*powm1(q0);
    let q2 = 2.0*x.val;
    let q3 = q2*x.d2val1;
    let q4 = pow2(x.d1val1);
    let q5 = 0.25*powm1(pow3(f_sqrt(x.val)));
    let q6 = q2*x.d2val2;
    let q7 = pow2(x.d1val2);
    let q8 = x.d1val1*x.val;
    let q9 = 4.0*pow2(x.val);
    let q10 = 0.125*powm1(pow5(f_sqrt(x.val)));
    let q11 = 4.0*x.d1val1_d1val2;
    let q12 = x.d1val2*x.val;
    unary.val = q0;
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q5*(q3 - q4);
    unary.d1val1_d1val2 = q5*(q2*x.d1val1_d1val2 - x.d1val1*x.d1val2);
    unary.d2val2 = q5*(q6 - q7);
    unary.d3val1 = q10*(-6.0*q8*x.d2val1 + 3.0*pow3(x.d1val1) + q9*x.d3val1);
    unary.d2val1_d1val2 = q10*(3.0*q4*x.d1val2 - q11*q8 - q3*x.d1val2 + q9*x.d2val1_d1val2);
    unary.d1val1_d2val2 = q10*(-q11*q12 + q9*x.d1val1_d2val2 + x.d1val1*(3.0*q7 - q6));
    unary.d3val2 = q10*(-6.0*q12*x.d2val2 + 3.0*pow3(x.d1val2) + q9*x.d3val2);
    unary
}

/// `pow2_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow2_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 2.0*x.val;
    let q1 = 2.0*x.d1val2;
    let q2 = 4.0*x.d1val1_d1val2;
    unary.val = pow2(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = 2.0*pow2(x.d1val1) + q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2 + q1*x.d1val1;
    unary.d2val2 = 2.0*pow2(x.d1val2) + q0*x.d2val2;
    unary.d3val1 = 6.0*x.d1val1*x.d2val1 + q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2 + q1*x.d2val1 + q2*x.d1val1;
    unary.d1val1_d2val2 = 2.0*x.d1val1*x.d2val2 + q0*x.d1val1_d2val2 + q2*x.d1val2;
    unary.d3val2 = 6.0*x.d1val2*x.d2val2 + q0*x.d3val2;
    unary
}

/// `pow3_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow3_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 3.0*pow2(x.val);
    let q1 = x.d2val1*x.val;
    let q2 = 2.0*pow2(x.d1val1) + q1;
    let q3 = 3.0*x.val;
    let q4 = x.d1val1_d1val2*x.val;
    let q5 = x.d2val2*x.val;
    let q6 = pow2(x.d1val2);
    unary.val = pow3(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q3*(2.0*x.d1val1*x.d1val2 + q4);
    unary.d2val2 = q3*(2.0*q6 + q5);
    unary.d3val1 = 18.0*q1*x.d1val1 + 6.0*pow3(x.d1val1) + q0*x.d3val1;
    unary.d2val1_d1val2 = 3.0*q2*x.d1val2  + q3*(4.0*x.d1val1*x.d1val1_d1val2 + x.d1val2*x.d2val1 + x.d2val1_d1val2*x.val);
    unary.d1val1_d2val2 = 12.0*q4*x.d1val2 + 6.0*x.d1val1*(q5 + q6) + q0*x.d1val1_d2val2;
    unary.d3val2 = 18.0*q5*x.d1val2 + 6.0*pow3(x.d1val2) + q0*x.d3val2;
    unary
}

/// `pow4_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow4_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 4.0*pow3(x.val);
    let q1 = x.d2val1*x.val;
    let q2 = 3.0*pow2(x.d1val1) + q1;
    let q3 = pow2(x.val);
    let q4 = 4.0*q3;
    let q5 = x.d1val1_d1val2*x.val;
    let q6 = 3.0*x.d1val1;
    let q7 = x.d2val2*x.val;
    let q8 = pow2(x.d1val2);
    let q9 = 4.0*x.val;
    unary.val = pow4(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*q4;
    unary.d1val1_d1val2 = q4*(q5 + q6*x.d1val2);
    unary.d2val2 = q4*(3.0*q8 + q7);
    unary.d3val1 = q9*(6.0*pow3(x.d1val1) + 9.0*q1*x.d1val1 + q3*x.d3val1);
    unary.d2val1_d1val2 = q9*(2.0*q2*x.d1val2  + x.val*(6.0*x.d1val1*x.d1val1_d1val2 + x.d1val2*x.d2val1 + x.d2val1_d1val2*x.val));
    unary.d1val1_d2val2 = q9*(6.0*q5*x.d1val2 + q3*x.d1val1_d2val2 + q6*(2.0*q8 + q7));
    unary.d3val2 = q9*(6.0*pow3(x.d1val2) + 9.0*q7*x.d1val2 + q3*x.d3val2);
    unary
}

/// `pow5_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow5_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 5.0*pow4(x.val);
    let q1 = x.d2val1*x.val;
    let q2 = 4.0*pow2(x.d1val1) + q1;
    let q3 = 5.0*pow3(x.val);
    let q4 = x.d1val1_d1val2*x.val;
    let q5 = 4.0*x.d1val1;
    let q6 = x.d2val2*x.val;
    let q7 = pow2(x.d1val2);
    let q8 = pow2(x.val);
    let q9 = 5.0*q8;
    unary.val = pow5(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q3*(q4 + q5*x.d1val2);
    unary.d2val2 = q3*(4.0*q7 + q6);
    unary.d3val1 = q9*(12.0*q1*x.d1val1 + 12.0*pow3(x.d1val1) + q8*x.d3val1);
    unary.d2val1_d1val2 = q9*(3.0*q2*x.d1val2  + x.val*(8.0*x.d1val1*x.d1val1_d1val2 + x.d1val2*x.d2val1 + x.d2val1_d1val2*x.val));
    unary.d1val1_d2val2 = q9*(8.0*q4*x.d1val2 + q5*(3.0*q7 + q6) + q8*x.d1val1_d2val2);
    unary.d3val2 = q9*(12.0*q6*x.d1val2 + 12.0*pow3(x.d1val2) + q8*x.d3val2);
    unary
}

/// `pow6_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow6_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 6.0*pow5(x.val);
    let q1 = x.d2val1*x.val;
    let q2 = 5.0*pow2(x.d1val1) + q1;
    let q3 = 6.0*pow4(x.val);
    let q4 = x.d1val1_d1val2*x.val;
    let q5 = 5.0*x.d1val1;
    let q6 = x.d2val2*x.val;
    let q7 = pow2(x.d1val2);
    let q8 = pow2(x.val);
    let q9 = 6.0*pow3(x.val);
    unary.val = pow6(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q3*(q4 + q5*x.d1val2);
    unary.d2val2 = q3*(5.0*q7 + q6);
    unary.d3val1 = q9*(15.0*q1*x.d1val1 + 20.0*pow3(x.d1val1) + q8*x.d3val1);
    unary.d2val1_d1val2 = q9*(4.0*q2*x.d1val2 +  x.val*(10.0*x.d1val1*x.d1val1_d1val2 + x.d1val2*x.d2val1 + x.d2val1_d1val2*x.val));
    unary.d1val1_d2val2 = q9*(10.0*q4*x.d1val2 + q5*(4.0*q7 + q6) + q8*x.d1val1_d2val2);
    unary.d3val2 = q9*(15.0*q6*x.d1val2 + 20.0*pow3(x.d1val2) + q8*x.d3val2);
    unary
}

/// `pow7_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow7_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 7.0*pow6(x.val);
    let q1 = x.d2val1*x.val;
    let q2 = 6.0*pow2(x.d1val1) + q1;
    let q3 = 7.0*pow5(x.val);
    let q4 = x.d1val1_d1val2*x.val;
    let q5 = 6.0*x.d1val1;
    let q6 = x.d2val2*x.val;
    let q7 = pow2(x.d1val2);
    let q8 = pow2(x.val);
    let q9 = 7.0*pow4(x.val);
    unary.val = pow7(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q3*(q4 + q5*x.d1val2);
    unary.d2val2 = q3*(6.0*q7 + q6);
    unary.d3val1 = q9*(18.0*q1*x.d1val1 + 30.0*pow3(x.d1val1) + q8*x.d3val1);
    unary.d2val1_d1val2 = q9*(5.0*q2*x.d1val2  + x.val*(12.0*x.d1val1*x.d1val1_d1val2 + x.d1val2*x.d2val1 + x.d2val1_d1val2*x.val));
    unary.d1val1_d2val2 = q9*(12.0*q4*x.d1val2 + q5*(5.0*q7 + q6) + q8*x.d1val1_d2val2);
    unary.d3val2 = q9*(18.0*q6*x.d1val2 + 30.0*pow3(x.d1val2) + q8*x.d3val2);
    unary
}

/// `pow8_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow8_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = 8.0*pow7(x.val);
    let q1 = x.d2val1*x.val;
    let q2 = 7.0*pow2(x.d1val1) + q1;
    let q3 = 8.0*pow6(x.val);
    let q4 = x.d1val1_d1val2*x.val;
    let q5 = 7.0*x.d1val1;
    let q6 = x.d2val2*x.val;
    let q7 = pow2(x.d1val2);
    let q8 = pow2(x.val);
    let q9 = 8.0*pow5(x.val);
    unary.val = pow8(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q3*(q4 + q5*x.d1val2);
    unary.d2val2 = q3*(7.0*q7 + q6);
    unary.d3val1 = q9*(21.0*q1*x.d1val1 + 42.0*pow3(x.d1val1) + q8*x.d3val1);
    unary.d2val1_d1val2 = q9*(6.0*q2*x.d1val2  + x.val*(14.0*x.d1val1*x.d1val1_d1val2 + x.d1val2*x.d2val1 + x.d2val1_d1val2*x.val));
    unary.d1val1_d2val2 = q9*(14.0*q4*x.d1val2 + q5*(6.0*q7 + q6) + q8*x.d1val1_d2val2);
    unary.d3val2 = q9*(21.0*q6*x.d1val2 + 42.0*pow3(x.d1val2) + q8*x.d3val2);
    unary
}

/// `abs_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn abs_self(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = sgn(x.val);
    unary.val = f_abs(x.val);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary.d3val1 = q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2;
    unary.d1val1_d2val2 = q0*x.d1val1_d2val2;
    unary.d3val2 = q0*x.d3val2;
    unary
}

/// `add_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn add_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    binary.val = x.val + y.val;
    binary.d1val1 = x.d1val1 + y.d1val1;
    binary.d1val2 = x.d1val2 + y.d1val2;
    binary.d2val1 = x.d2val1 + y.d2val1;
    binary.d1val1_d1val2 = x.d1val1_d1val2 + y.d1val1_d1val2;
    binary.d2val2 = x.d2val2 + y.d2val2;
    binary.d3val1 = x.d3val1 + y.d3val1;
    binary.d2val1_d1val2 = x.d2val1_d1val2 + y.d2val1_d1val2;
    binary.d1val1_d2val2 = x.d1val1_d2val2 + y.d1val1_d2val2;
    binary.d3val2 = x.d3val2 + y.d3val2;
    binary
}

/// `add_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn add_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = x.val + y;
    unary.d1val1 = x.d1val1;
    unary.d1val2 = x.d1val2;
    unary.d2val1 = x.d2val1;
    unary.d1val1_d1val2 = x.d1val1_d1val2;
    unary.d2val2 = x.d2val2;
    unary.d3val1 = x.d3val1;
    unary.d2val1_d1val2 = x.d2val1_d1val2;
    unary.d1val1_d2val2 = x.d1val1_d2val2;
    unary.d3val2 = x.d3val2;
    unary
}

/// `add_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn add_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = x.val + z;
    unary.d1val1 = x.d1val1;
    unary.d1val2 = x.d1val2;
    unary.d2val1 = x.d2val1;
    unary.d1val1_d1val2 = x.d1val1_d1val2;
    unary.d2val2 = x.d2val2;
    unary.d3val1 = x.d3val1;
    unary.d2val1_d1val2 = x.d2val1_d1val2;
    unary.d1val1_d2val2 = x.d1val1_d2val2;
    unary.d3val2 = x.d3val2;
    unary
}

/// `sub_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sub_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    binary.val = x.val - y.val;
    binary.d1val1 = x.d1val1 - y.d1val1;
    binary.d1val2 = x.d1val2 - y.d1val2;
    binary.d2val1 = x.d2val1 - y.d2val1;
    binary.d1val1_d1val2 = x.d1val1_d1val2 - y.d1val1_d1val2;
    binary.d2val2 = x.d2val2 - y.d2val2;
    binary.d3val1 = x.d3val1 - y.d3val1;
    binary.d2val1_d1val2 = x.d2val1_d1val2 - y.d2val1_d1val2;
    binary.d1val1_d2val2 = x.d1val1_d2val2 - y.d1val1_d2val2;
    binary.d3val2 = x.d3val2 - y.d3val2;
    binary
}

/// `sub_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sub_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = x.val - y;
    unary.d1val1 = x.d1val1;
    unary.d1val2 = x.d1val2;
    unary.d2val1 = x.d2val1;
    unary.d1val1_d1val2 = x.d1val1_d1val2;
    unary.d2val2 = x.d2val2;
    unary.d3val1 = x.d3val1;
    unary.d2val1_d1val2 = x.d2val1_d1val2;
    unary.d1val1_d2val2 = x.d1val1_d2val2;
    unary.d3val2 = x.d3val2;
    unary
}

/// `sub_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn sub_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = -x.val + z;
    unary.d1val1 = -x.d1val1;
    unary.d1val2 = -x.d1val2;
    unary.d2val1 = -x.d2val1;
    unary.d1val1_d1val2 = -x.d1val1_d1val2;
    unary.d2val2 = -x.d2val2;
    unary.d3val1 = -x.d3val1;
    unary.d2val1_d1val2 = -x.d2val1_d1val2;
    unary.d1val1_d2val2 = -x.d1val1_d2val2;
    unary.d3val2 = -x.d3val2;
    unary
}

/// `mul_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn mul_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = 2.0*x.d1val1;
    let q1 = 2.0*y.d1val2;
    binary.val = x.val*y.val;
    binary.d1val1 = x.d1val1*y.val + x.val*y.d1val1;
    binary.d1val2 = x.d1val2*y.val + x.val*y.d1val2;
    binary.d2val1 = q0*y.d1val1 + x.d2val1*y.val + x.val*y.d2val1;
    binary.d1val1_d1val2 = x.d1val1*y.d1val2 + x.d1val1_d1val2*y.val + x.d1val2*y.d1val1 + x.val*y.d1val1_d1val2;
    binary.d2val2 = q1*x.d1val2 + x.d2val2*y.val + x.val*y.d2val2;
    binary.d3val1 = 3.0*x.d1val1*y.d2val1 + 3.0*x.d2val1*y.d1val1 + x.d3val1*y.val + x.val*y.d3val1;
    binary.d2val1_d1val2 = 2.0*x.d1val1_d1val2*y.d1val1 + q0*y.d1val1_d1val2  + x.d1val2*y.d2val1 + x.d2val1*y.d1val2 + x.d2val1_d1val2*y.val + x.val*y.d2val1_d1val2;
    binary.d1val1_d2val2 = 2.0*x.d1val2*y.d1val1_d1val2 + q1*x.d1val1_d1val2  + x.d1val1*y.d2val2 + x.d1val1_d2val2*y.val + x.d2val2*y.d1val1 + x.val*y.d1val1_d2val2;
    binary.d3val2 = 3.0*x.d1val2*y.d2val2 + 3.0*x.d2val2*y.d1val2 + x.d3val2*y.val + x.val*y.d3val2;
    binary
}

/// `mul_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn mul_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = x.val*y;
    unary.d1val1 = x.d1val1*y;
    unary.d1val2 = x.d1val2*y;
    unary.d2val1 = x.d2val1*y;
    unary.d1val1_d1val2 = x.d1val1_d1val2*y;
    unary.d2val2 = x.d2val2*y;
    unary.d3val1 = x.d3val1*y;
    unary.d2val1_d1val2 = x.d2val1_d1val2*y;
    unary.d1val1_d2val2 = x.d1val1_d2val2*y;
    unary.d3val2 = x.d3val2*y;
    unary
}

/// `mul_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn mul_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    unary.val = x.val*z;
    unary.d1val1 = x.d1val1*z;
    unary.d1val2 = x.d1val2*z;
    unary.d2val1 = x.d2val1*z;
    unary.d1val1_d1val2 = x.d1val1_d1val2*z;
    unary.d2val2 = x.d2val2*z;
    unary.d3val1 = x.d3val1*z;
    unary.d2val1_d1val2 = x.d2val1_d1val2*z;
    unary.d1val1_d2val2 = x.d1val1_d2val2*z;
    unary.d3val2 = x.d3val2*z;
    unary
}

/// `div_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn div_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(y.val);
    let q1 = powm1(q0);
    let q2 = x.d1val1*y.val;
    let q3 = x.val*y.d1val1;
    let q4 = x.d1val2*y.val;
    let q5 = x.val*y.d1val2;
    let q6 = pow3(y.val);
    let q7 = powm1(q6);
    let q8 = q0*x.d2val1;
    let q9 = 2.0*y.d1val1;
    let q10 = y.d2val1*y.val;
    let q11 = pow2(y.d1val1);
    let q12 = 2.0*q11;
    let q13 = -q10 + q12;
    let q14 = q0*x.d1val1_d1val2;
    let q15 = 2.0*y.d1val2;
    let q16 = x.d1val2*y.d1val1;
    let q17 = q0*x.d2val2;
    let q18 = y.d2val2*y.val;
    let q19 = pow2(y.d1val2);
    let q20 = 2.0*q19 - q18;
    let q21 = powm1(pow4(y.val));
    let q22 = 2.0*y.d1val1_d1val2;
    binary.val = x.val*powm1(y.val);
    binary.d1val1 = q1*(q2 - q3);
    binary.d1val2 = q1*(q4 - q5);
    binary.d2val1 = q7*(q13*x.val - q2*q9 + q8);
    binary.d1val1_d1val2 = q7*(q14 + q15*q3 - y.val*(q16 + x.d1val1*y.d1val2 + x.val*y.d1val1_d1val2));
    binary.d2val2 = q7*(-q15*q4 + q17 + q20*x.val);
    binary.d3val1 = q21*(-3.0*q8*y.d1val1 + 3.0*q13*q2 + q6*x.d3val1  - x.val*(-6.0*q10*y.d1val1 + 6.0*pow3(y.d1val1) + q0*y.d3val1));
    binary.d2val1_d1val2 = q21*(-6.0*q11*q5 + 2.0*q10*q5 + 4.0*q2*y.d1val1*y.d1val2  + 4.0*q3*y.d1val1_d1val2*y.val - q0*q22*x.d1val1  - q0*x.d1val2*y.d2val1 - q0*x.val*y.d2val1_d1val2 + q12*q4 - q14*q9 + q6*x.d2val1_d1val2 - q8*y.d1val2);
    binary.d1val1_d2val2 = q21*(-6.0*q19*q3 + 2.0*y.val*(q15*q16 + q19*x.d1val1 + q22*q5 + q3*y.d2val2)  - q0*(q15*x.d1val1_d1val2 + q22*x.d1val2  + x.d1val1*y.d2val2 + x.d2val2*y.d1val1 + x.val*y.d1val1_d2val2)  + q6*x.d1val1_d2val2);
    binary.d3val2 = q21*(-3.0*q17*y.d1val2 + 3.0*q20*q4  + q6*x.d3val2 - x.val*(-6.0*q18*y.d1val2 + 6.0*pow3(y.d1val2) + q0*y.d3val2));
    binary
}

/// `div_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn div_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = powm1(y);
    unary.val = q0*x.val;
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary.d3val1 = q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2;
    unary.d1val1_d2val2 = q0*x.d1val1_d2val2;
    unary.d3val2 = q0*x.d3val2;
    unary
}

/// `div_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn div_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow2(x.val);
    let q1 = z*powm1(q0);
    let q2 = x.d2val1*x.val;
    let q3 = pow2(x.d1val1);
    let q4 = z*powm1(pow3(x.val));
    let q5 = 2.0*x.d1val1;
    let q6 = x.d1val1_d1val2*x.val;
    let q7 = x.d2val2*x.val;
    let q8 = -q7;
    let q9 = pow2(x.d1val2);
    let q10 = z*powm1(pow4(x.val));
    let q11 = 4.0*q6;
    let q12 = 6.0*x.d1val2;
    unary.val = z*powm1(x.val);
    unary.d1val1 = -q1*x.d1val1;
    unary.d1val2 = -q1*x.d1val2;
    unary.d2val1 = q4*(2.0*q3 - q2);
    unary.d1val1_d1val2 = q4*(q5*x.d1val2 - q6);
    unary.d2val2 = q4*(2.0*q9 + q8);
    unary.d3val1 = q10*(-6.0*pow3(x.d1val1) + 6.0*q2*x.d1val1 - q0*x.d3val1);
    unary.d2val1_d1val2 = q10*(2.0*q2*x.d1val2 - q0*x.d2val1_d1val2 + q11*x.d1val1 - q12*q3);
    unary.d1val1_d2val2 = -q10*(q0*x.d1val1_d2val2 - q11*x.d1val2 + q5*(3.0*q9 + q8));
    unary.d3val2 = q10*(-6.0*pow3(x.d1val2) - q0*x.d3val2 + q12*q7);
    unary
}

/// `pow_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = pow(x.val, y.val - 1.0);
    let q1 = x.d1val1*y.val;
    let q2 = log(x.val);
    let q3 = q2*x.val;
    let q4 = q1 + q3*y.d1val1;
    let q5 = x.d1val2*y.val;
    let q6 = q3*y.d1val2 + q5;
    let q7 = pow(x.val, -2.0 + y.val);
    let q8 = pow2(x.d1val1);
    let q9 = pow2(x.val);
    let q10 = q2*q9;
    let q11 = x.d2val1*y.val;
    let q12 = x.d1val1*y.d1val1;
    let q13 = q10*y.d2val1 - q8*y.val + x.val*(2.0*q12 + q11);
    let q14 = q13 + pow2(q4);
    let q15 = x.d1val1*y.d1val2;
    let q16 = x.d1val2*y.d1val1;
    let q17 = -q1*x.d1val2 + q10*y.d1val1_d1val2 + x.val*(q15 + q16 + x.d1val1_d1val2*y.val);
    let q18 = pow2(x.d1val2);
    let q19 = x.d2val2*y.val;
    let q20 = x.d1val2*y.d1val2;
    let q21 = q10*y.d2val2 - q18*y.val + x.val*(2.0*q20 + q19);
    let q22 = q21 + pow2(q6);
    let q23 = pow(x.val, -3.0 + y.val);
    let q24 = 2.0*y.val;
    let q25 = q2*pow3(x.val);
    let q26 = 3.0*x.d1val1;
    let q27 = 2.0*y.d1val1_d1val2;
    let q28 = 2.0*x.d1val1_d1val2;
    let q29 = 2.0*q17;
    let q30 = 3.0*x.d1val2;
    binary.val = pow(x.val, y.val);
    binary.d1val1 = q0*q4;
    binary.d1val2 = q0*q6;
    binary.d2val1 = q14*q7;
    binary.d1val1_d1val2 = (q17 + q4*q6)*pow(x.val, 3.0 + y.val)*powm1(pow5(x.val));
    binary.d2val2 = q22*q7;
    binary.d3val1 = q23*(3.0*q13*q4 + q24*pow3(x.d1val1) + q25*y.d3val1  - q26*x.val*(q11 + q12) + q9*(3.0*x.d2val1*y.d1val1  + q26*y.d2val1 + x.d3val1*y.val) + pow3(q4));
    binary.d2val1_d1val2 = (2.0*q5*q8 + q14*q6 + q25*y.d2val1_d1val2 + q29*q4  + q9*(q27*x.d1val1 + q28*y.d1val1 + x.d1val2*y.d2val1  + x.d2val1*y.d1val2 + x.d2val1_d1val2*y.val)  - x.val*(2.0*q16*x.d1val1 + q1*q28 + q5*x.d2val1 + q8*y.d1val2))  * pow(x.val, 10.0 + y.val)*powm1(pow(x.val, 13.0));
    binary.d1val1_d2val2 = q23*(2.0*q1*q18 + q22*q4 + q25*y.d1val1_d2val2 + q29*q6  + q9*(q27*x.d1val2 + q28*y.d1val2 + x.d1val1*y.d2val2 + x.d1val1_d2val2*y.val + x.d2val2*y.d1val1)  - x.val*(2.0*q15*x.d1val2 + q1*x.d2val2 + q18*y.d1val1 + q28*q5));
    binary.d3val2 = q23 *  ( 3.0*q21*q6 + q24*pow3(x.d1val2) + q25*y.d3val2 - q30*x.val*(q19 + q20)  + q9*(3.0*x.d2val2*y.d1val2 + q30*y.d2val2 + x.d3val2*y.val) + pow3(q6));
    binary
}

/// `pow_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = y - 1.0;
    let q1 = y*pow(x.val, q0);
    let q2 = x.d2val1*x.val;
    let q3 = pow2(x.d1val1);
    let q4 = q3*y;
    let q5 = pow(x.val, -2.0 + y);
    let q6 = q5*y;
    let q7 = x.d1val1_d1val2*x.val;
    let q8 = x.d1val1*x.d1val2;
    let q9 = x.d2val2*x.val;
    let q10 = pow2(x.d1val2);
    let q11 = q10*y;
    let q12 = y*(-q10 + q11 + q9);
    let q13 = pow2(x.val);
    let q14 = pow2(y);
    let q15 = -3.0*y + 2.0 + q14;
    let q16 = y*pow(x.val, -3.0 + y);
    let q17 = 2.0*q7;
    let q18 = q17*x.d1val1;
    let q19 = q2*x.d1val2;
    let q20 = q3*x.d1val2;
    let q21 = 3.0*x.d1val2;
    unary.val = pow(x.val, y);
    unary.d1val1 = q1*x.d1val1;
    unary.d1val2 = q1*x.d1val2;
    unary.d2val1 = q6*(q2 - q3 + q4);
    unary.d1val1_d1val2 = q6*(q7 + q8*y - q8);
    unary.d2val2 = q12*q5;
    unary.d3val1 = q16*(3.0*q0*q2*x.d1val1 + q13*x.d3val1 + q15*pow3(x.d1val1));
    unary.d2val1_d1val2 = q16*(2.0*q20 + q13*x.d2val1_d1val2 + q14*q20 + q18*y - q18 + q19*y - q19 - q21*q4);
    unary.d1val1_d2val2 = q16*(q0*q17*x.d1val2 + q13*x.d1val1_d2val2 - x.d1val1*(-2.0*q10 + 2.0*q11 - q12 + q9));
    unary.d3val2 = q16*(q0*q21*q9 + q13*x.d3val2 + q15*pow3(x.d1val2));
    unary
}

/// `pow_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn pow_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = pow(z, x.val);
    let q1 = log(z);
    let q2 = q0*q1;
    let q3 = q1*pow2(x.d1val1) + x.d2val1;
    let q4 = q1*x.d1val2;
    let q5 = q1*pow2(x.d1val2) + x.d2val2;
    let q6 = q1*x.d1val1;
    let q7 = pow2(q1);
    let q8 = 2.0*x.d1val1_d1val2;
    unary.val = q0;
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*q3;
    unary.d1val1_d1val2 = q2*(q4*x.d1val1 + x.d1val1_d1val2);
    unary.d2val2 = q2*q5;
    unary.d3val1 = q2*(3.0*q6*x.d2val1 + q7*pow3(x.d1val1) + x.d3val1);
    unary.d2val1_d1val2 = q2*(q3*q4 + q6*q8 + x.d2val1_d1val2);
    unary.d1val1_d2val2 = q2*(q4*q8 + q5*q6 + x.d1val1_d2val2);
    unary.d3val2 = q2*(3.0*q4*x.d2val2 + q7*pow3(x.d1val2) + x.d3val2);
    unary
}

/// `max_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn max_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(x.val - y.val);
    let q1 = heaviside(-x.val + y.val);
    binary.val = f_max(x.val, y.val);
    binary.d1val1 = q0*x.d1val1 + q1*y.d1val1;
    binary.d1val2 = q0*x.d1val2 + q1*y.d1val2;
    binary.d2val1 = q0*x.d2val1 + q1*y.d2val1;
    binary.d1val1_d1val2 = q0*x.d1val1_d1val2 + q1*y.d1val1_d1val2;
    binary.d2val2 = q0*x.d2val2 + q1*y.d2val2;
    binary.d3val1 = q0*x.d3val1 + q1*y.d3val1;
    binary.d2val1_d1val2 = q0*x.d2val1_d1val2 + q1*y.d2val1_d1val2;
    binary.d1val1_d2val2 = q0*x.d1val1_d2val2 + q1*y.d1val1_d2val2;
    binary.d3val2 = q0*x.d3val2 + q1*y.d3val2;
    binary
}

/// `max_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn max_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(x.val - y);
    unary.val = f_max(x.val, y);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary.d3val1 = q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2;
    unary.d1val1_d2val2 = q0*x.d1val1_d2val2;
    unary.d3val2 = q0*x.d3val2;
    unary
}

/// `max_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn max_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(x.val - z);
    unary.val = f_max(x.val, z);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary.d3val1 = q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2;
    unary.d1val1_d2val2 = q0*x.d1val1_d2val2;
    unary.d3val2 = q0*x.d3val2;
    unary
}

/// `min_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn min_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(-x.val + y.val);
    let q1 = heaviside(x.val - y.val);
    binary.val = f_min(x.val, y.val);
    binary.d1val1 = q0*x.d1val1 + q1*y.d1val1;
    binary.d1val2 = q0*x.d1val2 + q1*y.d1val2;
    binary.d2val1 = q0*x.d2val1 + q1*y.d2val1;
    binary.d1val1_d1val2 = q0*x.d1val1_d1val2 + q1*y.d1val1_d1val2;
    binary.d2val2 = q0*x.d2val2 + q1*y.d2val2;
    binary.d3val1 = q0*x.d3val1 + q1*y.d3val1;
    binary.d2val1_d1val2 = q0*x.d2val1_d1val2 + q1*y.d2val1_d1val2;
    binary.d1val1_d2val2 = q0*x.d1val1_d2val2 + q1*y.d1val1_d2val2;
    binary.d3val2 = q0*x.d3val2 + q1*y.d3val2;
    binary
}

/// `min_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn min_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(-x.val + y);
    unary.val = f_min(x.val, y);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary.d3val1 = q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2;
    unary.d1val1_d2val2 = q0*x.d1val1_d2val2;
    unary.d3val2 = q0*x.d3val2;
    unary
}

/// `min_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn min_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = heaviside(-x.val + z);
    unary.val = f_min(x.val, z);
    unary.d1val1 = q0*x.d1val1;
    unary.d1val2 = q0*x.d1val2;
    unary.d2val1 = q0*x.d2val1;
    unary.d1val1_d1val2 = q0*x.d1val1_d1val2;
    unary.d2val2 = q0*x.d2val2;
    unary.d3val1 = q0*x.d3val1;
    unary.d2val1_d1val2 = q0*x.d2val1_d1val2;
    unary.d1val1_d2val2 = q0*x.d1val1_d2val2;
    unary.d3val2 = q0*x.d3val2;
    unary
}

/// `dim_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn dim_self(x: AutoDiffReal2VarOrder3, y: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut binary = AutoDiffReal2VarOrder3::default();
    let q0 = x.val - y.val;
    let q1 = 0.5*sgn(q0);
    binary.val = -0.5*y.val + 0.5*x.val + 0.5*f_abs(q0);
    binary.d1val1 = -0.5*y.d1val1 + 0.5*x.d1val1 + q1*(x.d1val1 - y.d1val1);
    binary.d1val2 = -0.5*y.d1val2 + 0.5*x.d1val2 + q1*(x.d1val2 - y.d1val2);
    binary.d2val1 = -0.5*y.d2val1 + 0.5*x.d2val1 + q1*(x.d2val1 - y.d2val1);
    binary.d1val1_d1val2 = -0.5*y.d1val1_d1val2 + 0.5*x.d1val1_d1val2 + q1*(x.d1val1_d1val2 - y.d1val1_d1val2);
    binary.d2val2 = -0.5*y.d2val2 + 0.5*x.d2val2 + q1*(x.d2val2 - y.d2val2);
    binary.d3val1 = -0.5*y.d3val1 + 0.5*x.d3val1 + q1*(x.d3val1 - y.d3val1);
    binary.d2val1_d1val2 = -0.5*y.d2val1_d1val2 + 0.5*x.d2val1_d1val2 + q1*(x.d2val1_d1val2 - y.d2val1_d1val2);
    binary.d1val1_d2val2 = -0.5*y.d1val1_d2val2 + 0.5*x.d1val1_d2val2 + q1*(x.d1val1_d2val2 - y.d1val1_d2val2);
    binary.d3val2 = -0.5*y.d3val2 + 0.5*x.d3val2 + q1*(x.d3val2 - y.d3val2);
    binary
}

/// `dim_self_real` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn dim_self_real(x: AutoDiffReal2VarOrder3, y: f64) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = x.val - y;
    let q1 = sgn(q0);
    let q2 = 0.5*q1 + 0.5;
    let q3 = 0.5*x.d1val1_d1val2;
    let q4 = 0.5*x.d2val1_d1val2;
    let q5 = 0.5*x.d1val1_d2val2;
    unary.val = -0.5*y + 0.5*x.val + 0.5*f_abs(q0);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*x.d2val1;
    unary.d1val1_d1val2 = q1*q3 + q3;
    unary.d2val2 = q2*x.d2val2;
    unary.d3val1 = q2*x.d3val1;
    unary.d2val1_d1val2 = q1*q4 + q4;
    unary.d1val1_d2val2 = q1*q5 + q5;
    unary.d3val2 = q2*x.d3val2;
    unary
}

/// `dim_real_self` in the upstream module.
#[inline]
#[must_use]
pub(crate) fn dim_real_self(z: f64, x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    let mut unary = AutoDiffReal2VarOrder3::default();
    let q0 = x.val - z;
    let q1 = sgn(q0);
    let q2 = -0.5 + 0.5*q1;
    let q3 = 0.5*x.d1val1_d1val2;
    let q4 = 0.5*x.d2val1_d1val2;
    let q5 = 0.5*x.d1val1_d2val2;
    unary.val = -0.5*x.val + 0.5*z + 0.5*f_abs(q0);
    unary.d1val1 = q2*x.d1val1;
    unary.d1val2 = q2*x.d1val2;
    unary.d2val1 = q2*x.d2val1;
    unary.d1val1_d1val2 = q1*q3 - q3;
    unary.d2val2 = q2*x.d2val2;
    unary.d3val1 = q2*x.d3val1;
    unary.d2val1_d1val2 = q1*q4 - q4;
    unary.d1val1_d2val2 = q1*q5 - q5;
    unary.d3val2 = q2*x.d3val2;
    unary
}
