// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity and property-based tests for `flexmesa-autodiff`.
//!
//! Verifies MESA's `auto_diff/test/src/test_auto_diff.f90` cases across
//! `AutoDiffReal1VarOrder1`, `AutoDiffReal2VarOrder1`,
//! `AutoDiffReal4VarOrder1`, `AutoDiffRealStarOrder1`, and
//! `AutoDiffReal2VarOrder2` to `<= 1e-14`, plus `proptest` central
//! finite-difference and chain-rule checks.

#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use flexmesa_autodiff::{
    AutoDiffReal1VarOrder1, AutoDiffReal2VarOrder1, AutoDiffReal2VarOrder2,
    AutoDiffReal4VarOrder1, AutoDiffRealStarOrder1,
};
use flexmesa_const::{LN10, PI};
use flexmesa_math::{cos, cosh, exp, log, pow2, sin, sinh, tan, tanh};
use flexmesa_testing::assert_values_match;
use proptest::prelude::*;

const TOL: f64 = 1.0e-14;

fn assert_allclose(actual: &[f64], expected: &[f64], rtol: f64, atol: f64) {
    assert_values_match(actual, expected, rtol, atol, "autodiff_tests")
        .expect("autodiff_tests parity mismatch");
}

fn check_2var_order1(z: AutoDiffReal2VarOrder1, val: f64, d1: f64, d2: f64) {
    assert_allclose(&[z.val, z.d1[0], z.d1[1]], &[val, d1, d2], TOL, TOL);
}

#[test]
fn test_auto_diff_2var_order1_binary_operators_match_mesa_test_suite() {
    let x = AutoDiffReal2VarOrder1::new(1.0, [1.0, 0.0]);
    let y = AutoDiffReal2VarOrder1::new(2.0, [1.0, 0.0]);
    check_2var_order1(x + y, 3.0, 2.0, 0.0);
    check_2var_order1(x + 2.0, 3.0, 1.0, 0.0);
    check_2var_order1(2.0 + x, 3.0, 1.0, 0.0);

    let x = AutoDiffReal2VarOrder1::new(3.0, [1.0, 0.0]);
    let y = AutoDiffReal2VarOrder1::new(2.0, [0.0, 1.0]);
    check_2var_order1(x * y, 6.0, 2.0, 3.0);
    check_2var_order1(x * 2.0, 6.0, 2.0, 0.0);
    check_2var_order1(2.0 * x, 6.0, 2.0, 0.0);
    check_2var_order1(x / y, 1.5, 0.5, -0.75);
    check_2var_order1(x / 2.0, 1.5, 0.5, 0.0);

    let x = AutoDiffReal2VarOrder1::new(1.0, [1.0, 0.0]);
    let y = AutoDiffReal2VarOrder1::new(5.0, [0.0, 2.0]);
    check_2var_order1(x - y, -4.0, 1.0, -2.0);
    check_2var_order1(x - 1.0, 0.0, 1.0, 0.0);

    let x = AutoDiffReal2VarOrder1::new(1.0, [1.0, 0.0]);
    let y = AutoDiffReal2VarOrder1::new(2.0, [0.0, 2.0]);
    check_2var_order1(x.max(y), 2.0, 0.0, 2.0);
    check_2var_order1(x.max_scalar(2.0), 2.0, 0.0, 0.0);
    check_2var_order1(x.max_scalar(0.0), 1.0, 1.0, 0.0);
    check_2var_order1(x.min(y), 1.0, 1.0, 0.0);
    check_2var_order1(x.min_scalar(2.0), 1.0, 1.0, 0.0);
    check_2var_order1(x.min_scalar(0.0), 0.0, 0.0, 0.0);

    let x = AutoDiffReal2VarOrder1::new(2.0, [2.0, 0.0]);
    let y = AutoDiffReal2VarOrder1::new(1.0, [0.0, 1.0]);
    check_2var_order1(x.dim(y), 1.0, 2.0, -1.0);
    check_2var_order1(y.dim(x), 0.0, 0.0, 0.0);
    check_2var_order1(x.dim_scalar(1.0), 1.0, 2.0, 0.0);

    let x = AutoDiffReal2VarOrder1::new(2.0, [1.0, 0.0]);
    let y = AutoDiffReal2VarOrder1::new(2.0, [0.0, 1.0]);
    check_2var_order1(x.pow(y), 4.0, 4.0, 4.0 * log(2.0));
}

#[test]
fn test_auto_diff_2var_order1_unary_operators_match_mesa_test_suite() {
    let x1 = AutoDiffReal2VarOrder1::new(1.0, [1.0, 0.0]);
    check_2var_order1(x1.exp(), exp(1.0), exp(1.0), 0.0);
    check_2var_order1(x1.abs(), 1.0, 1.0, 0.0);
    check_2var_order1((-x1).abs(), 1.0, 1.0, 0.0);
    check_2var_order1(-x1, -1.0, -1.0, 0.0);
    check_2var_order1(x1.log(), 0.0, 1.0, 0.0);
    check_2var_order1(x1.safe_log(), 0.0, 1.0, 0.0);

    let x_pi = AutoDiffReal2VarOrder1::new(PI, [1.0, 0.0]);
    check_2var_order1(x_pi.sin(), sin(PI), cos(PI), 0.0);
    check_2var_order1(x_pi.cos(), cos(PI), -sin(PI), 0.0);
    check_2var_order1(x_pi.tan(), tan(PI), 1.0 / pow2(cos(PI)), 0.0);
    check_2var_order1(x_pi.sinh(), sinh(PI), cosh(PI), 0.0);
    check_2var_order1(x_pi.cosh(), cosh(PI), sinh(PI), 0.0);
    check_2var_order1(x_pi.tanh(), tanh(PI), 1.0 / pow2(cosh(PI)), 0.0);

    let x_half = AutoDiffReal2VarOrder1::new(0.5, [1.0, 0.0]);
    check_2var_order1(x_half.asin(), 0.5_f64.asin(), 2.0 / 3.0_f64.sqrt(), 0.0);
    check_2var_order1(
        x_half.acos(),
        0.5_f64.acos(),
        -2.0 / 3.0_f64.sqrt(),
        0.0,
    );
    check_2var_order1(x_half.atan(), 0.5_f64.atan(), 0.8, 0.0);
    check_2var_order1(x_half.atanh(), 0.5_f64.atanh(), 4.0 / 3.0, 0.0);

    let x2 = AutoDiffReal2VarOrder1::new(2.0, [1.0, 0.0]);
    check_2var_order1(x2.asinh(), 2.0_f64.asinh(), 1.0 / 5.0_f64.sqrt(), 0.0);
    check_2var_order1(x2.acosh(), 2.0_f64.acosh(), 1.0 / 3.0_f64.sqrt(), 0.0);
    check_2var_order1(x2.pow3(), 8.0, 12.0, 0.0);
    check_2var_order1(x2.pow4(), 16.0, 32.0, 0.0);
    check_2var_order1(x2.pow6(), 64.0, 192.0, 0.0);
    check_2var_order1(x2.pow7(), 128.0, 448.0, 0.0);
    check_2var_order1(x2.pow8(), 256.0, 1024.0, 0.0);
    check_2var_order1(x2.rpow(exp(1.0)), exp(2.0), exp(2.0), 0.0);
    check_2var_order1(x2.rpow(2.0), 4.0, 4.0 * log(2.0), 0.0);

    let x3 = AutoDiffReal2VarOrder1::new(3.0, [1.0, 0.0]);
    check_2var_order1(x3.pow2(), 9.0, 6.0, 0.0);
    check_2var_order1(x3.pow5(), 243.0, 405.0, 0.0);

    let x4 = AutoDiffReal2VarOrder1::new(4.0, [1.0, 0.0]);
    check_2var_order1(x4.powf(0.5), 2.0, 0.25, 0.0);
    check_2var_order1(x4.powi(2), 16.0, 8.0, 0.0);

    let x10 = AutoDiffReal2VarOrder1::new(10.0, [1.0, 0.0]);
    check_2var_order1(x10.log10(), 1.0, 1.0 / (10.0 * LN10), 0.0);
    check_2var_order1(x10.safe_log10(), 1.0, 1.0 / (10.0 * LN10), 0.0);
}

#[test]
fn test_auto_diff_1var_4var_and_star_order1_propagate_partials_correctly() {
    let u = AutoDiffReal1VarOrder1::independent(2.0, 0);
    let fu = (u.pow3() + u.exp()) / u;
    let expected_val = (8.0 + exp(2.0)) / 2.0;
    let expected_d1 = (2.0 * (12.0 + exp(2.0)) - (8.0 + exp(2.0))) / 4.0;
    assert_allclose(&[fu.val, fu.d1[0]], &[expected_val, expected_d1], TOL, TOL);

    let a = AutoDiffReal4VarOrder1::independent(1.5, 0);
    let b = AutoDiffReal4VarOrder1::independent(2.5, 1);
    let c = AutoDiffReal4VarOrder1::independent(-0.5, 2);
    let d = AutoDiffReal4VarOrder1::independent(0.25, 3);
    let q = a * b + c.exp() - d.sinpi();
    assert_allclose(
        &q.d1,
        &[2.5, 1.5, (-0.5_f64).exp(), -PI * (0.25 * PI).cos()],
        TOL,
        TOL,
    );

    let mut star_sum = AutoDiffRealStarOrder1::constant(0.0);
    for idx in 0..15 {
        let xi = AutoDiffRealStarOrder1::independent(1.0 + idx as f64, idx);
        star_sum = star_sum + xi.pow2();
    }
    for idx in 0..15 {
        let expected = 2.0 * (1.0 + idx as f64);
        assert_allclose(&[star_sum.d1[idx]], &[expected], TOL, TOL);
    }
    for idx in 15..33 {
        assert_eq!(star_sum.d1[idx], 0.0);
    }
}

#[test]
fn test_auto_diff_2var_order2_hessian_and_differentiate_match_exact_calculus() {
    let x = AutoDiffReal2VarOrder2::var1(1.2);
    let y = AutoDiffReal2VarOrder2::var2(0.7);
    let f = (x * y).exp() + x.pow3() / y.sin();

    let xv = 1.2_f64;
    let yv = 0.7_f64;
    let exy = (xv * yv).exp();
    let sy = yv.sin();
    let cy = yv.cos();

    let expected_val = exy + xv.powi(3) / sy;
    let expected_dx = yv * exy + 3.0 * xv.powi(2) / sy;
    let expected_dy = xv * exy - xv.powi(3) * cy / (sy * sy);
    let expected_dxx = yv * yv * exy + 6.0 * xv / sy;
    let expected_dxy =
        exy * (1.0 + xv * yv) - 3.0 * xv.powi(2) * cy / (sy * sy);
    let expected_dyy = xv * xv * exy
        + xv.powi(3) * (1.0 + cy * cy) / (sy * sy * sy);

    assert_allclose(
        &[f.val, f.d1val1, f.d1val2, f.d2val1, f.d1val1_d1val2, f.d2val2],
        &[
            expected_val,
            expected_dx,
            expected_dy,
            expected_dxx,
            expected_dxy,
            expected_dyy,
        ],
        TOL,
        TOL,
    );

    let df_dx = f.differentiate_1();
    assert_allclose(
        &[df_dx.val, df_dx.d1val1, df_dx.d1val2],
        &[expected_dx, expected_dxx, expected_dxy],
        TOL,
        TOL,
    );
    let df_dy = f.differentiate_2();
    assert_allclose(
        &[df_dy.val, df_dy.d1val1, df_dy.d1val2],
        &[expected_dy, expected_dxy, expected_dyy],
        TOL,
        TOL,
    );
}

proptest! {
    #[test]
    fn test_order2_elementary_chain_matches_finite_differences(
        xv in 0.25_f64..0.85_f64,
        yv in 0.25_f64..0.85_f64
    ) {
        let eval_hyper = |x: AutoDiffReal2VarOrder2, y: AutoDiffReal2VarOrder2| {
            x.sinpi() * y.cospi()
                + x.safe_sqrt() * y.log1p()
                + x.exp10() / (y.cosh() + 1.0)
                + x.atanh() * y.asin()
        };

        let z = eval_hyper(
            AutoDiffReal2VarOrder2::var1(xv),
            AutoDiffReal2VarOrder2::var2(yv),
        );
        let h = 1.0e-4;
        let f00 = z.val;
        let fpx = eval_hyper(
            AutoDiffReal2VarOrder2::constant(xv + h),
            AutoDiffReal2VarOrder2::constant(yv),
        ).val;
        let fmx = eval_hyper(
            AutoDiffReal2VarOrder2::constant(xv - h),
            AutoDiffReal2VarOrder2::constant(yv),
        ).val;
        let fpy = eval_hyper(
            AutoDiffReal2VarOrder2::constant(xv),
            AutoDiffReal2VarOrder2::constant(yv + h),
        ).val;
        let fmy = eval_hyper(
            AutoDiffReal2VarOrder2::constant(xv),
            AutoDiffReal2VarOrder2::constant(yv - h),
        ).val;

        let fd_dx = (fpx - fmx) / (2.0 * h);
        let fd_dy = (fpy - fmy) / (2.0 * h);
        let fd_dxx = (fpx - 2.0 * f00 + fmx) / (h * h);
        let fd_dyy = (fpy - 2.0 * f00 + fmy) / (h * h);

        prop_assert!((z.d1val1 - fd_dx).abs() < 1.0e-6, "dx: {} vs {}", z.d1val1, fd_dx);
        prop_assert!((z.d1val2 - fd_dy).abs() < 1.0e-6, "dy: {} vs {}", z.d1val2, fd_dy);
        prop_assert!((z.d2val1 - fd_dxx).abs() < 1.0e-5, "dxx: {} vs {}", z.d2val1, fd_dxx);
        prop_assert!((z.d2val2 - fd_dyy).abs() < 1.0e-5, "dyy: {} vs {}", z.d2val2, fd_dyy);
    }
}
