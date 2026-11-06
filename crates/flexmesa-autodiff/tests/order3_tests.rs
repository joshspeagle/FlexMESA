// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical oracle tests for [`flexmesa_autodiff::AutoDiffReal2VarOrder3`].

#![allow(clippy::similar_names)]

use flexmesa_autodiff::AutoDiffReal2VarOrder3;

fn assert_close(actual: f64, expected: f64, tol: f64) {
    let scale = 1.0_f64.max(expected.abs());
    let err = (actual - expected).abs() / scale;
    assert!(
        err <= tol,
        "actual={actual:.16e}, expected={expected:.16e}, rel={err:.3e}"
    );
}

#[test]
fn test_order3_polynomial_and_differentiate_operators_match_exact_partials() {
    // f(x, y) = x^3 * y^2 + 2.0 * x^2 * y^3
    let x0 = 1.7_f64;
    let y0 = 2.3_f64;
    let x = AutoDiffReal2VarOrder3::var1(x0);
    let y = AutoDiffReal2VarOrder3::var2(y0);
    let f = x.pow3() * y.pow2() + 2.0 * x.pow2() * y.pow3();

    let val = x0.powi(3) * y0.powi(2) + 2.0 * x0.powi(2) * y0.powi(3);
    let fx = 3.0 * x0.powi(2) * y0.powi(2) + 4.0 * x0 * y0.powi(3);
    let fy = 2.0 * x0.powi(3) * y0 + 6.0 * x0.powi(2) * y0.powi(2);
    let fxx = 6.0 * x0 * y0.powi(2) + 4.0 * y0.powi(3);
    let fxy = 6.0 * x0.powi(2) * y0 + 12.0 * x0 * y0.powi(2);
    let fyy = 2.0 * x0.powi(3) + 12.0 * x0.powi(2) * y0;
    let fxxx = 6.0 * y0.powi(2);
    let fxxy = 12.0 * x0 * y0 + 12.0 * y0.powi(2);
    let fxyy = 6.0 * x0.powi(2) + 24.0 * x0 * y0;
    let fyyy = 12.0 * x0.powi(2);

    assert_close(f.val, val, 1e-15);
    assert_close(f.d1val1, fx, 1e-15);
    assert_close(f.d1val2, fy, 1e-15);
    assert_close(f.d2val1, fxx, 1e-15);
    assert_close(f.d1val1_d1val2, fxy, 1e-15);
    assert_close(f.d2val2, fyy, 1e-15);
    assert_close(f.d3val1, fxxx, 1e-15);
    assert_close(f.d2val1_d1val2, fxxy, 1e-15);
    assert_close(f.d1val1_d2val2, fxyy, 1e-15);
    assert_close(f.d3val2, fyyy, 1e-15);

    let df_dx = f.differentiate_1();
    assert_close(df_dx.val, fx, 1e-15);
    assert_close(df_dx.d1val1, fxx, 1e-15);
    assert_close(df_dx.d1val2, fxy, 1e-15);
    assert_close(df_dx.d2val1, fxxx, 1e-15);
    assert_close(df_dx.d1val1_d1val2, fxxy, 1e-15);
    assert_close(df_dx.d2val2, fxyy, 1e-15);

    let df_dy = f.differentiate_2();
    assert_close(df_dy.val, fy, 1e-15);
    assert_close(df_dy.d1val1, fxy, 1e-15);
    assert_close(df_dy.d1val2, fyy, 1e-15);
    assert_close(df_dy.d2val1, fxxy, 1e-15);
    assert_close(df_dy.d1val1_d1val2, fxyy, 1e-15);
    assert_close(df_dy.d2val2, fyyy, 1e-15);
}

#[test]
fn test_order3_transcendental_quotient_and_pow_match_analytic_oracle() {
    // f(x, y) = exp(0.5 * x) * ln(y) / sqrt(x)
    // Test via product of exp(0.5 * x - 0.5 * ln(x)) * ln(y) versus direct
    // quotient and sqrt.
    let x0 = 1.4_f64;
    let y0 = 2.1_f64;
    let x = AutoDiffReal2VarOrder3::var1(x0);
    let y = AutoDiffReal2VarOrder3::var2(y0);

    let f_direct = (x * 0.5).exp() * y.log() / x.sqrt();
    let f_equiv = (x * 0.5 - 0.5 * x.log()).exp() * y.log();

    assert_close(f_direct.val, f_equiv.val, 1e-14);
    assert_close(f_direct.d1val1, f_equiv.d1val1, 1e-14);
    assert_close(f_direct.d1val2, f_equiv.d1val2, 1e-14);
    assert_close(f_direct.d2val1, f_equiv.d2val1, 1e-14);
    assert_close(f_direct.d1val1_d1val2, f_equiv.d1val1_d1val2, 1e-14);
    assert_close(f_direct.d2val2, f_equiv.d2val2, 1e-14);
    assert_close(f_direct.d3val1, f_equiv.d3val1, 1e-14);
    assert_close(f_direct.d2val1_d1val2, f_equiv.d2val1_d1val2, 1e-14);
    assert_close(f_direct.d1val1_d2val2, f_equiv.d1val1_d2val2, 1e-14);
    assert_close(f_direct.d3val2, f_equiv.d3val2, 1e-14);

    // Also verify pow(x, y) against exp(y * ln(x))
    let p_direct = x.pow(y);
    let p_equiv = (y * x.log()).exp();
    assert_close(p_direct.val, p_equiv.val, 1e-14);
    assert_close(p_direct.d1val1, p_equiv.d1val1, 1e-14);
    assert_close(p_direct.d1val2, p_equiv.d1val2, 1e-14);
    assert_close(p_direct.d2val1, p_equiv.d2val1, 1e-14);
    assert_close(p_direct.d1val1_d1val2, p_equiv.d1val1_d1val2, 1e-14);
    assert_close(p_direct.d2val2, p_equiv.d2val2, 1e-14);
    assert_close(p_direct.d3val1, p_equiv.d3val1, 1e-14);
    assert_close(p_direct.d2val1_d1val2, p_equiv.d2val1_d1val2, 1e-14);
    assert_close(p_direct.d1val1_d2val2, p_equiv.d1val1_d2val2, 1e-14);
    assert_close(p_direct.d3val2, p_equiv.d3val2, 1e-14);
}

#[test]
fn test_order3_compose_binary_variable_substitution() {
    // Let F(u, v) = u^2 * v + u * v^2.
    // Substitute u = g1(x, y) = 3.0 * y^2, v = g2(x, y) = y.
    let y0 = 1.8_f64;
    let u = AutoDiffReal2VarOrder3::var1(3.0 * y0 * y0);
    let v = AutoDiffReal2VarOrder3::var2(y0);
    let f_uv = u.pow2() * v + u * v.pow2();

    let y = AutoDiffReal2VarOrder3::var2(y0);
    let g1 = 3.0 * y.pow2();
    let g2 = y;
    let composed = f_uv.compose_binary(g1, g2);
    let direct = g1.pow2() * g2 + g1 * g2.pow2();

    assert_close(composed.val, direct.val, 1e-15);
    assert_close(composed.d1val1, direct.d1val1, 1e-15);
    assert_close(composed.d1val2, direct.d1val2, 1e-15);
    assert_close(composed.d2val2, direct.d2val2, 1e-15);
    assert_close(composed.d3val2, direct.d3val2, 1e-15);
}
