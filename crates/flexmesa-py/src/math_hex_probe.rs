// SPDX-License-Identifier: GPL-3.0-or-later
//! Level-2 exact-hex test-suite probes for `math` and `auto_diff` (`#220`).

use flexmesa_autodiff::{
    AutoDiffReal1VarOrder1, AutoDiffReal2VarOrder1, AutoDiffRealStarOrder1,
    DualOrder1, STAR_ORDER1_VARS,
};
use flexmesa_const::PI;
use pyo3::prelude::*;

#[pyfunction]
fn math_test_suite_eval(x: f64) -> Vec<(&'static str, f64)> {
    vec![
        ("log", flexmesa_math::log(x)),
        ("log10", flexmesa_math::log10(x)),
        ("log1p", flexmesa_math::log1p(x)),
        ("log2", flexmesa_math::log2(x)),
        ("safe_log", flexmesa_math::safe_log(x)),
        ("safe_log10", flexmesa_math::safe_log10(x)),
        ("exp", flexmesa_math::exp(x)),
        ("exp10", flexmesa_math::exp10(x)),
        ("expm1", flexmesa_math::expm1(x)),
        ("pow(,2)", flexmesa_math::powi(x, 2)),
        ("pow(,2.)", flexmesa_math::pow(x, 2.0)),
        ("pow2", flexmesa_math::pow2(x)),
        ("pow3", flexmesa_math::pow3(x)),
        ("pow4", flexmesa_math::pow4(x)),
        ("pow5", flexmesa_math::pow5(x)),
        ("pow6", flexmesa_math::pow6(x)),
        ("pow7", flexmesa_math::pow7(x)),
        ("pow8", flexmesa_math::pow8(x)),
        ("cos", flexmesa_math::cos(x)),
        ("sin", flexmesa_math::sin(x)),
        ("tan", flexmesa_math::tan(x)),
        ("cospi", flexmesa_math::cospi(x)),
        ("sinpi", flexmesa_math::sinpi(x)),
        ("tanpi", flexmesa_math::tanpi(x)),
        ("acos", flexmesa_math::acos(x)),
        ("asin", flexmesa_math::asin(x)),
        ("atan", flexmesa_math::atan(x)),
        ("acospi", flexmesa_math::acospi(x)),
        ("asinpi", flexmesa_math::asinpi(x)),
        ("atanpi", flexmesa_math::atanpi(x)),
        ("cosh", flexmesa_math::cosh(x)),
        ("sinh", flexmesa_math::sinh(x)),
        ("tanh", flexmesa_math::tanh(x)),
    ]
}

fn push_dual<const N: usize>(out: &mut Vec<f64>, z: DualOrder1<N>) {
    out.push(z.val);
    out.extend_from_slice(&z.d1);
}

fn push_common_unary_tail<const N: usize>(out: &mut Vec<f64>) {
    let e = flexmesa_math::exp(1.0);
    let u = |v: f64| DualOrder1::<N>::independent(v, 0);
    for z in [
        u(PI).sin(),
        u(PI).cos(),
        u(PI).tan(),
        u(PI).sinh(),
        u(PI).cosh(),
        u(PI).tanh(),
        u(0.5).asin(),
        u(0.5).acos(),
        u(0.5).atan(),
        u(2.0).asinh(),
        u(2.0).acosh(),
        u(0.5).atanh(),
        -u(1.0),
        -u(-1.0),
        u(1.0).log(),
        u(1.0).safe_log(),
        u(10.0).log10(),
        u(10.0).safe_log10(),
        u(e).log(),
        u(e).safe_log(),
        u(100.0).log10(),
        u(100.0).safe_log10(),
        u(3.0).pow2(),
        u(2.0).pow3(),
        u(2.0).pow4(),
        u(3.0).pow5(),
        u(2.0).pow6(),
        u(2.0).pow7(),
        u(4.0).powf(0.5),
        u(2.0).rpow(e),
        u(4.0).powi(2),
        u(2.0).rpow(2.0),
    ] {
        push_dual(out, z);
    }
}

fn push_1var_order1_test_values(out: &mut Vec<f64>) {
    type D1 = AutoDiffReal1VarOrder1;
    let u = |v: f64| D1::independent(v, 0);

    push_dual(out, D1::constant(3.1));

    let x1 = u(1.0);
    let y2 = u(2.0);
    let x3 = u(3.0);
    let y2_d2 = D1::new(2.0, [2.0]);
    for z in [
        x1 + y2,
        x1 + 2.0,
        2.0 + x1,
        x3 * y2,
        x3 * 2.0,
        2.0 * x3,
        x1.max(y2_d2),
        x1.max_scalar(2.0),
        x1.max_scalar(0.0),
        x1.max_scalar(2.0),
        x1.max_scalar(0.0),
        y2_d2.dim(x1),
        x1.dim(y2_d2),
        y2_d2.dim_scalar(1.0),
        u(1.0).exp(),
        u(1.0).abs(),
        u(-1.0).abs(),
        u(2.0).exp(),
    ] {
        push_dual(out, z);
    }
    push_common_unary_tail::<1>(out);
}

fn push_2var_order1_binary_values(out: &mut Vec<f64>) {
    type D2 = AutoDiffReal2VarOrder1;
    push_dual(out, D2::constant(3.1));

    let x1 = D2::var1(1.0);
    let y2_v1 = D2::var1(2.0);
    let x3 = D2::var1(3.0);
    let y2_v2 = D2::var2(2.0);
    let y5_v2 = D2::new(5.0, [0.0, 2.0]);
    let y2_d2 = D2::new(2.0, [0.0, 2.0]);
    let x2_d2 = D2::new(2.0, [2.0, 0.0]);
    let y1_v2 = D2::var2(1.0);
    for z in [
        x1 + y2_v1,
        x1 + 2.0,
        2.0 + x1,
        x1 + 2.0,
        2.0 + x1,
        x3 * y2_v2,
        x3 * 2.0,
        x3 / y2_v2,
        x3 / 2.0,
        x1 - y2_v2,
        x1 - y5_v2,
        y2_v1 - 1.0,
        2.0 * x3,
        x1.max(y2_d2),
        x1.max_scalar(2.0),
        x1.max_scalar(0.0),
        x1.max_scalar(2.0),
        x1.max_scalar(0.0),
        x1.min(y2_d2),
        x1.min_scalar(2.0),
        x1.min_scalar(0.0),
        x1.min_scalar(2.0),
        x1.min_scalar(0.0),
        x2_d2.dim(y1_v2),
        y1_v2.dim(x2_d2),
        x2_d2.dim_scalar(1.0),
        y2_v1.pow(y2_v2),
    ] {
        push_dual(out, z);
    }
}

fn push_2var_order1_test_values(out: &mut Vec<f64>) {
    type D2 = AutoDiffReal2VarOrder1;
    push_2var_order1_binary_values(out);
    for z in [
        D2::var1(1.0).exp(),
        D2::var2(1.0).exp(),
        D2::var1(2.0).exp(),
        D2::var1(1.0).abs(),
        D2::var1(-1.0).abs(),
    ] {
        push_dual(out, z);
    }
    push_common_unary_tail::<2>(out);
}

fn push_star_order1_test_values(out: &mut Vec<f64>) {
    type DS = AutoDiffRealStarOrder1;

    let x_assign = DS::independent(3.0, 3);
    out.extend_from_slice(&x_assign.d1[..15]);

    let mut d1_seq = [0.0; STAR_ORDER1_VARS];
    for (i, slot) in d1_seq.iter_mut().enumerate().take(15) {
        *slot = (i + 1) as f64;
    }
    let x_base = DS::new(3.0, d1_seq);
    let y_pow2 = x_base.pow2();
    out.extend_from_slice(&y_pow2.d1[..15]);
    out.extend_from_slice(&y_pow2.d1[..15]);

    let y_add1 = x_base + 1.0;
    out.extend_from_slice(&y_add1.d1[..15]);
    let y_addx = x_base + x_base;
    out.extend_from_slice(&y_addx.d1[..15]);

    let mut z_d1 = [0.0; STAR_ORDER1_VARS];
    for (i, slot) in z_d1.iter_mut().enumerate().take(15) {
        // Keep `1.0 - (i + 1) as f64` so `i = 0` emits `+0.0` rather than `-0.0`.
        *slot = 1.0 - (i + 1) as f64;
    }
    let z = DS::new(2.0, z_d1);
    let y_chain = x_base.exp() * z;
    out.extend_from_slice(&y_chain.d1[..15]);
}

#[pyfunction]
fn auto_diff_test_suite_values() -> Vec<f64> {
    let mut out = Vec::with_capacity(387);
    push_1var_order1_test_values(&mut out);
    push_2var_order1_test_values(&mut out);
    push_star_order1_test_values(&mut out);
    out
}

/// Registers Level-2 exact-hex test-suite probes for `math` and `auto_diff`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(math_test_suite_eval, m)?)?;
    m.add_function(wrap_pyfunction!(auto_diff_test_suite_values, m)?)?;
    Ok(())
}
