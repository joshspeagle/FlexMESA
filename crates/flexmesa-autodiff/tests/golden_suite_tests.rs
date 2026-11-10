// SPDX-License-Identifier: LGPL-3.0-or-later
//! Complete upstream MESA `auto_diff/test/test_output` golden parity suite.
//!
//! Executes every `assignment`, `comparison`, `binary`, `unary`, and
//! `chain_rule` case from `auto_diff/test/src/test_auto_diff.f90` across
//! `AutoDiffReal1VarOrder1`, `AutoDiffReal2VarOrder1`,
//! `AutoDiffReal2VarOrder2`, and `AutoDiffRealStarOrder1`, and verifies all
//! 209 numeric records and 42 boolean records against
//! `.mesa_temp_cache/mesa_root/auto_diff/test/test_output`.

#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_autodiff::{
    AutoDiffReal1VarOrder1, AutoDiffReal2VarOrder1, AutoDiffReal2VarOrder2,
    AutoDiffRealStarOrder1,
};
use flexmesa_const::{LN10, PI};
use flexmesa_math::{
    acos, acosh, asin, asinh, atan, atanh, cos, cosh, exp, log, pow2, sin,
    sinh, tan, tanh,
};
use flexmesa_testing::assert_values_match;

const TOL: f64 = 1.0e-14;

fn parse_fortran_f64(tok: &str) -> f64 {
    tok.replace('D', "E")
        .replace('d', "e")
        .parse::<f64>()
        .expect("valid Fortran float")
}

type ParsedAutoDiffOutput = (Vec<(Vec<f64>, Vec<f64>)>, Vec<(bool, bool)>);

/// Parse all `(expected_vec, actual_vec)` numeric lines and `(expected_b, actual_b)`
/// boolean lines from upstream `auto_diff/test/test_output`.
fn parse_upstream_auto_diff_output(text: &str) -> ParsedAutoDiffOutput {
    let mut numeric = Vec::new();
    let mut booleans = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("Should print T:") {
            booleans.push((true, rest.trim() == "T"));
        } else if let Some(rest) = line.strip_prefix("Should print F:") {
            booleans.push((false, rest.trim() == "T"));
        } else if line.starts_with("Should print") && line.contains(':') {
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            let lhs = parts[0]
                .trim_start_matches("Should print approximately")
                .trim_start_matches("Should print")
                .trim();
            let rhs = parts[1].trim();
            let exp_vals: Vec<f64> =
                lhs.split_whitespace().map(parse_fortran_f64).collect();
            let act_vals: Vec<f64> =
                rhs.split_whitespace().map(parse_fortran_f64).collect();
            numeric.push((exp_vals, act_vals));
        }
    }
    (numeric, booleans)
}

/// Generate the exact sequence of `(comparison_booleans, numeric_records)`
/// emitted by `do_test_auto_diff_1var_order1` in `test_auto_diff.f90`.
fn run_1var_order1_suite() -> (Vec<bool>, Vec<Vec<f64>>) {
    let mut bools = Vec::new();
    let mut nums = Vec::new();
    let rec = |z: AutoDiffReal1VarOrder1| vec![z.val, z.d1[0]];

    // Assignment and comparison (lines 126-218 of test_auto_diff.f90)
    let mut x = AutoDiffReal1VarOrder1::constant(3.1);
    nums.push(rec(x));
    let mut y = AutoDiffReal1VarOrder1::constant(3.1);
    bools.push(x == y);
    y = AutoDiffReal1VarOrder1::constant(2.9);
    bools.extend([x > y, x < y, x >= y, x <= y]);
    y = AutoDiffReal1VarOrder1::constant(3.1);
    bools.extend([x >= y, x <= y]);
    bools.extend([
        x == 3.1,
        x > 2.9,
        x < 2.9,
        x >= 2.9,
        x <= 2.9,
        x >= 3.1,
        x <= 3.1,
    ]);
    x = AutoDiffReal1VarOrder1::constant(3.0);
    bools.extend([
        x == 3.0,
        x > 2.0,
        x < 2.0,
        x >= 3.0,
        x <= 2.0,
        x >= 2.0,
        x <= 3.0,
    ]);

    // Binary operators (lines 220-315)
    let x1 = AutoDiffReal1VarOrder1::new(1.0, [1.0]);
    let y2 = AutoDiffReal1VarOrder1::new(2.0, [1.0]);
    nums.extend([rec(x1 + y2), rec(x1 + 2.0), rec(2.0 + x1)]);

    let x3 = AutoDiffReal1VarOrder1::new(3.0, [1.0]);
    nums.extend([rec(x3 * y2), rec(x3 * 2.0), rec(2.0 * x3)]);

    let y2_d2 = AutoDiffReal1VarOrder1::new(2.0, [2.0]);
    nums.extend([
        rec(x1.max(y2_d2)),
        rec(x1.max_scalar(2.0)),
        rec(x1.max_scalar(0.0)),
        rec(x1.max_scalar(2.0)),
        rec(x1.max_scalar(0.0)),
    ]);

    let x2_d2 = AutoDiffReal1VarOrder1::new(2.0, [2.0]);
    nums.extend([
        rec(x2_d2.dim(x1)),
        rec(x1.dim(x2_d2)),
        rec(x2_d2.dim_scalar(1.0)),
    ]);

    // Unary operators (lines 316-533)
    let u = |v: f64| AutoDiffReal1VarOrder1::new(v, [1.0]);
    nums.extend([
        rec(u(1.0).exp()),
        rec(u(1.0).abs()),
        rec(u(-1.0).abs()),
        rec(u(2.0).exp()),
        rec(u(PI).sin()),
        rec(u(PI).cos()),
        rec(u(PI).tan()),
        rec(u(PI).sinh()),
        rec(u(PI).cosh()),
        rec(u(PI).tanh()),
        rec(u(0.5).asin()),
        rec(u(0.5).acos()),
        rec(u(0.5).atan()),
        rec(u(2.0).asinh()),
        rec(u(2.0).acosh()),
        rec(u(0.5).atanh()),
        rec(-u(1.0)),
        rec(-u(-1.0)),
        rec(u(1.0).log()),
        rec(u(1.0).safe_log()),
        rec(u(10.0).log10()),
        rec(u(10.0).safe_log10()),
        rec(u(exp(1.0)).log()),
        rec(u(exp(1.0)).safe_log()),
        rec(u(100.0).log10()),
        rec(u(100.0).safe_log10()),
        rec(u(3.0).pow2()),
        rec(u(2.0).pow3()),
        rec(u(2.0).pow4()),
        rec(u(3.0).pow5()),
        rec(u(2.0).pow6()),
        rec(u(2.0).pow7()),
        rec(u(4.0).powf(0.5)),
        rec(u(2.0).rpow(exp(1.0))),
        rec(u(4.0).powi(2)),
        rec(u(2.0).rpow(2.0)),
    ]);

    (bools, nums)
}

/// Generate the exact sequence of `(comparison_booleans, numeric_records)`
/// emitted by `do_test_auto_diff_2var_order1` in `test_auto_diff.f90`.
fn run_2var_order1_suite() -> (Vec<bool>, Vec<Vec<f64>>) {
    let mut bools = Vec::new();
    let mut nums = Vec::new();
    let rec = |z: AutoDiffReal2VarOrder1| vec![z.val, z.d1[0], z.d1[1]];

    // Assignment and comparison (lines 539-631)
    let mut x = AutoDiffReal2VarOrder1::constant(3.1);
    nums.push(rec(x));
    let mut y = AutoDiffReal2VarOrder1::constant(3.1);
    bools.push(x == y);
    y = AutoDiffReal2VarOrder1::constant(2.9);
    bools.extend([x > y, x < y, x >= y, x <= y]);
    y = AutoDiffReal2VarOrder1::constant(3.1);
    bools.extend([x >= y, x <= y]);
    bools.extend([
        x == 3.1,
        x > 2.9,
        x < 2.9,
        x >= 2.9,
        x <= 2.9,
        x >= 3.1,
        x <= 3.1,
    ]);
    x = AutoDiffReal2VarOrder1::constant(3.0);
    bools.extend([
        x == 3.0,
        x > 2.0,
        x < 2.0,
        x >= 3.0,
        x <= 2.0,
        x >= 2.0,
        x <= 3.0,
    ]);

    // Binary operators (lines 633-815)
    let x1 = AutoDiffReal2VarOrder1::new(1.0, [1.0, 0.0]);
    let y2_v1 = AutoDiffReal2VarOrder1::new(2.0, [1.0, 0.0]);
    nums.extend([
        rec(x1 + y2_v1),
        rec(x1 + 2.0),
        rec(2.0 + x1),
        rec(x1 + 2.0),
        rec(2.0 + x1),
    ]);

    let x3 = AutoDiffReal2VarOrder1::new(3.0, [1.0, 0.0]);
    let y2_v2 = AutoDiffReal2VarOrder1::new(2.0, [0.0, 1.0]);
    nums.extend([rec(x3 * y2_v2), rec(x3 * 2.0), rec(x3 / y2_v2), rec(x3 / 2.0)]);

    let y5_v2 = AutoDiffReal2VarOrder1::new(5.0, [0.0, 2.0]);
    let x2_v1 = AutoDiffReal2VarOrder1::new(2.0, [1.0, 0.0]);
    nums.extend([
        rec(x1 - y2_v2),
        rec(x1 - y5_v2),
        rec(x2_v1 - 1.0),
        rec(2.0 * x3),
    ]);

    let y2_d2 = AutoDiffReal2VarOrder1::new(2.0, [0.0, 2.0]);
    nums.extend([
        rec(x1.max(y2_d2)),
        rec(x1.max_scalar(2.0)),
        rec(x1.max_scalar(0.0)),
        rec(x1.max_scalar(2.0)),
        rec(x1.max_scalar(0.0)),
        rec(x1.min(y2_d2)),
        rec(x1.min_scalar(2.0)),
        rec(x1.min_scalar(0.0)),
        rec(x1.min_scalar(2.0)),
        rec(x1.min_scalar(0.0)),
    ]);

    let x2_d2 = AutoDiffReal2VarOrder1::new(2.0, [2.0, 0.0]);
    let y1_v2 = AutoDiffReal2VarOrder1::new(1.0, [0.0, 1.0]);
    nums.extend([
        rec(x2_d2.dim(y1_v2)),
        rec(y1_v2.dim(x2_d2)),
        rec(x2_d2.dim_scalar(1.0)),
        rec(x2_v1.pow(y2_v2)),
    ]);

    // Unary operators (lines 816-1039)
    let u1 = |v: f64| AutoDiffReal2VarOrder1::new(v, [1.0, 0.0]);
    let u2 = |v: f64| AutoDiffReal2VarOrder1::new(v, [0.0, 1.0]);
    nums.extend([
        rec(u1(1.0).exp()),
        rec(u2(1.0).exp()),
        rec(u1(2.0).exp()),
        rec(u1(1.0).abs()),
        rec(u1(-1.0).abs()),
        rec(u1(PI).sin()),
        rec(u1(PI).cos()),
        rec(u1(PI).tan()),
        rec(u1(PI).sinh()),
        rec(u1(PI).cosh()),
        rec(u1(PI).tanh()),
        rec(u1(0.5).asin()),
        rec(u1(0.5).acos()),
        rec(u1(0.5).atan()),
        rec(u1(2.0).asinh()),
        rec(u1(2.0).acosh()),
        rec(u1(0.5).atanh()),
        rec(-u1(1.0)),
        rec(-u1(-1.0)),
        rec(u1(1.0).log()),
        rec(u1(1.0).safe_log()),
        rec(u1(10.0).log10()),
        rec(u1(10.0).safe_log10()),
        rec(u1(exp(1.0)).log()),
        rec(u1(exp(1.0)).safe_log()),
        rec(u1(100.0).log10()),
        rec(u1(100.0).safe_log10()),
        rec(u1(3.0).pow2()),
        rec(u1(2.0).pow3()),
        rec(u1(2.0).pow4()),
        rec(u1(3.0).pow5()),
        rec(u1(2.0).pow6()),
        rec(u1(2.0).pow7()),
        rec(u1(4.0).powf(0.5)),
        rec(u1(2.0).rpow(exp(1.0))),
        rec(u1(4.0).powi(2)),
        rec(u1(2.0).rpow(2.0)),
    ]);

    (bools, nums)
}

/// Generate the exact sequence of 90 scalar derivative records emitted by
/// `do_test_auto_diff_star_order1` in `test_auto_diff.f90`.
fn run_star_order1_suite() -> Vec<Vec<f64>> {
    let mut nums = Vec::new();

    // Assignment: x = 3.0, x%d1Array(4) = 1.0 (1-indexed -> index 3)
    let x_assign = AutoDiffRealStarOrder1::independent(3.0, 3);
    for i in 0..15 {
        nums.push(vec![x_assign.d1[i]]);
    }

    // Unary: y = pow2(x), then x = pow2(x)
    let mut d1_seq = [0.0; 33];
    for (i, slot) in d1_seq.iter_mut().enumerate().take(15) {
        *slot = (i + 1) as f64;
    }
    let x_base = AutoDiffRealStarOrder1::new(3.0, d1_seq);
    let y_pow2 = x_base.pow2();
    for i in 0..15 {
        nums.push(vec![y_pow2.d1[i]]);
    }
    let x_pow2 = x_base.pow2();
    for i in 0..15 {
        nums.push(vec![x_pow2.d1[i]]);
    }

    // Binary: y = x + 1.0, y = x + x, y = exp(x) * z
    let y_add1 = x_base + 1.0;
    for i in 0..15 {
        nums.push(vec![y_add1.d1[i]]);
    }
    let y_addx = x_base + x_base;
    for i in 0..15 {
        nums.push(vec![y_addx.d1[i]]);
    }
    let mut z_d1 = [0.0; 33];
    for (i, slot) in z_d1.iter_mut().enumerate().take(15) {
        *slot = 1.0 - (i + 1) as f64;
    }
    let z = AutoDiffRealStarOrder1::new(2.0, z_d1);
    let y_chain = x_base.exp() * z;
    for i in 0..15 {
        nums.push(vec![y_chain.d1[i]]);
    }

    nums
}

#[test]
fn test_auto_diff_full_golden_test_output_parity() {
    let (bools_1v, nums_1v) = run_1var_order1_suite();
    let (bools_2v, nums_2v) = run_2var_order1_suite();
    let nums_star = run_star_order1_suite();

    let mut all_bools = Vec::new();
    all_bools.extend(bools_1v);
    all_bools.extend(bools_2v);
    let expected_bools_21 = [
        true, true, false, true, false, true, true, true, true, false, true,
        false, true, true, true, true, false, true, false, true, true,
    ];
    let mut expected_bools = Vec::with_capacity(42);
    expected_bools.extend_from_slice(&expected_bools_21);
    expected_bools.extend_from_slice(&expected_bools_21);
    assert_eq!(all_bools, expected_bools);

    let mut all_nums = Vec::new();
    all_nums.extend(nums_1v);
    all_nums.extend(nums_2v);
    all_nums.extend(nums_star);
    assert_eq!(all_nums.len(), 206);

    let pinned_1v_2v: &[(usize, &[f64])] = &[
        (0, &[3.1, 0.0]),
        (1, &[3.0, 2.0]),
        (4, &[6.0, 5.0]),
        (12, &[1.0, 1.0]),
        (15, &[exp(1.0), exp(1.0)]),
        (50, &[4.0, 4.0 * log(2.0)]),
        (51, &[3.1, 0.0, 0.0]),
        (57, &[6.0, 2.0, 3.0]),
        (59, &[1.5, 0.5, -0.75]),
        (78, &[4.0, 4.0, 4.0 * log(2.0)]),
    ];
    for &(idx, expected) in pinned_1v_2v {
        assert_values_match(
            &all_nums[idx],
            expected,
            TOL,
            TOL,
            &format!("unconditional golden record {idx}"),
        )
        .expect("match pinned golden record");
    }
    for i in 0..15 {
        let exp_assign = if i == 3 { 1.0 } else { 0.0 };
        let exp_pow2 = 6.0 * ((i + 1) as f64);
        let exp_chain = exp(3.0) * ((i + 2) as f64);
        assert_values_match(
            &all_nums[116 + i],
            &[exp_assign],
            TOL,
            TOL,
            "star_order1 assign",
        )
        .unwrap();
        assert_values_match(
            &all_nums[131 + i],
            &[exp_pow2],
            TOL,
            TOL,
            "star_order1 pow2",
        )
        .unwrap();
        assert_values_match(
            &all_nums[191 + i],
            &[exp_chain],
            TOL,
            TOL,
            "star_order1 chain",
        )
        .unwrap();
    }

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_path = manifest_dir
        .join("../../.mesa_temp_cache/mesa_root/auto_diff/test/test_output");
    if output_path.exists() {
        let text = std::fs::read_to_string(&output_path)
            .expect("read auto_diff/test/test_output");
        let (golden_nums, golden_bools) = parse_upstream_auto_diff_output(&text);
        assert_eq!(golden_bools.len(), all_bools.len());
        for (idx, (&actual_b, &(exp_b, ref_b))) in
            all_bools.iter().zip(golden_bools.iter()).enumerate()
        {
            assert_eq!(actual_b, exp_b, "boolean mismatch at index {idx}");
            assert_eq!(actual_b, ref_b, "reference boolean mismatch at {idx}");
        }

        assert_eq!(golden_nums.len(), all_nums.len());
        for (idx, (actual_vec, (exp_vec, ref_vec))) in
            all_nums.iter().zip(golden_nums.iter()).enumerate()
        {
            assert_values_match(
                actual_vec,
                exp_vec,
                TOL,
                TOL,
                &format!("auto_diff golden expected line {idx}"),
            )
            .expect("match expected column");
            assert_values_match(
                actual_vec,
                ref_vec,
                TOL,
                TOL,
                &format!("auto_diff golden actual line {idx}"),
            )
            .expect("match reference column");
        }
    }
}

/// Complete operator sweep (`comparison`, `binary`, `unary`, `chain_rule`) on
/// `AutoDiffReal2VarOrder2` (`Order2Var2`) verifying primal, gradient, and
/// Hessian components against exact analytical formulas and `Order1Var2`.
#[test]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn test_auto_diff_2var_order2_complete_operator_and_chain_rule_sweep() {
    let x_cmp = AutoDiffReal2VarOrder2::constant(3.1);
    let y_cmp = AutoDiffReal2VarOrder2::constant(2.9);
    assert_eq!(x_cmp, AutoDiffReal2VarOrder2::constant(3.1));
    assert!(x_cmp > y_cmp && !(x_cmp < y_cmp));
    assert!(x_cmp >= y_cmp && !(x_cmp <= y_cmp));
    assert!(x_cmp == 3.1 && x_cmp > 2.9 && x_cmp <= 3.1);

    // Binary operator parity between Order2Var2 and Order1Var2
    let x = AutoDiffReal2VarOrder2::var1(3.0);
    let y = AutoDiffReal2VarOrder2::var2(2.0);
    let z_mul = x * y;
    assert_values_match(
        &[
            z_mul.val,
            z_mul.d1val1,
            z_mul.d1val2,
            z_mul.d2val1,
            z_mul.d1val1_d1val2,
            z_mul.d2val2,
        ],
        &[6.0, 2.0, 3.0, 0.0, 1.0, 0.0],
        TOL,
        TOL,
        "Order2Var2 mul",
    )
    .unwrap();

    let z_div = x / y;
    assert_values_match(
        &[
            z_div.val,
            z_div.d1val1,
            z_div.d1val2,
            z_div.d2val1,
            z_div.d1val1_d1val2,
            z_div.d2val2,
        ],
        &[1.5, 0.5, -0.75, 0.0, -0.25, 0.75],
        TOL,
        TOL,
        "Order2Var2 div",
    )
    .unwrap();

    let x_p = AutoDiffReal2VarOrder2::var1(2.0);
    let y_p = AutoDiffReal2VarOrder2::var2(2.0);
    let z_pow = x_p.pow(y_p);
    let ln2 = log(2.0);
    assert_values_match(
        &[
            z_pow.val,
            z_pow.d1val1,
            z_pow.d1val2,
            z_pow.d2val1,
            z_pow.d1val1_d1val2,
            z_pow.d2val2,
        ],
        &[4.0, 4.0, 4.0 * ln2, 2.0, 2.0 + 4.0 * ln2, 4.0 * ln2 * ln2],
        TOL,
        TOL,
        "Order2Var2 pow",
    )
    .unwrap();

    // Unary operator sweep on Order2Var2 matching Order1Var2 values and 1st partials
    let u2 = |v: f64| AutoDiffReal2VarOrder2::var1(v);
    let cases = [
        (u2(1.0).exp(), exp(1.0), exp(1.0)),
        (u2(PI).sin(), sin(PI), cos(PI)),
        (u2(PI).cos(), cos(PI), -sin(PI)),
        (u2(PI).tan(), tan(PI), 1.0 / pow2(cos(PI))),
        (u2(PI).sinh(), sinh(PI), cosh(PI)),
        (u2(PI).cosh(), cosh(PI), sinh(PI)),
        (u2(PI).tanh(), tanh(PI), 1.0 / pow2(cosh(PI))),
        (u2(0.5).asin(), asin(0.5), 2.0 / 3.0_f64.sqrt()),
        (u2(0.5).acos(), acos(0.5), -2.0 / 3.0_f64.sqrt()),
        (u2(0.5).atan(), atan(0.5), 0.8),
        (u2(2.0).asinh(), asinh(2.0), 1.0 / 5.0_f64.sqrt()),
        (u2(2.0).acosh(), acosh(2.0), 1.0 / 3.0_f64.sqrt()),
        (u2(0.5).atanh(), atanh(0.5), 4.0 / 3.0),
        (u2(10.0).log10(), 1.0, 1.0 / (10.0 * LN10)),
        (u2(3.0).pow5(), 243.0, 405.0),
    ];
    for (z, ev, ed1) in cases {
        assert_values_match(
            &[z.val, z.d1val1],
            &[ev, ed1],
            TOL,
            TOL,
            "Order2Var2 unary",
        )
        .unwrap();
    }

    // Chain rule composition f(g(x, y), h(x, y)) across Order1Var1, Order1Var2, Order2Var2
    let xv = 0.6_f64;
    let yv = 1.1_f64;
    let x1 = AutoDiffReal1VarOrder1::independent(xv, 0);
    let chain_1v = (x1.pow2() + x1.sin()).exp();
    let x2 = AutoDiffReal2VarOrder1::var1(xv);
    let y2 = AutoDiffReal2VarOrder1::var2(yv);
    let chain_2v = (x2.pow2() + x2.sin()).exp() * y2.cos();
    let x22 = AutoDiffReal2VarOrder2::var1(xv);
    let y22 = AutoDiffReal2VarOrder2::var2(yv);
    let chain_22 = (x22.pow2() + x22.sin()).exp() * y22.cos();

    assert_values_match(
        &[chain_2v.val, chain_2v.d1[0], chain_2v.d1[1]],
        &[chain_22.val, chain_22.d1val1, chain_22.d1val2],
        TOL,
        TOL,
        "Order1Var2 vs Order2Var2 chain rule",
    )
    .unwrap();
    assert_values_match(
        &[chain_1v.val * cos(yv), chain_1v.d1[0] * cos(yv)],
        &[chain_22.val, chain_22.d1val1],
        TOL,
        TOL,
        "Order1Var1 vs Order2Var2 chain rule",
    )
    .unwrap();
}
