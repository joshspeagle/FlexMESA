//! Integration and golden-baseline tests for `flexmesa-interp1d`.

#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_interp1d::{
    integrate_values, interp_3_to_1, interp_3_to_2, interp_4_to_1,
    interp_4pt_pm, interp_m3a, interp_m3b, interp_m3q, interp_pm,
    interp_pm_slopes_only, interp_value_and_slope, interp_values,
    interp_values_and_slopes, interpolate_vector_pm,
    interpolate_vector_pm_autodiff,
};
use flexmesa_testing::assert_values_match;
use proptest::prelude::*;
use std::f64::consts::PI;

fn assert_close(actual: f64, expected: f64, ctx: &str) {
    assert_values_match(&[actual], &[expected], 1.0e-12, 1.0e-14, ctx)
        .expect(ctx);
}

fn assert_allclose(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_values_match(actual, expected, 1.0e-12, 1.0e-14, ctx).expect(ctx);
}

const TEST_U: [f64; 100] = [
    0.3691382918814334,
    0.3695899829574669,
    0.3699633032242006,
    0.3709860746618878,
    0.3724175838188959,
    0.3742482756438662,
    0.3764768199547272,
    0.3790910321656262,
    0.3820871425096383,
    0.3854500131142883,
    0.3891727128876379,
    0.3932377816410259,
    0.3976355585696444,
    0.4023461172079230,
    0.4073570798427897,
    0.4126461990878889,
    0.4181985492965749,
    0.4239897218739848,
    0.4300024071530404,
    0.4362102270543894,
    0.4425936978527515,
    0.4491247203356548,
    0.4557819260544076,
    0.4625358769752868,
    0.4693638888826394,
    0.4762362700345437,
    0.4831315999154387,
    0.4900124769496845,
    0.4968509249135122,
    0.5036231656397859,
    0.5102978108992456,
    0.5168503329341820,
    0.5232493058916751,
    0.5294708374154181,
    0.5354843215442489,
    0.5412671682413143,
    0.5467901937111218,
    0.5520326792853251,
    0.5569674077513700,
    0.5615760727117703,
    0.5658339305000317,
    0.5697255981615711,
    0.5732292848999179,
    0.5763330504836860,
    0.5790185013283228,
    0.5812774415045446,
    0.5830949206727917,
    0.5844671751637984,
    0.5853828024764719,
    0.5858432670435806,
    0.5858432670435451,
    0.5853828024764413,
    0.5844671751637710,
    0.5830949206727681,
    0.5812774415045240,
    0.5790185013283052,
    0.5763330504836706,
    0.5732292848999045,
    0.5697255981615594,
    0.5658339305000216,
    0.5615760727117616,
    0.5569674077513626,
    0.5520326792853185,
    0.5467901937111161,
    0.5412671682413093,
    0.5354843215442447,
    0.5294708374154145,
    0.5232493058916718,
    0.5168503329341793,
    0.5102978108992431,
    0.5036231656397836,
    0.4968509249135103,
    0.4900124769496828,
    0.4831315999154373,
    0.4762362700345427,
    0.4693638888826384,
    0.4625358769752859,
    0.4557819260544067,
    0.4491247203356541,
    0.4425936978527509,
    0.4362102270543888,
    0.4300024071530398,
    0.4239897218739841,
    0.4181985492965744,
    0.4126461990878886,
    0.4073570798427895,
    0.4023461172079228,
    0.3976355585696443,
    0.3932377816410258,
    0.3891727128876379,
    0.3854500131142883,
    0.3820871425096385,
    0.3790910321656264,
    0.3764768199547278,
    0.3742482756438669,
    0.3724175838188968,
    0.3709860746618888,
    0.3699633032242018,
    0.3695899829574566,
    0.3691382918814347,
];

#[test]
fn test_upstream_mesa_test1_m3q_matches_golden_output() {
    let dx = 0.594059405940594_f64;
    for increasing in [true, false] {
        let sign = if increasing { 1.0 } else { -1.0 };
        let init_xs: Vec<f64> =
            (0..100).map(|i| sign * dx * (i as f64)).collect();
        let coeffs = interp_m3q(&init_xs, &TEST_U).unwrap();

        assert_close(coeffs[0][1], sign * 6.2842244900173001e-4, "m3q f(2,1)");
        assert_close(coeffs[1][1], sign * 6.6140349791541427e-4, "m3q f(2,2)");
        assert_close(coeffs[2][1], sign * 1.1067133492894063e-3, "m3q f(2,3)");

        let xs = [sign * 10.0, sign * 10.4, sign * 10.8, sign * 11.0];
        let vals = interp_values(&init_xs, &coeffs, &xs).unwrap();
        let expected = [
            4.2300860237857557e-1,
            4.2701010015389296e-1,
            4.3110602983550195e-1,
            4.3318662265017888e-1,
        ];
        assert_allclose(&vals, &expected, "m3q interp_values");
    }
}

#[test]
fn test_upstream_mesa_test2_sin_and_integral_match_golden_output() {
    let n = 43usize;
    let dx = 1.0 / ((n - 1) as f64);
    for increasing in [true, false] {
        let sign = if increasing { 1.0 } else { -1.0 };
        let init_xs: Vec<f64> =
            (0..n).map(|i| sign * 4.0 * dx * (i as f64)).collect();
        let u: Vec<f64> = init_xs.iter().map(|&x| x.sin()).collect();
        let coeffs = interp_m3q(&init_xs, &u).unwrap();

        let xs = [
            0.0,
            sign * PI / 4.0,
            sign * PI / 3.0,
            sign * PI / 2.0,
            sign * 2.0 * PI / 3.0,
            sign * 3.0 * PI / 4.0,
            sign * 19.0 * PI / 20.0,
        ];
        let vals = interp_values(&init_xs, &coeffs, &xs).unwrap();
        let expected_vals = [
            0.0,
            sign * 7.0710667730056076e-1,
            sign * 8.6602540429696606e-1,
            sign * 9.9999978275888857e-1,
            sign * 8.6602540242370141e-1,
            sign * 7.0710667151355655e-1,
            sign * 1.5643445870792264e-1,
        ];
        assert_allclose(&vals, &expected_vals, "test2 sin vals");

        let integrals = integrate_values(&init_xs, &coeffs, &xs).unwrap();
        let expected_integrals = [
            0.0,
            2.9288520516627042e-1,
            2.0710675623584285e-1,
            4.9999994192722058e-1,
            4.9999994167149031e-1,
            2.0710675639571818e-1,
            2.8058152844437595e-1,
        ];
        assert_allclose(&integrals, &expected_integrals, "test2 sin integrals");
    }
}

#[test]
fn test_upstream_mesa_test3_and_min_pm_match_golden_output() {
    let dx = 0.594059405940594_f64;
    for increasing in [true, false] {
        let sign = if increasing { 1.0 } else { -1.0 };
        let init_xs: Vec<f64> =
            (0..100).map(|i| sign * dx * (i as f64)).collect();
        let coeffs = interp_pm(&init_xs, &TEST_U).unwrap();

        assert_close(coeffs[0][1], sign * 8.2630874248383526e-4, "pm f(2,1)");
        assert_close(coeffs[1][1], sign * 6.9438454682909843e-4, "pm f(2,2)");
        assert_close(coeffs[2][1], sign * 1.1750438512209234e-3, "pm f(2,3)");

        let xs = [sign * 10.0, sign * 10.4, sign * 10.8];
        let vals = interp_values(&init_xs, &coeffs, &xs).unwrap();
        let expected = [
            4.2300894327454436e-1,
            4.2701013523153308e-1,
            4.3110567262196459e-1,
        ];
        assert_allclose(&vals, &expected, "pm interp_values");

        let slopes_only = interp_pm_slopes_only(&init_xs, &TEST_U).unwrap();
        assert_close(slopes_only[0][1], coeffs[0][1], "slopes_only[0]");
        assert_close(slopes_only[2][1], coeffs[2][1], "slopes_only[2]");
        assert_eq!(slopes_only[0][2], 0.0);
    }
}

#[test]
fn test_upstream_mesa_4pt_and_3_to_2_match_golden_output() {
    let x_full = [
        PI / 4.0,
        PI / 3.0,
        PI / 2.0,
        2.0 * PI / 3.0,
        3.0 * PI / 4.0,
        19.0 * PI / 20.0,
    ];
    let y_full = x_full.map(f64::sin);
    let x4 = [x_full[1], x_full[2], x_full[3], x_full[4]];
    let y4 = [y_full[1], y_full[2], y_full[3], y_full[4]];
    let a = interp_4pt_pm(&x4, &y4).unwrap();
    let dx = 0.5 * (x_full[3] - x_full[2]);
    let result = y_full[2] + dx * (a[0] + dx * (a[1] + dx * a[2]));
    assert_close(result, 9.6508141383418289e-1, "test_4pt");

    let (nf00, nfp1) = interp_3_to_2(
        2.0114947208182310e-3,
        2.0097307373083619e-3,
        1.5784933404892154e-3,
        1.3270582904786616e-3,
        1.8984458478374221e30,
        1.4579738233866260e30,
        1.1136297079131214e30,
    )
    .unwrap();
    assert_close(nf00, 1.5446941380706298e30, "nf00");
    assert_close(nfp1, 1.2929705208875774e30, "nfp1");

    let nf_single = interp_3_to_1(
        2.0114947208182310e-3,
        2.0097307373083619e-3,
        1.5784933404892154e-3,
        1.8984458478374221e30,
        1.4579738233866260e30,
        1.1136297079131214e30,
    )
    .unwrap();
    assert_close(nf_single, nf00, "interp_3_to_1");

    let nf4 =
        interp_4_to_1(1.0, 1.0, 1.0, 1.5, 0.0, 1.0, 8.0, 27.0).unwrap();
    let direct = interpolate_vector_pm(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 8.0, 27.0],
        &[1.5],
    )
    .unwrap()[0];
    assert_close(nf4, direct, "interp_4_to_1");
}

#[test]
fn test_autodiff_interp_pm_matches_analytic_slope_and_parameter_derivative() {
    let x_vals = [0.0, 1.0, 2.0, 3.0, 4.0];
    let scale = AutoDiffReal2VarOrder1::independent(2.5, 1);
    let x_ad: Vec<AutoDiffReal2VarOrder1> = x_vals
        .iter()
        .map(|&xv| AutoDiffReal2VarOrder1::constant(xv))
        .collect();
    let y_ad: Vec<AutoDiffReal2VarOrder1> = x_vals
        .iter()
        .map(|&xv| scale * (xv * xv + 1.0))
        .collect();
    let x_query = [AutoDiffReal2VarOrder1::independent(1.6, 0)];
    let res = interpolate_vector_pm_autodiff(&x_ad, &y_ad, &x_query).unwrap();

    let y_f64: Vec<f64> =
        x_vals.iter().map(|&xv| 2.5 * (xv * xv + 1.0)).collect();
    let coeffs_f64 = interp_pm(&x_vals, &y_f64).unwrap();
    let (val_ref, slope_ref) =
        interp_value_and_slope(&x_vals, &coeffs_f64, 1.6).unwrap();

    assert_close(res[0].val, val_ref, "autodiff val");
    assert_close(res[0].d1[0], slope_ref, "autodiff d/dx");
    assert_close(res[0].d1[1], val_ref / 2.5, "autodiff d/dscale");
}

proptest! {
    #[test]
    fn prop_interp_pm_preserves_monotonicity_on_subintervals(
        y0 in -100.0_f64..100.0,
        y1 in -100.0_f64..100.0,
        y2 in -100.0_f64..100.0,
        y3 in -100.0_f64..100.0,
        y4 in -100.0_f64..100.0,
        t in 0.0_f64..=1.0,
    ) {
        let x = [0.0, 1.0, 2.5, 3.2, 5.0];
        let y = [y0, y1, y2, y3, y4];
        let coeffs = interp_pm(&x, &y).unwrap();
        for i in 0..4 {
            let xq = x[i] + t * (x[i + 1] - x[i]);
            let (val, _) = interp_value_and_slope(&x, &coeffs, xq).unwrap();
            let lo = y[i].min(y[i + 1]) - 1e-12;
            let hi = y[i].max(y[i + 1]) + 1e-12;
            prop_assert!(val >= lo && val <= hi);
        }
    }

    #[test]
    fn prop_m3a_m3b_m3q_reproduce_linear_functions_exactly(
        slope in -50.0_f64..50.0,
        intercept in -50.0_f64..50.0,
        xq in 0.0_f64..4.0,
    ) {
        let x = [0.0, 0.8, 1.9, 3.1, 4.0];
        let y = x.map(|xv| slope * xv + intercept);
        for builder in [interp_pm, interp_m3a, interp_m3b, interp_m3q] {
            let coeffs = builder(&x, &y).unwrap();
            let (vals, slopes) =
                interp_values_and_slopes(&x, &coeffs, &[xq]).unwrap();
            let exact = slope * xq + intercept;
            prop_assert!((vals[0] - exact).abs() <= 1e-11 * (1.0 + exact.abs()));
            prop_assert!((slopes[0] - slope).abs() <= 1e-11 * (1.0 + slope.abs()));
        }
    }
}
