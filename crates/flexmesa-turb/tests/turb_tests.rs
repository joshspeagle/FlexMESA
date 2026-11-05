// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden-baseline parity tests for `flexmesa-turb` (`turb/test/test_output`).
//!
//! Verifies:
//! 1. `check_efficient_MLT_scaling`: Convective speeds $v_{\mathrm{conv}}$ at
//!    $L_1 = 10^5\,L_\odot$ and $L_2 = 10^8\,L_\odot$ and cubic scaling ratio
//!    $v_2 / v_1$ to $\le 10^{-12}$ relative error.
//! 2. `compare_TDC_and_Cox_MLT`: Steady-state convergence of `set_tdc` at
//!    $\Delta t = 10^{40}\,\mathrm{s}$ and exact Cox MLT parity.
//! 3. `check_TDC`: All 31 time-step rows $\Delta t = 500 \times 1.02^j$ s
//!    ($j = 0, \dots, 30$) on the negative-$Y$ decaying-convection branch,
//!    including the transition from damped convection to radiative equilibrium
//!    between $j = 15$ and $j = 16$.

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::inconsistent_digit_grouping,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_const::{
    MixingType, BOLTZ_SIGMA, KERG, LSUN, MP, MSUN, PI, RSUN, STANDARD_CGRAV,
};
use flexmesa_math::{pow, pow2, pow4};
use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltOption, MltParams, TdcInputs, TdcParams,
};

fn assert_f64_rel_eq(actual: f64, expected: f64, rtol: f64) {
    let rel = (actual - expected).abs() / expected.abs();
    assert!(
        rel <= rtol,
        "actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    );
}

/// Golden `(dt, gradT, conv_vel)` rows from `check_TDC` in
/// `turb/test/test_output`.
const GOLDEN_TDC_ROWS: [(f64, f64, f64); 31] = [
    (
        5.0000000000000000e+02,
        2.5204370043250246e-01,
        6.0597238914198903e+06,
    ),
    (
        5.1000000000000000e+02,
        2.5204285749162930e-01,
        5.9050075093203261e+06,
    ),
    (
        5.2020000000000005e+02,
        2.5204196393651124e-01,
        5.7494001285570739e+06,
    ),
    (
        5.3060399999999993e+02,
        2.5204101302204984e-01,
        5.5925658656495046e+06,
    ),
    (
        5.4121608000000003e+02,
        2.5203999647052466e-01,
        5.4341006185611701e+06,
    ),
    (
        5.5204040160000000e+02,
        2.5203890395733641e-01,
        5.2735100646096319e+06,
    ),
    (
        5.6308120963200008e+02,
        2.5203772235599331e-01,
        5.1101775684554409e+06,
    ),
    (
        5.7434283382464002e+02,
        2.5203643459023894e-01,
        4.9433157147388468e+06,
    ),
    (
        5.8582969050113286e+02,
        2.5203501781331550e-01,
        4.7718899504500106e+06,
    ),
    (
        5.9754628431115555e+02,
        2.5203344036462588e-01,
        4.5944918361562928e+06,
    ),
    (
        6.0949720999737872e+02,
        2.5203165633254127e-01,
        4.4091141978058126e+06,
    ),
    (
        6.2168715419732621e+02,
        2.5202959494944044e-01,
        4.2127157422839301e+06,
    ),
    (
        6.3412089728127273e+02,
        2.5202713722579184e-01,
        4.0002689176325197e+06,
    ),
    (
        6.4680331522689812e+02,
        2.5202405410296042e-01,
        3.7622592559380517e+06,
    ),
    (
        6.5973938153143615e+02,
        2.5201978092426058e-01,
        3.4756430172060304e+06,
    ),
    (
        6.7293416916206490e+02,
        2.5201140133059619e-01,
        3.0239000429141554e+06,
    ),
    (6.8639285254530625e+02, -1.2020505435123705e+05, 0.0),
    (7.0012070959621246e+02, -1.2020505435123705e+05, 0.0),
    (7.1412312378813670e+02, -1.2020505435123705e+05, 0.0),
    (7.2840558626389941e+02, -1.2020505435123705e+05, 0.0),
    (7.4297369798917748e+02, -1.2020505435123705e+05, 0.0),
    (7.5783317194896108e+02, -1.2020505435123705e+05, 0.0),
    (7.7298983538794027e+02, -1.2020505435123705e+05, 0.0),
    (7.8844963209569903e+02, -1.2020505435123705e+05, 0.0),
    (8.0421862473761303e+02, -1.2020505435123705e+05, 0.0),
    (8.2030299723236533e+02, -1.2020505435123705e+05, 0.0),
    (8.3670905717701260e+02, -1.2020505435123705e+05, 0.0),
    (8.5344323832055295e+02, -1.2020505435123705e+05, 0.0),
    (8.7051210308696398e+02, -1.2020505435123705e+05, 0.0),
    (8.8792234514870324e+02, -1.2020505435123705e+05, 0.0),
    (9.0568079205167737e+02, -1.2020505435123705e+05, 0.0),
];

#[test]
fn test_efficient_mlt_scaling_golden_parity() {
    let params = MltParams {
        option: MltOption::Cox,
        mixing_length_alpha: 1.0,
        henyey_nu: 0.0,
        henyey_y: 0.0,
        max_conv_vel: 1.0e99,
    };
    let t = 1.0e5;
    let rho = 1.0e-5;
    let grav = 1.0e4;
    let r = RSUN;
    let cp = 2.5 * KERG / MP;
    let p = rho * t * KERG / MP;
    let lambda = p / (rho * grav);
    let opacity = 1.0;
    let grada = 0.4;
    let gradl = grada;

    let l1 = 1.0e5 * LSUN;
    let gradr1 = 3.0 * p * opacity * l1
        / (64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r));
    let inp1 = MltInputs {
        chi_t: 1.0,
        chi_rho: 1.0,
        cp,
        grav,
        lambda,
        rho,
        p,
        t,
        opacity,
        gradr: gradr1,
        grada,
        gradl,
    };
    let out1 = set_mlt(&params, &inp1).unwrap();
    assert_f64_rel_eq(out1.conv_vel, 3.965_420_378_586_662e+06, 1.0e-12);
    assert_eq!(out1.mixing_type, MixingType::ConvectiveMixing);

    let l2 = 1.0e8 * LSUN;
    let gradr2 = 3.0 * p * opacity * l2
        / (64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r));
    let inp2 = MltInputs {
        gradr: gradr2,
        ..inp1
    };
    let out2 = set_mlt(&params, &inp2).unwrap();
    assert_f64_rel_eq(out2.conv_vel, 3.976_581_170_678_065e+07, 1.0e-12);
    assert_f64_rel_eq(
        out2.conv_vel / out1.conv_vel,
        1.002_814_529_362_806_4e+01,
        1.0e-12,
    );
}

#[test]
fn test_compare_tdc_and_cox_mlt_golden_parity() {
    let mixing_length_alpha = 2.0;
    let t = 1.0e5;
    let rho = 1.0e-5;
    let r = RSUN;
    let m = MSUN;
    let cgrav = STANDARD_CGRAV;
    let grav = m * cgrav / pow2(r);
    let cp = 2.5 * KERG / MP;
    let p = rho * t * KERG / MP;
    let scale_height = p / (rho * grav);
    let lambda = mixing_length_alpha * scale_height;
    let opacity = 1.0;
    let grada = 0.4;
    let gradl = grada;

    let mut l = 70.0 * LSUN;
    let mut gradr = 3.0 * p * opacity * l
        / (64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r));
    l = l * (1.0 + 1.0e-5) * (grada / gradr);
    gradr = 3.0 * p * opacity * l
        / (64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r));
    assert_f64_rel_eq(gradr - grada, 4.000_000_000_004_000_4e-06, 1.0e-12);

    let tdc_params = TdcParams {
        conv_vel_start: 0.0,
        mixing_length_alpha,
        tdc_alpha_d: 1.0,
        tdc_alpha_r: 0.0,
        tdc_alpha_pt: 0.0,
        tdc_alpha_c: 1.0,
        tdc_alpha_s: 1.0,
        dt: 1.0e40,
        cgrav,
        m,
        scale: l * 1.0e-3,
        max_conv_vel: 1.0e99,
        include_mlt_corr_to_tdc: true,
        use_tdc_enthalpy_flux_limiter: false,
        y_face_guess: 0.0,
    };
    let tdc_inp = TdcInputs {
        chi_t: 1.0,
        chi_rho: 1.0,
        gradr,
        r,
        p,
        t,
        rho,
        dv: 0.0,
        cp,
        opacity,
        scale_height,
        gradl,
        grada,
        eq_div_w: 0.0,
        grav,
        energy: 0.0,
    };
    let tdc_res = set_tdc(&tdc_params, &tdc_inp).unwrap();
    // Condition number of L - L0*gradL at gradr - grada = 4e-6 is 1e5, so 1 ULP
    // in L corresponds to 1.5e-11 relative in Y_face and < 2e-16 in gradT.
    assert_f64_rel_eq(
        tdc_res.output.grad_t,
        0.4 + 3.999_985_220_802_392_6e-06,
        1.0e-15,
    );
    assert_f64_rel_eq(
        tdc_res.output.y_face,
        3.999_985_220_802_392_6e-06,
        2.0e-11,
    );
    assert_f64_rel_eq(
        tdc_res.output.conv_vel,
        1.505_108_505_206_907_5e+02,
        2.0e-11,
    );

    let mlt_params = MltParams::cox(mixing_length_alpha);
    let mlt_inp = MltInputs {
        chi_t: 1.0,
        chi_rho: 1.0,
        cp,
        grav,
        lambda,
        rho,
        p,
        t,
        opacity,
        gradr,
        grada,
        gradl,
    };
    let mlt_res = set_mlt(&mlt_params, &mlt_inp).unwrap();
    assert_f64_rel_eq(mlt_res.y_face, 3.999_983_030_023_024_4e-06, 1.0e-12);
    assert_f64_rel_eq(mlt_res.conv_vel, 1.505_108_093_041_575_9e+02, 1.0e-12);
    assert_f64_rel_eq(mlt_res.gamma, 1.374_099_194_781_847_9e-03, 1.0e-12);
}

#[test]
fn test_tdc_dt_sweep_31_golden_rows() {
    let cgrav = 6.674_300_000_000_000_4e-8;
    let m = 5.870_740_045_687_566_4e34;
    let scale = 5.038_651_936_224_629_4e45;
    let l = 1.094_152_881_588_35 * (-5.038_651_936_224_629_4e45);
    let r = 10_314_294_541.567_163;
    let p = 5.058_158_724_980_889_4e20;
    let t = 613_193_666.517_836_8;
    let rho = 5204.573_257_474_575;
    let dv = 3.825_649_446_348_260_4e-7;
    let cp = 6_628_075_118.460_659;
    let opacity = 9.075_017_123_146_995e-2;
    let scale_height = 2_638_686_602.006_378;
    let gradl = 0.252_075_872_673_435;
    let grada = 0.252_046_972_568_727_4;
    let gradr =
        3.0 * p * opacity * l / (64.0 * PI * BOLTZ_SIGMA * pow4(t) * cgrav * m);
    let grav = m * cgrav / pow2(r);

    let inp = TdcInputs {
        chi_t: 1.0,
        chi_rho: 1.0,
        gradr,
        r,
        p,
        t,
        rho,
        dv,
        cp,
        opacity,
        scale_height,
        gradl,
        grada,
        eq_div_w: 0.0,
        grav,
        energy: 0.0,
    };

    for (j, &(expected_dt, expected_grad_t, expected_vc)) in
        GOLDEN_TDC_ROWS.iter().enumerate()
    {
        let dt = 500.0 * pow(1.02, j as f64);
        assert_f64_rel_eq(dt, expected_dt, 1.0e-14);
        let params = TdcParams {
            conv_vel_start: 52_320_587.415_154_05,
            mixing_length_alpha: 2.0,
            tdc_alpha_d: 1.0,
            tdc_alpha_r: 0.0,
            tdc_alpha_pt: 0.0,
            tdc_alpha_c: 1.0,
            tdc_alpha_s: 1.0,
            dt,
            cgrav,
            m,
            scale,
            max_conv_vel: 1.0e99,
            include_mlt_corr_to_tdc: true,
            use_tdc_enthalpy_flux_limiter: false,
            y_face_guess: 0.0,
        };
        let res = set_tdc(&params, &inp).unwrap();
        assert_f64_rel_eq(res.output.grad_t, expected_grad_t, 1.0e-12);
        if expected_vc == 0.0 {
            assert_eq!(res.output.conv_vel, 0.0);
            assert_eq!(res.output.mixing_type, MixingType::NoMixing);
        } else {
            assert_f64_rel_eq(res.output.conv_vel, expected_vc, 1.0e-12);
            assert_eq!(res.output.mixing_type, MixingType::ConvectiveMixing);
        }
    }
}
