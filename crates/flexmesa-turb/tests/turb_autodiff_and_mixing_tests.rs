// SPDX-License-Identifier: LGPL-3.0-or-later
//! Autodiff and diffusive-mixing tests for `flexmesa-turb`.
//!
//! Verifies:
//! 1. All six geometric MLT options (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`,
//!    `Mihalas`) and `DualOrder1` forward-mode automatic differentiation
//!    against centered finite differences.
//! 2. `DualOrder1` Implicit Function Theorem derivatives for `set_tdc`
//!    (including the quintic enthalpy flux limiter) against centered finite
//!    differences.
//! 3. Both Langer (1985) semiconvection prescriptions (`Langer_85` and
//!    `Langer_85 mixing; gradT = gradr`) and all three thermohaline
//!    prescriptions (`Kippenhahn`, `Traxler_Garaud_Stellmach_11`, and
//!    `Brown_Garaud_Stellmach_13`).

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

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{
    MixingType, BOLTZ_SIGMA, KERG, MP, MSUN, PI, RSUN, STANDARD_CGRAV,
};
use flexmesa_math::{pow2, pow4};
use flexmesa_turb::{
    set_mlt, set_semiconvection, set_tdc, set_thermohaline, DrivingSpecies,
    MltInputs, MltOption, MltParams, SemiconvectionInputs,
    SemiconvectionOption, SemiconvectionParams, TdcInputs, TdcParams,
    ThermohalineInputs, ThermohalineOption, ThermohalineParams,
};

fn assert_f64_rel_eq(actual: f64, expected: f64, rtol: f64) {
    if expected == 0.0 {
        assert!(actual.abs() <= 1.0e-100, "actual={actual:.16e} != 0");
        return;
    }
    let rel = (actual - expected).abs() / expected.abs();
    assert!(
        rel <= rtol,
        "actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    );
}

#[test]
fn test_mlt_options_and_autodiff_vs_finite_difference() {
    let t = 1.0e5;
    let rho = 1.0e-5;
    let grav = 1.0e4;
    let cp = 2.5 * KERG / MP;
    let p = rho * t * KERG / MP;
    let lambda = p / (rho * grav);
    let opacity = 1.0;
    let grada = 0.4;
    let gradl = 0.4;
    let gradr = 0.8;

    for opt in [
        MltOption::Cox,
        MltOption::Tdc,
        MltOption::Henyey,
        MltOption::Ml1,
        MltOption::Ml2,
        MltOption::Mihalas,
    ] {
        let params = MltParams {
            option: opt,
            mixing_length_alpha: 1.8,
            henyey_nu: 8.0,
            henyey_y: 1.0 / 3.0,
            max_conv_vel: 1.0e99,
        };
        let dual_inp = MltInputs {
            chi_t: DualOrder1::<3>::constant(1.0),
            chi_rho: DualOrder1::<3>::constant(1.0),
            cp: DualOrder1::<3>::constant(cp),
            grav: DualOrder1::<3>::constant(grav),
            lambda: DualOrder1::<3>::constant(lambda),
            rho: DualOrder1::<3>::independent(rho, 0),
            p: DualOrder1::<3>::constant(p),
            t: DualOrder1::<3>::independent(t, 1),
            opacity: DualOrder1::<3>::constant(opacity),
            gradr: DualOrder1::<3>::independent(gradr, 2),
            grada: DualOrder1::<3>::constant(grada),
            gradl: DualOrder1::<3>::constant(gradl),
        };
        let dual_out = set_mlt(&params, &dual_inp).unwrap();
        assert!(dual_out.conv_vel.val > 0.0);

        let step = 1.0e-6;
        let d_gradr = gradr * step;
        let base_inp = MltInputs {
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
        let up = set_mlt(
            &params,
            &MltInputs {
                gradr: gradr + d_gradr,
                ..base_inp
            },
        )
        .unwrap();
        let dn = set_mlt(
            &params,
            &MltInputs {
                gradr: gradr - d_gradr,
                ..base_inp
            },
        )
        .unwrap();
        let fd_grad_t = (up.grad_t - dn.grad_t) / (2.0 * d_gradr);
        assert_f64_rel_eq(dual_out.grad_t.d1[2], fd_grad_t, 1.0e-6);
    }
}

#[test]
fn test_tdc_autodiff_implicit_function_theorem_matches_finite_difference() {
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
    let opacity = 1.0;
    let grada = 0.4;
    let gradl = 0.4;
    let gradr = 0.45;
    let l = 64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r) * gradr
        / (3.0 * p * opacity);

    let params = TdcParams {
        conv_vel_start: 1.0e4,
        mixing_length_alpha,
        tdc_alpha_d: 1.0,
        tdc_alpha_r: 1.0,
        tdc_alpha_pt: 0.0,
        tdc_alpha_c: 1.0,
        tdc_alpha_s: 1.0,
        dt: 1.0e4,
        cgrav,
        m,
        scale: l * 1.0e-3,
        max_conv_vel: 1.0e99,
        include_mlt_corr_to_tdc: true,
        use_tdc_enthalpy_flux_limiter: true,
        y_face_guess: 0.01,
    };

    let c = DualOrder1::<2>::constant;
    let dual_inp = TdcInputs {
        chi_t: c(1.0),
        chi_rho: c(1.0),
        gradr: DualOrder1::<2>::independent(gradr, 0),
        r: c(r),
        p: c(p),
        t: DualOrder1::<2>::independent(t, 1),
        rho: c(rho),
        dv: c(0.0),
        cp: c(cp),
        opacity: c(opacity),
        scale_height: c(scale_height),
        gradl: c(gradl),
        grada: c(grada),
        eq_div_w: c(0.0),
        grav: c(grav),
        energy: c(1.5 * p / rho),
    };
    let dual_res = set_tdc(&params, &dual_inp).unwrap();

    let base_inp = TdcInputs {
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
        energy: 1.5 * p / rho,
    };
    let d_gradr = gradr * 1.0e-6;
    let up = set_tdc(
        &params,
        &TdcInputs {
            gradr: gradr + d_gradr,
            ..base_inp
        },
    )
    .unwrap();
    let dn = set_tdc(
        &params,
        &TdcInputs {
            gradr: gradr - d_gradr,
            ..base_inp
        },
    )
    .unwrap();
    let fd_y = (up.output.y_face - dn.output.y_face) / (2.0 * d_gradr);
    let fd_vc = (up.output.conv_vel - dn.output.conv_vel) / (2.0 * d_gradr);
    assert_f64_rel_eq(dual_res.output.y_face.d1[0], fd_y, 1.0e-5);
    assert_f64_rel_eq(dual_res.output.conv_vel.d1[0], fd_vc, 1.0e-5);
}

#[test]
fn test_semiconvection_and_thermohaline_prescriptions() {
    let sc_inp = SemiconvectionInputs {
        l: 1.0e35,
        lambda: 1.0e10,
        t: 2.0e7,
        p: 1.0e16,
        pr: 1.0e15,
        beta: 0.9,
        opacity: 0.34,
        rho: 10.0,
        cp: 2.5e8,
        gradr: 0.42,
        grada: 0.40,
        gradl: 0.45,
    };
    for opt in [
        SemiconvectionOption::Langer85GradTIsGradr,
        SemiconvectionOption::Langer85,
    ] {
        let sc_params = SemiconvectionParams {
            option: opt,
            alpha_semiconvection: 0.1,
            cgrav: STANDARD_CGRAV,
            gradl_composition_term: 0.05,
            m: MSUN,
        };
        let out = set_semiconvection(&sc_params, &sc_inp).unwrap();
        assert!(out.d > 0.0);
        assert!(out.conv_vel > 0.0);
        assert_eq!(out.mixing_type, MixingType::SemiconvectiveMixing);

        let d_gr = 1.0e-7 * sc_inp.gradr;
        let up = set_semiconvection(
            &sc_params,
            &SemiconvectionInputs {
                gradr: sc_inp.gradr + d_gr,
                ..sc_inp
            },
        )
        .unwrap();
        let dn = set_semiconvection(
            &sc_params,
            &SemiconvectionInputs {
                gradr: sc_inp.gradr - d_gr,
                ..sc_inp
            },
        )
        .unwrap();
        let c = |x: f64| DualOrder1::<1>::constant(x);
        let dual_out = set_semiconvection(
            &sc_params,
            &SemiconvectionInputs {
                l: c(sc_inp.l),
                lambda: c(sc_inp.lambda),
                t: c(sc_inp.t),
                p: c(sc_inp.p),
                pr: c(sc_inp.pr),
                beta: c(sc_inp.beta),
                opacity: c(sc_inp.opacity),
                rho: c(sc_inp.rho),
                cp: c(sc_inp.cp),
                gradr: DualOrder1::<1>::independent(sc_inp.gradr, 0),
                grada: c(sc_inp.grada),
                gradl: c(sc_inp.gradl),
            },
        )
        .unwrap();
        assert_f64_rel_eq(
            dual_out.d.d1[0],
            (up.d - dn.d) / (2.0 * d_gr),
            1.0e-6,
        );
        assert_f64_rel_eq(
            dual_out.grad_t.d1[0],
            (up.grad_t - dn.grad_t) / (2.0 * d_gr),
            1.0e-6,
        );
    }

    let th_inp = ThermohalineInputs {
        lambda: 1.0e9,
        grada: 0.40,
        gradr: 0.38,
        t: 3.0e7,
        opacity: 0.3,
        rho: 5.0,
        cp: 2.5e8,
    };
    for opt in [
        ThermohalineOption::Kippenhahn,
        ThermohalineOption::TraxlerGaraudStellmach11,
        ThermohalineOption::BrownGaraudStellmach13,
    ] {
        let th_params = ThermohalineParams {
            option: opt,
            gradl_composition_term: -1.0e-4,
            xh1: 0.7,
            thermohaline_coeff: 1.0,
            species: DrivingSpecies::he3(),
        };
        let out = set_thermohaline(&th_params, &th_inp).unwrap();
        assert!(out.d > 0.0);
        assert!(out.conv_vel > 0.0);
        assert_eq!(out.mixing_type, MixingType::ThermohalineMixing);
    }
    let kip_params = ThermohalineParams {
        option: ThermohalineOption::Kippenhahn,
        gradl_composition_term: -1.0e-4,
        xh1: 0.7,
        thermohaline_coeff: 1.0,
        species: DrivingSpecies::he3(),
    };
    let d_lam = 1.0e-7 * th_inp.lambda;
    let up = set_thermohaline(
        &kip_params,
        &ThermohalineInputs {
            lambda: th_inp.lambda + d_lam,
            ..th_inp
        },
    )
    .unwrap();
    let dn = set_thermohaline(
        &kip_params,
        &ThermohalineInputs {
            lambda: th_inp.lambda - d_lam,
            ..th_inp
        },
    )
    .unwrap();
    let c = |x: f64| DualOrder1::<1>::constant(x);
    let dual_th = set_thermohaline(
        &kip_params,
        &ThermohalineInputs {
            lambda: DualOrder1::<1>::independent(th_inp.lambda, 0),
            grada: c(th_inp.grada),
            gradr: c(th_inp.gradr),
            t: c(th_inp.t),
            opacity: c(th_inp.opacity),
            rho: c(th_inp.rho),
            cp: c(th_inp.cp),
        },
    )
    .unwrap();
    assert_f64_rel_eq(
        dual_th.conv_vel.d1[0],
        (up.conv_vel - dn.conv_vel) / (2.0 * d_lam),
        1.0e-6,
    );
}

#[test]
fn test_mlt_marginal_superadiabaticity_series_eliminates_cardano_cancellation()
{
    let params = MltParams {
        option: MltOption::Cox,
        mixing_length_alpha: 1.8,
        henyey_nu: 8.0,
        henyey_y: 1.0 / 3.0,
        max_conv_vel: 1.0e99,
    };
    let c = |x: f64| DualOrder1::<1>::constant(x);

    // 1. Low-efficiency optically thin envelope (`A ~ 1e-2`): tiny `gradr - gradl`
    //    must smoothly approach the radiative limit (`grad_t -> gradr`,
    //    `d(grad_t)/d(gradr) -> 1`) without positive-ULP Cardano spikes.
    let t_env = 1.0e5;
    let rho_env = 1.0e-9;
    let grav_env = 1.0e3;
    let cp_env = 2.5 * KERG / MP;
    let p_env = rho_env * t_env * KERG / MP;
    let lam_env = p_env / (rho_env * grav_env);
    let gradl = 0.4;
    for exp_idx in [16_i32, 15, 14, 13, 12, 10, 8] {
        let dgrad = 10.0_f64.powi(-exp_idx);
        let gradr = gradl + dgrad;
        let out = set_mlt(
            &params,
            &MltInputs {
                chi_t: c(1.0),
                chi_rho: c(1.0),
                cp: c(cp_env),
                grav: c(grav_env),
                lambda: c(lam_env),
                rho: c(rho_env),
                p: c(p_env),
                t: c(t_env),
                opacity: c(0.4),
                gradr: DualOrder1::<1>::independent(gradr, 0),
                grada: c(gradl),
                gradl: c(gradl),
            },
        )
        .unwrap();
        assert!(
            (out.grad_t.val - gradr).abs() <= dgrad,
            "spurious jump at dgrad={dgrad:.1e}: grad_t={} vs gradr={gradr}",
            out.grad_t.val
        );
        assert!(
            (out.grad_t.d1[0] - 1.0).abs() < 1.0e-8,
            "bad d(grad_t)/d(gradr)={} at dgrad={dgrad:.1e}",
            out.grad_t.d1[0]
        );
        assert!(out.gamma.val > 0.0 && out.conv_vel.val > 0.0);
    }

    // 2. High-efficiency deep convective core (`A >> 1e7`): tiny `gradr - gradl = 1e-13`
    //    still has `delta = A^2 * dgrad >> 1` and must remain strongly adiabatic (`grad_t ~ gradl`).
    let t_core = 2.0e7;
    let rho_core = 100.0;
    let grav_core = 2.0e5;
    let p_core = rho_core * t_core * KERG / MP;
    let lam_core = p_core / (rho_core * grav_core);
    let out_core = set_mlt(
        &params,
        &MltInputs {
            chi_t: c(1.0),
            chi_rho: c(1.0),
            cp: c(cp_env),
            grav: c(grav_core),
            lambda: c(lam_core),
            rho: c(rho_core),
            p: c(p_core),
            t: c(t_core),
            opacity: c(1.0),
            gradr: DualOrder1::<1>::independent(gradl + 1.0e-13, 0),
            grada: c(gradl),
            gradl: c(gradl),
        },
    )
    .unwrap();
    assert!(
        (out_core.grad_t.val - gradl).abs() < 1.0e-13,
        "deep convective core lost adiabaticity: grad_t={}",
        out_core.grad_t.val
    );
    assert!(out_core.conv_vel.val > 0.0);
}
