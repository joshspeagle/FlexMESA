// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` `turb` parity tests (`#161`).
//!
//! Exercises regimes and code paths across `mlt`, `semiconvection`,
//! `thermohaline`, `tdc`, and `tdc_dual` relevant to `star/1M_thermohaline`,
//! `star/5M_cepheid_blue_loop`, `star/make_o_ne_wd`, `star/ns_c`,
//! `star/wd_c_core_ignition`, `star/7M_prems_to_AGB`, and `star/rsp_*`:
//! - All six geometric `MltOption` prescriptions, radiative zones, and value-only
//!   `max_conv_vel` clamping (`turb/private/mlt.f90:73-179`).
//! - `set_tdc` velocity capping with/without `include_mlt_corr_to_tdc`
//!   (`turb/public/turb.f90:180-194`), Wuchterl & Feuchtinger (1998) enthalpy
//!   flux limiter (`turb/private/tdc_support.f90:534-597`), radiative/Pt work
//!   damping and `Eq_div_w` (`tdc_support.f90:549-562`), negative-$Y$ branches
//!   (`turb/private/tdc.f90:103-208`), and extreme-scale `Q` bisection
//!   (`tdc_support.f90:133, 161-173, 284-296`).
//! - `Langer_85` `alpha.min(1.0)` cap and luminosity balance
//!   (`turb/private/semiconvection.f90:77-127`).
//! - `thermohaline` prescriptions for ${}^3\mathrm{He}$, ${}^4\mathrm{He}$, and
//!   heavy-species (${}^{12}\mathrm{C}$) driving ions, `r_th >= 1` and
//!   `Pr < 0` cutoffs, and all three Brown et al. (2013) initial-guess regimes
//!   (`turb/private/thermohaline.f90:62-274`).

#![forbid(unsafe_code)]
#![allow(
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::str::FromStr;

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{
    MixingType, BOLTZ_SIGMA, CLIGHT, CRAD, KERG, MP, MSUN, PI, RSUN,
    SQRT_2_DIV_3, STANDARD_CGRAV,
};
use flexmesa_math::{pow2, pow3, pow4, with_numerics_mode, NumericsMode};
use flexmesa_turb::tdc_dual::{AutoDiffRealTdc, TdcLift};
use flexmesa_turb::{
    get_d_thermohaline, set_mlt, set_semiconvection, set_tdc, set_thermohaline,
    DrivingSpecies, MltInputs, MltOption, MltParams, SemiconvectionInputs,
    SemiconvectionOption, SemiconvectionParams, TdcInputs, TdcParams,
    ThermohalineInputs, ThermohalineOption, ThermohalineParams, TurbError,
    TurbScalar,
};

fn assert_rel_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() <= rtol,
            "{label}: actual={actual:.16e}, expected=0"
        );
        return;
    }
    let rel = (actual - expected).abs() / expected.abs();
    assert!(
        rel <= rtol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e} > {rtol:.3e}"
    );
}

#[test]
fn test_mlt_all_options_radiative_and_max_conv_vel_value_only_cap() {
    with_numerics_mode(NumericsMode::Strict, || {
        let t = 1.0e5;
        let rho = 1.0e-5;
        let grav = 1.0e4;
        let cp = 2.5 * KERG / MP;
        let p = rho * t * KERG / MP;
        let hp = p / (rho * grav);
        let alpha = 1.8;
        let lambda = alpha * hp;
        let opacity = 1.0;
        let grada = 0.4;
        let gradl = 0.42;

        // 1. Unknown option strings reject cleanly (`mlt.f90:106-109`).
        assert!(matches!(
            MltOption::from_str("bogus"),
            Err(TurbError::UnknownMltOption(_))
        ));

        // 2. Radiative zone (`gradr <= gradL`, `mlt.f90:157-163`).
        for gradr in [gradl, gradl - 0.05] {
            let out = set_mlt(
                &MltParams::cox(alpha),
                &MltInputs {
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
                },
            )
            .expect("radiative set_mlt");
            assert_eq!(out.gamma, -1.0e99);
            assert_eq!(out.conv_vel, 0.0);
            assert_eq!(out.d, 0.0);
            assert_eq!(out.grad_t, gradr);
            assert_eq!(out.y_face, gradr - gradl);
            assert_eq!(out.mixing_type, MixingType::NoMixing);
        }

        // 3. `TDC` alias matches `Cox` bit for bit in value and AD partials
        // (`mlt.f90:77`), and `conv_vel%val = max_conv_vel` caps only the
        // primal value while preserving uncapped partials (`mlt.f90:151`).
        let c = |v: f64| DualOrder1::<4>::constant(v);
        let dual_inp = MltInputs {
            chi_t: c(1.0),
            chi_rho: c(1.0),
            cp: c(cp),
            grav: c(grav),
            lambda: DualOrder1::<4>::independent(lambda, 0),
            rho: DualOrder1::<4>::independent(rho, 1),
            p: c(p),
            t: DualOrder1::<4>::independent(t, 2),
            opacity: c(opacity),
            gradr: DualOrder1::<4>::independent(2.5, 3),
            grada: c(grada),
            gradl: c(gradl),
        };
        let cox_uncapped = set_mlt(
            &MltParams {
                option: MltOption::Cox,
                mixing_length_alpha: alpha,
                henyey_nu: 0.0,
                henyey_y: 0.0,
                max_conv_vel: 1.0e99,
            },
            &dual_inp,
        )
        .expect("cox uncapped");
        let tdc_uncapped = set_mlt(
            &MltParams {
                option: MltOption::Tdc,
                mixing_length_alpha: alpha,
                henyey_nu: 0.0,
                henyey_y: 0.0,
                max_conv_vel: 1.0e99,
            },
            &dual_inp,
        )
        .expect("tdc alias");
        assert_eq!(
            cox_uncapped.gamma.val.to_bits(),
            tdc_uncapped.gamma.val.to_bits()
        );
        assert_eq!(
            cox_uncapped.conv_vel.val.to_bits(),
            tdc_uncapped.conv_vel.val.to_bits()
        );
        for i in 0..4 {
            assert_eq!(
                cox_uncapped.conv_vel.d1[i].to_bits(),
                tdc_uncapped.conv_vel.d1[i].to_bits()
            );
            assert_eq!(
                cox_uncapped.grad_t.d1[i].to_bits(),
                tdc_uncapped.grad_t.d1[i].to_bits()
            );
        }

        for opt in [
            MltOption::Cox,
            MltOption::Henyey,
            MltOption::Ml1,
            MltOption::Ml2,
            MltOption::Mihalas,
        ] {
            let p_uncapped = MltParams {
                option: opt,
                mixing_length_alpha: alpha,
                henyey_nu: 8.0,
                henyey_y: 1.0 / 3.0,
                max_conv_vel: 1.0e99,
            };
            let u = set_mlt(&p_uncapped, &dual_inp).expect("uncapped");
            let v_cap = 0.5 * u.conv_vel.val;
            let p_capped = MltParams {
                max_conv_vel: v_cap,
                ..p_uncapped
            };
            let cap = set_mlt(&p_capped, &dual_inp).expect("capped");
            assert_eq!(cap.conv_vel.val, v_cap);
            assert_eq!(cap.conv_vel.d1, u.conv_vel.d1);
            assert_eq!(cap.gamma.val.to_bits(), u.gamma.val.to_bits());
            assert_eq!(cap.grad_t.val.to_bits(), u.grad_t.val.to_bits());
            assert_eq!(
                cap.d.val.to_bits(),
                (v_cap * lambda).div_real(3.0).to_bits()
            );
        }
    });
}

#[test]
fn test_tdc_max_conv_vel_cap_enthalpy_limiter_damping_and_extreme_brackets() {
    with_numerics_mode(NumericsMode::Strict, || {
        let alpha = 2.0;
        let t = 1.0e5;
        let rho = 1.0e-5;
        let r = RSUN;
        let m = MSUN;
        let cgrav = STANDARD_CGRAV;
        let grav = m * cgrav / pow2(r);
        let cp = 2.5 * KERG / MP;
        let p = rho * t * KERG / MP;
        let hp = p / (rho * grav);
        let opacity = 1.0;
        let grada = 0.4;
        let gradl = 0.4;
        let gradr = 0.6;
        let l = 64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r) * gradr
            / (3.0 * p * opacity);

        // 1. `max_conv_vel` cap with `include_mlt_corr_to_tdc = true/false`
        // (`turb/public/turb.f90:180-194`).
        let base_params = TdcParams {
            conv_vel_start: 1.0e6,
            mixing_length_alpha: alpha,
            tdc_alpha_d: 1.0,
            tdc_alpha_r: 0.0,
            tdc_alpha_pt: 0.0,
            tdc_alpha_c: 1.0,
            tdc_alpha_s: 1.0,
            dt: 1.0e8,
            cgrav,
            m,
            scale: l * 1.0e-3,
            max_conv_vel: 1.0e99,
            include_mlt_corr_to_tdc: true,
            use_tdc_enthalpy_flux_limiter: false,
        };
        let c = |v: f64| DualOrder1::<4>::constant(v);
        let dual_inp = TdcInputs {
            chi_t: c(1.0),
            chi_rho: c(1.0),
            gradr: DualOrder1::<4>::independent(gradr, 0),
            r: c(r),
            p: DualOrder1::<4>::independent(p, 1),
            t: DualOrder1::<4>::independent(t, 2),
            rho: DualOrder1::<4>::independent(rho, 3),
            dv: c(0.0),
            cp: c(cp),
            opacity: c(opacity),
            scale_height: c(hp),
            gradl: c(gradl),
            grada: c(grada),
            eq_div_w: c(0.0),
            grav: c(grav),
            energy: c(1.5 * p / rho),
        };
        let uncapped = set_tdc(&base_params, &dual_inp).expect("uncapped tdc");
        let v_cap = 0.25 * uncapped.output.conv_vel.val;
        assert!(v_cap > 0.0);

        for include_corr in [true, false] {
            let capped_params = TdcParams {
                max_conv_vel: v_cap,
                include_mlt_corr_to_tdc: include_corr,
                ..base_params
            };
            let res = set_tdc(&capped_params, &dual_inp).expect("capped tdc");
            assert_eq!(res.output.conv_vel.val, v_cap);
            assert_eq!(res.output.conv_vel.d1, [0.0; 4]);
            assert_eq!(res.output.mixing_type, MixingType::ConvectiveMixing);
            assert_eq!(
                res.output.d.val.to_bits(),
                (v_cap * hp * alpha).div_real(3.0).to_bits()
            );

            let l_ad = (DualOrder1::<4>::constant(64.0 * PI * BOLTZ_SIGMA)
                * dual_inp.t.pow4()
                * dual_inp.grav
                * dual_inp.r.pow2()
                * dual_inp.gradr)
                / (DualOrder1::<4>::constant(3.0)
                    * dual_inp.p
                    * dual_inp.opacity);
            let l0_ad = (DualOrder1::<4>::constant(
                (16.0 * PI * CRAD * CLIGHT / 3.0) * cgrav * m,
            ) * dual_inp.t.pow4())
                / (dual_inp.p * dual_inp.opacity);
            let c0_ad = DualOrder1::<4>::constant(
                base_params.tdc_alpha_c * alpha * (0.5 * SQRT_2_DIV_3),
            ) * dual_inp.rho
                * dual_inp.t
                * dual_inp.cp
                * DualOrder1::<4>::constant(4.0)
                * DualOrder1::<4>::constant(PI)
                * dual_inp.r.pow2();
            let gamma_ad = res.output.gamma;
            let denom = if include_corr {
                l0_ad
                    + c0_ad
                        * DualOrder1::<4>::constant(SQRT_2_DIV_3)
                        * DualOrder1::<4>::constant(v_cap)
                        * (gamma_ad
                            / (DualOrder1::<4>::constant(1.0) + gamma_ad))
            } else {
                l0_ad
                    + c0_ad
                        * DualOrder1::<4>::constant(SQRT_2_DIV_3)
                        * DualOrder1::<4>::constant(v_cap)
            };
            let want_y = (l_ad - l0_ad * dual_inp.gradl) / denom;
            assert_eq!(res.output.y_face.val.to_bits(), want_y.val.to_bits());
            for i in 0..4 {
                assert_eq!(
                    res.output.y_face.d1[i].to_bits(),
                    want_y.d1[i].to_bits()
                );
            }
        }

        // 2. Wuchterl & Feuchtinger (1998) enthalpy flux limiter
        // (`tdc_support.f90:534-597`): linear below X=0.95 and saturated above.
        let scalar_inp = TdcInputs {
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
            scale_height: hp,
            gradl,
            grada,
            eq_div_w: 0.0,
            grav,
            energy: 1.5 * p / rho,
        };
        let fl_off = set_tdc(
            &TdcParams {
                use_tdc_enthalpy_flux_limiter: false,
                ..base_params
            },
            &scalar_inp,
        )
        .expect("flux limiter off");
        let fl_linear = set_tdc(
            &TdcParams {
                use_tdc_enthalpy_flux_limiter: true,
                ..base_params
            },
            &scalar_inp,
        )
        .expect("flux limiter linear");
        assert_eq!(
            fl_off.output.conv_vel.to_bits(),
            fl_linear.output.conv_vel.to_bits()
        );
        assert_eq!(
            fl_off.output.y_face.to_bits(),
            fl_linear.output.y_face.to_bits()
        );

        let sat_inp = TdcInputs {
            gradr: 20.0,
            energy: 0.0,
            ..scalar_inp
        };
        let sat_scale = 20.0 * l / gradr * 1.0e-3;
        let sat_off = set_tdc(
            &TdcParams {
                scale: sat_scale,
                use_tdc_enthalpy_flux_limiter: false,
                ..base_params
            },
            &sat_inp,
        )
        .expect("saturated off");
        let sat_on = set_tdc(
            &TdcParams {
                scale: sat_scale,
                use_tdc_enthalpy_flux_limiter: true,
                ..base_params
            },
            &sat_inp,
        )
        .expect("saturated on");
        assert!(sat_on.output.conv_vel < sat_off.output.conv_vel);
        assert!(sat_on.output.y_face > sat_off.output.y_face);

        // 3. Radiative damping (`tdc_alpha_r`), Pt work (`tdc_alpha_pt` with
        // +/- `dV`), and `eq_div_w` (`tdc_support.f90:549-562`), plus AD vs FD.
        let damp_params = TdcParams {
            tdc_alpha_r: 1.0,
            tdc_alpha_pt: 1.0,
            dt: 1.0e5,
            ..base_params
        };
        let c2 = DualOrder1::<2>::constant;
        let damp_dual_inp = TdcInputs {
            chi_t: c2(1.0),
            chi_rho: c2(1.0),
            gradr: DualOrder1::<2>::independent(gradr, 0),
            r: c2(r),
            p: c2(p),
            t: DualOrder1::<2>::independent(t, 1),
            rho: c2(rho),
            dv: c2(0.05 / rho),
            cp: c2(cp),
            opacity: c2(opacity),
            scale_height: c2(hp),
            gradl: c2(gradl),
            grada: c2(grada),
            eq_div_w: c2(1.0e-2),
            grav: c2(grav),
            energy: c2(1.5 * p / rho),
        };
        let damp_dual = set_tdc(&damp_params, &damp_dual_inp).expect("damp ad");
        let d_gr = 1.0e-6 * gradr;
        let base_damp_scalar = TdcInputs {
            dv: 0.05 / rho,
            eq_div_w: 1.0e-2,
            ..scalar_inp
        };
        let up = set_tdc(
            &damp_params,
            &TdcInputs {
                gradr: gradr + d_gr,
                ..base_damp_scalar
            },
        )
        .expect("up");
        let dn = set_tdc(
            &damp_params,
            &TdcInputs {
                gradr: gradr - d_gr,
                ..base_damp_scalar
            },
        )
        .expect("dn");
        assert_rel_close(
            damp_dual.output.y_face.d1[0],
            (up.output.y_face - dn.output.y_face) / (2.0 * d_gr),
            1.0e-5,
            "damped Y_face d/dgradr",
        );
        assert_rel_close(
            damp_dual.output.conv_vel.d1[0],
            (up.output.conv_vel - dn.output.conv_vel) / (2.0 * d_gr),
            1.0e-5,
            "damped conv_vel d/dgradr",
        );

        // Contracting (`dV < 0`) enhances turbulence vs expanding (`dV > 0`).
        let neg_dv = set_tdc(
            &damp_params,
            &TdcInputs {
                dv: -0.05 / rho,
                ..base_damp_scalar
            },
        )
        .expect("contracting");
        assert!(neg_dv.output.conv_vel > damp_dual.output.conv_vel.val);

        // 4. Radiative negative-Y fast path (`A0 < 1e-10`, `tdc.f90:106-114`).
        let rad_inp = TdcInputs {
            gradr: 0.35,
            ..scalar_inp
        };
        let rad_res = set_tdc(
            &TdcParams {
                conv_vel_start: 1.0e-12,
                ..base_params
            },
            &rad_inp,
        )
        .expect("radY fast path");
        assert_eq!(rad_res.num_iters, 0);
        assert_eq!(rad_res.output.conv_vel, 0.0);
        assert_eq!(rad_res.output.mixing_type, MixingType::NoMixing);
        assert_rel_close(rad_res.output.y_face, 0.35 - gradl, 1.0e-14, "radY");

        // 5. Opposite-sign `-Inf` upper bracket at `Z = 100` still bisects
        // cleanly (`tdc_support.f90:133`).
        let r_huge = 1.0e120;
        let m_huge = grav * r_huge * r_huge / cgrav;
        let l_huge =
            64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * r_huge * r_huge * 0.5
                / (3.0 * p * opacity);
        let huge_res = set_tdc(
            &TdcParams {
                conv_vel_start: 0.0,
                cgrav,
                m: m_huge,
                scale: l_huge * 1.0e-3,
                ..base_params
            },
            &TdcInputs {
                gradr: 0.5,
                r: r_huge,
                ..scalar_inp
            },
        )
        .expect("TDC with -Inf Q(Zub) must still bisect and converge");
        assert_eq!(huge_res.output.mixing_type, MixingType::ConvectiveMixing);
        assert!(huge_res.output.conv_vel > 0.0);
        assert!(huge_res.output.y_face > 0.0);
        assert!(huge_res.output.y_face < 0.5 - gradl);
    });
}

#[test]
fn test_semiconvection_alpha_cap_early_exit_and_langer85_luminosity_balance() {
    with_numerics_mode(NumericsMode::Strict, || {
        assert!(matches!(
            SemiconvectionOption::from_str("unknown"),
            Err(TurbError::UnknownSemiconvectionOption(_))
        ));

        let m = MSUN;
        let cgrav = STANDARD_CGRAV;
        let t = 2.0e7;
        let p = 1.0e16;
        let pr = CRAD * pow4(t) / 3.0;
        let beta = (p - pr) / p;
        let opacity = 0.34;
        let rho = 10.0;
        let cp = 2.5e8;
        let grada = 0.40;
        let gradl_comp = 0.05;
        let gradl = grada + gradl_comp;
        let gradr = 0.42;
        let lg = (16.0 * PI * CLIGHT * m * cgrav * pr) / (p * opacity);
        let l = gradr * lg;
        let inp = SemiconvectionInputs {
            l,
            lambda: 1.0e10,
            t,
            p,
            pr,
            beta,
            opacity,
            rho,
            cp,
            gradr,
            grada,
            gradl,
        };

        // 1. Early return when `D <= 0` (`semiconvection.f90:81`).
        let no_mix = set_semiconvection(
            &SemiconvectionParams {
                option: SemiconvectionOption::Langer85,
                alpha_semiconvection: 0.1,
                cgrav,
                gradl_composition_term: gradl_comp,
                m,
            },
            &SemiconvectionInputs { gradr: 0.48, ..inp },
        )
        .expect("D <= 0 early exit");
        assert_eq!(no_mix.mixing_type, MixingType::NoMixing);
        assert!(no_mix.d < 0.0);

        // 2. `alpha = min(1, alpha_semiconvection)` applies only to the
        // `Langer_85` quadratic `gradT`, while `D` scales linearly with
        // `alpha_semiconvection` (`semiconvection.f90:79, 97`).
        let out_a1 = set_semiconvection(
            &SemiconvectionParams {
                option: SemiconvectionOption::Langer85,
                alpha_semiconvection: 1.0,
                cgrav,
                gradl_composition_term: gradl_comp,
                m,
            },
            &inp,
        )
        .expect("alpha=1.0");
        let out_a15 = set_semiconvection(
            &SemiconvectionParams {
                option: SemiconvectionOption::Langer85,
                alpha_semiconvection: 1.5,
                cgrav,
                gradl_composition_term: gradl_comp,
                m,
            },
            &inp,
        )
        .expect("alpha=1.5");
        assert_eq!(out_a1.grad_t.to_bits(), out_a15.grad_t.to_bits());
        assert_rel_close(out_a15.d / out_a1.d, 1.5, 1.0e-14, "D linear scale");

        // 3. Verify Langer (1985) luminosity balance (`semiconvection.f90:91-95`):
        // L/Lrad - Lsc/Lrad - 1 == 0.
        let grad = out_a1.grad_t;
        let bc = 32.0 - 24.0 * beta - beta * beta;
        let grad_mu = (4.0 - 3.0 * beta) / beta * gradl_comp;
        let l_div_lrad = l / (grad * lg);
        let lsc_div_lrad = 1.0 * (grad - grada) / (2.0 * grad * (gradl - grad))
            * (grad - grada - (beta * (8.0 - 3.0 * beta)) / bc * grad_mu);
        assert_rel_close(
            l_div_lrad - lsc_div_lrad - 1.0,
            0.0,
            1.0e-12,
            "Langer85 luminosity balance",
        );
    });
}

#[test]
fn test_thermohaline_species_cutoffs_and_all_brown13_regimes() {
    with_numerics_mode(NumericsMode::Strict, || {
        assert!(matches!(
            ThermohalineOption::from_str("invalid"),
            Err(TurbError::UnknownThermohalineOption(_))
        ));

        let t = 3.0e7;
        let rho = 5.0;
        let cp = 2.5e8;
        let opacity = 0.3;
        let grada = 0.40;
        let xh1 = 0.7;

        // 1. All three Brown et al. (2013) `analytical_estimate_th` regimes
        // (`thermohaline.f90:250-272`):
        // (a) r_th < 0.5 && r_th <= Pr
        let d_b1 = get_d_thermohaline(
            ThermohalineOption::BrownGaraudStellmach13,
            grada,
            0.3998,
            t,
            opacity,
            rho,
            cp,
            -1.0e-4,
            DrivingSpecies::he3(),
            xh1,
            1.0,
        );
        // (b) r_th < 0.5 && r_th > Pr
        let d_b2 = get_d_thermohaline(
            ThermohalineOption::BrownGaraudStellmach13,
            grada,
            0.30,
            t,
            opacity,
            rho,
            cp,
            -1.0e-4,
            DrivingSpecies::he3(),
            xh1,
            1.0,
        );
        // (c) r_th >= 0.5 && r_th < 1.0
        let d_b3 = get_d_thermohaline(
            ThermohalineOption::BrownGaraudStellmach13,
            grada,
            0.35,
            t,
            opacity,
            rho,
            cp,
            -2.5e-9,
            DrivingSpecies::he3(),
            xh1,
            1.0,
        );
        assert!(d_b1 > 0.0 && d_b1.is_finite());
        assert!(d_b2 > 0.0 && d_b2.is_finite());
        assert!(d_b3 > 0.0 && d_b3.is_finite());
        assert!(d_b1 > d_b2 && d_b2 > d_b3);

        // Linear scaling by `thermohaline_coeff` (`thermohaline.f90:91`).
        let d_b2_x3 = get_d_thermohaline(
            ThermohalineOption::BrownGaraudStellmach13,
            grada,
            0.30,
            t,
            opacity,
            rho,
            cp,
            -1.0e-4,
            DrivingSpecies::he3(),
            xh1,
            3.0,
        );
        assert_eq!(d_b2_x3.to_bits(), (3.0 * d_b2).to_bits());

        // 2. Heavy-ion `chem_z > 2` branch (`c12`, `thermohaline.f90:120-135`),
        // plus `he4` vs `he3`.
        let d_c12 = get_d_thermohaline(
            ThermohalineOption::TraxlerGaraudStellmach11,
            0.40,
            0.35,
            1.0e8,
            0.1,
            1.0e4,
            2.5e8,
            -1.0e-3,
            DrivingSpecies::c12(),
            0.0,
            1.0,
        );
        assert!(d_c12 > 0.0 && d_c12.is_finite());

        let d_he4_bgs = get_d_thermohaline(
            ThermohalineOption::BrownGaraudStellmach13,
            grada,
            0.30,
            t,
            opacity,
            rho,
            cp,
            -1.0e-4,
            DrivingSpecies::he4(),
            xh1,
            1.0,
        );
        assert_eq!(d_he4_bgs.to_bits(), d_b2.to_bits());

        // 3. `r_th >= 1.0` cutoff (`thermohaline.f90:74-75`) and `Pr < 0.0`
        // cutoff (`thermohaline.f90:76-78`).
        let d_stable = get_d_thermohaline(
            ThermohalineOption::BrownGaraudStellmach13,
            grada,
            0.30,
            t,
            opacity,
            rho,
            cp,
            -1.0e-11,
            DrivingSpecies::he3(),
            xh1,
            1.0,
        );
        assert_eq!(d_stable, 0.0);

        let d_neg_pr = get_d_thermohaline(
            ThermohalineOption::TraxlerGaraudStellmach11,
            0.40,
            0.39,
            1.0e4,
            1.0e4,
            1.0e8,
            2.5e8,
            -1.0e-4,
            DrivingSpecies::he3(),
            xh1,
            1.0,
        );
        assert_eq!(d_neg_pr, 0.0);

        // 4. `Kippenhahn` `dgrad` floor `max(1e-40, grada - gradr)`
        // (`thermohaline.f90:62, 66`) and `set_thermohaline` AD propagation.
        let d_floor = get_d_thermohaline(
            ThermohalineOption::Kippenhahn,
            0.40,
            0.45,
            t,
            opacity,
            rho,
            cp,
            -1.0e-4,
            DrivingSpecies::he3(),
            xh1,
            1.0,
        );
        let k_therm = 4.0 * CRAD * CLIGHT * pow3(t) / (3.0 * opacity * rho);
        let want_floor =
            -3.0 * k_therm / (2.0 * rho * cp) * (-1.0e-4) / 1.0e-40;
        assert_eq!(d_floor.to_bits(), want_floor.to_bits());

        let th_out = set_thermohaline(
            &ThermohalineParams {
                option: ThermohalineOption::BrownGaraudStellmach13,
                gradl_composition_term: -1.0e-11,
                xh1,
                thermohaline_coeff: 1.0,
                species: DrivingSpecies::he3(),
            },
            &ThermohalineInputs {
                lambda: 1.0e9,
                grada,
                gradr: 0.30,
                t,
                opacity,
                rho,
                cp,
            },
        )
        .expect("set_thermohaline stable");
        assert_eq!(th_out.d, 0.0);
        assert_eq!(th_out.conv_vel, 0.0);
        assert_eq!(th_out.grad_t, 0.30);
        assert_eq!(th_out.y_face, 0.30 - grada);
        assert_eq!(th_out.mixing_type, MixingType::ThermohalineMixing);
    });
}

#[test]
fn test_tdc_dual_div_real_and_star_order1_lifting() {
    let s = DualOrder1::<4>::new(6.0, [1.0, -2.0, 0.5, 4.0]);
    let lifted: AutoDiffRealTdc<4> = s.convert().with_d1val1(3.0);
    let scaled = lifted.div_real(3.0);
    assert_eq!(scaled.val, 2.0);
    assert_eq!(scaled.d1val1, 1.0);
    assert_eq!(
        scaled.d1_array,
        [1.0 / 3.0, -2.0 / 3.0, 0.5 / 3.0, 4.0 / 3.0]
    );
    let deriv = scaled.differentiate_1();
    assert_eq!(deriv.val, 1.0);
    assert_eq!(DualOrder1::<4>::unconvert(scaled).d1, scaled.d1_array);
}
