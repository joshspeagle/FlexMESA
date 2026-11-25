// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` parity tests for Issue `#158`:
//! - `hydro_rotation`: Fabry+2022 fits, `eval_fp_ft`, `eval_i_rot_dual`,
//!   `w_div_w_roche_{omega,jrot}` sigmoid & hard-cap branches, `set_j_rot`,
//!   `set_i_rot_from_omega_and_j_rot`, `get1_am_sig`, `get_rotation_sigmas`,
//!   and `TidalDeformationHook` blending.
//! - `rotation`: `update_rotation_distortion_factors`,
//!   `conserve_shell_j_rot_and_distortion`, and Wave-5 invariant checks.
//! - `rotation_mix`: `mix_info.f90:1561-1956` branches (`d_omega_flag = false`,
//!   `am_nu_rot_flag = true`, `set_uniform_am_nu_non_rot`,
//!   `set_min_am_nu_non_rot`, `augment_d_mix_with_rotation`,
//!   `sync_post_henyey_rotation`, and error validation).
//! - `omega_mix`: `solve_omega_mix`, `do_solve_omega_mix_with_extras`
//!   zero-gradient surface BC, conservation across stiffness clamps, and
//!   `BinaryOtherTorqueHook` implicit Newton convergence.

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use std::sync::Arc;

use flexmesa_const::{
    MixingType, M_SUN, R_SUN, SECYER, STANDARD_CGRAV, TWO_THIRDS,
};
use flexmesa_star::controls::{StarControls, StarError, StarJobControls};
use flexmesa_star::hydro_rotation::{
    eval_fp_ft, eval_fp_ft_single, eval_fp_ft_single_dual, eval_i_rot,
    eval_i_rot_dual, get1_am_sig, get_rotation_sigmas, re_from_rpsi_factor,
    roche_a, roche_b, roche_c, roche_sigmoid, rpsi_from_re_factor, set_i_rot,
    set_i_rot_from_omega_and_j_rot, set_j_rot, set_omega, set_rotation_info,
    total_angular_momentum, use_xh_to_update_i_rot,
    use_xh_to_update_i_rot_and_j_rot, w_div_w_roche_jrot, w_div_w_roche_omega,
    DEFAULT_W_DIV_WCRIT_MAX2, LOG_TERM_POWER,
};
use flexmesa_star::hydro_state::{
    BinaryOtherTorqueHandle, BinaryOtherTorqueHook, TidalDeformationHandle,
    TidalDeformationHook, TidalShellDeformation,
};
use flexmesa_star::inlist_runner::{DefaultStarExtras, StarExtras};
use flexmesa_star::mix_info::set_mixing_info;
use flexmesa_star::omega_mix::{
    do_solve_omega_mix_with_extras, solve_omega_mix, OmegaMixOutcome,
};
use flexmesa_star::rotation::{
    augment_d_mix_with_rotation, conserve_shell_j_rot_and_distortion,
    eval_magnetic_diffusivity, eval_omega_crit, eval_rotation_fp_ft,
    eval_rotational_mixing_zone, eval_thermal_diffusivity,
    sync_post_henyey_rotation, update_rotation_distortion_factors,
};
use flexmesa_star::{load_zams_model, set_hydro_vars, StarModel};

struct ConstTidalHook {
    f_switch: f64,
    fp: f64,
    d_fp_dlnr: f64,
    ft: f64,
    d_ft_dlnr: f64,
    i_rot_fac: f64,
    d_irot_dlnr_fac: f64,
}

impl TidalDeformationHook for ConstTidalHook {
    fn eval_shell(
        &self,
        _k: usize,
        r_cm: f64,
        _omega: f64,
    ) -> TidalShellDeformation {
        TidalShellDeformation {
            fp: self.fp,
            d_fp_dlnr: self.d_fp_dlnr,
            ft: self.ft,
            d_ft_dlnr: self.d_ft_dlnr,
            i_rot: self.i_rot_fac * r_cm * r_cm,
            d_irot_dlnr: self.d_irot_dlnr_fac * r_cm * r_cm,
            f_switch: self.f_switch,
        }
    }
}

struct SpinOrbitTargetTorqueHook {
    target_omega: f64,
    rate_hz: f64,
}

impl BinaryOtherTorqueHook for SpinOrbitTargetTorqueHook {
    fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
        for k in 0..model.nz {
            let om = model.omega[k];
            let ir = model.aux.i_rot[k];
            let coeff = -self.rate_hz * ir;
            model.aux.extra_jdot[k] = coeff * (om - self.target_omega);
            model.aux.d_extra_jdot_domega_00[k] = coeff;
            model.aux.d_extra_jdot_domega_m1[k] = 0.0;
            model.aux.d_extra_jdot_domega_p1[k] = 0.0;
            model.aux.extra_omegadot[k] = 0.0;
            model.aux.d_extra_omegadot_domega_00[k] = 0.0;
            model.aux.d_extra_omegadot_domega_m1[k] = 0.0;
            model.aux.d_extra_omegadot_domega_p1[k] = 0.0;
        }
        Ok(())
    }
}

struct FailingImplicitExtras;

impl StarExtras for FailingImplicitExtras {
    fn other_torque_implicit(
        &mut self,
        _model: &mut StarModel,
    ) -> Result<(), StarError> {
        Err(StarError::ConvergenceFailure(String::from(
            "synthetic implicit torque retry",
        )))
    }
}

fn load_1m_zams_rotating(omega: f64) -> StarModel {
    let mut ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        rotation_flag: true,
        ..StarControls::default()
    };
    ctrl.mix_wind.am_d_mix_factor = 1.0 / 30.0;
    ctrl.mix_wind.d_dsi_factor = 0.0;
    ctrl.mix_wind.d_sh_factor = 1.0;
    ctrl.mix_wind.d_ssi_factor = 1.0;
    ctrl.mix_wind.d_es_factor = 1.0;
    ctrl.mix_wind.d_gsf_factor = 1.0;
    ctrl.mix_wind.d_st_factor = 1.0;
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("1M ZAMS");
    model.omega.fill(omega);
    update_rotation_distortion_factors(&mut model);
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

#[test]
fn test_roche_polynomials_radii_and_autodiff_derivatives() {
    assert_eq!(LOG_TERM_POWER, 5.626);
    assert_eq!(DEFAULT_W_DIV_WCRIT_MAX2, 0.90);

    // At w = 0, every Fabry+2022 polynomial equals 1.0 and r_p = r_e = r.
    assert_eq!(rpsi_from_re_factor(0.0, 0.0, 0.0), 1.0);
    assert_eq!(re_from_rpsi_factor(0.0, 0.0, 0.0), 1.0);
    assert_eq!(roche_a(0.0, 0.0, 0.0), 1.0);
    assert_eq!(roche_b(0.0, 0.0, 0.0, 0.0), 1.0);
    assert_eq!(roche_c(0.0, 0.0, 0.0, 0.0), 1.0);
    assert_eq!(eval_fp_ft(R_SUN, 0.0), (1.0, 1.0, R_SUN, R_SUN));

    // Sigmoid maps [xmax1, inf) monotonically into [xmax1, xmax2], saturating
    // cleanly at xmax2 when the exponent underflows in IEEE-754 f64.
    assert_eq!(roche_sigmoid(0.89, 0.89, 0.90), 0.89);
    let mid_sig = roche_sigmoid(0.895, 0.89, 0.90);
    let hi_sig = roche_sigmoid(0.905, 0.89, 0.90);
    assert!(mid_sig > 0.89 && mid_sig < hi_sig && hi_sig < 0.90);
    assert_eq!(roche_sigmoid(-0.905, 0.89, 0.90), hi_sig);
    assert_eq!(roche_sigmoid(5.0, 0.89, 0.90), 0.90);

    // Central-difference verification of Fabry+2022 fp/ft and Paxton+2019 i_rot
    // partial derivatives in `eval_fp_ft_single_dual` and `eval_i_rot_dual`.
    let w = 0.65_f64;
    let dw = 1.0e-7_f64;
    let r0 = 2.0 * R_SUN;
    let (fp_ad, ft_ad) = eval_fp_ft_single_dual(w);
    let (fp_p, ft_p) = eval_fp_ft_single(w + dw);
    let (fp_m, ft_m) = eval_fp_ft_single(w - dw);
    let dfp_fd = (fp_p - fp_m) / (2.0 * dw);
    let dft_fd = (ft_p - ft_m) / (2.0 * dw);
    assert!((fp_ad.d1[0] - dfp_fd).abs() <= 1.0e-8 * dfp_fd.abs());
    assert!((ft_ad.d1[0] - dft_fd).abs() <= 1.0e-8 * dft_fd.abs());
    assert_eq!(eval_fp_ft_single(-w), eval_fp_ft_single(w));

    // `eval_fp_ft` geometry ordering: r_polar < r < r_equatorial for w > 0.
    let (fp_v, ft_v, rp, re) = eval_fp_ft(r0, -w);
    assert_eq!(fp_v, fp_ad.val);
    assert_eq!(ft_v, ft_ad.val);
    let w2 = w * w;
    assert!(
        (re / r0 - re_from_rpsi_factor(w2, w2 * w2, w2 * w2 * w2)).abs()
            < 1.0e-15
    );
    assert!((re / rp - (1.0 + 0.5 * w2)).abs() <= 1.0e-15);
    assert!(rp < r0 && r0 < re);

    // `eval_i_rot_dual` partials wrt w and ln(r), plus `simple_i_rot_flag`.
    let ir_ad = eval_i_rot_dual(r0, w, false);
    let dir_dw_fd = (eval_i_rot(r0, w + dw, false)
        - eval_i_rot(r0, w - dw, false))
        / (2.0 * dw);
    let eps_lnr = 1.0e-7_f64;
    let dir_dlnr_fd = (eval_i_rot(r0 * (1.0 + eps_lnr), w, false)
        - eval_i_rot(r0 * (1.0 - eps_lnr), w, false))
        / (2.0 * eps_lnr);
    assert!((ir_ad.d1[0] - dir_dw_fd).abs() <= 1.0e-7 * dir_dw_fd.abs());
    assert!((ir_ad.d1[1] - dir_dlnr_fd).abs() <= 1.0e-6 * dir_dlnr_fd.abs());

    let ir_simple = eval_i_rot_dual(r0, w, true);
    assert_eq!(ir_simple.val, TWO_THIRDS * r0 * r0);
    assert_eq!(ir_simple.d1[0], 0.0);
    assert_eq!(ir_simple.d1[1], 2.0 * ir_simple.val);
}

#[test]
fn test_w_div_w_roche_consistency_across_omega_and_jrot_branches() {
    // Verify self-consistency: for target w in (0, max_w), Fabry+2022 forward
    // omega(w) and j_rot(w) invert back to w within MESA's 1e-6 bisection
    // bound for both prograde and retrograde rotation. Notice `eval_i_rot`
    // uses Paxton+2019 while `w_div_w_roche_jrot` uses Fabry+2022 C/A, so
    // `eval_i_rot * omega` agrees closely but not to 1e-6 at high `w`.
    for w_target in [0.15_f64, 0.45, 0.78] {
        let w2 = w_target * w_target;
        let w4 = w2 * w2;
        let w6 = w4 * w2;
        let dimless_r = flexmesa_math::pow(w_target, TWO_THIRDS)
            * rpsi_from_re_factor(w2, w4, w6);
        let om_c_scale =
            (STANDARD_CGRAV * M_SUN / (R_SUN * R_SUN * R_SUN)).sqrt();
        let omega = flexmesa_math::pow(dimless_r, 1.5) * om_c_scale;
        let w_log_term = flexmesa_math::log(
            1.0 - flexmesa_math::pow(w_target, LOG_TERM_POWER),
        );
        let j_rot = TWO_THIRDS * w_target * roche_c(w2, w4, w6, w_log_term)
            / roche_a(w4, w6, w_log_term)
            * (STANDARD_CGRAV * M_SUN * R_SUN).sqrt();
        let j_rot_paxton = eval_i_rot(R_SUN, w_target, false) * omega;

        let w_om = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            omega,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        let w_jr = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            j_rot,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        let w_jr_paxton = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            j_rot_paxton,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        assert!((w_om - w_target).abs() <= 5.0e-7);
        assert!((w_jr - w_target).abs() <= 5.0e-7);
        assert!((w_jr_paxton - w_target).abs() < 2.0e-2);

        let w_om_neg = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            -omega,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        let w_jr_neg = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            -j_rot,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        assert_eq!(w_om_neg, -w_om);
        assert_eq!(w_jr_neg, -w_jr);
    }

    // Mild supercritical rotation lands strictly inside (-max_w2, -max_w),
    // while extreme supercritical input saturates to -max_w2 within bisection.
    let jr_lim = |w: f64| -> f64 {
        let w2 = w * w;
        let w4 = w2 * w2;
        let w6 = w4 * w2;
        let log_t =
            flexmesa_math::log(1.0 - flexmesa_math::pow(w, LOG_TERM_POWER));
        TWO_THIRDS * w * roche_c(w2, w4, w6, log_t) / roche_a(w4, w6, log_t)
    };
    let j_mid = -0.5
        * (jr_lim(0.89) + jr_lim(0.90))
        * (STANDARD_CGRAV * M_SUN * R_SUN).sqrt();
    let w_sig_neg_om = w_div_w_roche_omega(
        R_SUN,
        M_SUN,
        -4.48e-4,
        STANDARD_CGRAV,
        0.89,
        0.90,
        true,
    );
    let w_sig_neg_jr = w_div_w_roche_jrot(
        R_SUN,
        M_SUN,
        j_mid,
        STANDARD_CGRAV,
        0.89,
        0.90,
        true,
    );
    assert!((-0.90..-0.89).contains(&w_sig_neg_om));
    assert!((-0.90..-0.89).contains(&w_sig_neg_jr));
    let w_sat_neg_om = w_div_w_roche_omega(
        R_SUN,
        M_SUN,
        -1.0e-2,
        STANDARD_CGRAV,
        0.89,
        0.90,
        true,
    );
    let w_sat_neg_jr = w_div_w_roche_jrot(
        R_SUN,
        M_SUN,
        -1.0e22,
        STANDARD_CGRAV,
        0.89,
        0.90,
        true,
    );
    assert!((w_sat_neg_om + 0.90).abs() <= 1.0e-6);
    assert!((w_sat_neg_jr + 0.90).abs() <= 1.0e-6);
}

#[test]
fn test_hydro_rotation_setters_tidal_blending_and_sigmas() {
    let mut model = load_1m_zams_rotating(3.0e-5);
    let nz = model.nz;

    // `use_xh_to_update_i_rot_and_j_rot` + `set_j_rot` + `set_omega`.
    use_xh_to_update_i_rot_and_j_rot(&mut model);
    let j_tot_init = total_angular_momentum(&model);
    assert!(j_tot_init > 0.0);

    // Perturb `omega`, verify `set_j_rot` updates `j_rot`, then `set_omega`
    // restores `omega = j_rot / i_rot` and `use_xh_to_update_i_rot` conserves J.
    let saved_j = model.j_rot.clone();
    model.omega.fill(6.0e-5);
    set_j_rot(&mut model);
    assert!((total_angular_momentum(&model) - 2.0 * j_tot_init).abs() < 1.0e40);

    model.j_rot.copy_from_slice(&saved_j);
    use_xh_to_update_i_rot(&mut model);
    set_omega(&mut model);
    assert!(
        (total_angular_momentum(&model) - j_tot_init).abs()
            <= 1e-12 * j_tot_init
    );

    // `set_i_rot_from_omega_and_j_rot`: exact ratio when `omega != 0`, and
    // falls back to `update1_i_rot_from_xh` when `omega == 0`.
    model.omega[0] = 0.0;
    model.omega[1] = 2.5e-5;
    model.j_rot[1] = 5.0e17;
    set_i_rot_from_omega_and_j_rot(&mut model);
    assert!((model.aux.i_rot[1] - 2.0e22).abs() <= 1.0e8);
    let expected_i0 = eval_i_rot(
        flexmesa_math::exp(model.ln_r[0]),
        model.aux.w_div_wc[0],
        false,
    );
    assert!((model.aux.i_rot[0] - expected_i0).abs() <= 1.0e-12 * expected_i0);

    // `TidalDeformationHook` blending across `set_rotation_info`,
    // `update_rotation_distortion_factors`, and `conserve_shell_j_rot_and_distortion`.
    model.omega.fill(3.0e-5);
    update_rotation_distortion_factors(&mut model);
    let j_before_tide = total_angular_momentum(&model);
    model.aux.tidal_deformation =
        Some(TidalDeformationHandle(Arc::new(ConstTidalHook {
            f_switch: 0.25,
            fp: 0.72,
            d_fp_dlnr: -0.16,
            ft: 0.84,
            d_ft_dlnr: -0.08,
            i_rot_fac: 0.80,
            d_irot_dlnr_fac: 1.60,
        })));
    conserve_shell_j_rot_and_distortion(&mut model);
    assert!(
        (total_angular_momentum(&model) - j_before_tide).abs()
            <= 1.0e-13 * j_before_tide
    );
    let (fp_s0, ft_s0) = eval_fp_ft_single(model.aux.w_div_wc[0]);
    assert!(
        (model.aux.fp_rot[0] - (0.25 * 0.72 + 0.75 * fp_s0)).abs() < 1.0e-14
    );
    assert!(
        (model.aux.ft_rot[0] - (0.25 * 0.84 + 0.75 * ft_s0)).abs() < 1.0e-14
    );
    assert!((model.aux.d_fp_rot_dlnr[0] - (-0.04)).abs() < 1.0e-15);
    assert!((model.aux.d_ft_rot_dlnr[0] - (-0.02)).abs() < 1.0e-15);

    // `get1_am_sig` & `get_rotation_sigmas`: closed-form harmonic average,
    // `am_sig[nz-1] == 0`, and `am_sig_term_limit * dmbar / dt` stiffness cap.
    model.aux.am_nu_j.fill(1.0e10);
    model.aux.am_nu_omega.fill(3.0e10);
    model.controls.mix_wind.am_sig_term_limit = 1.0e13;
    let dt = 1.0e11;
    get_rotation_sigmas(&mut model, 0, nz, dt);
    assert_eq!(model.aux.am_sig_j[nz - 1], 0.0);
    assert_eq!(model.aux.am_sig_omega[nz - 1], 0.0);

    // Check stiffness clamp using tiny `am_sig_term_limit`.
    let mut am_sig_clamped = vec![0.0_f64; nz];
    let nu_huge = vec![1.0e28_f64; nz];
    model.controls.mix_wind.am_sig_term_limit = 50.0;
    get1_am_sig(&model, 0, nz, &nu_huge, &mut am_sig_clamped, 1.0e5);
    for k in 0..(nz - 1) {
        let dmbar = model.xmstar * model.dq[k].min(model.dq[k + 1]);
        let expected_lim = 50.0 * dmbar / 1.0e5;
        assert!(am_sig_clamped[k] <= expected_lim * (1.0 + 1e-15));
    }
    set_rotation_info(&mut model, false);
    set_i_rot(&mut model, true);
    assert_eq!(model.aux.i_rot.len(), nz);
}

#[test]
fn test_rotation_mix_non_d_omega_uniform_min_non_rot_and_errors() {
    let mut m = load_1m_zams_rotating(2.0e-5);
    let nz = m.nz;

    // 1. Non-D_omega mode: `D_mix_rotation = f * am_D_mix_factor * sum(D_*)`,
    //    `set_uniform_am_nu_non_rot`, `augment_d_mix_with_rotation`.
    m.controls.d_omega_flag = false;
    m.controls.am_nu_rot_flag = false;
    m.controls.mix_wind.set_uniform_am_nu_non_rot = true;
    m.controls.mix_wind.uniform_am_nu_non_rot = 4.2e15;
    set_mixing_info(&mut m).expect("set_mixing_info non-D_omega");
    assert!(m.aux.am_nu_non_rot.iter().all(|&v| v == 4.2e15));
    assert_eq!(m.aux.rotmix.d_mix_rotation[0], 0.0);

    let k_rad = (1..(nz - 1))
        .find(|&k| {
            m.mixing_type[k] == MixingType::RotationMixing
                && m.aux.rotmix.d_mix_rotation[k] > 0.0
        })
        .expect("radiative shear face marked RotationMixing");
    let mw = &m.controls.mix_wind;
    let rm = &m.aux.rotmix;
    let expected_rot = mw.am_d_mix_factor
        * (mw.d_dsi_factor * rm.d_dsi[k_rad]
            + mw.d_sh_factor * rm.d_sh[k_rad]
            + mw.d_ssi_factor * rm.d_ssi[k_rad]
            + mw.d_es_factor * rm.d_es[k_rad]
            + mw.d_gsf_factor * rm.d_gsf[k_rad]
            + mw.d_st_factor * rm.d_st[k_rad]);
    assert!(
        (m.aux.rotmix.d_mix_rotation[k_rad] - expected_rot).abs()
            <= 1.0e-12 * expected_rot
    );

    // 2. `set_min_am_nu_non_rot` threshold on center `ye[nz - 1]`.
    m.controls.mix_wind.set_uniform_am_nu_non_rot = false;
    m.controls.mix_wind.set_min_am_nu_non_rot = true;
    m.controls.mix_wind.min_am_nu_non_rot = 1.0e10;
    m.controls.mix_wind.min_center_ye_for_min_am_nu_non_rot = 0.50;
    m.ye[nz - 1] = 0.60;
    set_mixing_info(&mut m).expect("min_am_nu_non_rot active");
    assert!(m.aux.am_nu_non_rot.iter().all(|&v| v >= 1.0e10));
    m.ye[nz - 1] = 0.40;
    m.d_mix.fill(0.0);
    m.aux.mixing.d_mix_non_rotation.fill(0.0);
    augment_d_mix_with_rotation(&mut m).expect("min_am_nu_non_rot inactive");
    assert!(m.aux.am_nu_non_rot.iter().all(|&v| v == 0.0));

    // 3. Error paths matching `star/private/mix_info.f90:1648-1667, 1706-1724`.
    m.controls.d_omega_flag = true;
    m.controls.am_nu_rot_flag = false;
    m.job.use_d_omega_for_am_nu_rot = false;
    m.controls.mix_wind.am_nu_st_factor = 5.0; // differs from d_st_factor=1.0
    let err_mismatch = set_mixing_info(&mut m)
        .expect_err("mismatched am_nu >= 0 factor must fail loudly");
    assert!(matches!(err_mismatch, StarError::InvalidInput(_)));

    // Negative `am_nu_omega` error path.
    m.controls.mix_wind.am_nu_st_factor = 1.0;
    m.controls.mix_wind.set_uniform_am_nu_non_rot = true;
    m.controls.mix_wind.uniform_am_nu_non_rot = -10.0;
    let err_neg =
        set_mixing_info(&mut m).expect_err("negative am_nu_omega must error");
    assert!(matches!(err_neg, StarError::ConvergenceFailure(_)));
}

#[test]
fn test_rotation_mix_am_nu_rot_convective_zeroing_and_sync_post_henyey() {
    let mut m = load_1m_zams_rotating(2.0e-5);
    let nz = m.nz;
    m.controls.d_omega_flag = false;
    m.controls.am_nu_rot_flag = true;
    m.controls
        .mix_wind
        .max_q_for_nu_omega_zero_in_convection_region = 0.80;
    m.controls.mix_wind.smooth_am_nu_rot = 0;
    m.controls.mix_wind.nu_omega_mixing_rate = 0.0;
    set_mixing_info(&mut m).expect("am_nu_rot_flag setup");

    // Inner convective faces (`q <= 0.8`) must have `am_nu_rot == 0`.
    let mut found_zeroed_conv = false;
    for k in 0..nz {
        if m.mixing_type[k] == MixingType::ConvectiveMixing && m.q[k] <= 0.80 {
            assert_eq!(m.aux.am_nu_rot[k], 0.0);
            found_zeroed_conv = true;
        }
    }
    assert!(found_zeroed_conv);

    // `sync_post_henyey_rotation` populates viscosities if empty, respects
    // `premix_omega`, diffuses differential shear while conserving J, and
    // updates distortion factors.
    for (k, w) in m.omega.iter_mut().enumerate().take(nz) {
        *w = if k < nz / 2 { 1.0e-5 } else { 4.0e-5 };
    }
    update_rotation_distortion_factors(&mut m);
    let j_before = total_angular_momentum(&m);
    m.aux.am_nu_omega.fill(0.0);
    m.aux.am_nu_j.fill(0.0);
    m.controls.mix_wind.premix_omega = true;
    sync_post_henyey_rotation(&mut m, 1.0e4 * SECYER)
        .expect("sync_post_henyey_rotation");
    let j_after = total_angular_momentum(&m);
    assert!((j_after - j_before).abs() <= 1.0e-12 * j_before);
    assert!(m.aux.am_nu_omega.iter().any(|&v| v > 0.0));
    assert_eq!(m.omega[0], m.omega[1]);
}

#[test]
fn test_omega_mix_standalone_conservation_and_binary_implicit_spin_orbit() {
    let mut m = load_1m_zams_rotating(2.0e-5);
    let nz = m.nz;

    // Differential rotation in standalone `solve_omega_mix`: zero-gradient
    // surface BC and exact J conservation.
    for k in 0..nz {
        m.omega[k] = 1.0e-5 + 3.0e-5 * (1.0 - m.q[k]);
    }
    update_rotation_distortion_factors(&mut m);
    let j0 = total_angular_momentum(&m);
    let nu = vec![1.0e14_f64; nz];
    solve_omega_mix(&mut m, 1.0e11, &nu, 4);
    let j1 = total_angular_momentum(&m);
    assert!((j1 - j0).abs() <= 1.0e-12 * j0);
    assert_eq!(m.omega[0], m.omega[1]);
    assert!(m.omega[nz - 1] < 4.0e-5 && m.omega[0] > 1.0e-5);

    // `BinaryOtherTorqueHook` implicit synchronization toward `target_omega`:
    // exponential relaxation Omega(t) -> target_omega without false J-retry
    // errors (`has_external_torque` preserved for binary hooks).
    let target = 5.0e-5_f64;
    m.omega.fill(1.0e-5);
    update_rotation_distortion_factors(&mut m);
    m.aux.am_nu_omega.fill(1.0e10);
    m.aux.am_nu_j.fill(0.0);
    m.aux.binary_other_torque_implicit =
        Some(BinaryOtherTorqueHandle::new(SpinOrbitTargetTorqueHook {
            target_omega: target,
            rate_hz: 1.0e-10,
        }));
    let mut extras = DefaultStarExtras;
    let out = do_solve_omega_mix_with_extras(&mut m, 5.0e11, &mut extras)
        .expect("binary_other_torque_implicit");
    assert_eq!(out, OmegaMixOutcome::KeepGoing);
    assert!(m.omega.iter().all(|&w| w > 3.5e-5 && w < target));

    // Upstream `solve_omega_mix.f90:201-207`: error in `other_torque_implicit`
    // returns `OmegaMixOutcome::Retry`.
    m.controls.use_other_torque_implicit = true;
    let mut fail_extras = FailingImplicitExtras;
    let retry_out =
        do_solve_omega_mix_with_extras(&mut m, 1.0e10, &mut fail_extras)
            .expect("implicit hook error yields Retry");
    assert!(matches!(retry_out, OmegaMixOutcome::Retry(_)));
}

#[test]
fn test_rotation_wave5_kernels_physical_invariants() {
    let (fp0, ft0, ir0) = eval_rotation_fp_ft(0.0, 0.5, 0.5);
    assert_eq!((fp0, ft0, ir0), (1.0, 1.0, TWO_THIRDS));

    let (fp_hi, ft_hi, ir_hi) = eval_rotation_fp_ft(0.85, 0.55, 0.65);
    assert!((0.55..1.0).contains(&fp_hi));
    assert!((0.65..1.0).contains(&ft_hi));
    assert!(ir_hi > TWO_THIRDS);

    let om_c =
        eval_omega_crit(STANDARD_CGRAV, M_SUN, R_SUN, 3.828e33, 0.34, 0.99);
    assert!(om_c.is_finite() && om_c > 0.0);

    let k_th = eval_thermal_diffusivity(1.0e7, 10.0, 1.0, 2.0e8);
    let eta = eval_magnetic_diffusivity(1.0e7, 10.0, 1.3, 1.1);
    assert!(k_th > 0.0 && eta > 0.0);

    let z_zero = eval_rotational_mixing_zone(
        3.0e10, 5.0e9, 2.5e4, 5.0, 1.5e7, 3.5e8, 0.5, 1.0, 1.0, 0.4, 0.2,
        1.0e-6, 1.0e-7, 0.0, 0.5, 1.3, 1.1, 1.0,
    );
    assert_eq!(z_zero.d_st, 0.0);
    assert_eq!(z_zero.nu_st, 0.0);
    assert_eq!(z_zero.total_d_rot(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 0.0);
}
