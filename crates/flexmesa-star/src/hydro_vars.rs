// SPDX-License-Identifier: LGPL-3.0-or-later
//! Face-centered temperature gradients, Mixing-Length Theory, Time-Dependent
//! Convection, Semiconvection, and Thermohaline Mixing (`set_mlt_vars`,
//! `do1_mlt_2`, `check_for_redo_MLT`, `set_gradT_excess_alpha` in
//! `star/private/turb_info.f90` and `star/private/turb_support.f90`), and
//! top-level microphysics/hydro variable evaluation (`set_hydro_vars` in
//! `star/private/hydro_vars.f90`).

use flexmesa_const::{MixingType, CLIGHT, CRAD, FOUR_PI, LN10};
use flexmesa_eos::{
    I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD, I_LN_E, I_LN_PGAS, I_PHASE,
};
use flexmesa_turb::{
    set_mlt, set_semiconvection, set_tdc, set_thermohaline, MltInputs,
    MltOption, MltParams, SemiconvectionInputs, SemiconvectionParams,
    TdcInputs, TdcParams, ThermohalineInputs, ThermohalineParams,
};

use crate::controls::StarError;
pub use crate::hydro_brunt_surf::{do_brunt_b, do_brunt_n2, eval_surf_bc};
use crate::micro_cache::MicrophysicsCache;
use crate::mixing_state::driving_species_for_cid;
use crate::model::StarModel;
use crate::rotation::update_rotation_distortion_factors;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LNR_00, I_L_00,
};

/// Evaluates face-centered radiative, adiabatic, Ledoux, and actual temperature
/// gradients ($\nabla_{\mathrm{rad}}, \nabla_{\mathrm{ad}}, \nabla_L, \nabla_T$),
/// pressure scale height $H_P$, convective velocity $v_{\mathrm{conv}}$, and
/// diffusivity $D_{\mathrm{MLT}}$ with 3-cell stencil [`StarAdOrder1`] partials
/// across all faces `k = 0..nz` (`set_mlt_vars` in
/// `star/private/turb_info.f90:40-67`).
///
/// Writes only to MLT outputs (`mlt_mixing_type`, `mlt_d`, `mlt_cdc`, `mlt_vc`,
/// `mlt_gamma`, `mlt_mixing_length`, `gradt`, `gradr`, `gradl`, `grada_face`,
/// `scale_height`) and never overwrites `model.mixing_type`, `model.d_mix`, or
/// `model.conv_vel`.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `set_mlt` / `set_tdc` fails at
/// any face.
pub fn set_mlt_vars(model: &mut StarModel) -> Result<(), StarError> {
    if model.controls.rotation_flag && !model.aux.doing_struct_burn_mix {
        if model.aux.solver_mlt_eval_count > 0 {
            crate::rotation::conserve_shell_j_rot_and_distortion(model);
        } else {
            update_rotation_distortion_factors(model);
        }
    }
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.aux.fixed_gradr_sticky.len() != nz {
        model.aux.fixed_gradr_sticky.resize(nz, false);
    }
    if model.aux.mixing.mlt_mixing_type.len() != nz {
        model.aux.mixing.resize(nz);
    }
    if !model.aux.doing_struct_burn_mix {
        model.aux.solver_mlt_eval_count = 0;
        model.aux.fixed_gradr_sticky.fill(false);
    }
    set_mlt_vars_range(model, 0, nz - 1)
}

/// Evaluates `do1_mlt_2` over the inclusive face range `k_lo..=k_hi` and
/// applies `make_gradr_sticky_in_solver_iters` after solver iteration 3
/// (`set_mlt_vars` in `star/private/turb_info.f90:40-67`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if any face MLT/TDC evaluation fails.
pub(crate) fn set_mlt_vars_range(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 || k_lo > k_hi || k_hi >= nz {
        return Ok(());
    }
    if model.aux.fixed_gradr_sticky.len() != nz {
        model.aux.fixed_gradr_sticky.resize(nz, false);
    }
    if model.aux.mixing.mlt_mixing_type.len() != nz {
        model.aux.mixing.resize(nz);
    }
    let solver_iter = if model.aux.doing_struct_burn_mix {
        model.aux.solver_mlt_eval_count
    } else {
        0
    };
    for k in k_lo..=k_hi {
        let make_sticky = do1_mlt_2(model, k, None, None)?;
        if make_sticky
            && solver_iter > 3
            && !model.aux.fixed_gradr_sticky[k]
        {
            model.aux.fixed_gradr_sticky[k] =
                model.aux.mixing.mlt_mixing_type[k] == MixingType::NoMixing;
        }
    }
    Ok(())
}

/// Updates Lagrangian mass-diffusion coefficients `model.sig` from `model.d_mix`
/// by delegating to [`crate::mix_info::compute_convection_sigmas`].
pub fn update_sig_from_d_mix(
    model: &mut StarModel,
) -> Result<(), StarError> {
    let eff_mix_factor = if model.controls.do_mix {
        model.controls.mix_factor
    } else {
        0.0
    };
    crate::mix_info::compute_convection_sigmas(
        model,
        model.dt,
        eff_mix_factor,
        model.controls.sig_term_limit,
    )
}

/// Switches face `k` to radiative transport for MLT variables
/// (`switch_to_radiative` in `star/private/turb_info.f90:389-399`).
pub(crate) fn switch_to_radiative(model: &mut StarModel, k: usize) {
    if k >= model.nz {
        return;
    }
    model.aux.mixing.mlt_mixing_type[k] = MixingType::NoMixing;
    model.aux.mixing.mlt_mixing_length[k] = 0.0;
    model.mlt_d[k] = 0.0;
    model.mlt_cdc[k] = 0.0;
    model.mlt_vc[k] = 0.0;
    model.gradt_ad[k] = model.gradr_ad[k];
    model.gradt[k] = model.gradt_ad[k].val;
    model.aux.mixing.gradt_sub_grada[k] =
        model.gradt[k] - model.aux.mixing.grada_face[k];
}

/// Evaluates MLT/TDC/semiconvective/thermohaline quantities at a single face `k`
/// (`do1_mlt_2` in `star/private/turb_info.f90:70-334` and `do1_mlt_eval` /
/// `Get_results` in `star/private/turb_support.f90:111-510`).
///
/// Returns `Ok(make_gradr_sticky_in_solver_iters)`.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `k >= nz`, `gradr_factor` is
/// non-finite, or `set_mlt` / `set_tdc` / `set_semiconvection` /
/// `set_thermohaline` fails.
pub(crate) fn do1_mlt_2(
    model: &mut StarModel,
    k: usize,
    mixing_length_alpha_in: Option<f64>,
    gradl_composition_term_in: Option<f64>,
) -> Result<bool, StarError> {
    let nz = model.nz;
    if k >= nz {
        return Err(StarError::MicrophysicsError(format!(
            "bad k for do1_mlt_2: k={k}, nz={nz}"
        )));
    }
    if model.aux.mixing.mlt_mixing_type.len() != nz {
        model.aux.mixing.resize(nz);
    }

    let mixing_length_alpha =
        mixing_length_alpha_in.unwrap_or(model.controls.mixing_length_alpha);
    let gradl_composition_term =
        if let Some(b_in) = gradl_composition_term_in {
            b_in
        } else if model.controls.use_ledoux_criterion {
            model.aux.mixing.gradl_composition_term[k]
        } else {
            0.0
        };

    let (alfa, beta_w) = model.get_face_weights(k);

    // Match `wrap_d_00` and `wrap_T_00` (`auto_diff_support.f90:257-276`),
    // reading `model.rho[k]` and `model.t[k]` directly (reproduces upstream
    // `#169 / #277` during `predictive_mix` trial MLT calls).
    let rho_00 =
        StarAdOrder1::from_dlnd_dlnt_00(model.rho[k], model.rho[k], 0.0);
    let t_00 = StarAdOrder1::from_dlnd_dlnt_00(model.t[k], 0.0, model.t[k]);
    let pgas_00 = StarAdOrder1::from_dlnd_dlnt_00(
        model.ln_pgas[k],
        model.eos_d_dlnd[k][I_LN_PGAS],
        model.eos_d_dlnt[k][I_LN_PGAS],
    )
    .exp();
    let prad_00 = (CRAD / 3.0) * t_00.pow4();
    let p_00 = pgas_00 + prad_00;

    let (rho_f, t_f, p_f) = if k == 0 {
        (rho_00, t_00, p_00)
    } else {
        let km1 = k - 1;
        let rho_m1 = StarAdOrder1::from_dlnd_dlnt_m1(
            model.rho[km1],
            model.rho[km1],
            0.0,
        );
        let t_m1 =
            StarAdOrder1::from_dlnd_dlnt_m1(model.t[km1], 0.0, model.t[km1]);
        let pgas_m1 = StarAdOrder1::from_dlnd_dlnt_m1(
            model.ln_pgas[km1],
            model.eos_d_dlnd[km1][I_LN_PGAS],
            model.eos_d_dlnt[km1][I_LN_PGAS],
        )
        .exp();
        let prad_m1 = (CRAD / 3.0) * t_m1.pow4();
        let p_m1 = pgas_m1 + prad_m1;
        (
            rho_00 * alfa + rho_m1 * beta_w,
            t_00 * alfa + t_m1 * beta_w,
            p_00 * alfa + p_m1 * beta_w,
        )
    };
    // `get_Prad_face = crad*pow4(get_T_face(s,k))/3d0`
    // (`star/private/star_utils.f90:3838-3843`): the face radiation pressure
    // comes from the face temperature, not from the cell pressures.
    let prad_f = CRAD * t_f.pow4() / 3.0;

    let kap_f = model.interp_cell_ad_to_face(
        &model.opacity,
        &model.d_opacity_dlnd,
        &model.d_opacity_dlnt,
        k,
    );
    let grada_face_ad = model.interp_eos_ad_to_face(&model.grada, I_GRAD_AD, k);
    let cp_f = model.interp_eos_ad_to_face(&model.cp, I_CP, k);
    let chi_rho_f = model.interp_eos_ad_to_face(&model.chi_rho, I_CHI_RHO, k);
    let chi_t_f = model.interp_eos_ad_to_face(&model.chi_t, I_CHI_T, k);

    let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_k = lnr_k.exp();
    let lum_k = StarAdOrder1::variable(model.lum[k], I_L_00);
    let cgrav_k = model.cgrav[k];
    let m_grav_k = model.m_grav[k];
    let gm_k = cgrav_k * m_grav_k;
    // Upstream `get_scale_height_face` (`star_utils.f90:4001`) and
    // `Get_results` (`turb_support.f90:261`) use spherical `G*m_grav/r^2`.
    let grav_f = gm_k / (r_k * r_k);
    let hp_std = p_f / (grav_f * rho_f);
    let scale_height_ad = if model.controls.mix_wind.alt_scale_height_flag {
        let hp_alt = (p_f / cgrav_k).sqrt() / rho_f;
        if !hp_std.val.is_finite() || hp_alt.val < hp_std.val {
            hp_alt
        } else {
            hp_std
        }
    } else {
        hp_std
    };

    let gradr_base =
        p_f * kap_f * lum_k / (4.0 * FOUR_PI * CLIGHT * m_grav_k * cgrav_k * prad_f);
    let base_gradr_fac = model.aux.mixing.gradr_factor[k];
    let gradr_factor = if model.controls.rotation_flag
        && model.controls.mix_wind.mlt_use_rotation_correction
    {
        let fp = model.aux.fp_rot.get(k).copied().unwrap_or(1.0);
        let ft = model.aux.ft_rot.get(k).copied().unwrap_or(1.0);
        (ft / fp) * base_gradr_fac
    } else {
        base_gradr_fac
    };
    if !gradr_factor.is_finite() {
        return Err(StarError::MicrophysicsError(format!(
            "non-finite gradr_factor at face {k}: {gradr_factor}"
        )));
    }
    let gradr_ad = gradr_base * gradr_factor;

    let mw = &model.controls.mix_wind;

    // Early no-mixing gates (`star/private/turb_info.f90:138-223`).
    if k == 0 && mw.mlt_make_surface_no_mixing {
        set_no_mixing_face(
            model,
            k,
            grada_face_ad,
            gradr_ad,
            scale_height_ad,
            mixing_length_alpha,
            MixingType::NoMixing,
        );
        return Ok(false);
    }

    let crystal_pad = mw.min_dq * model.m[0] * 0.5;
    let phase_k = model.eos_res.get(k).map_or(0.0, |r| r[I_PHASE]);
    let m_cryst = model.aux.crystal_core_boundary_mass;
    let m_mix = model.aux.phase_sep_mixing_mass;
    // `turb_info.f90:143-159` tests the boundary masses without a sign gate;
    // the `-1` sentinels (`init.f90:770-771`) never satisfy either test.
    if (phase_k > 0.5 && model.mu[k] > 1.7)
        || m_cryst + crystal_pad > model.m[k]
    {
        set_no_mixing_face(
            model,
            k,
            grada_face_ad,
            gradr_ad,
            scale_height_ad,
            mixing_length_alpha,
            MixingType::Crystallized,
        );
        return Ok(false);
    }

    if model.m[k] <= m_mix {
        set_no_mixing_face(
            model,
            k,
            grada_face_ad,
            gradr_ad,
            scale_height_ad,
            mixing_length_alpha,
            MixingType::PhaseSeparationMixing,
        );
        return Ok(false);
    }

    let lnt_start_k = model
        .ln_t_start
        .get(k)
        .copied()
        .filter(|&v| v > 0.0)
        .unwrap_or(model.ln_t[k]);
    if lnt_start_k / LN10 > mw.max_logt_for_mlt {
        set_no_mixing_face(
            model,
            k,
            grada_face_ad,
            gradr_ad,
            scale_height_ad,
            mixing_length_alpha,
            MixingType::NoMixing,
        );
        return Ok(false);
    }

    if mw.no_mlt_below_shock
        && (model.controls.u_flag || model.controls.v_flag)
    {
        let vel = if model.controls.u_flag {
            &model.u
        } else {
            &model.v
        };
        if vel.len() == nz {
            for i in (0..k).rev() {
                let cs = model.csound[i];
                if vel[i + 1] >= cs && vel[i] < cs {
                    set_no_mixing_face(
                        model,
                        k,
                        grada_face_ad,
                        gradr_ad,
                        scale_height_ad,
                        mixing_length_alpha,
                        MixingType::NoMixing,
                    );
                    return Ok(false);
                }
            }
        }
    }

    let cs_start_k = model.aux.csound_start.get(k).copied().unwrap_or(0.0);
    if cs_start_k > 0.0 && (model.controls.u_flag || model.controls.v_flag) {
        let vel = if model.controls.u_flag {
            &model.u_start
        } else {
            &model.v_start
        };
        if vel.len() == nz {
            let mut no_mix = vel[k] / 1.0e5 > mw.max_v_for_convection
                || model.q[k] > mw.max_q_for_convection_with_hydro_on
                || vel[k].abs() >= cs_start_k * mw.max_v_div_cs_for_convection;
            if !no_mix && model.controls.u_flag {
                let abs_du_div_cs = if k == 0 {
                    1.0e99
                } else if k + 1 < nz {
                    (vel[k] - vel[k + 1])
                        .abs()
                        .max((vel[k] - vel[k - 1]).abs())
                        / cs_start_k
                } else {
                    0.0
                };
                if abs_du_div_cs > mw.max_abs_du_div_cs_for_convection {
                    no_mix = true;
                }
            }
            if no_mix {
                set_no_mixing_face(
                    model,
                    k,
                    grada_face_ad,
                    gradr_ad,
                    scale_height_ad,
                    mixing_length_alpha,
                    MixingType::NoMixing,
                );
                return Ok(false);
            }
        }
    }

    let mut make_gradr_sticky = mw.make_gradr_sticky_in_solver_iters;
    if !make_gradr_sticky
        && mw.min_logt_for_make_gradr_sticky_in_solver_iters < 1.0e20
    {
        let max_lnt_start = if model.ln_t_start.len() == nz {
            model
                .ln_t_start
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        make_gradr_sticky = max_lnt_start / LN10
            >= mw.min_logt_for_make_gradr_sticky_in_solver_iters;
    }
    if make_gradr_sticky
        && model.aux.doing_struct_burn_mix
        && model.aux.fixed_gradr_sticky.get(k).copied().unwrap_or(false)
    {
        set_no_mixing_face(
            model,
            k,
            grada_face_ad,
            gradr_ad,
            scale_height_ad,
            mixing_length_alpha,
            MixingType::NoMixing,
        );
        return Ok(make_gradr_sticky);
    }

    // `Get_results` (`star/private/turb_support.f90:198-510`).
    let mlt_opt = if !model.aux.doing_relax
        && model.controls.steps_before_use_tdc > 0
    {
        if (model.model_number as isize)
            >= model.controls.steps_before_use_tdc
        {
            MltOption::Tdc
        } else {
            MltOption::Cox
        }
    } else {
        model.controls.mlt_option
    };

    let using_tdc = mlt_opt == MltOption::Tdc
        && model.aux.have_mlt_vc
        && k > 0
        && model.dt > 0.0
        && !(model.mstar_dot.abs() > 1.0e-99 && k < model.aux.k_const_mass);

    let ptot_f = p_f;
    let pr_f = (CRAD / 3.0) * t_f.pow4();
    let pg_f = ptot_f - pr_f;
    let beta_ad = pg_f / ptot_f;
    let lambda_f = scale_height_ad * mixing_length_alpha;
    let gradl_f = if model.controls.use_ledoux_criterion {
        grada_face_ad + gradl_composition_term
    } else {
        grada_face_ad
    };

    let max_conv_vel = if k > 0
        && model.controls.max_conv_vel_div_csound >= 0.0
        && model.q[k] <= model.controls.output.max_conv_vel_div_csound_maxq
    {
        (model.interp_val_to_pt(&model.csound, k)
            * model.controls.max_conv_vel_div_csound)
            .max(0.0)
    } else {
        1.0e99
    };

    let mut mix_type = MixingType::NoMixing;
    let mut grad_t_f = gradr_ad;
    let mut y_face_f = grad_t_f - gradl_f;
    let mut conv_vel_f = StarAdOrder1::ZERO;
    let mut d_f = StarAdOrder1::ZERO;
    let mut gamma_f = StarAdOrder1::ZERO;
    model.aux.mixing.superad_reduction_factor[k] = 1.0;

    let bad_params = beta_ad.val < 1.0e-10
        || mixing_length_alpha <= 0.0
        || kap_f.val < 1.0e-10
        || p_f.val < 1.0e-20
        || t_f.val < 1.0e-10
        || rho_f.val < 1.0e-20
        || m_grav_k < 1.0e-10
        || r_k.val < 1.0e-10
        || cgrav_k < 1.0e-10;

    if !bad_params {
        if using_tdc {
            let vc_old = if model.aux.okay_to_set_mlt_vc {
                model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0)
            } else {
                model.mlt_vc.get(k).copied().unwrap_or(0.0)
            };
            let e_f =
                model.interp_eos_ad_to_face(&model.ln_e, I_LN_E, k).exp();
            let dv_f = if model.lnd_old.len() == nz {
                let rho_old_00 = flexmesa_math::exp(model.lnd_old[k]);
                let rho_old_m1 = flexmesa_math::exp(model.lnd_old[k - 1]);
                let rho_old_f =
                    (rho_old_00 * alfa + rho_old_m1 * beta_w).max(1.0e-30);
                (StarAdOrder1::constant(1.0) / rho_f)
                    - StarAdOrder1::constant(1.0 / rho_old_f)
            } else {
                StarAdOrder1::ZERO
            };
            let l_ref = if model.aux.doing_struct_burn_mix
                && model.lum_start.len() == nz
            {
                &model.lum_start
            } else {
                &model.lum
            };
            let l_max =
                l_ref.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.0);
            let scale = l_ref[k].abs().max(1.0e-3 * l_max).max(1.0e-30);
            let tdc_params = TdcParams {
                conv_vel_start: vc_old,
                mixing_length_alpha,
                tdc_alpha_d: model.controls.tdc_alpha_d,
                tdc_alpha_r: model.controls.tdc_alpha_r,
                tdc_alpha_pt: model.controls.tdc_alpha_pt,
                tdc_alpha_c: model.controls.tdc_alpha_c,
                tdc_alpha_s: model.controls.tdc_alpha_s,
                dt: model.dt,
                cgrav: cgrav_k,
                m: m_grav_k.max(1.0),
                scale,
                max_conv_vel,
                include_mlt_corr_to_tdc: model.controls.include_mlt_corr_to_tdc,
                use_tdc_enthalpy_flux_limiter: model
                    .controls
                    .rsp2
                    .use_tdc_enthalpy_flux_limiter,
                y_face_guess: (gradr_ad.val - gradl_f.val).max(0.0),
            };
            let mut tdc_in = TdcInputs {
                chi_t: chi_t_f,
                chi_rho: chi_rho_f,
                gradr: gradr_ad,
                r: r_k,
                p: ptot_f,
                t: t_f,
                rho: rho_f,
                dv: dv_f,
                cp: cp_f,
                opacity: kap_f,
                scale_height: scale_height_ad,
                gradl: gradl_f,
                grada: grada_face_ad,
                eq_div_w: StarAdOrder1::ZERO,
                grav: grav_f,
                energy: e_f,
            };
            let tdc_out = set_tdc(&tdc_params, &tdc_in)
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "TDC evaluation failed at face {k} (`star/private/turb_support.f90:337-340`): {e}"
                    ))
                })?
                .output;
            mix_type = tdc_out.mixing_type;
            grad_t_f = tdc_out.grad_t;
            y_face_f = tdc_out.y_face;
            conv_vel_f = tdc_out.conv_vel;
            d_f = tdc_out.d;
            gamma_f = tdc_out.gamma;

            if model.controls.mix_wind.use_superad_reduction {
                let (gamma_factor, gradr_scaled) = eval_superad_reduction(
                    &model.controls.mix_wind,
                    t_f,
                    p_f,
                    beta_ad,
                    grad_t_f,
                    gradr_ad,
                    gradl_f,
                );
                model.aux.mixing.superad_reduction_factor[k] = gamma_factor.val;
                if gamma_factor.val > 1.0 {
                    tdc_in.gradr = gradr_scaled;
                    let tdc_out2 = set_tdc(&tdc_params, &tdc_in)
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "TDC superad_reduction failed at face {k}: {e}"
                            ))
                        })?
                        .output;
                    mix_type = tdc_out2.mixing_type;
                    grad_t_f = tdc_out2.grad_t;
                    y_face_f = tdc_out2.y_face;
                    conv_vel_f = tdc_out2.conv_vel;
                    d_f = tdc_out2.d;
                    gamma_f = tdc_out2.gamma;
                }
            }
        } else if gradr_ad.val > gradl_f.val {
            let mlt_params = MltParams {
                option: mlt_opt,
                mixing_length_alpha,
                henyey_nu: model.controls.henyey_mlt_nu_param,
                henyey_y: model.controls.henyey_mlt_y_param,
                max_conv_vel,
            };
            let mut mlt_in = MltInputs {
                chi_t: chi_t_f,
                chi_rho: chi_rho_f,
                cp: cp_f,
                grav: grav_f,
                lambda: lambda_f,
                rho: rho_f,
                p: ptot_f,
                t: t_f,
                opacity: kap_f,
                gradr: gradr_ad,
                grada: grada_face_ad,
                gradl: gradl_f,
            };
            let out = set_mlt(&mlt_params, &mlt_in).map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "set_mlt failed at face {k}: {e}"
                ))
            })?;
            mix_type = out.mixing_type;
            grad_t_f = out.grad_t;
            y_face_f = out.y_face;
            conv_vel_f = out.conv_vel;
            d_f = out.d;
            gamma_f = out.gamma;

            if model.controls.mix_wind.use_superad_reduction {
                let (gamma_factor, gradr_scaled) = eval_superad_reduction(
                    &model.controls.mix_wind,
                    t_f,
                    p_f,
                    beta_ad,
                    grad_t_f,
                    gradr_ad,
                    gradl_f,
                );
                model.aux.mixing.superad_reduction_factor[k] = gamma_factor.val;
                if gamma_factor.val > 1.0 {
                    mlt_in.gradr = gradr_scaled;
                    let out2 = set_mlt(&mlt_params, &mlt_in).map_err(|e| {
                        StarError::MicrophysicsError(format!(
                            "set_mlt superad_reduction failed at face {k}: {e}"
                        ))
                    })?;
                    mix_type = out2.mixing_type;
                    grad_t_f = out2.grad_t;
                    y_face_f = out2.y_face;
                    conv_vel_f = out2.conv_vel;
                    d_f = out2.d;
                    gamma_f = out2.gamma;
                }
            }
        }

        // Semiconvection and thermohaline (`turb_support.f90:398-419`).
        if mix_type == MixingType::NoMixing {
            if gradl_composition_term < 0.0 {
                if model.controls.thermohaline_coeff > 0.0 {
                    let iso_cid = model
                        .aux
                        .mixing
                        .dominant_iso_for_thermohaline
                        .get(k)
                        .copied()
                        .unwrap_or(0);
                    let th_params = ThermohalineParams {
                        option: model.controls.thermohaline_option,
                        gradl_composition_term,
                        xh1: model.xh_mass[k],
                        thermohaline_coeff: model.controls.thermohaline_coeff,
                        species: driving_species_for_cid(iso_cid),
                    };
                    let th_in = ThermohalineInputs {
                        lambda: lambda_f,
                        grada: grada_face_ad,
                        gradr: gradr_ad,
                        t: t_f,
                        opacity: kap_f,
                        rho: rho_f,
                        cp: cp_f,
                    };
                    let th_out =
                        set_thermohaline(&th_params, &th_in).map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "thermohaline mixing failed at face {k} (`star/private/turb_support.f90:404-407`): {e}"
                            ))
                        })?;
                    if th_out.mixing_type != MixingType::NoMixing {
                        grad_t_f = th_out.grad_t;
                        y_face_f = th_out.y_face;
                        conv_vel_f = th_out.conv_vel;
                        d_f = th_out.d;
                        gamma_f = th_out.gamma;
                        mix_type = th_out.mixing_type;
                    }
                }
            } else if gradr_ad.val > grada_face_ad.val
                && model.controls.alpha_semiconvection > 0.0
                && gradl_f.val > gradr_ad.val
            {
                let sc_params = SemiconvectionParams {
                    option: model.controls.semiconvection_option,
                    alpha_semiconvection: model.controls.alpha_semiconvection,
                    cgrav: cgrav_k,
                    gradl_composition_term,
                    m: m_grav_k.max(1.0),
                };
                let sc_in = SemiconvectionInputs {
                    l: lum_k,
                    lambda: lambda_f,
                    t: t_f,
                    p: ptot_f,
                    pr: pr_f,
                    beta: beta_ad,
                    opacity: kap_f,
                    rho: rho_f,
                    cp: cp_f,
                    gradr: gradr_ad,
                    grada: grada_face_ad,
                    gradl: gradl_f,
                };
                let sc_out =
                    set_semiconvection(&sc_params, &sc_in).map_err(|e| {
                        StarError::MicrophysicsError(format!(
                            "semiconvection failed at face {k} (`star/private/turb_support.f90:414-417`): {e}"
                        ))
                    })?;
                if sc_out.mixing_type != MixingType::NoMixing {
                    grad_t_f = sc_out.grad_t;
                    y_face_f = sc_out.y_face;
                    conv_vel_f = sc_out.conv_vel;
                    d_f = sc_out.d;
                    gamma_f = sc_out.gamma;
                    mix_type = sc_out.mixing_type;
                }
            }
        }

        // `remove_small_D_limit` (`star/private/turb_support.f90:422-430`).
        if d_f.val < model.controls.mix_wind.remove_small_d_limit
            || !d_f.val.is_finite()
        {
            mix_type = MixingType::NoMixing;
            grad_t_f = gradr_ad;
            y_face_f = grad_t_f - gradl_f;
            conv_vel_f = StarAdOrder1::ZERO;
            d_f = StarAdOrder1::ZERO;
            gamma_f = StarAdOrder1::ZERO;
        }

        // Forced non-turbulent boundary cells (`turb_support.f90:434-443`).
        let n_in = model
            .controls
            .mix_wind
            .tdc_num_innermost_cells_forced_nonturbulent;
        let n_out = model
            .controls
            .mix_wind
            .tdc_num_outermost_cells_forced_nonturbulent;
        if (n_in > 0 && k + n_in >= nz) || (k + 1 < n_out) {
            mix_type = MixingType::NoMixing;
            grad_t_f = gradr_ad;
            y_face_f = grad_t_f - gradl_f;
            conv_vel_f = StarAdOrder1::ZERO;
            d_f = StarAdOrder1::ZERO;
            gamma_f = StarAdOrder1::ZERO;
        }
    }

    // `store_results` (`star/private/turb_info.f90:252-289`).
    model.aux.mixing.mlt_mixing_type[k] = mix_type;
    model.aux.mixing.grada_face_ad[k] = grada_face_ad;
    model.aux.mixing.grada_face[k] = grada_face_ad.val;
    model.gradt_ad[k] = grad_t_f;
    model.gradt[k] = grad_t_f.val;
    model.aux.mixing.mlt_gradt[k] = model.gradt[k];
    model.aux.mixing.y_face[k] = y_face_f.val;
    model.mlt_vc_ad[k] = conv_vel_f;
    if model.aux.okay_to_set_mlt_vc {
        model.mlt_vc[k] = conv_vel_f.val;
    }
    model.mlt_d_ad[k] = d_f;
    model.mlt_d[k] = d_f.val;
    let area_rho = FOUR_PI * flexmesa_math::pow2(model.r[k]) * rho_f.val;
    let cdc_val = model.mlt_d[k] * flexmesa_math::pow2(area_rho);
    let area_rho_ad = FOUR_PI * r_k * r_k * rho_f;
    model.mlt_cdc[k] = cdc_val;
    model.mlt_cdc_ad[k] = d_f * area_rho_ad * area_rho_ad;
    model.mlt_gamma[k] = gamma_f.val;
    model.gradr_ad[k] = gradr_ad;
    model.gradr[k] = gradr_ad.val;
    model.gradl_ad[k] = grada_face_ad + gradl_composition_term;
    model.gradl[k] = model.gradl_ad[k].val;
    model.scale_height_ad[k] = scale_height_ad;
    model.scale_height[k] = scale_height_ad.val;
    model.aux.mixing.mlt_mixing_length[k] = lambda_f.val;

    // `adjust_gradT_fraction` + `adjust_gradT_excess` (`turb_info.f90:237-248,337-386`).
    let mlt_gf = model.controls.mix_wind.mlt_gradt_fraction;
    let f_blend = if (0.0..=1.0).contains(&mlt_gf) {
        mlt_gf
    } else {
        model.aux.mixing.adjust_mlt_gradt_fraction[k]
    };
    adjust_gradt_fraction(model, k, f_blend);

    if model.aux.mixing.mlt_mixing_type[k] == MixingType::NoMixing
        || model.gradr[k].abs() < 1.0e-20
    {
        model.aux.mixing.l_conv[k] = 0.0;
    } else {
        model.aux.mixing.l_conv[k] =
            model.lum[k] * (1.0 - model.gradt[k] / model.gradr[k]);
    }

    Ok(make_gradr_sticky)
}

fn set_no_mixing_face(
    model: &mut StarModel,
    k: usize,
    grada_face_ad: StarAdOrder1,
    gradr_ad: StarAdOrder1,
    scale_height_ad: StarAdOrder1,
    mixing_length_alpha: f64,
    mix_type: MixingType,
) {
    model.aux.mixing.mlt_mixing_type[k] = mix_type;
    model.aux.mixing.grada_face_ad[k] = grada_face_ad;
    model.aux.mixing.grada_face[k] = grada_face_ad.val;
    model.gradt_ad[k] = gradr_ad;
    model.gradt[k] = gradr_ad.val;
    model.aux.mixing.mlt_gradt[k] = gradr_ad.val;
    let y_face_ad = gradr_ad - grada_face_ad;
    model.aux.mixing.y_face[k] = y_face_ad.val;
    model.aux.mixing.gradt_sub_grada[k] = gradr_ad.val - grada_face_ad.val;
    model.mlt_vc_ad[k] = StarAdOrder1::ZERO;
    if model.aux.okay_to_set_mlt_vc {
        model.mlt_vc[k] = 0.0;
    }
    model.mlt_d_ad[k] = StarAdOrder1::ZERO;
    model.mlt_d[k] = 0.0;
    model.mlt_cdc[k] = 0.0;
    model.mlt_cdc_ad[k] = StarAdOrder1::ZERO;
    model.mlt_gamma[k] = 0.0;
    model.gradr_ad[k] = gradr_ad;
    model.gradr[k] = gradr_ad.val;
    model.gradl_ad[k] = StarAdOrder1::ZERO;
    model.gradl[k] = 0.0;
    model.scale_height_ad[k] = scale_height_ad;
    model.scale_height[k] = scale_height_ad.val;
    model.aux.mixing.mlt_mixing_length[k] =
        (scale_height_ad * mixing_length_alpha).val;
    model.aux.mixing.l_conv[k] = 0.0;
    if matches!(
        mix_type,
        MixingType::Crystallized | MixingType::PhaseSeparationMixing
    ) {
        model.sig[k] = 0.0;
    }
}

/// Evaluates implicit superadiabaticity reduction (`set_superad_reduction` in
/// `star/private/turb_support.f90:448-509`).
fn eval_superad_reduction(
    mw: &crate::controls_mix_wind::MixWindControls,
    t_f: StarAdOrder1,
    p_f: StarAdOrder1,
    beta_ad: StarAdOrder1,
    grad_t_f: StarAdOrder1,
    gradr_ad: StarAdOrder1,
    gradl_f: StarAdOrder1,
) -> (StarAdOrder1, StarAdOrder1) {
    let gamma_limit = mw.superad_reduction_gamma_limit;
    let scale_value1 = mw.superad_reduction_gamma_limit_scale;
    // Upstream `turb_support.f90:486` reads `superad_reduction_gamma_inv_scale`
    // into `scale_value2` but uses `scale_value1` in the `beta_inv` formula
    // (tracked under #169 / #267).
    let diff_grads_limit = mw.superad_reduction_diff_grads_limit;
    let reduction_limit = mw.superad_reduction_limit;

    let lrad_div_ledd = (4.0 * CRAD / 3.0) * t_f.pow4() / p_f * grad_t_f;
    let gamma_inv_threshold = (StarAdOrder1::constant(1.0) - beta_ad) * 4.0
        / (StarAdOrder1::constant(4.0) - beta_ad * 3.0);

    let mut gamma_factor = StarAdOrder1::constant(1.0);
    if grad_t_f.val > gradl_f.val
        && (lrad_div_ledd.val > gamma_limit
            || lrad_div_ledd.val > gamma_inv_threshold.val)
    {
        let alfa0 = (grad_t_f - gradl_f) / diff_grads_limit;
        let diff_grads_factor = if alfa0.val < 1.0 {
            let a3 = alfa0 * alfa0 * alfa0;
            (StarAdOrder1::constant(0.0) - a3)
                * (StarAdOrder1::constant(-10.0)
                    + alfa0 * (StarAdOrder1::constant(15.0) - alfa0 * 6.0))
        } else {
            StarAdOrder1::constant(1.0)
        };

        let mut gamma_term = StarAdOrder1::ZERO;
        if lrad_div_ledd.val > gamma_limit {
            let a = lrad_div_ledd / gamma_limit - 1.0;
            if a.val < 1.0 {
                gamma_term = gamma_term + (a * a * 0.5) * scale_value1;
            } else {
                gamma_term = gamma_term + (a - 0.5) * scale_value1;
            }
        }
        // Verbatim upstream `star/private/turb_support.f90:483-490`: uses
        // `scale_value1` (`superad_reduction_Gamma_limit_scale`) in the
        // `Gamma_inv_threshold` branch as well (tracked under #169 / #267).
        if lrad_div_ledd.val > gamma_inv_threshold.val {
            let a = lrad_div_ledd / gamma_inv_threshold - 1.0;
            if a.val < 1.0 {
                gamma_term = gamma_term + (a * a * 0.5) * scale_value1;
            } else {
                gamma_term = gamma_term + (a - 0.5) * scale_value1;
            }
        }
        if gamma_term.val > 0.0 {
            gamma_factor =
                gamma_term / beta_ad.sqrt() * diff_grads_factor + 1.0;
            if reduction_limit > 1.0 {
                let lambda_limit = 2.0 / (reduction_limit - 1.0);
                let exp_limit = ((gamma_factor - 1.0) * (-lambda_limit)).exp();
                gamma_factor = ((StarAdOrder1::constant(1.0)
                    / (exp_limit + 1.0))
                    - 0.5)
                    * (2.0 * (reduction_limit - 1.0))
                    + 1.0;
            }
        }
    }

    let gradr_scaled = if gamma_factor.val > 1.0 {
        let grad_scale =
            (gradr_ad - gradl_f) / (gamma_factor * gradr_ad) + gradl_f / gradr_ad;
        grad_scale * gradr_ad
    } else {
        gradr_ad
    };
    (gamma_factor, gradr_scaled)
}

fn adjust_gradt_fraction(model: &mut StarModel, k: usize, f: f64) {
    if (0.0..=1.0).contains(&f) {
        if f == 0.0 {
            model.gradt_ad[k] = model.gradr_ad[k];
        } else {
            model.gradt_ad[k] = model.aux.mixing.grada_face_ad[k] * f
                + model.gradr_ad[k] * (1.0 - f);
        }
        model.gradt[k] = model.gradt_ad[k].val;
    }
    adjust_gradt_excess(model, k);
    model.aux.mixing.gradt_sub_grada[k] =
        model.gradt[k] - model.aux.mixing.grada_face[k];
}

fn adjust_gradt_excess(model: &mut StarModel, k: usize) {
    let mw = &model.controls.mix_wind;
    let mut gradt_excess_alpha = model.aux.mixing.gradt_excess_alpha;
    model.aux.mixing.gradt_excess_effect[k] = 0.0;
    let gradt_sub_grada = model.gradt[k] - model.aux.mixing.grada_face[k];
    if gradt_excess_alpha <= 0.0 || gradt_sub_grada <= mw.gradt_excess_f1 {
        return;
    }
    if model.ln_t[k] / LN10 > mw.gradt_excess_max_logt {
        return;
    }
    let log_tau = flexmesa_math::log10(model.tau[k]);
    if log_tau < mw.gradt_excess_max_log_tau_full_off {
        return;
    }
    if log_tau < mw.gradt_excess_min_log_tau_full_on {
        gradt_excess_alpha *= (log_tau - mw.gradt_excess_max_log_tau_full_off)
            / (mw.gradt_excess_min_log_tau_full_on
                - mw.gradt_excess_max_log_tau_full_off);
    }
    let mut alfa = mw.gradt_excess_f2;
    if gradt_excess_alpha < 1.0 {
        alfa += (1.0 - alfa) * (1.0 - gradt_excess_alpha);
    }
    model.gradt_ad[k] = model.gradt_ad[k] * alfa
        + model.aux.mixing.grada_face_ad[k] * (1.0 - alfa);
    model.gradt[k] = model.gradt_ad[k].val;
    model.aux.mixing.gradt_excess_effect[k] = 1.0 - alfa;
}

fn get_lrad_div_ledd(model: &StarModel, k: usize) -> f64 {
    if model.cgrav[k] <= 0.0 || model.nz < 2 {
        return 0.0;
    }
    let j = if k == 0 { 1 } else { k };
    let del_m = model.dm_bar[j];
    let del_t4 =
        flexmesa_math::pow4(model.t[j - 1]) - flexmesa_math::pow4(model.t[j]);
    let area = FOUR_PI * model.r[j] * model.r[j];
    -(area * area * CRAD * (del_t4 / del_m) / 3.0)
        / (FOUR_PI * model.cgrav[j] * model.m_grav[j])
}

/// Evaluates the start-of-step MLT++ boost scalar `gradT_excess_alpha`
/// (`set_gradT_excess_alpha` in `star/private/turb_info.f90:411-519`).
pub fn set_gradt_excess_alpha(model: &mut StarModel) {
    let mw = &model.controls.mix_wind;
    if !mw.okay_to_reduce_gradt_excess || model.nz == 0 {
        model.aux.mixing.gradt_excess_alpha = 0.0;
        return;
    }
    let nz = model.nz;
    let h1_c = model.xh_mass[nz - 1];
    if h1_c > mw.gradt_excess_max_center_h1 {
        model.aux.mixing.gradt_excess_alpha = 0.0;
        return;
    }
    let he4_c = model.yhe_mass[nz - 1];
    if he4_c < mw.gradt_excess_min_center_he4 {
        model.aux.mixing.gradt_excess_alpha = 0.0;
        return;
    }

    let mut beta = 1.0_f64;
    let mut k_beta = 0_usize;
    for k in 0..nz {
        let tmp = flexmesa_math::exp(model.ln_pgas[k]) / model.peos[k];
        if tmp < beta {
            k_beta = k;
            beta = tmp;
        }
    }
    if model.species > 0 {
        beta *= 1.0 + model.xa[(nz - 1) * model.species];
    }
    model.aux.mixing.gradt_excess_min_beta = beta;

    let mut lambda = 0.0_f64;
    for k in 1..=k_beta {
        let tmp = get_lrad_div_ledd(model, k);
        if tmp > lambda {
            lambda = tmp;
        }
    }
    lambda = lambda.min(1.0);
    model.aux.mixing.gradt_excess_max_lambda = lambda;

    let lambda1 = mw.gradt_excess_lambda1;
    let beta1 = mw.gradt_excess_beta1;
    let lambda2 = mw.gradt_excess_lambda2;
    let beta2 = mw.gradt_excess_beta2;
    let dlambda = mw.gradt_excess_dlambda;
    let dbeta = mw.gradt_excess_dbeta;

    let alpha = if lambda1 < 0.0 {
        1.0
    } else if lambda >= lambda1 {
        if beta <= beta1 {
            1.0
        } else if beta < beta1 + dbeta {
            (beta1 + dbeta - beta) / dbeta
        } else {
            0.0
        }
    } else if lambda >= lambda2 {
        let beta_limit =
            beta2 + (lambda - lambda2) * (beta1 - beta2) / (lambda1 - lambda2);
        if beta <= beta_limit {
            1.0
        } else if beta < beta_limit + dbeta {
            (beta_limit + dbeta - beta) / dbeta
        } else {
            0.0
        }
    } else if lambda > lambda2 - dlambda {
        if beta <= beta2 {
            1.0
        } else if beta < beta2 + dbeta {
            (lambda - (lambda2 - dlambda)) / dlambda
        } else {
            0.0
        }
    } else {
        0.0
    };

    let alpha_old = model.aux.mixing.gradt_excess_alpha_old;
    let mut new_alpha = if model.model_number > 1 && lambda1 >= 0.0 {
        let mut sm = (1.0 - mw.gradt_excess_age_fraction) * alpha
            + mw.gradt_excess_age_fraction * alpha_old;
        if mw.gradt_excess_max_change > 0.0 {
            if sm > alpha_old {
                sm = sm.min(alpha_old + mw.gradt_excess_max_change);
            } else {
                sm = sm.max(alpha_old - mw.gradt_excess_max_change);
            }
        }
        sm
    } else {
        alpha
    };
    if new_alpha < 1.0e-4 {
        new_alpha = 0.0;
    }
    if new_alpha > 0.9999 {
        new_alpha = 1.0;
    }
    model.aux.mixing.gradt_excess_alpha = new_alpha;
}

/// Checks for geometrically thin convection zones where
/// $\Delta r_{\mathrm{cz}} < \alpha_{\mathrm{MLT}} \min(H_{P,\mathrm{top}}, H_{P,\mathrm{bot}})$
/// and re-evaluates MLT using $\alpha_{\mathrm{MLT}} = \Delta r_{\mathrm{cz}} / H_{P,k}$
/// (`check_for_redo_MLT` in `star/private/turb_info.f90:522-605`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `do1_mlt_2` fails on re-evaluation.
pub fn check_for_redo_mlt(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if nz < 2 || !model.controls.redo_conv_for_dr_lt_mixing_length {
        return Ok(());
    }
    if model.aux.mixing.mlt_mixing_type.len() != nz {
        return Ok(());
    }
    let mut in_conv =
        model.aux.mixing.mlt_mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    let mut k_bot = nz - 1;
    let mut bot_r = model.r[k_bot];
    let mut bot_hp = model.scale_height[k_bot];

    for k in (1..(nz - 1)).rev() {
        if in_conv {
            if model.aux.mixing.mlt_mixing_type[k]
                != MixingType::ConvectiveMixing
            {
                redo_convective_region(model, k, k_bot, bot_r, bot_hp)?;
                in_conv = false;
            }
        } else if model.aux.mixing.mlt_mixing_type[k]
            == MixingType::ConvectiveMixing
        {
            k_bot = k + 1;
            in_conv = true;
            bot_r = model.r[k_bot];
            bot_hp = model.scale_height[k_bot];
        }
    }
    if in_conv {
        redo_convective_region(model, 0, k_bot, bot_r, bot_hp)?;
    }
    Ok(())
}

fn redo_convective_region(
    model: &mut StarModel,
    k_top: usize,
    k_bot: usize,
    bot_r: f64,
    bot_hp: f64,
) -> Result<(), StarError> {
    let top_r = model.r[k_top];
    let top_hp = model.scale_height[k_top];
    let dr = top_r - bot_r;
    let alpha_k = if model.aux.mixing.alpha_mlt[k_top] > 0.0 {
        model.aux.mixing.alpha_mlt[k_top]
    } else {
        model.controls.mixing_length_alpha
    };
    if dr < alpha_k * top_hp.min(bot_hp) {
        for kk in k_top..=k_bot {
            if dr < model.aux.mixing.mlt_mixing_length[kk] {
                let new_alpha = dr / model.scale_height[kk];
                let _ = do1_mlt_2(model, kk, Some(new_alpha), None)?;
            }
        }
    }
    Ok(())
}

/// Evaluates all zone and face microphysics and hydrodynamic variables
/// (`set_hydro_vars` in `star/private/hydro_vars.f90:500-635`).
///
/// # Errors
/// Returns [`StarError`] if any microphysics, MLT, mixing, or boundary
/// evaluation fails.
pub fn set_hydro_vars(
    model: &mut StarModel,
    skip_net: bool,
    skip_net_jacobian: bool,
) -> Result<(), StarError> {
    MicrophysicsCache::eval_or_restore(model, skip_net, skip_net_jacobian)?;
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.aux.mixing.mlt_mixing_type.len() != nz {
        model.aux.mixing.resize(nz);
    }
    let sig_was_zero = model.sig.iter().all(|&s| s == 0.0);

    if !model.aux.doing_struct_burn_mix {
        do_brunt_b(model)?;
        crate::mixing_state::set_grads(model);
    }

    if !model.controls.rsp_flag {
        if !model.aux.doing_struct_burn_mix
            && model.aux.mixing.okay_to_set_mixing_info
        {
            if model.controls.mix_wind.make_gradr_sticky_in_solver_iters {
                model.aux.fixed_gradr_sticky.fill(false);
            }
            model.aux.mixing.alpha_mlt[..nz]
                .fill(model.controls.mixing_length_alpha);
        }
        if model.controls.mix_wind.use_t_tau_gradr_factor {
            let t_tau_id = model.controls.atm_t_tau_relation.id();
            for k in 0..nz {
                let dq = flexmesa_atm::eval_t_tau_dq_dtau(
                    t_tau_id,
                    model.tau[k],
                )
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "eval_t_tau_dq_dtau failed at face {k}: {e}"
                    ))
                })?;
                model.aux.mixing.gradr_factor[k] = 1.0 + dq;
            }
        } else {
            model.aux.mixing.gradr_factor[..nz].fill(1.0);
        }

        set_mlt_vars(model)?;
        check_for_redo_mlt(model)?;
    }

    if model.controls.rsp2_flag {
        eval_surf_bc(model)?;
        crate::eps_grav::eval_eps_grav(model);
        crate::rsp2_ad::update_rsp2_vars_ad(model);
        let alpha = model.controls.mixing_length_alpha;
        for k in 0..nz {
            if k == 0 || crate::rsp2_eqns::is_nonturbulent_cell(model, k) {
                model.conv_vel[k] = 0.0;
                model.d_mix[k] = 0.0;
                model.mixing_type[k] = MixingType::NoMixing;
                model.aux.mixing.mlt_mixing_type[k] = MixingType::NoMixing;
                model.gradt[k] = model.gradr[k];
            } else {
                let (alfa, beta) =
                    crate::rsp2_eqns::rsp2_face_weights(model, k);
                let w_00 = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
                let w_m1 = model.aux.rsp.et_rsp[k - 1].max(0.0).sqrt();
                let vc =
                    flexmesa_const::SQRT_2_DIV_3 * (alfa * w_00 + beta * w_m1);
                let d_mix = vc * alpha * model.aux.rsp.hp_face[k] / 3.0;
                model.conv_vel[k] = vc;
                model.d_mix[k] = d_mix;
                let mt = if d_mix > 0.0 {
                    MixingType::ConvectiveMixing
                } else {
                    MixingType::NoMixing
                };
                model.mixing_type[k] = mt;
                model.aux.mixing.mlt_mixing_type[k] = mt;
                let grada_f = alfa * model.grada[k] + beta * model.grada[k - 1];
                model.gradt[k] = model.aux.rsp.y_face_rsp[k] + grada_f;
                model.aux.mixing.gradt_sub_grada[k] =
                    model.gradt[k] - grada_f;
            }
        }
        if !model.aux.doing_struct_burn_mix && sig_was_zero {
            model.sig.fill(0.0);
            update_sig_from_d_mix(model)?;
        }
        do_brunt_n2(model);
        if model.controls.u_flag {
            crate::riemann_ad::do_uface_and_pface(model)?;
        }
        return Ok(());
    }

    if !model.aux.doing_struct_burn_mix {
        do_brunt_n2(model);
        if model.aux.mixing.okay_to_set_mixing_info {
            let skip_set_cz_bdy_mass = !model.aux.mixing.have_new_generation;
            crate::mix_info::set_mixing_info_with_skip(
                model,
                skip_set_cz_bdy_mass,
            )?;
            if sig_was_zero {
                update_sig_from_d_mix(model)?;
            }
        }
    }
    eval_surf_bc(model)?;
    crate::eps_grav::eval_eps_grav(model);
    if model.controls.u_flag {
        crate::riemann_ad::do_uface_and_pface(model)?;
    }
    Ok(())
}
