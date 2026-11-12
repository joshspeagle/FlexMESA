// SPDX-License-Identifier: LGPL-3.0-or-later
//! Face-centered temperature gradients, Mixing-Length Theory, Time-Dependent
//! Convection, Semiconvection, and Thermohaline Mixing (`set_mlt_vars` in
//! `star/private/mlt_info.f90`, `turb_support.f90`, `turb_info.f90`), and
//! top-level microphysics/hydro variable evaluation (`set_hydro_vars` in
//! `star/private/hydro_vars.f90`).

use flexmesa_const::{MixingType, CLIGHT, CRAD, FOUR_PI};
use flexmesa_eos::{I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD, I_LN_E, I_LN_PGAS};
use flexmesa_turb::{
    set_mlt, set_semiconvection, set_tdc, set_thermohaline, DrivingSpecies,
    MltInputs, MltOption, MltParams, SemiconvectionInputs,
    SemiconvectionParams, TdcInputs, TdcParams, ThermohalineInputs,
    ThermohalineParams,
};

use crate::controls::StarError;
pub use crate::hydro_brunt_surf::{do_brunt_b, do_brunt_n2, eval_surf_bc};
use crate::micro::{
    do_eos, do_kap, do_net, do_neu, resolve_model_net, set_basic_vars,
};
use crate::model::StarModel;
use crate::rotation::update_rotation_distortion_factors;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LNR_00, I_LNT_00, I_L_00,
};

/// Evaluates face-centered radiative, adiabatic, Ledoux, and actual temperature
/// gradients ($\nabla_{\mathrm{rad}}, \nabla_{\mathrm{ad}}, \nabla_L, \nabla_T$),
/// pressure scale height $H_P$, convective velocity $v_{\mathrm{conv}}$, and
/// diffusivity $D_{\mathrm{MLT}}$ with 3-cell stencil [`StarAdOrder1`] partials
/// across all faces `k = 0..nz-1` (`set_mlt_vars` in `star/private/mlt_info.f90`,
/// `turb_support.f90`, and `turb_info.f90`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `set_mlt` fails at any face.
pub fn set_mlt_vars(model: &mut StarModel) -> Result<(), StarError> {
    if model.controls.rotation_flag {
        update_rotation_distortion_factors(model);
    }
    let nz = model.nz;
    for k in 0..nz {
        eval_face_mlt(model, k, None)?;
    }

    if model.controls.redo_conv_for_dr_lt_mixing_length {
        check_for_redo_mlt(model)?;
    }
    if !model.aux.doing_struct_burn_mix && model.sig.iter().all(|&s| s == 0.0) {
        update_sig_from_d_mix(model);
    }
    Ok(())
}

/// Computes the Lagrangian mass-diffusion coefficient
/// $\sigma_k = (4\pi r_k^2 \rho_{\mathrm{face},k})^2 D_{\mathrm{mix},k} / \overline{\Delta m}_k$
/// across interior faces `k = 1..nz-1` (`sig[0] = 0`), applying the diffusive
/// stiffness cap $\sigma_k \Delta t / \min(\Delta m_{k-1}, \Delta m_k) \le 10^6$.
pub fn update_sig_from_d_mix(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    model.sig[0] = 0.0;
    for k in 1..nz {
        let rho_f = model.interp_val_to_pt(&model.rho, k);
        let area_rho = FOUR_PI * model.r[k] * model.r[k] * rho_f;
        let dm_bar_k = model.dm_bar[k].max(1.0e-30);
        let mut sig_k = (area_rho * area_rho) * model.d_mix[k] / dm_bar_k;
        if model.dt > 0.0 {
            let dm_min = model.dm[k - 1].min(model.dm[k]).max(1.0e-30);
            sig_k = sig_k.min(1.0e6 * dm_min / model.dt);
        }
        model.sig[k] = sig_k;
    }
}

/// Evaluates MLT/TDC/semiconvective/thermohaline quantities at a single face `k`,
/// optionally overriding the mixing length $\Lambda$ with `custom_length`
/// (`get_mlt_vars` in `turb_support.f90:328-444` and `turb_info.f90:132-205`).
fn eval_face_mlt(
    model: &mut StarModel,
    k: usize,
    custom_length: Option<f64>,
) -> Result<(), StarError> {
    let (alfa, beta) = model.get_face_weights(k);

    let lnd_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00);
    let lnt_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00);
    let rho_00 = lnd_00.exp();
    let t_00 = lnt_00.exp();
    let pgas_00 = StarAdOrder1::from_dlnd_dlnt_00(
        model.ln_pgas[k],
        model.eos_d_dlnd[k][I_LN_PGAS],
        model.eos_d_dlnt[k][I_LN_PGAS],
    )
    .exp();
    let prad_00 = (CRAD / 3.0) * t_00.pow4();
    let p_00 = pgas_00 + prad_00;

    let (rho_f, t_f, prad_f, p_f) = if k == 0 {
        (rho_00, t_00, prad_00, p_00)
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
            rho_00 * alfa + rho_m1 * beta,
            t_00 * alfa + t_m1 * beta,
            prad_00 * alfa + prad_m1 * beta,
            p_00 * alfa + p_m1 * beta,
        )
    };

    let kap_f = model.interp_cell_ad_to_face(
        &model.opacity,
        &model.d_opacity_dlnd,
        &model.d_opacity_dlnt,
        k,
    );
    let grada_f = model.interp_eos_ad_to_face(&model.grada, I_GRAD_AD, k);
    let cp_f = model.interp_eos_ad_to_face(&model.cp, I_CP, k);
    let chi_rho_f = model.interp_eos_ad_to_face(&model.chi_rho, I_CHI_RHO, k);
    let chi_t_f = model.interp_eos_ad_to_face(&model.chi_t, I_CHI_T, k);

    let (fp_k, ft_k) = if model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction
    {
        let fp = model.aux.fp_rot.get(k).copied().unwrap_or(1.0);
        let ft = model.aux.ft_rot.get(k).copied().unwrap_or(1.0);
        (
            if fp > 0.0 { fp } else { 1.0 },
            if ft > 0.0 { ft } else { 1.0 },
        )
    } else {
        (1.0, 1.0)
    };

    let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_k = lnr_k.exp();
    let lum_k = StarAdOrder1::variable(model.lum[k], I_L_00);
    let gm_k = model.cgrav[k] * model.m_grav[k];
    let grav_f = (gm_k / (r_k * r_k)) * fp_k;
    let hp_std = p_f / (grav_f * rho_f);
    let hp_alt = (p_f / (model.cgrav[k] * rho_f * rho_f)).sqrt();
    let hp_f = if !hp_std.val.is_finite() || hp_std.val > hp_alt.val {
        hp_alt
    } else {
        hp_std
    };

    let mut gradr_f = (p_f * kap_f * lum_k
        / (4.0 * FOUR_PI * CLIGHT * gm_k * prad_f))
        * (ft_k / fp_k);
    if k == 0 && model.tau[0] < 2.0 / 3.0 && model.tau[0] > 0.0 {
        gradr_f = gradr_f * (1.0 + 2.0 / (3.0 * model.tau[0]));
    }

    let b_term = if model.controls.use_ledoux_criterion {
        model.brunt_b[k]
    } else {
        0.0
    };
    let gradl_f = grada_f + b_term;

    // In `check_for_redo_mlt` (`mlt_info.f90:312`), `alfa = dr / scale_height(k)`
    // and `mixing_length = alfa * scale_height(k)`.
    let alpha_mlt = if let Some(dr) = custom_length {
        (dr / hp_f.val.max(1.0)).max(1.0e-6)
    } else {
        model.controls.mixing_length_alpha
    };
    let lambda_f = hp_f * alpha_mlt;

    let (mut grad_t_f, mut conv_vel_f, mut d_f, mut gamma_val, mut mix_type) =
        if alpha_mlt <= 0.0 {
            (
                gradr_f,
                StarAdOrder1::ZERO,
                StarAdOrder1::ZERO,
                0.0,
                MixingType::NoMixing,
            )
        } else {
            let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
            let tdc_res = if model.controls.mlt_option == MltOption::Tdc
                && model.dt > 0.0
                && k > 0
                && vc_old > 0.0
            {
                let e_f =
                    model.interp_eos_ad_to_face(&model.ln_e, I_LN_E, k).exp();
                let dv_f = if model.lnd_old.len() == model.nz {
                    let rho_old_00 = model.lnd_old[k].exp();
                    let rho_old_m1 = model.lnd_old[k - 1].exp();
                    let rho_old_f =
                        (rho_old_00 * alfa + rho_old_m1 * beta).max(1.0e-30);
                    (StarAdOrder1::constant(1.0) / rho_f)
                        - StarAdOrder1::constant(1.0 / rho_old_f)
                } else {
                    StarAdOrder1::ZERO
                };
                let tdc_params = TdcParams {
                    conv_vel_start: vc_old,
                    mixing_length_alpha: alpha_mlt,
                    tdc_alpha_d: 1.0,
                    tdc_alpha_r: 1.0,
                    tdc_alpha_pt: 0.0,
                    tdc_alpha_c: 1.0,
                    tdc_alpha_s: 1.0,
                    dt: model.dt,
                    cgrav: model.cgrav[k],
                    m: model.m_grav[k].max(1.0),
                    scale: model
                        .lum
                        .first()
                        .copied()
                        .unwrap_or(1.0)
                        .abs()
                        .max(model.lum[k].abs())
                        .max(1.0e20),
                    max_conv_vel: 1.0e99,
                    include_mlt_corr_to_tdc: true,
                    use_tdc_enthalpy_flux_limiter: true,
                    y_face_guess: (gradr_f.val - gradl_f.val).max(0.0),
                };
                let tdc_in = TdcInputs {
                    chi_t: chi_t_f,
                    chi_rho: chi_rho_f,
                    gradr: gradr_f,
                    r: r_k,
                    p: p_f,
                    t: t_f,
                    rho: rho_f,
                    dv: dv_f,
                    cp: cp_f,
                    opacity: kap_f,
                    scale_height: hp_f,
                    gradl: gradl_f,
                    grada: grada_f,
                    eq_div_w: StarAdOrder1::ZERO,
                    grav: grav_f,
                    energy: e_f,
                };
                set_tdc(&tdc_params, &tdc_in).ok().map(|r| r.output)
            } else {
                None
            };

            let out = if let Some(tdc_out) = tdc_res {
                tdc_out
            } else {
                let mlt_params = MltParams {
                    option: model.controls.mlt_option,
                    mixing_length_alpha: alpha_mlt,
                    henyey_nu: 8.0,
                    henyey_y: 1.0 / 3.0,
                    max_conv_vel: 1.0e99,
                };
                let mlt_in = MltInputs {
                    chi_t: chi_t_f,
                    chi_rho: chi_rho_f,
                    cp: cp_f,
                    grav: grav_f,
                    lambda: lambda_f,
                    rho: rho_f,
                    p: p_f,
                    t: t_f,
                    opacity: kap_f,
                    gradr: gradr_f,
                    grada: grada_f,
                    gradl: gradl_f,
                };
                set_mlt(&mlt_params, &mlt_in).map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "set_mlt failed at face {k}: {e}"
                    ))
                })?
            };
            (
                out.grad_t,
                out.conv_vel,
                out.d,
                out.gamma.val,
                out.mixing_type,
            )
        };

    // Semiconvective and thermohaline mixing (`turb_info.f90:132-205`)
    if mix_type == MixingType::NoMixing
        && model.controls.use_ledoux_criterion
        && k > 0
        && lambda_f.val > 0.0
    {
        if b_term < 0.0 && model.controls.thermohaline_coeff > 0.0 {
            let th_params = ThermohalineParams {
                option: model.controls.thermohaline_option,
                gradl_composition_term: b_term,
                xh1: model.xh_mass[k],
                thermohaline_coeff: model.controls.thermohaline_coeff,
                species: DrivingSpecies::he4(),
            };
            let th_in = ThermohalineInputs {
                lambda: lambda_f,
                grada: grada_f,
                gradr: gradr_f,
                t: t_f,
                opacity: kap_f,
                rho: rho_f,
                cp: cp_f,
            };
            if let Ok(th_out) = set_thermohaline(&th_params, &th_in) {
                if th_out.d.val > 0.0 {
                    grad_t_f = th_out.grad_t;
                    conv_vel_f = th_out.conv_vel;
                    d_f = th_out.d;
                    gamma_val = th_out.gamma.val;
                    mix_type = th_out.mixing_type;
                }
            }
        } else if b_term > 0.0
            && model.controls.alpha_semiconvection > 0.0
            && (gradr_f.val - grada_f.val) > 1.0e-40
        {
            let sc_params = SemiconvectionParams {
                option: model.controls.semiconvection_option,
                alpha_semiconvection: model.controls.alpha_semiconvection,
                cgrav: model.cgrav[k],
                gradl_composition_term: b_term,
                m: model.m_grav[k].max(1.0),
            };
            let sc_in = SemiconvectionInputs {
                l: lum_k,
                lambda: lambda_f,
                t: t_f,
                p: p_f,
                pr: prad_f,
                beta: (p_f - prad_f) / p_f,
                opacity: kap_f,
                rho: rho_f,
                cp: cp_f,
                gradr: gradr_f,
                grada: grada_f,
                gradl: gradl_f,
            };
            if let Ok(sc_out) = set_semiconvection(&sc_params, &sc_in) {
                if sc_out.mixing_type == MixingType::SemiconvectiveMixing {
                    grad_t_f = sc_out.grad_t;
                    conv_vel_f = sc_out.conv_vel;
                    d_f = sc_out.d;
                    gamma_val = sc_out.gamma.val;
                    mix_type = sc_out.mixing_type;
                }
            }
        }
    }

    let area_rho = FOUR_PI * r_k * r_k * rho_f;
    let cdc_f = area_rho * area_rho * conv_vel_f * lambda_f / 3.0;
    let mlt_vc_ad = if mix_type == MixingType::ConvectiveMixing {
        conv_vel_f
    } else {
        StarAdOrder1::ZERO
    };

    model.scale_height[k] = hp_f.val;
    model.scale_height_ad[k] = hp_f;
    model.gradr[k] = gradr_f.val;
    model.gradr_ad[k] = gradr_f;
    model.gradl[k] = gradl_f.val;
    model.gradl_ad[k] = gradl_f;
    model.gradt[k] = grad_t_f.val;
    model.gradt_ad[k] = grad_t_f;
    model.mlt_vc[k] = mlt_vc_ad.val;
    model.mlt_vc_ad[k] = mlt_vc_ad;
    model.mlt_d[k] = d_f.val;
    model.mlt_d_ad[k] = d_f;
    model.mlt_cdc[k] = cdc_f.val;
    model.mlt_cdc_ad[k] = cdc_f;
    model.mlt_gamma[k] = gamma_val;
    model.mixing_type[k] = mix_type;
    model.d_mix[k] = d_f.val;
    model.conv_vel[k] = conv_vel_f.val;

    Ok(())
}

/// Checks for geometrically thin convection zones where
/// $\Delta r_{\mathrm{cz}} < \alpha_{\mathrm{MLT}} H_P$ and re-evaluates MLT
/// using $\Lambda = \Delta r_{\mathrm{cz}}$ (`check_for_redo_mlt` in
/// `star/private/mlt_info.f90:233-348`).
pub fn check_for_redo_mlt(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if nz < 2 {
        return Ok(());
    }
    let alpha = model.controls.mixing_length_alpha;
    let mut k_top = 1_usize;
    while k_top < nz {
        if model.mixing_type[k_top] != MixingType::ConvectiveMixing {
            k_top += 1;
            continue;
        }
        let mut k_bot = k_top;
        while k_bot + 1 < nz
            && model.mixing_type[k_bot + 1] == MixingType::ConvectiveMixing
        {
            k_bot += 1;
        }
        let r_top = model.r[k_top];
        let r_bot = if k_bot + 1 < nz {
            model.r[k_bot + 1]
        } else {
            model.r_center
        };
        let dr = (r_top - r_bot).max(0.0);
        if dr > 0.0 {
            let mut min_lambda = f64::INFINITY;
            for k in k_top..=k_bot {
                min_lambda = min_lambda.min(alpha * model.scale_height[k]);
            }
            if dr < min_lambda {
                for k in k_top..=k_bot {
                    eval_face_mlt(model, k, Some(dr))?;
                }
            }
        }
        k_top = k_bot + 1;
    }
    Ok(())
}

/// Evaluates all zone and face microphysics and hydrodynamic variables
/// (`set_hydro_vars` in `star/private/hydro_vars.f90:101-326`).
///
/// # Errors
/// Returns [`StarError`] if any microphysics or boundary evaluation fails.
pub fn set_hydro_vars(
    model: &mut StarModel,
    skip_net: bool,
    skip_net_jacobian: bool,
) -> Result<(), StarError> {
    set_basic_vars(model)?;
    do_eos(model)?;
    do_kap(model)?;
    do_neu(model)?;
    if !skip_net {
        if let Some(net_info) = resolve_model_net(model) {
            do_net(model, &net_info, skip_net_jacobian)?;
        }
    }
    let sig_was_zero = model.sig.iter().all(|&s| s == 0.0);
    do_brunt_b(model);
    set_mlt_vars(model)?;
    if model.controls.rsp2_flag {
        eval_surf_bc(model)?;
        crate::eps_grav::eval_eps_grav(model);
        crate::rsp2_ad::update_rsp2_vars_ad(model);
        let alpha = model.controls.mixing_length_alpha;
        for k in 0..model.nz {
            if k == 0 || crate::rsp2_eqns::is_nonturbulent_cell(model, k) {
                model.conv_vel[k] = 0.0;
                model.d_mix[k] = 0.0;
                model.mixing_type[k] = MixingType::NoMixing;
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
                model.mixing_type[k] = if d_mix > 0.0 {
                    MixingType::ConvectiveMixing
                } else {
                    MixingType::NoMixing
                };
                let grada_f = alfa * model.grada[k] + beta * model.grada[k - 1];
                model.gradt[k] = model.aux.rsp.y_face_rsp[k] + grada_f;
            }
        }
        if !model.aux.doing_struct_burn_mix && sig_was_zero {
            model.sig.fill(0.0);
            update_sig_from_d_mix(model);
        }
        do_brunt_n2(model);
        if model.controls.u_flag {
            crate::riemann_ad::do_uface_and_pface(model)?;
        }
        return Ok(());
    }
    do_brunt_n2(model);
    eval_surf_bc(model)?;
    crate::eps_grav::eval_eps_grav(model);
    if model.controls.u_flag {
        crate::riemann_ad::do_uface_and_pface(model)?;
    }
    Ok(())
}
