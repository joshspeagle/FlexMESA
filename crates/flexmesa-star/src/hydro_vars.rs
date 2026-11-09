// SPDX-License-Identifier: LGPL-3.0-or-later
//! Face-centered temperature gradients, Mixing-Length Theory (`set_mlt_vars`),
//! Brunt-Väisälä frequency (`do_brunt_b`, `do_brunt_n2`), surface atmosphere
//! boundary conditions (`eval_surf_bc`), and top-level microphysics/hydro
//! variable evaluation (`set_hydro_vars` in `star/private/hydro_vars.f90`,
//! `mlt_info.f90`, and `brunt.f90`).

use flexmesa_atm::{
    atm_teff, eval_t_tau_uniform, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, ATM_T_TAU_EDDINGTON,
};
use flexmesa_const::{MixingType, CLIGHT, CRAD, FOUR_PI};
use flexmesa_eos::{I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD, I_LN_PGAS};
use flexmesa_turb::{set_mlt, MltInputs, MltParams};

use crate::controls::{AtmOption, StarError};
use crate::micro::{
    do_eos, do_kap, do_net, do_neu, resolve_model_net, set_basic_vars,
};
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LNR_00, I_LNT_00, I_L_00,
};

/// Evaluates the face-centered composition gradient term $B_k$ (`do_brunt_B`
/// in `star/private/brunt.f90:57-293`).
pub fn do_brunt_b(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    model.brunt_b.fill(0.0);
    if !model.controls.calculate_brunt_n2
        && !model.controls.use_ledoux_criterion
    {
        return;
    }

    for k in 1..nz {
        let dlnp = model.ln_peos[k - 1] - model.ln_peos[k];
        if dlnp.abs() < 1.0e-14 {
            model.brunt_b[k] = 0.0;
            continue;
        }
        let mu_m1 = model.mu[k - 1].max(1.0e-10);
        let mu_00 = model.mu[k].max(1.0e-10);
        let dln_mu = mu_m1.ln() - mu_00.ln();
        let chi_rho_f = model.interp_val_to_pt(&model.chi_rho, k);
        let chi_t_f = model.interp_val_to_pt(&model.chi_t, k).max(1.0e-10);
        // Ledoux composition gradient term B = (chi_rho / chi_T) * (d ln mu / d ln P)
        let b_k = (chi_rho_f / chi_t_f) * (dln_mu / dlnp);
        model.brunt_b[k] = if b_k.is_finite() {
            b_k.clamp(-10.0, 10.0)
        } else {
            0.0
        };
    }
}

/// Evaluates face-centered radiative, adiabatic, Ledoux, and actual temperature
/// gradients ($\nabla_{\mathrm{rad}}, \nabla_{\mathrm{ad}}, \nabla_L, \nabla_T$),
/// pressure scale height $H_P$, convective velocity $v_{\mathrm{conv}}$, and
/// diffusivity $D_{\mathrm{MLT}}$ with 3-cell stencil [`StarAdOrder1`] partials
/// across all faces `k = 0..nz-1` (`set_mlt_vars` in `star/private/mlt_info.f90`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `set_mlt` fails at any face.
pub fn set_mlt_vars(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    for k in 0..nz {
        eval_face_mlt(model, k, None)?;
    }

    if model.controls.redo_conv_for_dr_lt_mixing_length {
        check_for_redo_mlt(model)?;
    }
    if model.sig.iter().all(|&s| s == 0.0) {
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

/// Evaluates MLT quantities at a single face `k`, optionally overriding the
/// mixing length $\Lambda$ with `custom_length`.
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

    let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_k = lnr_k.exp();
    let lum_k = StarAdOrder1::variable(model.lum[k], I_L_00);
    let gm_k = model.cgrav[k] * model.m_grav[k];
    let grav_f = gm_k / (r_k * r_k);
    let hp_std = p_f / (grav_f * rho_f);
    let hp_alt = (p_f / (model.cgrav[k] * rho_f * rho_f)).sqrt();
    let hp_f = if !hp_std.val.is_finite() || hp_std.val > hp_alt.val {
        hp_alt
    } else {
        hp_std
    };

    let mut gradr_f =
        p_f * kap_f * lum_k / (4.0 * FOUR_PI * CLIGHT * gm_k * prad_f);
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
    let mlt_out = set_mlt(&mlt_params, &mlt_in).map_err(|e| {
        StarError::MicrophysicsError(format!("set_mlt failed at face {k}: {e}"))
    })?;

    // Convective conductivity cdc = (4*pi*r^2*rho)^2 * vc * Lambda / 3
    let area_rho = FOUR_PI * r_k * r_k * rho_f;
    let cdc_f = area_rho * area_rho * mlt_out.conv_vel * lambda_f / 3.0;

    model.scale_height[k] = hp_f.val;
    model.scale_height_ad[k] = hp_f;
    model.gradr[k] = gradr_f.val;
    model.gradr_ad[k] = gradr_f;
    model.gradl[k] = gradl_f.val;
    model.gradl_ad[k] = gradl_f;
    model.gradt[k] = mlt_out.grad_t.val;
    model.gradt_ad[k] = mlt_out.grad_t;
    model.mlt_vc[k] = mlt_out.conv_vel.val;
    model.mlt_vc_ad[k] = mlt_out.conv_vel;
    model.mlt_d[k] = mlt_out.d.val;
    model.mlt_d_ad[k] = mlt_out.d;
    model.mlt_cdc[k] = cdc_f.val;
    model.mlt_cdc_ad[k] = cdc_f;
    model.mlt_gamma[k] = mlt_out.gamma.val;
    model.mixing_type[k] = mlt_out.mixing_type;
    model.d_mix[k] = mlt_out.d.val;
    model.conv_vel[k] = mlt_out.conv_vel.val;

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

/// Evaluates the Brunt-Väisälä buoyancy frequency squared $N_k^2$ at all faces
/// (`do_brunt_N2` in `star/private/brunt.f90:297-334`).
pub fn do_brunt_n2(model: &mut StarModel) {
    let nz = model.nz;
    model.brunt_n2.fill(0.0);
    if !model.controls.calculate_brunt_n2 {
        return;
    }
    for k in 1..nz {
        let rho_f = model.interp_val_to_pt(&model.rho, k);
        let p_f = model.interp_val_to_pt(&model.peos, k).max(1.0e-30);
        let chi_t_f = model.interp_val_to_pt(&model.chi_t, k);
        let chi_rho_f = model.interp_val_to_pt(&model.chi_rho, k).max(1.0e-10);
        let grada_f = model.interp_val_to_pt(&model.grada, k);
        let g = model.grav[k];
        model.brunt_n2[k] = (g * g * rho_f / p_f)
            * (chi_t_f / chi_rho_f)
            * (grada_f + model.brunt_b[k] - model.gradt[k]);
    }
}

/// Evaluates the surface atmosphere boundary condition
/// $(P_{\mathrm{surf}}, T_{\mathrm{surf}})$ and its 3-cell stencil
/// [`StarAdOrder1`] partials (`eval_surf_bc` in
/// `star/private/hydro_vars.f90:2902-3224`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `eval_t_tau_uniform` fails.
pub fn eval_surf_bc(model: &mut StarModel) -> Result<(), StarError> {
    if model.nz == 0 {
        return Ok(());
    }
    let l_surf = model.lum[0].max(1.0);
    let r_surf = model.r[0].max(1.0);
    let m_surf = model.m_grav[0].max(1.0);
    let cgrav = model.cgrav[0];
    let tau_surf = model.controls.tau_factor * model.tau_base;
    let teff = atm_teff(l_surf, r_surf);
    model.teff = teff;

    let (kap_surf, dlnkap_dlnd, dlnkap_dlnt) =
        if model.controls.atm_option == AtmOption::Table {
            let dln_d = if model.lnd_start.is_empty() {
                0.0
            } else {
                model.lnd_start[0] - model.lnd[0]
            };
            let dln_t = if model.ln_t_start.is_empty() {
                0.0
            } else {
                model.ln_t_start[0] - model.ln_t[0]
            };
            let p_start = model.peos[0]
                * (model.chi_rho[0] * dln_d + model.chi_t[0] * dln_t)
                    .clamp(-5.0, 5.0)
                    .exp();
            let r_s = if model.ln_r_start.is_empty() {
                r_surf
            } else {
                model.ln_r_start[0].exp().max(1.0)
            };
            let dp_0 = (cgrav * m_surf * model.dm[0] / (2.0 * FOUR_PI))
                / r_s.powi(4);
            let p_rad = (1.0 / 3.0) * CRAD * teff.powi(4);
            let p_gas_surf = (p_start - dp_0 - 0.5 * p_rad).max(1.0);
            let g_s = cgrav * m_surf / (r_s * r_s);
            let kap_eff = (tau_surf * g_s / p_gas_surf).max(1.0e-10);
            (kap_eff, 0.0, 0.0)
        } else {
            let k0 = model.opacity[0].max(1.0e-10);
            (
                k0,
                model.d_opacity_dlnd[0] / k0,
                model.d_opacity_dlnt[0] / k0,
            )
        };

    let relation_id = match model.controls.atm_option {
        AtmOption::TTau | AtmOption::Table => {
            model.controls.atm_t_tau_relation.id()
        }
        AtmOption::SimplePhotosphere => ATM_T_TAU_EDDINGTON,
    };

    let dummy_eos =
        |_lnp: f64, _lnt: f64| -> Result<AtmEosCallbackResult, AtmError> {
            Err(AtmError::InvalidArgument {
                context: "eval_surf_bc",
                detail: String::from("unused callback when max_iters == 0"),
            })
        };
    let dummy_kap = |_lnrho: f64,
                     _lnt: f64,
                     _eos: &AtmEosCallbackResult|
     -> Result<AtmKapCallbackResult, AtmError> {
        Err(AtmError::InvalidArgument {
            context: "eval_surf_bc",
            detail: String::from("unused callback when max_iters == 0"),
        })
    };

    let atm_out = eval_t_tau_uniform(
        tau_surf,
        l_surf,
        r_surf,
        m_surf,
        cgrav,
        kap_surf,
        1.0,
        relation_id,
        dummy_eos,
        dummy_kap,
        1.0e-4,
        0,
        false,
        teff,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("eval_t_tau_uniform failed: {e}"))
    })?;

    let mut d1_lnp = [0.0; 18];
    d1_lnp[I_LND_00] = atm_out.dln_p_dln_kap * dlnkap_dlnd;
    d1_lnp[I_LNT_00] = atm_out.dln_p_dln_kap * dlnkap_dlnt;
    d1_lnp[I_LNR_00] = atm_out.dln_p_dln_r;
    d1_lnp[I_L_00] = atm_out.dln_p_dl;
    model.ln_peos_surf_ad = StarAdOrder1 {
        val: atm_out.ln_p,
        d1: d1_lnp,
    };

    let mut d1_lnt = [0.0; 18];
    d1_lnt[I_LND_00] = atm_out.dln_t_dln_kap * dlnkap_dlnd;
    d1_lnt[I_LNT_00] = atm_out.dln_t_dln_kap * dlnkap_dlnt;
    d1_lnt[I_LNR_00] = atm_out.dln_t_dln_r;
    d1_lnt[I_L_00] = atm_out.dln_t_dl;
    let ln_t_surf_val = if model.controls.atm_option == AtmOption::Table
        && !model.ln_t_start.is_empty()
        && !model.ln_r_start.is_empty()
        && !model.lum_start.is_empty()
    {
        let teff_s = atm_teff(
            model.lum_start[0].max(1.0),
            model.ln_r_start[0].exp().max(1.0),
        );
        model.ln_t_start[0] + (teff / teff_s.max(1.0)).ln()
    } else {
        atm_out.ln_t
    };
    model.ln_t_surf_ad = StarAdOrder1 {
        val: ln_t_surf_val,
        d1: d1_lnt,
    };

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
    do_brunt_b(model);
    set_mlt_vars(model)?;
    do_brunt_n2(model);
    eval_surf_bc(model)?;
    crate::eps_grav::eval_eps_grav(model);
    Ok(())
}
