// SPDX-License-Identifier: LGPL-3.0-or-later
//! Brunt-Väisälä composition gradient and buoyancy frequency (`do_brunt_b`,
//! `do_brunt_n2` in `star/private/brunt.f90`) and surface atmosphere boundary
//! conditions (`eval_surf_bc` in `star/private/hydro_vars.f90`).

use flexmesa_atm::{
    atm_teff, eval_irradiated, eval_t_tau_uniform, eval_t_tau_varying,
    eval_table, get_t_tau_base, get_table_alfa_beta, get_table_base,
    AtmEosCallbackResult, AtmError, AtmEvalOutput, AtmKapCallbackResult,
};
use flexmesa_const::{CLIGHT, CRAD, FOUR_PI, LN10, L_SUN, M_SUN, PI};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, EosConfig, HelmTable, I_ETA, I_LN_FREE_E,
    I_LN_PGAS,
};
use flexmesa_kap::{get_kap_results, KapConfig};

use crate::controls::{
    AtmIrradiatedOpacity, AtmOption, AtmTTauOpacity, StarControls, StarError,
};
use crate::model::StarModel;
use crate::rsp_eos_kap::RspChemCache;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LNR_00, I_LNT_00, I_L_00,
    NUM_STAR_AD_VARS,
};

/// Evaluates the face-centered composition gradient term $B_k$ (`do_brunt_B`
/// in `star/private/brunt.f90:37-106,184-333`), stores `unsmoothed_brunt_B`,
/// and applies `threshold_smoothing` to `model.brunt_b`.
///
/// Supports both the default Brassard et al. (1991) finite-difference EOS
/// formulation (`!model.controls.use_brunt_gradmux_form`, `brunt.f90:231-333`)
/// and the explicit mean-molecular-weight logarithmic derivative form
/// (`model.controls.use_brunt_gradmux_form`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if face EOS evaluation fails or
/// produces non-finite $B_k$ (`star/private/brunt.f90:258-331`).
pub fn do_brunt_b(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.aux.mixing.unsmoothed_brunt_b.len() != nz {
        model.aux.mixing.resize(nz);
    }
    model.brunt_b.fill(0.0);
    model.aux.mixing.unsmoothed_brunt_b.fill(0.0);
    model.aux.brunt_n2_composition_term.fill(0.0);
    if !model.controls.calculate_brunt_b {
        return Ok(());
    }

    let species = model.species;
    let use_gradmux = model.controls.use_brunt_gradmux_form || species == 0;
    let use_mc = model.controls.use_mass_corrections;
    let ht = crate::micro::builtin_helm_table()?;

    for k in 1..nz {
        let rho_f = model.interp_val_to_pt(&model.rho, k);
        let t_f = model.interp_val_to_pt(&model.t, k);
        let chi_t_f = model.interp_val_to_pt(&model.chi_t, k);
        let chi_rho_f = model.interp_val_to_pt(&model.chi_rho, k);

        let mut delta_lnp = model.ln_peos[k - 1] - model.ln_peos[k];
        if delta_lnp > -1.0e-50 {
            // Hydrostatic equilibrium fallback (`star/private/brunt.f90:299-304`).
            let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
            let p_point =
                alfa * model.peos[k] + (1.0 - alfa) * model.peos[k - 1];
            let dlnp_dm = -model.cgrav[k] * model.m[k]
                / (FOUR_PI * flexmesa_math::pow4(model.r[k]) * p_point);
            delta_lnp = dlnp_dm * model.dm_bar[k];
        }

        let mut b_k = if use_gradmux {
            eval_gradmux_brunt_b(model, k, delta_lnp, chi_t_f, chi_rho_f)
        } else {
            let xa_00 = &model.xa[k * species..(k + 1) * species];
            let xa_m1 = &model.xa[(k - 1) * species..k * species];
            if !(rho_f > 0.0 && t_f > 0.0) {
                return Err(StarError::MicrophysicsError(format!(
                    "do_brunt_b invalid face rho/T at face {k}: rho={rho_f}, T={t_f}"
                )));
            }
            if xa_00 == xa_m1 {
                0.0
            } else {
                let prad_f = CRAD * t_f * t_f * t_f * t_f / 3.0;
                let out_00 = eos_dt_get(
                    ht,
                    rho_f,
                    t_f,
                    &model.chem_id,
                    xa_00,
                    &model.eos_config,
                )
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "do_brunt_b EOS call (zone {k}) failed: {e}"
                    ))
                })?;
                let out_m1 = eos_dt_get(
                    ht,
                    rho_f,
                    t_f,
                    &model.chem_id,
                    xa_m1,
                    &model.eos_config,
                )
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "do_brunt_b EOS call (zone {}) failed: {e}",
                        k - 1
                    ))
                })?;
                let ln_p1 = flexmesa_math::log(
                    prad_f + flexmesa_math::exp(out_00.res[I_LN_PGAS]),
                );
                let ln_p2 = flexmesa_math::log(
                    prad_f + flexmesa_math::exp(out_m1.res[I_LN_PGAS]),
                );
                (ln_p1 - ln_p2) / delta_lnp / chi_t_f
            }
        };

        if use_mc {
            let delta_ln_mbar =
                flexmesa_math::log(model.aux.mass_correction[k - 1])
                    - flexmesa_math::log(model.aux.mass_correction[k]);
            b_k -= chi_rho_f * delta_ln_mbar / delta_lnp / chi_t_f;
        }

        if !b_k.is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "non-finite brunt_B at face {k}: {b_k}"
            )));
        }
        model.brunt_b[k] = b_k;
    }

    model
        .aux
        .mixing
        .unsmoothed_brunt_b
        .copy_from_slice(&model.brunt_b[..nz]);

    let ns = model.controls.mix_wind.num_cells_for_smooth_brunt_b;
    if ns > 0 {
        let thresh = model.controls.mix_wind.threshold_for_smooth_brunt_b;
        let mut work = vec![0.0_f64; nz];
        crate::mixing_state::threshold_smoothing(
            &mut model.brunt_b[..nz],
            thresh,
            ns,
            false,
            &mut work,
        );
    }
    Ok(())
}

#[inline]
fn eval_gradmux_brunt_b(
    model: &StarModel,
    k: usize,
    dlnp: f64,
    chi_t_f: f64,
    chi_rho_f: f64,
) -> f64 {
    let mu_m1 = model.mu[k - 1].max(1.0e-10);
    let mu_00 = model.mu[k].max(1.0e-10);
    let dln_mu = flexmesa_math::log1p((mu_m1 - mu_00) / mu_00);
    (chi_rho_f / chi_t_f) * (dln_mu / dlnp)
}

/// Evaluates the Brunt-Väisälä buoyancy frequency squared $N_k^2$ and its
/// compositional contribution $N_{\mu,k}^2$ at all faces (`do_brunt_N2` in
/// `star/private/brunt.f90:110-181`).
pub fn do_brunt_n2(model: &mut StarModel) {
    let nz = model.nz;
    model.brunt_n2.fill(0.0);
    model.aux.brunt_n2_composition_term.fill(0.0);
    if nz == 0
        || !(model.controls.calculate_brunt_b
            && model.controls.calculate_brunt_n2)
    {
        return;
    }

    let use_mc = model.controls.use_mass_corrections;
    let mut cell_fac = vec![0.0_f64; nz];
    for k in 0..nz {
        let mut v =
            (model.rho[k] / model.peos[k]) * (model.chi_t[k] / model.chi_rho[k]);
        if use_mc {
            v *= model.aux.mass_correction[k];
        }
        cell_fac[k] = v;
    }
    let mut face_fac = vec![0.0_f64; nz];
    face_fac[0] = cell_fac[0];
    for k in 1..nz {
        face_fac[k] = model.interp_val_to_pt(&cell_fac, k);
    }

    let min_mag = model.controls.mix_wind.min_magnitude_brunt_b;
    let has_gradt_sub_grada = model.aux.mixing.gradt_sub_grada.len() == nz;
    for k in 0..nz {
        let gradt_sub_grada = if has_gradt_sub_grada {
            model.aux.mixing.gradt_sub_grada[k]
        } else {
            let grada_f = if k == 0 {
                model.grada[0]
            } else {
                model.interp_val_to_pt(&model.grada, k)
            };
            model.gradt[k] - grada_f
        };
        if model.brunt_b[k].abs() < min_mag
            || model.gradt[k] == 0.0
            || !gradt_sub_grada.is_finite()
        {
            model.brunt_b[k] = 0.0;
            model.brunt_n2[k] = 0.0;
            model.aux.brunt_n2_composition_term[k] = 0.0;
            continue;
        }
        let g = if model.grav[k] > 0.0 {
            model.grav[k]
        } else {
            let r_k = model.r[k].max(1.0);
            model.cgrav[k] * model.m_grav[k] / (r_k * r_k)
        };
        let f = flexmesa_math::pow2(g) * face_fac[k];
        if !f.is_finite() || !model.brunt_b[k].is_finite() {
            model.brunt_b[k] = 0.0;
            model.brunt_n2[k] = 0.0;
            model.aux.brunt_n2_composition_term[k] = 0.0;
            continue;
        }
        model.brunt_n2[k] = f * (model.brunt_b[k] - gradt_sub_grada);
        model.aux.brunt_n2_composition_term[k] = f * model.brunt_b[k];
    }

    let coeff = model.controls.mix_wind.brunt_n2_coefficient;
    if coeff != 1.0 {
        for k in 0..nz {
            model.brunt_n2[k] *= coeff;
            model.aux.brunt_n2_composition_term[k] *= coeff;
        }
    }
}

/// `get_atm_tau_base` (`star/private/atm_support.f90:150-215`): the base
/// optical depth for the `atm_option` in `controls`. `do_star_job_controls_before`
/// stores it in `s% tau_base` before the model is loaded or built
/// (`star/job/run_star_support.f90:1546`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] for an unknown $T(\tau)$ relation
/// or atmosphere table.
pub(crate) fn get_atm_tau_base(
    controls: &StarControls,
) -> Result<f64, StarError> {
    match controls.atm_option {
        AtmOption::TTau => get_t_tau_base(controls.atm_t_tau_relation.id())
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "Call to atm_get_T_tau_base failed in get_atm_tau_base: {e}"
                ))
            }),
        AtmOption::Table => {
            let table_id = crate::inlist_populate::atm_table_id(
                &controls.atm_table,
            )
            .ok_or_else(|| {
                StarError::MicrophysicsError(format!(
                    "Call to get_table_id failed in get_atm_tau_base: {:?}",
                    controls.atm_table
                ))
            })?;
            get_table_base(table_id).map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "Call to atm_get_table_base failed in get_atm_tau_base: {e}"
                ))
            })
        }
        // 'irradiated_grey' and every other option (`atm_support.f90:201-209`).
        AtmOption::IrradiatedGrey | AtmOption::SimplePhotosphere => {
            Ok(2.0 / 3.0)
        }
    }
}

/// Evaluates the surface atmosphere boundary condition
/// $(P_{\mathrm{surf}}, T_{\mathrm{surf}})$ and its 3-cell stencil
/// [`StarAdOrder1`] partials (`get_surf_PT` in
/// `star/private/hydro_vars.f90:779-1030` and `get_atm_PT` in
/// `star/private/atm_support.f90:37-147`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `eval_t_tau_uniform` fails.
pub fn eval_surf_bc(model: &mut StarModel) -> Result<(), StarError> {
    if model.nz == 0 {
        return Ok(());
    }
    // `L_surf = s% L(1)`, `R_surf = s% r(1)` and `s% m(1)`, unclamped
    // (`star/private/hydro_vars.f90:817-821, 962-966`).
    let l_surf = model.lum[0];
    let r_surf = model.r[0];
    let m_surf = model.m[0];
    let cgrav = model.cgrav[0];
    let tau_surf = model.controls.tau_factor * model.tau_base;
    let teff = atm_teff(l_surf, r_surf);
    model.teff = teff;

    if model.controls.use_zero_pgas_outer_bc {
        let t_start_0 = flexmesa_math::exp(
            model.ln_t_start.first().copied().unwrap_or(model.ln_t[0]),
        )
        .max(1.0);
        model.ln_t_surf_ad =
            StarAdOrder1::constant(flexmesa_math::log(t_start_0));
        let p_rad_0 =
            // `Radiation_Pressure(s% T_start(1))` (`star/private/hydro_eqns.f90:761`,
            // `eos/public/eos_lib.f90:91`: `crad*T*T*T*T/3d0`).
            (CRAD * t_start_0 * t_start_0 * t_start_0 * t_start_0 / 3.0)
                .max(1.0e-30);
        model.ln_peos_surf_ad =
            StarAdOrder1::constant(flexmesa_math::log(p_rad_0));
        return Ok(());
    }

    if model.controls.atm_option == AtmOption::SimplePhotosphere {
        // Direct radiation-pressure corrected simple photosphere (`hydro_vars.f90:857-916`):
        // P_surf = (tau_surf * g / kap_0) * (1 + 1.6e-4 * pextra_factor * kap_0 * (L/Lsun)/(M/Msun))
        // T_surf = Teff * (0.75 * tau_surf + 0.5)^(1/4)
        let lnr_0 = StarAdOrder1::variable(model.ln_r[0], I_LNR_00);
        let r_0 = lnr_0.exp();
        let lum_0 = StarAdOrder1::variable(l_surf, I_L_00);
        let kap_0 = StarAdOrder1::from_dlnd_dlnt_00(
            model.opacity[0].max(1.0e-10),
            model.d_opacity_dlnd[0],
            model.d_opacity_dlnt[0],
        );
        let g_0 = (cgrav * m_surf) / (r_0 * r_0);
        let pextra_coeff = 1.6e-4 * model.controls.pextra_factor
            / ((m_surf / M_SUN).max(1.0e-10) * L_SUN);
        let pextra = StarAdOrder1::constant(1.0) + kap_0 * lum_0 * pextra_coeff;
        let p_surf = (g_0 * tau_surf / kap_0) * pextra;
        model.ln_peos_surf_ad = p_surf.log();

        let t_surf4 = lum_0 * (0.75 * tau_surf.max(0.0) + 0.5)
            / (PI * CRAD * CLIGHT * r_0 * r_0);
        model.ln_t_surf_ad = t_surf4.log() * 0.25;
        return Ok(());
    }

    if model.controls.atm_option == AtmOption::Table {
        if model.controls.atm_table.trim().is_empty() {
            return Err(StarError::InvalidInput(String::from(
                "atm_option = 'table' requires a non-empty atm_table",
            )));
        }
        let table_id =
            crate::inlist_populate::atm_table_id(&model.controls.atm_table)
                .ok_or_else(|| {
                    StarError::MicrophysicsError(format!(
                        "unknown atm_table {:?}",
                        model.controls.atm_table
                    ))
                })?;
        model.tau_base = get_table_base(table_id).map_err(|e| {
            StarError::MicrophysicsError(format!("get_table_base failed: {e}"))
        })?;
        let tbl_tau_surf = model.controls.tau_factor * model.tau_base;
        let z_surf = model.z_mass.first().copied().unwrap_or(0.02);
        let (alfa, beta) =
            get_table_alfa_beta(l_surf, teff, r_surf, m_surf, cgrav, table_id)
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "get_table_alfa_beta failed for '{}': {e}",
                        model.controls.atm_table
                    ))
                })?;
        if beta > 0.0 && alfa < 1.0 {
            let tbl_out = eval_table(
                l_surf, r_surf, m_surf, cgrav, table_id, z_surf, false, teff,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "atm table '{}' evaluation failed: {e}",
                    model.controls.atm_table
                ))
            })?;
            let (tbl_lnp, tbl_lnt) = atm_out_to_ad(&tbl_out, 0.0, 0.0);
            if alfa <= 0.0 {
                model.ln_peos_surf_ad = tbl_lnp;
                model.ln_t_surf_ad = tbl_lnt;
            } else {
                let (grey_lnp, grey_lnt) = eval_grey_t_tau_surf(
                    model,
                    tbl_tau_surf,
                    l_surf,
                    r_surf,
                    m_surf,
                    cgrav,
                    teff,
                )?;
                model.ln_peos_surf_ad =
                    blend_surf_ad(alfa, grey_lnp, beta, tbl_lnp);
                model.ln_t_surf_ad =
                    blend_surf_ad(alfa, grey_lnt, beta, tbl_lnt);
            }
            return Ok(());
        }
        let (grey_lnp, grey_lnt) = eval_grey_t_tau_surf(
            model,
            tbl_tau_surf,
            l_surf,
            r_surf,
            m_surf,
            cgrav,
            teff,
        )?;
        model.ln_peos_surf_ad = grey_lnp;
        model.ln_t_surf_ad = grey_lnt;
        return Ok(());
    }

    if model.controls.atm_option == AtmOption::IrradiatedGrey {
        return eval_irradiated_surf(model);
    }

    let (lnp_ad, lnt_ad) = eval_grey_t_tau_surf(
        model, tau_surf, l_surf, r_surf, m_surf, cgrav, teff,
    )?;
    model.ln_peos_surf_ad = lnp_ad;
    model.ln_t_surf_ad = lnt_ad;
    Ok(())
}

fn atm_out_to_ad(
    atm_out: &AtmEvalOutput,
    dlnkap_dlnd: f64,
    dlnkap_dlnt: f64,
) -> (StarAdOrder1, StarAdOrder1) {
    let mut d1_lnp = [0.0; NUM_STAR_AD_VARS];
    d1_lnp[I_LND_00] = atm_out.dln_p_dln_kap * dlnkap_dlnd;
    d1_lnp[I_LNT_00] = atm_out.dln_p_dln_kap * dlnkap_dlnt;
    d1_lnp[I_LNR_00] = atm_out.dln_p_dln_r;
    d1_lnp[I_L_00] = atm_out.dln_p_dl;

    let mut d1_lnt = [0.0; NUM_STAR_AD_VARS];
    d1_lnt[I_LND_00] = atm_out.dln_t_dln_kap * dlnkap_dlnd;
    d1_lnt[I_LNT_00] = atm_out.dln_t_dln_kap * dlnkap_dlnt;
    d1_lnt[I_LNR_00] = atm_out.dln_t_dln_r;
    d1_lnt[I_L_00] = atm_out.dln_t_dl;

    (
        StarAdOrder1 {
            val: atm_out.ln_p,
            d1: d1_lnp,
        },
        StarAdOrder1 {
            val: atm_out.ln_t,
            d1: d1_lnt,
        },
    )
}

fn blend_surf_ad(
    alfa: f64,
    grey: StarAdOrder1,
    beta: f64,
    tbl: StarAdOrder1,
) -> StarAdOrder1 {
    let mut d1 = [0.0; NUM_STAR_AD_VARS];
    for (i, slot) in d1.iter_mut().enumerate() {
        *slot = alfa * grey.d1[i] + beta * tbl.d1[i];
    }
    StarAdOrder1 {
        val: alfa * grey.val + beta * tbl.val,
        d1,
    }
}

fn atm_eos_proc(
    ht: &HelmTable,
    chem: &RspChemCache,
    eos_config: &EosConfig,
    ln_p: f64,
    ln_t: f64,
    context: &'static str,
) -> Result<AtmEosCallbackResult, AtmError> {
    let t = flexmesa_math::exp(ln_t);
    let p = flexmesa_math::exp(ln_p);
    // `Prad = radiation_pressure(T)` (`star/private/atm_support.f90:1208`,
    // `eos/public/eos_lib.f90:91`: `crad*T*T*T*T/3d0`).
    let prad = CRAD * t * t * t * t / 3.0;
    let pgas = (p - prad).max(1.0e-99);
    let rho_guess =
        crate::rsp_eos_inv::eval1_gamma_pt_get_rho(chem, p, t).max(1.0e-40);
    let inv = eos_dt_get_rho(
        ht,
        &chem.chem_id,
        &chem.xa,
        ln_t / LN10,
        I_LN_PGAS as isize,
        flexmesa_math::log(pgas),
        flexmesa_math::log10(rho_guess),
        1.0e-11,
        1.0e-11 * LN10,
        100,
        eos_config,
    )
    .map_err(|e| AtmError::CallbackError {
        context,
        detail: e.to_string(),
    })?;
    Ok(AtmEosCallbackResult {
        ln_rho: inv.log10_result * LN10,
        res: inv.eos.res,
        dres_dln_rho: inv.eos.d_dlnd,
        dres_dln_t: inv.eos.d_dln_t,
    })
}

fn atm_kap_proc(
    controls: &StarControls,
    kap_config: &KapConfig,
    chem: &RspChemCache,
    ln_rho: f64,
    ln_t: f64,
    eos_res: &AtmEosCallbackResult,
    context: &'static str,
) -> Result<AtmKapCallbackResult, AtmError> {
    let log_rho = ln_rho / LN10;
    let log_t = ln_t / LN10;
    let (kap_raw, mut dlnkap_dln_rho, mut dlnkap_dln_t) =
        if controls.use_simple_es_for_kap {
            (0.2 * (1.0 + chem.x), 0.0, 0.0)
        } else {
            let ko = get_kap_results(
                kap_config,
                chem.zbar,
                chem.x,
                chem.z,
                chem.xc,
                chem.xn,
                chem.xo,
                chem.xne,
                log_rho,
                log_t,
                eos_res.res[I_LN_FREE_E],
                eos_res.dres_dln_rho[I_LN_FREE_E],
                eos_res.dres_dln_t[I_LN_FREE_E],
                eos_res.res[I_ETA],
                eos_res.dres_dln_rho[I_ETA],
                eos_res.dres_dln_t[I_ETA],
            )
            .map_err(|e| AtmError::CallbackError {
                context,
                detail: e.to_string(),
            })?;
            (ko.kap, ko.dlnkap_dln_rho, ko.dlnkap_dln_t)
        };
    let f = crate::rsp_eos_kap::effective_opacity_factor(controls, log_t);
    let mut kap = kap_raw * f;
    if controls.opacity_max > 0.0 && kap > controls.opacity_max {
        kap = controls.opacity_max;
        dlnkap_dln_rho = 0.0;
        dlnkap_dln_t = 0.0;
    }
    if controls.opacity_min > 0.0 && kap < controls.opacity_min {
        kap = controls.opacity_min;
        dlnkap_dln_rho = 0.0;
        dlnkap_dln_t = 0.0;
    }
    if !(kap.is_finite() && kap > 0.0) {
        return Err(AtmError::CallbackError {
            context,
            detail: format!(
                "bad opacity {kap} at logRho={log_rho}, logT={log_t}"
            ),
        });
    }
    Ok(AtmKapCallbackResult {
        kap,
        dlnkap_dln_rho,
        dlnkap_dln_t,
    })
}

fn unused_eos_proc(
    _ln_p: f64,
    _ln_t: f64,
) -> Result<AtmEosCallbackResult, AtmError> {
    Err(AtmError::InvalidArgument {
        context: "eval_surf_bc",
        detail: String::from("unused callback when max_iters == 0"),
    })
}

fn unused_kap_proc(
    _ln_rho: f64,
    _ln_t: f64,
    _eos: &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    Err(AtmError::InvalidArgument {
        context: "eval_surf_bc",
        detail: String::from("unused callback when max_iters == 0"),
    })
}

/// Evaluates surface $(P_{\mathrm{surf}}, T_{\mathrm{surf}})$ and partials for
/// `atm_option = 'irradiated_grey'`, updating `model.tau_base` and
/// `model.controls.tau_factor` (`get_atm_tau_base` and `get_irradiated` in
/// `star/private/atm_support.f90:201-204, 736-888`, `atm/private/atm_irradiated.f90:35-332`,
/// and `star/private/hydro_eqns.f90:922-966`).
fn eval_irradiated_surf(model: &mut StarModel) -> Result<(), StarError> {
    let c = &model.controls;
    let t_eq = c.atm_irradiated_t_eq;
    let p_surf = c.atm_irradiated_p_surf;
    let kap_v = c.atm_irradiated_kap_v;
    let gamma = c.atm_irradiated_kap_v_div_kap_th;
    let errtol = c.atm_irradiated_errtol;
    let max_iters = c.atm_irradiated_max_iters;

    if !p_surf.is_finite() || p_surf <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "atm_irradiated_P_surf must be > 0, got {p_surf}"
        )));
    }
    if !t_eq.is_finite() || t_eq < 0.0 {
        return Err(StarError::InvalidInput(format!(
            "atm_irradiated_T_eq must be >= 0, got {t_eq}"
        )));
    }
    if !gamma.is_finite() || gamma < 0.0 {
        return Err(StarError::InvalidInput(format!(
            "atm_irradiated_kap_v_div_kap_th must be >= 0, got {gamma}"
        )));
    }
    if !kap_v.is_finite() || kap_v < 0.0 || (gamma == 0.0 && kap_v <= 0.0) {
        return Err(StarError::InvalidInput(format!(
            "atm_irradiated_kap_v must be >= 0 (and > 0 when atm_irradiated_kap_v_div_kap_th == 0), got kap_v={kap_v}, gamma={gamma}"
        )));
    }
    if c.atm_irradiated_opacity == AtmIrradiatedOpacity::Iterated {
        if max_iters == 0 {
            return Err(StarError::InvalidInput(String::from(
                "atm_irradiated_opacity = 'iterated' requires atm_irradiated_max_iters > 0",
            )));
        }
        if !errtol.is_finite() || errtol <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "atm_irradiated_errtol must be > 0 when atm_irradiated_opacity = 'iterated', got {errtol}"
            )));
        }
    }

    let l_surf = model.lum[0];
    let r_surf = model.r[0];
    let m_surf = model.m_grav[0];
    let cgrav = model.cgrav[0];
    let k0 = model.opacity[0];
    if !k0.is_finite() || k0 <= 0.0 {
        return Err(StarError::MicrophysicsError(format!(
            "eval_irradiated_surf requires finite surface opacity > 0, got {k0}"
        )));
    }

    // `star/private/atm_support.f90:769-773`: hold `kap_for_atm` at
    // `opacity_start(1)` during solver iterations.
    let in_solver_iter =
        model.aux.doing_struct_burn_mix && model.aux.solver_mlt_eval_count > 0;
    let kap_for_atm = if in_solver_iter && model.aux.opacity_start_surf > 0.0 {
        model.aux.opacity_start_surf
    } else {
        k0
    };

    let (irr_out, dlnkap_dlnd, dlnkap_dlnt) = match c.atm_irradiated_opacity {
        AtmIrradiatedOpacity::Fixed => {
            let out = eval_irradiated(
                l_surf,
                r_surf,
                m_surf,
                cgrav,
                t_eq,
                p_surf,
                kap_for_atm,
                kap_v,
                gamma,
                unused_eos_proc,
                unused_kap_proc,
                errtol,
                0,
                false,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "eval_irradiated (fixed) failed: {e}"
                ))
            })?;
            (
                out,
                model.d_opacity_dlnd[0] / k0,
                model.d_opacity_dlnt[0] / k0,
            )
        }
        AtmIrradiatedOpacity::Iterated => {
            let chem = RspChemCache::from_model_zone0(model)?;
            let ht = crate::micro::builtin_helm_table()?;
            let eos_cb = |ln_p: f64, ln_t: f64| {
                atm_eos_proc(
                    ht,
                    &chem,
                    &model.eos_config,
                    ln_p,
                    ln_t,
                    "eval_irradiated_surf::eos_proc",
                )
            };
            let kap_cb =
                |ln_rho: f64, ln_t: f64, eos_res: &AtmEosCallbackResult| {
                    atm_kap_proc(
                        &model.controls,
                        &model.kap_config,
                        &chem,
                        ln_rho,
                        ln_t,
                        eos_res,
                        "eval_irradiated_surf::kap_proc",
                    )
                };
            let out = eval_irradiated(
                l_surf,
                r_surf,
                m_surf,
                cgrav,
                t_eq,
                p_surf,
                kap_for_atm,
                kap_v,
                gamma,
                eos_cb,
                kap_cb,
                errtol,
                max_iters,
                false,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "eval_irradiated (iterated) failed: {e}"
                ))
            })?;
            (out, 0.0, 0.0)
        }
    };

    // `star/private/atm_support.f90:201-204, 830-840`: fixed `lnP_surf`, zero
    // pressure partials, `tau_base = 2/3`, and `tau_factor = tau_surf / tau_base`.
    let atm_out = AtmEvalOutput {
        ln_t: irr_out.ln_t,
        dln_t_dl: irr_out.dln_t_dl,
        dln_t_dln_r: irr_out.dln_t_dln_r,
        dln_t_dln_m: irr_out.dln_t_dln_m,
        dln_t_dln_kap: irr_out.dln_t_dln_kap,
        ln_p: flexmesa_math::log(p_surf),
        dln_p_dl: 0.0,
        dln_p_dln_r: 0.0,
        dln_p_dln_m: 0.0,
        dln_p_dln_kap: 0.0,
        kap: irr_out.kap,
    };
    let (lnp_ad, lnt_ad) = atm_out_to_ad(&atm_out, dlnkap_dlnd, dlnkap_dlnt);
    model.ln_peos_surf_ad = lnp_ad;
    model.ln_t_surf_ad = lnt_ad;
    model.tau_base = 2.0 / 3.0;
    model.controls.tau_factor = irr_out.tau / model.tau_base;
    Ok(())
}

/// Evaluates surface $(P_{\mathrm{surf}}, T_{\mathrm{surf}})$ and partials for
/// `atm_option = 'T_tau'` across `'fixed'`, `'iterated'`, and `'varying'`
/// opacity prescriptions (`get_T_tau`, `eos_proc`, and `kap_proc` in
/// `star/private/atm_support.f90:218-392, 1176-1287`).
#[allow(clippy::too_many_arguments)]
fn eval_grey_t_tau_surf(
    model: &StarModel,
    tau_surf: f64,
    l_surf: f64,
    r_surf: f64,
    m_surf: f64,
    cgrav: f64,
    teff: f64,
) -> Result<(StarAdOrder1, StarAdOrder1), StarError> {
    let k0 = model.opacity[0].max(1.0e-10);
    let relation_id = model.controls.atm_t_tau_relation.id();
    let errtol = model.controls.atm_t_tau_errtol;

    if model.controls.atm_t_tau_opacity == AtmTTauOpacity::Fixed {
        let (kap_surf, dlnkap_dlnd, dlnkap_dlnt) = (
            k0,
            model.d_opacity_dlnd[0] / k0,
            model.d_opacity_dlnt[0] / k0,
        );
        let atm_out = eval_t_tau_uniform(
            tau_surf,
            l_surf,
            r_surf,
            m_surf,
            cgrav,
            kap_surf,
            model.controls.pextra_factor,
            relation_id,
            unused_eos_proc,
            unused_kap_proc,
            errtol,
            0,
            false,
            teff,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "eval_t_tau_uniform failed: {e}"
            ))
        })?;
        return Ok(atm_out_to_ad(&atm_out, dlnkap_dlnd, dlnkap_dlnt));
    }

    // Live EOS and opacity callbacks matching `eos_proc` and `kap_proc` in
    // `star/private/atm_support.f90:1176-1287`.
    let chem = RspChemCache::from_model_zone0(model)?;
    let ht = crate::micro::builtin_helm_table()?;
    let eos_cb =
        |ln_p: f64, ln_t: f64| -> Result<AtmEosCallbackResult, AtmError> {
            atm_eos_proc(
                ht,
                &chem,
                &model.eos_config,
                ln_p,
                ln_t,
                "eval_grey_t_tau_surf::eos_proc",
            )
        };
    let kap_cb = |ln_rho: f64,
                  ln_t: f64,
                  eos_res: &AtmEosCallbackResult|
     -> Result<AtmKapCallbackResult, AtmError> {
        atm_kap_proc(
            &model.controls,
            &model.kap_config,
            &chem,
            ln_rho,
            ln_t,
            eos_res,
            "eval_grey_t_tau_surf::kap_proc",
        )
    };

    let atm_out = match model.controls.atm_t_tau_opacity {
        AtmTTauOpacity::Fixed => unreachable!("handled above"),
        AtmTTauOpacity::Iterated => {
            // `star/private/atm_support.f90:301-306`: start iterations from
            // `opacity_start(1)` during solver iterations for consistency.
            let in_solver_iter = model.aux.doing_struct_burn_mix
                && model.aux.solver_mlt_eval_count > 0;
            let kap_guess =
                if in_solver_iter && model.aux.opacity_start_surf > 0.0 {
                    model.aux.opacity_start_surf
                } else {
                    k0
                }
                .max(1.0e-10);
            eval_t_tau_uniform(
                tau_surf,
                l_surf,
                r_surf,
                m_surf,
                cgrav,
                kap_guess,
                model.controls.pextra_factor,
                relation_id,
                eos_cb,
                kap_cb,
                errtol,
                model.controls.atm_t_tau_max_iters,
                false,
                teff,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "eval_t_tau_uniform (iterated) failed: {e}"
                ))
            })?
        }
        AtmTTauOpacity::Varying => eval_t_tau_varying(
            tau_surf,
            l_surf,
            r_surf,
            m_surf,
            cgrav,
            relation_id,
            eos_cb,
            kap_cb,
            errtol,
            model.controls.atm_t_tau_max_steps,
            false,
            teff,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "eval_t_tau_varying failed: {e}"
            ))
        })?,
    };

    Ok(atm_out_to_ad(&atm_out, 0.0, 0.0))
}
