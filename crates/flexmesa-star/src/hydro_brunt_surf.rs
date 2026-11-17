// SPDX-License-Identifier: LGPL-3.0-or-later
//! Brunt-Väisälä composition gradient and buoyancy frequency (`do_brunt_b`,
//! `do_brunt_n2` in `star/private/brunt.f90`) and surface atmosphere boundary
//! conditions (`eval_surf_bc` in `star/private/hydro_vars.f90`).

use flexmesa_atm::{
    atm_teff, eval_t_tau_uniform, eval_table, get_table_alfa_beta,
    get_table_base, AtmEosCallbackResult, AtmError, AtmEvalOutput,
    AtmKapCallbackResult,
};
use flexmesa_const::{CLIGHT, CRAD, L_SUN, M_SUN, PI};
use flexmesa_eos::{eos_dt_get, I_LN_PGAS};

use crate::controls::{AtmOption, StarError};
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LNR_00, I_LNT_00, I_L_00,
    NUM_STAR_AD_VARS,
};

/// Evaluates the face-centered composition gradient term $B_k$ (`do_brunt_B`
/// in `star/private/brunt.f90:57-330`).
///
/// Supports both the default Brassard et al. (1991) finite-difference EOS
/// formulation (`!model.controls.use_brunt_gradmux_form`, `brunt.f90:282-316`)
/// and the explicit mean-molecular-weight logarithmic derivative form
/// (`model.controls.use_brunt_gradmux_form`, `brunt.f90:202-238`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if face EOS evaluation fails or
/// produces non-finite $B_k$ (`star/private/brunt.f90:282-296,314-316`).
pub fn do_brunt_b(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    model.brunt_b.fill(0.0);
    model.aux.brunt_n2_composition_term.fill(0.0);
    if !model.controls.calculate_brunt_n2
        && !model.controls.use_ledoux_criterion
    {
        return Ok(());
    }

    let species = model.species;
    let use_gradmux = model.controls.use_brunt_gradmux_form || species == 0;
    let ht = crate::micro::builtin_helm_table()?;

    for k in 1..nz {
        let dlnp_peos = model.ln_peos[k - 1] - model.ln_peos[k];
        if dlnp_peos.abs() < 1.0e-14 {
            model.brunt_b[k] = 0.0;
            continue;
        }
        let chi_t_f = model.interp_val_to_pt(&model.chi_t, k).max(1.0e-10);

        let b_k = if use_gradmux {
            eval_gradmux_brunt_b(model, k, dlnp_peos, chi_t_f)
        } else {
            let xa_00 = &model.xa[k * species..(k + 1) * species];
            let xa_m1 = &model.xa[(k - 1) * species..k * species];
            let max_dx = xa_00
                .iter()
                .zip(xa_m1.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_dx < 1.0e-14 {
                0.0
            } else {
                let rho_f = model.interp_val_to_pt(&model.rho, k).max(1.0e-99);
                let t_f = model.interp_val_to_pt(&model.t, k).max(1.0);
                let prad_f = CRAD * flexmesa_math::pow4(t_f) / 3.0;
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
                (ln_p1 - ln_p2) / (dlnp_peos * chi_t_f)
            }
        };

        if !b_k.is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "non-finite brunt_B at face {k}: {b_k}"
            )));
        }
        model.brunt_b[k] = b_k;
    }
    Ok(())
}

#[inline]
fn eval_gradmux_brunt_b(
    model: &StarModel,
    k: usize,
    dlnp: f64,
    chi_t_f: f64,
) -> f64 {
    let mu_m1 = model.mu[k - 1].max(1.0e-10);
    let mu_00 = model.mu[k].max(1.0e-10);
    let dln_mu = flexmesa_math::log1p((mu_m1 - mu_00) / mu_00);
    let chi_rho_f = model.interp_val_to_pt(&model.chi_rho, k);
    (chi_rho_f / chi_t_f) * (dln_mu / dlnp)
}

/// Evaluates the Brunt-Väisälä buoyancy frequency squared $N_k^2$ and its
/// compositional contribution $N_{\mu,k}^2$ at all faces (`do_brunt_N2` in
/// `star/private/brunt.f90:297-334`).
pub fn do_brunt_n2(model: &mut StarModel) {
    let nz = model.nz;
    model.brunt_n2.fill(0.0);
    model.aux.brunt_n2_composition_term.fill(0.0);
    if !model.controls.calculate_brunt_n2 {
        return;
    }
    let use_rot = model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction;
    for k in 1..nz {
        let rho_f = model.interp_val_to_pt(&model.rho, k);
        let p_f = model.interp_val_to_pt(&model.peos, k).max(1.0e-30);
        let chi_t_f = model.interp_val_to_pt(&model.chi_t, k);
        let chi_rho_f = model.interp_val_to_pt(&model.chi_rho, k).max(1.0e-10);
        let grada_f = model.interp_val_to_pt(&model.grada, k);
        let fp_k = if use_rot {
            let fp = model.aux.fp_rot.get(k).copied().unwrap_or(1.0);
            if fp > 0.0 {
                fp
            } else {
                1.0
            }
        } else {
            1.0
        };
        let r_k = model.r[k].max(1.0);
        let g_sph = model.cgrav[k] * model.m_grav[k] / (r_k * r_k);
        let g = g_sph * fp_k;
        let prefactor = (g * g * rho_f / p_f) * (chi_t_f / chi_rho_f);
        let n2_comp = prefactor * model.brunt_b[k];
        model.aux.brunt_n2_composition_term[k] = n2_comp;
        model.brunt_n2[k] = prefactor * (grada_f - model.gradt[k]) + n2_comp;
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
    let l_surf = model.lum[0].max(1.0);
    let r_surf = model.r[0].max(1.0);
    let m_surf = model.m_grav[0].max(1.0);
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
            ((CRAD / 3.0) * flexmesa_math::pow4(t_start_0)).max(1.0e-30);
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
    let (kap_surf, dlnkap_dlnd, dlnkap_dlnt) = (
        k0,
        model.d_opacity_dlnd[0] / k0,
        model.d_opacity_dlnt[0] / k0,
    );

    let relation_id = model.controls.atm_t_tau_relation.id();

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

    let (lnp_ad, lnt_ad) = atm_out_to_ad(&atm_out, dlnkap_dlnd, dlnkap_dlnt);
    Ok((lnp_ad, lnt_ad))
}
