// SPDX-License-Identifier: GPL-3.0-or-later
//! Brunt-Väisälä composition gradient and buoyancy frequency (`do_brunt_b`,
//! `do_brunt_n2` in `star/private/brunt.f90`) and surface atmosphere boundary
//! conditions (`eval_surf_bc` in `star/private/hydro_vars.f90`).

use flexmesa_atm::{
    atm_black_body_t, eval_irradiated, eval_t_tau_uniform, eval_t_tau_varying,
    eval_table, get_t_tau_base, get_table_alfa_beta, get_table_base,
    AtmEosCallbackResult, AtmError, AtmEvalOutput, AtmKapCallbackResult,
};
use flexmesa_const::{BOLTZ_SIGMA, CRAD, FOUR_PI, LN10};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, EosConfig, HelmTable, I_ETA, I_LN_FREE_E,
    I_LN_PGAS,
};
use flexmesa_kap::{get_kap_results, KapConfig};

use crate::controls::{
    AtmIrradiatedOpacity, AtmOption, AtmTTauOpacity, StarControls, StarError,
};
use crate::model::{StarModel, SurfPt};
use crate::rsp_eos_kap::RspChemCache;
use crate::star_ad::{StarAdOrder1, I_LND_00, I_LNR_00, I_LNT_00, I_L_00};

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
    if !model.controls.calculate_brunt_b {
        // `brunt.f90:53-57`: set `brunt_B` and `unsmoothed_brunt_B` to NaN.
        model.brunt_b.fill(f64::NAN);
        model.aux.mixing.unsmoothed_brunt_b.fill(f64::NAN);
        return Ok(());
    }
    model.brunt_b.fill(0.0);
    model.aux.mixing.unsmoothed_brunt_b.fill(0.0);
    model.aux.brunt_n2_composition_term.fill(0.0);

    let species = model.species;
    let use_gradmux = model.controls.use_brunt_gradmux_form || species == 0;
    let use_mc = model.controls.use_mass_corrections;
    let ht = crate::micro::builtin_helm_table()?;

    for k in 1..nz {
        // `get_face_values` (`brunt.f90:207-216`).
        let rho_f = model.get_face_value(&model.rho, k);
        let t_f = model.get_face_value(&model.t, k);
        let chi_t_f = model.get_face_value(&model.chi_t, k);
        let chi_rho_f = model.get_face_value(&model.chi_rho, k);

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
pub fn do_brunt_n2(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if !(model.controls.calculate_brunt_b && model.controls.calculate_brunt_n2)
    {
        // `brunt.f90:128-132`: set `brunt_N2` and composition term to NaN.
        model.brunt_n2.fill(f64::NAN);
        model.aux.brunt_n2_composition_term.fill(f64::NAN);
        return Ok(());
    }
    model.brunt_n2.fill(0.0);
    model.aux.brunt_n2_composition_term.fill(0.0);

    let use_mc = model.controls.use_mass_corrections;
    let mut cell_fac = vec![0.0_f64; nz];
    for k in 0..nz {
        let mut v = (model.rho[k] / model.peos[k])
            * (model.chi_t[k] / model.chi_rho[k]);
        if use_mc {
            v *= model.aux.mass_correction[k];
        }
        cell_fac[k] = v;
    }
    // `get_face_values` (`star/private/star_utils.f90:2046-2060`).
    let mut face_fac = vec![0.0_f64; nz];
    face_fac[0] = cell_fac[0];
    for k in 1..nz {
        // `get_face_values` (`brunt.f90:144-145`).
        face_fac[k] = model.get_face_value(&cell_fac, k);
    }

    let min_mag = model.controls.mix_wind.min_magnitude_brunt_b;
    if model.aux.mixing.gradt_sub_grada.len() != nz {
        return Err(StarError::InvalidInput(String::from(
            "do_brunt_N2: gradT_sub_grada is unset (star/private/brunt.f90:139)",
        )));
    }
    for k in 0..nz {
        // `s% gradT_sub_grada(k)` (`brunt.f90:153-154`).
        let gradt_sub_grada = model.aux.mixing.gradt_sub_grada[k];
        if model.brunt_b[k].abs() < min_mag
            || model.gradt[k] == 0.0
            || !gradt_sub_grada.is_finite()
        {
            model.brunt_b[k] = 0.0;
            model.brunt_n2[k] = 0.0;
            model.aux.brunt_n2_composition_term[k] = 0.0;
            continue;
        }
        let f = flexmesa_math::pow2(model.grav[k]) * face_fac[k];
        if !f.is_finite() || !model.brunt_b[k].is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "do_brunt_N2: bad f = {f:e} or brunt_B = {:e} at face {} (star/private/brunt.f90:150-157)",
                model.brunt_b[k],
                k + 1
            )));
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
    Ok(())
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
        // 'irradiated_grey' (`atm_support.f90:201-204`).
        AtmOption::IrradiatedGrey => Ok(2.0 / 3.0),
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
    // `set_Teff_info_for_eqns` (`star/private/hydro_vars.f90:388-469`):
    // `set_phot_info` sets `s% Teff` from the photosphere's `L` and `r`
    // (`hydro_vars.f90:418-419`, `star/private/star_utils.f90:1094-1106`),
    // and `get_surf_PT` passes that `Teff` to the atmosphere
    // (`hydro_vars.f90:821, 962-966`).
    let phot = crate::report_utils::eval_photosphere_info(model);
    let teff = atm_black_body_t(phot.l_erg, phot.r_cm);
    model.teff = teff;
    if model.controls.rsp_flag {
        // `hydro_vars.f90:421-433`: `RSP_flag` sets `T_surf = T(1)` and
        // `P_surf = Peos(1)` with zero atmosphere partials.
        model.surf_pt = SurfPt {
            ln_t: model.ln_t[0],
            ln_p: model.ln_peos[0],
            ..SurfPt::default()
        };
        model.ln_peos_surf_ad = StarAdOrder1::constant(model.ln_peos[0]);
        model.ln_t_surf_ad = StarAdOrder1::constant(model.ln_t[0]);
        return Ok(());
    }
    if model.controls.do_starspots {
        // `starspot_tweak_PT` before `get_surf_PT` and `starspot_restore_PT`
        // after it (`hydro_vars.f90:443-456`,
        // `star/private/starspots.f90:68-102`): the atmosphere sees the
        // unspotted surface luminosity `L(1)/alp` and its `Teff`.
        let alp = 1.0 - model.controls.fspot
            + model.controls.fspot * flexmesa_math::pow4(model.controls.xspot);
        let l_init = model.lum[0];
        model.lum[0] /= alp;
        let teff_spot = flexmesa_math::pow(
            model.lum[0]
                / (FOUR_PI * flexmesa_math::pow2(model.r[0]) * BOLTZ_SIGMA),
            0.25,
        );
        let result = get_surf_pt(model, teff_spot);
        model.lum[0] = l_init;
        result?;
    } else {
        get_surf_pt(model, teff)?;
    }
    // `set_phot_info` is called again after `T_surf` is set
    // (`hydro_vars.f90:467`); `Teff` depends only on the photosphere's `L`
    // and `r`, so it is unchanged.
    set_surf_diagnostic_ad(model);
    Ok(())
}

/// Fills the diagnostic [`StarModel::ln_peos_surf_ad`] and
/// [`StarModel::ln_t_surf_ad`] from [`StarModel::surf_pt`], with the
/// `lnd`/`lnT` partials `dlnX_dlnkap*dlnkap_dlnd` from the cell-1 opacity
/// (`star/private/hydro_eqns.f90:922-928`). The surface equations do not read
/// these; they use `get_pt_bc_ad` in `hydro_eqns.rs`.
fn set_surf_diagnostic_ad(model: &mut StarModel) {
    let sp = model.surf_pt;
    let dlnkap_dlnd = model.d_opacity_dlnd[0] / model.opacity[0];
    let dlnkap_dlnt = model.d_opacity_dlnt[0] / model.opacity[0];
    let mut lnp = StarAdOrder1::constant(sp.ln_p);
    lnp.d1[I_LND_00] = sp.dlnp_dlnkap * dlnkap_dlnd;
    lnp.d1[I_LNT_00] = sp.dlnp_dlnkap * dlnkap_dlnt;
    lnp.d1[I_LNR_00] = sp.dlnp_dlnr;
    lnp.d1[I_L_00] = sp.dlnp_dl;
    let mut lnt = StarAdOrder1::constant(sp.ln_t);
    lnt.d1[I_LND_00] = sp.dlnt_dlnkap * dlnkap_dlnd;
    lnt.d1[I_LNT_00] = sp.dlnt_dlnkap * dlnkap_dlnt;
    lnt.d1[I_LNR_00] = sp.dlnt_dlnr;
    lnt.d1[I_L_00] = sp.dlnt_dl;
    model.ln_peos_surf_ad = lnp;
    model.ln_t_surf_ad = lnt;
}

/// `get_surf_PT` (`star/private/hydro_vars.f90:779-1030`) with `Teff = s% Teff`
/// as set by `set_Teff_info_for_eqns`. Stores the result in
/// [`StarModel::surf_pt`].
///
/// # Errors
/// Returns [`StarError`] from the atmosphere evaluation; `atm_option`
/// values other than `'T_tau'`, `'table'` and `'irradiated_grey'` cannot be
/// set (the `fixed_*` options are not ported, and the deprecated options stop
/// in `get_legacy`, `star/private/atm_support.f90:891-998`).
pub(crate) fn get_surf_pt(
    model: &mut StarModel,
    teff: f64,
) -> Result<(), StarError> {
    // `L_surf = s% L(1)`, `R_surf = s% r(1)` and `s% m(1)`, unclamped
    // (`star/private/hydro_vars.f90:817-821, 962-966`).
    let l_surf = model.lum[0];
    let r_surf = model.r[0];
    let m_surf = model.m[0];
    let cgrav = model.cgrav[0];
    let tau_surf = model.controls.tau_factor * model.tau_base;

    // `need_P_surf` and `need_T_surf` (`star/private/hydro_eqns.f90:757-779`),
    // both forced on with starspots (`hydro_vars.f90:443-445`). The solver
    // always has an `equL` equation here (`do_equL`).
    let c = &model.controls;
    let need_p_surf = !(c.use_compression_outer_bc
        || c.use_zero_pgas_outer_bc
        || c.use_fixed_psurf_outer_bc
        || c.use_fixed_vsurf_outer_bc);
    let need_t_surf = !(c.rsp2_flag || c.use_rsp_l_eqn_outer_bc);
    let (need_p_surf, need_t_surf) = if c.do_starspots {
        (true, true)
    } else {
        (need_p_surf, need_t_surf)
    };

    if !(need_p_surf || need_t_surf) {
        // Special-case boundary condition (`hydro_vars.f90:839-853`): the
        // start-of-step cell-1 values, with zero partials.
        let mut ln_p = model.ln_peos_start[0];
        if !ln_p.is_finite() {
            ln_p = 0.0;
        }
        let t_surf = flexmesa_math::exp(model.ln_t_start[0]);
        model.surf_pt = SurfPt {
            ln_t: flexmesa_math::log(t_surf),
            ln_p,
            ..SurfPt::default()
        };
        return Ok(());
    }

    model.surf_pt = match model.controls.atm_option {
        AtmOption::Table => {
            get_table(model, l_surf, r_surf, m_surf, cgrav, teff)?
        }
        AtmOption::IrradiatedGrey => eval_irradiated_surf(model)?,
        AtmOption::TTau => eval_grey_t_tau_surf(
            model, tau_surf, l_surf, r_surf, m_surf, cgrav, teff,
        )?,
    };
    Ok(())
}

/// `get_table` (`star/private/atm_support.f90:430-689`): blends the
/// tabulated atmosphere (weight `beta`) with the off-table `T_tau`
/// atmosphere at `tau = s% tau_base` (weight `alfa`), resetting `s% tau_base`
/// to the `T_tau` base when completely off the table and back to the table
/// base otherwise.
///
/// `atm_off_table_option` is not a supported control, so it always has its
/// default `'T_tau'` (`star/defaults/controls.defaults:4335`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when `tau_factor /= 1`
/// (`atm_support.f90:491-494`, a `mesa_error` upstream) and
/// [`StarError::MicrophysicsError`] for `L < 0` or a failing table or `T_tau`
/// evaluation.
fn get_table(
    model: &mut StarModel,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    teff: f64,
) -> Result<SurfPt, StarError> {
    if model.controls.tau_factor != 1.0 {
        return Err(StarError::InvalidInput(String::from(
            "Cannot use atm_option == 'table' with tau_factor /= 1",
        )));
    }
    let z = model.z_mass[0];
    if l < 0.0 {
        return Err(StarError::MicrophysicsError(String::from(
            "atm get_table: L < 0",
        )));
    }
    let table_id =
        crate::inlist_populate::atm_table_id(&model.controls.atm_table)
            .ok_or_else(|| {
                StarError::MicrophysicsError(format!(
                    "get_table_id failed in get_table: {:?}",
                    model.controls.atm_table
                ))
            })?;
    let (alfa, beta) = get_table_alfa_beta(l, teff, r, m, cgrav, table_id)
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "Call to atm_get_table_alfa_beta failed in get_table: {e}"
            ))
        })?;
    // `atm_support.f90:531-566`.
    let tau_base = if beta == 0.0 {
        get_t_tau_base(model.controls.atm_t_tau_relation.id()).map_err(|e| {
            StarError::MicrophysicsError(format!(
                "Call to atm_get_T_tau_base failed in get_table: {e}"
            ))
        })?
    } else {
        get_table_base(table_id).map_err(|e| {
            StarError::MicrophysicsError(format!(
                "Call to atm_get_table_base failed in get_table: {e}"
            ))
        })?
    };
    if model.tau_base != tau_base {
        model.tau_base = tau_base;
    }
    // `atm_support.f90:571-600`.
    let b = if beta == 0.0 {
        SurfPt::default()
    } else {
        let out = eval_table(l, r, m, cgrav, table_id, z, false, teff)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "Call to atm_eval_table failed in get_table: {e}"
                ))
            })?;
        surf_pt_from_atm(&out)
    };
    // `atm_support.f90:604-650`: `atm_off_table_option = 'T_tau'`.
    let a = if alfa == 0.0 {
        SurfPt::default()
    } else {
        let tau = model.tau_base;
        eval_grey_t_tau_surf(model, tau, l, r, m, cgrav, teff)?
    };
    // `atm_support.f90:654-668`: values and raw partials blended separately.
    Ok(SurfPt {
        ln_t: alfa * a.ln_t + beta * b.ln_t,
        ln_p: alfa * a.ln_p + beta * b.ln_p,
        dlnt_dl: alfa * a.dlnt_dl + beta * b.dlnt_dl,
        dlnt_dlnr: alfa * a.dlnt_dlnr + beta * b.dlnt_dlnr,
        dlnt_dlnm: alfa * a.dlnt_dlnm + beta * b.dlnt_dlnm,
        dlnt_dlnkap: alfa * a.dlnt_dlnkap + beta * b.dlnt_dlnkap,
        dlnp_dl: alfa * a.dlnp_dl + beta * b.dlnp_dl,
        dlnp_dlnr: alfa * a.dlnp_dlnr + beta * b.dlnp_dlnr,
        dlnp_dlnm: alfa * a.dlnp_dlnm + beta * b.dlnp_dlnm,
        dlnp_dlnkap: alfa * a.dlnp_dlnkap + beta * b.dlnp_dlnkap,
    })
}

/// The `lnT`/`lnP` outputs and raw partials of an `atm_lib` evaluation.
fn surf_pt_from_atm(o: &AtmEvalOutput) -> SurfPt {
    SurfPt {
        ln_t: o.ln_t,
        dlnt_dl: o.dln_t_dl,
        dlnt_dlnr: o.dln_t_dln_r,
        dlnt_dlnm: o.dln_t_dln_m,
        dlnt_dlnkap: o.dln_t_dln_kap,
        ln_p: o.ln_p,
        dlnp_dl: o.dln_p_dl,
        dlnp_dlnr: o.dln_p_dln_r,
        dlnp_dlnm: o.dln_p_dln_m,
        dlnp_dlnkap: o.dln_p_dln_kap,
    }
}

/// `s% opacity_start(1)`, saved by `save_start_values` before the solve.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if it was never saved (DECISIONS #43).
fn opacity_start_1(model: &StarModel) -> Result<f64, StarError> {
    let kap = model.aux.opacity_start_surf;
    if kap > 0.0 && kap.is_finite() {
        Ok(kap)
    } else {
        Err(StarError::InvalidInput(format!(
            "atmosphere kap guess: opacity_start(1) = {kap} is unset during solver iterations (star/private/atm_support.f90:301-306, 769-773; DECISIONS #43)"
        )))
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
        // `solve_eos_given_PgasT` passes `logPgas*ln10` with
        // `logPgas = log10(Pgas)` (`star/private/atm_support.f90:1221-1225`,
        // `star/private/eos_support.f90:340-346`).
        flexmesa_math::log10(pgas) * LN10,
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
fn eval_irradiated_surf(model: &mut StarModel) -> Result<SurfPt, StarError> {
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
    let m_surf = model.m[0];
    let cgrav = model.cgrav[0];
    let k0 = model.opacity[0];
    if !k0.is_finite() || k0 <= 0.0 {
        return Err(StarError::MicrophysicsError(format!(
            "eval_irradiated_surf requires finite surface opacity > 0, got {k0}"
        )));
    }

    // `if (s% solver_iter > 0) kap_for_atm = s% opacity_start(1)`
    // (`star/private/atm_support.f90:769-773`).
    let kap_for_atm = if model.aux.solver_iter > 0 {
        opacity_start_1(model)?
    } else {
        k0
    };

    let irr_out = match c.atm_irradiated_opacity {
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
            out
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
            out
        }
    };

    // `star/private/atm_support.f90:201-204, 830-840`: fixed `lnP_surf`, zero
    // pressure partials, `tau_base = 2/3`, and `tau_factor = tau_surf / tau_base`.
    let surf = SurfPt {
        ln_t: irr_out.ln_t,
        dlnt_dl: irr_out.dln_t_dl,
        dlnt_dlnr: irr_out.dln_t_dln_r,
        dlnt_dlnm: irr_out.dln_t_dln_m,
        dlnt_dlnkap: irr_out.dln_t_dln_kap,
        ln_p: flexmesa_math::log(p_surf),
        dlnp_dl: 0.0,
        dlnp_dlnr: 0.0,
        dlnp_dlnm: 0.0,
        dlnp_dlnkap: 0.0,
    };
    model.tau_base = 2.0 / 3.0;
    model.controls.tau_factor = irr_out.tau / model.tau_base;
    Ok(surf)
}

/// Evaluates surface $(P_{\mathrm{surf}}, T_{\mathrm{surf}})$ and partials for
/// `atm_option = 'T_tau'` across `'fixed'`, `'iterated'`, and `'varying'`
/// opacity prescriptions (`get_T_tau`, `eos_proc`, and `kap_proc` in
/// `star/private/atm_support.f90:218-392, 1176-1287`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_grey_t_tau_surf(
    model: &StarModel,
    tau_surf: f64,
    l_surf: f64,
    r_surf: f64,
    m_surf: f64,
    cgrav: f64,
    teff: f64,
) -> Result<SurfPt, StarError> {
    let relation_id = model.controls.atm_t_tau_relation.id();
    let errtol = model.controls.atm_t_tau_errtol;

    if model.controls.atm_t_tau_opacity == AtmTTauOpacity::Fixed {
        // `s% opacity(1)` (`star/private/atm_support.f90:280-290`).
        let kap_surf = model.opacity[0];
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
        return Ok(surf_pt_from_atm(&atm_out));
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
            // `if (s% solver_iter > 0) kap_guess = s% opacity_start(1)`
            // (`star/private/atm_support.f90:301-306`).
            let kap_guess = if model.aux.solver_iter > 0 {
                opacity_start_1(model)?
            } else {
                model.opacity[0]
            };
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

    // `'iterated'` and `'varying'` return zero `dlnT_dlnkap` and
    // `dlnP_dlnkap` (`atm/private/atm_t_tau_uniform.f90:231-232`,
    // `atm/private/atm_t_tau_varying.f90:180, 186`).
    Ok(surf_pt_from_atm(&atm_out))
}
