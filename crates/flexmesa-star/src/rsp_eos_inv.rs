// SPDX-License-Identifier: LGPL-3.0-or-later
//! Inverse `eosDT` root-finders, thermodynamic cell updaters, and Eddington
//! grey $T(\tau)$ surface boundary solver for `RSP`
//! (`star/private/rsp_eval_eos_and_kap.f90:409-1047` and
//! `star/private/atm_support.f90:1149-1220`).

use flexmesa_atm::{
    atm_teff, eval_t_tau_uniform, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, ATM_T_TAU_EDDINGTON,
};
use flexmesa_const::{AVO, CRAD, KERG, LN10, STANDARD_CGRAV};
use flexmesa_eos::{
    eos_dt_get_rho, eos_dt_get_t, HelmTable, I_ETA, I_LN_E, I_LN_FREE_E,
    I_LN_PGAS, WHICH_OTHER_EGAS, WHICH_OTHER_LOG_PTOT,
};
use flexmesa_kap::get_kap_results;

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_eos_kap::{
    effective_opacity_factor, eval1_mesa_eos_and_kap, RspChemCache,
    RspEosKapOut,
};

/// Ideal $\gamma$-law initial density estimate
/// $\rho = (P / T)\bar{A} / (N_{\mathrm{A}} k_{\mathrm{B}})$
/// (`eval1_gamma_PT_getRho` in `rsp_eval_eos_and_kap.f90:780-798`).
#[must_use]
pub fn eval1_gamma_pt_get_rho(chem: &RspChemCache, p: f64, t: f64) -> f64 {
    (p / t) * chem.abar / (AVO * KERG)
}

/// Solves $P_{\mathrm{tot}}(\rho, T) = P$ for $\rho$ with up to 9 tolerance
/// relaxations (`eval1_mesa_Rho_given_PT` in `rsp_eval_eos_and_kap.f90:725-777`).
pub fn eval1_mesa_rho_given_pt(
    model: &StarModel,
    chem: &RspChemCache,
    p: f64,
    t: f64,
    rho_guess: f64,
) -> Result<f64, StarError> {
    let (log_t, log_p, log_rho_guess) =
        (t.log10(), p.log10(), rho_guess.log10());
    let (mut rho_tol, mut p_tol) = (5.0e-10, 5.0e-13);
    for _ in 0..9 {
        if let Ok(inv) = eos_dt_get_rho(
            HelmTable::builtin(),
            &chem.chem_id,
            &chem.xa,
            log_t,
            WHICH_OTHER_LOG_PTOT,
            log_p,
            log_rho_guess,
            rho_tol,
            p_tol,
            100,
            &model.eos_config,
        ) {
            return Ok(inv.linear_result);
        }
        rho_tol *= 3.0;
        p_tol *= 3.0;
    }
    Err(StarError::MicrophysicsError(format!(
        "eval1_mesa_rho_given_pt failed at P={p}, T={t}"
    )))
}

/// Solves $P_{\mathrm{tot}}(1/V, T) = P$ for $T$ (`eval1_mesa_T_given_DP`,
/// `rsp_eval_eos_and_kap.f90:696-722`).
pub fn eval1_mesa_t_given_dp(
    model: &StarModel,
    chem: &RspChemCache,
    vol: f64,
    p: f64,
    t_guess: f64,
) -> Result<f64, StarError> {
    eos_dt_get_t(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        (1.0 / vol).log10(),
        WHICH_OTHER_LOG_PTOT,
        p.log10(),
        t_guess.log10(),
        1.0e-11,
        1.0e-11,
        100,
        &model.eos_config,
    )
    .map(|inv| inv.linear_result)
    .map_err(|e| {
        StarError::MicrophysicsError(format!("eval1_mesa_t_given_dp: {e}"))
    })
}

/// Solves $e_{\mathrm{gas}}(1/V, T) = e_{\mathrm{gas}}$ for $T$ and evaluates
/// `RSP` EOS + opacity (`eval1_mesa_eosDEgas_and_kap`, `rsp_eval_eos_and_kap.f90:409-504`).
pub fn eval1_mesa_eos_degas_and_kap(
    model: &mut StarModel,
    chem: &RspChemCache,
    cell_k: usize,
    skip_kap: bool,
    egas: f64,
    vol: f64,
) -> Result<(f64, RspEosKapOut), StarError> {
    let t_guess = model.t.get(cell_k).copied().unwrap_or(1.0e4).max(100.0);
    let inv = eos_dt_get_t(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        (1.0 / vol).log10(),
        WHICH_OTHER_EGAS,
        egas,
        t_guess.log10(),
        1.0e-11,
        egas.abs() * 1.0e-11,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("eval1_mesa_eos_degas: {e}"))
    })?;
    let t = inv.linear_result;
    let out =
        eval1_mesa_eos_and_kap(model, chem, Some(cell_k), skip_kap, t, vol)?;
    Ok((t, out))
}

/// Solves $e(1/V, T) = e$ for $T$ and evaluates `RSP` EOS + opacity
/// (`eval1_mesa_eosDE_and_kap`, `rsp_eval_eos_and_kap.f90:507-591`).
pub fn eval1_mesa_eos_de_and_kap(
    model: &mut StarModel,
    chem: &RspChemCache,
    cell_k: usize,
    skip_kap: bool,
    energy: f64,
    vol: f64,
) -> Result<(f64, RspEosKapOut), StarError> {
    let t_guess = model.t.get(cell_k).copied().unwrap_or(1.0e4).max(100.0);
    let inv = eos_dt_get_t(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        (1.0 / vol).log10(),
        I_LN_E as isize,
        energy.ln(),
        t_guess.log10(),
        1.0e-11,
        1.0e-11 * LN10,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("eval1_mesa_eos_de: {e}"))
    })?;
    let t = inv.linear_result;
    let out =
        eval1_mesa_eos_and_kap(model, chem, Some(cell_k), skip_kap, t, vol)?;
    Ok((t, out))
}

/// Updates cell `k`'s `rho` and `lnd` to match `model.pgas[k]` at fixed `T`
/// (`set_Rho_for_new_Pgas` in `rsp_eval_eos_and_kap.f90:849-891`).
pub fn set_rho_for_new_pgas(
    model: &mut StarModel,
    chem: &RspChemCache,
    k: usize,
) -> Result<(), StarError> {
    let inv = eos_dt_get_rho(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        model.ln_t[k] / LN10,
        I_LN_PGAS as isize,
        model.pgas[k].ln(),
        model.rho[k].log10(),
        1.0e-11,
        1.0e-11,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("set_rho_for_new_pgas({k}): {e}"))
    })?;
    model.lnd[k] = inv.log10_result * LN10;
    model.rho[k] = model.lnd[k].exp();
    Ok(())
}

/// Updates cell `k`'s `t` and `ln_t` to match `model.aux.rsp.egas_rsp[k]` at
/// fixed `lnd` (`set_T_for_new_egas` in `rsp_eval_eos_and_kap.f90:894-964`).
pub fn set_t_for_new_egas(
    model: &mut StarModel,
    chem: &RspChemCache,
    k: usize,
) -> Result<(), StarError> {
    let egas_want = model.aux.rsp.egas_rsp[k];
    let inv = eos_dt_get_t(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        model.lnd[k] / LN10,
        WHICH_OTHER_EGAS,
        egas_want,
        model.t[k].log10(),
        1.0e-11,
        egas_want.abs() * 1.0e-12,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("set_t_for_new_egas({k}): {e}"))
    })?;
    model.ln_t[k] = inv.log10_result * LN10;
    model.t[k] = model.ln_t[k].exp();
    Ok(())
}

/// Updates cell `k`'s `t` and `ln_t` to match `model.pgas[k]` at fixed `lnd`
/// (`set_T_for_new_Pgas` in `rsp_eval_eos_and_kap.f90:967-1006`).
pub fn set_t_for_new_pgas(
    model: &mut StarModel,
    chem: &RspChemCache,
    k: usize,
) -> Result<(), StarError> {
    let inv = eos_dt_get_t(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        model.lnd[k] / LN10,
        I_LN_PGAS as isize,
        model.pgas[k].ln(),
        model.t[k].log10(),
        1.0e-11,
        1.0e-11,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("set_t_for_new_pgas({k}): {e}"))
    })?;
    model.ln_t[k] = inv.log10_result * LN10;
    model.t[k] = model.ln_t[k].exp();
    Ok(())
}

/// Updates cell `k`'s `t` and `ln_t` to match `model.energy[k]` at fixed `lnd`
/// (`set_T_for_new_energy` in `rsp_eval_eos_and_kap.f90:1009-1045`).
pub fn set_t_for_new_energy(
    model: &mut StarModel,
    chem: &RspChemCache,
    k: usize,
    log_t_tol: f64,
    other_tol: f64,
) -> Result<(), StarError> {
    let inv = eos_dt_get_t(
        HelmTable::builtin(),
        &chem.chem_id,
        &chem.xa,
        model.lnd[k] / LN10,
        I_LN_E as isize,
        model.energy[k].ln(),
        model.t[k].log10(),
        log_t_tol,
        other_tol,
        100,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("set_t_for_new_energy({k}): {e}"))
    })?;
    model.ln_t[k] = inv.log10_result * LN10;
    model.t[k] = model.ln_t[k].exp();
    Ok(())
}

/// Solves Eddington grey $T(\tau)$ with iterated uniform opacity for surface
/// $(T, P, \kappa, T_{\mathrm{eff}})$ (`get_surf_P_T_kap` in
/// `rsp_eval_eos_and_kap.f90:801-828` and `atm_support.f90:1149-1220`).
pub fn get_surf_p_t_kap(
    model: &StarModel,
    chem: &RspChemCache,
    m: f64,
    r: f64,
    lum: f64,
    tau: f64,
    kap_guess: f64,
) -> Result<(f64, f64, f64, f64), StarError> {
    let teff = atm_teff(lum, r);
    let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
    let kap_init = model
        .opacity
        .first()
        .copied()
        .filter(|&k| k > 0.0)
        .unwrap_or(if kap_guess > 0.0 { kap_guess } else { 1.0e-2 });
    let eos_cb =
        |ln_p: f64, ln_t: f64| -> Result<AtmEosCallbackResult, AtmError> {
            let t = ln_t.exp();
            let p = ln_p.exp();
            let prad = CRAD * t.powi(4) / 3.0;
            let pgas = (p - prad).max(1.0e-99);
            let rho_guess = eval1_gamma_pt_get_rho(chem, p, t).max(1.0e-40);
            let inv = eos_dt_get_rho(
                HelmTable::builtin(),
                &chem.chem_id,
                &chem.xa,
                t.log10(),
                I_LN_PGAS as isize,
                pgas.ln(),
                rho_guess.log10(),
                1.0e-11,
                1.0e-11 * LN10,
                100,
                &model.eos_config,
            )
            .map_err(|e| AtmError::CallbackError {
                context: "get_surf_p_t_kap::eos",
                detail: e.to_string(),
            })?;
            Ok(AtmEosCallbackResult {
                ln_rho: inv.linear_result.ln(),
                res: inv.eos.res,
                dres_dln_rho: inv.eos.d_dlnd,
                dres_dln_t: inv.eos.d_dln_t,
            })
        };
    let kap_cb = |ln_rho: f64,
                  ln_t: f64,
                  eos_res: &AtmEosCallbackResult|
     -> Result<AtmKapCallbackResult, AtmError> {
        let log_rho = ln_rho / LN10;
        let log_t = ln_t / LN10;
        let ko = get_kap_results(
            &model.kap_config,
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
            context: "get_surf_p_t_kap::kap",
            detail: e.to_string(),
        })?;
        let f = effective_opacity_factor(&model.controls, log_t);
        Ok(AtmKapCallbackResult {
            kap: ko.kap * f,
            dlnkap_dln_rho: ko.dlnkap_dln_rho,
            dlnkap_dln_t: ko.dlnkap_dln_t,
        })
    };
    let atm = eval_t_tau_uniform(
        tau,
        lum,
        r,
        m,
        cgrav,
        kap_init,
        model.controls.pextra_factor,
        ATM_T_TAU_EDDINGTON,
        eos_cb,
        kap_cb,
        model.controls.rsp.atm_t_tau_errtol,
        model.controls.rsp.atm_t_tau_max_iters,
        true,
        teff,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!("get_surf_p_t_kap: {e}"))
    })?;
    Ok((atm.ln_t.exp(), atm.ln_p.exp(), atm.kap, teff))
}
