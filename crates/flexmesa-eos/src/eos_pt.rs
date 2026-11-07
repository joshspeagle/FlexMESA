// SPDX-License-Identifier: LGPL-3.0-or-later
//! `(P_gas, T)` equation of state evaluation (`eos_pt_get`) and inverse solvers
//! (`eos_pt_get_t`, `eos_pt_get_pgas`).
//!
//! Ports `Get_eosPT_Results`, `Get_PT_Results_using_DT`, `get_T`, `get_Pgas`,
//! and `do_safe_get_Pgas_T` from upstream MESA `eos/private/eospt_eval.f90`.

use crate::blend_dt::EosConfig;
use crate::helm::HelmTable;
use crate::inverse::{brent_safe_zero, eos_dt_get_rho, look_for_brackets};
use crate::types::{
    EosError, EosEvalOutput, I_CHI_RHO, I_CHI_T, I_LN_PGAS,
    NUM_EOS_BASIC_RESULTS,
};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CRAD, KERG, LN10, MP};
use flexmesa_math::exp10;

/// Result of a `(P_gas, T)` equation of state evaluation (`eos_pt_get`).
#[derive(Debug, Clone, PartialEq)]
pub struct EosPtOutput {
    /// Converged mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
    pub rho: f64,
    /// Converged $\log_{10}(\rho / \mathrm{g\,cm^{-3}})$.
    pub log10_rho: f64,
    /// $(\partial \ln \rho / \partial \ln P_{\mathrm{gas}})_T$ via the Implicit Function Theorem.
    pub dln_rho_dln_pgas_const_t: f64,
    /// $(\partial \ln \rho / \partial \ln T)_{P_{\mathrm{gas}}}$ via the Implicit Function Theorem.
    pub dln_rho_dln_t_const_pgas: f64,
    /// Number of `eosDT_get` evaluations used in the root solve.
    pub eos_calls: usize,
    /// Full `EosEvalOutput` at the converged $(\rho, T)$ state.
    pub eos: EosEvalOutput,
}

/// Result of a 1D inverse `eosPT` root search (`eos_pt_get_t` or `eos_pt_get_pgas`).
#[derive(Debug, Clone, PartialEq)]
pub struct EosPtInverseOutput {
    /// Converged root (`log10(T)` for `eos_pt_get_t` or `log10(P_gas)` for `eos_pt_get_pgas`).
    pub log10_result: f64,
    /// Linear value of the converged root (`10^log10_result`).
    pub linear_result: f64,
    /// Number of `eosPT_get` evaluations used in the outer root solve.
    pub eos_calls: usize,
    /// Full `EosPtOutput` at the converged `(P_gas, T)` state.
    pub pt: EosPtOutput,
}

/// Evaluates the equation of state at given `(log10(P_gas), log10(T))` (`eosPT_get`).
///
/// Mirrors `Get_PT_Results_using_DT` in `eos/private/eospt_eval.f90` (lines 145..263).
pub fn eos_pt_get(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    log_pgas: f64,
    log_t: f64,
    cfg: &EosConfig,
) -> Result<EosPtOutput, EosError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|e| {
            EosError::InvalidState {
                context: "eos_pt_get",
                detail: e.to_string(),
            }
        })?;
    let pgas = exp10(log_pgas);
    let t = exp10(log_t);

    let rho_guess = pgas * comp.abar * MP / (KERG * t * (1.0 + comp.zbar));
    let log_rho_guess = rho_guess.log10();
    let target_ln_pgas = log_pgas * LN10;

    let inv = eos_dt_get_rho(
        ht,
        chem_id,
        xa,
        log_t,
        I_LN_PGAS as isize,
        target_ln_pgas,
        log_rho_guess,
        1.0e-8,
        1.0e-8,
        20,
        cfg,
    )?;

    let log_rho = inv.log10_result;
    let rho = inv.linear_result;
    let prad = CRAD * t * t * t * t / 3.0;
    let p = pgas + prad;

    let dp_drho = inv.eos.res[I_CHI_RHO] * p / rho;
    let dpgas_drho = dp_drho;
    let drho_dpgas = 1.0 / dpgas_drho;
    let dln_rho_dln_pgas_const_t = drho_dpgas * pgas / rho;

    let dprad_dt = 4.0 * CRAD * t * t * t / 3.0;
    let dp_dt = inv.eos.res[I_CHI_T] * p / t;
    let dpgas_dt = dp_dt - dprad_dt;
    let drho_dt = -dpgas_dt / dpgas_drho;
    let dln_rho_dln_t_const_pgas = drho_dt * t / rho;

    Ok(EosPtOutput {
        rho,
        log10_rho: log_rho,
        dln_rho_dln_pgas_const_t,
        dln_rho_dln_t_const_pgas,
        eos_calls: inv.eos_calls,
        eos: inv.eos,
    })
}

/// Core 1D bracketed Brent solver for `eosPT_get_T` and `eosPT_get_Pgas`.
///
/// Mirrors `do_safe_get_Pgas_T` in `eos/private/eospt_eval.f90` (lines 392..604).
#[allow(clippy::too_many_arguments)]
fn do_safe_get_pgas_t(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    the_other_log: f64,
    which_other: usize,
    other_value: f64,
    doing_get_t: bool,
    initial_guess: f64,
    xacc: f64,
    yacc: f64,
    ntry: usize,
    cfg: &EosConfig,
) -> Result<EosPtInverseOutput, EosError> {
    if which_other >= NUM_EOS_BASIC_RESULTS {
        return Err(EosError::InvalidState {
            context: "do_safe_get_pgas_t",
            detail: format!(
                "which_other={which_other} >= {NUM_EOS_BASIC_RESULTS}"
            ),
        });
    }
    let mut eos_calls = 0usize;
    let mut last_pt: Option<EosPtOutput> = None;

    let mut get_f_df = |x: f64| -> Result<(f64, f64), EosError> {
        let (log_pgas, log_t) = if doing_get_t {
            (the_other_log, x)
        } else {
            (x, the_other_log)
        };
        let pt = eos_pt_get(ht, chem_id, xa, log_pgas, log_t, cfg)?;
        eos_calls += 1;
        let f = pt.eos.res[which_other] - other_value;
        let dfdx = if doing_get_t {
            (pt.eos.d_dln_t[which_other]
                + pt.dln_rho_dln_t_const_pgas * pt.eos.d_dlnd[which_other])
                * LN10
        } else {
            pt.dln_rho_dln_pgas_const_t * pt.eos.d_dlnd[which_other] * LN10
        };
        last_pt = Some(pt);
        Ok((f, dfdx))
    };

    let (xb1, xb3, y1, y3) =
        look_for_brackets(initial_guess, 0.1, ntry, &mut get_f_df)?;
    let root = brent_safe_zero(
        xb1,
        xb3,
        1.0e-14,
        0.5 * xacc,
        0.5 * yacc,
        y1,
        y3,
        &mut get_f_df,
    )?;

    Ok(EosPtInverseOutput {
        log10_result: root,
        linear_result: exp10(root),
        eos_calls,
        pt: last_pt.expect("evaluated at least once"),
    })
}

/// Solves for $\log_{10} T$ at fixed $\log_{10} P_{\mathrm{gas}}$ and target `other_value` (`eosPT_get_T`).
#[allow(clippy::too_many_arguments)]
pub fn eos_pt_get_t(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    log_pgas: f64,
    which_other: usize,
    other_value: f64,
    log_t_guess: f64,
    log_t_tol: f64,
    other_tol: f64,
    max_iter: usize,
    cfg: &EosConfig,
) -> Result<EosPtInverseOutput, EosError> {
    do_safe_get_pgas_t(
        ht,
        chem_id,
        xa,
        log_pgas,
        which_other,
        other_value,
        true,
        log_t_guess,
        log_t_tol,
        other_tol,
        max_iter,
        cfg,
    )
}

/// Solves for $\log_{10} P_{\mathrm{gas}}$ at fixed $\log_{10} T$ and target `other_value` (`eosPT_get_Pgas`).
#[allow(clippy::too_many_arguments)]
pub fn eos_pt_get_pgas(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    log_t: f64,
    which_other: usize,
    other_value: f64,
    log_pgas_guess: f64,
    log_pgas_tol: f64,
    other_tol: f64,
    max_iter: usize,
    cfg: &EosConfig,
) -> Result<EosPtInverseOutput, EosError> {
    do_safe_get_pgas_t(
        ht,
        chem_id,
        xa,
        log_t,
        which_other,
        other_value,
        false,
        log_pgas_guess,
        log_pgas_tol,
        other_tol,
        max_iter,
        cfg,
    )
}
