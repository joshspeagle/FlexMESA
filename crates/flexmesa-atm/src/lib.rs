// SPDX-License-Identifier: LGPL-3.0-or-later
//! Surface boundary conditions, $T(\tau)$ relations, irradiated atmospheres, and
//! atmosphere table interpolation (`flexmesa-atm`).
//!
//! Ports upstream MESA `atm` (`atm_def.f90`, `atm_lib.f90`, `atm_t_tau_relations.f90`,
//! `atm_utils.f90`, `table_atm.f90`, `atm_table.f90`, `atm_t_tau_uniform.f90`,
//! `atm_t_tau_varying.f90`, and `atm_irradiated.f90`) into Safe Rust in `f64`
//! precision.

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::imprecise_flops,
    clippy::inline_always,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod irradiated;
pub mod t_tau_relations;
pub mod t_tau_uniform;
pub mod t_tau_varying;
pub mod table;
pub mod types;
pub mod utils;

pub use irradiated::eval_irradiated;
pub use t_tau_relations::{eval_t_tau, eval_t_tau_dq_dtau, get_t_tau_base};
pub use t_tau_uniform::{build_t_tau_uniform, eval_t_tau_uniform};
pub use t_tau_varying::{build_t_tau_varying, eval_t_tau_varying};
pub use table::{
    eval_table, get_table_alfa_beta, get_table_base, get_table_values,
    AtmTableInfo, AtmTables,
};
pub use types::{
    get_t_tau_id, get_table_id, AtmEosCallbackResult, AtmError, AtmEvalOutput,
    AtmIrradiatedOutput, AtmKapCallbackResult, AtmStructure, AtmTTauOpacity,
    ATM_CHIRHO, ATM_CHIT, ATM_CP, ATM_CV, ATM_DELTA_R, ATM_DLNKAP_DLND,
    ATM_DLNKAP_DLNT, ATM_GAMMA1, ATM_GRADA, ATM_GRADR, ATM_GRADT, ATM_KAP,
    ATM_LND, ATM_LNFREE_E, ATM_LNP, ATM_LNPGAS, ATM_LNT,
    ATM_TABLE_DB_WD_TAU_25, ATM_TABLE_PHOTOSPHERE, ATM_TABLE_TAU_1,
    ATM_TABLE_TAU_10, ATM_TABLE_TAU_100, ATM_TABLE_TAU_1M1,
    ATM_TABLE_WD_TAU_25, ATM_TAU, ATM_T_TAU_EDDINGTON, ATM_T_TAU_KRISHNA_SWAMY,
    ATM_T_TAU_SOLAR_HOPF, ATM_T_TAU_TRAMPEDACH_SOLAR, ATM_XM,
    NUM_RESULTS_FOR_BUILD_ATM, TABLE_ATM_VERSION,
};
pub use utils::{
    atm_black_body_t, atm_l, atm_teff, eval_e2, eval_paczynski_gradr,
    eval_teff_g, radiation_pressure,
};

/// Dispatches $T(\tau)$ outer boundary condition evaluation across `'fixed'`,
/// `'iterated'`, and `'varying'` opacity modes with `Pextra_factor` in exact
/// upstream order (`get_T_tau` in `star/private/atm_support.f90:218-343`).
///
/// - `'fixed'` (`atm_support.f90:280-291`): calls [`eval_t_tau_uniform`] with
///   `max_iters = 0` and `pextra_factor`, returning `kap = kap_guess`.
/// - `'iterated'` (`atm_support.f90:292-315`): requires `max_iters > 0` and
///   calls [`eval_t_tau_uniform`] with `pextra_factor`, 0.5-underrelaxed fixed-point
///   opacity iteration, and implicit $(\chi_\rho, \chi_T, \kappa_\rho, \kappa_T)$
///   partial derivative correction, returning converged `kap`.
/// - `'varying'` (`atm_support.f90:316-335`): calls [`eval_t_tau_varying`] via
///   inward `dopri5` integration and central finite differencing in $\ln T_{\mathrm{eff}}$,
///   returning `kap = 0.0`.
///
/// # Errors
/// Returns [`AtmError`] on non-positive $(L, R, M)$, unknown `t_tau_relation` or
/// `t_tau_opacity` strings, `max_iters == 0` when `'iterated'` is requested,
/// callback failure, or non-convergence.
pub fn eval_t_tau_atm<E, K>(
    tau_surf: f64,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    kap_guess: f64,
    pextra_factor: f64,
    t_tau_relation: &str,
    t_tau_opacity: &str,
    eos_proc: E,
    kap_proc: K,
    errtol: f64,
    max_iters: usize,
    max_steps: usize,
    skip_partials: bool,
    teff: f64,
) -> Result<AtmEvalOutput, AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
    K: FnMut(
        f64,
        f64,
        &AtmEosCallbackResult,
    ) -> Result<AtmKapCallbackResult, AtmError>,
{
    if l <= 0.0 || r <= 0.0 || m <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "eval_t_tau_atm",
            detail: format!("require L, R, M > 0, got L={l}, R={r}, M={m}"),
        });
    }
    let t_tau_id = get_t_tau_id(t_tau_relation)?;
    let opacity_mode = AtmTTauOpacity::from_name(t_tau_opacity)?;
    match opacity_mode {
        AtmTTauOpacity::Fixed => eval_t_tau_uniform(
            tau_surf,
            l,
            r,
            m,
            cgrav,
            kap_guess,
            pextra_factor,
            t_tau_id,
            eos_proc,
            kap_proc,
            errtol,
            0,
            skip_partials,
            teff,
        ),
        AtmTTauOpacity::Iterated => {
            if max_iters == 0 {
                return Err(AtmError::InvalidArgument {
                    context: "eval_t_tau_atm",
                    detail:
                        "atm_T_tau_opacity = 'iterated' requires max_iters > 0"
                            .to_string(),
                });
            }
            eval_t_tau_uniform(
                tau_surf,
                l,
                r,
                m,
                cgrav,
                kap_guess,
                pextra_factor,
                t_tau_id,
                eos_proc,
                kap_proc,
                errtol,
                max_iters,
                skip_partials,
                teff,
            )
        }
        AtmTTauOpacity::Varying => {
            let mut out = eval_t_tau_varying(
                tau_surf,
                l,
                r,
                m,
                cgrav,
                t_tau_id,
                eos_proc,
                kap_proc,
                errtol,
                max_steps,
                skip_partials,
                teff,
            )?;
            out.kap = 0.0;
            Ok(out)
        }
    }
}

/// Evaluates a tabulated model atmosphere with upstream boundary blending
/// (`get_table` in `star/private/atm_support.f90:430-689`), returning
/// `(AtmEvalOutput, tau_base)`.
///
/// Rejects `tau_factor != 1.0` (`atm_support.f90:491`), propagates table and
/// callback errors without silent fallback, and when `alfa != 0.0` dispatches
/// `atm_off_table_option = 'T_tau'` via [`eval_t_tau_atm`] (failing loudly when
/// `atm_off_table_option` is `""` or unrecognized).
///
/// # Errors
/// Returns [`AtmError`] on `tau_factor != 1.0`, non-positive $(L, R, M)$,
/// unknown `atm_table` or `atm_off_table_option`, off-table evaluation with
/// empty `atm_off_table_option`, or underlying table/`T-tau` failure.
pub fn eval_table_atm<E, K>(
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    atm_table: &str,
    z: f64,
    tau_factor: f64,
    atm_off_table_option: &str,
    t_tau_relation: &str,
    t_tau_opacity: &str,
    kap_guess: f64,
    pextra_factor: f64,
    eos_proc: E,
    kap_proc: K,
    errtol: f64,
    max_iters: usize,
    max_steps: usize,
    skip_partials: bool,
    teff: f64,
) -> Result<(AtmEvalOutput, f64), AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
    K: FnMut(
        f64,
        f64,
        &AtmEosCallbackResult,
    ) -> Result<AtmKapCallbackResult, AtmError>,
{
    if tau_factor != 1.0 {
        return Err(AtmError::InvalidArgument {
            context: "eval_table_atm",
            detail: format!(
                "cannot use atm_option = 'table' with tau_factor != 1 (got {tau_factor})"
            ),
        });
    }
    if l <= 0.0 || r <= 0.0 || m <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "eval_table_atm",
            detail: format!("require L, R, M > 0, got L={l}, R={r}, M={m}"),
        });
    }
    let table_id = get_table_id(atm_table)?;
    let (alfa, beta) = get_table_alfa_beta(l, teff, r, m, cgrav, table_id)?;

    let tau_base = if beta == 0.0 {
        let t_tau_id = get_t_tau_id(t_tau_relation)?;
        get_t_tau_base(t_tau_id)?
    } else {
        get_table_base(table_id)?
    };

    let b = if beta == 0.0 {
        AtmEvalOutput {
            ln_t: 0.0,
            dln_t_dl: 0.0,
            dln_t_dln_r: 0.0,
            dln_t_dln_m: 0.0,
            dln_t_dln_kap: 0.0,
            ln_p: 0.0,
            dln_p_dl: 0.0,
            dln_p_dln_r: 0.0,
            dln_p_dln_m: 0.0,
            dln_p_dln_kap: 0.0,
            kap: 0.0,
        }
    } else {
        eval_table(l, r, m, cgrav, table_id, z, skip_partials, teff)?
    };

    let a = if alfa == 0.0 {
        AtmEvalOutput {
            ln_t: 0.0,
            dln_t_dl: 0.0,
            dln_t_dln_r: 0.0,
            dln_t_dln_m: 0.0,
            dln_t_dln_kap: 0.0,
            ln_p: 0.0,
            dln_p_dl: 0.0,
            dln_p_dln_r: 0.0,
            dln_p_dln_m: 0.0,
            dln_p_dln_kap: 0.0,
            kap: 0.0,
        }
    } else {
        match atm_off_table_option.trim() {
            "T_tau" => eval_t_tau_atm(
                tau_base,
                l,
                r,
                m,
                cgrav,
                kap_guess,
                pextra_factor,
                t_tau_relation,
                t_tau_opacity,
                eos_proc,
                kap_proc,
                errtol,
                max_iters,
                max_steps,
                skip_partials,
                teff,
            )?,
            "" => {
                return Err(AtmError::InvalidOption {
                    context: "eval_table_atm",
                    option: "attempt to interpolate outside atmosphere table with empty atm_off_table_option".to_string(),
                });
            }
            other => {
                return Err(AtmError::InvalidOption {
                    context: "eval_table_atm",
                    option: other.to_string(),
                });
            }
        }
    };

    let ln_t = alfa * a.ln_t + beta * b.ln_t;
    let ln_p = alfa * a.ln_p + beta * b.ln_p;
    let out = if skip_partials {
        AtmEvalOutput {
            ln_t,
            dln_t_dl: 0.0,
            dln_t_dln_r: 0.0,
            dln_t_dln_m: 0.0,
            dln_t_dln_kap: 0.0,
            ln_p,
            dln_p_dl: 0.0,
            dln_p_dln_r: 0.0,
            dln_p_dln_m: 0.0,
            dln_p_dln_kap: 0.0,
            kap: 0.0,
        }
    } else {
        AtmEvalOutput {
            ln_t,
            dln_t_dl: alfa * a.dln_t_dl + beta * b.dln_t_dl,
            dln_t_dln_r: alfa * a.dln_t_dln_r + beta * b.dln_t_dln_r,
            dln_t_dln_m: alfa * a.dln_t_dln_m + beta * b.dln_t_dln_m,
            dln_t_dln_kap: alfa * a.dln_t_dln_kap + beta * b.dln_t_dln_kap,
            ln_p,
            dln_p_dl: alfa * a.dln_p_dl + beta * b.dln_p_dl,
            dln_p_dln_r: alfa * a.dln_p_dln_r + beta * b.dln_p_dln_r,
            dln_p_dln_m: alfa * a.dln_p_dln_m + beta * b.dln_p_dln_m,
            dln_p_dln_kap: alfa * a.dln_p_dln_kap + beta * b.dln_p_dln_kap,
            kap: 0.0,
        }
    };
    Ok((out, tau_base))
}
