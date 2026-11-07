// SPDX-License-Identifier: LGPL-3.0-or-later
//! Inverse equation of state solvers (`eosDT_get_Rho`, `eosDT_get_T`, `eosPT_get`,
//! and `eos_gamma_DE_get`).
//!
//! Ports the inverse root-finding routines from upstream MESA
//! `eos/private/eosdt_eval.f90` (`get_Rho`, `get_T`, `do_safe_get_Rho_T`) and
//! `eos/private/eospt_eval.f90` (`Get_eosPT_Results`, `Get_PT_Results_using_DT`).

use crate::blend_dt::{
    eos_dt_get_from_weights, extract_species_weights, EosConfig,
};
pub use crate::gamma_de::{eos_gamma_de_get, EosGammaDeOutput};
use crate::helm::HelmTable;
use crate::types::{
    EosError, EosEvalOutput, I_CHI_RHO, I_CHI_T, I_LN_E, I_LN_PGAS,
};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CRAD, KERG, LN10, MP};
use flexmesa_math::{exp10, pow4};

/// Result of a 1D inverse `eosDT` root search (`eos_dt_get_rho` or `eos_dt_get_t`).
#[derive(Debug, Clone, PartialEq)]
pub struct EosInverseOutput {
    /// Converged root (`log10(rho)` for `eos_dt_get_rho` or `log10(T)` for `eos_dt_get_t`).
    pub log10_result: f64,
    /// Linear value of the converged root (`10^log10_result`).
    pub linear_result: f64,
    /// Number of `eosDT_get` evaluations required to converge.
    pub eos_calls: usize,
    /// Full `EosEvalOutput` at the converged state.
    pub eos: EosEvalOutput,
}

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

/// Evaluates the residual $f(x)$ and analytical derivative $df/dx$ for `do_safe_get_Rho_T`.
///
/// `which_other` follows upstream MESA conventions (`-1` for $E_{\mathrm{gas}}$,
/// `-2` for $\log_{10} P_{\mathrm{total}}$, or `0..NUM_EOS_BASIC_RESULTS`).
fn eval_other_residual(
    which_other: isize,
    other_value: f64,
    doing_rho: bool,
    rho: f64,
    t: f64,
    out: &EosEvalOutput,
) -> (f64, f64) {
    let pgas = out.res[I_LN_PGAS].exp();
    let prad = CRAD * pow4(t) / 3.0;
    let energy = out.res[I_LN_E].exp();

    if which_other == -1 {
        let erad = CRAD * pow4(t) / rho;
        let egas = energy - erad;
        let f = egas - other_value;
        let dfdx = if doing_rho {
            energy * out.d_dlnd[I_LN_E] * LN10 + erad
        } else {
            (energy * out.d_dln_t[I_LN_E] - 4.0 * erad) * LN10
        };
        (f, dfdx)
    } else if which_other == -2 {
        let p_tot = pgas + prad;
        let f = p_tot.log10() - other_value;
        let dfdx = if doing_rho {
            pgas * out.d_dlnd[I_LN_PGAS] / p_tot
        } else {
            (pgas * out.d_dln_t[I_LN_PGAS] + 4.0 * prad) / p_tot
        };
        (f, dfdx)
    } else {
        let idx = which_other as usize;
        let f = out.res[idx] - other_value;
        let dfdx = if doing_rho {
            out.d_dlnd[idx] * LN10
        } else {
            out.d_dln_t[idx] * LN10
        };
        (f, dfdx)
    }
}

/// Core 1D bracketed Newton-secant solver matching `do_safe_get_Rho_T` in `eosdt_eval.f90`.
#[allow(clippy::too_many_arguments)]
fn do_safe_get_rho_t(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    the_other_log: f64,
    which_other: isize,
    other_value: f64,
    doing_rho: bool,
    initial_guess: f64,
    xacc: f64,
    yacc: f64,
    ntry: usize,
    cfg: &EosConfig,
) -> Result<EosInverseOutput, EosError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|e| {
            EosError::InvalidState {
                context: "do_safe_get_rho_t",
                detail: e.to_string(),
            }
        })?;
    let (z_int, azion, acmi, a_baryon) =
        extract_species_weights(catalog, chem_id, "do_safe_get_rho_t")?;

    let the_other_val = exp10(the_other_log);
    let mut eos_calls = 0usize;

    let mut eval_at = |x: f64| -> Result<(f64, f64, EosEvalOutput), EosError> {
        eos_calls += 1;
        let (rho, log_rho, t, log_t) = if doing_rho {
            (exp10(x), x, the_other_val, the_other_log)
        } else {
            (the_other_val, the_other_log, exp10(x), x)
        };
        let out = eos_dt_get_from_weights(
            ht, rho, log_rho, t, log_t, comp.abar, comp.zbar, xa, &z_int,
            &azion, &acmi, &a_baryon, cfg,
        )?;
        let (f, dfdx) = eval_other_residual(
            which_other,
            other_value,
            doing_rho,
            rho,
            t,
            &out,
        );
        Ok((f, dfdx, out))
    };

    let mut x = initial_guess;
    let (mut f, mut dfdx, mut last_out) = eval_at(x)?;
    if f.abs() <= yacc {
        return Ok(EosInverseOutput {
            log10_result: x,
            linear_result: exp10(x),
            eos_calls,
            eos: last_out,
        });
    }

    let newt_imax = ntry.min(6);
    let mut bracket: Option<(f64, f64, f64, f64)> = None;
    let mut prev_x = x;
    let mut prev_f = f;

    for _ in 0..newt_imax {
        if dfdx.abs() <= 1.0e-30 || !dfdx.is_finite() {
            break;
        }
        let step = (-f / dfdx).clamp(-1.0, 1.0);
        let x_next = x + step;
        let Ok((f_next, dfdx_next, out_next)) = eval_at(x_next) else {
            break;
        };
        last_out = out_next;

        if prev_f * f_next <= 0.0 {
            bracket = Some(if prev_x < x_next {
                (prev_x, prev_f, x_next, f_next)
            } else {
                (x_next, f_next, prev_x, prev_f)
            });
        }
        prev_x = x_next;
        prev_f = f_next;
        x = x_next;
        f = f_next;
        dfdx = dfdx_next;

        if f.abs() <= yacc || step.abs() <= xacc {
            return Ok(EosInverseOutput {
                log10_result: x,
                linear_result: exp10(x),
                eos_calls,
                eos: last_out,
            });
        }
    }

    if bracket.is_none() {
        let mut dx = 0.1;
        let (mut x1, mut y1, mut x3, mut y3) = (x, f, x, f);
        for _ in 0..ntry.max(20) {
            if (dfdx > 0.0 && y1 > 0.0) || (dfdx < 0.0 && y1 < 0.0) {
                x1 -= dx;
                if let Ok((f1, _, out1)) = eval_at(x1) {
                    y1 = f1;
                    last_out = out1;
                }
            } else {
                x3 += dx;
                if let Ok((f3, _, out3)) = eval_at(x3) {
                    y3 = f3;
                    last_out = out3;
                }
            }
            if y1 * y3 <= 0.0 {
                bracket = Some((x1, y1, x3, y3));
                break;
            }
            dx *= 2.0;
        }
    }

    let Some((mut x_lo, mut f_lo, mut x_hi, _f_hi)) = bracket else {
        return Err(EosError::ConvergenceFailure {
            context: "do_safe_get_rho_t",
            detail: format!("failed to bracket root within {ntry} iterations"),
        });
    };

    x = 0.5 * (x_lo + x_hi);
    for _ in 0..ntry.max(20) {
        let (fx, dfx, out_x) = eval_at(x)?;
        last_out = out_x;
        if fx.abs() <= yacc || (x_hi - x_lo).abs() <= xacc {
            return Ok(EosInverseOutput {
                log10_result: x,
                linear_result: exp10(x),
                eos_calls,
                eos: last_out,
            });
        }
        if f_lo * fx <= 0.0 {
            x_hi = x;
        } else {
            x_lo = x;
            f_lo = fx;
        }
        let x_newt = if dfx.abs() > 1.0e-30 {
            x - fx / dfx
        } else {
            0.5 * (x_lo + x_hi)
        };
        x = if x_newt > x_lo && x_newt < x_hi {
            x_newt
        } else {
            0.5 * (x_lo + x_hi)
        };
    }

    if last_out.res[I_LN_PGAS].is_finite() && (x_hi - x_lo).abs() <= 10.0 * xacc
    {
        Ok(EosInverseOutput {
            log10_result: x,
            linear_result: exp10(x),
            eos_calls,
            eos: last_out,
        })
    } else {
        Err(EosError::ConvergenceFailure {
            context: "do_safe_get_rho_t",
            detail: format!("failed to converge within {ntry} iterations"),
        })
    }
}

/// Solves for $\log_{10} \rho$ at fixed $\log_{10} T$ and target `other_value` (`eosDT_get_Rho`).
#[allow(clippy::too_many_arguments)]
pub fn eos_dt_get_rho(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    log_t: f64,
    which_other: isize,
    other_value: f64,
    log_rho_guess: f64,
    log_rho_tol: f64,
    other_tol: f64,
    max_iter: usize,
    cfg: &EosConfig,
) -> Result<EosInverseOutput, EosError> {
    do_safe_get_rho_t(
        ht,
        chem_id,
        xa,
        log_t,
        which_other,
        other_value,
        true,
        log_rho_guess,
        log_rho_tol,
        other_tol,
        max_iter,
        cfg,
    )
}

/// Solves for $\log_{10} T$ at fixed $\log_{10} \rho$ and target `other_value` (`eosDT_get_T`).
#[allow(clippy::too_many_arguments)]
pub fn eos_dt_get_t(
    ht: &HelmTable,
    chem_id: &[usize],
    xa: &[f64],
    log_rho: f64,
    which_other: isize,
    other_value: f64,
    log_t_guess: f64,
    log_t_tol: f64,
    other_tol: f64,
    max_iter: usize,
    cfg: &EosConfig,
) -> Result<EosInverseOutput, EosError> {
    do_safe_get_rho_t(
        ht,
        chem_id,
        xa,
        log_rho,
        which_other,
        other_value,
        false,
        log_t_guess,
        log_t_tol,
        other_tol,
        max_iter,
        cfg,
    )
}

/// Evaluates the equation of state at given `(log10(P_gas), log10(T))` (`eosPT_get`).
///
/// Mirrors `Get_PT_Results_using_DT` in `eos/private/eospt_eval.f90`.
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
