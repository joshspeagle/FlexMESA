// SPDX-License-Identifier: LGPL-3.0-or-later
//! Inverse `eosDT` root-finding solvers (`eos_dt_get_rho`, `eos_dt_get_t`).
//!
//! Ports `get_Rho`, `get_T`, and `do_safe_get_Rho_T` from upstream MESA
//! `eos/private/eosdt_eval.f90` (lines 2348..2535), together with
//! `do_safe_root_with_guess`, `do_look_for_brackets` (`num/private/mod_root.f90`),
//! and `eval_brent_safe_zero` (`num/private/mod_brent.f90`).

use crate::blend_dt::{
    eos_dt_get_from_weights_with_tables, extract_species_weights, EosConfig,
};
use crate::cms::CmsTableSet;
use crate::helm::HelmTable;
use crate::tables_4d::EosDtTableSet;
use crate::types::{
    EosError, EosEvalOutput, I_CV, I_LN_E, I_LN_PGAS, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CRAD, LN10};
use flexmesa_math::{exp10, pow4};
use flexmesa_utils::is_bad;

/// Special `which_other` code for specific gas internal energy $E_{\mathrm{gas}} = E - E_{\mathrm{rad}}$ (`-1`).
pub const WHICH_OTHER_EGAS: isize = -1;
/// Special `which_other` code for $\log_{10} P_{\mathrm{total}}$ (`-2`).
pub const WHICH_OTHER_LOG_PTOT: isize = -2;

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
    let prad = CRAD * t * t * t * t / 3.0;
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
            pgas * out.d_dlnd[I_LN_PGAS] / p_tot * LN10
        } else {
            (pgas * out.d_dln_t[I_LN_PGAS] + 4.0 * prad) / p_tot * LN10
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

/// Expands `[x1, x3]` around `x_guess` with step `dx * 2^i` until `y1 * y3 <= 0`.
///
/// Mirrors `do_look_for_brackets` in `num/private/mod_root.f90` (lines 168..256).
pub(crate) fn look_for_brackets<F>(
    x_guess: f64,
    dx: f64,
    imax: usize,
    mut eval_f: F,
) -> Result<(f64, f64, f64, f64), EosError>
where
    F: FnMut(f64) -> Result<(f64, f64), EosError>,
{
    let mut x1 = x_guess;
    let mut x3 = x_guess;
    let mut y1 = 0.0;
    let mut y3 = 0.0;

    for i in 1..=imax {
        let jump = ((1u64 << i.min(50)) as f64) * dx;
        let (move_x1, move_x3) = if i == 1 {
            x1 -= jump;
            x3 += jump;
            (true, true)
        } else if y1 > 0.0 {
            if y1 < y3 {
                x1 -= jump;
                (true, false)
            } else {
                x3 += jump;
                (false, true)
            }
        } else if y1 > y3 {
            x1 -= jump;
            (true, false)
        } else {
            x3 += jump;
            (false, true)
        };

        if move_x1 {
            let (f1, _) = eval_f(x1)?;
            y1 = f1;
        }
        if move_x3 {
            let (f3, _) = eval_f(x3)?;
            y3 = f3;
        }
        if y1 * y3 <= 0.0 {
            return Ok((x1, x3, y1, y3));
        }
    }

    Err(EosError::ConvergenceFailure {
        context: "look_for_brackets",
        detail: format!("failed to bracket root within {imax} steps"),
    })
}

/// Brent's root-finding algorithm on a sign-change bracket `[a, b]`.
///
/// Mirrors `eval_brent_safe_zero` in `num/private/mod_brent.f90` (lines 513..705).
pub(crate) fn brent_safe_zero<F>(
    a: f64,
    b: f64,
    machep: f64,
    t: f64,
    epsy: f64,
    fa_in: f64,
    fb_in: f64,
    mut eval_f: F,
) -> Result<f64, EosError>
where
    F: FnMut(f64) -> Result<(f64, f64), EosError>,
{
    let mut sa = a;
    let mut sb = b;
    let mut fa = fa_in;
    let mut fb = fb_in;
    let mut c = sa;
    let mut fc = fa;
    let mut e = sb - sa;
    let mut d = e;

    loop {
        if fc.abs() < fb.abs() {
            sa = sb;
            sb = c;
            c = sa;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol = 2.0 * machep * sb.abs() + t;
        let m = 0.5 * (c - sb);

        if m.abs() <= tol || fb == 0.0 {
            break;
        }

        if e.abs() < tol || fa.abs() <= fb.abs() {
            e = m;
            d = e;
        } else {
            let s = fb / fa;
            let (mut p, mut q) = if sa == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q0 * (q0 - r) - (sb - sa) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };

            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }

            let s_prev = e;
            e = d;

            if 2.0 * p < 3.0 * m * q - (tol * q).abs()
                && p < (0.5 * s_prev * q).abs()
            {
                d = p / q;
            } else {
                e = m;
                d = e;
            }
        }

        sa = sb;
        fa = fb;

        if d.abs() > tol {
            sb += d;
        } else if m > 0.0 {
            sb += tol;
        } else {
            sb -= tol;
        }

        let (fb_new, _) = eval_f(sb)?;
        fb = fb_new;
        if fb.abs() < epsy {
            break;
        }

        if (fb > 0.0 && fc > 0.0) || (fb <= 0.0 && fc <= 0.0) {
            c = sa;
            fc = fa;
            e = sb - sa;
            d = e;
        }
    }

    Ok(sb)
}

/// Core 1D Newton + Brent solver matching `do_safe_get_Rho_T` in `eosdt_eval.f90`.
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
    if which_other < WHICH_OTHER_LOG_PTOT
        || which_other >= NUM_EOS_BASIC_RESULTS as isize
    {
        return Err(EosError::InvalidState {
            context: "do_safe_get_rho_t",
            detail: format!("which_other={which_other} out of range (-2..26)"),
        });
    }
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
    let dt_tables = EosDtTableSet::builtin();
    let cms_tables = CmsTableSet::builtin();

    let the_other_val = exp10(the_other_log);
    let mut eos_calls = 0usize;
    let mut last_out: Option<EosEvalOutput> = None;

    let mut eval_f = |x: f64| -> Result<(f64, f64), EosError> {
        eos_calls += 1;
        if x > 50.0 {
            return Err(EosError::InvalidState {
                context: "do_safe_get_rho_t",
                detail: format!("x={x} > 50"),
            });
        }
        let (rho, log_rho, t, log_t) = if doing_rho {
            (exp10(x), x, the_other_val, the_other_log)
        } else {
            (the_other_val, the_other_log, exp10(x), x)
        };
        let out = eos_dt_get_from_weights_with_tables(
            ht, dt_tables, cms_tables, comp.xh, comp.xz, rho, log_rho, t,
            log_t, comp.abar, comp.zbar, xa, &z_int, &azion, &acmi, &a_baryon,
            cfg,
        )?;
        if is_bad(out.res[I_CV]) {
            return Err(EosError::InvalidState {
                context: "do_safe_get_rho_t",
                detail: "bad Cv".to_string(),
            });
        }
        let (f, dfdx) = eval_other_residual(
            which_other,
            other_value,
            doing_rho,
            rho,
            t,
            &out,
        );
        last_out = Some(out);
        Ok((f, dfdx))
    };

    let newt_imax = ntry.min(6);
    let mut x = initial_guess;
    let mut absy_prev = 0.0;
    let mut x1 = 0.0;
    let mut y1 = 0.0;
    let mut have_x1 = false;
    let mut x3 = 0.0;
    let mut y3 = 0.0;
    let mut have_x3 = false;

    for i in 1..=newt_imax {
        let Ok((y, dydx)) = eval_f(x) else {
            break;
        };
        let absy = y.abs();
        if absy < yacc.max(dydx.abs() * xacc) {
            return Ok(EosInverseOutput {
                log10_result: x,
                linear_result: exp10(x),
                eos_calls,
                eos: last_out.expect("evaluated at least once"),
            });
        }
        if dydx == 0.0 || (i > 1 && absy > absy_prev) {
            break;
        }
        absy_prev = absy;
        if y < 0.0 {
            x1 = x;
            y1 = y;
            have_x1 = true;
        } else {
            x3 = x;
            y3 = y;
            have_x3 = true;
        }
        x -= y / dydx;
    }

    if !(have_x1 && have_x3) {
        let (bx1, bx3, by1, by3) =
            look_for_brackets(initial_guess, 0.1, ntry, &mut eval_f)?;
        x1 = bx1;
        x3 = bx3;
        y1 = by1;
        y3 = by3;
    }

    let root =
        brent_safe_zero(x1, x3, 1.0e-14, xacc, yacc, y1, y3, &mut eval_f)?;
    Ok(EosInverseOutput {
        log10_result: root,
        linear_result: exp10(root),
        eos_calls,
        eos: last_out.expect("evaluated at least once"),
    })
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
