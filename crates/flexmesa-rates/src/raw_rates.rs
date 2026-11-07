// SPDX-License-Identifier: LGPL-3.0-or-later
//! Raw unscreened nuclear reaction rate evaluation (`set_raw_rate`,
//! `set_raw_rates`, `eval_reaclib_handle`), `interp_pm` rate-table evaluation,
//! and Fushiki & Lamb (1987) pycnonuclear triple-alpha (`pycno.f90`).

use flexmesa_const::ONE_THIRD;
use flexmesa_interp1d::{interp_value_and_slope, CubicCoeffs};

use crate::loader::RatesCatalog;
use crate::ratelib::eval_analytic_rate;
use crate::tfactors::{eval_reaclib_terms, TFactors};
use crate::types::{RatesError, ReactionInfo, DEFAULT_REACLIB_MIN_T9};

/// Evaluates the raw unscreened nuclear reaction rate
/// `set_raw_rate(ir, temp, tf)` for a 1-based reaction ID `ir` (`1..=316`)
/// matching `rates/private/raw_rates.f90`.
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if `ir` is out of bounds.
pub fn set_raw_rate(
    ir: usize,
    temp: f64,
    tf: &TFactors,
) -> Result<f64, RatesError> {
    let cat = RatesCatalog::builtin();
    let info = cat.reaction_by_id(ir)?;
    Ok(eval_reaction_raw(cat, info, temp, tf))
}

/// Evaluates `set_raw_rate` across a slice of 1-based reaction IDs `irs`.
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if any reaction ID is out of bounds.
pub fn set_raw_rates(
    irs: &[usize],
    temp: f64,
    tf: &TFactors,
) -> Result<Vec<f64>, RatesError> {
    let cat = RatesCatalog::builtin();
    let mut out = Vec::with_capacity(irs.len());
    for &ir in irs {
        let info = cat.reaction_by_id(ir)?;
        out.push(eval_reaction_raw(cat, info, temp, tf));
    }
    Ok(out)
}

fn eval_reaclib_with_cat(
    cat: &RatesCatalog,
    handle: &str,
    t9: f64,
    check_e_flag: bool,
) -> Result<(f64, f64, f64, f64), RatesError> {
    let (is_rev, terms) = cat.reaclib_terms(handle)?;
    let (lam, dlam, rlam, drlam) =
        eval_reaclib_terms(terms, t9, DEFAULT_REACLIB_MIN_T9, check_e_flag);
    if is_rev {
        Ok((rlam, drlam, lam, dlam))
    } else {
        Ok((lam, dlam, rlam, drlam))
    }
}

/// Evaluates a REACLIB reaction handle
/// `(forward_rate, dforward_dlnT, reverse_rate, dreverse_dlnT)` at temperature
/// $T_9 = T / 10^9\,\mathrm{K}$.
///
/// # Errors
/// Returns [`RatesError::UnknownReaction`] if `handle` is not in the catalog.
pub fn eval_reaclib_handle(
    handle: &str,
    t9: f64,
    check_e_flag: bool,
) -> Result<(f64, f64, f64, f64), RatesError> {
    eval_reaclib_with_cat(RatesCatalog::builtin(), handle, t9, check_e_flag)
}

/// Evaluates `set_raw_rate` for a [`ReactionInfo`] entry given [`RatesCatalog`]
/// and [`TFactors`] at temperature `temp` ($\mathrm{K}$).
#[must_use]
pub fn eval_reaction_raw(
    cat: &RatesCatalog,
    info: &ReactionInfo,
    temp: f64,
    tf: &TFactors,
) -> f64 {
    match info.op_code {
        0 => eval_reaclib_with_cat(cat, &info.target_handle, tf.t9, false)
            .map_or(0.0, |(f, _, _, _)| f),
        1 => cat.eval_rate_table(info.id, temp).map_or(0.0, |(v, _)| v),
        2 => -1.0,
        3..=28 => {
            let (fr, rr) = eval_analytic_rate(info.op_code, tf);
            if info.is_reverse {
                rr
            } else {
                fr
            }
        }
        29 => eval_reaclib_with_cat(cat, &info.target_handle, tf.t9, true)
            .map_or(0.0, |(f, _, r, _)| if info.is_reverse { r } else { f }),
        30 => {
            let fr1 = eval_reaclib_with_cat(cat, "r_fe54_ag_ni58", tf.t9, true)
                .map_or(0.0, |(f, _, _, _)| f);
            let fr2 = eval_reaclib_with_cat(cat, "r_ni57_na_fe54", tf.t9, true)
                .map_or(0.0, |(_, _, r, _)| r);
            let fr3 = eval_reaclib_with_cat(cat, "r_fe54_ap_co57", tf.t9, true)
                .map_or(0.0, |(f, _, _, _)| f);
            fr1 + fr2 + fr3
        }
        31 => {
            let (f1, _, r1, _) = eval_reaclib_with_cat(
                cat,
                "r_o16_o16_to_he4_si28",
                tf.t9,
                true,
            )
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
            let (f2, _, r2, _) =
                eval_reaclib_with_cat(cat, "r_o16_o16_to_h1_p31", tf.t9, true)
                    .unwrap_or((0.0, 0.0, 0.0, 0.0));
            if info.is_reverse {
                r1 + r2
            } else {
                f1 + f2
            }
        }
        _ => 0.0,
    }
}

/// Evaluates an `interp_pm` 1D cubic spline rate table `(t8s, f1)` at
/// temperature `temp` ($\mathrm{K}$), returning `(rate, drate_dt)`.
#[must_use]
pub fn eval_interp_rate_table(
    t8s: &[f64],
    f1: &[CubicCoeffs],
    temp: f64,
) -> (f64, f64) {
    if t8s.is_empty() {
        return (0.0, 0.0);
    }
    let t8 = (temp * 1.0e-8).clamp(t8s[0], t8s[t8s.len() - 1]);
    if let Ok((val, slope)) = interp_value_and_slope(t8s, f1, t8) {
        (val, slope * 1.0e-8)
    } else {
        (0.0, 0.0)
    }
}

/// Evaluates the Fushiki & Lamb (1987, ApJ 317, 368) pycnonuclear + thermally
/// enhanced triple-alpha energy generation rate $\epsilon_{3\alpha}$
/// (`erg g^-1 s^-1`) matching `rates/private/pycno.f90:fl_epsnuc_3alf`.
///
/// # Arguments
/// - `t`: Temperature $T$ ($\mathrm{K}$).
/// - `rho`: Mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
/// - `y`: Helium-4 mass fraction $Y$ (`X(He4)`).
/// - `ue`: Mean nucleon-to-electron ratio $\mu_e \approx 1 / Y_e$ (typically `2.0`).
#[must_use]
pub fn eval_fl_epsnuc_3alf(t: f64, rho: f64, y: f64, ue: f64) -> f64 {
    let r6 = 1.0e-6 * rho;
    let r6t = 2.0 * r6 / ue;
    let t6 = 1.0e-6 * t;
    let r6t13 = r6t.powf(ONE_THIRD);
    let t623 = t6.powf(2.0 / 3.0);
    let u = 1.35 * r6t13 / t623;
    let t612 = t6.sqrt();
    let t632 = t6.powf(1.5);

    let (g1, g2) = if u < 1.0 {
        let t613 = t6.powf(ONE_THIRD);
        let t653 = t6.powf(5.0 / 3.0);
        let a1 = (1.0 - 4.222e-2 * t623).powi(2) + 2.643e-5 * t653;
        let mut b1 = 16.16 / (a1 * t623) * (-134.92 / t613).exp();
        if r6t < 5.458e3 {
            b1 += (-1065.1 / t6).exp() / t632;
        }
        let g1 = b1 * (60.492 * r6t13 / t6).exp();
        let a2 = (1.0 - 2.807e-2 * t623).powi(2) + 2.704e-6 * t653;
        let mut b2 = 244.6 / (a2 * t623)
            * (1.0 + 3.528e-3 * t623).powi(5)
            * (-235.72 / t613).exp();
        if r6t < 1.836e4 {
            b2 += (-3336.4 / t6).exp() / t632;
        }
        let g2 = b2 * (106.35 * r6t13 / t6).exp();
        (g1, g2)
    } else {
        let r6t16 = r6t.powf(1.0 / 6.0);
        let af = u.powf(-1.5) + 1.0;
        let b1_num = 1.178 * af * (-77.554 / r6t16).exp();
        let b1_den =
            t612 * ((1.0 - 5.680e-2 * r6t13).powi(2) + 8.815e-7 * t6 * t6);
        let b1 = b1_num / b1_den;
        let g1 = if r6t < 5.458e3 {
            b1 + ((60.492 * r6t13 - 1065.1) / t6).exp() / t632
        } else {
            b1
        };
        let b2_num = 13.48
            * af
            * (1.0 + 5.070e-3 * r6t13).powi(5)
            * (-135.08 / r6t16).exp();
        let b2_den =
            t612 * ((1.0 - 3.791e-2 * r6t13).powi(2) + 5.162e-8 * t6 * t6);
        let b2 = b2_num / b2_den;
        let g2 = if r6t < 1.836e4 {
            b2 + ((106.35 * r6t13 - 3336.4) / t6).exp() / t632
        } else {
            b2
        };
        (g1, g2)
    };

    5.120e29 * g1 * g2 * y.powi(3) * r6 * r6
}
