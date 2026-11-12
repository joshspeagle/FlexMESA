// SPDX-License-Identifier: LGPL-3.0-or-later
//! Exact upstream MESA residual (`sizequ`) and correction (`sizeB`) norm
//! evaluators and 3-stage `gold2` / `gold` / standard Newton tolerance
//! schedule (`star/private/solver_support.f90:185-555`,
//! `star/private/struct_burn_mix.f90:531-553`, and
//! `star/private/star_solver.f90:198-239, 494-510`).

use crate::controls::{StarControls, VarLayout};
use crate::model::StarModel;

#[derive(Default)]
struct NormAcc {
    sum_abs: f64,
    max_abs: f64,
    count: usize,
}

impl NormAcc {
    #[inline]
    fn push(&mut self, a: f64) {
        self.sum_abs += a;
        self.count += 1;
        if a > self.max_abs || a.is_nan() {
            self.max_abs = a;
        }
    }

    #[inline]
    fn finish(self) -> (f64, f64) {
        if self.count == 0 {
            (0.0, 0.0)
        } else {
            (self.sum_abs / (self.count as f64), self.max_abs)
        }
    }
}

/// Computes `(mean_abs, max_abs)` across all elements of `v` (`sizequ` raw
/// helper).
#[must_use]
pub fn compute_vector_norms(v: &[f64]) -> (f64, f64) {
    let mut acc = NormAcc::default();
    for &x in v {
        acc.push(x.abs());
    }
    acc.finish()
}

/// Computes equation residual norms `(residual_norm, max_residual)` (`sizequ`
/// in `star/private/solver_support.f90:185-308`), honoring
/// `convergence_ignore_equl_residuals` and
/// `convergence_ignore_alpha_rti_residuals`.
#[must_use]
pub fn compute_residual_norms(
    model: &StarModel,
    nvar: usize,
    equ: &[f64],
) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 || nvar == 0 || equ.is_empty() {
        return (0.0, 0.0);
    }
    let c = &model.controls;
    let layout = &model.layout;
    let ignore_rti = c.convergence_ignore_alpha_rti_residuals
        && layout.i_dalpha_rti_dt.is_some();
    if !(c.convergence_ignore_equl_residuals || ignore_rti) {
        return compute_vector_norms(equ);
    }
    let mut acc = NormAcc::default();
    for k in 0..nz {
        let base = k * nvar;
        for j in 0..nvar {
            let val = equ[base + j];
            if val.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            if c.convergence_ignore_equl_residuals && j == layout.i_equ_l {
                continue;
            }
            if ignore_rti && Some(j) == layout.i_dalpha_rti_dt {
                continue;
            }
            acc.push(val.abs());
        }
    }
    acc.finish()
}

/// Computes `(correction_norm, max_correction)` from layout-only metadata,
/// skipping `i_lum`, `i_hp`, and `i_alpha_rti` (`sizeB` baseline helper).
#[must_use]
pub fn compute_correction_norms(
    layout: &VarLayout,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
) -> (f64, f64) {
    if nz == 0 || nvar == 0 || dx_scaled.is_empty() {
        return (0.0, 0.0);
    }
    let mut acc = NormAcc::default();
    for k in 0..nz {
        let base = k * nvar;
        for j in 0..nvar {
            let val = dx_scaled[base + j];
            if val.is_nan() {
                return (f64::NAN, f64::NAN);
            }
            if j == layout.i_lum
                || Some(j) == layout.i_hp
                || Some(j) == layout.i_alpha_rti
            {
                continue;
            }
            acc.push(val.abs());
        }
    }
    acc.finish()
}

/// Computes `(correction_norm, max_correction)` matching `sizeB` in
/// `star/private/solver_support.f90:316-555`:
/// - `i_lum`, `i_v`, `i_u`, `i_w` skipped unless their respective
///   `include_*_in_correction_limits` flags are enabled,
/// - `i_hp` and `i_alpha_rti` always skipped,
/// - trace species with `xa_start[k, j] < correction_xa_limit` (`5e-3` default)
///   excluded from correction norms (`ignore_species_in_max_correction`
///   additionally excludes species from `max_correction`).
#[must_use]
pub fn compute_model_correction_norms(
    model: &StarModel,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
    x_scale: &[f64],
) -> (f64, f64) {
    if nz == 0 || nvar == 0 || dx_scaled.is_empty() {
        return (0.0, 0.0);
    }
    let layout = &model.layout;
    let c = &model.controls;
    let lum_surf_ref =
        0.1 * model.lum_start.first().copied().unwrap_or(0.0).abs();

    let mut acc = NormAcc::default();

    for k in 0..nz {
        let base = k * nvar;
        let mut sum_xa_corr = 0.0;
        let mut num_xa_terms = 0_usize;
        for j in 0..nvar {
            if dx_scaled[base + j].is_nan()
                || x_scale.get(base + j).is_some_and(|s| s.is_nan())
            {
                return (f64::NAN, f64::NAN);
            }
            if Some(j) == layout.i_hp || Some(j) == layout.i_alpha_rti {
                continue;
            }
            if j == layout.i_lum {
                if c.include_l_in_correction_limits {
                    let dx = (dx_scaled[base + j] * x_scale[base + j]).abs();
                    let denom =
                        (lum_surf_ref + model.lum[k].abs()).max(1.0e-30);
                    acc.push(dx / denom);
                }
                continue;
            }
            if Some(j) == layout.i_v || Some(j) == layout.i_u {
                let (active, vel_k) = if Some(j) == layout.i_v {
                    (c.v_flag && c.include_v_in_correction_limits, model.v[k])
                } else {
                    (c.u_flag && c.include_u_in_correction_limits, model.u[k])
                };
                if active {
                    let cs_s = model
                        .aux
                        .csound_start
                        .get(k)
                        .copied()
                        .filter(|&cs| cs > 0.0)
                        .unwrap_or(model.csound[k]);
                    let dx = (dx_scaled[base + j] * x_scale[base + j]).abs();
                    acc.push(dx / (0.1 * cs_s + vel_k.abs()).max(1.0));
                }
                continue;
            }
            if Some(j) == layout.i_w {
                if c.rsp2_flag && c.include_w_in_correction_limits {
                    let w_k = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
                    let a = if w_k < 1.0 {
                        0.0
                    } else {
                        (dx_scaled[base + j] * x_scale[base + j]).abs()
                            / (1.0e3 + w_k)
                    };
                    acc.push(a);
                }
                continue;
            }
            if j >= layout.nvar_hydro {
                if !(c.do_burn || c.do_mix) {
                    continue;
                }
                let j_sp = j - layout.nvar_hydro;
                let xa_s = model
                    .xa_start
                    .get(k * model.species + j_sp)
                    .copied()
                    .unwrap_or(0.0);
                if xa_s < c.correction_xa_limit {
                    continue;
                }
                let a = dx_scaled[base + j].abs();
                sum_xa_corr += a;
                num_xa_terms += 1;
                if !c.ignore_species_in_max_correction
                    && (a > acc.max_abs || a.is_nan())
                {
                    acc.max_abs = a;
                }
                continue;
            }

            acc.push(dx_scaled[base + j].abs());
        }
        if num_xa_terms > 0 {
            acc.sum_abs += sum_xa_corr / (num_xa_terms as f64);
            acc.count += 1;
        }
    }

    acc.finish()
}

pub(crate) fn active_gold_level(model: &StarModel) -> u8 {
    let c = &model.controls;
    let mnum = model.model_number as isize;
    let use_gold2 = (c.use_gold2_tolerances
        && c.steps_before_use_gold2_tolerances < 0)
        || (c.steps_before_use_gold2_tolerances >= 0
            && mnum > c.steps_before_use_gold2_tolerances);
    if use_gold2 {
        return 2;
    }

    let use_gold_step = (c.use_gold_tolerances
        && c.steps_before_use_gold_tolerances < 0)
        || (c.steps_before_use_gold_tolerances >= 0
            && mnum > c.steps_before_use_gold_tolerances);
    if !use_gold_step {
        return 0;
    }
    if c.max_t_for_gold_tolerances <= 0.0 {
        return 1;
    }
    let max_t = model
        .ln_t
        .iter()
        .fold(0.0_f64, |acc, &lnt| acc.max(lnt.exp()));
    u8::from(max_t <= c.max_t_for_gold_tolerances)
}

pub(crate) fn active_solver_tolerances_with_gold(
    c: &StarControls,
    gold_level: u8,
    iter: usize,
) -> (f64, f64, f64, f64) {
    // Honor explicit caller overrides of the legacy scalar `tol_residual_norm` /
    // `tol_max_residual` fields when `tol_residual_norm1` was left untouched.
    let has_custom_scalar_tol = ((c.tol_residual_norm - 1.0e-4).abs()
        > 1.0e-15
        || (c.tol_max_residual - 1.0e-2).abs() > 1.0e-15)
        && (c.tol_residual_norm1 - 1.0e-10).abs() < 1.0e-20;
    if has_custom_scalar_tol {
        return (
            c.tol_residual_norm,
            c.tol_max_residual,
            c.tol_correction_norm,
            c.tol_max_correction,
        );
    }

    let (rn, mr) = match gold_level {
        2 => {
            if iter < c.gold2_iter_for_resid_tol2 {
                (c.gold2_tol_residual_norm1, c.gold2_tol_max_residual1)
            } else if iter < c.gold2_iter_for_resid_tol3 {
                (c.gold2_tol_residual_norm2, c.gold2_tol_max_residual2)
            } else {
                (c.gold2_tol_residual_norm3, c.gold2_tol_max_residual3)
            }
        }
        1 => {
            if iter < c.gold_iter_for_resid_tol2 {
                (c.gold_tol_residual_norm1, c.gold_tol_max_residual1)
            } else if iter < c.gold_iter_for_resid_tol3 {
                (c.gold_tol_residual_norm2, c.gold_tol_max_residual2)
            } else {
                (c.gold_tol_residual_norm3, c.gold_tol_max_residual3)
            }
        }
        _ => {
            if iter < c.iter_for_resid_tol2 {
                (c.tol_residual_norm1, c.tol_max_residual1)
            } else if iter < c.iter_for_resid_tol3 {
                (c.tol_residual_norm2, c.tol_max_residual2)
            } else {
                (c.tol_residual_norm3, c.tol_max_residual3)
            }
        }
    };
    (rn, mr, c.tol_correction_norm, c.tol_max_correction)
}

/// Returns `(tol_residual_norm, tol_max_residual, tol_correction_norm, tol_max_correction)`
/// for 1-based Newton iteration `iter` (`struct_burn_mix.f90:531-553` and
/// `star_solver.f90:198-239`).
#[must_use]
pub fn active_solver_tolerances(
    model: &StarModel,
    iter: usize,
) -> (f64, f64, f64, f64) {
    active_solver_tolerances_with_gold(
        &model.controls,
        active_gold_level(model),
        iter,
    )
}
