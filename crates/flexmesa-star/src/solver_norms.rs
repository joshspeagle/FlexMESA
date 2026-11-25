// SPDX-License-Identifier: GPL-3.0-or-later
//! Residual norms (`sizequ`), per-variable filtered correction norms (`sizeB`),
//! 3-iteration flailing detection, high-$T$ correction tolerance relaxation
//! (`set_tol_correction`), `max_tries` retry schedule, and 3-stage `gold` /
//! `gold2` solver tolerance schedules (`star/private/solver_support.f90` and
//! `star/private/struct_burn_mix.f90`).

use crate::model::StarModel;

/// Three-iteration monotonic correction-growth tracker (`sizeB` in
/// `star/private/solver_support.f90:539-567`).
#[derive(Debug, Clone, Default)]
pub struct FlailingTracker {
    abs_max_corr1: f64,
    abs_max_corr2: f64,
    max_var1: Option<usize>,
    max_var2: Option<usize>,
    max_zone1: Option<usize>,
    max_zone2: Option<usize>,
}

impl FlailingTracker {
    /// Resets the tracker before iteration 1 of a Newton solve.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Applies the upstream `sizeB` history shift and diverging-solve check
    /// (`solver_support.f90:537-567`).
    ///
    /// Upstream overwrites `abs_max_corr1 = abs_corr` (`:539`) BEFORE testing
    /// `abs_corr > abs_max_corr1` (`:560`), so the flailing branch is always
    /// false in MESA `26.4.1` (reproduced unconditionally per `#260` / `#169`).
    pub fn observe(
        &mut self,
        solver_iter: usize,
        tol_max_correction: f64,
        raw_max_corr: f64,
        arg_max: Option<(usize, usize)>,
    ) -> f64 {
        if solver_iter <= 1 {
            self.reset();
        }
        let (zone_opt, var_opt) = match arg_max {
            Some((z, v)) => (Some(z), Some(v)),
            None => (None, None),
        };
        // `solver_support.f90:539-541` (`#260`): shift history BEFORE comparing.
        self.abs_max_corr2 = self.abs_max_corr1;
        self.abs_max_corr1 = raw_max_corr;
        self.max_zone2 = self.max_zone1;
        self.max_zone1 = zone_opt;
        self.max_var2 = self.max_var1;
        self.max_var1 = var_opt;
        if solver_iter < 3 {
            return raw_max_corr;
        }
        // `solver_support.f90:558-567` (`#260`): `raw_max_corr > self.abs_max_corr1`
        // is always false because `self.abs_max_corr1 == raw_max_corr`.
        if let Some((zone, var)) = arg_max {
            if raw_max_corr > tol_max_correction
                && raw_max_corr > self.abs_max_corr1
                && self.abs_max_corr1 > self.abs_max_corr2
                && Some(zone) == self.max_zone1
                && self.max_zone1 == self.max_zone2
                && Some(var) == self.max_var1
                && self.max_var1 == self.max_var2
            {
                return 1.0e99;
            }
        }
        raw_max_corr
    }
}

/// Computes `(mean_abs_norm, max_abs_norm, arg_max_idx)` across slice `v`,
/// propagating `NaN`/`Inf` immediately so non-finite entries are never masked.
#[must_use]
pub fn compute_vector_norms_with_argmax(
    v: &[f64],
) -> (f64, f64, Option<usize>) {
    if v.is_empty() {
        return (0.0, 0.0, None);
    }
    let mut sum_abs = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut max_idx = Some(0_usize);
    for (idx, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return (f64::NAN, f64::NAN, Some(idx));
        }
        let a = x.abs();
        sum_abs += a;
        if a > max_abs {
            max_abs = a;
            max_idx = Some(idx);
        }
    }
    (sum_abs / (v.len() as f64), max_abs, max_idx)
}

/// Computes `(mean_abs_norm, max_abs_norm)` across slice `v`, propagating
/// `NaN`/`Inf` immediately (`sizequ` / `sizeB` in `solver_support.f90:185-450`).
#[must_use]
pub fn compute_vector_norms(v: &[f64]) -> (f64, f64) {
    let (mean_abs, max_abs, _) = compute_vector_norms_with_argmax(v);
    (mean_abs, max_abs)
}

/// Computes `(residual_norm, max_residual, arg_max_zone_equ)` (`sizequ` in
/// `star/private/solver_support.f90:185-294`).
///
/// Honors `convergence_ignore_equL_residuals` and
/// `convergence_ignore_alpha_RTI_residuals` while checking every entry of `equ`
/// for non-finite values first (`solver_support.f90:224-232`). Reproduces
/// MESA `26.4.1`'s chemistry residual double-count when `do_burn || do_mix` is
/// active (`#177`, `solver_support.f90:219-236, 273-286`).
#[must_use]
pub fn compute_residual_norms_with_argmax(
    model: &StarModel,
    nvar: usize,
    equ: &[f64],
) -> (f64, f64, Option<(usize, usize)>) {
    let nz = model.nz;
    if nz == 0 || nvar == 0 || equ.len() < nz * nvar {
        let (rn, mr, idx) = compute_vector_norms_with_argmax(equ);
        return (
            rn,
            mr,
            idx.map(|flat| (flat / nvar.max(1), flat % nvar.max(1))),
        );
    }
    for (flat, &val) in equ[..nz * nvar].iter().enumerate() {
        if !val.is_finite() {
            return (f64::NAN, f64::NAN, Some((flat / nvar, flat % nvar)));
        }
    }

    let layout = &model.layout;
    let nvar_hydro = layout.nvar_hydro.min(nvar);
    let do_chem_in_sizequ =
        (model.controls.do_burn || model.controls.do_mix) && nvar > nvar_hydro;
    let jmax = if do_chem_in_sizequ { nvar } else { nvar_hydro };

    let ignore_equl = model.controls.convergence_ignore_equl_residuals
        && layout.i_equ_l < jmax;
    let ignore_rti = model.controls.convergence_ignore_alpha_rti_residuals
        && layout.i_dalpha_rti_dt.is_some_and(|i| i < jmax);

    let mut num_terms = jmax * nz;
    if ignore_equl {
        num_terms = num_terms.saturating_sub(nz);
    }
    if ignore_rti {
        num_terms = num_terms.saturating_sub(nz);
    }

    let mut sum_equ = 0.0_f64;
    let mut equmax = 0.0_f64;
    let mut max_loc: Option<(usize, usize)> = None;
    for k in 0..nz {
        let base = k * nvar;
        for j in 0..jmax {
            if ignore_equl && j == layout.i_equ_l {
                continue;
            }
            if ignore_rti && layout.i_dalpha_rti_dt == Some(j) {
                continue;
            }
            let a = equ[base + j].abs();
            sum_equ += a;
            if a > equmax || max_loc.is_none() {
                if a > equmax {
                    equmax = a;
                    max_loc = Some((k, j));
                } else if max_loc.is_none() {
                    max_loc = Some((k, j));
                }
            }
        }
    }

    // Upstream MESA `26.4.1` `sizequ` (`solver_support.f90:219-236, 273-286`,
    // `#177`): when `do_chem_in_sizequ` is true, `jmax` was already set to
    // `nvar`, and lines 273-286 accumulate `nvar_hydro+1..nvar` into `sum_equ`
    // and `num_terms` a second time. Reproduced unconditionally for `26.4.1`
    // parity (`CLAUDE.md`, `#177`).
    if do_chem_in_sizequ {
        num_terms += (nvar - nvar_hydro) * nz;
        for k in 0..nz {
            let base = k * nvar;
            for j in nvar_hydro..nvar {
                let a = equ[base + j].abs();
                sum_equ += a;
                if a > equmax {
                    equmax = a;
                    max_loc = Some((k, j));
                }
            }
        }
    }

    if num_terms == 0 {
        (0.0, 0.0, None)
    } else {
        (sum_equ / (num_terms as f64), equmax, max_loc)
    }
}

/// Computes `(residual_norm, max_residual)` (`sizequ` in
/// `star/private/solver_support.f90:185-294`).
#[must_use]
pub fn compute_residual_norms(
    model: &StarModel,
    nvar: usize,
    equ: &[f64],
) -> (f64, f64) {
    let (rn, mr, _) = compute_residual_norms_with_argmax(model, nvar, equ);
    (rn, mr)
}

/// Computes `(correction_norm, max_correction)` across `dx_scaled` for all
/// `nz` zones (`sizeB` fallback when a full [`StarModel`] is not provided).
#[must_use]
pub fn compute_correction_norms(
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
) -> (f64, f64) {
    if nz == 0 || nvar == 0 || dx_scaled.len() < nz * nvar {
        return compute_vector_norms(dx_scaled);
    }
    compute_vector_norms(&dx_scaled[..nz * nvar])
}

/// Computes `(correction_norm, max_correction, arg_max_zone_var)` with full
/// upstream `sizeB` variable weighting, species filtering, and optional
/// 3-iteration flailing detection (`star/private/solver_support.f90:316-568`).
///
/// Note that `dx_scaled` (`B` in upstream MESA) is already scaled by
/// `1 / x_scale` (`solver_dx = B * x_scale`); upstream `sizeB` multiplies `B`
/// by `correction_weight` without multiplying by `x_scale`
/// (`solver_support.f90:366-384, 428`).
#[must_use]
pub fn compute_model_correction_norms_with_argmax(
    model: &StarModel,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
    _x_scale: &[f64],
) -> (f64, f64, Option<(usize, usize)>) {
    compute_model_correction_norms_in_solver(model, nz, nvar, dx_scaled, None)
}

/// Internal `sizeB` evaluator supporting optional `(solver_iter, tol_max_corr, tracker)`
/// for the 3-iteration diverging-correction check (`solver_support.f90:539-567`).
#[must_use]
pub fn compute_model_correction_norms_in_solver(
    model: &StarModel,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
    flailing: Option<(usize, f64, &mut FlailingTracker)>,
) -> (f64, f64, Option<(usize, usize)>) {
    if nz == 0 || nvar == 0 || dx_scaled.len() < nz * nvar {
        let (cn, mc, idx) = compute_vector_norms_with_argmax(dx_scaled);
        return (
            cn,
            mc,
            idx.map(|flat| (flat / nvar.max(1), flat % nvar.max(1))),
        );
    }

    for (flat, &val) in dx_scaled[..nz * nvar].iter().enumerate() {
        if !val.is_finite() {
            return (f64::NAN, f64::NAN, Some((flat / nvar, flat % nvar)));
        }
    }

    let c = &model.controls;
    let layout = &model.layout;
    let nvar_hydro = layout.nvar_hydro.min(nvar);
    // `sizeB` checks `s% do_burn .or. s% do_mix`
    // (`star/private/solver_support.f90:400, 466`).
    let include_chem = nvar > nvar_hydro && (c.do_burn || c.do_mix);
    let lum_surf_start = model
        .lum_start
        .first()
        .copied()
        .unwrap_or_else(|| model.lum.first().copied().unwrap_or(0.0));

    let mut sum_corr = 0.0_f64;
    let mut max_corr = 0.0_f64;
    let mut max_loc: Option<(usize, usize)> = None;
    let mut num_terms = 0_usize;

    // Upstream MESA `26.4.1` `sizeB` (`solver_support.f90:387-388, 452-501`):
    // `num_xa_terms` and `sum_xa_corr` are initialized once outside `cell_loop`
    // rather than reset per zone, and `species_loop` (`:466-501`) adds each
    // qualifying species a second time. Reproduced verbatim for `26.4.1` parity.
    let mut num_xa_terms = 0_usize;
    let mut sum_xa_corr = 0.0_f64;

    for k in 0..nz {
        let base = k * nvar;
        for j in 0..nvar_hydro {
            // `skip4 = s% i_Hp` and `j == s% i_alpha_RTI` are always
            // skipped (`star/private/solver_support.f90:380, 405-411`).
            if layout.i_hp == Some(j) || layout.i_alpha_rti == Some(j) {
                continue;
            }
            let weight = if j == layout.i_lum {
                if !c.include_l_in_correction_limits {
                    continue;
                }
                let denom = 0.1 * lum_surf_start + model.lum[k].abs();
                if denom == 0.0 { 0.0 } else { 1.0 / denom }
            } else if layout.i_v == Some(j) {
                if !c.include_v_in_correction_limits {
                    continue;
                }
                let cs_s = model
                    .aux
                    .csound_start
                    .get(k)
                    .copied()
                    .unwrap_or(model.csound[k]);
                let denom =
                    0.1 * cs_s + model.v.get(k).copied().unwrap_or(0.0).abs();
                if denom == 0.0 { 0.0 } else { 1.0 / denom }
            } else if layout.i_u == Some(j) {
                if !c.include_u_in_correction_limits {
                    continue;
                }
                let cs_s = model
                    .aux
                    .csound_start
                    .get(k)
                    .copied()
                    .unwrap_or(model.csound[k]);
                let denom =
                    0.1 * cs_s + model.u.get(k).copied().unwrap_or(0.0).abs();
                if denom == 0.0 { 0.0 } else { 1.0 / denom }
            } else if layout.i_w == Some(j) {
                // `star/private/solver_support.f90:368-379`.
                if !c.rsp2_flag || !c.include_w_in_correction_limits {
                    continue;
                }
                let w_k = model
                    .aux
                    .rsp
                    .et_rsp
                    .get(k)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
                    .sqrt();
                if w_k.abs() < 1.0 {
                    0.0
                } else {
                    1.0 / (1.0e3 + w_k.abs())
                }
            } else {
                1.0
            };

            let abs_corr = (dx_scaled[base + j] * weight).abs();
            if !abs_corr.is_finite() {
                return (f64::NAN, f64::NAN, Some((k, j)));
            }
            sum_corr += abs_corr;
            num_terms += 1;
            if abs_corr > max_corr {
                max_corr = abs_corr;
                max_loc = Some((k, j));
            }
        }

        if include_chem {
            // Upstream MESA `26.4.1` `sizeB` (`solver_support.f90:387, 389,
            // 452-501`, `#251` under `#169`): `num_xa_terms` and `sum_xa_corr`
            // are accumulated across zones without resetting per cell, and
            // `species_loop` adds qualifying species a second time. Reproduced
            // verbatim for `26.4.1` parity.
            let n_chem = (nvar - nvar_hydro).min(model.species);
            for sp in 0..n_chem {
                let xa_s = model
                    .xa_start
                    .get(k * model.species + sp)
                    .copied()
                    .unwrap_or_else(|| model.xa[k * model.species + sp]);
                if xa_s >= c.correction_xa_limit {
                    num_xa_terms += 1;
                    sum_xa_corr += dx_scaled[base + nvar_hydro + sp].abs();
                }
            }
            if num_xa_terms > 0 {
                sum_corr += sum_xa_corr / (num_xa_terms as f64);
                num_terms += 1;
            }
            for sp in 0..n_chem {
                let xa_s = model
                    .xa_start
                    .get(k * model.species + sp)
                    .copied()
                    .unwrap_or_else(|| model.xa[k * model.species + sp]);
                if xa_s < c.correction_xa_limit {
                    continue;
                }
                let ivar = nvar_hydro + sp;
                let abs_corr = dx_scaled[base + ivar].abs();
                sum_corr += abs_corr;
                num_terms += 1;
                if !c.ignore_species_in_max_correction && abs_corr > max_corr {
                    max_corr = abs_corr;
                    max_loc = Some((k, ivar));
                }
            }
        }
    }

    let corr_norm = if num_terms == 0 {
        0.0
    } else {
        sum_corr / (num_terms as f64)
    };
    if let Some((solver_iter, tol_max_corr, tracker)) = flailing {
        max_corr =
            tracker.observe(solver_iter, tol_max_corr, max_corr, max_loc);
    }
    (corr_norm, max_corr, max_loc)
}

/// Computes `(correction_norm, max_correction)` with full upstream `sizeB`
/// per-variable and species filtering (`star/private/solver_support.f90:316-568`).
#[must_use]
pub fn compute_model_correction_norms(
    model: &StarModel,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
    x_scale: &[f64],
) -> (f64, f64) {
    let (cn, mc, _) = compute_model_correction_norms_with_argmax(
        model, nz, nvar, dx_scaled, x_scale,
    );
    (cn, mc)
}

/// Selects `(tol_correction_norm, tol_max_correction)` using peak temperature
/// relaxation (`set_tol_correction` in `star/private/struct_burn_mix.f90:467-483`
/// and `relax.f90:4002-4005`).
#[must_use]
pub fn active_correction_tolerances(model: &StarModel) -> (f64, f64) {
    let c = &model.controls;
    let base_norm =
        if model.aux.doing_relax && c.relax_tol_correction_norm != 0.0 {
            c.relax_tol_correction_norm
        } else {
            c.tol_correction_norm
        };
    let base_max = if model.aux.doing_relax && c.relax_tol_max_correction != 0.0
    {
        c.relax_tol_max_correction
    } else {
        c.tol_max_correction
    };
    let max_t = model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let t_max = if max_t.is_finite() {
        flexmesa_math::exp(max_t)
    } else {
        0.0
    };
    if t_max >= c.tol_correction_extreme_t_limit {
        (
            c.tol_correction_norm_extreme_t,
            c.tol_max_correction_extreme_t,
        )
    } else if t_max >= c.tol_correction_high_t_limit {
        (c.tol_correction_norm_high_t, c.tol_max_correction_high_t)
    } else {
        (base_norm, base_max)
    }
}

/// Determines the active gold tolerance level (`0`, `1`, or `2`,
/// `star/private/struct_burn_mix.f90:514-537`).
#[must_use]
pub fn active_gold_level(model: &StarModel) -> u8 {
    let c = &model.controls;
    if model.aux.doing_relax {
        if !c.relax_use_gold_tolerances {
            return 0;
        }
        let max_t_lim = if c.relax_maxt_for_gold_tolerances > 0.0 {
            c.relax_maxt_for_gold_tolerances
        } else {
            c.max_t_for_gold_tolerances
        };
        if max_t_lim > 0.0 {
            let max_lnt =
                model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max_lnt.is_finite() && flexmesa_math::exp(max_lnt) > max_t_lim {
                return 0;
            }
        }
        return 1;
    }

    let mnum = model.model_number as isize;
    let init_offset = model.init_model_number.max(0);
    let gold2_on = (c.use_gold2_tolerances
        && c.steps_before_use_gold2_tolerances < 0)
        || (c.steps_before_use_gold2_tolerances >= 0
            && mnum > c.steps_before_use_gold2_tolerances + init_offset);
    if gold2_on {
        return 2;
    }

    let gold1_on = (c.use_gold_tolerances
        && c.steps_before_use_gold_tolerances < 0)
        || (c.steps_before_use_gold_tolerances >= 0
            && mnum > c.steps_before_use_gold_tolerances + init_offset);
    if gold1_on {
        if c.max_t_for_gold_tolerances > 0.0 {
            let max_lnt =
                model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max_lnt.is_finite()
                && flexmesa_math::exp(max_lnt) > c.max_t_for_gold_tolerances
            {
                return 0;
            }
        }
        return 1;
    }
    0
}

/// Computes `max_tries` for the current solve (`star/private/star_solver.f90:285-297`).
#[must_use]
pub fn active_max_tries(model: &StarModel) -> usize {
    let c = &model.controls;
    if model.doing_first_model_of_run {
        c.max_tries1
    } else if model.step_retry_cnt > 20 {
        c.max_tries_after_20_retries
    } else if model.step_retry_cnt > 10 {
        c.max_tries_after_10_retries
    } else if model.step_retry_cnt > 5 {
        c.max_tries_after_5_retries
    } else if model.step_retry_cnt > 0 {
        c.max_tries_for_retry
    } else {
        c.solver_max_tries_before_reject
    }
}

/// Returns `(tol_residual_norm, tol_max_residual)` for `iter` (1-based) and a
/// fixed `gold_level` (`star_solver.f90:196-236, 309-322`).
#[must_use]
pub fn active_residual_tolerances_with_gold(
    c: &crate::controls::StarControls,
    doing_relax: bool,
    level: u8,
    iter: usize,
) -> (f64, f64) {
    let (it2, it3, rn1, mr1, rn2, mr2, rn3, mr3) = match level {
        2 => (
            c.gold2_iter_for_resid_tol2,
            c.gold2_iter_for_resid_tol3,
            c.gold2_tol_residual_norm1,
            c.gold2_tol_max_residual1,
            c.gold2_tol_residual_norm2,
            c.gold2_tol_max_residual2,
            c.gold2_tol_residual_norm3,
            c.gold2_tol_max_residual3,
        ),
        1 => (
            c.gold_iter_for_resid_tol2,
            c.gold_iter_for_resid_tol3,
            c.gold_tol_residual_norm1,
            c.gold_tol_max_residual1,
            c.gold_tol_residual_norm2,
            c.gold_tol_max_residual2,
            c.gold_tol_residual_norm3,
            c.gold_tol_max_residual3,
        ),
        _ if doing_relax => (
            if c.relax_iter_for_resid_tol2 == 0 {
                c.iter_for_resid_tol2
            } else {
                c.relax_iter_for_resid_tol2
            },
            if c.relax_iter_for_resid_tol3 == 0 {
                c.iter_for_resid_tol3
            } else {
                c.relax_iter_for_resid_tol3
            },
            if c.relax_tol_residual_norm1 == 0.0 {
                c.tol_residual_norm1
            } else {
                c.relax_tol_residual_norm1
            },
            if c.relax_tol_max_residual1 == 0.0 {
                c.tol_max_residual1
            } else {
                c.relax_tol_max_residual1
            },
            if c.relax_tol_residual_norm2 == 0.0 {
                c.tol_residual_norm2
            } else {
                c.relax_tol_residual_norm2
            },
            if c.relax_tol_max_residual2 == 0.0 {
                c.tol_max_residual2
            } else {
                c.relax_tol_max_residual2
            },
            if c.relax_tol_residual_norm3 == 0.0 {
                c.tol_residual_norm3
            } else {
                c.relax_tol_residual_norm3
            },
            if c.relax_tol_max_residual3 == 0.0 {
                c.tol_max_residual3
            } else {
                c.relax_tol_max_residual3
            },
        ),
        _ => (
            c.iter_for_resid_tol2,
            c.iter_for_resid_tol3,
            c.tol_residual_norm1,
            c.tol_max_residual1,
            c.tol_residual_norm2,
            c.tol_max_residual2,
            c.tol_residual_norm3,
            c.tol_max_residual3,
        ),
    };

    if iter >= it2 {
        if iter < it3 { (rn2, mr2) } else { (rn3, mr3) }
    } else {
        (rn1, mr1)
    }
}

/// Returns `(tol_residual_norm, tol_max_residual, tol_correction_norm, tol_max_correction)`
/// for `iter` (1-based) using the 3-stage `gold2` / `gold` / standard tolerance
/// schedule (`star_solver.f90:196-236, 309-322` and `struct_burn_mix.f90:467-537`).
#[must_use]
pub fn active_solver_tolerances(
    model: &StarModel,
    iter: usize,
) -> (f64, f64, f64, f64) {
    let (tol_corr_norm, tol_max_corr) = active_correction_tolerances(model);
    let level = active_gold_level(model);
    let (tol_rn, tol_mr) = active_residual_tolerances_with_gold(
        &model.controls,
        model.aux.doing_relax,
        level,
        iter,
    );
    (tol_rn, tol_mr, tol_corr_norm, tol_max_corr)
}
