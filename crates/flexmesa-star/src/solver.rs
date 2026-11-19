// SPDX-License-Identifier: LGPL-3.0-or-later
//! Block-tridiagonal damped Newton-Raphson Henyey solver (`star_solver.f90`
//! and `solver_support.f90`).
//!
//! Solves the coupled 1D stellar structure (and optional nuclear composition)
//! equations across all `nz` Lagrangian zones simultaneously at each
//! evolutionary timestep $\Delta t$ using [`solve_block_tridiag_bcyclic`]
//! (Hirshman et al. 2010 odd-even cyclic reduction) or sequential block-Thomas
//! elimination ([`solve_block_tridiagonal`]), reproducing MESA `26.4.1`'s
//! `do_solver_work`, `adjust_correction` quadratic/cubic backtracking line
//! search, `Bdomain`, `sizequ`/`sizeB` norms, `max_tries` retry schedule,
//! `solver_itermin`, and post-convergence checks.

use flexmesa_mtx::{
    block_multiply_xa, solve_block_tridiag_bcyclic, solve_block_tridiagonal,
};

use crate::controls::{StarError, StepFailureInfo};
use crate::hydro_eqns::{HenyeySystem, eval_henyey_equations_into};
use crate::model::StarModel;
pub use crate::solver_norms::{
    FlailingTracker, active_correction_tolerances, active_gold_level,
    active_max_tries, active_solver_tolerances, compute_correction_norms,
    compute_model_correction_norms, compute_residual_norms,
    compute_vector_norms,
};
use crate::solver_norms::{
    active_residual_tolerances_with_gold,
    compute_model_correction_norms_in_solver,
    compute_residual_norms_with_argmax,
};
use crate::solver_support::{
    apply_bdomain, check_after_converge, force_another_iteration,
    rti_check_after_converge, set_vars_for_solver,
};

/// Convergence and diagnostic summary returned by [`solve_henyey_step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HenyeySolverReport {
    /// Whether both residual and scaled-correction tolerances were satisfied.
    pub converged: bool,
    /// Number of Newton-Raphson iterations executed.
    pub iterations: usize,
    /// Mean absolute equation residual $\langle |\mathrm{equ}| \rangle$.
    pub residual_norm: f64,
    /// Maximum absolute equation residual $\max |\mathrm{equ}|$.
    pub max_residual: f64,
    /// Mean absolute dimensionless correction $\langle |\delta\tilde{x}| \rangle$.
    pub correction_norm: f64,
    /// Maximum absolute dimensionless correction $\max |\delta\tilde{x}|$.
    pub max_correction: f64,
}

/// Computes the initial Newton step damping factor `correction_factor` $\in (0, 1]$
/// from `scale_correction_norm`, `scale_max_correction`, and [`apply_bdomain`]
/// (`star_solver.f90:391-416` and `solver_support.f90:623-730`).
#[must_use]
pub fn compute_corr_coeff(
    model: &StarModel,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
    x_scale: &[f64],
) -> f64 {
    if nz == 0 || nvar == 0 || dx_scaled.len() < nz * nvar {
        return 1.0;
    }
    let (correction_norm, max_correction) =
        compute_model_correction_norms(model, nz, nvar, dx_scaled, x_scale);
    let c = &model.controls;
    let scale_max_corr = if c.scale_max_correction == 0.0 {
        1.0e99
    } else {
        c.scale_max_correction
    };
    let min_corr_coeff = if model.solver_skip_global_corr_coeff_limit {
        1.0
    } else {
        c.corr_coeff_limit
    };
    let mut correction_factor = 1.0_f64;
    let mut temp_correction_factor = 1.0_f64;
    if correction_norm * correction_factor > c.scale_correction_norm
        && correction_norm > 0.0
    {
        correction_factor =
            correction_factor.min(c.scale_correction_norm / correction_norm);
    }
    if max_correction * correction_factor > scale_max_corr
        && max_correction > 0.0
    {
        temp_correction_factor = scale_max_corr / max_correction;
    }
    correction_factor = correction_factor.max(min_corr_coeff);
    if !c.ignore_min_corr_coeff_for_scale_max_correction {
        temp_correction_factor = temp_correction_factor.max(min_corr_coeff);
    }
    correction_factor = correction_factor.min(temp_correction_factor);

    let nvar_hydro = model.layout.nvar_hydro;
    let xh_ref = build_xh_start_buffer(model);
    let solver_dx = vec![0.0_f64; nz * nvar];
    let mut soln_copy = dx_scaled[..nz * nvar].to_vec();
    let _ = nvar_hydro;
    apply_bdomain(
        model,
        nvar,
        &xh_ref,
        &solver_dx,
        x_scale,
        &mut soln_copy,
        &mut correction_factor,
    );
    correction_factor
}

fn build_xh_start_buffer(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    let nvar_hydro = model.layout.nvar_hydro;
    let mut buf = vec![0.0_f64; nz * nvar_hydro];
    let layout = &model.layout;
    for k in 0..nz {
        let base = k * nvar_hydro;
        buf[base + layout.i_lnd] =
            model.lnd_start.get(k).copied().unwrap_or(model.lnd[k]);
        buf[base + layout.i_lnt] =
            model.ln_t_start.get(k).copied().unwrap_or(model.ln_t[k]);
        buf[base + layout.i_lnr] =
            model.ln_r_start.get(k).copied().unwrap_or(model.ln_r[k]);
        buf[base + layout.i_lum] =
            model.lum_start.get(k).copied().unwrap_or(model.lum[k]);
        if let Some(iv) = layout.i_v {
            buf[base + iv] = model
                .v_start
                .get(k)
                .copied()
                .unwrap_or_else(|| model.v.get(k).copied().unwrap_or(0.0));
        }
        if let Some(iu) = layout.i_u {
            buf[base + iu] = model
                .u_start
                .get(k)
                .copied()
                .unwrap_or_else(|| model.u.get(k).copied().unwrap_or(0.0));
        }
        if let Some(iw) = layout.i_w {
            buf[base + iw] =
                model.aux.rsp.w_start.get(k).copied().unwrap_or_else(|| {
                    model
                        .aux
                        .rsp
                        .et_rsp
                        .get(k)
                        .copied()
                        .unwrap_or(0.0)
                        .max(0.0)
                        .sqrt()
                });
        }
        if let Some(ihp) = layout.i_hp {
            buf[base + ihp] = model
                .xh_start
                .get(base + ihp)
                .copied()
                .filter(|&v| v != 0.0)
                .unwrap_or_else(|| {
                    model.aux.rsp.hp_face.get(k).copied().unwrap_or(1.0)
                });
        }
        if let Some(ia) = layout.i_alpha_rti {
            buf[base + ia] =
                model.aux.alpha_rti_start.get(k).copied().unwrap_or_else(
                    || model.aux.alpha_rti.get(k).copied().unwrap_or(0.0),
                );
        }
    }
    buf
}

#[inline]
fn eval_f(nvar: usize, nz: usize, equ: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for k in 0..nz {
        let base = k * nvar;
        for i in 0..nvar {
            let q = equ[base + i];
            sum_sq += q * q;
        }
    }
    0.5 * sum_sq
}

#[inline]
fn eval_slope(nvar: usize, nz: usize, grad_f: &[f64], soln: &[f64]) -> f64 {
    let mut slope = 0.0_f64;
    for i in 0..nvar {
        for k in 0..nz {
            let idx = k * nvar + i;
            slope += grad_f[idx] * soln[idx];
        }
    }
    slope
}

#[inline]
fn apply_coeff(
    nvar: usize,
    nz: usize,
    dxsave: &[f64],
    soln: &[f64],
    x_scale: &[f64],
    coeff: f64,
    just_use_dx: bool,
    solver_dx: &mut [f64],
) {
    let len = nz * nvar;
    if just_use_dx {
        if coeff == 1.0 {
            for idx in 0..len {
                solver_dx[idx] += x_scale[idx] * soln[idx];
            }
        } else {
            for idx in 0..len {
                solver_dx[idx] += coeff * x_scale[idx] * soln[idx];
            }
        }
    } else if coeff == 1.0 {
        for idx in 0..len {
            solver_dx[idx] = dxsave[idx] + x_scale[idx] * soln[idx];
        }
    } else {
        for idx in 0..len {
            solver_dx[idx] = dxsave[idx] + coeff * x_scale[idx] * soln[idx];
        }
    }
}

/// Solves $J\,\delta\tilde{x} = -\mathrm{equ}$ using `bcyclic` or block-Thomas.
fn solve_linear_henyey_system(
    sys: &HenyeySystem,
    use_bcyclic: bool,
    rhs_buf: &mut Vec<f64>,
) -> Result<Vec<f64>, StarError> {
    rhs_buf.clear();
    rhs_buf.extend(sys.equ.iter().map(|&r| -r));
    let res = if use_bcyclic && sys.nz >= 2 {
        solve_block_tridiag_bcyclic(
            sys.nvar, sys.nz, &sys.lblk, &sys.dblk, &sys.ublk, rhs_buf, true,
        )
    } else {
        solve_block_tridiagonal(
            sys.nvar, sys.nz, &sys.lblk, &sys.dblk, &sys.ublk, rhs_buf,
        )
    };
    res.map_err(|e| {
        StarError::ConvergenceFailure(format!(
            "block-tridiagonal linear solve failed: {e}"
        ))
    })
}

fn worst_residual_site(
    model: &StarModel,
    sys: &HenyeySystem,
) -> Option<(usize, usize, String, String, f64)> {
    if sys.equ.is_empty() || sys.nvar == 0 {
        return None;
    }
    let (zone, eq, val) = crate::step_trace::argmax_abs(&sys.equ, sys.nvar);
    Some((
        zone,
        eq,
        model.layout.equation_name(eq, &model.iso_names),
        model.layout.paired_variable_name(eq, &model.iso_names),
        val,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_step_failure_info(
    model: &StarModel,
    sys: &HenyeySystem,
    iterations: usize,
    residual_norm: f64,
    max_residual: f64,
    correction_norm: f64,
    max_correction: f64,
    corr_site: Option<(usize, usize, f64)>,
    detail: String,
) -> StepFailureInfo {
    let res_site = worst_residual_site(model, sys);
    let equation = res_site.as_ref().map(|s| s.2.clone());
    let (zone, variable, value) = if let Some((k, v_idx, v_val)) = corr_site {
        (
            Some(k),
            Some(model.layout.variable_name(v_idx, &model.iso_names)),
            Some(v_val),
        )
    } else if let Some((k, _, _, var, v_val)) = res_site {
        (Some(k), Some(var), Some(v_val))
    } else {
        (None, None, None)
    };
    StepFailureInfo {
        model_number: model.model_number,
        zone,
        equation,
        variable,
        value,
        residual_norm,
        max_residual,
        correction_norm,
        max_correction,
        dt: model.dt,
        iterations,
        retries: model.step_retry_cnt,
        retry_reason: String::new(),
        detail,
    }
}

/// Executes a single implicit damped Newton-Raphson Henyey solve
/// (`do_solver_converge` and `do_solver_work` in
/// `star/private/struct_burn_mix.f90:359-799` and
/// `star/private/star_solver.f90:164-992`).
///
/// # Arguments
/// * `model` - Mutable [`StarModel`] state with `*_start` arrays already
///   populated for the current step.
/// * `include_chem` - When `true`, solves the fully coupled `(nvar_hydro + species)`
///   system; when `false`, solves the `nvar_hydro` hydrostatic/hydrodynamic
///   equations at fixed composition `xa`.
/// * `use_bcyclic` - When `true`, uses Hirshman et al. (2010) odd-even parallel
///   cyclic reduction ([`solve_block_tridiag_bcyclic`]); when `false`, uses
///   sequential block-Thomas elimination ([`solve_block_tridiagonal`]).
///
/// # Errors
/// Returns [`StarError`] if initial microphysics fails or [`StarError::StepFailure`]
/// if linear solve, line search, tolerance convergence, or post-convergence
/// checks fail.
pub fn solve_henyey_step(
    model: &mut StarModel,
    include_chem: bool,
    use_bcyclic: bool,
) -> Result<HenyeySolverReport, StarError> {
    if model.dt <= 0.0 || model.nz == 0 {
        return Ok(HenyeySolverReport {
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
            max_residual: 0.0,
            correction_norm: 0.0,
            max_correction: 0.0,
        });
    }

    let nz = model.nz;
    let nvar_hydro = model.layout.nvar_hydro;
    let nvar = model.layout.nvar(include_chem);

    model.total_num_solver_calls += 1;
    model.used_extra_iter_in_solver_for_accretion = false;
    let gold_level = active_gold_level(model);
    model.using_gold_tolerances = gold_level > 0;
    let (tol_cn, tol_mc) = active_correction_tolerances(model);

    model.pack_xh();
    let xh_start_buf = build_xh_start_buffer(model);
    let mut solver_dx = vec![0.0_f64; nz * nvar];
    for k in 0..nz {
        let h_base = k * nvar_hydro;
        let s_base = k * nvar;
        for j in 0..nvar_hydro {
            solver_dx[s_base + j] =
                model.xh[h_base + j] - xh_start_buf[h_base + j];
        }
    }
    if include_chem && model.species > 0 {
        let species = model.species;
        model.xa_sub_xa_start.resize(nz * species, 0.0);
        for k in 0..nz {
            let c_base = k * species;
            let s_base = k * nvar + nvar_hydro;
            for j in 0..species {
                let xs = model
                    .xa_start
                    .get(c_base + j)
                    .copied()
                    .unwrap_or(model.xa[c_base + j]);
                let dx_kj = model.xa[c_base + j] - xs;
                model.xa_sub_xa_start[c_base + j] = dx_kj;
                solver_dx[s_base + j] = dx_kj;
            }
        }
    } else {
        model.xa_sub_xa_start.clear();
    }

    let saved_doing_sbm = model.aux.doing_struct_burn_mix;
    model.aux.solver_mlt_eval_count = 0;
    model.aux.solver_had_mlt_flip = false;
    model.aux.fixed_gradr_sticky.resize(nz, false);
    model.aux.fixed_gradr_sticky.fill(false);
    model.aux.mlt_flip_count.resize(nz, 0);
    model.aux.mlt_flip_count.fill(0);
    model.aux.doing_struct_burn_mix = true;

    let mut sys = HenyeySystem::new(0, 0, Vec::new());
    if let Err(e) = eval_henyey_equations_into(model, include_chem, &mut sys) {
        model.aux.doing_struct_burn_mix = saved_doing_sbm;
        model.xa_sub_xa_start.clear();
        return Err(e);
    }

    let (mut residual_norm, mut max_residual, _) =
        compute_residual_norms_with_argmax(model, sys.nvar, &sys.equ);
    if !residual_norm.is_finite() || !max_residual.is_finite() {
        model.aux.doing_struct_burn_mix = saved_doing_sbm;
        model.xa_sub_xa_start.clear();
        return Err(StarError::StepFailure(Box::new(build_step_failure_info(
            model,
            &sys,
            0,
            residual_norm,
            max_residual,
            0.0,
            0.0,
            None,
            String::from("solver failure: initial sizequ returned NaN or Inf"),
        ))));
    }

    let corr_param_factor = if model.controls.corr_param_factor == 0.0 {
        10.0
    } else {
        model.controls.corr_param_factor
    };
    let scale_max_correction = if model.controls.scale_max_correction == 0.0 {
        1.0e99
    } else {
        model.controls.scale_max_correction
    };
    let corr_norm_jump_limit = if model.controls.corr_norm_jump_limit == 0.0 {
        1.0e99
    } else {
        model.controls.corr_norm_jump_limit
    };
    let max_corr_jump_limit = if model.controls.max_corr_jump_limit == 0.0 {
        1.0e99
    } else {
        model.controls.max_corr_jump_limit
    };

    let mut min_corr_coeff = if model.solver_skip_global_corr_coeff_limit {
        1.0
    } else {
        model.controls.corr_coeff_limit
    };
    let max_tries = active_max_tries(model);
    let mut first_iter = true;
    let mut iter = 1_usize;
    let mut iter_used = 0_usize;
    let mut tiny_corr_cnt = 0_usize;
    let mut passed_tol_tests = false;
    let mut failure_detail = String::new();

    let mut correction_norm = 0.0_f64;
    let mut max_correction = 0.0_f64;
    let mut corr_norm_min = 1.0e99_f64;
    let mut max_corr_min = 1.0e99_f64;
    let mut resid_norm_min = 1.0e99_f64;
    let mut max_resid_min = 1.0e99_f64;
    let mut flailing = FlailingTracker::default();
    let mut last_corr_site: Option<(usize, usize, f64)> = None;

    let mut rhs_buf = Vec::with_capacity(nz * nvar);
    let mut grad_f = Vec::new();
    let mut dxsave = Vec::new();
    let mut trial_sys = HenyeySystem::new(0, 0, Vec::new());

    if max_tries == 0 {
        failure_detail = String::from("out of tries (max_tries == 0)");
    } else {
        while !passed_tol_tests {
            iter_used = iter;
            model.total_num_solver_iterations += 1;

            let (tol_rn, tol_mr) = active_residual_tolerances_with_gold(
                &model.controls,
                model.aux.doing_relax,
                gold_level,
                iter,
            );

            let mut soln = match solve_linear_henyey_system(
                &sys,
                use_bcyclic,
                &mut rhs_buf,
            ) {
                Ok(v) => v,
                Err(e) => {
                    failure_detail = format!("bad_matrix: {e}");
                    break;
                }
            };

            let (cn_raw, mc_raw, corr_argmax) =
                compute_model_correction_norms_in_solver(
                    model,
                    sys.nz,
                    sys.nvar,
                    &soln,
                    Some((iter, tol_mc, &mut flailing)),
                );
            correction_norm = cn_raw.abs();
            max_correction = mc_raw.abs();
            corr_norm_min = corr_norm_min.min(correction_norm);
            max_corr_min = max_corr_min.min(max_correction);

            if !correction_norm.is_finite() || !max_correction.is_finite() {
                failure_detail = String::from(
                    "bad result from sizeB: correction info either NaN or Inf",
                );
                break;
            }

            if !model.controls.ignore_too_large_correction
                && correction_norm
                    > corr_param_factor * model.controls.scale_correction_norm
                && !model.doing_first_model_of_run
            {
                failure_detail = String::from("avg corr too large");
                break;
            }

            let mut correction_factor = 1.0_f64;
            let mut temp_correction_factor = 1.0_f64;
            if correction_norm * correction_factor
                > model.controls.scale_correction_norm
                && correction_norm > 0.0
            {
                correction_factor = correction_factor.min(
                    model.controls.scale_correction_norm / correction_norm,
                );
            }
            if max_correction * correction_factor > scale_max_correction
                && max_correction > 0.0
            {
                temp_correction_factor = scale_max_correction / max_correction;
            }
            if iter > model.controls.solver_itermin_until_reduce_min_corr_coeff
                && min_corr_coeff == 1.0
                && model.controls.solver_reduced_min_corr_coeff < 1.0
            {
                min_corr_coeff = model.controls.solver_reduced_min_corr_coeff;
            }
            correction_factor = correction_factor.max(min_corr_coeff);
            if !model
                .controls
                .ignore_min_corr_coeff_for_scale_max_correction
            {
                temp_correction_factor =
                    temp_correction_factor.max(min_corr_coeff);
            }
            correction_factor = correction_factor.min(temp_correction_factor);

            apply_bdomain(
                model,
                sys.nvar,
                &xh_start_buf,
                &solver_dx,
                &sys.x_scale,
                &mut soln,
                &mut correction_factor,
            );

            let mut slope = 0.0_f64;
            if min_corr_coeff < 1.0 {
                grad_f.resize(sys.nz * sys.nvar, 0.0);
                if let Err(e) = block_multiply_xa(
                    sys.nvar,
                    sys.nz,
                    &sys.lblk,
                    &sys.dblk,
                    &sys.ublk,
                    &sys.equ,
                    &mut grad_f,
                ) {
                    failure_detail = format!("block_multiply_xa failed: {e}");
                    break;
                }
                slope = eval_slope(sys.nvar, sys.nz, &grad_f, &soln);
                if !slope.is_finite() || slope > 0.0 {
                    slope = 0.0;
                    min_corr_coeff = 1.0;
                }
            }

            // `adjust_correction` (`star/private/star_solver.f90:737-951`).
            let skip_eval_f = min_corr_coeff == 1.0;
            let need_dxsave = !skip_eval_f
                || model.model_number <= 1
                || model.doing_first_model_of_run;
            if need_dxsave {
                dxsave.clone_from(&solver_dx);
            }
            let fold = if skip_eval_f {
                0.0
            } else {
                let f0 = eval_f(sys.nvar, sys.nz, &sys.equ);
                if !f0.is_finite() {
                    failure_detail =
                        String::from("adjust_correction failed in eval_f");
                    break;
                }
                f0
            };

            let eff_min_coeff = min_corr_coeff.min(correction_factor);
            let mut alam = correction_factor;
            let mut first_time = true;
            let mut f2 = 0.0_f64;
            let mut alam2 = 0.0_f64;
            let mut accepted_coeff: Option<f64> = None;
            let mut adjust_err = String::new();

            for _adj_iter in 1..=1000_usize {
                let coeff_try = eff_min_coeff.max(alam);
                apply_coeff(
                    sys.nvar,
                    sys.nz,
                    &dxsave,
                    &soln,
                    &sys.x_scale,
                    coeff_try,
                    !need_dxsave,
                    &mut solver_dx,
                );

                let eq_res = set_vars_for_solver(
                    model,
                    include_chem,
                    &xh_start_buf,
                    &mut solver_dx,
                )
                .and_then(|()| {
                    eval_henyey_equations_into(
                        model,
                        include_chem,
                        &mut trial_sys,
                    )
                });

                if let Err(e) = eq_res {
                    if alam > eff_min_coeff
                        && (model.model_number <= 1
                            || model.doing_first_model_of_run)
                    {
                        alam = (alam / 10.0).max(eff_min_coeff);
                        continue;
                    }
                    adjust_err = format!(
                        "adjust_correction failed in eval_equations: {e}"
                    );
                    break;
                }

                std::mem::swap(&mut sys, &mut trial_sys);

                if eff_min_coeff == 1.0 {
                    accepted_coeff = Some(coeff_try);
                    break;
                }

                let f = eval_f(sys.nvar, sys.nz, &sys.equ);
                if !f.is_finite() {
                    if alam > eff_min_coeff {
                        alam = (alam / 10.0).max(eff_min_coeff);
                        continue;
                    }
                    adjust_err =
                        String::from("equ norm is NaN or other bad num");
                    break;
                }

                let f_target =
                    (0.5 * fold).max(fold + 1.0e-2 * coeff_try * slope);
                if f <= f_target || alam <= eff_min_coeff {
                    accepted_coeff = Some(coeff_try);
                    break;
                }

                let alam_factor = 0.2_f64;
                let tmplam_raw = if first_time {
                    first_time = false;
                    -slope / (2.0 * (f - fold - slope))
                } else {
                    let rhs1 = f - fold - alam * slope;
                    let rhs2 = f2 - fold - alam2 * slope;
                    let denom = alam - alam2;
                    let a1 =
                        (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / denom;
                    let a2 = (-alam2 * rhs1 / (alam * alam)
                        + alam * rhs2 / (alam2 * alam2))
                        / denom;
                    if a1 == 0.0 {
                        -slope / (2.0 * a2)
                    } else {
                        let disc = a2 * a2 - 3.0 * a1 * slope;
                        if disc < 0.0 {
                            alam * alam_factor
                        } else if a2 <= 0.0 {
                            (-a2 + disc.sqrt()) / (3.0 * a1)
                        } else {
                            -slope / (a2 + disc.sqrt())
                        }
                    }
                };
                let tmplam = if tmplam_raw.is_finite() {
                    tmplam_raw.min(alam * alam_factor)
                } else {
                    alam * alam_factor
                };
                alam2 = alam;
                f2 = f;
                alam = tmplam.max(alam * alam_factor).max(eff_min_coeff);
            }

            let Some(coeff) = accepted_coeff else {
                if !skip_eval_f {
                    solver_dx.clone_from(&dxsave);
                }
                if crate::step_trace::enabled(model.model_number) {
                    if let Some((k_max, _, eq_name, var_name, v_max)) =
                        worst_residual_site(model, &sys)
                    {
                        let mut jl = crate::step_trace::JsonLine::new(
                            "solver_line_search_failed",
                        );
                        jl.field_usize("model", model.model_number);
                        jl.field_f64("dt", model.dt);
                        jl.field_usize("iter", iter);
                        jl.field_f64("rnorm", residual_norm);
                        jl.field_f64("rmax", max_residual);
                        jl.field_usize("zone", k_max);
                        jl.field_str("eq", &eq_name);
                        jl.field_str("var", &var_name);
                        jl.field_f64("val", v_max);
                        jl.emit();
                    }
                }
                failure_detail = if adjust_err.is_empty() {
                    format!("adjust_correction exhausted at iteration {iter}")
                } else {
                    adjust_err
                };
                break;
            };

            if coeff > model.controls.tiny_corr_factor * min_corr_coeff
                || min_corr_coeff >= 1.0
            {
                tiny_corr_cnt = 0;
            } else {
                tiny_corr_cnt += 1;
            }

            let (rn_raw, mr_raw, _) =
                compute_residual_norms_with_argmax(model, sys.nvar, &sys.equ);
            if !rn_raw.is_finite() || !mr_raw.is_finite() {
                failure_detail = String::from(
                    "residual_norm or max_residual is a bad number (NaN or Inf)",
                );
                break;
            }
            residual_norm = rn_raw.abs();
            max_residual = mr_raw.abs();
            resid_norm_min = resid_norm_min.min(residual_norm);
            max_resid_min = max_resid_min.min(max_residual);

            if crate::step_trace::enabled(model.model_number) {
                let (kz, eqi, eq_n, var_n, vmax) = worst_residual_site(
                    model, &sys,
                )
                .unwrap_or((0, 0, String::new(), String::new(), 0.0));
                let (ckz, cvi) = corr_argmax.unwrap_or((0, 0));
                let cvar_n = model.layout.variable_name(cvi, &model.iso_names);
                let mut jl = crate::step_trace::JsonLine::new("solver_iter");
                jl.field_usize("model", model.model_number);
                jl.field_usize("iter", iter);
                jl.field_f64("coeff", coeff);
                jl.field_f64("rnorm", residual_norm);
                jl.field_f64("rmax", vmax);
                jl.field_usize("rzone", kz);
                jl.field_usize("req_idx", eqi);
                jl.field_str("req", &eq_n);
                jl.field_str("rvar", &var_n);
                jl.field_f64("cnorm", correction_norm);
                jl.field_f64("cmax", max_correction);
                jl.field_usize("czone", ckz);
                jl.field_str("cvar", &cvar_n);
                jl.emit();
            }

            let disabled_resid_tests = tol_mr > 1.0e2 && tol_rn > 1.0e2;
            let pass_resid_tests = !disabled_resid_tests
                && max_residual <= tol_mr
                && residual_norm <= tol_rn;
            let pass_corr_without_coeff =
                max_correction <= tol_mc && correction_norm <= tol_cn;
            let pass_corr_with_coeff = max_correction <= tol_mc * coeff
                && correction_norm <= tol_cn * coeff;
            passed_tol_tests = (pass_resid_tests && pass_corr_with_coeff)
                || (disabled_resid_tests && pass_corr_without_coeff);

            last_corr_site = if pass_resid_tests || disabled_resid_tests {
                corr_argmax.map(|(z, v)| {
                    (z, v, soln.get(z * sys.nvar + v).copied().unwrap_or(0.0))
                })
            } else {
                None
            };

            if !passed_tol_tests {
                if iter >= max_tries {
                    failure_detail = format!(
                        "solver did not converge within {iter} iterations"
                    );
                    break;
                } else if !first_iter && !model.doing_first_model_of_run {
                    if correction_norm > corr_norm_jump_limit * corr_norm_min {
                        failure_detail = String::from("avg correction jumped");
                        break;
                    } else if residual_norm
                        > model.controls.resid_norm_jump_limit * resid_norm_min
                    {
                        failure_detail = String::from("avg residual jumped");
                        break;
                    } else if max_correction
                        > max_corr_jump_limit * max_corr_min
                    {
                        failure_detail = String::from("max correction jumped");
                        break;
                    } else if max_residual
                        > model.controls.max_resid_jump_limit * max_resid_min
                    {
                        failure_detail = String::from("max residual jumped");
                        break;
                    } else if tiny_corr_cnt
                        >= model.controls.tiny_corr_coeff_limit
                        && min_corr_coeff < 1.0
                    {
                        failure_detail = String::from("tiny corrections");
                        break;
                    }
                } else if !model.doing_first_model_of_run
                    && coeff < min_corr_coeff.min(correction_factor)
                {
                    failure_detail = String::from("coeff too small");
                    break;
                }
            }

            if passed_tol_tests && iter + 1 < max_tries {
                let itermin = model.controls.solver_itermin;
                if force_another_iteration(model, iter, itermin) {
                    passed_tol_tests = false;
                    tiny_corr_cnt = 0;
                    corr_norm_min = 1.0e99;
                    resid_norm_min = 1.0e99;
                    max_corr_min = 1.0e99;
                    max_resid_min = 1.0e99;
                }
            }

            iter += 1;
            first_iter = false;
        }
    }

    model.aux.doing_struct_burn_mix = saved_doing_sbm;
    model.xa_sub_xa_start.clear();

    if passed_tol_tests {
        if let Err(msg) = check_after_converge(model) {
            passed_tol_tests = false;
            failure_detail = msg;
        } else if model.controls.rti_flag {
            if let Err(e) = rti_check_after_converge(model) {
                passed_tol_tests = false;
                failure_detail =
                    format!("RTI_check_after_converge failed: {e}");
            }
        }
    }

    if crate::step_trace::enabled(model.model_number) {
        if let Some((k_max, i_max, eq_name, var_name, v_max)) =
            worst_residual_site(model, &sys)
        {
            let (tol_rn, tol_mr) = active_residual_tolerances_with_gold(
                &model.controls,
                model.aux.doing_relax,
                gold_level,
                iter_used.max(1),
            );
            let mut jl = crate::step_trace::JsonLine::new("solver");
            jl.field_usize("model", model.model_number);
            jl.field_f64("dt", model.dt);
            jl.field_usize("nz", sys.nz);
            jl.field_bool("converged", passed_tol_tests);
            jl.field_usize("iters", iter_used);
            jl.field_f64("rnorm", residual_norm);
            jl.field_f64("tol_rn", tol_rn);
            jl.field_f64("rmax", v_max);
            jl.field_f64("tol_mr", tol_mr);
            jl.field_usize("zone", k_max);
            jl.field_usize("equ_idx", i_max);
            jl.field_str("eq", &eq_name);
            jl.field_str("var", &var_name);
            jl.field_f64("val", v_max);
            jl.field_usize("nvar_hydro", model.layout.nvar_hydro);
            jl.field_f64("cnorm", correction_norm);
            jl.field_f64("tol_cn", tol_cn);
            jl.field_f64("cmax", max_correction);
            jl.field_f64("tol_mc", tol_mc);
            jl.emit();
        }
    }

    if !passed_tol_tests {
        return Err(StarError::StepFailure(Box::new(build_step_failure_info(
            model,
            &sys,
            iter_used,
            residual_norm,
            max_residual,
            correction_norm,
            max_correction,
            last_corr_site,
            failure_detail,
        ))));
    }

    Ok(HenyeySolverReport {
        converged: true,
        iterations: iter_used,
        residual_norm,
        max_residual,
        correction_norm,
        max_correction,
    })
}
