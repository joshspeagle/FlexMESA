// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar evolution step driver (`star_evolve_step`), automatic retry/backup
//! cascade on solver non-convergence, and multi-step evolutionary loop
//! (`star_evolve`) corresponding to `star/private/evolve.f90`.
//!
//! # Evolutionary Step Sequence (`evolve.f90:155-560`)
//! Each evolutionary step $n \to n+1$ executes:
//! 1. **Snapshot `old` state**: Saves $(\ln\rho, \ln T, \ln R, L, X, \Delta q, M_*)$
//!    into `*_old` so the step can be rolled back if Newton-Raphson fails and
//!    so `varcontrol` can measure the structural change over the step.
//! 2. **Adaptive Remeshing (`do_adjust_mesh`)**: Evaluates mesh spacing
//!    functions $g_k$ and splits/merges zones when `do_remesh` is enabled.
//! 3. **Stellar Winds & Mass Adjustment (`set_mdot` + `do_adjust_mass`)**:
//!    Applies empirical mass loss $\Delta M = \dot{M}\Delta t$ across the
//!    3-region pseudo-Lagrangian grid when wind scaling factors are active.
//! 4. **Structure + Burn + Mix Solve with Automatic Retry Cascade**:
//!    Executes `do_struct_burn_mix(model, dt_try, ...)`. If convergence fails,
//!    restores the pre-step state from `*_old`, halves
//!    $\Delta t_{\mathrm{try}} \leftarrow 0.5\,\Delta t_{\mathrm{try}}$,
//!    increments `model.num_retries`, and retries up to `max_retries`.
//! 5. **Bookkeeping & Timestep Control (`pick_next_timestep`)**:
//!    Increments `model.model_number`, advances `model.star_age` by
//!    $\Delta t / \mathrm{SECYER}$, and computes $\Delta t_{\mathrm{next}}$ via
//!    the Söderlind & Wang (2006) H211 digital filter.

use flexmesa_const::{LN10, R_SUN, SECYER};

use crate::adjust_mass::do_adjust_mass_step;
use crate::controls::{StarError, StepFailureInfo};
use crate::evolve_support::{
    apply_step_pre_physics, eval_total_energy_integrals,
    remesh_star_before_step, update_step_energy_conservation,
};
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;
use crate::solver::HenyeySolverReport;
use crate::struct_burn_mix::{do_struct_burn_mix, refresh_mixing_state};
use crate::termination::check_termination_limits;
use crate::timestep::{
    eval_h211_factor, filter_limit, TimestepAction, TimestepLimitReason,
    TimestepReport,
};
use crate::timestep_limits::timestep_controller;
use crate::winds::do_set_mdot;

/// Diagnostic report returned by a single evolutionary step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvolveStepReport {
    /// Evolutionary step number after completion (`model_number`).
    pub model_number: usize,
    /// Stellar age in years after completion (`star_age`).
    pub star_age: f64,
    /// Timestep $\Delta t$ actually taken in seconds (`s`).
    pub dt: f64,
    /// Proposed next timestep $\Delta t_{\mathrm{next}}$ in seconds (`s`).
    pub dt_next: f64,
    /// Number of Henyey Newton-Raphson iterations on the successful attempt.
    pub iterations: usize,
    /// Number of step retries (`dt` halvings) required during this step.
    pub retries: usize,
    /// Number of zone splits performed during remeshing.
    pub num_splits: usize,
    /// Number of zone merges performed during remeshing.
    pub num_merges: usize,
    /// Measured normalized structural variation `varcontrol`.
    pub varcontrol: f64,
    /// Whether the step converged within tolerances.
    pub converged: bool,
}

/// Restores `model` to its pre-step `*_old` snapshot after a failed solver
/// attempt (`do_restore` / `prepare_for_retry` in `evolve.f90:1012-1120`).
///
/// # Errors
/// Returns [`StarError`] if mesh synchronization or microphysics fails.
pub fn restore_from_old(model: &mut StarModel) -> Result<(), StarError> {
    let nz_old = model.dq_old.len();
    if nz_old != model.nz && nz_old >= 2 {
        model.resize_zones(nz_old);
    }
    model.star_age = model.star_age_old;
    model.mstar = model.mstar_old;
    model.xmstar = model.mstar - model.m_center;
    model.lnd.clone_from(&model.lnd_old);
    model.ln_t.clone_from(&model.ln_t_old);
    model.ln_r.clone_from(&model.ln_r_old);
    model.lum.clone_from(&model.lum_old);
    model.v.clone_from(&model.v_old);
    model.u.clone_from(&model.u_old);
    model.xa.clone_from(&model.xa_old);
    model.dq.clone_from(&model.dq_old);
    model.aux.rsp.et_rsp.clone_from(&model.aux.et_rsp_old);
    model.aux.rsp.hp_face.clone_from(&model.aux.hp_face_old);
    model.aux.alpha_rti.clone_from(&model.aux.alpha_rti_old);
    if model.aux.omega_old.len() == model.nz {
        model.omega.clone_from(&model.aux.omega_old);
        model.j_rot.clone_from(&model.aux.j_rot_old);
    }
    if model.aux.eps_mdot_old.len() == model.nz {
        model.eps_mdot.clone_from(&model.aux.eps_mdot_old);
    } else {
        model.eps_mdot.fill(0.0);
    }
    model.mstar_dot = model.aux.step_hist.mstar_dot_old;
    model.aux.total_energy = model.aux.step_hist.total_energy_old;
    model.aux.cumulative_energy_error =
        model.aux.step_hist.cumulative_energy_error_old;
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.set_m_grav_and_grav();
    model.pack_xh();
    crate::rotation::conserve_shell_j_rot_and_distortion(model);
    set_hydro_vars(model, false, false)?;
    model.aux.burn_avg_epsnuc.fill(0.0);
    model.save_start_values()?;
    Ok(())
}

/// Executes a single evolutionary step (`star_evolve_step` in `evolve.f90`):
/// remeshing $\to$ stellar wind mass adjustment $\to$ pre-physics hooks $\to$
/// structure/burn/mix solve with automatic $\Delta t$ retry cascade $\to$
/// energy conservation and `timestep_controller`.
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if all retry attempts fail.
pub fn star_evolve_step(
    model: &mut StarModel,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
) -> Result<EvolveStepReport, StarError> {
    if model.controls.rsp_flag {
        if model.dt_next > 0.0 {
            model.dt = model.dt_next;
        }
        let prev_dt_old = model.dt_old;
        model.save_old_values();
        model.dt_old = prev_dt_old;
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
        let (e_old, _) = crate::rsp_driver::rsp_total_energy_integrals(model);
        model.aux.rsp.total_energy_old = e_old;
        model.model_number += 1;

        let (iterations, retries) =
            crate::rsp_driver::rsp_one_step(model, None)?;
        model.total_num_solver_iterations += iterations;
        model.num_retries += retries;
        model.star_age = model.aux.rsp.time_sec / SECYER;
        model.dt_next = model.aux.rsp.rsp_dt;
        model.dt_old = model.dt;

        let (etot, mut total_radiation) =
            crate::rsp_driver::rsp_total_energy_integrals(model);
        if model.aux.rsp.just_set_velocities {
            model.aux.rsp.total_energy_old = etot;
            total_radiation = 0.0;
            model.aux.cumulative_energy_error = 0.0;
        }
        let err_e = etot - (model.aux.rsp.total_energy_old - total_radiation);
        model.aux.cumulative_energy_error += err_e;
        model.aux.total_energy = etot;

        return Ok(EvolveStepReport {
            model_number: model.model_number,
            star_age: model.star_age,
            dt: model.dt,
            dt_next: model.dt_next,
            iterations,
            retries,
            num_splits: 0,
            num_merges: 0,
            varcontrol: 0.0,
            converged: true,
        });
    }

    if model.dt_next > 0.0 {
        model.dt = model.dt_next;
    }
    let prev_dt_old = model.dt_old;
    let prev_mstar_dot_old = if model.controls.use_other_adjust_mdot {
        model.aux.step_hist.mstar_dot_old
    } else {
        model.mstar_dot
    };
    let explicit_step_mdot = model.mstar_dot;

    let (mut num_splits, mut num_merges, mut pre_remesh_model) = if do_remesh {
        remesh_star_before_step(model, model.dt)?
    } else {
        (0, 0, None)
    };

    // Snapshot post-remesh state into `*_old` so `varcontrol` and
    // `restore_from_old` operate on the current `nz` zone grid, preserving the
    // previous step's `dt_old` for the H211 filter.
    if !model.controls.rsp2_flag && model.aux.total_energy == 0.0 {
        model.aux.total_energy = eval_total_energy_integrals(model).5;
    }
    model.save_old_values();
    model.aux.step_hist.mstar_dot_old = prev_mstar_dot_old;
    model.dt_old = prev_dt_old;
    let rsp2_e_old = if model.controls.rsp2_flag {
        crate::rsp_driver::rsp_total_energy_integrals(model).0
    } else {
        0.0
    };

    let dt_floor = model.controls.min_timestep_limit.max(0.0);
    let mut dt_try = model.dt;
    // Upstream stops on `dt_next <= 0` (`star/private/evolve.f90:1829-1843`)
    // and on `dt_next < min_timestep_limit` with a strict comparison
    // (`star/private/evolve.f90:1845-1849`).
    if dt_try <= 0.0 || dt_try < model.controls.min_timestep_limit {
        return Err(StarError::ConvergenceFailure(format!(
            "dt ({dt_try:.3e} s) is zero or below min_timestep_limit \
             ({:.3e} s) (`star/private/evolve.f90:1829-1849`)",
            model.controls.min_timestep_limit
        )));
    }
    if let Some(max_yr) = model.controls.max_years_for_timestep {
        dt_try = dt_try.min(max_yr * SECYER);
    }
    if model.controls.rsp.max_timestep > 0.0 {
        dt_try = dt_try.min(model.controls.rsp.max_timestep);
    }
    let prev_dt_limit_ratio_old = model.dt_limit_ratio_old;
    let prev_rot_boost_old = model.aux.step_hist.rotational_mdot_boost_old;
    let prev_was_in_limit = model.aux.step_hist.was_in_implicit_wind_limit;
    let max_retries = model.controls.ts_stop.retry_limit;
    let retry_fac = model.controls.ts_stop.timestep_factor_for_retries;
    if !(retry_fac.is_finite() && retry_fac > 0.0 && retry_fac < 1.0) {
        return Err(StarError::InvalidInput(format!(
            "timestep_factor_for_retries must be in (0, 1), got {retry_fac}"
        )));
    }
    let mut retries = 0_usize;

    loop {
        model.dt = dt_try;
        if model.controls.use_other_adjust_mdot {
            model.mstar_dot = explicit_step_mdot;
        } else {
            do_set_mdot(model)?;
        }
        let mut mdot_redo = crate::mdot_redo::MdotRedoState::setup(model);
        let mut last_conv_mdot: Option<f64> = None;
        let mut force_outer_retry = false;
        let reset_for_redo =
            |m: &mut StarModel, next_mdot: f64| -> Result<(), StarError> {
                m.num_redos += 1;
                restore_from_old(m)?;
                m.dt = dt_try;
                m.dt_old = prev_dt_old;
                m.dt_limit_ratio_old = prev_dt_limit_ratio_old;
                m.aux.step_hist.rotational_mdot_boost_old = prev_rot_boost_old;
                m.mstar_dot = next_mdot;
                Ok(())
            };
        let attempt = loop {
            let step_try = do_adjust_mass_step(model, dt_try)
                .and_then(|()| {
                    if model.mstar_dot != 0.0 && !model.controls.rsp_flag {
                        refresh_mixing_state(model, dt_try.max(1.0), &[]);
                    }
                    apply_step_pre_physics(model, dt_try)
                })
                .and_then(|()| {
                    do_struct_burn_mix(
                        model,
                        dt_try,
                        op_split_burn,
                        use_bcyclic,
                        &[],
                    )
                });
            match step_try {
                Ok(rep) if rep.converged => {
                    crate::rotation::sync_post_henyey_rotation(model, dt_try);
                    match mdot_redo.select_action(model, dt_try) {
                        crate::mdot_redo::SelectMdotAction::Accept => {
                            break Ok(rep)
                        }
                        crate::mdot_redo::SelectMdotAction::RedoWithMdot(
                            new_mdot,
                        ) => {
                            if mdot_redo.last_w_div_w_crit()
                                > model
                                    .controls
                                    .surf_omega_div_omega_crit_limit
                                    .unwrap_or(0.99)
                            {
                                last_conv_mdot = Some(model.mstar_dot);
                            }
                            reset_for_redo(model, new_mdot)?;
                        }
                        crate::mdot_redo::SelectMdotAction::ForceRetry => {
                            force_outer_retry = true;
                            break Ok(rep);
                        }
                    }
                }
                other => {
                    // Used as given, with no floor
                    // (`star/private/evolve.f90:943,1004`).
                    let tol =
                        model.controls.mix_wind.surf_omega_div_omega_crit_tol;
                    if let Some(next_mdot) =
                        mdot_redo.note_solver_failure(model.mstar_dot, tol)
                    {
                        reset_for_redo(model, next_mdot)?;
                        continue;
                    }
                    if let Some(ok_mdot) = last_conv_mdot.take() {
                        reset_for_redo(model, ok_mdot)?;
                        mdot_redo.force_accept_next();
                        continue;
                    }
                    break other;
                }
            }
        };
        let mut ts_retry_reason: Option<String> = None;
        if let Ok(ref rep) = attempt {
            if rep.converged && !force_outer_retry {
                model.model_number += 1;
                model.star_age += dt_try / SECYER;
                model.dt = dt_try;
                update_step_energy_conservation(model, dt_try);
                let (mut ts_action, mut ts_rep) = timestep_controller(model);
                apply_delta_mdot_limits(
                    model,
                    dt_try,
                    prev_dt_old,
                    prev_dt_limit_ratio_old,
                    retries,
                    &mut ts_action,
                    &mut ts_rep,
                );
                if crate::step_trace::enabled(model.model_number) {
                    let mut jl = crate::step_trace::JsonLine::new("star_step");
                    jl.field_usize("model", model.model_number);
                    jl.field_f64("dt", dt_try);
                    jl.field_usize("retries", retries);
                    jl.field_str("action", &format!("{ts_action:?}"));
                    jl.field_f64("dt_next", ts_rep.dt_next);
                    jl.field_str("reason", &format!("{:?}", ts_rep.reason));
                    jl.field_f64("varcontrol", ts_rep.varcontrol);
                    jl.field_f64("mdot", model.mstar_dot);
                    jl.emit();
                }
                match ts_action {
                    TimestepAction::KeepGoing => {
                        return Ok(finalize_accepted_step(
                            model,
                            rep,
                            ts_rep.varcontrol,
                            ts_rep.dt_next,
                            dt_try,
                            rsp2_e_old,
                            retries,
                            num_splits,
                            num_merges,
                        ));
                    }
                    TimestepAction::Retry(reason) => {
                        model.model_number -= 1;
                        ts_retry_reason = Some(reason);
                    }
                    TimestepAction::Terminate(reason) => {
                        let terminated_model = model.model_number;
                        model.model_number -= 1;
                        restore_from_old(model)?;
                        model.dt_old = prev_dt_old;
                        model.dt_limit_ratio_old = prev_dt_limit_ratio_old;
                        model.aux.step_hist.rotational_mdot_boost_old =
                            prev_rot_boost_old;
                        model.aux.step_hist.was_in_implicit_wind_limit =
                            prev_was_in_limit;
                        return Err(StarError::Terminated {
                            model_number: terminated_model,
                            reason,
                        });
                    }
                }
            }
        }
        // Why this attempt is being retried. Kept so that the final failure
        // names the cause of the last attempt (for example a missing table
        // or a rejected option) instead of only counting retries.
        let target_mod = model.model_number + 1;
        let (why, attempt_info) = match &attempt {
            Ok(rep) if rep.converged && force_outer_retry => {
                let msg =
                    String::from("implicit wind redo requested outer retry");
                let info = StepFailureInfo {
                    model_number: target_mod,
                    zone: None,
                    equation: None,
                    variable: None,
                    value: None,
                    residual_norm: rep.residual_norm,
                    max_residual: rep.max_residual,
                    correction_norm: rep.correction_norm,
                    max_correction: rep.max_correction,
                    dt: dt_try,
                    iterations: rep.iterations,
                    retries,
                    retry_reason: msg.clone(),
                    detail: msg.clone(),
                };
                (msg, info)
            }
            Ok(rep) if rep.converged => {
                let msg = ts_retry_reason.unwrap_or_else(|| {
                    String::from("timestep controller requested retry")
                });
                let info = StepFailureInfo {
                    model_number: target_mod,
                    zone: None,
                    equation: None,
                    variable: None,
                    value: None,
                    residual_norm: rep.residual_norm,
                    max_residual: rep.max_residual,
                    correction_norm: rep.correction_norm,
                    max_correction: rep.max_correction,
                    dt: dt_try,
                    iterations: rep.iterations,
                    retries,
                    retry_reason: msg.clone(),
                    detail: msg.clone(),
                };
                (msg, info)
            }
            Ok(rep) => {
                let msg = format!(
                    "solver did not converge (iters={}, rmax={:.3e}, cmax={:.3e})",
                    rep.iterations, rep.max_residual, rep.max_correction
                );
                let info = StepFailureInfo {
                    model_number: target_mod,
                    zone: None,
                    equation: None,
                    variable: None,
                    value: None,
                    residual_norm: rep.residual_norm,
                    max_residual: rep.max_residual,
                    correction_norm: rep.correction_norm,
                    max_correction: rep.max_correction,
                    dt: dt_try,
                    iterations: rep.iterations,
                    retries,
                    retry_reason: msg.clone(),
                    detail: msg.clone(),
                };
                (msg, info)
            }
            Err(StarError::StepFailure(info)) => {
                let mut preserved = (**info).clone();
                preserved.model_number = target_mod;
                preserved.retries = retries;
                let msg = preserved.detail.clone();
                preserved.retry_reason.clone_from(&msg);
                (msg, preserved)
            }
            Err(e) => {
                let msg = format!("error: {e}");
                let info = StepFailureInfo {
                    model_number: target_mod,
                    zone: None,
                    equation: None,
                    variable: None,
                    value: None,
                    residual_norm: 0.0,
                    max_residual: 0.0,
                    correction_norm: 0.0,
                    max_correction: 0.0,
                    dt: dt_try,
                    iterations: 0,
                    retries,
                    retry_reason: msg.clone(),
                    detail: msg.clone(),
                };
                (msg, info)
            }
        };
        if crate::step_trace::enabled(model.model_number + 1) {
            let mut jl = crate::step_trace::JsonLine::new("star_retry");
            jl.field_usize("model", model.model_number + 1);
            jl.field_f64("dt", dt_try);
            jl.field_usize("retries", retries);
            jl.field_str("why", &why);
            jl.emit();
        }
        model.dt_limit_ratio_old = prev_dt_limit_ratio_old;
        model.aux.step_hist.rotational_mdot_boost_old = prev_rot_boost_old;
        model.aux.step_hist.was_in_implicit_wind_limit = prev_was_in_limit;
        retries += 1;
        let next_dt = dt_try * retry_fac;
        if (max_retries > 0 && retries > max_retries)
            || next_dt <= dt_floor
            || !next_dt.is_finite()
            || next_dt <= 0.0
        {
            restore_from_old(model)?;
            model.dt_old = prev_dt_old;
            let limit_str = if max_retries == 0 {
                String::from("unlimited")
            } else {
                max_retries.to_string()
            };
            let mut final_info = attempt_info;
            final_info.retries = retries;
            final_info.dt = next_dt;
            final_info.retry_reason = why;
            final_info.detail = format!(
                "star_evolve_step failed after {retries}/{limit_str} retries \
                 (dt={next_dt:.3e} s, dt_floor={dt_floor:.3e} s)"
            );
            return Err(StarError::StepFailure(Box::new(final_info)));
        }
        model.num_retries += 1;
        dt_try = next_dt;
        if let Some(snap) = pre_remesh_model.take() {
            snap.restore_into(model)?;
            num_splits = 0;
            num_merges = 0;
            if !model.controls.rsp2_flag && model.aux.total_energy == 0.0 {
                model.aux.total_energy = eval_total_energy_integrals(model).5;
            }
            model.save_old_values();
            model.aux.step_hist.mstar_dot_old = prev_mstar_dot_old;
            model.dt_old = prev_dt_old;
        } else {
            restore_from_old(model)?;
            model.dt_old = prev_dt_old;
        }
        let target_mod = model.model_number + 1;
        model.aux.step_hist.model_number_for_last_retry = Some(target_mod);
        model.aux.step_hist.timestep_hold = model
            .aux
            .step_hist
            .timestep_hold
            .max(target_mod + model.controls.ts_stop.retry_hold);
    }
}

fn apply_delta_mdot_limits(
    model: &mut StarModel,
    dt_try: f64,
    prev_dt_old: f64,
    prev_dt_limit_ratio_old: f64,
    retries: usize,
    action: &mut TimestepAction,
    ts_rep: &mut TimestepReport,
) {
    if !matches!(action, TimestepAction::KeepGoing) {
        return;
    }
    let mw = &model.controls.mix_wind;
    let ts = &model.controls.ts_stop;
    let tdc = ts.time_delta_coeff;
    let lim = mw.delta_mdot_limit * tdc;
    let hard_lim = mw.delta_mdot_hard_limit * tdc;
    if lim <= 0.0 && hard_lim <= 0.0 {
        return;
    }
    let (mdot, mdot_old) = (model.mstar_dot, model.aux.step_hist.mstar_dot_old);
    let denom = mw.delta_mdot_atol.max(0.0) * (flexmesa_const::M_SUN / SECYER)
        + mw.delta_mdot_rtol.max(0.0) * mdot.abs().max(mdot_old.abs());
    if denom <= 0.0 || (mdot - mdot_old).abs() == 0.0 {
        return;
    }
    let delta_mdot = (mdot - mdot_old).abs() / denom;
    let skip_hard = model.model_number <= 1
        || (retries > 0
            && ts.relax_hard_limits_after_retry
            && !ts.never_skip_hard_limits);
    if hard_lim > 0.0 && !skip_hard && delta_mdot > hard_lim {
        *action = TimestepAction::Retry(String::from("delta_mdot_hard_limit"));
        return;
    }
    if lim > 0.0 && delta_mdot > lim {
        let r_dmdot = delta_mdot / lim;
        if r_dmdot > ts_rep.dt_limit_ratio {
            let log_max_t =
                model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    / LN10;
            let max_tf =
                if log_max_t > ts.min_logt_for_max_timestep_factor_at_high_t {
                    ts.max_timestep_factor_at_high_t
                } else {
                    ts.max_timestep_factor
                };
            let min_tf = ts.min_timestep_factor;
            let factor = if ts.use_dt_low_pass_controller {
                eval_h211_factor(
                    r_dmdot,
                    prev_dt_limit_ratio_old,
                    dt_try,
                    prev_dt_old,
                    max_tf,
                    min_tf,
                )
                .0
            } else {
                let mut f = filter_limit(1.0 / r_dmdot.max(1.0e-10));
                if max_tf > 0.0 {
                    f = f.min(max_tf);
                }
                if min_tf > 0.0 {
                    f = f.max(min_tf);
                }
                f
            };
            let mut dt_cand = dt_try * factor;
            if model.aux.step_hist.timestep_hold > model.model_number
                && dt_cand > dt_try
            {
                dt_cand = dt_try;
            }
            if dt_cand < ts_rep.dt_next {
                ts_rep.dt_next = dt_cand;
                ts_rep.reason = TimestepLimitReason::MassChange;
            }
            ts_rep.dt_limit_ratio = r_dmdot;
            model.dt_next = ts_rep.dt_next;
            model.dt_limit_ratio_old = r_dmdot;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_accepted_step(
    model: &mut StarModel,
    rep: &HenyeySolverReport,
    varcontrol: f64,
    mut dt_next: f64,
    dt_try: f64,
    rsp2_e_old: f64,
    retries: usize,
    num_splits: usize,
    num_merges: usize,
) -> EvolveStepReport {
    if let Some(max_yr) = model.controls.max_years_for_timestep {
        dt_next = dt_next.min(max_yr * SECYER);
    }
    if model.controls.rsp.max_timestep > 0.0 {
        dt_next = dt_next.min(model.controls.rsp.max_timestep);
    }
    if model.controls.rsp2_flag {
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
        model.aux.rsp.time_sec += dt_try;
        let (etot, total_rad) =
            crate::rsp_driver::rsp_total_energy_integrals(model);
        model.aux.rsp.total_energy_old = rsp2_e_old;
        model.aux.cumulative_energy_error += etot - (rsp2_e_old - total_rad);
        model.aux.total_energy = etot;
        update_rsp2_cycle_stats(model, dt_try, &mut dt_next);
    }
    model.dt_next = dt_next;
    model.dt_old = dt_try;
    model.aux.step_hist.mstar_dot_old = model.mstar_dot;

    EvolveStepReport {
        model_number: model.model_number,
        star_age: model.star_age,
        dt: dt_try,
        dt_next,
        iterations: rep.iterations,
        retries,
        num_splits,
        num_merges,
        varcontrol,
        converged: true,
    }
}

fn update_rsp2_cycle_stats(
    model: &mut StarModel,
    dt_try: f64,
    dt_next: &mut f64,
) {
    if model.nz == 0 {
        return;
    }
    let r_surf = model.r[0];
    if model.aux.rsp.rmax <= 0.0 {
        model.aux.rsp.rmax = r_surf;
        model.aux.rsp.rmin = r_surf;
    } else {
        model.aux.rsp.rmax = model.aux.rsp.rmax.max(r_surf);
        model.aux.rsp.rmin = model.aux.rsp.rmin.min(r_surf);
    }
    if model.controls.v_flag {
        let v_beg = model.v_start[0];
        let v_end = model.v[0];
        if v_end <= 0.0 && v_beg >= 0.0 && (v_beg - v_end).abs() > 0.0 {
            let frac = (v_beg / (v_beg - v_end)).clamp(0.0, 1.0);
            let t_cross = model.aux.rsp.time_old_sec + frac * dt_try;
            if model.aux.rsp.tt1 > 0.0 {
                let p_new = t_cross - model.aux.rsp.tt1;
                if p_new > 0.0 {
                    model.aux.rsp.period = p_new;
                    model.aux.rsp.num_periods += 1;
                    model.aux.rsp.delta_r =
                        (model.aux.rsp.rmax - model.aux.rsp.rmin) / R_SUN;
                }
            }
            model.aux.rsp.tt1 = t_cross;
            model.aux.rsp.rmax = r_surf;
            model.aux.rsp.rmin = r_surf;
        }
    }
    if model.controls.rsp2.target_steps_per_cycle > 0
        && model.aux.rsp.period > 0.0
    {
        let dt_cap = model.aux.rsp.period
            / (model.controls.rsp2.target_steps_per_cycle as f64);
        *dt_next = (*dt_next).min(dt_cap);
    }
}

/// Evolves `model` across up to `max_steps` steps (`star_evolve` in
/// `evolve.f90`), stopping early if `model.controls.max_model_number` or
/// `model.controls.max_age` is reached.
///
/// # Errors
/// Returns [`StarError`] if any step fails to converge after retries.
pub fn star_evolve(
    model: &mut StarModel,
    max_steps: usize,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
) -> Result<Vec<EvolveStepReport>, StarError> {
    let mut reports = Vec::with_capacity(max_steps);
    for _ in 0..max_steps {
        if let Some(max_mod) = model.controls.max_model_number {
            if model.model_number >= max_mod {
                break;
            }
        }
        if let Some(max_age) = model.controls.max_age {
            if model.star_age >= max_age * (1.0 - 1.0e-12) {
                break;
            }
            let dt_base = if model.dt_next > 0.0 {
                model.dt_next
            } else {
                model.dt
            };
            let rem_sec = (max_age - model.star_age) * SECYER;
            if rem_sec > 0.0 && dt_base > rem_sec {
                model.dt_next = rem_sec;
                model.dt = model.dt_next;
            }
        }
        let rep =
            star_evolve_step(model, do_remesh, op_split_burn, use_bcyclic)?;
        reports.push(rep);
        if check_termination_limits(model).is_some() {
            break;
        }
    }
    Ok(reports)
}
