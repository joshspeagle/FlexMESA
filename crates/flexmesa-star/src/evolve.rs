// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar evolution step driver (`star_evolve_step`), automatic retry/redo
//! cascade on solver or timestep-limit rejection, and multi-step evolutionary
//! loop (`star_evolve`) corresponding to `star/private/evolve.f90` and
//! `star/job/run_star_support.f90:176-275`.

use flexmesa_const::{CLIGHT, R_SUN, SECDAY, SECYER};

use crate::adjust_mass::do_adjust_mass_step;
use crate::controls::{StarError, StepFailureInfo};
pub use crate::evolve_support::snapshot_step_old_extras;
use crate::evolve_support::{
    apply_step_pre_physics_with_extras, record_start_of_step_energy_integrals,
    remesh_star_before_step, snapshot_step_start_extras,
    update_step_energy_conservation,
};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist_runner::{DefaultStarExtras, ExtrasDirective, StarExtras};
use crate::model::StarModel;
use crate::solver::HenyeySolverReport;
use crate::struct_burn_mix::{do_struct_burn_mix, refresh_mixing_state};
use crate::termination::{LimitCheckOutcome, do_check_limits};
use crate::timestep::TimestepAction;
use crate::timestep_limits::{
    timestep_controller, timestep_controller_with_extras,
};
use crate::winds::do_set_mdot_with_extras;

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
    /// Number of step retries (`dt` reductions) required during this step.
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

/// Prepared single-attempt state returned by [`star_evolve_step_part1`] and
/// consumed by [`star_evolve_step_part2`] (`star/private/evolve.f90:282-730`).
pub struct StarStepPart1State {
    dt_try: f64,
    dt_prev: f64,
    has_prev_generation: bool,
    rsp2_e_old: f64,
    num_splits: usize,
    num_merges: usize,
    pre_remesh_model: Option<crate::PrimaryStateBuffer>,
    saved: PreAttemptScalars,
    target_mod: usize,
}

/// Converged single-attempt solver payload returned inside
/// [`StarStepAttemptOutcome::Converged`] and passed into
/// [`star_pick_next_timestep_after_step`] or [`star_finalize_terminated_step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarStepConverged {
    /// Converged Henyey solver report.
    pub solver_report: HenyeySolverReport,
    /// Timestep $\Delta t$ taken in this attempt in seconds (`s`).
    pub dt: f64,
    /// RSP2 start-of-step total energy (`0.0` when `rsp2_flag` is `false`).
    pub rsp2_e_old: f64,
    /// Number of zone splits performed in `part1`.
    pub num_splits: usize,
    /// Number of zone merges performed in `part1`.
    pub num_merges: usize,
}

/// Outcome of a single non-looping solver attempt from
/// [`star_evolve_step_part2`] (`star/private/evolve.f90:472-730`).
#[derive(Debug)]
pub enum StarStepAttemptOutcome {
    /// Single attempt converged; `model` is advanced to `target_mod` and ready
    /// for coupled binary evolution, limit checks, and
    /// [`star_pick_next_timestep_after_step`].
    Converged(StarStepConverged),
    /// Single attempt failed to converge or requested a retry
    /// (`do_step_part2 = retry` in `star/private/evolve.f90:678-680`); `model`
    /// is restored to its pre-attempt state and `model.num_retries` is left
    /// untouched for the outer driver's retry preparation.
    Retry {
        /// Diagnostic retry cause string.
        reason: String,
        /// Whether `why_Tlim == Tlim_neg_X` was set during the attempt.
        why_tlim_neg_x: bool,
        /// Underlying retriable [`StarError`], if any.
        cause: Option<Box<StarError>>,
    },
}

/// Outcome of [`star_pick_next_timestep_after_step`]
/// (`star/private/evolve.f90:2037-2145`).
#[derive(Debug)]
pub enum StarPickTimestepOutcome {
    /// Step accepted and finalized (`dt_next`, `dt_old`, `mstar_dot_old`,
    /// `have_mlt_vc = true`, `doing_first_model_of_run = false`).
    KeepGoing(EvolveStepReport),
    /// A stellar hard timestep limit requested a retry (`TimestepAction::Retry`).
    Retry {
        /// Diagnostic retry reason string.
        reason: String,
        /// Whether `why_Tlim == Tlim_neg_X` was set during timestep checks.
        why_tlim_neg_x: bool,
    },
}

/// Restores `model` to its pre-step `*_old` snapshot after a failed solver
/// attempt or step redo (`set_current_to_old` in `star/private/evolve_support.f90:147-240`).
///
/// # Errors
/// Returns [`StarError`] if mesh synchronization or microphysics fails.
pub fn restore_from_old(model: &mut StarModel) -> Result<(), StarError> {
    let nz_old = model.dq_old.len();
    if nz_old != model.nz && nz_old >= 2 {
        model.resize_zones(nz_old);
    }
    model.star_age = model.star_age_old;
    model.time_sec = model.time_old_sec;
    model.mstar = model.mstar_old;
    model.m_center = model.m_center_old;
    model.r_center = model.r_center_old;
    model.l_center = model.l_center_old;
    model.v_center = model.v_center_old;
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
    model.mlt_vc.clone_from(&model.aux.mlt_vc_old);
    model.conv_vel.clone_from(&model.aux.mlt_vc_old);
    model.aux.okay_to_set_mlt_vc = false;
    if model.aux.omega_old.len() == model.nz {
        model.omega.clone_from(&model.aux.omega_old);
        model.j_rot.clone_from(&model.aux.j_rot_old);
    }
    if model.aux.eps_mdot_old.len() == model.nz {
        model.eps_mdot.clone_from(&model.aux.eps_mdot_old);
    } else {
        model.eps_mdot.fill(0.0);
    }
    if model.aux.step_hist.teff_old > 0.0 {
        model.teff = model.aux.step_hist.teff_old;
    }
    model.mstar_dot = model.aux.step_hist.mstar_dot_old;
    model.aux.step_hist.time_sec = model.aux.step_hist.time_old_sec;
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
    model.mlt_vc.clone_from(&model.aux.mlt_vc_old);
    model.conv_vel.clone_from(&model.aux.mlt_vc_old);
    model.aux.okay_to_set_mlt_vc = true;
    model.aux.burn_avg_epsnuc.fill(0.0);
    record_start_of_step_energy_integrals(model);
    model.save_start_values()?;
    snapshot_step_start_extras(model);
    Ok(())
}

/// Executes a single evolutionary step (`star_evolve_step` in `evolve.f90`):
/// remeshing $\to$ stellar wind mass adjustment $\to$ pre-physics hooks $\to$
/// structure/burn/mix solve with automatic $\Delta t$ retry cascade $\to$
/// energy conservation and `timestep_controller`.
///
/// # Errors
/// Returns [`StarError`] if a termination limit is reached or all retry
/// attempts fail.
pub fn star_evolve_step(
    model: &mut StarModel,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
) -> Result<EvolveStepReport, StarError> {
    let mut extras = DefaultStarExtras;
    star_evolve_step_with_extras(
        model,
        do_remesh,
        op_split_burn,
        use_bcyclic,
        &mut extras,
    )
}

/// Executes a single evolutionary step with explicit [`StarExtras`] hooks
/// (`star_evolve_step` in `evolve.f90`).
///
/// # Errors
/// Returns [`StarError`] if a required `other_*` hook is missing, a termination
/// limit is triggered, or all retry attempts fail.
pub fn star_evolve_step_with_extras(
    model: &mut StarModel,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
    extras: &mut dyn StarExtras,
) -> Result<EvolveStepReport, StarError> {
    let (rep, _) = star_evolve_step_outcome_with_extras(
        model,
        do_remesh,
        op_split_burn,
        use_bcyclic,
        extras,
    )?;
    Ok(rep)
}

/// Executes upstream `star_evolve_step_part1` (`star/private/evolve.f90:282-469`)
/// for a single non-looping step attempt, as invoked by coupled binary
/// evolution (`binary/private/run_binary_support.f90:455-456`).
///
/// Unlike [`star_evolve_step`], when `first_try` is `false` this preserves
/// `model.step_retry_cnt` and `model.num_retries` across outer binary retries
/// so the solver's `active_max_tries` schedule and retry line search match
/// MESA `26.4.1`.
///
/// # Errors
/// Returns [`StarError`] if timestep controls are invalid, `dt <= 0` or below
/// `min_timestep_limit`, remeshing fails, or `set_mdot` fails.
#[allow(clippy::too_many_lines)]
pub fn star_evolve_step_part1(
    model: &mut StarModel,
    first_try: bool,
    do_remesh: bool,
    extras: &mut dyn StarExtras,
) -> Result<StarStepPart1State, StarError> {
    if model.controls.rsp_flag {
        return Err(StarError::InvalidInput(String::from(
            "star_evolve_step_part1 does not support rsp_flag",
        )));
    }
    model.aux.step_hist.termination_code = None;
    let retry_fac = model.controls.ts_stop.timestep_factor_for_retries;
    if !(retry_fac.is_finite() && retry_fac > 0.0 && retry_fac < 1.0) {
        return Err(StarError::InvalidInput(format!(
            "timestep_factor_for_retries must be in (0, 1), got {retry_fac}"
        )));
    }

    sync_time_sec_from_age(model);
    if model.doing_first_model_of_run {
        model.aux.step_hist.timestep_hold = 0;
    }

    let dt_prev = model.dt;
    let mut dt_next_in = if model.dt_next > 0.0 || model.dt <= 0.0 {
        model.dt_next
    } else {
        model.dt
    };

    // `prepare_for_new_step` (`star/private/evolve.f90:1830-1852`)
    if dt_next_in <= 0.0 {
        let reason = check_max_age_reached_reason(model)
            .unwrap_or_else(|| String::from("dt_is_zero"));
        return Err(StarError::Terminated {
            model_number: model.model_number,
            reason,
        });
    }
    if dt_next_in < model.controls.min_timestep_limit {
        return Err(StarError::Terminated {
            model_number: model.model_number,
            reason: String::from("min_timestep_limit"),
        });
    }

    let has_prev_generation = !model.lnd_old.is_empty();
    let prev_dt_old = model.dt_old;
    let prev_mstar_dot_old = if model.controls.use_other_adjust_mdot {
        model.aux.step_hist.mstar_dot_old
    } else {
        model.mstar_dot
    };
    let explicit_step_mdot = model.mstar_dot;

    model.aux.okay_to_set_mlt_vc = false;
    let (num_splits, num_merges, mut pre_remesh_model) =
        if first_try && do_remesh {
            match remesh_star_before_step(model, dt_prev) {
                Ok(res) => res,
                Err(e) => {
                    model.aux.okay_to_set_mlt_vc = true;
                    return Err(e);
                }
            }
        } else {
            (0, 0, None)
        };

    snapshot_step_old_extras(model);
    model.save_old_values();
    record_start_of_step_energy_integrals(model);
    snapshot_step_start_extras(model);
    model.aux.okay_to_set_mlt_vc = true;
    model.aux.step_hist.mstar_dot_old = prev_mstar_dot_old;
    model.aux.step_hist.time_old_sec = model.aux.step_hist.time_sec;
    model.dt_old = prev_dt_old;
    let rsp2_e_old = if model.controls.rsp2_flag {
        model.aux.step_hist.total_energy_old
    } else {
        0.0
    };

    let pre_step_time_sec = model.aux.step_hist.time_sec;
    let prev_dt_limit_ratio_old = model.dt_limit_ratio_old;
    let prev_rot_boost_old = model.aux.step_hist.rotational_mdot_boost_old;
    let prev_was_in_limit = model.aux.step_hist.was_in_implicit_wind_limit;
    let saved = PreAttemptScalars {
        dt_old: prev_dt_old,
        mstar_dot_old: prev_mstar_dot_old,
        dt_limit_ratio_old: prev_dt_limit_ratio_old,
        rot_boost_old: prev_rot_boost_old,
        was_in_limit: prev_was_in_limit,
        time_sec: pre_step_time_sec,
        time_old_sec: model.aux.step_hist.time_old_sec,
        phase_of_evolution: model.aux.step_hist.phase_of_evolution,
        using_revised_max_yr_dt: model.aux.step_hist.using_revised_max_yr_dt,
        revised_max_yr_dt: model.aux.step_hist.revised_max_yr_dt,
        max_years_for_timestep: model.controls.max_years_for_timestep,
        max_timestep: model.controls.ts_stop.max_timestep,
    };

    if first_try {
        // Clip `dt_next_in` against `max_age*` (`evolve.f90:1899-1915`).
        if model.controls.max_age > 0.0
            && (pre_step_time_sec + dt_next_in)
                > model.controls.max_age * SECYER
        {
            dt_next_in =
                (model.controls.max_age * SECYER - pre_step_time_sec).max(0.0);
        } else if let Some(max_sec) = model
            .controls
            .max_age_in_seconds
            .filter(|&s| s > 0.0 && (pre_step_time_sec + dt_next_in) > s)
        {
            dt_next_in = (max_sec - pre_step_time_sec).max(0.0);
        } else if model.controls.ts_stop.max_age_in_days > 0.0
            && (pre_step_time_sec + dt_next_in)
                > model.controls.ts_stop.max_age_in_days * SECDAY
        {
            dt_next_in = (model.controls.ts_stop.max_age_in_days * SECDAY
                - pre_step_time_sec)
                .max(0.0);
        }
        if dt_next_in <= 0.0 {
            let _ = restore_attempt_state(model, pre_remesh_model.take(), saved);
            model.dt = dt_prev;
            let reason = check_max_age_reached_reason(model)
                .unwrap_or_else(|| String::from("dt_is_zero"));
            return Err(StarError::Terminated {
                model_number: model.model_number,
                reason,
            });
        }
    }

    let mut dt_try = dt_next_in;
    {
        let ts = &model.controls.ts_stop;
        if first_try {
            let force_min = if ts.force_timestep_min == 0.0 {
                SECYER * ts.force_timestep_min_years
            } else {
                ts.force_timestep_min
            };
            if force_min > 0.0 && dt_try < force_min {
                dt_try = (dt_try * ts.force_timestep_min_factor).min(force_min);
            }
            let force_dt = if ts.force_timestep == 0.0 {
                SECYER * ts.force_timestep_years
            } else {
                ts.force_timestep
            };
            if force_dt > 0.0 {
                dt_try = force_dt;
            } else {
                if model.controls.max_years_for_timestep > 0.0 {
                    dt_try = dt_try
                        .min(model.controls.max_years_for_timestep * SECYER);
                }
                if ts.max_timestep > 0.0 {
                    dt_try = dt_try.min(ts.max_timestep);
                }
                if model.controls.rsp.max_timestep > 0.0 {
                    dt_try = dt_try.min(model.controls.rsp.max_timestep);
                }
            }
        } else {
            if model.controls.max_years_for_timestep > 0.0 {
                dt_try =
                    dt_try.min(model.controls.max_years_for_timestep * SECYER);
            }
            if ts.max_timestep > 0.0 {
                dt_try = dt_try.min(ts.max_timestep);
            }
            if model.controls.rsp.max_timestep > 0.0 {
                dt_try = dt_try.min(model.controls.rsp.max_timestep);
            }
        }
    }

    if first_try {
        // `prepare_for_new_step` resets `retry_cnt` once per step (`evolve.f90:1938`).
        model.step_retry_cnt = 0;
    }
    model.dt = dt_try;
    model.solver_first_try_of_step = first_try;
    model.why_tlim_neg_x = false;
    model.aux.step_hist.num_rotation_solver_steps = 0;

    // `set_start_of_step_info` (`star/private/evolve.f90:1784-1787`)
    if model.nz > 0 {
        model.aux.step_hist.starting_t_center = Some(model.t[model.nz - 1]);
        model.aux.step_hist.prev_ledd =
            Some(crate::report_utils::eval_ledd_erg_s(model));
    }
    if model.controls.use_other_adjust_mdot {
        let prev_rot_boost = model.aux.step_hist.rotational_mdot_boost_old;
        do_set_mdot_with_extras(model, extras)?;
        if let Some(adj_mdot) =
            extras.other_adjust_mdot(model, model.mstar_dot)?
        {
            model.mstar_dot = adj_mdot;
        } else {
            model.mstar_dot = explicit_step_mdot;
            model.aux.step_hist.rotational_mdot_boost_old = prev_rot_boost;
        }
    } else {
        do_set_mdot_with_extras(model, extras)?;
    }

    let target_mod = model.model_number + 1;
    Ok(StarStepPart1State {
        dt_try,
        dt_prev,
        has_prev_generation,
        rsp2_e_old,
        num_splits,
        num_merges,
        pre_remesh_model,
        saved,
        target_mod,
    })
}

fn run_post_henyey_omega_mix(
    model: &mut StarModel,
    dt_try: f64,
    target_mod: usize,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if !model.controls.rotation_flag {
        return Ok(());
    }
    crate::hydro_rotation::set_i_rot(model, false);
    crate::hydro_rotation::set_omega(model);
    let dt_omega = if model.controls.mix_wind.premix_omega {
        0.5 * dt_try
    } else {
        dt_try
    };
    match crate::omega_mix::do_solve_omega_mix_with_extras(
        model, dt_omega, extras,
    )? {
        crate::omega_mix::OmegaMixOutcome::KeepGoing => {
            crate::hydro_rotation::set_rotation_info(model, false);
            Ok(())
        }
        crate::omega_mix::OmegaMixOutcome::Retry(msg) => {
            Err(StarError::ConvergenceFailure(msg))
        }
        crate::omega_mix::OmegaMixOutcome::Terminate(msg) => {
            Err(StarError::Terminated {
                model_number: target_mod,
                reason: msg,
            })
        }
    }
}

/// Executes upstream `star_evolve_step_part2` (`star/private/evolve.f90:472-730`)
/// for a single non-looping solver attempt (with only the single-star
/// rotational `implicit_mdot_loop`, never an internal retry cascade).
///
/// # Errors
/// Returns [`StarError`] only for non-retriable input, I/O, or termination
/// errors; retriable solver failures return [`StarStepAttemptOutcome::Retry`].
#[allow(clippy::too_many_lines)]
pub fn star_evolve_step_part2(
    model: &mut StarModel,
    mut state: StarStepPart1State,
    first_try: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
    extras: &mut dyn StarExtras,
) -> Result<StarStepAttemptOutcome, StarError> {
    let dt_try = state.dt_try;
    let target_mod = state.target_mod;
    let saved = state.saved;
    let mut mdot_redo = crate::mdot_redo::MdotRedoState::setup_with_generation(
        model,
        state.has_prev_generation,
    );
    let mut force_outer_retry = false;
    let mut attempt_first_try = first_try;
    let reset_for_mdot_redo =
        |m: &mut StarModel, next_mdot: f64| -> Result<(), StarError> {
            m.num_redos += 1;
            restore_from_old(m)?;
            m.dt = dt_try;
            m.dt_old = saved.dt_old;
            m.dt_limit_ratio_old = saved.dt_limit_ratio_old;
            m.aux.step_hist.rotational_mdot_boost_old = saved.rot_boost_old;
            m.mstar_dot = next_mdot;
            Ok(())
        };

    let attempt = loop {
        model.time_sec = model.time_old_sec + dt_try;
        model.star_age = model.time_sec / SECYER;
        if model.v_center != 0.0 {
            model.r_center = model.r_center_old + dt_try * model.v_center;
            if model.r_center < 0.0 {
                break Err(StarError::ConvergenceFailure(format!(
                    "R_center < 0 ({:.3e}) at model {target_mod}",
                    model.r_center
                )));
            }
        }
        // `star/private/evolve.f90:674-675`:
        // `skip_global_corr_coeff_limit = first_try || model_number_for_last_retry != model_number`.
        model.solver_first_try_of_step = attempt_first_try;
        model.solver_skip_global_corr_coeff_limit = attempt_first_try
            || model.aux.step_hist.model_number_for_last_retry
                != Some(target_mod);

        let step_try = do_adjust_mass_step(model, dt_try)
            .and_then(|()| {
                if model.mstar_dot != 0.0 && !model.controls.rsp_flag {
                    refresh_mixing_state(model, dt_try.max(1.0), &[])?;
                }
                apply_step_pre_physics_with_extras(model, dt_try, extras)
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
                if exceeds_clight_velocity(model) {
                    break Err(StarError::ConvergenceFailure(format!(
                        "velocity exceeded clight at model {target_mod}"
                    )));
                }
                if let Err(e) = run_post_henyey_omega_mix(
                    model, dt_try, target_mod, extras,
                ) {
                    break Err(e);
                }
                match mdot_redo
                    .select_action_with_extras(model, dt_try, extras)
                {
                    Ok(crate::mdot_redo::SelectMdotAction::Accept) => {
                        break Ok(rep);
                    }
                    Ok(crate::mdot_redo::SelectMdotAction::RedoWithMdot(
                        new_mdot,
                    )) => {
                        reset_for_mdot_redo(model, new_mdot)?;
                        attempt_first_try = false;
                    }
                    Ok(crate::mdot_redo::SelectMdotAction::ForceRetry) => {
                        force_outer_retry = true;
                        break Ok(rep);
                    }
                    Err(e) => break Err(e),
                }
            }
            other => {
                // `star/private/evolve.f90:678-680`: return `retry` directly to
                // the outer driver without swallowing retries internally.
                break other;
            }
        }
    };

    if matches!(&attempt, Err(e) if !e.is_retriable()) {
        restore_attempt_state(model, state.pre_remesh_model.take(), saved)?;
        model.dt = state.dt_prev;
        return Err(attempt.expect_err("matched non-retriable StarError above"));
    }

    if let Ok(rep) = attempt {
        if rep.converged && !force_outer_retry {
            model.model_number = target_mod;
            model.aux.step_hist.time_old_sec = saved.time_sec;
            model.aux.step_hist.time_sec = saved.time_sec + dt_try;
            model.time_old_sec = saved.time_sec;
            model.time_sec = saved.time_sec + dt_try;
            model.star_age = model.aux.step_hist.time_sec / SECYER;
            model.dt = dt_try;
            update_step_energy_conservation(model, dt_try);
            if !model
                .aux
                .step_hist
                .error_in_energy_conservation
                .is_finite()
                || !model.aux.cumulative_energy_error.is_finite()
            {
                let why = String::from(
                    "okay_energy_conservation: non-finite energy error",
                );
                model.model_number -= 1;
                restore_attempt_state(model, None, saved)?;
                model.dt = state.dt_prev;
                return Ok(StarStepAttemptOutcome::Retry {
                    reason: why,
                    why_tlim_neg_x: model.why_tlim_neg_x,
                    cause: None,
                });
            }

            if model.controls.ts_stop.max_timestep_hi_t_limit > 0.0
                && model.controls.max_years_for_timestep
                    != model.controls.ts_stop.hi_t_max_years_for_timestep
                && model.t.iter().copied().fold(0.0_f64, f64::max)
                    >= model.controls.ts_stop.max_timestep_hi_t_limit
            {
                model.controls.max_years_for_timestep =
                    model.controls.ts_stop.hi_t_max_years_for_timestep;
                model.controls.ts_stop.max_timestep =
                    SECYER * model.controls.max_years_for_timestep;
            }
            return Ok(StarStepAttemptOutcome::Converged(StarStepConverged {
                solver_report: rep,
                dt: dt_try,
                rsp2_e_old: state.rsp2_e_old,
                num_splits: state.num_splits,
                num_merges: state.num_merges,
            }));
        }
        let why_tlim_neg_x = model.why_tlim_neg_x;
        let why = describe_retry_cause(&Ok(rep), force_outer_retry, None);
        if crate::step_trace::enabled(target_mod) {
            let mut jl = crate::step_trace::JsonLine::new("star_retry");
            jl.field_usize("model", target_mod);
            jl.field_f64("dt", dt_try);
            jl.field_usize("retries", model.step_retry_cnt);
            jl.field_str("why", &why);
            jl.emit();
        }
        restore_attempt_state(model, None, saved)?;
        model.dt = state.dt_prev;
        model.why_tlim_neg_x = why_tlim_neg_x;
        return Ok(StarStepAttemptOutcome::Retry {
            reason: why,
            why_tlim_neg_x,
            cause: None,
        });
    }

    let why_tlim_neg_x = model.why_tlim_neg_x;
    let why = describe_retry_cause(&attempt, force_outer_retry, None);
    let err = attempt.expect_err("handled Ok branch above");
    if crate::step_trace::enabled(target_mod) {
        let mut jl = crate::step_trace::JsonLine::new("star_retry");
        jl.field_usize("model", target_mod);
        jl.field_f64("dt", dt_try);
        jl.field_usize("retries", model.step_retry_cnt);
        jl.field_str("why", &why);
        jl.emit();
    }
    restore_attempt_state(model, None, saved)?;
    model.dt = state.dt_prev;
    model.why_tlim_neg_x = why_tlim_neg_x;
    Ok(StarStepAttemptOutcome::Retry {
        reason: why,
        why_tlim_neg_x,
        cause: Some(Box::new(err)),
    })
}

/// Evaluates upstream `star_pick_next_timestep`
/// (`star/private/evolve.f90:2037-2145`) after a converged
/// [`star_evolve_step_part2`] attempt (`binary/private/run_binary_support.f90:546-558`).
///
/// # Errors
/// Returns [`StarError`] if a termination limit or hook error occurs.
pub fn star_pick_next_timestep_after_step(
    model: &mut StarModel,
    conv: &StarStepConverged,
    retries: usize,
    extras: &mut dyn StarExtras,
) -> Result<StarPickTimestepOutcome, StarError> {
    let (ts_action, ts_rep, ts_extra_redo) =
        timestep_controller_with_extras(model, extras)?;
    if ts_extra_redo {
        return Ok(StarPickTimestepOutcome::Retry {
            reason: String::from("other_timestep_limit requested redo"),
            why_tlim_neg_x: model.why_tlim_neg_x,
        });
    }
    if crate::step_trace::enabled(model.model_number) {
        let mut jl = crate::step_trace::JsonLine::new("star_step");
        jl.field_usize("model", model.model_number);
        jl.field_f64("dt", conv.dt);
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
            Ok(StarPickTimestepOutcome::KeepGoing(finalize_accepted_step(
                model,
                &conv.solver_report,
                ts_rep.varcontrol,
                ts_rep.dt_next,
                conv.dt,
                conv.rsp2_e_old,
                retries,
                conv.num_splits,
                conv.num_merges,
            )))
        }
        TimestepAction::Retry(reason) => Ok(StarPickTimestepOutcome::Retry {
            reason,
            why_tlim_neg_x: model.why_tlim_neg_x,
        }),
        TimestepAction::Terminate(reason) => Err(StarError::Terminated {
            model_number: model.model_number,
            reason,
        }),
    }
}

/// Finalizes an accepted terminating step after [`star_evolve_step_part2`]
/// without triggering timestep-controller retries
/// (`binary/private/run_binary_support.f90:546-561`).
pub fn star_finalize_terminated_step(
    model: &mut StarModel,
    conv: &StarStepConverged,
    retries: usize,
) -> EvolveStepReport {
    let (_, ts_rep) = timestep_controller(model);
    finalize_accepted_step(
        model,
        &conv.solver_report,
        ts_rep.varcontrol,
        ts_rep.dt_next,
        conv.dt,
        conv.rsp2_e_old,
        retries,
        conv.num_splits,
        conv.num_merges,
    )
}

fn check_max_age_reached_reason(model: &StarModel) -> Option<String> {
    let c = &model.controls;
    let time_sec = model.aux.step_hist.time_sec;
    if c.max_age > 0.0 && time_sec >= c.max_age * SECYER {
        return Some(String::from("max_age"));
    }
    if c.ts_stop.max_age_in_days > 0.0
        && time_sec >= c.ts_stop.max_age_in_days * SECDAY
    {
        return Some(String::from("max_age_in_days"));
    }
    if c.max_age_in_seconds
        .is_some_and(|s| s > 0.0 && time_sec >= s)
    {
        return Some(String::from("max_age_in_seconds"));
    }
    None
}

fn exceeds_clight_velocity(model: &StarModel) -> bool {
    if !model.controls.ts_stop.retry_for_v_above_clight {
        return false;
    }
    (model.controls.v_flag && model.v.iter().any(|v| v.abs() > CLIGHT))
        || (model.controls.u_flag && model.u.iter().any(|u| u.abs() > CLIGHT))
}

#[allow(clippy::too_many_lines)]
pub(crate) fn star_evolve_step_outcome_with_extras(
    model: &mut StarModel,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
    extras: &mut dyn StarExtras,
) -> Result<(EvolveStepReport, Option<String>), StarError> {
    model.aux.step_hist.termination_code = None;
    let retry_fac = model.controls.ts_stop.timestep_factor_for_retries;
    if !(retry_fac.is_finite() && retry_fac > 0.0 && retry_fac < 1.0) {
        return Err(StarError::InvalidInput(format!(
            "timestep_factor_for_retries must be in (0, 1), got {retry_fac}"
        )));
    }
    if model.controls.rsp_flag {
        if model.dt_next > 0.0 {
            model.dt = model.dt_next;
        }
        let prev_dt_old = model.dt_old;
        snapshot_step_old_extras(model);
        model.save_old_values();
        model.dt_old = prev_dt_old;
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
        let (e_old, _) = crate::rsp_driver::rsp_total_energy_integrals(model);
        model.aux.rsp.total_energy_old = e_old;
        model.model_number += 1;

        let (iterations, retries) =
            crate::rsp_driver::rsp_one_step_with_extras(model, None, extras)?;
        model.total_num_solver_iterations += iterations;
        model.num_retries += retries;
        model.time_sec = model.aux.rsp.time_sec;
        model.star_age = model.aux.rsp.time_sec / SECYER;
        model.aux.step_hist.time_old_sec = model.aux.rsp.time_old_sec;
        model.aux.step_hist.time_sec = model.aux.rsp.time_sec;
        model.dt_next = model.aux.rsp.rsp_dt;
        model.dt_old = model.dt;
        model.doing_first_model_of_run = false;

        let (etot, mut total_radiation) =
            crate::rsp_driver::rsp_total_energy_integrals(model);
        if model.aux.rsp.just_set_velocities {
            model.aux.rsp.total_energy_old = etot;
            total_radiation = 0.0;
            model.aux.cumulative_energy_error = 0.0;
            model.aux.step_hist.cumulative_energy_error_old = 0.0;
        }
        let err_e = etot - (model.aux.rsp.total_energy_old - total_radiation);
        model.aux.cumulative_energy_error =
            if model.controls.absolute_cumulative_energy_err {
                model.aux.step_hist.cumulative_energy_error_old + err_e.abs()
            } else {
                model.aux.step_hist.cumulative_energy_error_old + err_e
            };
        model.aux.total_energy = etot;
        let sh = &mut model.aux.step_hist;
        sh.total_internal_energy_end = model.aux.rsp.ethe;
        sh.total_gravitational_energy_end = model.aux.rsp.egrv;
        sh.total_radial_kinetic_energy_end = model.aux.rsp.ekin;
        sh.total_rotational_kinetic_energy_end = 0.0;
        sh.total_turbulent_energy_end = model.aux.rsp.econ;
        sh.total_energy_end = etot;
        sh.error_in_energy_conservation = err_e;
        sh.total_energy_sources_and_sinks = -total_radiation;

        let check_term = match extras.extras_check_model(model) {
            ExtrasDirective::Terminate(reason) => Some(reason),
            ExtrasDirective::Error(msg) => {
                return Err(StarError::ConvergenceFailure(msg));
            }
            _ => None,
        };

        return Ok((
            EvolveStepReport {
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
            },
            check_term,
        ));
    }

    sync_time_sec_from_age(model);
    if model.doing_first_model_of_run {
        model.aux.step_hist.timestep_hold = 0;
    }

    let dt_prev = model.dt;
    let mut dt_next_in = if model.dt_next > 0.0 || model.dt <= 0.0 {
        model.dt_next
    } else {
        model.dt
    };

    // `prepare_for_new_step` (`star/private/evolve.f90:1830-1852`)
    if dt_next_in <= 0.0 {
        let reason = check_max_age_reached_reason(model)
            .unwrap_or_else(|| String::from("dt_is_zero"));
        return Err(StarError::Terminated {
            model_number: model.model_number,
            reason,
        });
    }
    if dt_next_in < model.controls.min_timestep_limit {
        return Err(StarError::Terminated {
            model_number: model.model_number,
            reason: String::from("min_timestep_limit"),
        });
    }

    let has_prev_generation = !model.lnd_old.is_empty();
    let prev_dt_old = model.dt_old;
    let prev_mstar_dot_old = if model.controls.use_other_adjust_mdot {
        model.aux.step_hist.mstar_dot_old
    } else {
        model.mstar_dot
    };
    let explicit_step_mdot = model.mstar_dot;

    // Keep `okay_to_set_mlt_vc = false` across `remesh_star_before_step` until
    // `save_old_values()` copies `mlt_vc` into `mlt_vc_old` (`evolve.f90:321`,
    // `evolve_support.f90:58, 193-195`).
    model.aux.okay_to_set_mlt_vc = false;
    let (mut num_splits, mut num_merges, mut pre_remesh_model) = if do_remesh {
        match remesh_star_before_step(model, dt_prev) {
            Ok(res) => res,
            Err(e) => {
                model.aux.okay_to_set_mlt_vc = true;
                return Err(e);
            }
        }
    } else {
        (0, 0, None)
    };

    snapshot_step_old_extras(model);
    model.save_old_values();
    record_start_of_step_energy_integrals(model);
    snapshot_step_start_extras(model);
    model.aux.okay_to_set_mlt_vc = true;
    model.aux.step_hist.mstar_dot_old = prev_mstar_dot_old;
    model.aux.step_hist.time_old_sec = model.aux.step_hist.time_sec;
    model.dt_old = prev_dt_old;
    let rsp2_e_old = if model.controls.rsp2_flag {
        model.aux.step_hist.total_energy_old
    } else {
        0.0
    };

    let pre_step_time_sec = model.aux.step_hist.time_sec;
    let prev_dt_limit_ratio_old = model.dt_limit_ratio_old;
    let prev_rot_boost_old = model.aux.step_hist.rotational_mdot_boost_old;
    let prev_was_in_limit = model.aux.step_hist.was_in_implicit_wind_limit;
    let saved = PreAttemptScalars {
        dt_old: prev_dt_old,
        mstar_dot_old: prev_mstar_dot_old,
        dt_limit_ratio_old: prev_dt_limit_ratio_old,
        rot_boost_old: prev_rot_boost_old,
        was_in_limit: prev_was_in_limit,
        time_sec: pre_step_time_sec,
        time_old_sec: model.aux.step_hist.time_old_sec,
        phase_of_evolution: model.aux.step_hist.phase_of_evolution,
        using_revised_max_yr_dt: model.aux.step_hist.using_revised_max_yr_dt,
        revised_max_yr_dt: model.aux.step_hist.revised_max_yr_dt,
        max_years_for_timestep: model.controls.max_years_for_timestep,
        max_timestep: model.controls.ts_stop.max_timestep,
    };

    // Clip `dt_next_in` against `max_age*` (`evolve.f90:1899-1915, 550-555`).
    if model.controls.max_age > 0.0
        && (pre_step_time_sec + dt_next_in) > model.controls.max_age * SECYER
    {
        dt_next_in =
            (model.controls.max_age * SECYER - pre_step_time_sec).max(0.0);
    } else if let Some(max_sec) = model
        .controls
        .max_age_in_seconds
        .filter(|&s| s > 0.0 && (pre_step_time_sec + dt_next_in) > s)
    {
        dt_next_in = (max_sec - pre_step_time_sec).max(0.0);
    } else if model.controls.ts_stop.max_age_in_days > 0.0
        && (pre_step_time_sec + dt_next_in)
            > model.controls.ts_stop.max_age_in_days * SECDAY
    {
        dt_next_in = (model.controls.ts_stop.max_age_in_days * SECDAY
            - pre_step_time_sec)
            .max(0.0);
    }
    if dt_next_in <= 0.0 {
        let _ = restore_attempt_state(model, pre_remesh_model.take(), saved);
        model.dt = dt_prev;
        let reason = check_max_age_reached_reason(model)
            .unwrap_or_else(|| String::from("dt_is_zero"));
        return Err(StarError::Terminated {
            model_number: model.model_number,
            reason,
        });
    }

    let mut dt_try = dt_next_in;
    {
        let ts = &model.controls.ts_stop;
        let force_min = if ts.force_timestep_min == 0.0 {
            SECYER * ts.force_timestep_min_years
        } else {
            ts.force_timestep_min
        };
        if force_min > 0.0 && dt_try < force_min {
            dt_try = (dt_try * ts.force_timestep_min_factor).min(force_min);
        }
        let force_dt = if ts.force_timestep == 0.0 {
            SECYER * ts.force_timestep_years
        } else {
            ts.force_timestep
        };
        if force_dt > 0.0 {
            dt_try = force_dt;
        } else {
            if model.controls.max_years_for_timestep > 0.0 {
                dt_try =
                    dt_try.min(model.controls.max_years_for_timestep * SECYER);
            }
            if ts.max_timestep > 0.0 {
                dt_try = dt_try.min(ts.max_timestep);
            }
            if model.controls.rsp.max_timestep > 0.0 {
                dt_try = dt_try.min(model.controls.rsp.max_timestep);
            }
        }
    }

    let dt_floor = model.controls.min_timestep_limit.max(0.0);
    let max_retries = model.controls.ts_stop.retry_limit;
    let target_mod = model.model_number + 1;
    let mut retries = 0_usize;
    let mut redo_cnt = 0_usize;
    let mut first_try = true;

    loop {
        model.dt = dt_try;
        model.step_retry_cnt = retries;
        model.solver_first_try_of_step = first_try;
        model.why_tlim_neg_x = false;
        model.aux.step_hist.num_rotation_solver_steps = 0;
        // `set_start_of_step_info` (`star/private/evolve.f90:1784-1787`):
        // snapshot start-of-step `starting_T_center` and optical-depth-averaged
        // `prev_Ledd` before evaluating `set_mdot`.
        if model.nz > 0 {
            model.aux.step_hist.starting_t_center = Some(model.t[model.nz - 1]);
            model.aux.step_hist.prev_ledd =
                Some(crate::report_utils::eval_ledd_erg_s(model));
        }
        if model.controls.use_other_adjust_mdot {
            let prev_rot_boost = model.aux.step_hist.rotational_mdot_boost_old;
            do_set_mdot_with_extras(model, extras)?;
            if let Some(adj_mdot) =
                extras.other_adjust_mdot(model, model.mstar_dot)?
            {
                model.mstar_dot = adj_mdot;
            } else {
                model.mstar_dot = explicit_step_mdot;
                model.aux.step_hist.rotational_mdot_boost_old = prev_rot_boost;
            }
        } else {
            do_set_mdot_with_extras(model, extras)?;
        }

        let mut mdot_redo =
            crate::mdot_redo::MdotRedoState::setup_with_generation(
                model,
                has_prev_generation,
            );
        let mut force_outer_retry = false;
        let reset_for_mdot_redo =
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
            model.time_sec = model.time_old_sec + dt_try;
            model.star_age = model.time_sec / SECYER;
            if model.v_center != 0.0 {
                model.r_center = model.r_center_old + dt_try * model.v_center;
                if model.r_center < 0.0 {
                    break Err(StarError::ConvergenceFailure(format!(
                        "R_center < 0 ({:.3e}) at model {target_mod}",
                        model.r_center
                    )));
                }
            }
            model.solver_first_try_of_step = first_try;
            model.solver_skip_global_corr_coeff_limit = first_try
                || model.aux.step_hist.model_number_for_last_retry
                    != Some(target_mod);

            let step_try = do_adjust_mass_step(model, dt_try)
                .and_then(|()| {
                    if model.mstar_dot != 0.0 && !model.controls.rsp_flag {
                        refresh_mixing_state(model, dt_try.max(1.0), &[])?;
                    }
                    apply_step_pre_physics_with_extras(model, dt_try, extras)
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
                    if exceeds_clight_velocity(model) {
                        break Err(StarError::ConvergenceFailure(format!(
                            "velocity exceeded clight at model {target_mod}"
                        )));
                    }
                    if let Err(e) = run_post_henyey_omega_mix(
                        model, dt_try, target_mod, extras,
                    ) {
                        break Err(e);
                    }
                    match mdot_redo
                        .select_action_with_extras(model, dt_try, extras)
                    {
                        Ok(crate::mdot_redo::SelectMdotAction::Accept) => {
                            break Ok(rep);
                        }
                        Ok(
                            crate::mdot_redo::SelectMdotAction::RedoWithMdot(
                                new_mdot,
                            ),
                        ) => {
                            reset_for_mdot_redo(model, new_mdot)?;
                            first_try = false;
                        }
                        Ok(crate::mdot_redo::SelectMdotAction::ForceRetry) => {
                            force_outer_retry = true;
                            break Ok(rep);
                        }
                        Err(e) => break Err(e),
                    }
                }
                other => {
                    // Upstream `evolve.f90:678-680`: any non-converged solve inside
                    // `implicit_mdot_loop` immediately exits `do_step_part2` with
                    // `retry` (`#139`, `#152`).
                    break other;
                }
            }
        };

        let mut ts_retry_reason: Option<String> = None;
        let mut requested_redo = false;

        if let Ok(ref rep) = attempt {
            if rep.converged && !force_outer_retry {
                model.model_number = target_mod;
                model.aux.step_hist.time_old_sec = pre_step_time_sec;
                model.aux.step_hist.time_sec = pre_step_time_sec + dt_try;
                model.time_old_sec = pre_step_time_sec;
                model.time_sec = pre_step_time_sec + dt_try;
                model.star_age = model.aux.step_hist.time_sec / SECYER;
                model.dt = dt_try;
                update_step_energy_conservation(model, dt_try);
                if !model
                    .aux
                    .step_hist
                    .error_in_energy_conservation
                    .is_finite()
                    || !model.aux.cumulative_energy_error.is_finite()
                {
                    model.model_number -= 1;
                    ts_retry_reason = Some(String::from(
                        "okay_energy_conservation: non-finite energy error",
                    ));
                } else {
                    if model.controls.ts_stop.max_timestep_hi_t_limit > 0.0
                        && model.controls.max_years_for_timestep
                            != model
                                .controls
                                .ts_stop
                                .hi_t_max_years_for_timestep
                        && model.t.iter().copied().fold(0.0_f64, f64::max)
                            >= model.controls.ts_stop.max_timestep_hi_t_limit
                    {
                        model.controls.max_years_for_timestep =
                            model.controls.ts_stop.hi_t_max_years_for_timestep;
                        model.controls.ts_stop.max_timestep =
                            SECYER * model.controls.max_years_for_timestep;
                    }

                    // Upstream runs `do_check_limits` before `extras_check_model`
                    // and `pick_next_timestep` (`run_star_support.f90:211-213`).
                    let limit_outcome =
                        match do_check_limits(model, None, false) {
                            Ok(out) => out,
                            Err(e) => {
                                model.model_number -= 1;
                                restore_attempt_state(
                                    model,
                                    pre_remesh_model.take(),
                                    saved,
                                )?;
                                model.dt = dt_prev;
                                return Err(e);
                            }
                        };
                    match limit_outcome {
                        LimitCheckOutcome::Redo { dt_new, .. } => {
                            let max_redos = model.controls.ts_stop.redo_limit;
                            redo_cnt += 1;
                            if max_redos > 0 && redo_cnt > max_redos {
                                let terminated_model = model.model_number;
                                model.model_number -= 1;
                                restore_attempt_state(
                                    model,
                                    pre_remesh_model.take(),
                                    saved,
                                )?;
                                model.dt = dt_prev;
                                return Err(StarError::Terminated {
                                    model_number: terminated_model,
                                    reason: String::from("redo_limit"),
                                });
                            }
                            if !(dt_new.is_finite() && dt_new > 0.0) {
                                let terminated_model = model.model_number;
                                model.model_number -= 1;
                                restore_attempt_state(
                                    model,
                                    pre_remesh_model.take(),
                                    saved,
                                )?;
                                model.dt = dt_prev;
                                return Err(StarError::Terminated {
                                    model_number: terminated_model,
                                    reason: String::from("dt_is_zero"),
                                });
                            }
                            model.num_redos += 1;
                            model.model_number -= 1;
                            restore_attempt_state(model, None, saved)?;
                            dt_try = dt_new;
                            first_try = false;
                            continue;
                        }
                        LimitCheckOutcome::Retry { reason } => {
                            model.model_number -= 1;
                            ts_retry_reason = Some(reason);
                        }
                        LimitCheckOutcome::Terminate { code, .. } => {
                            model.aux.step_hist.termination_code =
                                Some(code.clone());
                            let (_, ts_rep) = timestep_controller(model);
                            return Ok((
                                finalize_accepted_step(
                                    model,
                                    rep,
                                    ts_rep.varcontrol,
                                    ts_rep.dt_next,
                                    dt_try,
                                    rsp2_e_old,
                                    retries,
                                    num_splits,
                                    num_merges,
                                ),
                                Some(code),
                            ));
                        }
                        LimitCheckOutcome::KeepGoing => {
                            let mut pending_check_term: Option<String> = None;
                            match extras.extras_check_model(model) {
                                ExtrasDirective::KeepGoing => {}
                                ExtrasDirective::Terminate(reason) => {
                                    pending_check_term = Some(reason);
                                }
                                ExtrasDirective::Error(msg) => {
                                    model.model_number -= 1;
                                    restore_attempt_state(
                                        model,
                                        pre_remesh_model.take(),
                                        saved,
                                    )?;
                                    model.dt = dt_prev;
                                    return Err(StarError::ConvergenceFailure(
                                        msg,
                                    ));
                                }
                                ExtrasDirective::Redo => {
                                    model.model_number -= 1;
                                    requested_redo = true;
                                }
                                ExtrasDirective::Retry => {
                                    model.model_number -= 1;
                                    ts_retry_reason = Some(String::from(
                                        "extras_check_model retry",
                                    ));
                                }
                            }

                            if !requested_redo && ts_retry_reason.is_none() {
                                let (ts_action, ts_rep, ts_extra_redo) =
                                    timestep_controller_with_extras(
                                        model, extras,
                                    )?;
                                if ts_extra_redo {
                                    model.model_number -= 1;
                                    requested_redo = true;
                                } else {
                                    if crate::step_trace::enabled(
                                        model.model_number,
                                    ) {
                                        let mut jl =
                                            crate::step_trace::JsonLine::new(
                                                "star_step",
                                            );
                                        jl.field_usize(
                                            "model",
                                            model.model_number,
                                        );
                                        jl.field_f64("dt", dt_try);
                                        jl.field_usize("retries", retries);
                                        jl.field_str(
                                            "action",
                                            &format!("{ts_action:?}"),
                                        );
                                        jl.field_f64("dt_next", ts_rep.dt_next);
                                        jl.field_str(
                                            "reason",
                                            &format!("{:?}", ts_rep.reason),
                                        );
                                        jl.field_f64(
                                            "varcontrol",
                                            ts_rep.varcontrol,
                                        );
                                        jl.field_f64("mdot", model.mstar_dot);
                                        jl.emit();
                                    }
                                    match ts_action {
                                        TimestepAction::KeepGoing => {
                                            return Ok((
                                                finalize_accepted_step(
                                                    model,
                                                    rep,
                                                    ts_rep.varcontrol,
                                                    ts_rep.dt_next,
                                                    dt_try,
                                                    rsp2_e_old,
                                                    retries,
                                                    num_splits,
                                                    num_merges,
                                                ),
                                                pending_check_term,
                                            ));
                                        }
                                        TimestepAction::Retry(reason) => {
                                            model.model_number -= 1;
                                            ts_retry_reason = Some(reason);
                                        }
                                        TimestepAction::Terminate(reason) => {
                                            let terminated_model =
                                                model.model_number;
                                            model.model_number -= 1;
                                            restore_attempt_state(
                                                model,
                                                pre_remesh_model.take(),
                                                saved,
                                            )?;
                                            model.dt = dt_prev;
                                            return Err(StarError::Terminated {
                                                model_number: terminated_model,
                                                reason,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Handle outer `prepare_to_redo` (`star/private/evolve.f90:2148-2173`).
        if requested_redo {
            redo_cnt += 1;
            model.num_redos += 1;
            if model.controls.ts_stop.redo_limit > 0
                && redo_cnt > model.controls.ts_stop.redo_limit
            {
                restore_attempt_state(model, pre_remesh_model.take(), saved)?;
                model.dt = dt_prev;
                return Err(StarError::Terminated {
                    model_number: target_mod,
                    reason: String::from("redo_limit"),
                });
            }
            restore_attempt_state(model, None, saved)?;
            first_try = false;
            continue;
        }

        // Fail immediately on non-retriable configuration, I/O, or termination
        // errors without burning retry budget or shrinking `dt`.
        if matches!(&attempt, Err(e) if !e.is_retriable()) {
            restore_attempt_state(model, pre_remesh_model.take(), saved)?;
            model.dt = dt_prev;
            return Err(
                attempt.expect_err("matched non-retriable StarError above")
            );
        }

        // `prepare_to_retry` (`star/private/evolve.f90:2176-2277`).
        let next_retries = retries + 1;
        let hit_retry_limit = max_retries > 0 && next_retries > max_retries;
        let mut next_dt = dt_try * retry_fac;
        let hit_min_dt = !next_dt.is_finite() || next_dt <= dt_floor;
        let tracing = crate::step_trace::enabled(target_mod);
        let why = if tracing || hit_retry_limit || hit_min_dt {
            describe_retry_cause(
                &attempt,
                force_outer_retry,
                ts_retry_reason.as_deref(),
            )
        } else {
            String::new()
        };
        if tracing {
            let mut jl = crate::step_trace::JsonLine::new("star_retry");
            jl.field_usize("model", target_mod);
            jl.field_f64("dt", dt_try);
            jl.field_usize("retries", retries);
            jl.field_str("why", &why);
            jl.emit();
        }

        retries = next_retries;
        if hit_retry_limit {
            restore_attempt_state(model, pre_remesh_model.take(), saved)?;
            model.dt = dt_prev;
            return Err(StarError::StepFailure(Box::new(
                build_exhausted_step_failure(
                    attempt,
                    target_mod,
                    dt_try,
                    next_dt,
                    dt_floor,
                    retries,
                    max_retries,
                    why,
                ),
            )));
        }

        if hit_min_dt {
            restore_attempt_state(model, pre_remesh_model.take(), saved)?;
            model.dt = dt_prev;
            return Err(StarError::Terminated {
                model_number: target_mod,
                reason: String::from("min_timestep_limit"),
            });
        }

        let restore_snap = if model.controls.mix_wind.restore_mesh_on_retry
            && !model.controls.rsp_flag
        {
            pre_remesh_model.take()
        } else {
            None
        };
        if restore_attempt_state(model, restore_snap, saved)? {
            num_splits = 0;
            num_merges = 0;
        }

        if model.controls.max_years_for_timestep > 0.0 {
            model.controls.ts_stop.max_timestep =
                SECYER * model.controls.max_years_for_timestep;
        }
        if model.controls.ts_stop.max_timestep > 0.0
            && next_dt > model.controls.ts_stop.max_timestep
        {
            next_dt = model.controls.ts_stop.max_timestep;
        }

        model.num_retries += 1;
        let max_num_retries = if model.aux.doing_relax {
            model.controls.relax_max_number_retries
        } else {
            model.controls.max_number_retries
        };
        if max_num_retries >= 0
            && (model.num_retries as isize) > max_num_retries
        {
            restore_attempt_state(model, pre_remesh_model.take(), saved)?;
            model.dt = dt_prev;
            return Err(StarError::Terminated {
                model_number: target_mod,
                reason: String::from("max_number_retries"),
            });
        }

        dt_try = next_dt;
        first_try = false;
        model.aux.step_hist.model_number_for_last_retry = Some(target_mod);
        let hold_span = if model.why_tlim_neg_x {
            model
                .controls
                .ts_stop
                .retry_hold
                .max(model.controls.ts_stop.neg_mass_fraction_hold)
        } else {
            model.controls.ts_stop.retry_hold
        };
        model.aux.step_hist.timestep_hold = target_mod + hold_span;
    }
}

fn describe_retry_cause(
    attempt: &Result<crate::solver::HenyeySolverReport, StarError>,
    force_outer_retry: bool,
    ts_retry_reason: Option<&str>,
) -> String {
    match attempt {
        Ok(rep) if rep.converged && force_outer_retry => {
            String::from("implicit wind redo requested outer retry")
        }
        Ok(rep) if rep.converged => ts_retry_reason
            .unwrap_or("timestep controller requested retry")
            .to_owned(),
        Ok(rep) => format!(
            "solver did not converge (iters={}, rmax={:.3e}, cmax={:.3e})",
            rep.iterations, rep.max_residual, rep.max_correction
        ),
        Err(StarError::StepFailure(info)) => info.detail.clone(),
        Err(e) => format!("error: {e}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_exhausted_step_failure(
    attempt: Result<crate::solver::HenyeySolverReport, StarError>,
    target_mod: usize,
    dt_try: f64,
    next_dt: f64,
    dt_floor: f64,
    retries: usize,
    max_retries: usize,
    why: String,
) -> StepFailureInfo {
    let limit_str = if max_retries == 0 {
        String::from("unlimited")
    } else {
        max_retries.to_string()
    };
    let detail = format!(
        "star_evolve_step failed after {retries}/{limit_str} retries \
         (dt={next_dt:.3e} s, dt_floor={dt_floor:.3e} s)"
    );
    match attempt {
        Ok(rep) => StepFailureInfo {
            model_number: target_mod,
            zone: None,
            equation: None,
            variable: None,
            value: None,
            residual_norm: rep.residual_norm,
            max_residual: rep.max_residual,
            correction_norm: rep.correction_norm,
            max_correction: rep.max_correction,
            dt: next_dt,
            iterations: rep.iterations,
            retries,
            retry_reason: why,
            detail,
        },
        Err(StarError::StepFailure(mut info)) => {
            info.model_number = target_mod;
            info.retries = retries;
            info.dt = next_dt;
            info.retry_reason = why;
            info.detail = detail;
            *info
        }
        Err(_) => StepFailureInfo {
            model_number: target_mod,
            zone: None,
            equation: None,
            variable: None,
            value: None,
            residual_norm: 0.0,
            max_residual: 0.0,
            correction_norm: 0.0,
            max_correction: 0.0,
            dt: dt_try.min(next_dt),
            iterations: 0,
            retries,
            retry_reason: why,
            detail,
        },
    }
}

#[derive(Clone, Copy)]
struct PreAttemptScalars {
    dt_old: f64,
    mstar_dot_old: f64,
    dt_limit_ratio_old: f64,
    rot_boost_old: f64,
    was_in_limit: bool,
    time_sec: f64,
    time_old_sec: f64,
    phase_of_evolution: i32,
    using_revised_max_yr_dt: bool,
    revised_max_yr_dt: f64,
    max_years_for_timestep: f64,
    max_timestep: f64,
}

fn sync_time_sec_from_age(model: &mut StarModel) {
    let expected = model.star_age * SECYER;
    let cur = model.aux.step_hist.time_sec;
    let tol = 1.0e-12 * model.star_age.abs().max(1.0);
    if (cur == 0.0 && model.star_age != 0.0)
        || ((cur / SECYER) - model.star_age).abs() > tol
    {
        model.aux.step_hist.time_sec = expected;
        model.aux.step_hist.time_old_sec = expected;
    }
    model.time_sec = model.aux.step_hist.time_sec;
    model.time_old_sec = model.aux.step_hist.time_old_sec;
}

fn is_at_or_past_max_age(model: &StarModel) -> bool {
    let time_sec = model.aux.step_hist.time_sec;
    let ts = &model.controls.ts_stop;
    (model.controls.max_age > 0.0
        && time_sec >= model.controls.max_age * SECYER)
        || (ts.max_age_in_days > 0.0 && time_sec >= ts.max_age_in_days * SECDAY)
        || model
            .controls
            .max_age_in_seconds
            .is_some_and(|lim| lim > 0.0 && time_sec >= lim)
}

fn restore_attempt_state(
    model: &mut StarModel,
    snap: Option<crate::PrimaryStateBuffer>,
    saved: PreAttemptScalars,
) -> Result<bool, StarError> {
    let restored_remesh = if let Some(buf) = snap {
        buf.restore_into(model)?;
        snapshot_step_old_extras(model);
        model.save_old_values();
        record_start_of_step_energy_integrals(model);
        snapshot_step_start_extras(model);
        model.aux.okay_to_set_mlt_vc = true;
        model.aux.step_hist.mstar_dot_old = saved.mstar_dot_old;
        true
    } else {
        restore_from_old(model)?;
        false
    };
    model.dt_old = saved.dt_old;
    model.dt_limit_ratio_old = saved.dt_limit_ratio_old;
    model.aux.step_hist.rotational_mdot_boost_old = saved.rot_boost_old;
    model.aux.step_hist.was_in_implicit_wind_limit = saved.was_in_limit;
    model.aux.step_hist.time_sec = saved.time_sec;
    model.aux.step_hist.time_old_sec = saved.time_old_sec;
    model.time_sec = saved.time_sec;
    model.time_old_sec = saved.time_old_sec;
    model.aux.step_hist.phase_of_evolution = saved.phase_of_evolution;
    model.aux.step_hist.using_revised_max_yr_dt = saved.using_revised_max_yr_dt;
    model.aux.step_hist.revised_max_yr_dt = saved.revised_max_yr_dt;
    model.controls.max_years_for_timestep = saved.max_years_for_timestep;
    model.controls.ts_stop.max_timestep = saved.max_timestep;
    Ok(restored_remesh)
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
    if model.controls.max_years_for_timestep > 0.0 {
        dt_next = dt_next.min(model.controls.max_years_for_timestep * SECYER);
    }
    if model.controls.rsp.max_timestep > 0.0 {
        dt_next = dt_next.min(model.controls.rsp.max_timestep);
    }

    if model.controls.rsp2_flag {
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
        model.aux.rsp.time_sec += dt_try;
        model.aux.rsp.total_energy_old = rsp2_e_old;
        update_rsp2_cycle_stats(model, dt_try, &mut dt_next);
    }
    apply_max_age_to_dt_next(model, &mut dt_next);
    model.dt_next = dt_next;
    model.dt_old = dt_try;
    model.aux.step_hist.mstar_dot_old = model.mstar_dot;
    model.aux.have_mlt_vc = true;
    model.doing_first_model_of_run = false;

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

/// Applies `pick_next_timestep` `max_age` clipping and graded timestep approach
/// (`star/private/evolve.f90:2079-2143`).
fn apply_max_age_to_dt_next(model: &mut StarModel, dt_next: &mut f64) {
    let time_sec = model.aux.step_hist.time_sec;
    let max_age = model.controls.max_age;
    let max_days = model.controls.ts_stop.max_age_in_days;
    let max_sec = model.controls.max_age_in_seconds;
    if time_sec < 0.0 && time_sec + *dt_next > 0.0 {
        *dt_next = -time_sec;
    } else if max_age > 0.0 && time_sec + *dt_next > max_age * SECYER {
        *dt_next = (max_age * SECYER - time_sec).max(0.0);
    } else if max_days > 0.0 && time_sec + *dt_next > max_days * SECDAY {
        *dt_next = (max_days * SECDAY - time_sec).max(0.0);
    } else if max_sec.is_some_and(|lim| lim > 0.0 && time_sec + *dt_next > lim)
    {
        if let Some(lim) = max_sec {
            *dt_next = (lim - time_sec).max(0.0);
        }
    } else if model.controls.ts_stop.num_adjusted_dt_steps_before_max_age > 0
        && model.controls.max_years_for_timestep > 0.0
    {
        let remaining_years = if max_age > 0.0 {
            max_age - model.star_age
        } else if max_days > 0.0 {
            (max_days * SECDAY - time_sec) / SECYER
        } else if let Some(lim) = max_sec.filter(|&v| v > 0.0) {
            (lim - time_sec) / SECYER
        } else {
            1.0e99
        };
        if model.aux.step_hist.using_revised_max_yr_dt {
            model.controls.max_years_for_timestep =
                model.aux.step_hist.revised_max_yr_dt;
        }
        let cur_max_yr = model.controls.max_years_for_timestep;
        let n_f = (remaining_years / cur_max_yr + 1.0e-6).floor();
        let j_f =
            model.controls.ts_stop.num_adjusted_dt_steps_before_max_age as f64;
        if remaining_years <= cur_max_yr {
            model.controls.max_years_for_timestep = remaining_years;
            model.aux.step_hist.using_revised_max_yr_dt = true;
            model.aux.step_hist.revised_max_yr_dt = remaining_years;
            *dt_next = remaining_years * SECYER;
        } else if n_f <= j_f {
            let prev_max_yr = cur_max_yr;
            let dt_y = model.controls.ts_stop.dt_years_for_steps_before_max_age;
            let mut new_max_yr = if dt_y > 0.0 {
                let i_f = (remaining_years / dt_y + 1.0e-6).floor();
                if (i_f + 1.0e-9) * dt_y < remaining_years {
                    remaining_years / (i_f + 1.0)
                } else if i_f > 0.0 {
                    remaining_years / i_f
                } else {
                    remaining_years
                }
            } else {
                prev_max_yr
            };
            let min_max = prev_max_yr
                * model.controls.ts_stop.reduction_factor_for_max_timestep;
            if new_max_yr < min_max {
                new_max_yr = min_max;
            }
            model.controls.max_years_for_timestep = new_max_yr;
            model.aux.step_hist.using_revised_max_yr_dt = true;
            model.aux.step_hist.revised_max_yr_dt = new_max_yr;
            if *dt_next / SECYER > new_max_yr {
                *dt_next = new_max_yr * SECYER;
            }
        }
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
        sync_time_sec_from_age(model);
        if let Some(max_mod) = model.controls.max_model_number {
            if model.model_number >= max_mod {
                break;
            }
        }
        if is_at_or_past_max_age(model) {
            break;
        }
        let rep = match star_evolve_step(
            model,
            do_remesh,
            op_split_burn,
            use_bcyclic,
        ) {
            Ok(rep) => rep,
            Err(StarError::Terminated { .. }) => break,
            Err(e) => return Err(e),
        };
        reports.push(rep);
        if model.aux.step_hist.termination_code.is_some()
            || matches!(
                do_check_limits(model, None, false)?,
                LimitCheckOutcome::Terminate { .. }
            )
        {
            break;
        }
    }
    Ok(reports)
}
