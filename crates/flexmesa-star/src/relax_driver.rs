// SPDX-License-Identifier: GPL-3.0-or-later
//! The relax driver `do_internal_evolve` (`star/private/relax.f90:3905-4286`)
//! and the relaxes built on it: `do_relax_num_steps`
//! (`star/private/relax.f90:3323-3394`) and `do_relax_to_radiative_core`
//! (`star/private/relax.f90:3421-3541`).
//!
//! A relax evolves the model with the ordinary step machinery while a set of
//! controls is overridden, then restores those controls and the clock. Each
//! relax supplies `before_evolve`, `adjust_model`, `check_model` and
//! `finish_model` through [`RelaxHooks`].

use flexmesa_const::{M_SUN, RSUN, SECYER};
use flexmesa_turb::MltOption;

use crate::controls::StarError;
use crate::evolve::{star_evolve_step_outcome_checked, StepChecks};
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;

/// Termination code of a relax that reached its goal
/// (`termination_code_str(t_relax_finished_okay)`,
/// `star/private/star_private_def.f90:205`).
pub(crate) const RELAX_FINISHED_OKAY: &str = "relax_finished_okay";

/// Termination code set by `extras_check_model`
/// (`termination_code_str(t_extras_check_model)`,
/// `star/private/star_private_def.f90:212`).
pub(crate) const EXTRAS_CHECK_MODEL: &str = "extras_check_model";

/// Result of a relax hook. Upstream hooks may also return `retry` or `redo`;
/// none of the relaxes ported here does, so those are not represented.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RelaxHookResult {
    /// `keep_going`.
    KeepGoing,
    /// `terminate`, with the `termination_code_str` the hook set.
    Terminate(String),
}

/// The procedure arguments of `do_internal_evolve`
/// (`star/private/relax.f90:3905-3950`).
pub(crate) trait RelaxHooks {
    /// `before_evolve`, called after the control overrides and before the
    /// first step.
    ///
    /// # Errors
    /// Returns [`StarError`] for `ierr /= 0`.
    fn before_evolve(&mut self, model: &mut StarModel)
        -> Result<(), StarError>;

    /// `adjust_model`, called between `do_evolve_step_part1` and
    /// `do_evolve_step_part2`.
    ///
    /// # Errors
    /// Returns [`StarError`] when the hook fails.
    fn adjust_model(
        &mut self,
        _model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        Ok(RelaxHookResult::KeepGoing)
    }

    /// `check_model`, called after `do_evolve_step_part2` and before
    /// `pick_next_timestep`.
    ///
    /// # Errors
    /// Returns [`StarError`] when the hook fails.
    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError>;

    /// `finish_model`, called after the step loop accepts a step; the relaxes
    /// ported here pass `null_finish_model` (`star/private/relax.f90:4290`).
    ///
    /// # Errors
    /// Returns [`StarError`] when the hook fails.
    fn finish_model(
        &mut self,
        _model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// The values `save_stuff` stores and `restore_stuff` puts back
/// (`star/private/relax.f90:4173-4283`). `warning_limit_for_max_residual` and
/// `recent_log_header` only affect terminal output, which `FlexMESA` does not
/// write (`DECISIONS #26`), so they are not kept.
#[derive(Clone)]
struct SavedStuff {
    do_history_file: bool,
    write_profiles_flag: bool,
    mass_change: f64,
    reimers_scaling_factor: f64,
    blocker_scaling_factor: f64,
    de_jager_scaling_factor: f64,
    van_loon_scaling_factor: f64,
    nieuwenhuijzen_scaling_factor: f64,
    vink_scaling_factor: f64,
    dutch_scaling_factor: f64,
    bjorklund_scaling_factor: f64,
    use_other_wind: bool,
    num_retries: usize,
    star_age: f64,
    time_sec: f64,
    hist_time_sec: f64,
    model_number: usize,
    dxdt_nuc_factor: f64,
    max_age: f64,
    max_age_in_days: f64,
    max_age_in_seconds: Option<f64>,
    max_timestep_factor: f64,
    varcontrol_target: f64,
    max_timestep: f64,
    model_number_for_last_retry: Option<usize>,
    photo_interval: isize,
    profile_interval: usize,
    priority_profile_interval: usize,
    dt_next: f64,
    max_number_retries: isize,
    mlt_option: MltOption,
    use_gold2_tolerances: bool,
    steps_before_use_gold2_tolerances: isize,
    use_gold_tolerances: bool,
    steps_before_use_gold_tolerances: isize,
    solver_iters_timestep_limit: isize,
    tol_correction_norm: f64,
    tol_max_correction: f64,
    iter_for_resid_tol2: usize,
    tol_residual_norm1: f64,
    tol_max_residual1: f64,
    iter_for_resid_tol3: usize,
    tol_residual_norm2: f64,
    tol_max_residual2: f64,
    tol_residual_norm3: f64,
    tol_max_residual3: f64,
    max_t_for_gold_tolerances: f64,
    // "selected history"
    star_age_old: f64,
    time_old_sec: f64,
    hist_time_old_sec: f64,
    model_number_old: usize,
}

impl SavedStuff {
    /// `save_stuff` (`star/private/relax.f90:4173-4232`).
    fn save(model: &StarModel) -> Self {
        let c = &model.controls;
        let mw = &c.mix_wind;
        let ts = &c.ts_stop;
        Self {
            do_history_file: c.output.do_history_file,
            write_profiles_flag: c.output.write_profiles_flag,
            mass_change: c.mass_change,
            reimers_scaling_factor: c.reimers_scaling_factor,
            blocker_scaling_factor: c.blocker_scaling_factor,
            de_jager_scaling_factor: mw.de_jager_scaling_factor,
            van_loon_scaling_factor: mw.van_loon_scaling_factor,
            nieuwenhuijzen_scaling_factor: mw.nieuwenhuijzen_scaling_factor,
            vink_scaling_factor: c.vink_scaling_factor,
            dutch_scaling_factor: c.dutch_scaling_factor,
            bjorklund_scaling_factor: mw.bjorklund_scaling_factor,
            use_other_wind: c.use_other_wind,
            num_retries: model.num_retries,
            star_age: model.star_age,
            time_sec: model.time_sec,
            hist_time_sec: model.aux.step_hist.time_sec,
            model_number: model.model_number,
            dxdt_nuc_factor: c.dxdt_nuc_factor,
            max_age: c.max_age,
            max_age_in_days: ts.max_age_in_days,
            max_age_in_seconds: c.max_age_in_seconds,
            max_timestep_factor: ts.max_timestep_factor,
            varcontrol_target: c.varcontrol_target,
            max_timestep: ts.max_timestep,
            model_number_for_last_retry: model
                .aux
                .step_hist
                .model_number_for_last_retry,
            photo_interval: c.output.photo_interval,
            profile_interval: c.output.profile_interval,
            priority_profile_interval: c.output.priority_profile_interval,
            dt_next: model.dt_next,
            max_number_retries: c.max_number_retries,
            mlt_option: c.mlt_option,
            use_gold2_tolerances: c.use_gold2_tolerances,
            steps_before_use_gold2_tolerances: c
                .steps_before_use_gold2_tolerances,
            use_gold_tolerances: c.use_gold_tolerances,
            steps_before_use_gold_tolerances: c
                .steps_before_use_gold_tolerances,
            solver_iters_timestep_limit: ts.solver_iters_timestep_limit,
            tol_correction_norm: c.tol_correction_norm,
            tol_max_correction: c.tol_max_correction,
            iter_for_resid_tol2: c.iter_for_resid_tol2,
            tol_residual_norm1: c.tol_residual_norm1,
            tol_max_residual1: c.tol_max_residual1,
            iter_for_resid_tol3: c.iter_for_resid_tol3,
            tol_residual_norm2: c.tol_residual_norm2,
            tol_max_residual2: c.tol_max_residual2,
            tol_residual_norm3: c.tol_residual_norm3,
            tol_max_residual3: c.tol_max_residual3,
            max_t_for_gold_tolerances: c.max_t_for_gold_tolerances,
            star_age_old: model.star_age_old,
            time_old_sec: model.time_old_sec,
            hist_time_old_sec: model.aux.step_hist.time_old_sec,
            model_number_old: model.model_number_old,
        }
    }

    /// `restore_stuff` (`star/private/relax.f90:4234-4283`).
    fn restore(self, model: &mut StarModel) {
        let c = &mut model.controls;
        c.output.do_history_file = self.do_history_file;
        c.output.write_profiles_flag = self.write_profiles_flag;
        c.mass_change = self.mass_change;
        c.reimers_scaling_factor = self.reimers_scaling_factor;
        c.blocker_scaling_factor = self.blocker_scaling_factor;
        c.mix_wind.de_jager_scaling_factor = self.de_jager_scaling_factor;
        c.mix_wind.van_loon_scaling_factor = self.van_loon_scaling_factor;
        c.mix_wind.nieuwenhuijzen_scaling_factor =
            self.nieuwenhuijzen_scaling_factor;
        c.vink_scaling_factor = self.vink_scaling_factor;
        c.dutch_scaling_factor = self.dutch_scaling_factor;
        c.mix_wind.bjorklund_scaling_factor = self.bjorklund_scaling_factor;
        c.use_other_wind = self.use_other_wind;
        model.num_retries = self.num_retries;
        model.star_age = self.star_age;
        model.time_sec = self.time_sec;
        model.aux.step_hist.time_sec = self.hist_time_sec;
        model.model_number = self.model_number;
        c.dxdt_nuc_factor = self.dxdt_nuc_factor;
        c.max_age = self.max_age;
        c.ts_stop.max_age_in_days = self.max_age_in_days;
        c.max_age_in_seconds = self.max_age_in_seconds;
        c.ts_stop.max_timestep_factor = self.max_timestep_factor;
        c.varcontrol_target = self.varcontrol_target;
        c.ts_stop.max_timestep = self.max_timestep;
        model.aux.step_hist.model_number_for_last_retry =
            self.model_number_for_last_retry;
        c.output.photo_interval = self.photo_interval;
        c.output.profile_interval = self.profile_interval;
        c.output.priority_profile_interval = self.priority_profile_interval;
        model.dt_next = self.dt_next;
        c.max_number_retries = self.max_number_retries;
        c.mlt_option = self.mlt_option;
        c.use_gold2_tolerances = self.use_gold2_tolerances;
        c.steps_before_use_gold2_tolerances =
            self.steps_before_use_gold2_tolerances;
        c.use_gold_tolerances = self.use_gold_tolerances;
        c.steps_before_use_gold_tolerances =
            self.steps_before_use_gold_tolerances;
        c.ts_stop.solver_iters_timestep_limit =
            self.solver_iters_timestep_limit;
        c.tol_correction_norm = self.tol_correction_norm;
        c.tol_max_correction = self.tol_max_correction;
        c.iter_for_resid_tol2 = self.iter_for_resid_tol2;
        c.tol_residual_norm1 = self.tol_residual_norm1;
        c.tol_max_residual1 = self.tol_max_residual1;
        c.iter_for_resid_tol3 = self.iter_for_resid_tol3;
        c.tol_residual_norm2 = self.tol_residual_norm2;
        c.tol_max_residual2 = self.tol_max_residual2;
        c.tol_residual_norm3 = self.tol_residual_norm3;
        c.tol_max_residual3 = self.tol_max_residual3;
        c.max_t_for_gold_tolerances = self.max_t_for_gold_tolerances;
        model.star_age_old = self.star_age_old;
        model.time_old_sec = self.time_old_sec;
        model.aux.step_hist.time_old_sec = self.hist_time_old_sec;
        model.model_number_old = self.model_number_old;
    }
}

/// Applies the control overrides at the start of `do_internal_evolve`
/// (`star/private/relax.f90:3976-4022`).
fn override_controls_for_relax(model: &mut StarModel) {
    let c = &mut model.controls;
    c.output.do_history_file = false;
    c.output.write_profiles_flag = false;
    c.max_number_retries = c.relax_max_number_retries;
    c.use_gold_tolerances = c.relax_use_gold_tolerances;
    c.steps_before_use_gold_tolerances = -1;
    c.use_gold2_tolerances = false;
    c.steps_before_use_gold2_tolerances = -1;
    if c.mlt_option == MltOption::Tdc {
        c.mlt_option = MltOption::Cox;
    }

    if c.relax_solver_iters_timestep_limit != 0 {
        c.ts_stop.solver_iters_timestep_limit =
            c.relax_solver_iters_timestep_limit;
    }

    if c.relax_tol_correction_norm != 0.0 {
        c.tol_correction_norm = c.relax_tol_correction_norm;
    }
    if c.relax_tol_max_correction != 0.0 {
        c.tol_max_correction = c.relax_tol_max_correction;
    }

    if c.relax_iter_for_resid_tol2 != 0 {
        c.iter_for_resid_tol2 = c.relax_iter_for_resid_tol2;
    }
    if c.relax_tol_residual_norm1 != 0.0 {
        c.tol_residual_norm1 = c.relax_tol_residual_norm1;
    }
    if c.relax_tol_max_residual1 != 0.0 {
        c.tol_max_residual1 = c.relax_tol_max_residual1;
    }

    if c.relax_iter_for_resid_tol3 != 0 {
        c.iter_for_resid_tol3 = c.relax_iter_for_resid_tol3;
    }
    if c.relax_tol_residual_norm2 != 0.0 {
        c.tol_residual_norm2 = c.relax_tol_residual_norm2;
    }
    if c.relax_tol_max_residual2 != 0.0 {
        c.tol_max_residual2 = c.relax_tol_max_residual2;
    }

    if c.relax_tol_residual_norm3 != 0.0 {
        c.tol_residual_norm3 = c.relax_tol_residual_norm3;
    }
    if c.relax_tol_max_residual3 != 0.0 {
        c.tol_max_residual3 = c.relax_tol_max_residual3;
    }

    if c.relax_maxt_for_gold_tolerances != 0.0 {
        c.max_t_for_gold_tolerances = c.relax_maxt_for_gold_tolerances;
    }
}

/// `setup_before_relax` (`star/private/relax.f90:3878-3887`) with
/// `turn_off_winds` (`star/private/relax.f90:3890-3902`).
pub(crate) fn setup_before_relax(model: &mut StarModel) {
    let c = &mut model.controls;
    c.dxdt_nuc_factor = 0.0;
    c.max_age = 1.0e50;
    c.ts_stop.max_age_in_days = 1.0e50;
    c.max_age_in_seconds = Some(1.0e50);
    c.ts_stop.max_timestep_factor = 2.0;
    // `max_model_number = -1111`: no limit.
    c.max_model_number = None;
    turn_off_winds(model);
}

/// `turn_off_winds` (`star/private/relax.f90:3890-3902`).
pub(crate) fn turn_off_winds(model: &mut StarModel) {
    let c = &mut model.controls;
    c.mass_change = 0.0;
    c.reimers_scaling_factor = 0.0;
    c.blocker_scaling_factor = 0.0;
    c.mix_wind.de_jager_scaling_factor = 0.0;
    c.mix_wind.van_loon_scaling_factor = 0.0;
    c.mix_wind.nieuwenhuijzen_scaling_factor = 0.0;
    c.vink_scaling_factor = 0.0;
    c.dutch_scaling_factor = 0.0;
    c.mix_wind.bjorklund_scaling_factor = 0.0;
    c.use_other_wind = false;
}

/// Per-step refresh the single-star run loop applies after every accepted
/// step (`inlist_runner::run_bundle_steps`), so a relax step leaves the model
/// in the same state a run step does.
fn refresh_after_accepted_step(model: &mut StarModel) -> Result<(), StarError> {
    model.mstar_start = model.mstar;
    model.sync_mesh_and_pack()?;
    crate::hydro_vars::set_hydro_vars(model, false, false)?;
    crate::struct_burn_mix::refresh_mixing_state(model, model.dt, &[])?;
    model.save_start_values()?;
    crate::evolve_support::snapshot_step_start_extras(model);
    Ok(())
}

/// `do_internal_evolve` (`star/private/relax.f90:3905-4166`): evolves `model`
/// under the relax control overrides until a hook terminates the loop, then
/// restores the saved controls and clock when `restore_at_end` is set.
///
/// # Errors
/// Returns [`StarError`] when `before_evolve` fails, a step fails, or the
/// loop ends with a termination code other than `relax_finished_okay` or
/// `extras_check_model` (`ierr = -1` at `star/private/relax.f90:4144-4145`).
pub(crate) fn do_internal_evolve(
    model: &mut StarModel,
    hooks: &mut dyn RelaxHooks,
    restore_at_end: bool,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let saved = SavedStuff::save(model);
    override_controls_for_relax(model);

    if model.doing_first_model_of_run {
        model.num_retries = 0;
        model.time_sec = 0.0;
        model.time_old_sec = 0.0;
        model.aux.step_hist.time_sec = 0.0;
        model.aux.step_hist.time_old_sec = 0.0;
        model.star_age = 0.0;
        model.star_age_old = 0.0;
        model.aux.step_hist.model_number_for_last_retry = Some(0);
        model.controls.output.photo_interval = 0;
        model.controls.output.profile_interval = 0;
        model.controls.output.priority_profile_interval = 0;
    }
    // `pgstar_flag` (`star/private/relax.f90:4038-4050`): `FlexMESA` runs
    // headless (`DECISIONS #25`).

    // Upstream returns `ierr` here without `restore_stuff`
    // (`star/private/relax.f90:4052-4056`).
    hooks.before_evolve(model)?;

    model.aux.step_hist.termination_code = None;
    model.aux.doing_relax = true;
    // `need_to_setvars = .true.`: every step starts by setting the variables
    // (`star/private/evolve.f90:319`).

    let do_remesh =
        model.controls.okay_to_remesh && model.controls.mesh_delta_coeff > 0.0;
    let op_split_burn = model.controls.op_split_burn;
    let mut termination: Option<String> = None;
    let mut step_error: Option<StarError> = None;
    // `other_cgrav` and `other_alpha_mlt` run inside `set_hydro_vars`
    // (`star/private/hydro_vars.f90:525,572-579,763-776`), not once per step.
    crate::inlist_runner::install_star_aux_hooks(&*extras, model);
    loop {
        match star_evolve_step_outcome_checked(
            model,
            do_remesh,
            op_split_burn,
            true,
            extras,
            StepChecks::Relax(&mut *hooks),
        ) {
            Ok((_, Some(code))) => {
                termination = Some(code);
                break;
            }
            Ok((_, None)) => {}
            // `prepare_to_retry`/`prepare_to_redo` returned `terminate`
            // (`star/private/evolve.f90:2148-2277`).
            Err(StarError::Terminated { reason, .. }) => {
                termination = Some(reason);
                break;
            }
            Err(e) => {
                step_error = Some(e);
                break;
            }
        }
        if let Err(e) = refresh_after_accepted_step(model) {
            step_error = Some(e);
            break;
        }

        match hooks.finish_model(model) {
            Ok(RelaxHookResult::KeepGoing) => {}
            Ok(RelaxHookResult::Terminate(code)) => {
                termination = Some(code);
                break;
            }
            Err(e) => {
                step_error = Some(e);
                break;
            }
        }
        // `finish_step` (`star/private/evolve.f90:2287-2387`) with history
        // and profiles off and no photo while relaxing: its remaining effect,
        // `doing_first_model_of_run = .false.`, is applied when the step is
        // accepted. `s% finish_relax_step` is not a `StarExtras` hook.
    }

    model.aux.doing_relax = false;
    // Matches `star/private/relax.f90:4141-4142`.
    model.aux.need_to_setvars = true;
    model.aux.st_start.have_st_start_info = false;
    let finished_okay = matches!(
        termination.as_deref(),
        Some(RELAX_FINISHED_OKAY | EXTRAS_CHECK_MODEL)
    );
    model.aux.step_hist.termination_code = None;

    if restore_at_end {
        saved.restore(model);
    }

    if model.job.set_cumulative_energy_error_each_relax {
        model.aux.cumulative_energy_error =
            model.job.new_cumulative_energy_error;
    }

    model.dt = 0.0;
    model.dt_old = 0.0;
    // `timestep_hold = -100` and `model_number_for_last_retry = -100`
    // (`star/private/relax.f90:4159-4160`): no hold and no retry, which this
    // port stores as `0` and `None`.
    model.aux.step_hist.timestep_hold = 0;
    model.aux.step_hist.model_number_for_last_retry = None;

    if let Some(e) = step_error {
        return Err(e);
    }
    if finished_okay {
        Ok(())
    } else {
        Err(StarError::ConvergenceFailure(format!(
            "relax ended with termination code '{}' instead of \
             relax_finished_okay (star/private/relax.f90:4144-4145)",
            termination.as_deref().unwrap_or("none")
        )))
    }
}

/// `before_evolve_relax_num_steps` and `relax_num_steps_check_model`
/// (`star/private/relax.f90:3377-3419`).
struct RelaxNumSteps {
    num_steps: usize,
    max_timestep: f64,
}

impl RelaxHooks for RelaxNumSteps {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        let c = &mut model.controls;
        c.ts_stop.max_timestep = self.max_timestep;
        c.max_years_for_timestep = c.ts_stop.max_timestep / SECYER;
        model.dt_next = model.controls.ts_stop.max_timestep;
        model.controls.max_model_number = Some(self.num_steps);
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        // `do_bare_bones_check_model` only gathers history values.
        if model.model_number >= self.num_steps {
            return Ok(RelaxHookResult::Terminate(String::from(
                RELAX_FINISHED_OKAY,
            )));
        }
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_num_steps` (`star/private/relax.f90:3323-3374`): takes
/// `num_steps` steps with `dxdt_nuc_factor = 0`, winds off and the timestep
/// held at or below `max_timestep` (seconds; one year when `<= 0`), then
/// restores the controls, clock and `model_number`.
///
/// # Errors
/// Returns [`StarError`] when the relax fails.
pub(crate) fn do_relax_num_steps(
    model: &mut StarModel,
    num_steps: isize,
    max_timestep: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let Ok(num_steps) = usize::try_from(num_steps) else {
        return Ok(());
    };
    if num_steps == 0 {
        return Ok(());
    }
    let max_timestep = if max_timestep <= 0.0 {
        SECYER
    } else {
        max_timestep
    };
    let max_model_number = model.controls.max_model_number;
    let model_number = model.model_number;
    let save_max_timestep = model.controls.ts_stop.max_timestep;
    let save_max_years_for_timestep = model.controls.max_years_for_timestep;
    let save_use_grad_t = model
        .controls
        .use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn;
    model
        .controls
        .use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn = false;
    model.model_number = 0;
    let mut hooks = RelaxNumSteps {
        num_steps,
        max_timestep,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    model.model_number = model_number;
    model.controls.ts_stop.max_timestep = save_max_timestep;
    model.controls.max_years_for_timestep = save_max_years_for_timestep;
    model
        .controls
        .use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn = save_use_grad_t;
    res
}

/// `before_evolve_relax_to_radiative_core` and
/// `relax_to_radiative_core_check_model`
/// (`star/private/relax.f90:3475-3541`).
struct RelaxToRadiativeCore {
    /// `rpar(2)`, the smallest `conv_mx1_bot` seen so far.
    min_conv_mx1_bot: f64,
}

impl RelaxHooks for RelaxToRadiativeCore {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        // The `max_timestep` settings are commented out upstream
        // (`star/private/relax.f90:3484-3486`); reproduced, deferred #415.
        setup_before_relax(model);
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        // `do_bare_bones_check_model` (`get_history_info`) and
        // `set_power_info` (`star/private/report.f90`).
        let ctx = crate::report_utils::StepReportContext::compute(model);
        let job = &model.job;
        if ctx.powers.power_nuc_burn / ctx.phot.l_lsun
            > job.pre_ms_check_radiative_core_lnuc_div_l_limit
        {
            return Ok(RelaxHookResult::Terminate(String::from(
                RELAX_FINISHED_OKAY,
            )));
        }
        let conv_mx1_bot = ctx.mix.conv_mx_q[1];
        if conv_mx1_bot < self.min_conv_mx1_bot {
            self.min_conv_mx1_bot = conv_mx1_bot;
        }
        if self.min_conv_mx1_bot < job.pre_ms_check_radiative_core_start
            && conv_mx1_bot > job.pre_ms_check_radiative_core_stop
        {
            return Ok(RelaxHookResult::Terminate(String::from(
                RELAX_FINISHED_OKAY,
            )));
        }
        Ok(RelaxHookResult::KeepGoing)
    }
}

/// `do_relax_to_radiative_core` (`star/private/relax.f90:3421-3473`): relaxes
/// a pre-main-sequence model until a radiative core starts to form or
/// $L_{\mathrm{nuc}}/L_{\mathrm{phot}}$ exceeds
/// `pre_ms_check_radiative_core_Lnuc_div_L_limit`. Skipped below
/// `pre_ms_check_radiative_core_min_mass`.
///
/// # Errors
/// Returns [`StarError`] when the relax fails.
pub(crate) fn do_relax_to_radiative_core(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if model.mstar / M_SUN < model.job.pre_ms_check_radiative_core_min_mass {
        return Ok(());
    }
    // `max_timestep = 1d3*secyer` fills `rpar(1)`, which the hooks never
    // read (`star/private/relax.f90:3447-3457`).
    let max_model_number = model.controls.max_model_number;
    let model_number = model.model_number;
    let save_max_timestep = model.controls.ts_stop.max_timestep;
    let save_max_years_for_timestep = model.controls.max_years_for_timestep;
    model.model_number = 0;
    let mut hooks = RelaxToRadiativeCore {
        min_conv_mx1_bot: 1.0e99,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);
    model.controls.max_model_number = max_model_number;
    model.model_number = model_number;
    model.controls.ts_stop.max_timestep = save_max_timestep;
    model.controls.max_years_for_timestep = save_max_years_for_timestep;
    res
}

/// `kind_of_relax` of `do_relax_uniform_omega`
/// (`star/private/star_private_def.f90`: `relax_to_new_omega`,
/// `relax_to_new_omega_div_omega_crit`, `relax_to_new_surface_rotation_v`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelaxOmegaKind {
    /// Target $\Omega$ in $\mathrm{rad\,s^{-1}}$.
    NewOmega,
    /// Target $\Omega/\Omega_{\mathrm{crit,avg\,surf}}$.
    NewOmegaDivOmegaCrit,
    /// Target surface rotation velocity in $\mathrm{km\,s^{-1}}$.
    NewSurfaceRotationV,
}

/// `before_evolve_relax_omega`, `relax_omega_adjust_model` and
/// `relax_omega_check_model` (`star/private/relax.f90:1313-1391`).
struct RelaxOmega {
    kind: RelaxOmegaKind,
    /// `rpar(1)`.
    target_value: f64,
    /// `rpar(2)`, the surface $\Omega$ when the relax starts.
    starting_omega: f64,
    /// `ipar(1)`.
    starting_model_number: usize,
    /// `ipar(2)`.
    num_steps_to_use: usize,
}

impl RelaxHooks for RelaxOmega {
    fn before_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        setup_before_relax(model);
        Ok(())
    }

    fn check_model(
        &mut self,
        model: &mut StarModel,
    ) -> Result<RelaxHookResult, StarError> {
        // `do_bare_bones_check_model` only gathers history values.
        let since_start =
            model.model_number as f64 - self.starting_model_number as f64;
        let frac = 0.0_f64
            .max(1.0_f64.min(since_start / self.num_steps_to_use as f64));
        let new_omega = match self.kind {
            RelaxOmegaKind::NewOmega => self.target_value,
            RelaxOmegaKind::NewOmegaDivOmegaCrit => {
                crate::hydro_rotation::set_i_rot(model, false);
                let surf =
                    crate::report_utils::eval_surf_avg_rotation_info(model);
                self.target_value * surf.omega_crit
            }
            RelaxOmegaKind::NewSurfaceRotationV => {
                let phot = crate::report_utils::eval_photosphere_info(model);
                self.target_value * 1.0e5 / (phot.r_rsun * RSUN)
            }
        };
        let this_step_omega =
            frac * new_omega + (1.0 - frac) * self.starting_omega;
        let omega_surf = model.omega[0];
        let result = if model.model_number
            > self.starting_model_number + self.num_steps_to_use + 50
            || (omega_surf - new_omega).abs()
                < 1.0e-4 * 1.0e-10_f64.max(new_omega.abs())
        {
            RelaxHookResult::Terminate(String::from(RELAX_FINISHED_OKAY))
        } else {
            RelaxHookResult::KeepGoing
        };
        // `set_uniform_omega` runs on the final step as well
        // (`star/private/relax.f90:1383-1389`).
        crate::relax::apply_uniform_rotation(model, this_step_omega);
        Ok(result)
    }
}

/// `do_relax_uniform_omega` (`star/private/relax.f90:1234-1310`): relaxes a
/// rotating model to a uniform $\Omega$ over `num_steps_to_relax_rotation`
/// steps with nuclear burning, mixing and angular-momentum mixing off, winds
/// off (`setup_before_relax`), and the timestep at most
/// `relax_omega_max_yrs_dt` years, setting a uniform $\Omega$ ramped
/// linearly from the starting surface value after every step.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for `num_steps_to_relax_rotation <= 0`
/// (`ierr = -1`, `relax.f90:1279-1283`) and the [`StarError`] of a failed
/// relax.
pub(crate) fn do_relax_uniform_omega(
    model: &mut StarModel,
    kind: RelaxOmegaKind,
    target_value: f64,
    num_steps_to_relax_rotation: isize,
    relax_omega_max_yrs_dt: f64,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    if !model.controls.rotation_flag {
        return Ok(());
    }
    if model.omega[..model.nz].iter().all(|&w| w == target_value) {
        return Ok(());
    }
    let starting_omega = model.omega[0];
    let starting_model_number = model.model_number;
    let num_steps_to_use = match usize::try_from(num_steps_to_relax_rotation) {
        Ok(n) if n > 0 => n,
        _ => {
            return Err(StarError::InvalidInput(format!(
                "invalid num_steps_to_relax_rotation {num_steps_to_relax_rotation} \
                 (star/private/relax.f90:1279-1283)"
            )));
        }
    };

    let c = &mut model.controls;
    let dxdt_nuc_factor = c.dxdt_nuc_factor;
    c.dxdt_nuc_factor = 0.0;
    let mix_factor = c.mix_factor;
    let am_d_mix_factor = c.mix_wind.am_d_mix_factor;
    c.mix_factor = 0.0;
    c.mix_wind.am_d_mix_factor = 0.0;
    let max_model_number = c.max_model_number;
    // `setup_before_relax` then sets `max_model_number = -1111` in
    // `before_evolve`, as upstream (`relax.f90:1292, 1320`, `:3885`).
    c.max_model_number = Some(num_steps_to_use + 1);
    let max_years_for_timestep = c.max_years_for_timestep;
    c.max_years_for_timestep = relax_omega_max_yrs_dt;

    let mut hooks = RelaxOmega {
        kind,
        target_value,
        starting_omega,
        starting_model_number,
        num_steps_to_use,
    };
    let res = do_internal_evolve(model, &mut hooks, true, extras);

    let c = &mut model.controls;
    c.dxdt_nuc_factor = dxdt_nuc_factor;
    c.mix_factor = mix_factor;
    c.mix_wind.am_d_mix_factor = am_d_mix_factor;
    c.max_model_number = max_model_number;
    c.max_years_for_timestep = max_years_for_timestep;
    // `s% D_omega(1:nz) = 0; s% am_nu_rot(1:nz) = 0` (`relax.f90:1306-1307`);
    // `D_omega` (`D_omega_flag`) is not ported.
    model.aux.am_nu_rot.fill(0.0);
    res
}
