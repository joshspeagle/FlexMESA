// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled `BinarySystem::step` / `BinarySystem::try_step` explicit and implicit
//! RLOF trial loop, single-trial Henyey + first-order Euler orbital update,
//! irradiation coupling, and binary termination checks
//! (`binary/private/run_binary_support.f90:385-660`,
//! `binary/private/binary_evolve.f90:257-620`, and
//! `binary/private/binary_irradiation.f90:33-60`).

use flexmesa_const::{LSUN, MSUN, PI, SECYER};
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::star_utils::set_rho_to_dm_div_dv_for_new_step;
use flexmesa_star::{
    compute_core_masses_msun, do_check_limits, eval_photosphere_info,
    remesh_star_before_step, star_finalize_terminated_step,
    star_pick_next_timestep_after_step, LimitCheckOutcome, StarError,
    StarModel, StarPickTimestepOutcome, StarStepAttemptOutcome,
    StarStepConverged,
};

use crate::extras::{BinaryExtras, BinaryExtrasAction, DefaultBinaryExtras};
use crate::mdot_implicit::{
    update_change_factor_after_step, ImplicitMdotSolver, ImplicitStepAction,
    RloIterReportContext,
};
use crate::orbit::{
    eval_l2_overflow_limit, period_from_separation, separation_from_j_orb,
};
use crate::system::{compute_rl_and_gaps, BinaryCompanion, BinarySystem};
use crate::system_jdot::{
    eval_coupled_edot_breakdown, eval_gated_jdot_step,
    eval_star_am_removed_post_step, eval_star_spin_j,
    star_photosphere_radius_cm, EdotBreakdown,
};
use crate::system_rlof::{
    begin_single_star_attempt_with_extras, blend_explicit_mdot,
    eval_scheme_explicit_mdot_with_steps,
    finish_single_star_attempt_with_extras, prepare_coupled_mdots_with_extras,
    sync_vars_for_new_step_star,
};
use crate::timestep::{
    binary_pick_next_timestep, check_binary_hard_limits,
    BinaryTimestepControls, B_TLIM_COMP,
};

#[allow(clippy::large_enum_variant)]
enum SingleTrialOutcome {
    Completed {
        dt_used: f64,
        ts_ctrl: BinaryTimestepControls,
        donor_conv: StarStepConverged,
        acc_conv: Option<StarStepConverged>,
    },
    CutDt(f64),
    /// Single-attempt stellar solver failure (`star_evolve_step_part2 = retry`).
    StarSolverRetry {
        dt_used: f64,
        cause: Option<Box<StarError>>,
    },
    /// Upstream `retry` raised inside `binary_evolve_step`: the next `dt` is
    /// the carried value times `timestep_factor_for_retries`, and also times
    /// `dt_reduction_factor_for_j` when `have_to_reduce_timestep_due_to_j`
    /// is set.
    RetryStep(f64),
}

enum StarTimestepsResult {
    KeepGoing,
    Retry { is_dmdot_hard: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TrialDecision {
    KeepGoing,
    Redo,
    Retry,
    Terminate(String),
}

/// Precedence combiner matching `worst_result` in `run_binary_support.f90:836-857`
/// (`Terminate > Retry > Redo > KeepGoing`).
fn combine_worst_result(a: TrialDecision, b: TrialDecision) -> TrialDecision {
    match (a, b) {
        (TrialDecision::Terminate(r), _) | (_, TrialDecision::Terminate(r)) => {
            TrialDecision::Terminate(r)
        }
        (TrialDecision::Retry, _) | (_, TrialDecision::Retry) => {
            TrialDecision::Retry
        }
        (TrialDecision::Redo, _) | (_, TrialDecision::Redo) => {
            TrialDecision::Redo
        }
        (TrialDecision::KeepGoing, TrialDecision::KeepGoing) => {
            TrialDecision::KeepGoing
        }
    }
}

fn is_binary_trial_retriable(err: &StarError) -> bool {
    err.is_retriable()
}

fn emit_outer_retry_trace(
    sys: &BinarySystem,
    outer_retry: usize,
    dt_try: f64,
    implicit_mode: bool,
) {
    let mut jl = flexmesa_star::step_trace::JsonLine::new("binary_outer_retry");
    jl.field_usize("model", sys.model_number + 1);
    jl.field_usize("outer_retry", outer_retry);
    jl.field_f64("dt_try", dt_try);
    jl.field_bool("implicit", implicit_mode);
    jl.field_usize("donor_i", sys.donor_i);
    jl.field_f64("mtransfer", sys.mtransfer_rate);
    jl.field_f64("gap_d", sys.rl_relative_gap[0]);
    jl.field_f64("gap_a", sys.rl_relative_gap[1]);
    jl.emit();
}

fn emit_dt_cut_trace(
    model: usize,
    outer_retry: usize,
    try_idx: usize,
    dt_try: f64,
    next_dt: f64,
) {
    let mut jl = flexmesa_star::step_trace::JsonLine::new("binary_dt_cut");
    jl.field_usize("model", model);
    jl.field_usize("outer_retry", outer_retry);
    jl.field_usize("try_idx", try_idx);
    jl.field_f64("dt_try", dt_try);
    jl.field_f64("dt_next", next_dt);
    jl.emit();
}

#[allow(clippy::too_many_arguments)]
fn emit_implicit_trial_trace(
    model: usize,
    outer_retry: usize,
    try_idx: usize,
    trial_mdot: f64,
    dt_used: Option<f64>,
    m_exp: Option<f64>,
    outcome: &str,
    detail: Option<&str>,
) {
    let mut jl =
        flexmesa_star::step_trace::JsonLine::new("binary_implicit_trial");
    jl.field_usize("model", model);
    jl.field_usize("outer_retry", outer_retry);
    jl.field_usize("try_idx", try_idx);
    jl.field_f64("trial_mdot", trial_mdot);
    if let Some(dt) = dt_used {
        jl.field_f64("dt_used", dt);
    }
    if let Some(me) = m_exp {
        jl.field_f64("m_exp", me);
    }
    jl.field_str("outcome", outcome);
    if let Some(det) = detail {
        jl.field_str("detail", det);
    }
    jl.emit();
}

fn emit_hard_limit_retry_trace(
    model: usize,
    outer_retry: usize,
    dt_used: f64,
    next_dt: f64,
    dmdot_hard: bool,
) {
    let mut jl =
        flexmesa_star::step_trace::JsonLine::new("binary_hard_limit_retry");
    jl.field_usize("model", model);
    jl.field_usize("outer_retry", outer_retry);
    jl.field_f64("dt_used", dt_used);
    jl.field_f64("dt_next", next_dt);
    jl.field_bool("dmdot_hard", dmdot_hard);
    jl.emit();
}

/// Evaluates `eval_irradiation_heat` (`star/private/star_utils.f90:1849-1866`)
/// on `star`'s current mesh (`irradiation_dq = 4 * pi * r(1)^2 * sigma / xmstar`
/// with partial-cell weighting on the straddling cell).
fn refresh_star_irradiation_heat(star: &mut StarModel) {
    let nz = star.nz;
    if star.aux.irradiation_heat.len() != nz {
        star.aux.irradiation_heat.resize(nz, 0.0);
    }
    let f_irr = star.controls.irradiation_flux;
    let sigma_max = star.controls.column_depth_for_irradiation;
    if f_irr == 0.0 || sigma_max <= 0.0 || nz == 0 || star.mstar <= 0.0 {
        star.aux.irradiation_heat.fill(0.0);
        return;
    }
    let r0 = star.r[0];
    let xmstar = (star.mstar - star.m_center).max(1.0e-99);
    let irradiation_dq = (4.0 * PI) * r0 * r0 * sigma_max / xmstar;
    let eps = 0.25 * f_irr / sigma_max;
    for k in 0..nz {
        let xq = 1.0 - star.q[k];
        let dq_k = star.dq[k];
        star.aux.irradiation_heat[k] = if irradiation_dq > xq {
            if irradiation_dq < xq + dq_k && dq_k > 0.0 {
                eps * (irradiation_dq - xq) / dq_k
            } else {
                eps
            }
        } else {
            0.0
        };
    }
}

impl BinarySystem {
    /// Advances the coupled binary system by one timestep `dt` (s) using [`DefaultBinaryExtras`].
    pub fn step(&mut self, dt: f64) -> (f64, i32) {
        self.step_with_extras(dt, &mut DefaultBinaryExtras)
    }

    /// Advances the coupled binary system by one timestep `dt` (s) with custom [`BinaryExtras`].
    pub fn step_with_extras(
        &mut self,
        dt: f64,
        extras: &mut dyn BinaryExtras,
    ) -> (f64, i32) {
        self.try_step_with_extras(dt, extras)
            .unwrap_or((0.5 * dt.max(1.0), B_TLIM_COMP))
    }

    /// Advances the coupled binary system by one timestep `dt` (s) using [`DefaultBinaryExtras`],
    /// returning `(next_dt_sec, limit_reason)` or [`StarError`] on unrecoverable solver failure.
    pub fn try_step(&mut self, dt: f64) -> Result<(f64, i32), StarError> {
        self.try_step_with_extras(dt, &mut DefaultBinaryExtras)
    }

    /// Advances the coupled binary system by one timestep `dt` (s) with custom [`BinaryExtras`],
    /// returning `(next_dt_sec, limit_reason)` or [`StarError`] on unrecoverable solver failure.
    pub fn try_step_with_extras(
        &mut self,
        dt: f64,
        extras: &mut dyn BinaryExtras,
    ) -> Result<(f64, i32), StarError> {
        self.controls.mass_transfer_alpha = self.mass_transfer_alpha;
        self.controls.mass_transfer_beta = self.mass_transfer_beta;
        self.controls.mass_transfer_delta = self.mass_transfer_delta;
        self.controls.mass_transfer_gamma = self.mass_transfer_gamma;
        if !self.use_implicit_rlof && self.controls.max_tries_to_achieve > 0 {
            self.controls.max_tries_to_achieve = 0;
        }
        self.last_termination_reason = None;
        self.donor_started_implicit_wind = false;
        self.have_to_reduce_timestep_due_to_j = false;
        self.maybe_switch_donor_at_step_start()?;

        self.donor.step_retry_cnt = 0;
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            s2.step_retry_cnt = 0;
        }
        let mut star_target_mods = [0_usize; 2];
        let d_idx_init = (self.donor_i - 1).min(1);
        star_target_mods[d_idx_init] = self.donor.model_number + 1;
        if let BinaryCompanion::Star(ref s2) = self.companion {
            star_target_mods[1 - d_idx_init] = s2.model_number + 1;
        }
        let pre_step_snap = self.capture_snapshot();
        let mut retry_holds = [(None, 0_usize); 2];
        let s2_min = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => s2.controls.min_timestep_limit,
        };
        let dt_floor =
            self.donor.controls.min_timestep_limit.max(s2_min).max(0.0);
        let retry_factor_base =
            self.donor.controls.ts_stop.timestep_factor_for_retries;
        if !(retry_factor_base.is_finite()
            && retry_factor_base > 0.0
            && retry_factor_base < 1.0)
        {
            return Err(StarError::InvalidInput(format!(
                "timestep_factor_for_retries must be in (0, 1), got {retry_factor_base}"
            )));
        }

        let check_cut_dt =
            |dt_cut: f64, cur_dt: f64| -> Result<f64, StarError> {
                if !dt_cut.is_finite() || dt_cut <= 0.0 {
                    return Err(StarError::ConvergenceFailure(format!(
                        "CutDt produced non-positive dt ({dt_cut:.3e} s)"
                    )));
                }
                if dt_floor > 0.0 && dt_cut < dt_floor {
                    return Err(StarError::ConvergenceFailure(format!(
                    "CutDt hit dt_floor ({dt_cut:.3e} s < {dt_floor:.3e} s)"
                )));
                }
                let next_dt = dt_cut.max(dt_floor);
                if next_dt >= 0.999_999 * cur_dt {
                    return Err(StarError::ConvergenceFailure(format!(
                    "CutDt stalled at dt ({next_dt:.3e} s >= {cur_dt:.3e} s)"
                )));
                }
                Ok(next_dt)
            };

        self.set_star_timesteps();
        let mut dt_try = dt.min(self.max_timestep_sec);
        if dt_try <= 0.0 || (dt_floor > 0.0 && dt_try < dt_floor) {
            return Err(StarError::ConvergenceFailure(format!(
                "initial binary dt ({dt_try:.3e} s) is non-positive or below min_timestep_limit ({dt_floor:.3e} s)"
            )));
        }

        let implicit_mode = !self.ignore_rlof_flag
            && self.controls.max_tries_to_achieve > 0
            && self.controls.implicit_scheme_tolerance > 0.0;
        let max_tries =
            usize::try_from(self.controls.max_tries_to_achieve).unwrap_or(0);
        let target_model = pre_step_snap.model_number + 1;
        let doing_first_model_of_run = pre_step_snap.model_number == 0;
        let tracing = flexmesa_star::step_trace::enabled(target_model);
        let mut outer_retry = 0_usize;
        let mut first_try = true;
        let mut star_limit_redo_cnt = 0_usize;
        // Upstream remeshes once per step in `prepare_for_new_step`
        // (`star/private/evolve.f90:1882-1889`) before `new_generation`
        // stores the old model (`evolve.f90:1895`), and a retry or redo
        // restores that stored model with `set_current_to_old`
        // (`evolve.f90:401`). A retry therefore starts from the remeshed
        // model, never from the mesh of the previous step, unless
        // `restore_mesh_on_retry` is enabled (`evolve.f90:2192-2209`).
        let mut post_mesh_snap: Option<crate::system_rlof::BinaryStepSnapshot> =
            None;
        // `prepare_for_new_step` stores `prev_mesh_xh` right after the
        // `rho = dm/dV` reset and before the remesh (`evolve.f90:1854-1872`),
        // and `restore_mesh_on_retry` restores that model
        // (`evolve.f90:2192-2209`), so it carries the reset density.
        let mut pre_mesh_snap: Option<crate::system_rlof::BinaryStepSnapshot> =
            None;
        let restore_pre_mesh =
            (self.donor.controls.mix_wind.restore_mesh_on_retry
                && !self.donor.controls.rsp_flag)
                || matches!(
                    &self.companion,
                    BinaryCompanion::Star(s2)
                        if s2.controls.mix_wind.restore_mesh_on_retry
                            && !s2.controls.rsp_flag
                );
        loop {
            if outer_retry > 0 {
                let snap = if restore_pre_mesh {
                    pre_mesh_snap.as_ref().unwrap_or(&pre_step_snap)
                } else {
                    post_mesh_snap.as_ref().unwrap_or(&pre_step_snap)
                };
                self.restore_snapshot(snap)?;
                if restore_pre_mesh {
                    self.donor.aux.need_to_setvars = true;
                    sync_vars_for_new_step_star(&mut self.donor)?;
                    if let BinaryCompanion::Star(ref mut s2) = self.companion {
                        s2.aux.need_to_setvars = true;
                        sync_vars_for_new_step_star(s2)?;
                    }
                }
                self.apply_retry_holds(&retry_holds);
            }
            if tracing {
                emit_outer_retry_trace(
                    self,
                    outer_retry,
                    dt_try,
                    implicit_mode,
                );
            }

            // Remesh once per step, before either branch (see `post_mesh_snap`).
            if outer_retry == 0 && star_limit_redo_cnt == 0 {
                // `evolve.f90:319, 1854-1857`: mark `need_to_setvars` at step
                // start and reset `rho = dm/dV` before the remesh.
                self.donor.aux.need_to_setvars = true;
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    s2.aux.need_to_setvars = true;
                }
                let reset = set_rho_to_dm_div_dv_for_new_step(&mut self.donor)
                    .and_then(|()| match self.companion {
                        BinaryCompanion::Star(ref mut s2) => {
                            set_rho_to_dm_div_dv_for_new_step(s2)
                        }
                        BinaryCompanion::PointMass { .. } => Ok(()),
                    });
                if let Err(e) = reset {
                    self.restore_snapshot(&pre_step_snap)?;
                    return Err(e);
                }
                // Store mesh info for the ST time smoothing
                // (`star/private/evolve.f90:1859-1879`).
                flexmesa_star::rotation_mix_info::store_prev_mesh_st_info(
                    &mut self.donor,
                );
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    flexmesa_star::rotation_mix_info::store_prev_mesh_st_info(
                        s2,
                    );
                }
                if restore_pre_mesh {
                    pre_mesh_snap = Some(self.capture_snapshot());
                }
                self.refresh_tidal_deformation_hooks(false)?;
                let donor_dt_prev = self.donor.dt;
                if self.donor.controls.mesh_delta_coeff > 0.0 {
                    self.donor.aux.okay_to_set_mlt_vc = false;
                    let remesh_res =
                        remesh_star_before_step(&mut self.donor, donor_dt_prev);
                    if let Err(e) = remesh_res {
                        self.donor.aux.okay_to_set_mlt_vc = true;
                        return Err(e);
                    }
                }
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    let s2_dt_prev = s2.dt;
                    if s2.controls.mesh_delta_coeff > 0.0 {
                        s2.aux.okay_to_set_mlt_vc = false;
                        let remesh_res =
                            remesh_star_before_step(s2, s2_dt_prev);
                        if let Err(e) = remesh_res {
                            s2.aux.okay_to_set_mlt_vc = true;
                            return Err(e);
                        }
                    }
                }
                self.refresh_tidal_deformation_hooks(false)?;
                // `star/private/evolve.f90:1891`: set derived variables once
                // post-remesh before capturing `post_mesh_snap`, so
                // `prepare_coupled_mdots`, `try_idx >= 2`, and retries match
                // upstream's post-`set_vars` state (`#140`, `#160`).
                sync_vars_for_new_step_star(&mut self.donor)?;
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    sync_vars_for_new_step_star(s2)?;
                }
                post_mesh_snap = Some(self.capture_snapshot());
            }
            if !implicit_mode {
                let mut redo_idx = 0_usize;
                let mut explicit_trial_mdot = if self.ignore_rlof_flag {
                    0.0
                } else {
                    self.mtransfer_rate
                };
                loop {
                    if redo_idx > 0 {
                        // `binary/private/binary_evolve.f90:596, 616-617`:
                        // `binary_set_current_to_old` restores
                        // `mtransfer_rate_old` and `change_factor_old` when
                        // `num_tries == 0`. Only `check_implicit_rlo`
                        // increments `num_tries` (`binary_mdot.f90:76`) and
                        // it is not called in explicit mode
                        // (`binary_evolve.f90:407-415`), so an explicit redo
                        // starts again from the old values.
                        self.restore_snapshot(
                            post_mesh_snap.as_ref().unwrap_or(&pre_step_snap),
                        )?;
                        self.apply_retry_holds(&retry_holds);
                        explicit_trial_mdot = if self.ignore_rlof_flag {
                            0.0
                        } else {
                            self.mtransfer_rate
                        };
                    }
                    let attempt_first_try = first_try && redo_idx == 0;
                    match self.run_single_trial(
                        explicit_trial_mdot,
                        dt_try,
                        attempt_first_try,
                        extras,
                    ) {
                        Ok(SingleTrialOutcome::Completed {
                            dt_used,
                            ts_ctrl,
                            donor_conv,
                            acc_conv,
                        }) => {
                            self.mtransfer_rate = if self.ignore_rlof_flag {
                                0.0
                            } else {
                                let m_exp =
                                    eval_scheme_explicit_mdot_with_steps(
                                        &self.controls.mdot_scheme,
                                        &self.donor,
                                        self.accretor_mass_g(),
                                        self.rl[0],
                                        self.separation,
                                        self.eccentricity,
                                        self.controls.anomaly_steps,
                                    )?;
                                blend_explicit_mdot(
                                    explicit_trial_mdot,
                                    m_exp,
                                    self.controls.cur_mdot_frac,
                                    self.controls.max_explicit_abs_mdot,
                                )
                            };
                            self.adjust_irradiation();

                            let mut star_redo_req: Option<(f64, usize)> = None;
                            let mut decision = if let Some(reason) =
                                self.eval_binary_overflow_termination()
                            {
                                TrialDecision::Terminate(reason)
                            } else {
                                let (star_dec, star_dt, star_lim) = match self
                                    .eval_star_stopping_limits(
                                        doing_first_model_of_run,
                                    ) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        self.restore_snapshot(&pre_step_snap)?;
                                        return Err(e);
                                    }
                                };
                                if let (Some(dt_new), Some(lim)) =
                                    (star_dt, star_lim)
                                {
                                    star_redo_req = Some((dt_new, lim));
                                }
                                star_dec
                            };

                            let extras_decision = match extras.check_model(self)
                            {
                                BinaryExtrasAction::KeepGoing => {
                                    TrialDecision::KeepGoing
                                }
                                BinaryExtrasAction::Redo => TrialDecision::Redo,
                                BinaryExtrasAction::Retry => {
                                    TrialDecision::Retry
                                }
                                BinaryExtrasAction::Terminate(r) => {
                                    TrialDecision::Terminate(r)
                                }
                                BinaryExtrasAction::Error(msg) => {
                                    self.restore_snapshot(&pre_step_snap)?;
                                    return Err(StarError::ConvergenceFailure(
                                        msg,
                                    ));
                                }
                            };
                            decision =
                                combine_worst_result(decision, extras_decision);

                            // `run_binary_support.f90:540-544`: the binary
                            // timestep runs first, on `keep_going` or `redo`,
                            // because `star_pick_next_timestep` needs it.
                            if matches!(
                                decision,
                                TrialDecision::KeepGoing | TrialDecision::Redo
                            ) {
                                match self.pick_binary_timestep(&ts_ctrl) {
                                    Ok(d) => {
                                        decision =
                                            combine_worst_result(decision, d);
                                    }
                                    Err(e) => {
                                        self.restore_snapshot(&pre_step_snap)?;
                                        return Err(e);
                                    }
                                }
                            }
                            let mut dmdot_hard = false;
                            if decision == TrialDecision::KeepGoing {
                                match self.pick_star_next_timesteps(
                                    &donor_conv,
                                    acc_conv.as_ref(),
                                    outer_retry,
                                ) {
                                    Ok(StarTimestepsResult::KeepGoing) => {}
                                    Ok(StarTimestepsResult::Retry {
                                        is_dmdot_hard,
                                    }) => {
                                        if is_dmdot_hard {
                                            dmdot_hard = true;
                                            self.counters
                                                .num_hard_limit_retries += 1;
                                        }
                                        decision = TrialDecision::Retry;
                                    }
                                    Err(e) => {
                                        self.restore_snapshot(&pre_step_snap)?;
                                        return Err(e);
                                    }
                                }
                            }

                            match decision {
                                TrialDecision::KeepGoing => {
                                    return self.finish_accepted_step(
                                        pre_step_snap.mtransfer_rate,
                                        extras,
                                    );
                                }
                                TrialDecision::Terminate(reason) => {
                                    self.finalize_terminated_star_steps(
                                        &donor_conv,
                                        acc_conv.as_ref(),
                                        outer_retry,
                                    );
                                    self.last_termination_reason = Some(reason);
                                    return self.finish_accepted_step(
                                        pre_step_snap.mtransfer_rate,
                                        extras,
                                    );
                                }
                                TrialDecision::Redo => {
                                    if let Some((dt_new, max_redos)) =
                                        star_redo_req
                                    {
                                        star_limit_redo_cnt += 1;
                                        if max_redos > 0
                                            && star_limit_redo_cnt > max_redos
                                        {
                                            self.restore_snapshot(
                                                &pre_step_snap,
                                            )?;
                                            return Err(
                                                StarError::Terminated {
                                                    model_number: target_model,
                                                    reason: String::from(
                                                        "redo_limit",
                                                    ),
                                                },
                                            );
                                        }
                                        if !(dt_new.is_finite() && dt_new > 0.0)
                                        {
                                            self.restore_snapshot(
                                                &pre_step_snap,
                                            )?;
                                            return Err(
                                                StarError::Terminated {
                                                    model_number: target_model,
                                                    reason: String::from(
                                                        "dt_is_zero",
                                                    ),
                                                },
                                            );
                                        }
                                        dt_try = dt_new;
                                    }
                                    // Upstream's `step_loop` has no limit on
                                    // binary-extras redos (`binary/private/
                                    // run_binary_support.f90:385-660`).
                                    first_try = false;
                                    redo_idx += 1;
                                }
                                TrialDecision::Retry => {
                                    let prev_outer = outer_retry;
                                    let next_dt = self.prepare_outer_retry(
                                        dt_used,
                                        None,
                                        retry_factor_base,
                                        dt_floor,
                                        &mut outer_retry,
                                        &mut first_try,
                                        &mut retry_holds,
                                        star_target_mods,
                                        target_model,
                                        &pre_step_snap,
                                        None,
                                    )?;
                                    if tracing {
                                        emit_hard_limit_retry_trace(
                                            target_model,
                                            prev_outer,
                                            dt_used,
                                            next_dt,
                                            dmdot_hard,
                                        );
                                    }
                                    dt_try = next_dt;
                                    break;
                                }
                            }
                        }
                        Ok(SingleTrialOutcome::CutDt(dt_cut)) => {
                            self.counters.num_dt_cuts += 1;
                            self.have_to_reduce_timestep_due_to_j = false;
                            let cut_dt = match check_cut_dt(dt_cut, dt_try) {
                                Ok(d) => d,
                                Err(e) => {
                                    self.restore_snapshot(&pre_step_snap)?;
                                    return Err(e);
                                }
                            };
                            dt_try = self.prepare_outer_retry(
                                dt_try,
                                Some(cut_dt),
                                retry_factor_base,
                                dt_floor,
                                &mut outer_retry,
                                &mut first_try,
                                &mut retry_holds,
                                star_target_mods,
                                target_model,
                                &pre_step_snap,
                                None,
                            )?;
                            break;
                        }
                        Ok(SingleTrialOutcome::StarSolverRetry {
                            dt_used,
                            cause,
                        }) => {
                            dt_try = self.prepare_outer_retry(
                                dt_used,
                                None,
                                retry_factor_base,
                                dt_floor,
                                &mut outer_retry,
                                &mut first_try,
                                &mut retry_holds,
                                star_target_mods,
                                target_model,
                                &pre_step_snap,
                                cause.map(|b| *b),
                            )?;
                            break;
                        }
                        Ok(SingleTrialOutcome::RetryStep(dt_used)) => {
                            self.counters.num_dt_cuts += 1;
                            dt_try = self.prepare_outer_retry(
                                dt_used,
                                None,
                                retry_factor_base,
                                dt_floor,
                                &mut outer_retry,
                                &mut first_try,
                                &mut retry_holds,
                                star_target_mods,
                                target_model,
                                &pre_step_snap,
                                None,
                            )?;
                            break;
                        }
                        Err(e) if is_binary_trial_retriable(&e) => {
                            dt_try = self.prepare_outer_retry(
                                dt_try,
                                None,
                                retry_factor_base,
                                dt_floor,
                                &mut outer_retry,
                                &mut first_try,
                                &mut retry_holds,
                                star_target_mods,
                                target_model,
                                &pre_step_snap,
                                Some(e),
                            )?;
                            break;
                        }
                        Err(e) => {
                            self.restore_snapshot(&pre_step_snap)?;
                            return Err(e);
                        }
                    }
                }
                continue;
            }

            // Implicit RLOF branch (`run_binary_support.f90:417-652`).
            let mut trial_snap = self.capture_snapshot();
            let mut solver = ImplicitMdotSolver::from_controls(
                &self.controls,
                self.change_factor,
            )?;
            let mut trial_mdot = if self.mtransfer_rate < 0.0 {
                self.mtransfer_rate.max(-solver.max_abs_mdot_cgs)
            } else {
                0.0
            };

            // `binary_mdot.f90:76, 205`: `num_tries` increments on each call to
            // `check_implicit_rlo` and triggers `retry` only when
            // `num_tries > max_tries_to_achieve`, so convergence is evaluated
            // through try `max_tries + 1`.
            let mut try_idx = 1_usize;
            let mut redo_dt_override = false;
            loop {
                if try_idx > 1 || redo_dt_override {
                    // `binary_evolve.f90:615-617`: `binary_set_current_to_old`
                    // restores `change_factor_old` only on outer retry
                    // (`num_tries == 0`), not on implicit RLOF redo trials.
                    let live_cf = self.change_factor;
                    self.restore_snapshot(&trial_snap)?;
                    self.change_factor = live_cf;
                    if redo_dt_override {
                        self.mtransfer_rate = trial_mdot;
                    }
                    redo_dt_override = false;
                }
                // `run_binary_support.f90:421-424`: preserve rotational implicit
                // wind state and double `change_factor` across subsequent trials.
                if self.donor_started_implicit_wind {
                    self.donor.aux.step_hist.was_in_implicit_wind_limit = true;
                    self.change_factor = 2.0 * self.controls.max_change_factor;
                }
                self.counters.num_implicit_trials += 1;
                let attempt_first_try = first_try && try_idx == 1;
                let (dt_used, ts_ctrl, donor_conv, acc_conv) = match self
                    .run_single_trial(
                        trial_mdot,
                        dt_try,
                        attempt_first_try,
                        extras,
                    ) {
                    Ok(SingleTrialOutcome::Completed {
                        dt_used,
                        ts_ctrl,
                        donor_conv,
                        acc_conv,
                    }) => {
                        if try_idx > 1 && dt_used < 0.999_999 * dt_try {
                            self.counters.num_dt_cuts += 1;
                            let cut_dt = match check_cut_dt(dt_used, dt_try) {
                                Ok(d) => d,
                                Err(e) => {
                                    self.restore_snapshot(&pre_step_snap)?;
                                    return Err(e);
                                }
                            };
                            dt_try = self.prepare_outer_retry(
                                dt_try,
                                Some(cut_dt),
                                retry_factor_base,
                                dt_floor,
                                &mut outer_retry,
                                &mut first_try,
                                &mut retry_holds,
                                star_target_mods,
                                target_model,
                                &pre_step_snap,
                                None,
                            )?;
                            break;
                        }
                        (dt_used, ts_ctrl, donor_conv, acc_conv)
                    }
                    Ok(SingleTrialOutcome::StarSolverRetry {
                        dt_used,
                        cause,
                    }) => {
                        if tracing {
                            emit_implicit_trial_trace(
                                target_model,
                                outer_retry,
                                try_idx,
                                trial_mdot,
                                Some(dt_used),
                                None,
                                "star_retry",
                                cause
                                    .as_deref()
                                    .map(ToString::to_string)
                                    .as_deref(),
                            );
                        }
                        dt_try = self.prepare_outer_retry(
                            dt_used,
                            None,
                            retry_factor_base,
                            dt_floor,
                            &mut outer_retry,
                            &mut first_try,
                            &mut retry_holds,
                            star_target_mods,
                            target_model,
                            &pre_step_snap,
                            cause.map(|b| *b),
                        )?;
                        break;
                    }
                    Ok(SingleTrialOutcome::RetryStep(dt_used)) => {
                        self.counters.num_dt_cuts += 1;
                        let prev_outer = outer_retry;
                        let next_dt = self.prepare_outer_retry(
                            dt_used,
                            None,
                            retry_factor_base,
                            dt_floor,
                            &mut outer_retry,
                            &mut first_try,
                            &mut retry_holds,
                            star_target_mods,
                            target_model,
                            &pre_step_snap,
                            None,
                        )?;
                        if tracing {
                            emit_dt_cut_trace(
                                target_model,
                                prev_outer,
                                try_idx,
                                dt_try,
                                next_dt,
                            );
                        }
                        dt_try = next_dt;
                        break;
                    }
                    Ok(SingleTrialOutcome::CutDt(dt_cut)) => {
                        self.counters.num_dt_cuts += 1;
                        self.have_to_reduce_timestep_due_to_j = false;
                        let prev_outer = outer_retry;
                        let cut_dt = match check_cut_dt(dt_cut, dt_try) {
                            Ok(d) => d,
                            Err(e) => {
                                self.restore_snapshot(&pre_step_snap)?;
                                return Err(e);
                            }
                        };
                        let next_dt = self.prepare_outer_retry(
                            dt_try,
                            Some(cut_dt),
                            retry_factor_base,
                            dt_floor,
                            &mut outer_retry,
                            &mut first_try,
                            &mut retry_holds,
                            star_target_mods,
                            target_model,
                            &pre_step_snap,
                            None,
                        )?;
                        if tracing {
                            emit_dt_cut_trace(
                                target_model,
                                prev_outer,
                                try_idx,
                                dt_try,
                                next_dt,
                            );
                        }
                        dt_try = next_dt;
                        break;
                    }
                    Err(e) => {
                        if tracing {
                            emit_implicit_trial_trace(
                                target_model,
                                outer_retry,
                                try_idx,
                                trial_mdot,
                                None,
                                None,
                                "error",
                                Some(&e.to_string()),
                            );
                        }
                        if !is_binary_trial_retriable(&e) {
                            self.restore_snapshot(&pre_step_snap)?;
                            return Err(e);
                        }
                        dt_try = self.prepare_outer_retry(
                            dt_try,
                            None,
                            retry_factor_base,
                            dt_floor,
                            &mut outer_retry,
                            &mut first_try,
                            &mut retry_holds,
                            star_target_mods,
                            target_model,
                            &pre_step_snap,
                            Some(e),
                        )?;
                        break;
                    }
                };
                if try_idx == 1 && dt_used != dt_try {
                    dt_try = dt_used;
                }

                let (m_d, m_a) = (self.donor.mstar, self.accretor_mass_g());
                let m_exp = if self
                    .controls
                    .mdot_scheme
                    .eq_ignore_ascii_case("roche_lobe")
                    || self.controls.mdot_scheme.eq_ignore_ascii_case("contact")
                {
                    0.0
                } else {
                    eval_scheme_explicit_mdot_with_steps(
                        &self.controls.mdot_scheme,
                        &self.donor,
                        m_a,
                        self.rl[0],
                        self.separation,
                        self.eccentricity,
                        self.controls.anomaly_steps,
                    )?
                };
                solver.change_factor = self.change_factor;
                let act = solver.evaluate_trial_with_context(
                    &self.controls.mdot_scheme,
                    try_idx,
                    max_tries,
                    trial_mdot,
                    m_exp,
                    self.rl_relative_gap[0],
                    self.rl_relative_gap[1],
                    m_a / m_d.max(1.0),
                    RloIterReportContext {
                        model_number: self.model_number,
                        donor_i: self.donor_i,
                        rl_gaps_by_star: self.rl_relative_gap_by_star(),
                        point_mass_i: self.point_mass_i,
                    },
                )?;
                self.change_factor = solver.change_factor;
                let outcome_name = match &act {
                    ImplicitStepAction::Converged(_) => "converged",
                    ImplicitStepAction::Detached => "detached",
                    ImplicitStepAction::TryNextMdot(_) => "try_next_mdot",
                    ImplicitStepAction::ExceededMaxTries => {
                        "exceeded_max_tries"
                    }
                };
                if tracing {
                    emit_implicit_trial_trace(
                        target_model,
                        outer_retry,
                        try_idx,
                        trial_mdot,
                        Some(dt_used),
                        Some(m_exp),
                        outcome_name,
                        None,
                    );
                }

                let mut next_mdot_candidate = 0.0_f64;
                let mut decision = match act {
                    ImplicitStepAction::Converged(m_conv) => {
                        self.mtransfer_rate = m_conv.min(0.0);
                        self.donor_started_implicit_wind = false;
                        TrialDecision::KeepGoing
                    }
                    ImplicitStepAction::Detached => {
                        self.mtransfer_rate = 0.0;
                        self.donor_started_implicit_wind = false;
                        TrialDecision::KeepGoing
                    }
                    ImplicitStepAction::TryNextMdot(next_m) => {
                        next_mdot_candidate = next_m;
                        self.mtransfer_rate = next_m;
                        TrialDecision::Redo
                    }
                    ImplicitStepAction::ExceededMaxTries => {
                        TrialDecision::Retry
                    }
                };
                self.donor_started_implicit_wind |=
                    self.donor.aux.step_hist.was_in_implicit_wind_limit;
                self.adjust_irradiation();

                if let Some(reason) = self.eval_binary_overflow_termination() {
                    decision = combine_worst_result(
                        decision,
                        TrialDecision::Terminate(reason),
                    );
                }
                let mut star_redo_req: Option<(f64, usize)> = None;
                if decision == TrialDecision::KeepGoing {
                    let (star_dec, star_dt, star_lim) = match self
                        .eval_star_stopping_limits(doing_first_model_of_run)
                    {
                        Ok(v) => v,
                        Err(e) => {
                            self.restore_snapshot(&pre_step_snap)?;
                            return Err(e);
                        }
                    };
                    if let (Some(dt_new), Some(lim)) = (star_dt, star_lim) {
                        star_redo_req = Some((dt_new, lim));
                    }
                    decision = combine_worst_result(decision, star_dec);
                }

                // `run_binary_support.f90:537-538`: `extras_binary_check_model`
                // runs on every trial inside `step_loop`.
                let extras_decision = match extras.check_model(self) {
                    BinaryExtrasAction::KeepGoing => TrialDecision::KeepGoing,
                    BinaryExtrasAction::Redo => TrialDecision::Redo,
                    BinaryExtrasAction::Retry => TrialDecision::Retry,
                    BinaryExtrasAction::Terminate(r) => {
                        TrialDecision::Terminate(r)
                    }
                    BinaryExtrasAction::Error(msg) => {
                        self.restore_snapshot(&pre_step_snap)?;
                        return Err(StarError::ConvergenceFailure(msg));
                    }
                };
                decision = combine_worst_result(decision, extras_decision);

                // `run_binary_support.f90:540-544`: the binary timestep runs
                // first, on `keep_going` or `redo` (so a hard limit stops the
                // implicit solver), because `star_pick_next_timestep` needs it.
                if matches!(
                    decision,
                    TrialDecision::KeepGoing | TrialDecision::Redo
                ) {
                    match self.pick_binary_timestep(&ts_ctrl) {
                        Ok(d) => decision = combine_worst_result(decision, d),
                        Err(e) => {
                            self.restore_snapshot(&pre_step_snap)?;
                            return Err(e);
                        }
                    }
                }
                let mut dmdot_hard = false;
                if decision == TrialDecision::KeepGoing {
                    match self.pick_star_next_timesteps(
                        &donor_conv,
                        acc_conv.as_ref(),
                        outer_retry,
                    ) {
                        Ok(StarTimestepsResult::KeepGoing) => {}
                        Ok(StarTimestepsResult::Retry { is_dmdot_hard }) => {
                            if is_dmdot_hard {
                                dmdot_hard = true;
                                self.counters.num_hard_limit_retries += 1;
                            }
                            decision = TrialDecision::Retry;
                        }
                        Err(e) => {
                            self.restore_snapshot(&pre_step_snap)?;
                            return Err(e);
                        }
                    }
                }

                match decision {
                    TrialDecision::KeepGoing => {
                        return self.finish_accepted_step(
                            pre_step_snap.mtransfer_rate,
                            extras,
                        );
                    }
                    TrialDecision::Terminate(reason) => {
                        if self.mtransfer_rate > 0.0 {
                            self.mtransfer_rate = 0.0;
                        }
                        self.finalize_terminated_star_steps(
                            &donor_conv,
                            acc_conv.as_ref(),
                            outer_retry,
                        );
                        self.last_termination_reason = Some(reason);
                        return self.finish_accepted_step(
                            pre_step_snap.mtransfer_rate,
                            extras,
                        );
                    }
                    TrialDecision::Redo => {
                        if let Some((dt_new, max_redos)) = star_redo_req {
                            star_limit_redo_cnt += 1;
                            if max_redos > 0 && star_limit_redo_cnt > max_redos
                            {
                                self.restore_snapshot(&pre_step_snap)?;
                                return Err(StarError::Terminated {
                                    model_number: target_model,
                                    reason: String::from("redo_limit"),
                                });
                            }
                            if !(dt_new.is_finite() && dt_new > 0.0) {
                                self.restore_snapshot(&pre_step_snap)?;
                                return Err(StarError::Terminated {
                                    model_number: target_model,
                                    reason: String::from("dt_is_zero"),
                                });
                            }
                            // `run_binary_support.f90:585-603`,
                            // `binary_evolve.f90:572-625`: stay inside
                            // `step_loop` on star limit-hitting redos so
                            // `ImplicitMdotSolver` brackets, `change_factor`,
                            // `try_idx`, and `trial_mdot` are preserved.
                            dt_try = dt_new;
                            redo_dt_override = true;
                        } else if try_idx > max_tries {
                            dt_try = self.prepare_outer_retry(
                                dt_used,
                                None,
                                retry_factor_base,
                                dt_floor,
                                &mut outer_retry,
                                &mut first_try,
                                &mut retry_holds,
                                star_target_mods,
                                target_model,
                                &pre_step_snap,
                                None,
                            )?;
                            break;
                        }
                        first_try = false;
                        if next_mdot_candidate > 0.0 && self.point_mass_i == 0 {
                            let live_cf = self.change_factor;
                            self.restore_snapshot(&trial_snap)?;
                            self.change_factor = live_cf;
                            let _ = self.swap_donor_and_accretor()?;
                            self.capture_snapshot_into(&mut trial_snap);
                            solver.flip_for_contact_donor_switch();
                            trial_mdot = -next_mdot_candidate;
                        } else if next_mdot_candidate != 0.0 {
                            trial_mdot = next_mdot_candidate.min(0.0);
                        } else {
                            trial_mdot = self.mtransfer_rate.min(0.0);
                        }
                        try_idx += 1;
                    }
                    TrialDecision::Retry => {
                        let prev_outer = outer_retry;
                        let next_dt = self.prepare_outer_retry(
                            dt_used,
                            None,
                            retry_factor_base,
                            dt_floor,
                            &mut outer_retry,
                            &mut first_try,
                            &mut retry_holds,
                            star_target_mods,
                            target_model,
                            &pre_step_snap,
                            None,
                        )?;
                        if tracing && dmdot_hard {
                            emit_hard_limit_retry_trace(
                                target_model,
                                prev_outer,
                                dt_used,
                                next_dt,
                                dmdot_hard,
                            );
                        }
                        dt_try = next_dt;
                        break;
                    }
                }
            }
        }
    }

    /// Executes upstream `star_prepare_to_retry` + `binary_prepare_to_retry` +
    /// `set_star_timesteps` J-reduction (`binary/private/run_binary_support.f90:604-615, 650`,
    /// `star/private/evolve.f90:2176-2277`, `binary/private/binary_timestep.f90:86-95`),
    /// advancing `step_retry_cnt` and `num_retries`, enforcing `retry_limit`,
    /// `min_timestep_limit`, and `max_number_retries`, updating
    /// `model_number_for_last_retry` and `timestep_hold`, and clearing `first_try`.
    #[allow(clippy::too_many_arguments)]
    fn prepare_outer_retry(
        &mut self,
        cur_dt: f64,
        explicit_next_dt: Option<f64>,
        retry_factor_base: f64,
        dt_floor: f64,
        outer_retry: &mut usize,
        first_try: &mut bool,
        retry_holds: &mut [(Option<usize>, usize); 2],
        star_target_mods: [usize; 2],
        target_model: usize,
        pre_step_snap: &crate::system_rlof::BinaryStepSnapshot,
        cause: Option<StarError>,
    ) -> Result<f64, StarError> {
        let next_retries = *outer_retry + 1;
        self.donor.step_retry_cnt = next_retries;
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            s2.step_retry_cnt = next_retries;
        }

        // `star/private/evolve.f90:2213-2218`: check per-step `retry_limit`.
        let donor_retry_limit = self.donor.controls.ts_stop.retry_limit;
        let acc_retry_limit = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0,
            BinaryCompanion::Star(s2) => s2.controls.ts_stop.retry_limit,
        };
        let hit_retry_limit = (donor_retry_limit > 0
            && next_retries > donor_retry_limit)
            || (acc_retry_limit > 0 && next_retries > acc_retry_limit);
        if hit_retry_limit {
            self.restore_snapshot(pre_step_snap)?;
            self.apply_retry_holds(retry_holds);
            if let Some(err) = cause {
                return Err(err);
            }
            let lim =
                if donor_retry_limit > 0 && next_retries > donor_retry_limit {
                    donor_retry_limit
                } else {
                    acc_retry_limit
                };
            return Err(StarError::ConvergenceFailure(format!(
                "BinarySystem::try_step exceeded retry_limit ({lim})"
            )));
        }

        // `evolve.f90:2222-2224` + `binary_timestep.f90:86-95`.
        let mut next_dt = if let Some(dt_cut) = explicit_next_dt {
            self.have_to_reduce_timestep_due_to_j = false;
            dt_cut
        } else {
            let mut d = cur_dt * retry_factor_base;
            if self.have_to_reduce_timestep_due_to_j {
                d *= self.controls.dt_reduction_factor_for_j;
                self.have_to_reduce_timestep_due_to_j = false;
            }
            d
        };

        // `star/private/evolve.f90:2240-2248`: check `min_timestep_limit`.
        if !next_dt.is_finite()
            || next_dt <= 0.0
            || (dt_floor > 0.0 && next_dt <= dt_floor)
        {
            self.restore_snapshot(pre_step_snap)?;
            self.apply_retry_holds(retry_holds);
            return Err(StarError::ConvergenceFailure(format!(
                "BinarySystem::try_step exhausted outer timestep retries (retry {next_retries}, next_dt={next_dt:.3e} s, min_timestep_limit={dt_floor:.3e} s)"
            )));
        }

        // `star/private/evolve.f90:2250-2255`: cap `next_dt` by `max_timestep`.
        let cap_star_max_dt = |s: &mut StarModel, dt_val: &mut f64| {
            if s.controls.max_years_for_timestep > 0.0 {
                s.controls.ts_stop.max_timestep =
                    SECYER * s.controls.max_years_for_timestep;
            }
            if s.controls.ts_stop.max_timestep > 0.0
                && *dt_val > s.controls.ts_stop.max_timestep
            {
                *dt_val = s.controls.ts_stop.max_timestep;
            }
        };
        cap_star_max_dt(&mut self.donor, &mut next_dt);
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            cap_star_max_dt(s2, &mut next_dt);
        }

        // `star/private/evolve.f90:2259-2266`: increment `num_retries` and
        // enforce `max_number_retries` (`relax_max_number_retries` if relaxing).
        self.counters.num_retries += 1;
        let bump_and_check_max_retries = |s: &mut StarModel| -> bool {
            s.num_retries += 1;
            let max_num = if s.aux.doing_relax {
                s.controls.relax_max_number_retries
            } else {
                s.controls.max_number_retries
            };
            usize::try_from(max_num).is_ok_and(|lim| s.num_retries > lim)
        };
        let mut exceeded_max_retries =
            bump_and_check_max_retries(&mut self.donor);
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            exceeded_max_retries |= bump_and_check_max_retries(s2);
        }
        if exceeded_max_retries {
            self.restore_snapshot(pre_step_snap)?;
            self.apply_retry_holds(retry_holds);
            return Err(StarError::Terminated {
                model_number: target_model,
                reason: String::from("max_number_retries"),
            });
        }

        // `star/private/evolve.f90:2268-2275`: update retry hold metadata on
        // each evolved star so subsequent attempts on `target_mod` engage retry
        // line search (`min_corr_coeff = 0.01`) and `timestep_hold`.
        let d_idx = (self.donor_i - 1).min(1);
        let update_hold = |s: &mut StarModel, star_target: usize| {
            s.aux.step_hist.model_number_for_last_retry = Some(star_target);
            let hold_span = if s.why_tlim_neg_x {
                s.controls
                    .ts_stop
                    .retry_hold
                    .max(s.controls.ts_stop.neg_mass_fraction_hold)
            } else {
                s.controls.ts_stop.retry_hold
            };
            s.aux.step_hist.timestep_hold = star_target + hold_span;
        };
        update_hold(&mut self.donor, star_target_mods[d_idx]);
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            update_hold(s2, star_target_mods[1 - d_idx]);
        }
        self.record_retry_holds(retry_holds);

        *outer_retry = next_retries;
        *first_try = false;
        Ok(next_dt)
    }

    fn pick_star_next_timesteps(
        &mut self,
        donor_conv: &StarStepConverged,
        acc_conv: Option<&StarStepConverged>,
        outer_retry: usize,
    ) -> Result<StarTimestepsResult, StarError> {
        let mut star_extras = DefaultStarExtras;
        match star_pick_next_timestep_after_step(
            &mut self.donor,
            donor_conv,
            outer_retry,
            &mut star_extras,
        )? {
            StarPickTimestepOutcome::KeepGoing(_) => {}
            StarPickTimestepOutcome::Retry { reason, .. } => {
                return Ok(StarTimestepsResult::Retry {
                    is_dmdot_hard: reason == "delta_mdot_hard_limit",
                });
            }
        }
        if let (BinaryCompanion::Star(ref mut s2), Some(conv_a)) =
            (&mut self.companion, acc_conv)
        {
            match star_pick_next_timestep_after_step(
                s2,
                conv_a,
                outer_retry,
                &mut star_extras,
            )? {
                StarPickTimestepOutcome::KeepGoing(_) => {}
                StarPickTimestepOutcome::Retry { reason, .. } => {
                    return Ok(StarTimestepsResult::Retry {
                        is_dmdot_hard: reason == "delta_mdot_hard_limit",
                    });
                }
            }
        }
        Ok(StarTimestepsResult::KeepGoing)
    }

    fn finalize_terminated_star_steps(
        &mut self,
        donor_conv: &StarStepConverged,
        acc_conv: Option<&StarStepConverged>,
        outer_retry: usize,
    ) {
        let _ = star_finalize_terminated_step(
            &mut self.donor,
            donor_conv,
            outer_retry,
        );
        if let (BinaryCompanion::Star(ref mut s2), Some(conv_a)) =
            (&mut self.companion, acc_conv)
        {
            let _ = star_finalize_terminated_step(s2, conv_a, outer_retry);
        }
    }

    /// Runs `binary_pick_next_timestep` (`binary/private/binary_timestep.f90:99-322`)
    /// at its place in the step loop: after `extras_binary_check_model` and
    /// before `star_pick_next_timestep`, on `keep_going` or `redo`
    /// (`run_binary_support.f90:540-544`).
    ///
    /// A violated hard limit returns `retry` before anything else changes
    /// (`binary_timestep.f90:166-262`; `fj_hard` also sets
    /// `have_to_reduce_timestep_due_to_j`). Otherwise the function stores
    /// the softened `max_timestep` and `dt_why_reason`, which a later redo or
    /// retry restores (`binary_set_current_to_old`, `binary_evolve.f90:613,
    /// 635`), and sets the stars' `varcontrol_target` and clears
    /// `ignore_hard_limits_this_step`, which neither restores
    /// (`binary_timestep.f90:287-319`).
    fn pick_binary_timestep(
        &mut self,
        ts_ctrl: &BinaryTimestepControls,
    ) -> Result<TrialDecision, StarError> {
        // `b% ignore_hard_limits_this_step` and `b% max_timestep` are read
        // when the function runs, so a `set_*` call made by
        // `extras_binary_check_model` in this trial counts.
        let ctrl = BinaryTimestepControls {
            ignore_hard_limits_this_step: self.ignore_hard_limits_this_step,
            max_timestep_sec: self.max_timestep_sec,
            ..ts_ctrl.clone()
        };
        if let Some(viol) = check_binary_hard_limits(
            &ctrl,
            self.controls.fm_hard,
            self.controls.fr_hard,
            self.controls.fj_hard,
            self.controls.fa_hard,
            self.controls.fe_hard,
            self.controls.fdm_hard,
        ) {
            self.counters.num_hard_limit_retries += 1;
            if viol.reduces_timestep_due_to_j() {
                self.have_to_reduce_timestep_due_to_j = true;
            }
            return Ok(TrialDecision::Retry);
        }
        let (max_dt, reason) = binary_pick_next_timestep(&ctrl)?;
        self.dt_why_reason = reason;
        self.max_timestep_sec = max_dt;
        self.update_stage_varcontrol();
        self.ignore_hard_limits_this_step = false;
        Ok(TrialDecision::KeepGoing)
    }

    /// `set_star_timesteps` (`binary/private/binary_timestep.f90:33-84`) at the
    /// start of the step: caps every evolved star's `dt_next` at the binary
    /// limit and marks it `why_Tlim = Tlim_binary`. FlexMESA models only the
    /// `Tlim_neg_X` value of `why_Tlim` (`why_tlim_neg_x`, read by
    /// `prepare_to_retry`, `star/private/evolve.f90:2269`), so marking a star
    /// clears that flag. The `dt_reduction_factor_for_j` part
    /// (`binary_timestep.f90:86-95`) is applied with the retry, in
    /// `prepare_outer_retry`. On later passes of the step loop no star's
    /// `dt_next` exceeds the limit (`max_timestep` is restored by every redo
    /// and retry), so applying it once per step is equivalent.
    fn set_star_timesteps(&mut self) {
        let mut dt_min = 1.0e99_f64;
        if self.donor.dt_next < dt_min {
            dt_min = self.donor.dt_next;
        }
        if let BinaryCompanion::Star(ref s2) = self.companion {
            if s2.dt_next < dt_min {
                dt_min = s2.dt_next;
            }
        }
        if self.max_timestep_sec <= dt_min {
            dt_min = self.max_timestep_sec;
        } else {
            self.dt_why_reason = B_TLIM_COMP;
        }
        let cap = |s: &mut StarModel| {
            if s.dt_next > dt_min {
                s.dt_next = dt_min;
                s.why_tlim_neg_x = false;
            }
        };
        cap(&mut self.donor);
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            cap(s2);
        }
    }

    /// Evaluates `adjust_irradiation` (`binary/private/binary_irradiation.f90:33-60`).
    pub fn adjust_irradiation(&mut self) {
        let col_depth = self.controls.col_depth_for_eps_extra;
        if col_depth <= 0.0 || self.separation <= 0.0 {
            return;
        }
        let f_irr = if self.controls.accretion_powered_irradiation {
            let lx = self.last_mdot.accretion_luminosity;
            self.controls
                .max_f_irr
                .min(lx / (4.0 * PI * self.separation * self.separation))
        } else if self.controls.use_accretor_luminosity_for_irrad {
            let Some(s_acc) = (match &self.companion {
                BinaryCompanion::Star(s2) => Some(&**s2),
                BinaryCompanion::PointMass { .. } => None,
            }) else {
                return;
            };
            let lx = eval_photosphere_info(s_acc).l_lsun * LSUN;
            self.controls
                .max_f_irr
                .min(lx / (4.0 * PI * self.separation * self.separation))
        } else {
            if self.controls.irrad_flux_at_std_distance <= 0.0 {
                return;
            }
            let ratio =
                self.controls.std_distance_for_irradiation / self.separation;
            self.controls.irrad_flux_at_std_distance * (ratio * ratio)
        };
        self.donor.controls.irradiation_flux = f_irr;
        self.donor.controls.column_depth_for_irradiation = col_depth;
        refresh_star_irradiation_heat(&mut self.donor);
    }

    /// Checks `accretor_overflow_terminate`, `terminate_if_initial_overflow`,
    /// and `terminate_if_L2_overflow` (`binary/private/binary_evolve.f90:455-489`).
    fn eval_binary_overflow_termination(&self) -> Option<String> {
        if self.point_mass_i == 0
            && self.rl_relative_gap[1] >= 0.0
            && self.rl_relative_gap[1]
                >= self.controls.accretor_overflow_terminate
        {
            return Some(String::from("accretor_overflow_terminate"));
        }
        if self.model_number == 1
            && self.controls.terminate_if_initial_overflow
            && !self.ignore_rlof_flag
            && (self.rl_relative_gap[0] >= 0.0
                || (self.point_mass_i == 0 && self.rl_relative_gap[1] >= 0.0))
        {
            return Some(String::from("terminate_if_initial_overflow"));
        }
        if self.point_mass_i == 0 && self.controls.terminate_if_l2_overflow {
            let m_all = self.star_masses_g();
            let gaps = self.rl_relative_gap_by_star();
            let (l2_lim, gap_sec) = if m_all[0] > m_all[1] {
                (eval_l2_overflow_limit(m_all[0], m_all[1]), gaps[1])
            } else {
                (eval_l2_overflow_limit(m_all[1], m_all[0]), gaps[0])
            };
            if gap_sec > l2_lim {
                return Some(String::from("terminate_if_L2_overflow"));
            }
        }
        None
    }

    /// Evaluates `star_check_model` (`do_check_limits`) on the donor and, if
    /// still `keep_going`, on the companion star, propagating `Terminate`,
    /// `Retry`, `Redo` (with halved `dt_new` and `redo_limit`), and fatal
    /// errors verbatim (`binary/private/run_binary_support.f90:523-535`).
    fn eval_star_stopping_limits(
        &mut self,
        doing_first_model_of_run: bool,
    ) -> Result<(TrialDecision, Option<f64>, Option<usize>), StarError> {
        match do_check_limits(&mut self.donor, None, doing_first_model_of_run)?
        {
            LimitCheckOutcome::KeepGoing => {}
            LimitCheckOutcome::Terminate { code, .. } => {
                self.donor.aux.step_hist.termination_code = Some(code.clone());
                return Ok((TrialDecision::Terminate(code), None, None));
            }
            LimitCheckOutcome::Retry { .. } => {
                return Ok((TrialDecision::Retry, None, None));
            }
            LimitCheckOutcome::Redo { dt_new, .. } => {
                let lim = self.donor.controls.ts_stop.redo_limit;
                return Ok((TrialDecision::Redo, Some(dt_new), Some(lim)));
            }
        }
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            match do_check_limits(s2, None, doing_first_model_of_run)? {
                LimitCheckOutcome::KeepGoing => {}
                LimitCheckOutcome::Terminate { code, .. } => {
                    s2.aux.step_hist.termination_code = Some(code.clone());
                    return Ok((TrialDecision::Terminate(code), None, None));
                }
                LimitCheckOutcome::Retry { .. } => {
                    return Ok((TrialDecision::Retry, None, None));
                }
                LimitCheckOutcome::Redo { dt_new, .. } => {
                    let lim = s2.controls.ts_stop.redo_limit;
                    return Ok((TrialDecision::Redo, Some(dt_new), Some(lim)));
                }
            }
        }
        Ok((TrialDecision::KeepGoing, None, None))
    }

    fn run_single_trial(
        &mut self,
        trial_mdot: f64,
        dt_try: f64,
        first_try: bool,
        extras: &mut dyn BinaryExtras,
    ) -> Result<SingleTrialOutcome, StarError> {
        let doing_first_model_of_run = self.model_number == 0;
        self.t_sync = [0.0; 2];
        self.tide_sync_scratch.reset();
        self.refresh_tidal_deformation_hooks(false)?;
        let (m_d_old, m_a_old) = (self.donor.mstar, self.accretor_mass_g());
        let (j_old, a_old, p_old, e_old, gap_old) = (
            self.angular_momentum_j,
            self.separation,
            self.period,
            self.eccentricity,
            self.rl_relative_gap,
        );
        let env_d_old =
            (m_d_old / MSUN - compute_core_masses_msun(&self.donor).0).max(0.0);

        let s_pre_d = eval_star_spin_j(&self.donor);
        let s_pre_a = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => eval_star_spin_j(s2),
        };

        let donor_id = self.donor_i;
        let ignore_rlof = self.ignore_rlof_flag;
        let donor_part1 =
            flexmesa_star::stepdump::with_star_id(donor_id, || {
                begin_single_star_attempt_with_extras(
                    &mut self.donor,
                    dt_try,
                    first_try,
                    false,
                    donor_id,
                    ignore_rlof,
                    extras,
                )
            })?;
        let dt_star_try = self.donor.dt;
        let mut acc_part1 = match &mut self.companion {
            BinaryCompanion::PointMass { .. } => None,
            BinaryCompanion::Star(s2) => {
                match flexmesa_star::stepdump::with_star_id(
                    3 - donor_id,
                    || {
                        begin_single_star_attempt_with_extras(
                            s2,
                            dt_star_try,
                            first_try,
                            false,
                            3 - donor_id,
                            ignore_rlof,
                            extras,
                        )
                    },
                ) {
                    Ok(p1) => Some(p1),
                    Err(e) => {
                        donor_part1.restore_guards(&mut self.donor);
                        return Err(e);
                    }
                }
            }
        };

        let pm_state = match &self.companion {
            BinaryCompanion::PointMass {
                mass_g,
                m_bh_zero_eq_g,
                ..
            } => {
                let eq_m = if self.eq_initial_bh_mass > 0.0 {
                    self.eq_initial_bh_mass
                } else {
                    *m_bh_zero_eq_g
                };
                Some((*mass_g, eq_m))
            }
            BinaryCompanion::Star(_) => None,
        };
        let acc_opt = match &mut self.companion {
            BinaryCompanion::PointMass { .. } => None,
            BinaryCompanion::Star(s2) => Some(&mut **s2),
        };
        let mut acct = match prepare_coupled_mdots_with_extras(
            &self.controls,
            &mut self.donor,
            acc_opt,
            pm_state,
            self.donor_i,
            self.rl,
            self.separation,
            self.eccentricity,
            trial_mdot,
            self.ignore_rlof_flag,
            extras,
        ) {
            Ok(acct) => acct,
            Err(e) => {
                donor_part1.restore_guards(&mut self.donor);
                if let (Some(p1_a), BinaryCompanion::Star(s2)) =
                    (acc_part1.take(), &mut self.companion)
                {
                    p1_a.restore_guards(s2);
                }
                return Err(e);
            }
        };

        refresh_star_irradiation_heat(&mut self.donor);
        // Step dumps label each star with MESA's `s% id`: the donor is star
        // `donor_i` and the accretor `3 - donor_i` (#372).
        let donor_conv =
            match flexmesa_star::stepdump::with_star_id(donor_id, || {
                finish_single_star_attempt_with_extras(
                    &mut self.donor,
                    donor_part1,
                    first_try,
                    donor_id,
                    ignore_rlof,
                    extras,
                )
            }) {
                Ok(StarStepAttemptOutcome::Converged(conv)) => conv,
                Ok(StarStepAttemptOutcome::Retry { cause, .. }) => {
                    if let (Some(p1_a), BinaryCompanion::Star(s2)) =
                        (acc_part1.take(), &mut self.companion)
                    {
                        p1_a.restore_guards(s2);
                    }
                    return Ok(SingleTrialOutcome::StarSolverRetry {
                        dt_used: dt_star_try,
                        cause,
                    });
                }
                Err(e) => {
                    if let (Some(p1_a), BinaryCompanion::Star(s2)) =
                        (acc_part1.take(), &mut self.companion)
                    {
                        p1_a.restore_guards(s2);
                    }
                    return Err(e);
                }
            };
        let dt_used = donor_conv.dt;
        let mtr = if ignore_rlof {
            0.0
        } else {
            trial_mdot.min(0.0)
        };
        // `binary/private/binary_evolve.f90:266-317`: sync donor
        // `component_mdot` and `mdot_system_wind` when `part2` (`iwind` /
        // `mdot_redo`) revised `mstar_dot`, leaving point-mass accretor
        // accounting from `adjust_mdots` untouched.
        if self.donor.mstar_dot.to_bits() != acct.component_mdot[0].to_bits() {
            let delta_d = self.donor.mstar_dot - acct.component_mdot[0];
            acct.mdot_wind_intrinsic[0] += delta_d;
            acct.component_mdot[0] = self.donor.mstar_dot;
            acct.mdot_system_wind[0] = self.donor.mstar_dot - mtr
                + acct.mdot_wind_transfer[1]
                - acct.mdot_wind_transfer[0];
        }
        let mut acc_conv: Option<StarStepConverged> = None;
        if let (Some(p1_a), BinaryCompanion::Star(ref mut s2)) =
            (acc_part1.take(), &mut self.companion)
        {
            let conv_a = match flexmesa_star::stepdump::with_star_id(
                3 - donor_id,
                || {
                    finish_single_star_attempt_with_extras(
                        s2,
                        p1_a,
                        first_try,
                        3 - donor_id,
                        ignore_rlof,
                        extras,
                    )
                },
            )? {
                StarStepAttemptOutcome::Converged(conv) => conv,
                StarStepAttemptOutcome::Retry { cause, .. } => {
                    return Ok(SingleTrialOutcome::StarSolverRetry {
                        dt_used,
                        cause,
                    });
                }
            };
            if conv_a.dt < 0.999_999 * dt_used {
                return Ok(SingleTrialOutcome::CutDt(conv_a.dt));
            }
            // `binary/private/binary_evolve.f90:270, 308-310`: sync accretor
            // `component_mdot` and `mdot_system_wind` when `part2` revised it.
            if s2.mstar_dot.to_bits() != acct.component_mdot[1].to_bits() {
                let delta_a = s2.mstar_dot - acct.component_mdot[1];
                let fixed_xfer = (1.0
                    - self.controls.mass_transfer_alpha
                    - self.controls.mass_transfer_beta
                    - self.controls.mass_transfer_delta)
                    .clamp(0.0, 1.0);
                acct.mdot_wind_intrinsic[1] += delta_a;
                acct.component_mdot[1] = s2.mstar_dot;
                acct.mdot_system_wind[1] = s2.mstar_dot
                    + mtr * fixed_xfer
                    + acct.mdot_wind_transfer[0]
                    - acct.mdot_wind_transfer[1];
            }
            acc_conv = Some(conv_a);
        }

        let am_rem_d = eval_star_am_removed_post_step(&self.donor);
        let am_rem_a = match &self.companion {
            BinaryCompanion::Star(s2) => eval_star_am_removed_post_step(s2),
            BinaryCompanion::PointMass { .. } => 0.0,
        };

        if self.controls.do_tidal_sync {
            self.t_sync = self.tide_sync_scratch.load();
        }

        if let BinaryCompanion::PointMass { ref mut mass_g, .. } =
            self.companion
        {
            *mass_g =
                (m_a_old + acct.retained_accretion_rate * dt_used).max(1.0e20);
        }

        let (m_d_new, m_a_new) = (self.donor.mstar, self.accretor_mass_g());
        let s_post_d = eval_star_spin_j(&self.donor);
        let s_post_a = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => eval_star_spin_j(s2),
        };

        let acc_ref = match &self.companion {
            BinaryCompanion::PointMass { .. } => None,
            BinaryCompanion::Star(s2) => Some(&**s2),
        };
        let mtr_eff = if self.ignore_rlof_flag {
            0.0
        } else {
            trial_mdot.min(0.0)
        };
        let old_mb = self.using_jdot_mb;
        let last_jb = eval_gated_jdot_step(
            &self.controls,
            doing_first_model_of_run,
            &self.donor,
            acc_ref,
            self.donor_i,
            m_d_new,
            m_a_new,
            self.rl[0],
            self.rl[1],
            a_old,
            p_old,
            e_old,
            j_old,
            mtr_eff,
            acct.mdot_system_transfer[0],
            acct.mdot_system_wind[0],
            acct.mdot_system_transfer[1],
            acct.mdot_system_wind[1],
            acct.mdot_system_cct,
            s_pre_d,
            s_post_d,
            am_rem_d,
            s_pre_a,
            s_post_a,
            am_rem_a,
            dt_used,
            0.0,
            old_mb,
            &mut self.using_jdot_mb,
            &mut self.change_factor,
        );

        // `#175` (`binary/private/binary_evolve.f90:320-328`): upstream uses a
        // single explicit first-order Euler orbital update (`DECISIONS #36`) and
        // sets `have_to_reduce_timestep_due_to_j = .true.` + returns `retry`
        // (multiplying `timestep_factor_for_retries * dt_reduction_factor_for_j`)
        // when `J_orb <= 0`.
        // `binary_evolve.f90:321`: `jdot*time_step*secyer` with
        // `time_step = dt/secyer` (`binary_evolve.f90:276`,
        // `star/private/evolve.f90:1931`).
        let time_step_yr = dt_used / SECYER;
        let j_cur = j_old + last_jb.jdot_total * time_step_yr * SECYER;
        if j_cur <= 0.0 || !j_cur.is_finite() {
            self.have_to_reduce_timestep_due_to_j = true;
            return Ok(SingleTrialOutcome::RetryStep(dt_used));
        }

        // `binary/private/binary_evolve.f90:330-340`: eccentricity is held
        // fixed during `doing_first_model_of_run` with zeroed `edot` terms.
        let (e_cur, last_eb) = if doing_first_model_of_run {
            (e_old, EdotBreakdown::default())
        } else {
            let eb = eval_coupled_edot_breakdown(
                &self.controls,
                &self.donor,
                acc_ref,
                self.donor_i,
                m_d_new,
                m_a_new,
                a_old,
                p_old,
                e_old,
            )?;
            // `binary_evolve.f90:332-334`: same `time_step*secyer` product,
            // then two separate limit tests (not a clamp).
            let mut e_next = e_old + eb.edot_total * time_step_yr * SECYER;
            if e_next < self.controls.min_eccentricity {
                e_next = self.controls.min_eccentricity;
            }
            if e_next > self.controls.max_eccentricity {
                e_next = self.controls.max_eccentricity;
            }
            (e_next, eb)
        };
        let dj_non_ls_accum = (last_jb.jdot_total - last_jb.jdot_ls) * dt_used;

        self.angular_momentum_j = j_cur;
        self.eccentricity = e_cur;
        self.jdot = last_jb.jdot_total;
        self.separation = separation_from_j_orb(m_d_new, m_a_new, j_cur, e_cur);
        self.period = period_from_separation(m_d_new, m_a_new, self.separation);
        let (rl_new, gap_new) = compute_rl_and_gaps(
            m_d_new,
            m_a_new,
            star_photosphere_radius_cm(&self.donor),
            self.accretor_radius_cm(),
            self.separation,
            self.eccentricity,
        );
        if !gap_new[0].is_finite() || !gap_new[1].is_finite() {
            // `binary_evolve.f90:372-381` returns `retry`, so the star's
            // `timestep_factor_for_retries` applies (no J reduction flag).
            return Ok(SingleTrialOutcome::RetryStep(dt_used));
        }
        self.rl = rl_new;
        self.rl_relative_gap = gap_new;
        self.step_mtransfer_rate = mtr_eff;
        self.last_mdot = acct;
        self.last_jdot = last_jb;
        self.last_edot = last_eb;

        // `binary_evolve.f90:385-386`: increment `model_number` and `binary_age`
        // inside `binary_evolve_step` before `binary_check_model` and
        // `extras_binary_check_model` run (`restore_snapshot` rewinds on redo/retry).
        self.model_number += 1;
        self.binary_age_yr += dt_used / SECYER;
        self.refresh_binary_star_baselines();
        self.refresh_tidal_deformation_hooks(false)?;

        let am_wind_d = if self.donor.mstar_dot < 0.0 {
            am_rem_d * (acct.mdot_system_wind[0] / self.donor.mstar_dot).abs()
        } else {
            0.0
        };
        let am_wind_a = match &self.companion {
            BinaryCompanion::Star(s2) if s2.mstar_dot < 0.0 => am_rem_a,
            _ => 0.0,
        };
        let ext_spin_delta =
            if self.controls.do_jdot_ls && !doing_first_model_of_run {
                -(am_wind_d + am_wind_a)
            } else {
                (s_post_d - s_pre_d) + (s_post_a - s_pre_a)
            };
        self.cumulative_j_ext += dj_non_ls_accum + ext_spin_delta;

        let env_d_new =
            (m_d_new / MSUN - compute_core_masses_msun(&self.donor).0).max(0.0);
        let c = &self.controls;
        let ts_ctrl = BinaryTimestepControls {
            donor_time_step_yr: dt_used / SECYER,
            donor_dt_sec: dt_used,
            max_timestep_sec: self.max_timestep_sec,
            time_delta_coeff: c.time_delta_coeff,
            dt_softening_factor: c.dt_softening_factor,
            rl_relative_gap: self.rl_relative_gap,
            rl_relative_gap_old: gap_old,
            angular_momentum_j: self.angular_momentum_j,
            angular_momentum_j_old: j_old,
            separation: self.separation,
            separation_old: a_old,
            eccentricity: self.eccentricity,
            eccentricity_old: e_old,
            env_donor: env_d_new,
            env_donor_old: env_d_old,
            m: [m_d_new, m_a_new],
            m_old: [m_d_old, m_a_old],
            is_point_mass_companion: self.point_mass_i != 0,
            doing_first_model_of_run,
            ignore_hard_limits_this_step: self.ignore_hard_limits_this_step,
            fm: c.fm,
            fm_limit: c.fm_limit,
            fr: c.fr,
            fr_limit: c.fr_limit,
            fr_dt_limit: c.fr_dt_limit,
            fj: c.fj,
            fa: c.fa,
            fe: c.fe,
            fe_limit: c.fe_limit,
            fdm: c.fdm,
        };
        Ok(SingleTrialOutcome::Completed {
            dt_used,
            ts_ctrl,
            donor_conv,
            acc_conv,
        })
    }

    fn finish_accepted_step(
        &mut self,
        mtransfer_rate_old: f64,
        extras: &mut dyn BinaryExtras,
    ) -> Result<(f64, i32), StarError> {
        // The binary timestep was picked in the step loop
        // (`pick_binary_timestep`). The suggested next `dt` is the one the
        // next step's `set_star_timesteps` takes (`binary_timestep.f90:56-60`).
        let star_next = match &self.companion {
            BinaryCompanion::PointMass { .. } => self.donor.dt_next,
            BinaryCompanion::Star(s2) => self.donor.dt_next.min(s2.dt_next),
        };
        let (next_dt, reason) = if self.max_timestep_sec <= star_next {
            (self.max_timestep_sec, self.dt_why_reason)
        } else {
            (star_next, B_TLIM_COMP)
        };

        // `run_binary_support.f90:654-657`: `extras_binary_finish_step` runs
        // before `binary_finish_step` (so any `set_period_eccentricity` or
        // `set_point_mass_i` called by `finish_step` sets
        // `ignore_hard_limits_this_step = true` for the next step).
        match extras.finish_step(self) {
            BinaryExtrasAction::KeepGoing
            | BinaryExtrasAction::Redo
            | BinaryExtrasAction::Retry => {}
            BinaryExtrasAction::Terminate(reason_str) => {
                self.last_termination_reason = Some(reason_str);
            }
            BinaryExtrasAction::Error(msg) => {
                return Err(StarError::ConvergenceFailure(msg));
            }
        }

        if self.last_termination_reason.is_none() {
            self.change_factor = update_change_factor_after_step(
                self.model_number == 1,
                self.change_factor,
                mtransfer_rate_old,
                self.mtransfer_rate,
                &self.controls,
            );
        }
        Ok((next_dt, reason))
    }

    fn record_retry_holds(&self, holds: &mut [(Option<usize>, usize); 2]) {
        let upd = |slot: &mut (Option<usize>, usize), s: &StarModel| {
            let sh = &s.aux.step_hist;
            slot.0 = slot.0.max(sh.model_number_for_last_retry);
            slot.1 = slot.1.max(sh.timestep_hold);
        };
        let d_idx = (self.donor_i - 1).min(1);
        upd(&mut holds[d_idx], &self.donor);
        if let BinaryCompanion::Star(ref s2) = self.companion {
            upd(&mut holds[1 - d_idx], s2);
        }
    }

    fn apply_retry_holds(&mut self, holds: &[(Option<usize>, usize); 2]) {
        let app = |s: &mut StarModel, slot: (Option<usize>, usize)| {
            let sh = &mut s.aux.step_hist;
            sh.model_number_for_last_retry =
                sh.model_number_for_last_retry.max(slot.0);
            sh.timestep_hold = sh.timestep_hold.max(slot.1);
        };
        let d_idx = (self.donor_i - 1).min(1);
        app(&mut self.donor, holds[d_idx]);
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            app(s2, holds[1 - d_idx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inlist::BinaryInlistControls;
    use flexmesa_star::{load_zams_model, StarControls, StarJobControls};

    #[test]
    fn test_eval_irradiation_heat_matches_upstream_partial_cell_straddling() {
        let mut star = load_zams_model(
            StarControls {
                initial_mass: 1.0,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1 Msun ZAMS");
        let xmstar = star.mstar - star.m_center;
        let r0 = star.r[0];
        let f_irr = 1.0e12;
        // Case A: sigma_max smaller than half of cell 0 column depth still heats cell 0 proportionally.
        let dq_small = 0.25 * star.dq[0];
        let sigma_small = dq_small * xmstar / (4.0 * PI * r0 * r0);
        star.controls.irradiation_flux = f_irr;
        star.controls.column_depth_for_irradiation = sigma_small;
        refresh_star_irradiation_heat(&mut star);
        let eps_small = 0.25 * f_irr / sigma_small;
        let expected_0 = eps_small * (dq_small / star.dq[0]);
        assert!(
            (star.aux.irradiation_heat[0] - expected_0).abs()
                <= 1.0e-12 * expected_0.abs()
        );
        assert_eq!(star.aux.irradiation_heat[1], 0.0);

        // Case B: straddling cell 1 heats cell 0 fully and cell 1 by (irradiation_dq - xq1)/dq1.
        let dq_straddle = star.dq[0] + 0.4 * star.dq[1];
        let sigma_straddle = dq_straddle * xmstar / (4.0 * PI * r0 * r0);
        star.controls.column_depth_for_irradiation = sigma_straddle;
        refresh_star_irradiation_heat(&mut star);
        let eps_straddle = 0.25 * f_irr / sigma_straddle;
        let xq1 = 1.0 - star.q[1];
        let expected_1 = eps_straddle * (dq_straddle - xq1) / star.dq[1];
        assert!(
            (star.aux.irradiation_heat[0] - eps_straddle).abs()
                <= 1.0e-12 * eps_straddle.abs()
        );
        assert!(
            (star.aux.irradiation_heat[1] - expected_1).abs()
                <= 1.0e-12 * expected_1.abs()
        );
        assert_eq!(star.aux.irradiation_heat[2], 0.0);
    }

    #[test]
    fn test_implicit_redo_preserves_change_factor_and_final_try_redo_triggers_retry(
    ) {
        struct RedoProbeExtras {
            calls: usize,
            seen_cf: Vec<f64>,
            seen_models: Vec<usize>,
        }

        impl BinaryExtras for RedoProbeExtras {
            fn check_model(
                &mut self,
                bin: &mut BinarySystem,
            ) -> BinaryExtrasAction {
                self.calls += 1;
                self.seen_cf.push(bin.change_factor);
                self.seen_models.push(bin.model_number);
                match self.calls {
                    1 => {
                        // Simulate a mid-trial change_factor update (e.g. boost or MB reset)
                        // that must survive into try 2 rather than being wiped by trial_snap.
                        bin.change_factor = 1.75;
                        BinaryExtrasAction::Redo
                    }
                    2 => {
                        // Final allowed try (try_idx == max_tries + 1 = 2): returning Redo
                        // must convert to outer Retry rather than falling through the loop.
                        BinaryExtrasAction::Redo
                    }
                    _ => BinaryExtrasAction::KeepGoing,
                }
            }
        }

        let donor = load_zams_model(
            StarControls {
                initial_mass: 1.0,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1.0 Msun ZAMS");
        let bctrl = BinaryInlistControls {
            m1_msun: 1.0,
            m2_msun: 0.8,
            initial_period_in_days: 5.0,
            mdot_scheme: String::from("roche_lobe"),
            max_tries_to_achieve: 1,
            initial_change_factor: 1.25,
            max_change_factor: 2.0,
            terminate_if_initial_overflow: false,
            ..BinaryInlistControls::default()
        };
        let mut sys = BinarySystem::from_controls_point_mass(donor, bctrl)
            .expect("detached point-mass binary");
        let mut extras = RedoProbeExtras {
            calls: 0,
            seen_cf: Vec::new(),
            seen_models: Vec::new(),
        };
        let _ = sys
            .try_step_with_extras(SECYER, &mut extras)
            .expect("step succeeds after outer retry on final-try redo");
        assert_eq!(extras.calls, 3);
        assert_eq!(extras.seen_models, vec![1, 1, 1]);
        // Try 1 starts at 1.25, Try 2 sees preserved 1.75 across redo, Outer retry restores 1.25.
        assert!((extras.seen_cf[0] - 1.25).abs() < 1.0e-12);
        assert!((extras.seen_cf[1] - 1.75).abs() < 1.0e-12);
        assert!((extras.seen_cf[2] - 1.25).abs() < 1.0e-12);
        assert_eq!(sys.model_number, 1);
        assert_eq!(sys.counters.num_retries, 1);
    }

    #[test]
    fn test_eval_star_stopping_limits_propagates_redo_retry_terminate_and_err()
    {
        let mut donor = load_zams_model(
            StarControls {
                initial_mass: 1.5,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1.5 Msun donor");
        donor.dt = 1.0e3 * SECYER;
        donor.dt_next = donor.dt;
        let bctrl = BinaryInlistControls {
            m1_msun: 1.5,
            m2_msun: 1.0,
            initial_period_in_days: 5.0,
            terminate_if_initial_overflow: false,
            ..BinaryInlistControls::default()
        };
        let mut sys = BinarySystem::from_controls_point_mass(donor, bctrl)
            .expect("detached binary");

        // 1. KeepGoing
        let (dec, dt_opt, _) = sys
            .eval_star_stopping_limits(false)
            .expect("default keep_going");
        assert_eq!(dec, TrialDecision::KeepGoing);
        assert!(dt_opt.is_none());

        // 2. Redo (target-hitting `when_to_stop_atol` on `max_age`)
        sys.donor.star_age = 100.0;
        sys.donor.controls.max_age = 50.0;
        sys.donor.controls.ts_stop.when_to_stop_atol = 1.0;
        sys.donor.controls.ts_stop.when_to_stop_rtol = 0.0;
        let (dec_redo, dt_redo, _) =
            sys.eval_star_stopping_limits(false).expect("target redo");
        assert_eq!(dec_redo, TrialDecision::Redo);
        assert_eq!(dt_redo, Some(0.5 * sys.donor.dt));

        // 3. Terminate (within tolerance)
        sys.donor.controls.ts_stop.when_to_stop_atol = 100.0;
        let (dec_term, dt_term, _) = sys
            .eval_star_stopping_limits(false)
            .expect("target terminate");
        assert_eq!(dec_term, TrialDecision::Terminate(String::from("max_age")));
        assert!(dt_term.is_none());
        sys.donor.controls.max_age = -1.0;

        // 4. Retry (`retry_for_v_above_clight`)
        sys.donor.controls.v_flag = true;
        sys.donor.controls.ts_stop.retry_for_v_above_clight = true;
        sys.donor.v[0] = 1.1 * flexmesa_const::CLIGHT;
        let (dec_retry, _, _) =
            sys.eval_star_stopping_limits(false).expect("v > c retry");
        assert_eq!(dec_retry, TrialDecision::Retry);
        sys.donor.v[0] = 0.0;
        sys.donor.controls.v_flag = false;

        // 5. Fatal Err (`#264` `peak_burn_vconv_div_cs > 0.75 * limit`)
        let k_pk = sys.donor.nz - 2;
        sys.donor.eps_nuc.fill(0.0);
        sys.donor.eps_nuc[k_pk] = 1.0e10;
        sys.donor.conv_vel[k_pk] = 0.8 * sys.donor.csound[k_pk];
        sys.donor.controls.ts_stop.peak_burn_vconv_div_cs_limit = 1.0;
        let err = sys
            .eval_star_stopping_limits(false)
            .expect_err("#264 fatal error must propagate");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }

    fn point_mass_test_system(max_tries: i64) -> BinarySystem {
        let donor = load_zams_model(
            StarControls {
                initial_mass: 1.0,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1.0 Msun ZAMS");
        let bctrl = BinaryInlistControls {
            m1_msun: 1.0,
            m2_msun: 0.8,
            initial_period_in_days: 5.0,
            max_tries_to_achieve: max_tries,
            terminate_if_initial_overflow: false,
            varcontrol_ms: 3.0e-3,
            ..BinaryInlistControls::default()
        };
        BinarySystem::from_controls_point_mass(donor, bctrl)
            .expect("detached point-mass binary")
    }

    #[test]
    fn test_binary_pick_sets_max_timestep_varcontrol_and_clears_ignore_flag() {
        let mut sys = point_mass_test_system(0);
        sys.ignore_hard_limits_this_step = true;
        let vc0 = sys.donor.controls.varcontrol_target;
        assert!((vc0 - 3.0e-3).abs() > 0.0, "inlist value differs");
        let ctrl = BinaryTimestepControls {
            max_timestep_sec: 10.0 * SECYER,
            ignore_hard_limits_this_step: true,
            ..BinaryTimestepControls::default()
        };
        sys.max_timestep_sec = ctrl.max_timestep_sec;
        let want = binary_pick_next_timestep(&ctrl).expect("finite limits");
        let dec = sys.pick_binary_timestep(&ctrl).expect("pick");
        assert_eq!(dec, TrialDecision::KeepGoing);
        assert_eq!(sys.max_timestep_sec.to_bits(), want.0.to_bits());
        assert_eq!(sys.dt_why_reason, want.1);
        // `binary_timestep.f90:306-309`: detached, donor on the main sequence.
        assert_eq!(sys.donor.controls.varcontrol_target, 3.0e-3);
        assert!(!sys.ignore_hard_limits_this_step);
    }

    #[test]
    fn test_binary_pick_hard_limit_returns_retry_before_any_state_change() {
        let mut sys = point_mass_test_system(0);
        sys.controls.fj_hard = 0.01;
        sys.ignore_hard_limits_this_step = false;
        sys.max_timestep_sec = 7.0;
        let vc0 = sys.donor.controls.varcontrol_target;
        let ctrl = BinaryTimestepControls {
            angular_momentum_j: 0.9e52,
            angular_momentum_j_old: 1.0e52,
            fj: 0.001,
            ..BinaryTimestepControls::default()
        };
        let dec = sys.pick_binary_timestep(&ctrl).expect("pick");
        assert_eq!(dec, TrialDecision::Retry);
        assert!(sys.have_to_reduce_timestep_due_to_j);
        assert_eq!(sys.max_timestep_sec, 7.0);
        assert_eq!(sys.donor.controls.varcontrol_target, vc0);
        assert_eq!(sys.counters.num_hard_limit_retries, 1);
    }

    #[test]
    fn test_redo_trial_runs_binary_pick_and_keeps_varcontrol_and_ignore_reset()
    {
        // `run_binary_support.f90:540-544`: the binary timestep also runs on
        // `redo`. `binary_prepare_to_redo` restores `max_timestep` but not
        // the stars' `varcontrol_target` or `ignore_hard_limits_this_step`.
        struct Probe {
            seen: Vec<(f64, bool, f64)>,
        }
        impl BinaryExtras for Probe {
            fn check_model(
                &mut self,
                bin: &mut BinarySystem,
            ) -> BinaryExtrasAction {
                self.seen.push((
                    bin.donor.controls.varcontrol_target,
                    bin.ignore_hard_limits_this_step,
                    bin.max_timestep_sec,
                ));
                if self.seen.len() == 1 {
                    BinaryExtrasAction::Redo
                } else {
                    BinaryExtrasAction::KeepGoing
                }
            }
        }
        let mut sys = point_mass_test_system(0);
        let vc0 = sys.donor.controls.varcontrol_target;
        assert!(sys.ignore_hard_limits_this_step);
        let mut probe = Probe { seen: Vec::new() };
        let _ = sys
            .try_step_with_extras(SECYER, &mut probe)
            .expect("step succeeds after one redo");
        assert_eq!(probe.seen.len(), 2);
        assert_eq!(probe.seen[0], (vc0, true, 1.0e99));
        // The redo trial starts with the pick's varcontrol and ignore reset,
        // and with max_timestep restored.
        assert_eq!(probe.seen[1], (3.0e-3, false, 1.0e99));
        // The accepted trial's pick stores a new limit (larger than 1e99 on
        // the first model, where every relative change is zero).
        assert_ne!(sys.max_timestep_sec, 1.0e99);
    }

    #[test]
    fn test_set_star_timesteps_caps_dt_next_and_marks_tlim_binary() {
        let mut sys = point_mass_test_system(0);
        sys.donor.dt_next = 100.0;
        sys.donor.why_tlim_neg_x = true;
        sys.max_timestep_sec = 40.0;
        sys.dt_why_reason = crate::timestep::B_TLIM_ROCHE;
        sys.set_star_timesteps();
        assert_eq!(sys.donor.dt_next, 40.0);
        assert!(!sys.donor.why_tlim_neg_x);
        assert_eq!(sys.dt_why_reason, crate::timestep::B_TLIM_ROCHE);

        sys.donor.why_tlim_neg_x = true;
        sys.max_timestep_sec = 50.0;
        sys.set_star_timesteps();
        assert_eq!(sys.donor.dt_next, 40.0);
        assert!(sys.donor.why_tlim_neg_x);
        assert_eq!(sys.dt_why_reason, B_TLIM_COMP);
    }
}
