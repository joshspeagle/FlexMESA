// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled `BinarySystem::step` / `BinarySystem::try_step` explicit and implicit
//! RLOF trial loop, single-trial Henyey + first-order Euler orbital update,
//! irradiation coupling, and binary termination checks
//! (`binary/private/run_binary_support.f90:385-660`,
//! `binary/private/binary_evolve.f90:257-620`, and
//! `binary/private/binary_irradiation.f90:33-60`).

use flexmesa_const::{LSUN, MSUN, PI, SECYER};
use flexmesa_star::{
    LimitCheckOutcome, StarError, StarModel, compute_core_masses_msun,
    do_check_limits,
    eval_photosphere_info, eval_total_energy_integrals,
    remesh_star_before_step, set_hydro_vars, sync_post_henyey_rotation,
};

use crate::extras::{BinaryExtras, BinaryExtrasAction, DefaultBinaryExtras};
use crate::mdot_implicit::{
    ImplicitMdotSolver, ImplicitStepAction, RloIterReportContext,
    update_change_factor_after_step,
};
use crate::orbit::{
    eval_l2_overflow_limit, period_from_separation, separation_from_j_orb,
};
use crate::system::{BinaryCompanion, BinarySystem, compute_rl_and_gaps};
use crate::system_jdot::{
    EdotBreakdown, apply_tidal_spin_sync_step, eval_coupled_edot_breakdown,
    eval_gated_jdot_step, eval_star_am_removed_post_step, eval_star_spin_j,
    star_photosphere_radius_cm,
};
use crate::system_rlof::{
    blend_explicit_mdot, eval_scheme_explicit_mdot_with_steps,
    evolve_single_star_without_term_rewind, prepare_coupled_mdots_with_extras,
    star_exceeds_delta_mdot_hard_limit,
};
use crate::timestep::{
    B_TLIM_COMP, BinaryHardLimitViolation, BinaryTimestepControls,
    binary_pick_next_timestep, check_binary_hard_limits,
};

#[allow(clippy::large_enum_variant)]
enum SingleTrialOutcome {
    Completed {
        dt_used: f64,
        ts_ctrl: BinaryTimestepControls,
    },
    CutDt(f64),
    /// Upstream `retry` raised inside `binary_evolve_step`: the next `dt` is
    /// the carried value times `timestep_factor_for_retries`, and also times
    /// `dt_reduction_factor_for_j` when `have_to_reduce_timestep_due_to_j`
    /// is set.
    RetryStep(f64),
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
        let max_outer_retries = self.donor.controls.ts_stop.retry_limit;

        let check_cut_dt = |dt_cut: f64,
                            cur_dt: f64|
         -> Result<f64, StarError> {
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
        let mut star_limit_redo_cnt = 0_usize;
        // Upstream remeshes once per step in `prepare_for_new_step`
        // (`star/private/evolve.f90:1882-1889`) before `new_generation`
        // stores the old model (`evolve.f90:1895`), and a retry or redo
        // restores that stored model with `set_current_to_old`
        // (`evolve.f90:401`). A retry therefore starts from the remeshed
        // model, never from the mesh of the previous step.
        let mut post_mesh_snap: Option<crate::system_rlof::BinaryStepSnapshot> =
            None;
        let mut pending_star_redo_dt: Option<f64> = None;

        loop {
            if let Some(redo_dt) = pending_star_redo_dt.take() {
                // Star limit-hitting redo (`do_one_utils.f90:1436-1437`,
                // `run_binary_support.f90:585-603`): restore pre-step state
                // and repeat at halved `dt` without counting an outer retry.
                self.restore_snapshot(
                    post_mesh_snap.as_ref().unwrap_or(&pre_step_snap),
                )?;
                self.apply_retry_holds(&retry_holds);
                dt_try = redo_dt;
            } else if outer_retry > 0 {
                self.counters.num_retries += 1;
                self.record_retry_holds(&mut retry_holds);
                self.restore_snapshot(
                    post_mesh_snap.as_ref().unwrap_or(&pre_step_snap),
                )?;
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
                self.refresh_tidal_deformation_hooks(false)?;
                if self.donor.controls.mesh_delta_coeff > 0.0 {
                    let _ = remesh_star_before_step(&mut self.donor, dt_try)?;
                }
                if let BinaryCompanion::Star(ref mut s2) = self.companion {
                    if s2.controls.mesh_delta_coeff > 0.0 {
                        let _ = remesh_star_before_step(s2, dt_try)?;
                    }
                }
                self.refresh_tidal_deformation_hooks(false)?;
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
                    match self.run_single_trial(
                        explicit_trial_mdot,
                        dt_try,
                        extras,
                    ) {
                        Ok(SingleTrialOutcome::Completed {
                            dt_used,
                            ts_ctrl,
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

                            if matches!(
                                decision,
                                TrialDecision::KeepGoing | TrialDecision::Redo
                            ) {
                                if let Some(viol) =
                                    self.check_hard_limits(&ts_ctrl)
                                {
                                    self.counters.num_hard_limit_retries += 1;
                                    if viol.reduces_timestep_due_to_j() {
                                        self.have_to_reduce_timestep_due_to_j =
                                            true;
                                    }
                                    decision = TrialDecision::Retry;
                                }
                            }
                            let mut dmdot_hard = false;
                            if decision == TrialDecision::KeepGoing {
                                dmdot_hard = self.any_star_exceeds_dmdot_hard();
                                if dmdot_hard {
                                    self.counters.num_hard_limit_retries += 1;
                                    decision = TrialDecision::Retry;
                                }
                            }

                            match decision {
                                TrialDecision::KeepGoing => {
                                    return self.finish_accepted_step(
                                        dt_used,
                                        &ts_ctrl,
                                        pre_step_snap.mtransfer_rate,
                                        extras,
                                    );
                                }
                                TrialDecision::Terminate(reason) => {
                                    self.last_termination_reason = Some(reason);
                                    return self.finish_accepted_step(
                                        dt_used,
                                        &ts_ctrl,
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
                                            return Err(StarError::Terminated {
                                                model_number: target_model,
                                                reason: String::from(
                                                    "redo_limit",
                                                ),
                                            });
                                        }
                                        if !(dt_new.is_finite()
                                            && dt_new > 0.0)
                                        {
                                            self.restore_snapshot(
                                                &pre_step_snap,
                                            )?;
                                            return Err(StarError::Terminated {
                                                model_number: target_model,
                                                reason: String::from(
                                                    "dt_is_zero",
                                                ),
                                            });
                                        }
                                        dt_try = dt_new;
                                    }
                                    // Upstream's `step_loop` has no limit on
                                    // binary-extras redos (`binary/private/
                                    // run_binary_support.f90:385-660`).
                                    redo_idx += 1;
                                }
                                TrialDecision::Retry => {
                                    let next_dt = self.compute_retry_dt(
                                        dt_used,
                                        retry_factor_base,
                                        dt_floor,
                                        outer_retry,
                                        max_outer_retries,
                                        &pre_step_snap,
                                    )?;
                                    if tracing {
                                        emit_hard_limit_retry_trace(
                                            target_model,
                                            outer_retry,
                                            dt_used,
                                            next_dt,
                                            dmdot_hard,
                                        );
                                    }
                                    dt_try = next_dt;
                                    outer_retry += 1;
                                    break;
                                }
                            }
                        }
                        Ok(SingleTrialOutcome::CutDt(dt_cut)) => {
                            self.counters.num_dt_cuts += 1;
                            self.have_to_reduce_timestep_due_to_j = false;
                            dt_try = match check_cut_dt(dt_cut, dt_try) {
                                Ok(d) => d,
                                Err(e) => {
                                    self.restore_snapshot(&pre_step_snap)?;
                                    return Err(e);
                                }
                            };
                            outer_retry += 1;
                            if max_outer_retries > 0
                                && outer_retry > max_outer_retries
                            {
                                self.restore_snapshot(&pre_step_snap)?;
                                return Err(StarError::ConvergenceFailure(
                                    format!(
                                        "BinarySystem::try_step exceeded retry_limit ({max_outer_retries})"
                                    ),
                                ));
                            }
                            break;
                        }
                        Ok(SingleTrialOutcome::RetryStep(dt_used)) => {
                            self.counters.num_dt_cuts += 1;
                            dt_try = self.compute_retry_dt(
                                dt_used,
                                retry_factor_base,
                                dt_floor,
                                outer_retry,
                                max_outer_retries,
                                &pre_step_snap,
                            )?;
                            outer_retry += 1;
                            break;
                        }
                        Err(e) if is_binary_trial_retriable(&e) => {
                            dt_try = self.compute_retry_dt_with_cause(
                                dt_try,
                                retry_factor_base,
                                dt_floor,
                                outer_retry,
                                max_outer_retries,
                                &pre_step_snap,
                                Some(e),
                            )?;
                            outer_retry += 1;
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
            for try_idx in 1..=(max_tries + 1) {
                if try_idx > 1 {
                    // `binary_evolve.f90:615-617`: `binary_set_current_to_old`
                    // restores `change_factor_old` only on outer retry
                    // (`num_tries == 0`), not on implicit RLOF redo trials.
                    let live_cf = self.change_factor;
                    self.restore_snapshot(&trial_snap)?;
                    self.change_factor = live_cf;
                }
                // `run_binary_support.f90:421-424`: preserve rotational implicit
                // wind state and double `change_factor` across subsequent trials.
                if self.donor_started_implicit_wind {
                    self.donor.aux.step_hist.was_in_implicit_wind_limit = true;
                    self.change_factor = 2.0 * self.controls.max_change_factor;
                }
                self.counters.num_implicit_trials += 1;
                let (dt_used, ts_ctrl) = match self
                    .run_single_trial(trial_mdot, dt_try, extras)
                {
                    Ok(SingleTrialOutcome::Completed { dt_used, ts_ctrl }) => {
                        if try_idx > 1 && dt_used < 0.999_999 * dt_try {
                            self.counters.num_dt_cuts += 1;
                            let next_dt = match check_cut_dt(dt_used, dt_try) {
                                Ok(d) => d,
                                Err(e) => {
                                    self.restore_snapshot(&pre_step_snap)?;
                                    return Err(e);
                                }
                            };
                            dt_try = next_dt;
                            outer_retry += 1;
                            if max_outer_retries > 0
                                && outer_retry > max_outer_retries
                            {
                                self.restore_snapshot(&pre_step_snap)?;
                                return Err(StarError::ConvergenceFailure(
                                    format!(
                                        "BinarySystem::try_step exceeded retry_limit ({max_outer_retries})"
                                    ),
                                ));
                            }
                            break;
                        }
                        (dt_used, ts_ctrl)
                    }
                    Ok(SingleTrialOutcome::RetryStep(dt_used)) => {
                        self.counters.num_dt_cuts += 1;
                        let next_dt = self.compute_retry_dt(
                            dt_used,
                            retry_factor_base,
                            dt_floor,
                            outer_retry,
                            max_outer_retries,
                            &pre_step_snap,
                        )?;
                        if tracing {
                            emit_dt_cut_trace(
                                target_model,
                                outer_retry,
                                try_idx,
                                dt_try,
                                next_dt,
                            );
                        }
                        dt_try = next_dt;
                        outer_retry += 1;
                        break;
                    }
                    Ok(SingleTrialOutcome::CutDt(dt_cut)) => {
                        self.counters.num_dt_cuts += 1;
                        self.have_to_reduce_timestep_due_to_j = false;
                        let next_dt = match check_cut_dt(dt_cut, dt_try) {
                            Ok(d) => d,
                            Err(e) => {
                                self.restore_snapshot(&pre_step_snap)?;
                                return Err(e);
                            }
                        };
                        if tracing {
                            emit_dt_cut_trace(
                                target_model,
                                outer_retry,
                                try_idx,
                                dt_try,
                                next_dt,
                            );
                        }
                        dt_try = next_dt;
                        outer_retry += 1;
                        if max_outer_retries > 0
                            && outer_retry > max_outer_retries
                        {
                            self.restore_snapshot(&pre_step_snap)?;
                            return Err(StarError::ConvergenceFailure(
                                format!(
                                    "BinarySystem::try_step exceeded retry_limit ({max_outer_retries})"
                                ),
                            ));
                        }
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
                        dt_try = self.compute_retry_dt_with_cause(
                            dt_try,
                            retry_factor_base,
                            dt_floor,
                            outer_retry,
                            max_outer_retries,
                            &pre_step_snap,
                            Some(e),
                        )?;
                        outer_retry += 1;
                        break;
                    }
                };
                if try_idx == 1 && dt_used < 0.999_999 * dt_try {
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

                // `run_binary_support.f90:540-544`: check binary hard limits on
                // both `keep_going` and `redo` to stop the implicit solver
                // immediately if a hard limit is violated.
                if matches!(
                    decision,
                    TrialDecision::KeepGoing | TrialDecision::Redo
                ) {
                    if let Some(viol) = self.check_hard_limits(&ts_ctrl) {
                        self.counters.num_hard_limit_retries += 1;
                        if viol.reduces_timestep_due_to_j() {
                            self.have_to_reduce_timestep_due_to_j = true;
                        }
                        decision = TrialDecision::Retry;
                    }
                }
                let mut dmdot_hard = false;
                if decision == TrialDecision::KeepGoing {
                    dmdot_hard = self.any_star_exceeds_dmdot_hard();
                    if dmdot_hard {
                        self.counters.num_hard_limit_retries += 1;
                        decision = TrialDecision::Retry;
                    }
                }

                match decision {
                    TrialDecision::KeepGoing => {
                        return self.finish_accepted_step(
                            dt_used,
                            &ts_ctrl,
                            pre_step_snap.mtransfer_rate,
                            extras,
                        );
                    }
                    TrialDecision::Terminate(reason) => {
                        if self.mtransfer_rate > 0.0 {
                            self.mtransfer_rate = 0.0;
                        }
                        self.last_termination_reason = Some(reason);
                        return self.finish_accepted_step(
                            dt_used,
                            &ts_ctrl,
                            pre_step_snap.mtransfer_rate,
                            extras,
                        );
                    }
                    TrialDecision::Redo => {
                        if let Some((dt_new, max_redos)) = star_redo_req {
                            star_limit_redo_cnt += 1;
                            if max_redos > 0
                                && star_limit_redo_cnt > max_redos
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
                            pending_star_redo_dt = Some(dt_new);
                            break;
                        }
                        if try_idx > max_tries {
                            let next_dt = self.compute_retry_dt(
                                dt_used,
                                retry_factor_base,
                                dt_floor,
                                outer_retry,
                                max_outer_retries,
                                &pre_step_snap,
                            )?;
                            dt_try = next_dt;
                            outer_retry += 1;
                            break;
                        }
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
                    }
                    TrialDecision::Retry => {
                        let next_dt = self.compute_retry_dt(
                            dt_used,
                            retry_factor_base,
                            dt_floor,
                            outer_retry,
                            max_outer_retries,
                            &pre_step_snap,
                        )?;
                        if tracing && dmdot_hard {
                            emit_hard_limit_retry_trace(
                                target_model,
                                outer_retry,
                                dt_used,
                                next_dt,
                                dmdot_hard,
                            );
                        }
                        dt_try = next_dt;
                        outer_retry += 1;
                        break;
                    }
                }
            }
        }
    }

    fn compute_retry_dt(
        &mut self,
        cur_dt: f64,
        retry_factor_base: f64,
        dt_floor: f64,
        outer_retry: usize,
        max_outer_retries: usize,
        pre_step_snap: &crate::system_rlof::BinaryStepSnapshot,
    ) -> Result<f64, StarError> {
        self.compute_retry_dt_with_cause(
            cur_dt,
            retry_factor_base,
            dt_floor,
            outer_retry,
            max_outer_retries,
            pre_step_snap,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_retry_dt_with_cause(
        &mut self,
        cur_dt: f64,
        retry_factor_base: f64,
        dt_floor: f64,
        outer_retry: usize,
        max_outer_retries: usize,
        pre_step_snap: &crate::system_rlof::BinaryStepSnapshot,
        cause: Option<StarError>,
    ) -> Result<f64, StarError> {
        // The star retry scales `dt` by `timestep_factor_for_retries` first;
        // `set_star_timesteps` then multiplies the result by
        // `dt_reduction_factor_for_j` (`binary/private/
        // binary_timestep.f90:86-95`).
        let mut next_dt = cur_dt * retry_factor_base;
        if self.have_to_reduce_timestep_due_to_j {
            next_dt *= self.controls.dt_reduction_factor_for_j;
            self.have_to_reduce_timestep_due_to_j = false;
        }
        let next_retries = outer_retry + 1;
        let exhausted = (max_outer_retries > 0
            && next_retries > max_outer_retries)
            || !next_dt.is_finite()
            || next_dt <= 0.0
            || (dt_floor > 0.0 && next_dt < dt_floor);
        if exhausted {
            self.restore_snapshot(pre_step_snap)?;
            if let Some(err) = cause {
                return Err(err);
            }
            return Err(StarError::ConvergenceFailure(format!(
                "BinarySystem::try_step exhausted outer timestep retries (retry {next_retries}, next_dt={next_dt:.3e} s, min_timestep_limit={dt_floor:.3e} s)"
            )));
        }
        Ok(next_dt)
    }

    fn check_hard_limits(
        &self,
        ts_ctrl: &BinaryTimestepControls,
    ) -> Option<BinaryHardLimitViolation> {
        check_binary_hard_limits(
            ts_ctrl,
            self.controls.fm_hard,
            self.controls.fr_hard,
            self.controls.fj_hard,
            self.controls.fa_hard,
            self.controls.fe_hard,
            self.controls.fdm_hard,
        )
    }

    fn any_star_exceeds_dmdot_hard(&self) -> bool {
        star_exceeds_delta_mdot_hard_limit(&self.donor)
            || matches!(
                &self.companion,
                BinaryCompanion::Star(s2)
                    if star_exceeds_delta_mdot_hard_limit(s2)
            )
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
        extras: &mut dyn BinaryExtras,
    ) -> Result<SingleTrialOutcome, StarError> {
        let doing_first_model_of_run = self.model_number == 0;
        self.t_sync = [0.0; 2];
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
        let mut acct = prepare_coupled_mdots_with_extras(
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
        )?;

        refresh_star_irradiation_heat(&mut self.donor);
        let dt_used = evolve_single_star_without_term_rewind(
            &mut self.donor,
            dt_try,
            false,
        )?
        .dt;
        let extra_w_d =
            (self.donor.mstar_dot - acct.component_mdot[0]).min(0.0);
        if extra_w_d < 0.0 {
            acct.mdot_wind_intrinsic[0] += extra_w_d;
            let dw_tr = extra_w_d * acct.wind_xfer_fraction[0];
            acct.mdot_wind_transfer[0] += dw_tr;
            acct.mdot_system_wind[0] += extra_w_d - dw_tr;
            acct.component_mdot[0] = self.donor.mstar_dot;
        }
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            let rep_a = evolve_single_star_without_term_rewind(
                s2,
                dt_used,
                false,
            )?;
            if rep_a.dt < 0.999_999 * dt_used {
                return Ok(SingleTrialOutcome::CutDt(rep_a.dt));
            }
            let extra_w_a = (s2.mstar_dot - acct.component_mdot[1]).min(0.0);
            if extra_w_a < 0.0 {
                acct.mdot_wind_intrinsic[1] += extra_w_a;
                let dw_tr = extra_w_a * acct.wind_xfer_fraction[1];
                acct.mdot_wind_transfer[1] += dw_tr;
                acct.mdot_system_wind[1] += extra_w_a - dw_tr;
                acct.component_mdot[1] = s2.mstar_dot;
            }
        }

        let am_rem_d = eval_star_am_removed_post_step(&self.donor);
        let am_rem_a = match &self.companion {
            BinaryCompanion::Star(s2) => eval_star_am_removed_post_step(s2),
            BinaryCompanion::PointMass { .. } => 0.0,
        };

        if self.controls.do_tidal_sync {
            let (st_d, sm_d, ft_d, st_a, sm_a, ft_a) = if self.donor_i == 1 {
                (
                    self.controls.sync_type_1.as_str(),
                    self.controls.sync_mode_1.as_str(),
                    self.controls.ftid_1,
                    self.controls.sync_type_2.as_str(),
                    self.controls.sync_mode_2.as_str(),
                    self.controls.ftid_2,
                )
            } else {
                (
                    self.controls.sync_type_2.as_str(),
                    self.controls.sync_mode_2.as_str(),
                    self.controls.ftid_2,
                    self.controls.sync_type_1.as_str(),
                    self.controls.sync_mode_1.as_str(),
                    self.controls.ftid_1,
                )
            };
            let ts_d = apply_tidal_spin_sync_step(
                &mut self.donor,
                doing_first_model_of_run,
                st_d,
                sm_d,
                ft_d,
                self.controls.tidal_reduction,
                m_a_old,
                a_old,
                p_old,
                e_old,
                dt_used,
            )?;
            let ts_a = if let BinaryCompanion::Star(ref mut s2) = self.companion
            {
                apply_tidal_spin_sync_step(
                    s2,
                    doing_first_model_of_run,
                    st_a,
                    sm_a,
                    ft_a,
                    self.controls.tidal_reduction,
                    m_d_old,
                    a_old,
                    p_old,
                    e_old,
                    dt_used,
                )?
            } else {
                0.0
            };
            self.t_sync = if self.donor_i == 1 {
                [ts_d, ts_a]
            } else {
                [ts_a, ts_d]
            };
        }

        finalize_star_post_binary_torques(&mut self.donor)?;
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            finalize_star_post_binary_torques(s2)?;
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
        self.refresh_tidal_deformation_hooks(false)?;

        // `binary_evolve.f90:385-386`: increment `model_number` and `binary_age`
        // inside `binary_evolve_step` before `binary_check_model` and
        // `extras_binary_check_model` run (`restore_snapshot` rewinds on redo/retry).
        self.model_number += 1;
        self.binary_age_yr += dt_used / SECYER;

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
        Ok(SingleTrialOutcome::Completed { dt_used, ts_ctrl })
    }

    fn finish_accepted_step(
        &mut self,
        _dt_used: f64,
        ts_ctrl: &BinaryTimestepControls,
        mtransfer_rate_old: f64,
        extras: &mut dyn BinaryExtras,
    ) -> Result<(f64, i32), StarError> {
        // `binary_timestep.f90:99-321`: `binary_pick_next_timestep` runs on
        // every accepted step (including `doing_first_model_of_run`), updates
        // `varcontrol_target`, and clears `ignore_hard_limits_this_step`.
        let (bin_max_dt, bin_reason) = binary_pick_next_timestep(ts_ctrl);
        self.max_timestep_sec = bin_max_dt;
        self.update_stage_varcontrol();
        self.ignore_hard_limits_this_step = false;

        let star_next = match &self.companion {
            BinaryCompanion::PointMass { .. } => self.donor.dt_next,
            BinaryCompanion::Star(s2) => self.donor.dt_next.min(s2.dt_next),
        };
        let (next_dt, reason) = if bin_max_dt <= star_next {
            (bin_max_dt, bin_reason)
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

fn finalize_star_post_binary_torques(
    star: &mut StarModel,
) -> Result<(), StarError> {
    star.mstar_start = star.mstar;
    if !star.controls.rsp_flag {
        star.sync_mesh_and_pack()?;
        sync_post_henyey_rotation(star, 0.0);
        set_hydro_vars(star, false, false)?;
        sync_post_henyey_rotation(star, 0.0);
        star.aux.total_energy = eval_total_energy_integrals(star).5;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inlist::BinaryInlistControls;
    use flexmesa_star::{StarControls, StarJobControls, load_zams_model};

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
    fn test_implicit_redo_preserves_change_factor_and_final_try_redo_triggers_retry()
     {
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
        let (dec_redo, dt_redo, _) = sys
            .eval_star_stopping_limits(false)
            .expect("target redo");
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
        let (dec_retry, _, _) = sys
            .eval_star_stopping_limits(false)
            .expect("v > c retry");
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
}
