// SPDX-License-Identifier: LGPL-3.0-or-later
//! Top-level adaptive timestep controller (`timestep_controller` and
//! `do_timestep_limits` in `star/private/timestep.f90:39-381`).

use flexmesa_const::{LN10 as LN_10, M_SUN, SECYER};

use crate::controls::StarError;
use crate::inlist_runner::{DefaultStarExtras, ExtrasDirective, StarExtras};
use crate::mesa_history::{eval_min_cell_collapse_time, eval_min_dr_div_cs};
use crate::model::StarModel;
use crate::timestep::{
    TimestepAction, TimestepLimitReason, TimestepReport, eval_h211_factor,
    eval_varcontrol,
};
use crate::timestep_checks::{
    CheckFailure, check_change, check_cntr_abundance_limits, check_dx_limits,
    check_dx_nuc_drop_full, check_integer_limit,
    check_luminosity_and_hr_limits, check_structural_limits,
    collect_mixing_boundary_qs,
};

fn check_mass_mdot_courant_and_energy(
    model: &StarModel,
    skip_hard_limit: bool,
) -> Result<(f64, f64, f64, f64), String> {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let mw = &c.mix_wind;
    let coeff = ts.time_delta_coeff;
    let dt = model.dt;

    // `check_delta_mstar` (`star/private/timestep.f90:1915-1938`)
    let mut r_mass = if model.mstar > 0.0 && model.mstar_old > 0.0 {
        check_change(
            flexmesa_math::log10(model.mstar / model.mstar_old).abs(),
            ts.delta_lg_star_mass_limit,
            ts.delta_lg_star_mass_hard_limit,
            coeff,
            "delta_lg_star_mass_hard_limit",
            skip_hard_limit,
        )?
    } else {
        0.0
    };

    // `check_delta_mdot` (`star/private/timestep.f90:1884-1912`)
    let (mdot, mdot_old) = (model.mstar_dot, model.aux.step_hist.mstar_dot_old);
    let denom = mw.delta_mdot_atol * (M_SUN / SECYER)
        + mw.delta_mdot_rtol * mdot.abs().max(mdot_old.abs());
    if denom > 0.0 {
        let delta_mdot = (mdot - mdot_old).abs() / denom;
        if delta_mdot != 0.0 {
            let lim = mw.delta_mdot_limit * coeff;
            let hard_lim = mw.delta_mdot_hard_limit * coeff;
            if hard_lim > 0.0 && !skip_hard_limit && delta_mdot > hard_lim {
                return Err(String::from("delta_mdot_hard_limit"));
            }
            if lim > 0.0 {
                let r_mdot = delta_mdot / lim;
                if r_mdot > 1.0 {
                    r_mass = r_mass.max(r_mdot);
                }
            }
        }
    }

    if model.aux.doing_relax {
        return Ok((r_mass, 0.0, 0.0, 0.0));
    }

    // `check_adjust_J_q` (`star/private/timestep.f90:1941-1958`)
    let r_jq =
        if c.rotation_flag && mw.do_adjust_j_lost && model.mstar_dot < 0.0 {
            check_change(
                1.0 - model.adjust_j_q,
                1.0 - ts.adjust_j_q_limit,
                1.0 - ts.adjust_j_q_hard_limit,
                coeff,
                "check_adjust_J_q",
                false,
            )?
        } else {
            0.0
        };

    // `check_rel_error_in_energy` (`star/private/timestep.f90:2023-2043`)
    let sh = &model.aux.step_hist;
    let r_erg = if sh.total_energy_end.abs() > 1.0e-30 {
        check_change(
            (sh.error_in_energy_conservation / sh.total_energy_end).abs(),
            ts.limit_for_rel_error_in_energy_conservation,
            ts.hard_limit_for_rel_error_in_energy_conservation,
            coeff,
            "hard_limit_for_rel_error_in_energy_conservation",
            false,
        )?
    } else {
        0.0
    };

    // `check_dt_div_dt_cell_collapse` & `check_dt_div_min_dr_div_cs` (`timestep.f90:2046-2097`)
    let mut r_courant = 0.0_f64;
    let dt_coll = eval_min_cell_collapse_time(model);
    if dt_coll >= 1.0e-30 {
        r_courant = r_courant.max(check_change(
            dt / dt_coll,
            ts.dt_div_dt_cell_collapse_limit,
            ts.dt_div_dt_cell_collapse_hard_limit,
            coeff,
            "dt_div_dt_cell_collapse",
            skip_hard_limit,
        )?);
    }

    if ts.dt_div_min_dr_div_cs_limit > 0.0 {
        let (dt_x, _) = eval_min_dr_div_cs(model);
        if dt_x > 0.0 {
            r_courant = r_courant.max(check_change(
                dt / dt_x,
                ts.dt_div_min_dr_div_cs_limit,
                ts.dt_div_min_dr_div_cs_hard_limit,
                coeff,
                "dt_div_min_dr_div_cs",
                skip_hard_limit,
            )?);
        }
    }

    Ok((r_mass, r_jq, r_erg, r_courant))
}

/// Evaluates all soft and hard timestep limits (`timestep_controller` and
/// `do_timestep_limits` in `star/private/timestep.f90:39-381`), updating
/// `model.dt_next` and `model.dt_limit_ratio_old` when accepted.
pub fn timestep_controller(
    model: &mut StarModel,
) -> (TimestepAction, TimestepReport) {
    let mut extras = DefaultStarExtras;
    match timestep_controller_with_extras(model, &mut extras) {
        Ok((act, rep, _)) => (act, rep),
        Err(e) => (
            TimestepAction::Terminate(e.to_string()),
            TimestepReport {
                dt_next: model.dt,
                varcontrol: 0.0,
                dt_limit_ratio: 0.0,
                reason: TimestepLimitReason::Varcontrol,
            },
        ),
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn timestep_controller_with_extras(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(TimestepAction, TimestepReport, bool), StarError> {
    if model.controls.varcontrol_target
        < model.controls.ts_stop.min_allowed_varcontrol_target
    {
        let rep = TimestepReport {
            dt_next: model.dt,
            varcontrol: 0.0,
            dt_limit_ratio: 0.0,
            reason: TimestepLimitReason::Varcontrol,
        };
        return Ok((
            TimestepAction::Terminate(String::from(
                "varcontrol_target < min_allowed_varcontrol_target",
            )),
            rep,
            false,
        ));
    }

    let skip_hard_limit = if model.controls.ts_stop.never_skip_hard_limits {
        false
    } else {
        model.aux.step_hist.timestep_hold >= model.model_number
            || (model.controls.ts_stop.relax_hard_limits_after_retry
                && model.aux.step_hist.model_number_for_last_retry
                    == Some(model.model_number))
    };

    let varcontrol = eval_varcontrol(model);
    let vc_target = model.controls.varcontrol_target.max(1.0e-99);
    let r_vc = varcontrol / vc_target;
    let retry_fac = model.controls.ts_stop.timestep_factor_for_retries;
    if r_vc > model.controls.ts_stop.varcontrol_dt_limit_ratio_hard_max {
        let rep = TimestepReport {
            dt_next: model.dt * retry_fac,
            varcontrol,
            dt_limit_ratio: r_vc,
            reason: TimestepLimitReason::Varcontrol,
        };
        return Ok((
            TimestepAction::Retry(String::from(
                "varcontrol_dt_limit_ratio_hard_max",
            )),
            rep,
            false,
        ));
    }

    let retry_rep = |msg: String, dt_try: f64, reason: TimestepLimitReason| {
        Ok((
            TimestepAction::Retry(msg),
            TimestepReport {
                dt_next: dt_try * retry_fac,
                varcontrol,
                dt_limit_ratio: r_vc,
                reason,
            },
            false,
        ))
    };

    let log_max_t =
        model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max) / LN_10;
    let mut dt_limit_ratio = r_vc;
    let mut reason = TimestepLimitReason::Varcontrol;

    if !model.doing_first_model_of_run {
        let dt_done = model.dt;
        let mut r_other = 0.0_f64;
        if model.controls.ts_stop.use_other_timestep_limit {
            let (dir, r_ext) =
                extras.other_timestep_limit(model, skip_hard_limit, dt_done);
            match dir {
                ExtrasDirective::KeepGoing => {
                    r_other = r_ext;
                }
                ExtrasDirective::Retry => {
                    return retry_rep(
                        String::from("other_timestep_limit retry"),
                        dt_done,
                        TimestepLimitReason::SolverOrSubsteps,
                    );
                }
                ExtrasDirective::Redo => {
                    return Ok((
                        TimestepAction::KeepGoing,
                        TimestepReport {
                            dt_next: dt_done,
                            varcontrol,
                            dt_limit_ratio: r_vc,
                            reason: TimestepLimitReason::SolverOrSubsteps,
                        },
                        true,
                    ));
                }
                ExtrasDirective::Terminate(code) => {
                    return Ok((
                        TimestepAction::Terminate(code),
                        TimestepReport {
                            dt_next: dt_done,
                            varcontrol,
                            dt_limit_ratio: r_vc,
                            reason: TimestepLimitReason::SolverOrSubsteps,
                        },
                        false,
                    ));
                }
                ExtrasDirective::Error(msg) => {
                    return Err(StarError::ConvergenceFailure(msg));
                }
            }
        }

        let c = &model.controls;
        let ts = &c.ts_stop;
        let sh = &model.aux.step_hist;

        let solver_lim = if model.aux.doing_relax
            && c.relax_solver_iters_timestep_limit > 0
        {
            c.relax_solver_iters_timestep_limit
        } else if model.using_gold_tolerances {
            ts.gold_solver_iters_timestep_limit
        } else {
            ts.solver_iters_timestep_limit
        };
        let solver_iters = if model.used_extra_iter_in_solver_for_accretion {
            sh.last_solver_iterations.saturating_sub(1)
        } else {
            sh.last_solver_iterations
        };
        let mut r_sub = match check_integer_limit(
            solver_lim,
            -1,
            solver_iters,
            "num_solver_iterations",
            skip_hard_limit,
        ) {
            Ok(r) => r.max(r_other),
            Err(m) => {
                return retry_rep(
                    m,
                    dt_done,
                    TimestepLimitReason::SolverOrSubsteps,
                );
            }
        };

        let max_lnt_start = model
            .ln_t_start
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if c.op_split_burn
            && max_lnt_start.is_finite()
            && flexmesa_math::exp(max_lnt_start) >= c.op_split_burn_min_t
        {
            match check_integer_limit(
                ts.burn_steps_limit,
                c.burn_steps_hard_limit as isize,
                sh.burn_max_num_substeps,
                "num_burn_solver_steps",
                skip_hard_limit,
            ) {
                Ok(r) => r_sub = r_sub.max(r),
                Err(m) => {
                    return retry_rep(
                        m,
                        dt_done,
                        TimestepLimitReason::SolverOrSubsteps,
                    );
                }
            }
        }
        if c.do_element_diffusion {
            for res in [
                check_integer_limit(
                    ts.diffusion_steps_limit,
                    ts.diffusion_steps_hard_limit,
                    sh.num_diffusion_solver_steps,
                    "num_diffusion_solver_steps",
                    skip_hard_limit,
                ),
                check_integer_limit(
                    ts.diffusion_iters_limit,
                    ts.diffusion_iters_hard_limit,
                    sh.num_diffusion_solver_iters,
                    "num_diffusion_solver_iters",
                    skip_hard_limit,
                ),
            ] {
                match res {
                    Ok(r) => r_sub = r_sub.max(r),
                    Err(m) => {
                        return retry_rep(
                            m,
                            dt_done,
                            TimestepLimitReason::SolverOrSubsteps,
                        );
                    }
                }
            }
        }

        let mix_bdy_q = collect_mixing_boundary_qs(model);
        let (r_dx_abs, r_dx_rel) = match check_dx_limits(
            model,
            skip_hard_limit,
            &mix_bdy_q,
            log_max_t,
        ) {
            Ok(pair) => pair,
            Err(CheckFailure::Retry(m)) => {
                return retry_rep(
                    m,
                    dt_done,
                    TimestepLimitReason::AbundanceChange,
                );
            }
            Err(CheckFailure::Fatal(e)) => return Err(e),
        };
        let r_struct = match check_structural_limits(model, skip_hard_limit) {
            Ok(r) => r,
            Err(m) => {
                return retry_rep(
                    m,
                    dt_done,
                    TimestepLimitReason::StructuralChange,
                );
            }
        };
        let r_cntr_x = match check_cntr_abundance_limits(model, skip_hard_limit)
        {
            Ok(r) => r,
            Err(m) => {
                return retry_rep(
                    m,
                    dt_done,
                    TimestepLimitReason::AbundanceChange,
                );
            }
        };
        let r_lum = match check_luminosity_and_hr_limits(
            model,
            skip_hard_limit,
            log_max_t,
            &mix_bdy_q,
        ) {
            Ok(r) => r,
            Err(m) => {
                return retry_rep(
                    m,
                    dt_done,
                    TimestepLimitReason::LuminosityChange,
                );
            }
        };
        let (r_mass, r_jq, r_erg, r_cour) =
            match check_mass_mdot_courant_and_energy(model, skip_hard_limit) {
                Ok(t) => t,
                Err(m) => {
                    return retry_rep(
                        m,
                        dt_done,
                        TimestepLimitReason::CourantOrCollapse,
                    );
                }
            };
        let r_burn =
            match check_dx_nuc_drop_full(model, skip_hard_limit, log_max_t) {
                Ok(r) => r,
                Err(m) => {
                    return retry_rep(
                        m,
                        dt_done,
                        TimestepLimitReason::NuclearBurn,
                    );
                }
            };

        let candidates = [
            (r_sub, TimestepLimitReason::SolverOrSubsteps),
            (r_dx_abs.max(r_dx_rel), TimestepLimitReason::AbundanceChange),
            (r_struct.max(r_jq), TimestepLimitReason::StructuralChange),
            (r_cntr_x, TimestepLimitReason::AbundanceChange),
            (r_lum, TimestepLimitReason::LuminosityChange),
            (r_mass, TimestepLimitReason::MassChange),
            (r_cour, TimestepLimitReason::CourantOrCollapse),
            (r_erg, TimestepLimitReason::EnergyConservation),
            (r_burn, TimestepLimitReason::NuclearBurn),
        ];
        for &(r, why) in &candidates {
            if r > dt_limit_ratio {
                dt_limit_ratio = r;
                reason = why;
            }
        }
    }

    let c = &model.controls;
    let ts = &c.ts_stop;
    let max_tf = if log_max_t > ts.min_logt_for_max_timestep_factor_at_high_t {
        ts.max_timestep_factor_at_high_t
    } else {
        ts.max_timestep_factor
    };
    let min_tf = ts.min_timestep_factor;

    // `filter_dt_next` (`star/private/timestep.f90:2362-2427`)
    let (factor, hit_max, hit_min) = if ts.use_dt_low_pass_controller {
        eval_h211_factor(
            dt_limit_ratio,
            model.dt_limit_ratio_old,
            model.dt,
            model.dt_old,
            max_tf,
            min_tf,
        )
    } else {
        let raw = 1.0 / dt_limit_ratio.max(1.0e-10);
        let h_max = max_tf > 0.0 && raw > max_tf;
        let h_min = min_tf > 0.0 && raw < min_tf;
        let mut f = raw;
        if max_tf > 0.0 {
            f = f.min(max_tf);
        }
        if min_tf > 0.0 {
            f = f.max(min_tf);
        }
        (f, h_max, h_min)
    };

    if reason == TimestepLimitReason::Varcontrol {
        if hit_max {
            reason = TimestepLimitReason::MaxTimestepFactor;
        } else if hit_min {
            reason = TimestepLimitReason::MinTimestepFactor;
        }
    }

    let mut dt_next = model.dt * factor;

    let force_dt = if ts.force_timestep > 0.0 {
        ts.force_timestep
    } else if ts.force_timestep_years > 0.0 {
        ts.force_timestep_years * SECYER
    } else {
        0.0
    };
    if force_dt > 0.0 {
        dt_next = force_dt;
        reason = TimestepLimitReason::MaxTimestep;
    } else {
        let mut max_dt = if ts.max_timestep > 0.0 {
            ts.max_timestep
        } else {
            1.0e99
        };
        if c.max_years_for_timestep > 0.0 {
            max_dt = max_dt.min(c.max_years_for_timestep * SECYER);
        }
        if dt_next > max_dt {
            dt_next = max_dt;
            reason = TimestepLimitReason::MaxTimestep;
        }
        if model.aux.step_hist.timestep_hold > model.model_number
            && dt_next > model.dt
        {
            dt_next = model.dt;
            reason = TimestepLimitReason::TimestepHold;
        }
    }

    if !dt_next.is_finite() || dt_next <= 0.0 {
        let rep = TimestepReport {
            dt_next: model.dt,
            varcontrol,
            dt_limit_ratio,
            reason,
        };
        return Ok((
            TimestepAction::Terminate(String::from(
                "invalid dt_next in timestep_controller",
            )),
            rep,
            false,
        ));
    }

    model.dt_next = dt_next;
    model.dt_limit_ratio_old = dt_limit_ratio.max(1.0e-10);
    Ok((
        TimestepAction::KeepGoing,
        TimestepReport {
            dt_next,
            varcontrol,
            dt_limit_ratio,
            reason,
        },
        false,
    ))
}
