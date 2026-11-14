// SPDX-License-Identifier: LGPL-3.0-or-later
//! Top-level adaptive timestep controller (`timestep_controller` and
//! `do_timestep_limits` in `star/private/timestep.f90:39-381`).

use std::f64::consts::LN_10;

use flexmesa_const::SECYER;

use crate::model::StarModel;
use crate::timestep::{
    eval_h211_factor, eval_varcontrol, filter_limit, TimestepAction,
    TimestepLimitReason, TimestepReport,
};
use crate::timestep_checks::{
    check_change, check_dx_limits, check_dx_nuc_drop_full, check_integer_limit,
    check_luminosity_and_hr_limits, check_structural_limits,
    collect_mixing_boundary_qs,
};

fn check_courant_collapse_and_energy(
    model: &StarModel,
    skip_hard_limit: bool,
) -> Result<(f64, f64, f64), String> {
    let ts = &model.controls.ts_stop;
    let coeff = ts.time_delta_coeff;
    let (dt, nz) = (model.dt, model.nz);

    let r_mass = if model.mstar > 0.0 && model.mstar_old > 0.0 {
        check_change(
            (model.mstar / model.mstar_old).log10().abs(),
            ts.delta_lg_star_mass_limit,
            ts.delta_lg_star_mass_hard_limit,
            coeff,
            "delta_lg_star_mass",
            skip_hard_limit,
        )?
    } else {
        0.0
    };
    if model.aux.doing_relax {
        return Ok((r_mass, 0.0, 0.0));
    }

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

    let mut r_courant = 0.0_f64;
    if (ts.dt_div_min_dr_div_cs_limit > 0.0
        || ts.dt_div_min_dr_div_cs_hard_limit > 0.0)
        && nz > 0
    {
        let min_dt_cs = (0..nz)
            .filter_map(|k| {
                let r_in = if k + 1 < nz {
                    model.r[k + 1]
                } else {
                    model.r_center
                };
                let dr = model.r[k] - r_in;
                (dr > 0.0).then(|| dr / model.csound[k].max(1.0))
            })
            .fold(f64::INFINITY, f64::min);
        if min_dt_cs.is_finite() && min_dt_cs > 1.0e-30 {
            r_courant = r_courant.max(check_change(
                dt / min_dt_cs,
                ts.dt_div_min_dr_div_cs_limit,
                ts.dt_div_min_dr_div_cs_hard_limit,
                coeff,
                "dt_div_min_dr_div_cs",
                skip_hard_limit,
            )?);
        }
    }

    if (ts.dt_div_dt_cell_collapse_limit > 0.0
        || ts.dt_div_dt_cell_collapse_hard_limit > 0.0)
        && (model.controls.v_flag || model.controls.u_flag)
        && nz > 1
    {
        let vel = if model.controls.u_flag {
            &model.u
        } else {
            &model.v
        };
        let min_t_coll = (1..nz)
            .filter_map(|k| {
                let r_in = if k + 1 < nz {
                    model.r[k + 1]
                } else {
                    model.r_center
                };
                let v_in = if k + 1 < nz { vel[k + 1] } else { 0.0 };
                let (dv, dr) = (v_in - vel[k], model.r[k] - r_in);
                (dv > 0.0 && dr > 0.0).then_some(dr / dv)
            })
            .fold(f64::INFINITY, f64::min);
        if min_t_coll.is_finite() && min_t_coll > 1.0e-30 {
            r_courant = r_courant.max(check_change(
                dt / min_t_coll,
                ts.dt_div_dt_cell_collapse_limit,
                ts.dt_div_dt_cell_collapse_hard_limit,
                coeff,
                "dt_div_dt_cell_collapse",
                skip_hard_limit,
            )?);
        }
    }
    Ok((r_mass, r_erg, r_courant))
}

/// Evaluates all soft and hard timestep limits (`timestep_controller` and
/// `do_timestep_limits` in `star/private/timestep.f90:39-381`), updating
/// `model.dt_next` and `model.dt_limit_ratio_old` when accepted.
pub fn timestep_controller(
    model: &mut StarModel,
) -> (TimestepAction, TimestepReport) {
    let c = &model.controls;
    let ts = &c.ts_stop;
    let sh = &model.aux.step_hist;

    if c.varcontrol_target < ts.min_allowed_varcontrol_target {
        let rep = TimestepReport {
            dt_next: model.dt,
            varcontrol: 0.0,
            dt_limit_ratio: 0.0,
            reason: TimestepLimitReason::Varcontrol,
        };
        return (
            TimestepAction::Terminate(String::from(
                "varcontrol_target < min_allowed_varcontrol_target",
            )),
            rep,
        );
    }

    let skip_hard_limit = if ts.never_skip_hard_limits {
        false
    } else {
        sh.timestep_hold >= model.model_number
            || (ts.relax_hard_limits_after_retry
                && sh.model_number_for_last_retry == Some(model.model_number))
    };

    let varcontrol = eval_varcontrol(model);
    let vc_target = c.varcontrol_target.max(1.0e-99);
    let r_vc = varcontrol / vc_target;
    if r_vc > ts.varcontrol_dt_limit_ratio_hard_max {
        let rep = TimestepReport {
            dt_next: model.dt * ts.timestep_factor_for_retries,
            varcontrol,
            dt_limit_ratio: r_vc,
            reason: TimestepLimitReason::Varcontrol,
        };
        return (
            TimestepAction::Retry(String::from(
                "varcontrol_dt_limit_ratio_hard_max",
            )),
            rep,
        );
    }

    let retry_rep = |msg: String, reason: TimestepLimitReason| {
        (
            TimestepAction::Retry(msg),
            TimestepReport {
                dt_next: model.dt * ts.timestep_factor_for_retries,
                varcontrol,
                dt_limit_ratio: r_vc,
                reason,
            },
        )
    };

    let solver_lim = if c.use_gold2_tolerances {
        ts.gold2_solver_iters_timestep_limit
    } else if c.use_gold_tolerances {
        ts.gold_solver_iters_timestep_limit
    } else {
        ts.solver_iters_timestep_limit
    };

    let mut r_sub = match check_integer_limit(
        solver_lim,
        -1,
        sh.last_solver_iterations,
        "num_solver_iterations",
        skip_hard_limit,
    ) {
        Ok(r) => r,
        Err(m) => return retry_rep(m, TimestepLimitReason::SolverOrSubsteps),
    };
    if c.op_split_burn {
        match check_integer_limit(
            ts.burn_steps_limit,
            c.burn_steps_hard_limit as isize,
            sh.burn_max_num_substeps,
            "num_burn_solver_steps",
            skip_hard_limit,
        ) {
            Ok(r) => r_sub = r_sub.max(r),
            Err(m) => {
                return retry_rep(m, TimestepLimitReason::SolverOrSubsteps)
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
                    return retry_rep(m, TimestepLimitReason::SolverOrSubsteps)
                }
            }
        }
    }

    let log_max_t =
        model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max) / LN_10;
    let mix_bdy_q = collect_mixing_boundary_qs(model);

    let r_dx = match check_dx_limits(model, skip_hard_limit, &mix_bdy_q) {
        Ok(r) => r,
        Err(m) => return retry_rep(m, TimestepLimitReason::AbundanceChange),
    };
    let r_struct = match check_structural_limits(model, skip_hard_limit) {
        Ok(r) => r,
        Err(m) => return retry_rep(m, TimestepLimitReason::StructuralChange),
    };
    let r_lum =
        match check_luminosity_and_hr_limits(model, skip_hard_limit, log_max_t)
        {
            Ok(r) => r,
            Err(m) => {
                return retry_rep(m, TimestepLimitReason::LuminosityChange)
            }
        };
    let (r_mass, r_erg, r_cour) =
        match check_courant_collapse_and_energy(model, skip_hard_limit) {
            Ok(t) => t,
            Err(m) => {
                return retry_rep(m, TimestepLimitReason::CourantOrCollapse)
            }
        };
    let r_burn = match check_dx_nuc_drop_full(model, skip_hard_limit, log_max_t)
    {
        Ok(r) => r,
        Err(m) => return retry_rep(m, TimestepLimitReason::NuclearBurn),
    };

    let candidates = [
        (r_vc, TimestepLimitReason::Varcontrol),
        (r_sub, TimestepLimitReason::SolverOrSubsteps),
        (r_dx, TimestepLimitReason::AbundanceChange),
        (r_struct, TimestepLimitReason::StructuralChange),
        (r_lum, TimestepLimitReason::LuminosityChange),
        (r_mass, TimestepLimitReason::MassChange),
        (r_erg, TimestepLimitReason::EnergyConservation),
        (r_cour, TimestepLimitReason::CourantOrCollapse),
        (r_burn, TimestepLimitReason::NuclearBurn),
    ];
    let mut dt_limit_ratio = r_vc;
    let mut reason = TimestepLimitReason::Varcontrol;
    for &(r, why) in &candidates[1..] {
        if r > dt_limit_ratio {
            dt_limit_ratio = r;
            reason = why;
        }
    }

    let max_tf = if log_max_t > ts.min_logt_for_max_timestep_factor_at_high_t {
        ts.max_timestep_factor_at_high_t
    } else {
        ts.max_timestep_factor
    };
    let min_tf = ts.min_timestep_factor;

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
        let raw = filter_limit(1.0 / dt_limit_ratio.max(1.0e-10));
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
        let force_min = if ts.force_timestep_min > 0.0 {
            ts.force_timestep_min
        } else if ts.force_timestep_min_years > 0.0 {
            ts.force_timestep_min_years * SECYER
        } else {
            0.0
        };
        if force_min > 0.0 && dt_next < force_min {
            dt_next = (model.dt * ts.force_timestep_min_factor.max(1.0))
                .min(force_min);
        }

        let mut max_dt = if ts.max_timestep > 0.0 {
            ts.max_timestep
        } else {
            f64::INFINITY
        };
        if let Some(yrs) = c.max_years_for_timestep.filter(|&y| y > 0.0) {
            max_dt = max_dt.min(yrs * SECYER);
        }
        if ts.max_timestep_hi_t_limit > 0.0
            && ts.hi_t_max_years_for_timestep > 0.0
            && 10.0_f64.powf(log_max_t) >= ts.max_timestep_hi_t_limit
        {
            max_dt = max_dt.min(ts.hi_t_max_years_for_timestep * SECYER);
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
        return (
            TimestepAction::Terminate(String::from(
                "invalid dt_next in timestep_controller",
            )),
            rep,
        );
    }

    model.dt_next = dt_next;
    model.dt_limit_ratio_old = dt_limit_ratio.max(1.0e-10);
    (
        TimestepAction::KeepGoing,
        TimestepReport {
            dt_next,
            varcontrol,
            dt_limit_ratio,
            reason,
        },
    )
}
