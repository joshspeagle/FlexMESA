// SPDX-License-Identifier: GPL-3.0-or-later
//! Top-level adaptive timestep controller (`timestep_controller` and
//! `do_timestep_limits` in `star/private/timestep.f90:39-381`).

use flexmesa_const::SECYER;

use crate::controls::StarError;
use crate::inlist_runner::{DefaultStarExtras, ExtrasDirective, StarExtras};
use crate::model::StarModel;
use crate::timestep::{
    eval_varcontrol, h211_limtr, reason_for_tlim, tlim, DtLimitRatios,
    TimestepAction, TimestepLimitReason, TimestepReport,
};
use crate::timestep_checks::{
    check_adjust_j_q, check_delta_hr, check_delta_lgl, check_delta_mdot,
    check_delta_mstar, check_dl_div_l, check_dlg_e_change, check_dlg_p_change,
    check_dlg_p_cntr_change, check_dlg_r_change, check_dlg_rho_change,
    check_dlg_rho_cntr_change, check_dlg_t_change, check_dlg_t_cntr_change,
    check_dlg_t_max_at_high_t_change, check_dlg_t_max_change,
    check_dlg_teff_change, check_dlog_eps_nuc_change,
    check_dt_div_dt_cell_collapse, check_dt_div_min_dr_div_cs,
    check_dx_div_x_cntr, check_dx_limits, check_dx_nuc_drop,
    check_dye_hight_change, check_integer_limit, check_lg_xc_cntr,
    check_lg_xh_cntr, check_lg_xhe_cntr, check_lg_xne_cntr, check_lg_xo_cntr,
    check_lg_xsi_cntr, check_lgl_h_change, check_lgl_he_change,
    check_lgl_nuc_cat_change, check_lgl_nuc_change,
    check_lgl_power_photo_change, check_lgl_z_change,
    check_rel_error_in_energy, check_xc_cntr, check_xh_cntr, check_xhe_cntr,
    check_xne_cntr, check_xo_cntr, check_xsi_cntr, CheckFailure, LimitCtx,
};

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
                why_tlim: tlim::STRUC,
            },
        ),
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn timestep_controller_with_extras(
    model: &mut StarModel,
    extras: &mut dyn StarExtras,
) -> Result<(TimestepAction, TimestepReport, bool), StarError> {
    let skip_hard_limit = if model.controls.ts_stop.never_skip_hard_limits {
        false
    } else {
        (model.aux.step_hist.timestep_hold > 0
            && model.aux.step_hist.timestep_hold >= model.model_number)
            || (model.controls.ts_stop.relax_hard_limits_after_retry
                && model.aux.step_hist.model_number_for_last_retry
                    == Some(model.model_number))
    };

    let varcontrol = eval_varcontrol(model)?;
    // `check_varcontrol_limit` (`timestep.f90:2244-2290`) evaluates
    // `eval_varcontrol` before testing `min_allowed_varcontrol_target`.
    if model.controls.varcontrol_target
        < model.controls.ts_stop.min_allowed_varcontrol_target
    {
        let rep = TimestepReport {
            dt_next: model.dt,
            varcontrol: 0.0,
            dt_limit_ratio: 0.0,
            reason: TimestepLimitReason::Varcontrol,
            why_tlim: tlim::STRUC,
        };
        return Ok((
            TimestepAction::Terminate(String::from(
                "varcontrol_target < min_allowed_varcontrol_target",
            )),
            rep,
            false,
        ));
    }

    let vc_target = model.controls.varcontrol_target.max(1.0e-99);
    let r_vc = varcontrol / vc_target;
    let retry_fac = model.controls.ts_stop.timestep_factor_for_retries;
    if r_vc > model.controls.ts_stop.varcontrol_dt_limit_ratio_hard_max {
        model.why_tlim = tlim::STRUC;
        let rep = TimestepReport {
            dt_next: model.dt * retry_fac,
            varcontrol,
            dt_limit_ratio: r_vc,
            reason: TimestepLimitReason::Varcontrol,
            why_tlim: tlim::STRUC,
        };
        return Ok((
            TimestepAction::Retry(String::from(
                "varcontrol_dt_limit_ratio_hard_max",
            )),
            rep,
            false,
        ));
    }

    let retry_rep = |msg: String, dt_try: f64, i_limit: usize| {
        Ok((
            TimestepAction::Retry(msg),
            TimestepReport {
                dt_next: dt_try * retry_fac,
                varcontrol,
                dt_limit_ratio: r_vc,
                reason: reason_for_tlim(i_limit),
                why_tlim: i_limit,
            },
            false,
        ))
    };

    // `dt_limit_ratio(1:numTlim)` (`timestep.f90:99`); `Tlim_*` is 1-based.
    let mut dt_limit_ratio: DtLimitRatios = [0.0; tlim::NUM_TLIM];
    dt_limit_ratio[tlim::STRUC - 1] = r_vc;
    let log_max_t = model
        .ln_t
        .iter()
        .take(model.nz)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        / flexmesa_const::LN10;

    if !model.doing_first_model_of_run {
        let dt_done = model.dt;
        if model.controls.ts_stop.use_other_timestep_limit {
            let (dir, r_ext) =
                extras.other_timestep_limit(model, skip_hard_limit, dt_done);
            let why = tlim::OTHER_TIMESTEP_LIMIT;
            let stop_rep = |dt_next: f64| TimestepReport {
                dt_next,
                varcontrol,
                dt_limit_ratio: r_vc,
                reason: reason_for_tlim(why),
                why_tlim: why,
            };
            match dir {
                ExtrasDirective::KeepGoing => {
                    dt_limit_ratio[why - 1] = r_ext;
                }
                ExtrasDirective::Retry => {
                    model.why_tlim = why;
                    return retry_rep(
                        String::from("other_timestep_limit retry"),
                        dt_done,
                        why,
                    );
                }
                ExtrasDirective::Redo => {
                    return Ok((
                        TimestepAction::KeepGoing,
                        stop_rep(dt_done),
                        true,
                    ));
                }
                ExtrasDirective::Terminate(code) => {
                    return Ok((
                        TimestepAction::Terminate(code),
                        stop_rep(dt_done),
                        false,
                    ));
                }
                ExtrasDirective::Error(msg) => {
                    return Err(StarError::ConvergenceFailure(msg));
                }
            }
        }

        let ctx = LimitCtx::new(model, skip_hard_limit);
        let c = &model.controls;
        let ts = &c.ts_stop;
        let sh = &model.aux.step_hist;

        // Each check mirrors one `do_timestep_limits` call: it writes its
        // `Tlim_*` slot and `return_now` reports a hard-limit retry under that
        // index (`timestep.f90:370-379`).
        macro_rules! run {
            ($tl:expr, $call:expr) => {
                match $call {
                    Ok(()) => {}
                    Err(CheckFailure::Retry(m)) => {
                        model.why_tlim = $tl;
                        return retry_rep(m, dt_done, $tl);
                    }
                    Err(CheckFailure::Fatal(e)) => return Err(e),
                }
            };
        }
        macro_rules! slot {
            ($tl:expr) => {
                &mut dt_limit_ratio[$tl - 1]
            };
        }
        let int_limit = |r: Result<f64, String>| r.map_err(CheckFailure::Retry);

        // `check_solver_iters_timestep_limit` (`timestep.f90:427-445`): start
        // from `solver_iters_timestep_limit` (which `relax.f90:3999-4000`
        // replaces with `relax_solver_iters_timestep_limit` while relaxing when
        // that control is nonzero), then let the gold limit override it.
        let mut solver_lim = if model.aux.doing_relax
            && c.relax_solver_iters_timestep_limit != 0
        {
            c.relax_solver_iters_timestep_limit
        } else {
            ts.solver_iters_timestep_limit
        };
        if model.using_gold_tolerances {
            solver_lim = ts.gold_solver_iters_timestep_limit;
        }
        let solver_iters = if model.used_extra_iter_in_solver_for_accretion {
            sh.last_solver_iterations.saturating_sub(1)
        } else {
            sh.last_solver_iterations
        };
        run!(tlim::SOLVER, {
            int_limit(check_integer_limit(
                solver_lim,
                -1,
                solver_iters,
                "num_solver_iterations",
                skip_hard_limit,
            ))
            .map(|r| dt_limit_ratio[tlim::SOLVER - 1] = r)
        });

        // `check_burn_steps_limit` (`timestep.f90:411-424`).
        let max_lnt_start = model
            .ln_t_start
            .iter()
            .take(model.nz)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if c.op_split_burn
            && max_lnt_start.is_finite()
            && flexmesa_math::exp(max_lnt_start) >= c.op_split_burn_min_t
        {
            run!(tlim::NUM_BURN_STEPS, {
                int_limit(check_integer_limit(
                    ts.burn_steps_limit,
                    isize::try_from(c.burn_steps_hard_limit)
                        .unwrap_or(isize::MAX),
                    sh.burn_max_num_substeps,
                    "num_burn_solver_steps",
                    skip_hard_limit,
                ))
                .map(|r| dt_limit_ratio[tlim::NUM_BURN_STEPS - 1] = r)
            });
        }
        // `check_diffusion_steps_limit` and `check_diffusion_iters_limit`
        // (`timestep.f90:448-473`).
        if c.do_element_diffusion {
            run!(tlim::NUM_DIFF_SOLVER_STEPS, {
                int_limit(check_integer_limit(
                    ts.diffusion_steps_limit,
                    ts.diffusion_steps_hard_limit,
                    sh.num_diffusion_solver_steps,
                    "num_diffusion_solver_steps",
                    skip_hard_limit,
                ))
                .map(|r| dt_limit_ratio[tlim::NUM_DIFF_SOLVER_STEPS - 1] = r)
            });
            run!(tlim::NUM_DIFF_SOLVER_ITERS, {
                int_limit(check_integer_limit(
                    ts.diffusion_iters_limit,
                    ts.diffusion_iters_hard_limit,
                    sh.num_diffusion_solver_iters,
                    "num_diffusion_solver_iters",
                    skip_hard_limit,
                ))
                .map(|r| dt_limit_ratio[tlim::NUM_DIFF_SOLVER_ITERS - 1] = r)
            });
        }

        // `check_dX` reports its hard-limit retry under `Tlim_dX_div_X`
        // (`timestep.f90:149`).
        run!(tlim::DX_DIV_X, {
            check_dx_limits(
                model,
                skip_hard_limit,
                &ctx.mix_bdy_q,
                ctx.log_max_t,
            )
            .map(|(dx, dx_div_x)| {
                dt_limit_ratio[tlim::DX - 1] = dx;
                dt_limit_ratio[tlim::DX_DIV_X - 1] = dx_div_x;
            })
        });
        run!(tlim::DL_DIV_L, check_dl_div_l(&ctx, slot!(tlim::DL_DIV_L)));
        run!(tlim::DLGP, check_dlg_p_change(&ctx, slot!(tlim::DLGP)));
        run!(
            tlim::DLGRHO,
            check_dlg_rho_change(&ctx, slot!(tlim::DLGRHO))
        );
        run!(tlim::DLGT, check_dlg_t_change(&ctx, slot!(tlim::DLGT)));
        run!(tlim::DLGE, check_dlg_e_change(&ctx, slot!(tlim::DLGE)));
        run!(tlim::DLGR, check_dlg_r_change(&ctx, slot!(tlim::DLGR)));
        run!(
            tlim::DLGL_NUC_CAT,
            check_lgl_nuc_cat_change(&ctx, slot!(tlim::DLGL_NUC_CAT))
        );
        run!(tlim::DLGL_H, check_lgl_h_change(&ctx, slot!(tlim::DLGL_H)));
        run!(
            tlim::DLGL_HE,
            check_lgl_he_change(&ctx, slot!(tlim::DLGL_HE))
        );
        run!(tlim::DLGL_Z, check_lgl_z_change(&ctx, slot!(tlim::DLGL_Z)));
        run!(
            tlim::DLGL_POWER_PHOTO,
            check_lgl_power_photo_change(&ctx, slot!(tlim::DLGL_POWER_PHOTO))
        );
        run!(
            tlim::DLGL_NUC,
            check_lgl_nuc_change(&ctx, slot!(tlim::DLGL_NUC))
        );
        run!(
            tlim::DLGTEFF,
            check_dlg_teff_change(&ctx, slot!(tlim::DLGTEFF))
        );
        run!(
            tlim::DLGRHO_CNTR,
            check_dlg_rho_cntr_change(&ctx, slot!(tlim::DLGRHO_CNTR))
        );
        run!(
            tlim::DLGT_CNTR,
            check_dlg_t_cntr_change(&ctx, slot!(tlim::DLGT_CNTR))
        );
        run!(
            tlim::DLGT_MAX,
            check_dlg_t_max_change(&ctx, slot!(tlim::DLGT_MAX))
        );
        run!(
            tlim::DLGT_MAX_AT_HIGH_T,
            check_dlg_t_max_at_high_t_change(
                &ctx,
                slot!(tlim::DLGT_MAX_AT_HIGH_T)
            )
        );
        run!(
            tlim::DLGP_CNTR,
            check_dlg_p_cntr_change(&ctx, slot!(tlim::DLGP_CNTR))
        );
        run!(
            tlim::DELTA_YE_HIGH_T,
            check_dye_hight_change(&ctx, slot!(tlim::DELTA_YE_HIGH_T))
        );
        run!(
            tlim::DLOG_EPS_NUC,
            check_dlog_eps_nuc_change(&ctx, slot!(tlim::DLOG_EPS_NUC))
        );
        run!(
            tlim::DX_DIV_X_CNTR,
            check_dx_div_x_cntr(&ctx, slot!(tlim::DX_DIV_X_CNTR))
        );
        run!(
            tlim::LG_XH_CNTR,
            check_lg_xh_cntr(&ctx, slot!(tlim::LG_XH_CNTR))
        );
        run!(
            tlim::LG_XHE_CNTR,
            check_lg_xhe_cntr(&ctx, slot!(tlim::LG_XHE_CNTR))
        );
        run!(
            tlim::LG_XC_CNTR,
            check_lg_xc_cntr(&ctx, slot!(tlim::LG_XC_CNTR))
        );
        run!(
            tlim::LG_XNE_CNTR,
            check_lg_xne_cntr(&ctx, slot!(tlim::LG_XNE_CNTR))
        );
        run!(
            tlim::LG_XO_CNTR,
            check_lg_xo_cntr(&ctx, slot!(tlim::LG_XO_CNTR))
        );
        // Issue #617: upstream `Tlim_lg_XSi_cntr` equals `Tlim_lg_XNe_cntr` and
        // `Tlim_XH_cntr` equals `Tlim_lg_XO_cntr` (`star_data_def.inc`), so the
        // silicon and hydrogen checks assign the slot that the neon and oxygen
        // checks just filled. Reproduced as upstream has it.
        run!(
            tlim::LG_XSI_CNTR,
            check_lg_xsi_cntr(&ctx, slot!(tlim::LG_XSI_CNTR))
        );
        run!(tlim::XH_CNTR, check_xh_cntr(&ctx, slot!(tlim::XH_CNTR)));
        run!(tlim::XHE_CNTR, check_xhe_cntr(&ctx, slot!(tlim::XHE_CNTR)));
        run!(tlim::XC_CNTR, check_xc_cntr(&ctx, slot!(tlim::XC_CNTR)));
        run!(tlim::XNE_CNTR, check_xne_cntr(&ctx, slot!(tlim::XNE_CNTR)));
        run!(tlim::XO_CNTR, check_xo_cntr(&ctx, slot!(tlim::XO_CNTR)));
        run!(tlim::XSI_CNTR, check_xsi_cntr(&ctx, slot!(tlim::XSI_CNTR)));
        run!(tlim::DMSTAR, check_delta_mstar(&ctx, slot!(tlim::DMSTAR)));
        run!(
            tlim::DEL_MDOT,
            check_delta_mdot(&ctx, slot!(tlim::DEL_MDOT))
        );
        run!(
            tlim::ADJUST_J_Q,
            check_adjust_j_q(&ctx, slot!(tlim::ADJUST_J_Q))
        );
        run!(tlim::LGL, check_delta_lgl(&ctx, slot!(tlim::LGL)));
        run!(
            tlim::DT_DIV_DT_CELL_COLLAPSE,
            check_dt_div_dt_cell_collapse(
                &ctx,
                slot!(tlim::DT_DIV_DT_CELL_COLLAPSE)
            )
        );
        run!(
            tlim::DT_DIV_MIN_DR_DIV_CS,
            check_dt_div_min_dr_div_cs(&ctx, slot!(tlim::DT_DIV_MIN_DR_DIV_CS))
        );
        run!(tlim::DELTA_HR, check_delta_hr(&ctx, slot!(tlim::DELTA_HR)));
        run!(
            tlim::ERROR_IN_ENERGY_CONSERVATION,
            check_rel_error_in_energy(
                &ctx,
                slot!(tlim::ERROR_IN_ENERGY_CONSERVATION)
            )
        );
        run!(
            tlim::DX_NUC_DROP,
            check_dx_nuc_drop(&ctx, slot!(tlim::DX_NUC_DROP))
        );
    }

    // `i_limit = maxloc(dt_limit_ratio(1:numTlim), dim=1)` (`timestep.f90:332`):
    // the first maximum wins.
    let mut i_limit = tlim::STRUC;
    let mut dt_limit_ratio_max = dt_limit_ratio[0];
    for (i, &r) in dt_limit_ratio.iter().enumerate() {
        if r > dt_limit_ratio_max {
            dt_limit_ratio_max = r;
            i_limit = i + 1;
        }
    }
    let dt_limit_ratio = dt_limit_ratio_max;

    let c = &model.controls;
    let ts = &c.ts_stop;
    let max_tf = if log_max_t > ts.min_logt_for_max_timestep_factor_at_high_t {
        ts.max_timestep_factor_at_high_t
    } else {
        ts.max_timestep_factor
    };
    let min_tf = ts.min_timestep_factor;

    // `filter_dt_next` (`star/private/timestep.f90:2362-2411`): the low-pass
    // controller when the previous ratio and step are available, otherwise
    // the first-order `dt*dt_limit_ratio_target/dt_limit_ratio`.
    let ratio_floored = dt_limit_ratio.max(1.0e-10);
    let mut dt_next = if ts.use_dt_low_pass_controller
        && model.dt_limit_ratio_old > 0.0
        && model.dt_old > 0.0
    {
        model.dt
            * h211_limtr(
                ratio_floored,
                model.dt_limit_ratio_old,
                model.dt,
                model.dt_old,
            )
    } else {
        let dt_first_order = model.dt * 1.0 / ratio_floored;
        // `timestep.f90:2400`: upstream stops when the fallback gives zero.
        if dt_first_order == 0.0 {
            return Err(StarError::InvalidInput(String::from(
                "filter_dt_next: dt_next is zero",
            )));
        }
        dt_first_order
    };
    // `star/private/timestep.f90:343-363`.
    let hit_max = max_tf > 0.0 && dt_next > max_tf * model.dt;
    if hit_max {
        dt_next = max_tf * model.dt;
        // `timestep.f90:343-352`.
        if dt_next == 0.0 {
            return Err(StarError::InvalidInput(String::from(
                "filter_dt_next: dt_next is zero",
            )));
        }
    }
    let hit_min = min_tf > 0.0 && dt_next < min_tf * model.dt;
    if hit_min {
        dt_next = min_tf * model.dt;
        if dt_next == 0.0 {
            return Err(StarError::InvalidInput(String::from(
                "filter_dt_next: dt_next is zero",
            )));
        }
    }

    if i_limit == tlim::STRUC {
        if hit_max {
            i_limit = tlim::MAX_TIMESTEP_FACTOR;
        } else if hit_min {
            i_limit = tlim::MIN_TIMESTEP_FACTOR;
        }
    }

    // `timestep.f90:49-53`: `timestep_controller` checks only `force_timestep > 0`;
    // `force_timestep_years` is applied at step start in `prepare_for_new_step`
    // (`evolve.f90:1925-1928`).
    if ts.force_timestep > 0.0 {
        dt_next = ts.force_timestep;
        i_limit = tlim::FORCE_TIMESTEP;
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
            i_limit = tlim::MAX_TIMESTEP;
        }
        if model.aux.step_hist.timestep_hold > model.model_number
            && dt_next > model.dt
        {
            dt_next = model.dt;
            i_limit = tlim::TIMESTEP_HOLD;
        }
    }

    let reason = reason_for_tlim(i_limit);
    if !dt_next.is_finite() || dt_next <= 0.0 {
        let rep = TimestepReport {
            dt_next: model.dt,
            varcontrol,
            dt_limit_ratio,
            reason,
            why_tlim: i_limit,
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
    model.why_tlim = i_limit;
    model.dt_limit_ratio_old = dt_limit_ratio.max(1.0e-10);
    Ok((
        TimestepAction::KeepGoing,
        TimestepReport {
            dt_next,
            varcontrol,
            dt_limit_ratio,
            reason,
            why_tlim: i_limit,
        },
        false,
    ))
}
