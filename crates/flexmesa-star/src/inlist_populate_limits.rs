// SPDX-License-Identifier: GPL-3.0-or-later
//! Solver tolerance, evolutionary stopping limit, and user `x_ctrl` array
//! population helpers for [`InlistBundle::populate_typed_structs`].

use flexmesa_schema::ControlValue;

use crate::controls::StarError;
use crate::inlist::{InlistBundle, SectionControls, NUM_XA_LIMITS};
use crate::inlist_populate::{map_bools, map_f64s, map_opt_f64s, map_strs};

impl InlistBundle {
    #[rustfmt::skip]
    pub(crate) fn populate_controls_solver(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        let c = &mut self.controls;
        ctrl.try_usize_into(
            "controls",
            &mut c.solver_max_tries_before_reject,
            "solver_max_tries_before_reject",
            1,
        )?;
        ctrl.try_usize_into("controls", &mut c.max_tries1, "max_tries1", 1)?;
        ctrl.try_usize_into("controls", &mut c.max_tries_for_retry, "max_tries_for_retry", 1)?;
        ctrl.try_usize_into("controls", &mut c.max_tries_after_5_retries, "max_tries_after_5_retries", 1)?;
        ctrl.try_usize_into("controls", &mut c.max_tries_after_10_retries, "max_tries_after_10_retries", 1)?;
        ctrl.try_usize_into("controls", &mut c.max_tries_after_20_retries, "max_tries_after_20_retries", 1)?;
        ctrl.try_usize_into("controls", &mut c.solver_itermin, "solver_itermin", 0)?;
        // Solver partials self-test (`star/defaults/controls.defaults:11525-11539`).
        ctrl.try_f64_into("controls", &mut c.solver_test_partials_dx_0, "solver_test_partials_dx_0")?;
        ctrl.try_i64_into("controls", &mut c.solver_test_partials_k, "solver_test_partials_k")?;
        ctrl.try_i64_into("controls", &mut c.solver_test_partials_k_low, "solver_test_partials_k_low")?;
        ctrl.try_i64_into("controls", &mut c.solver_test_partials_k_high, "solver_test_partials_k_high")?;
        ctrl.try_i64_into("controls", &mut c.solver_test_partials_call_number, "solver_test_partials_call_number")?;
        ctrl.try_i64_into("controls", &mut c.solver_test_partials_iter_number, "solver_test_partials_iter_number")?;
        map_strs!("controls", ctrl, c, [
            solver_test_partials_var_name, solver_test_partials_sink_name,
            solver_test_partials_equ_name,
        ]);
        // `star/private/star_solver.f90:635-639` runs the self-test (and then
        // `mesa_error`s, `star_solver.f90:675`) only when `dx_0 > 0`, `k > 0`,
        // `solver_call_number == call_number` and `solver_iter == iter_number`.
        // `solver_call_number` counts from 1 (`struct_burn_mix.f90:390`), so a
        // call number below 1 never fires. Firing is not ported: fail loudly.
        if c.solver_test_partials_dx_0 > 0.0
            && c.solver_test_partials_k > 0
            && c.solver_test_partials_call_number >= 1
            && c.solver_test_partials_iter_number >= 0
        {
            return Err(StarError::InvalidInput(String::from(
                "'&controls::solver_test_partials_*' selects a reachable solver partials self-test (dx_0 > 0, k > 0, call_number >= 1); the self-test is not ported (star/private/star_solver.f90:635-675)",
            )));
        }
        ctrl.try_usize_into(
            "controls",
            &mut c.solver_itermin_until_reduce_min_corr_coeff,
            "solver_itermin_until_reduce_min_corr_coeff",
            0,
        )?;
        ctrl.try_usize_into("controls", &mut c.tiny_corr_coeff_limit, "tiny_corr_coeff_limit", 0)?;
        ctrl.try_usize_into("controls", &mut c.bad_max_correction_series_limit, "bad_max_correction_series_limit", 0)?;
        ctrl.try_usize_into("controls", &mut c.relax_iter_for_resid_tol2, "relax_iter_for_resid_tol2", 0)?;
        ctrl.try_usize_into("controls", &mut c.relax_iter_for_resid_tol3, "relax_iter_for_resid_tol3", 0)?;
        ctrl.try_usize_into("controls", &mut c.max_num_surf_revisions, "max_num_surf_revisions", 0)?;
        ctrl.try_usize_into("controls", &mut c.max_tries_for_implicit_wind, "max_tries_for_implicit_wind", 0)?;
        ctrl.try_isize_into(
            "controls",
            &mut c.relax_solver_iters_timestep_limit,
            "relax_solver_iters_timestep_limit",
        )?;
        map_f64s!("controls", ctrl, c, [
            tol_correction_norm, tol_max_correction,
            tol_correction_high_t_limit, tol_correction_norm_high_t, tol_max_correction_high_t,
            tol_correction_extreme_t_limit, tol_correction_norm_extreme_t, tol_max_correction_extreme_t,
            tol_bad_max_correction,
            relax_tol_correction_norm, relax_tol_max_correction,
            relax_tol_residual_norm1, relax_tol_max_residual1,
            relax_tol_residual_norm2, relax_tol_max_residual2,
            relax_tol_residual_norm3, relax_tol_max_residual3,
            relax_maxt_for_gold_tolerances,
            solver_reduced_min_corr_coeff,
            scale_correction_norm, corr_param_factor, scale_max_correction,
            corr_coeff_limit, tiny_corr_factor,
            corr_norm_jump_limit, max_corr_jump_limit,
            resid_norm_jump_limit, max_resid_jump_limit,
            min_xa_hard_limit, min_xa_hard_limit_for_hight,
            logt_max_for_min_xa_hard_limit, logt_min_for_min_xa_hard_limit_for_hight,
            sum_xa_hard_limit, sum_xa_hard_limit_for_hight,
            logt_max_for_sum_xa_hard_limit, logt_min_for_sum_xa_hard_limit_for_hight,
            max_frac_for_negative_surf_lum,
            hydro_mtx_max_allowed_abs_dlogt, hydro_mtx_max_allowed_abs_dlogrho,
            min_logt_for_hydro_mtx_max_allowed,
            hydro_mtx_max_allowed_logt, hydro_mtx_max_allowed_logrho,
            hydro_mtx_min_allowed_logt, hydro_mtx_min_allowed_logrho,
            xa_clip_limit, rti_energy_floor, max_abs_rel_change_surf_lns,
            iwind_tolerance, iwind_lambda,
            correction_xa_limit, xa_scale,
            min_chem_eqn_scale, dedt_eqn_r_scale,
            tol_residual_norm1, tol_max_residual1,
            tol_residual_norm2, tol_max_residual2,
            tol_residual_norm3, tol_max_residual3,
            gold_tol_residual_norm1, gold_tol_max_residual1,
            gold_tol_residual_norm2, gold_tol_max_residual2,
            gold_tol_residual_norm3, gold_tol_max_residual3,
            gold2_tol_residual_norm1, gold2_tol_max_residual1,
            gold2_tol_residual_norm2, gold2_tol_max_residual2,
            gold2_tol_residual_norm3, gold2_tol_max_residual3,
        ]);
        if let Some(v) = ctrl.try_get_f64("controls", "maxt_for_gold_tolerances")? {
            c.max_t_for_gold_tolerances = v;
        }
        map_bools!("controls", ctrl, c, [
            include_l_in_correction_limits, include_v_in_correction_limits,
            include_u_in_correction_limits, include_w_in_correction_limits,
            ignore_species_in_max_correction,
            ignore_too_large_correction,
            ignore_min_corr_coeff_for_scale_max_correction,
            relax_use_gold_tolerances,
            do_solver_damping_for_neg_xa,
            scale_max_correction_for_negative_surf_lum,
            convergence_ignore_equl_residuals, convergence_ignore_alpha_rti_residuals,
            use_gold_tolerances, use_gold2_tolerances,
        ]);
        ctrl.try_usize_into("controls", &mut c.iter_for_resid_tol2, "iter_for_resid_tol2", 0)?;
        ctrl.try_usize_into("controls", &mut c.iter_for_resid_tol3, "iter_for_resid_tol3", 0)?;
        ctrl.try_usize_into("controls", &mut c.gold_iter_for_resid_tol2, "gold_iter_for_resid_tol2", 0)?;
        ctrl.try_usize_into("controls", &mut c.gold_iter_for_resid_tol3, "gold_iter_for_resid_tol3", 0)?;
        ctrl.try_usize_into("controls", &mut c.gold2_iter_for_resid_tol2, "gold2_iter_for_resid_tol2", 0)?;
        ctrl.try_usize_into("controls", &mut c.gold2_iter_for_resid_tol3, "gold2_iter_for_resid_tol3", 0)?;
        ctrl.try_isize_into(
            "controls",
            &mut c.steps_before_use_gold_tolerances,
            "steps_before_use_gold_tolerances",
        )?;
        ctrl.try_isize_into(
            "controls",
            &mut c.steps_before_use_gold2_tolerances,
            "steps_before_use_gold2_tolerances",
        )?;
        Ok(())
    }

    #[rustfmt::skip]
    pub(crate) fn populate_controls_limits(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        let c = &mut self.controls;
        if let Some(v) = ctrl.try_get_i64("controls", "max_model_number")? {
            c.max_model_number = if v < 0 { None } else { Some(v as usize) };
        }
        ctrl.try_f64_into("controls", &mut c.max_age, "max_age")?;
        if let Some(v) = ctrl.try_get_f64("controls", "max_age_in_seconds")? {
            c.max_age_in_seconds = (v > 0.0).then_some(v);
        }
        ctrl.try_isize_into("controls", &mut c.max_number_retries, "max_number_retries")?;
        ctrl.try_isize_into("controls", &mut c.relax_max_number_retries, "relax_max_number_retries")?;
        ctrl.try_f64_into("controls", &mut c.relax_dy, "relax_dY")?;
        ctrl.try_f64_into("controls", &mut c.relax_dlnz, "relax_dlnZ")?;
        ctrl.try_f64_into("controls", &mut c.max_years_for_timestep, "max_years_for_timestep")?;
        ctrl.try_bool_into("controls", &mut c.stop_near_zams, "stop_near_zams")?;
        ctrl.try_f64_into("controls", &mut c.lnuc_div_l_zams_limit, "lnuc_div_l_zams_limit")?;
        c.xa_central_lower_limit_species = ctrl.try_get_indexed_strings_bounded(
            "controls",
            "xa_central_lower_limit_species",
            Some(NUM_XA_LIMITS),
        )?;
        c.xa_central_lower_limit = ctrl.try_get_indexed_f64s_bounded(
            "controls",
            "xa_central_lower_limit",
            Some(NUM_XA_LIMITS),
        )?;
        c.xa_central_upper_limit_species = ctrl.try_get_indexed_strings_bounded(
            "controls",
            "xa_central_upper_limit_species",
            Some(NUM_XA_LIMITS),
        )?;
        c.xa_central_upper_limit = ctrl.try_get_indexed_f64s_bounded(
            "controls",
            "xa_central_upper_limit",
            Some(NUM_XA_LIMITS),
        )?;

        map_f64s!("controls", ctrl, c, [
            log_center_temp_upper_limit, log_center_density_upper_limit,
            log_max_temp_upper_limit, envelope_mass_limit, he_layer_mass_lower_limit,
            star_mass_max_limit, star_mass_min_limit, remnant_mass_min_limit,
            star_species_mass_min_limit, star_species_mass_max_limit,
            fe_core_infall_limit, surf_omega_div_omega_crit_limit,
            conv_core_gap_dq_limit,
        ]);
        map_opt_f64s!("controls", ctrl, c, [
            log_l_upper_limit, log_l_lower_limit, teff_upper_limit, teff_lower_limit,
            log_teff_upper_limit, log_teff_lower_limit,
            log_center_temp_lower_limit, log_center_density_lower_limit,
            log_g_upper_limit, log_g_lower_limit,
            log_psurf_upper_limit, log_rsurf_upper_limit, center_r_lower_limit,
            he_core_mass_limit, co_core_mass_limit, one_core_mass_limit,
            power_nuc_burn_upper_limit, power_nuc_burn_lower_limit,
            power_he_burn_upper_limit,
            v_surf_div_v_kh_upper_limit, gamma_center_limit, v_surf_div_v_esc_limit,
        ]);
        map_strs!("controls", ctrl, c, [star_species_mass_min_limit_iso, star_species_mass_max_limit_iso]);
        map_bools!("controls", ctrl, c, [
            stop_at_phase_tp_agb, stop_at_phase_c_burn,
            stop_at_phase_ne_burn, stop_at_phase_o_burn,
        ]);

        let ts = &mut c.ts_stop;
        map_f64s!("controls", ctrl, ts, [
            delta_ye_hight_limit, delta_ye_hight_hard_limit, mint_for_hight_ye_limit,
            delta_log_eps_nuc_limit, delta_log_eps_nuc_hard_limit,
            delta_lgl_nuc_cat_limit, delta_lgl_nuc_cat_hard_limit,
            lgl_nuc_cat_burn_min, lgl_nuc_mix_dist_limit,
            lgl_h_burn_relative_limit, lgl_he_burn_relative_limit, lgl_z_burn_relative_limit,
            delta_lgl_nuc_at_high_t_limit, delta_lgl_nuc_at_high_t_hard_limit,
            delta_lgl_nuc_at_high_t_limit_lgt_min, max_lgt_for_lgl_nuc_limit,
            delta_lgt_max_limit_lgt_min,
            delta_lgt_max_at_high_t_limit, delta_lgt_max_at_high_t_hard_limit,
            delta_lgt_max_at_high_t_limit_lgt_min,
            delta_dx_div_x_cntr_min, delta_dx_div_x_cntr_max,
            delta_dx_div_x_cntr_limit, delta_dx_div_x_cntr_hard_limit,
            delta_lg_xne_cntr_min, delta_lg_xne_cntr_max,
            delta_lg_xne_cntr_limit, delta_lg_xne_cntr_hard_limit,
            delta_lg_xo_cntr_min, delta_lg_xo_cntr_max,
            delta_lg_xo_cntr_limit, delta_lg_xo_cntr_hard_limit,
            delta_lg_xsi_cntr_min, delta_lg_xsi_cntr_max,
            delta_lg_xsi_cntr_limit, delta_lg_xsi_cntr_hard_limit,
            delta_xh_cntr_limit, delta_xh_cntr_hard_limit,
            delta_xhe_cntr_limit, delta_xhe_cntr_hard_limit,
            delta_xc_cntr_limit, delta_xc_cntr_hard_limit,
            delta_xne_cntr_limit, delta_xne_cntr_hard_limit,
            delta_xo_cntr_limit, delta_xo_cntr_hard_limit,
            delta_xsi_cntr_limit, delta_xsi_cntr_hard_limit,
            adjust_j_q_limit, adjust_j_q_hard_limit,
        ]);
        map_bools!("controls", ctrl, ts, [
            check_remnant_only_for_dt_div_min_dr_div_cs_limit,
            delta_lgt_max_limit_only_after_near_zams,
            delta_lgt_cntr_limit_only_after_near_zams,
            delta_dx_div_x_drop_only,
            delta_lg_xne_drop_only, delta_lg_xo_drop_only, delta_lg_xsi_drop_only,
            delta_xh_drop_only, delta_xhe_drop_only, delta_xc_drop_only,
            delta_xne_drop_only, delta_xo_drop_only, delta_xsi_drop_only,
            use_other_timestep_limit,
        ]);
        for (key, dst) in [
            ("dx_div_x_at_high_t_limit", &mut ts.dx_div_x_at_high_t_limit),
            ("dx_div_x_at_high_t_hard_limit", &mut ts.dx_div_x_at_high_t_hard_limit),
            ("dx_div_x_at_high_t_limit_lgt_min", &mut ts.dx_div_x_at_high_t_limit_lgt_min),
        ] {
            Self::apply_fortran_f64_slice(ctrl, key, dst)?;
        }
        Ok(())
    }

    fn apply_fortran_f64_slice(
        ctrl: &SectionControls,
        key: &str,
        dst: &mut [f64],
    ) -> Result<(), StarError> {
        let extract = |val: &ControlValue| match val {
            ControlValue::Float(f) if f.is_finite() => Ok(*f),
            ControlValue::Float(f) => Err(format!("non-finite float {f}")),
            ControlValue::Integer(i) => Ok(*i as f64),
            other => Err(format!("expected real number, found {other:?}")),
        };
        if let Some(raw_val) = ctrl.scalars.get(key) {
            let v = extract(raw_val).map_err(|msg| {
                StarError::InvalidInput(format!(
                    "invalid value for '&controls::{key}': {msg}"
                ))
            })?;
            dst[0] = v;
        }
        if let Some(list) = ctrl.indexed.get(key) {
            let len = dst.len();
            for (sub, raw_val) in list {
                let v = extract(raw_val).map_err(|msg| {
                    StarError::InvalidInput(format!(
                        "invalid value for '&controls::{key}({sub})': {msg}"
                    ))
                })?;
                let s = sub.trim();
                if s == ":" {
                    dst.fill(v);
                } else if let Some(start_s) = s.strip_suffix(':') {
                    let idx1 = start_s.trim().parse::<usize>().map_err(|_| {
                        StarError::InvalidInput(format!("invalid slice subscript '({sub})' for '&controls::{key}'"))
                    })?;
                    if idx1 == 0 || idx1 > len {
                        return Err(StarError::InvalidInput(format!(
                            "subscript '({sub})' out of 1-based bounds (1..={len}) for '&controls::{key}'"
                        )));
                    }
                    dst[(idx1 - 1)..len].fill(v);
                } else if let Some((lo_s, hi_s)) = s.split_once(':') {
                    let lo1 = if lo_s.trim().is_empty() {
                        1
                    } else {
                        lo_s.trim().parse::<usize>().map_err(|_| {
                            StarError::InvalidInput(format!("invalid slice subscript '({sub})' for '&controls::{key}'"))
                        })?
                    };
                    let hi1 = hi_s.trim().parse::<usize>().map_err(|_| {
                        StarError::InvalidInput(format!("invalid slice subscript '({sub})' for '&controls::{key}'"))
                    })?;
                    if lo1 == 0 || hi1 == 0 || lo1 > hi1 || hi1 > len {
                        return Err(StarError::InvalidInput(format!(
                            "subscript '({sub})' out of 1-based bounds (1..={len}) for '&controls::{key}'"
                        )));
                    }
                    dst[(lo1 - 1)..hi1].fill(v);
                } else {
                    let idx1 = s.parse::<usize>().map_err(|_| {
                        StarError::InvalidInput(format!(
                            "invalid subscript '({sub})' for '&controls::{key}'"
                        ))
                    })?;
                    if idx1 == 0 || idx1 > len {
                        return Err(StarError::InvalidInput(format!(
                            "subscript '({idx1})' out of 1-based bounds (1..={len}) for '&controls::{key}'"
                        )));
                    }
                    dst[idx1 - 1] = v;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn populate_user_x_controls(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        Self::apply_indexed_slice(
            ctrl,
            "x_ctrl",
            &mut self.controls.x_ctrl,
            |val| match val {
                ControlValue::Float(f) if f.is_finite() => Ok(*f),
                ControlValue::Float(f) => Err(format!("non-finite float {f}")),
                ControlValue::Integer(i) => Ok(*i as f64),
                other => Err(format!("expected real number, found {other:?}")),
            },
        )?;
        Self::apply_indexed_slice(
            ctrl,
            "x_integer_ctrl",
            &mut self.controls.x_integer_ctrl,
            |val| match val {
                ControlValue::Integer(i) => Ok(*i),
                other => Err(format!("expected integer, found {other:?}")),
            },
        )?;
        Self::apply_indexed_slice(
            ctrl,
            "x_logical_ctrl",
            &mut self.controls.x_logical_ctrl,
            |val| match val {
                ControlValue::Logical(b) => Ok(*b),
                other => Err(format!("expected boolean, found {other:?}")),
            },
        )?;
        Self::apply_indexed_slice(
            ctrl,
            "x_character_ctrl",
            &mut self.controls.x_character_ctrl,
            |val| match val {
                ControlValue::String(s) => Ok(s.clone()),
                other => Err(format!("expected string, found {other:?}")),
            },
        )?;
        Ok(())
    }

    fn apply_indexed_slice<T>(
        ctrl: &SectionControls,
        key: &str,
        dst: &mut [T],
        extract: impl Fn(&ControlValue) -> Result<T, String>,
    ) -> Result<(), StarError> {
        if ctrl.scalars.contains_key(key) {
            return Err(StarError::InvalidInput(format!(
                "missing 1-based subscript '(1..={})' for array control '&controls::{key}'",
                dst.len()
            )));
        }
        if let Some(list) = ctrl.indexed.get(key) {
            for (sub, val) in list {
                let idx = sub.trim().parse::<usize>().map_err(|_| {
                    StarError::InvalidInput(format!(
                        "invalid subscript '({sub})' for '&controls::{key}'"
                    ))
                })?;
                if idx == 0 || idx > dst.len() {
                    return Err(StarError::InvalidInput(format!(
                        "subscript '({idx})' out of 1-based bounds (1..={}) for '&controls::{key}'",
                        dst.len()
                    )));
                }
                let v = extract(val).map_err(|msg| {
                    StarError::InvalidInput(format!(
                        "invalid value for '&controls::{key}({idx})': {msg}"
                    ))
                })?;
                dst[idx - 1] = v;
            }
        }
        Ok(())
    }
}
