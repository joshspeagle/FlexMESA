// SPDX-License-Identifier: LGPL-3.0-or-later
//! Solver tolerance, evolutionary stopping limit, and user `x_ctrl` array
//! population helpers for [`InlistBundle::populate_typed_structs`].

use flexmesa_schema::ControlValue;

use crate::inlist::{InlistBundle, SectionControls};
use crate::inlist_populate::{map_bools, map_f64s, map_opt_f64s, map_strs};

impl InlistBundle {
    #[rustfmt::skip]
    pub(crate) fn populate_controls_solver(&mut self) {
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        let c = &mut self.controls;
        if let Some(v) = ctrl
            .get_i64("solver_max_tries_before_reject")
            .or_else(|| ctrl.get_i64("max_tries1"))
            .filter(|&v| v > 0)
        {
            c.max_newton_iters = v as usize;
        }
        map_f64s!(ctrl, c, [
            tol_correction_norm, tol_max_correction,
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
        if let Some(v) = ctrl.get_f64("maxt_for_gold_tolerances") {
            c.max_t_for_gold_tolerances = v;
        }
        map_bools!(ctrl, c, [
            include_l_in_correction_limits, include_v_in_correction_limits,
            include_u_in_correction_limits, include_w_in_correction_limits,
            ignore_species_in_max_correction,
            convergence_ignore_equl_residuals, convergence_ignore_alpha_rti_residuals,
            use_gold_tolerances, use_gold2_tolerances,
        ]);
        ctrl.usize_into(&mut c.iter_for_resid_tol2, "iter_for_resid_tol2");
        ctrl.usize_into(&mut c.iter_for_resid_tol3, "iter_for_resid_tol3");
        ctrl.usize_into(&mut c.gold_iter_for_resid_tol2, "gold_iter_for_resid_tol2");
        ctrl.usize_into(&mut c.gold_iter_for_resid_tol3, "gold_iter_for_resid_tol3");
        ctrl.usize_into(&mut c.gold2_iter_for_resid_tol2, "gold2_iter_for_resid_tol2");
        ctrl.usize_into(&mut c.gold2_iter_for_resid_tol3, "gold2_iter_for_resid_tol3");
        ctrl.isize_into(
            &mut c.steps_before_use_gold_tolerances,
            "steps_before_use_gold_tolerances",
        );
        ctrl.isize_into(
            &mut c.steps_before_use_gold2_tolerances,
            "steps_before_use_gold2_tolerances",
        );
    }

    #[rustfmt::skip]
    pub(crate) fn populate_controls_limits(&mut self) {
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        let c = &mut self.controls;
        if let Some(v) = ctrl.get_i64("max_model_number").filter(|&v| v > 0) {
            c.max_model_number = Some(v as usize);
        }
        c.max_age = ctrl.get_f64("max_age").filter(|&v| v > 0.0);
        c.max_age_in_seconds = ctrl.get_f64("max_age_in_seconds").filter(|&v| v > 0.0);
        c.max_years_for_timestep = ctrl.get_f64("max_years_for_timestep").filter(|&v| v > 0.0);
        ctrl.bool_into(&mut c.stop_near_zams, "stop_near_zams");
        if let Some(v) = ctrl.get_f64("lnuc_div_l_zams_limit") {
            c.lnuc_div_l_zams_limit = v;
            if ctrl.get_bool("stop_near_zams").is_none() {
                c.stop_near_zams = true;
            }
        }
        c.xa_central_lower_limit_species = ctrl.get_indexed_strings("xa_central_lower_limit_species");
        c.xa_central_lower_limit = ctrl.get_indexed_f64s("xa_central_lower_limit");
        c.xa_central_upper_limit_species = ctrl.get_indexed_strings("xa_central_upper_limit_species");
        c.xa_central_upper_limit = ctrl.get_indexed_f64s("xa_central_upper_limit");

        map_opt_f64s!(ctrl, c, [
            log_l_upper_limit, log_l_lower_limit, teff_upper_limit, teff_lower_limit,
            log_teff_upper_limit, log_teff_lower_limit, log_center_temp_upper_limit,
            log_center_temp_lower_limit, log_center_density_upper_limit,
            log_center_density_lower_limit, log_max_temp_upper_limit, log_g_upper_limit,
            log_g_lower_limit, log_psurf_upper_limit, log_rsurf_upper_limit,
            center_r_lower_limit, he_core_mass_limit, co_core_mass_limit,
            one_core_mass_limit, envelope_mass_limit, he_layer_mass_lower_limit,
            power_nuc_burn_upper_limit, power_nuc_burn_lower_limit,
            power_he_burn_upper_limit, star_mass_max_limit, star_mass_min_limit,
            remnant_mass_min_limit, star_species_mass_min_limit,
            star_species_mass_max_limit, fe_core_infall_limit,
            v_surf_div_v_kh_upper_limit, gamma_center_limit, v_surf_div_v_esc_limit,
            surf_omega_div_omega_crit_limit, conv_core_gap_dq_limit,
        ]);
        map_strs!(ctrl, c, [star_species_mass_min_limit_iso, star_species_mass_max_limit_iso]);
        map_bools!(ctrl, c, [
            stop_at_phase_tp_agb, stop_at_phase_c_burn,
            stop_at_phase_ne_burn, stop_at_phase_o_burn,
        ]);
    }

    pub(crate) fn populate_user_x_controls(&mut self) {
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        Self::apply_indexed_slice(
            ctrl,
            "x_ctrl",
            &mut self.controls.x_ctrl,
            |val| match val {
                ControlValue::Float(f) => Some(*f),
                ControlValue::Integer(i) => Some(*i as f64),
                _ => None,
            },
        );
        Self::apply_indexed_slice(
            ctrl,
            "x_integer_ctrl",
            &mut self.controls.x_integer_ctrl,
            |val| match val {
                ControlValue::Integer(i) => Some(*i),
                _ => None,
            },
        );
        Self::apply_indexed_slice(
            ctrl,
            "x_logical_ctrl",
            &mut self.controls.x_logical_ctrl,
            |val| match val {
                ControlValue::Logical(b) => Some(*b),
                _ => None,
            },
        );
        Self::apply_indexed_slice(
            ctrl,
            "x_character_ctrl",
            &mut self.controls.x_character_ctrl,
            |val| match val {
                ControlValue::String(s) => Some(s.clone()),
                _ => None,
            },
        );
    }

    fn apply_indexed_slice<T>(
        ctrl: &SectionControls,
        key: &str,
        dst: &mut [T],
        extract: impl Fn(&ControlValue) -> Option<T>,
    ) {
        if let Some(list) = ctrl.indexed.get(key) {
            for (sub, val) in list {
                if let Ok(idx) = sub.trim().parse::<usize>() {
                    if idx < dst.len() {
                        if let Some(v) = extract(val) {
                            dst[idx] = v;
                        }
                    }
                }
            }
        }
    }
}
