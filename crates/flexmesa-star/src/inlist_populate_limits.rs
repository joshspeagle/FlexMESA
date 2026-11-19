// SPDX-License-Identifier: LGPL-3.0-or-later
//! Solver tolerance, evolutionary stopping limit, and user `x_ctrl` array
//! population helpers for [`InlistBundle::populate_typed_structs`].

use flexmesa_schema::ControlValue;

use crate::controls::StarError;
use crate::inlist::{InlistBundle, SectionControls};
use crate::inlist_populate::{map_bools, map_f64s, map_opt_f64s, map_strs};

impl InlistBundle {
    #[rustfmt::skip]
    pub(crate) fn populate_controls_solver(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        let c = &mut self.controls;
        let max_tries = match ctrl.try_get_i64("controls", "solver_max_tries_before_reject")? {
            Some(v) => Some(v),
            None => ctrl.try_get_i64("controls", "max_tries1")?,
        };
        if let Some(v) = max_tries {
            if v <= 0 {
                return Err(StarError::InvalidInput(format!(
                    "'&controls::solver_max_tries_before_reject' must be > 0, got {v}"
                )));
            }
            c.max_newton_iters = v as usize;
        }
        map_f64s!("controls", ctrl, c, [
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
        if let Some(v) = ctrl.try_get_f64("controls", "maxt_for_gold_tolerances")? {
            c.max_t_for_gold_tolerances = v;
        }
        map_bools!("controls", ctrl, c, [
            include_l_in_correction_limits, include_v_in_correction_limits,
            include_u_in_correction_limits, include_w_in_correction_limits,
            ignore_species_in_max_correction,
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
        ctrl.try_f64_into("controls", &mut c.max_years_for_timestep, "max_years_for_timestep")?;
        ctrl.try_bool_into("controls", &mut c.stop_near_zams, "stop_near_zams")?;
        ctrl.try_f64_into("controls", &mut c.lnuc_div_l_zams_limit, "lnuc_div_l_zams_limit")?;
        c.xa_central_lower_limit_species = ctrl.try_get_indexed_strings("controls", "xa_central_lower_limit_species")?;
        c.xa_central_lower_limit = ctrl.try_get_indexed_f64s("controls", "xa_central_lower_limit")?;
        c.xa_central_upper_limit_species = ctrl.try_get_indexed_strings("controls", "xa_central_upper_limit_species")?;
        c.xa_central_upper_limit = ctrl.try_get_indexed_f64s("controls", "xa_central_upper_limit")?;

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
