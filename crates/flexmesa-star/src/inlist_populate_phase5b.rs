// SPDX-License-Identifier: LGPL-3.0-or-later
//! Namelist population helpers for [`crate::controls_mix_wind::MixWindControls`]
//! and [`crate::controls_timestep_stop::TimestepStopControls`] invoked by
//! [`crate::InlistBundle::populate_typed_structs`].

use flexmesa_schema::ControlValue;

use crate::inlist::{InlistBundle, SectionControls};
use crate::inlist_populate::{map_bools, map_f64s, map_opt_f64s, map_strs};

fn apply_fortran_1based<T, F>(
    ctrl: &SectionControls,
    key: &str,
    len: usize,
    mut set_at: F,
    extract: impl Fn(&ControlValue) -> Option<T>,
) where
    T: Clone,
    F: FnMut(usize, T),
{
    let k_low = key.to_ascii_lowercase();
    if let Some(val) = ctrl.scalars.get(&k_low).and_then(&extract) {
        set_at(0, val);
    }
    if let Some(list) = ctrl.indexed.get(&k_low) {
        for (sub, raw_val) in list {
            let Some(v) = extract(raw_val) else {
                continue;
            };
            let s = sub.trim();
            if s == ":" {
                for idx0 in 0..len {
                    set_at(idx0, v.clone());
                }
            } else if let Some(start_s) = s.strip_suffix(':') {
                if let Ok(idx1) = start_s.trim().parse::<usize>() {
                    for idx0 in idx1.saturating_sub(1)..len {
                        set_at(idx0, v.clone());
                    }
                }
            } else if let Ok(idx1) = s.parse::<usize>() {
                if (1..=len).contains(&idx1) {
                    set_at(idx1 - 1, v);
                }
            }
        }
    }
}

fn val_f64(val: &ControlValue) -> Option<f64> {
    match val {
        ControlValue::Float(f) => Some(*f),
        ControlValue::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

fn val_bool(val: &ControlValue) -> Option<bool> {
    match val {
        ControlValue::Logical(b) => Some(*b),
        _ => None,
    }
}

fn val_str(val: &ControlValue) -> Option<String> {
    match val {
        ControlValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

impl InlistBundle {
    #[rustfmt::skip]
    pub(crate) fn populate_controls_phase5b(&mut self) {
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        let mw = &mut self.controls.mix_wind;
        map_f64s!(ctrl, mw, [
            overshoot_d_min, overshoot_brunt_b_max, min_overshoot_q, overshoot_alpha,
            burn_z_mix_region_logt, burn_he_mix_region_logt, burn_h_mix_region_logt,
            max_y_for_burn_z_mix_region, max_x_for_burn_he_mix_region,
            peclet_function_full_on, peclet_function_full_off,
            conv_premix_time_factor,
            rgb_to_agb_wind_switch, bjorklund_scaling_factor,
            de_jager_scaling_factor, van_loon_scaling_factor,
            nieuwenhuijzen_scaling_factor, min_wind, max_wind,
            mdot_omega_power, max_rotational_mdot_boost,
            max_mdot_jump_for_rotation, rotational_mdot_boost_fac,
            rotational_mdot_kh_fac, super_eddington_wind_ledd_factor,
            wind_boost_full_off_l_div_ledd, wind_boost_full_on_l_div_ledd,
            super_eddington_wind_max_boost, max_star_mass_for_gain,
            min_star_mass_for_loss, accreted_material_j,
            min_q_for_k_below_const_q, max_q_for_k_below_const_q,
            max_logt_for_k_below_const_q,
            min_q_for_k_const_mass, max_q_for_k_const_mass,
            max_logt_for_k_const_mass,
            eps_mdot_factor, eps_mdot_leak_frac_factor,
            eps_wd_sedimentation_factor, extra_power_source,
            inject_uniform_extra_heat, min_q_for_uniform_extra_heat,
            max_q_for_uniform_extra_heat,
            max_dq, min_dq, min_dq_for_split, max_surface_cell_dq,
            max_center_cell_dq, p_function_weight, t_function1_weight,
            r_function_weight, m_function_weight, omega_function_weight,
            gam_function_weight,
        ]);
        map_bools!(ctrl, mw, [
            limit_overshoot_hp_using_size_of_convection_zone,
            conv_premix_avoid_increase, conv_premix_fix_pgas,
            no_wind_if_no_rotation, use_accreted_material_j,
        ]);
        map_strs!(ctrl, mw, [hot_wind_scheme, dutch_wind_lowt_scheme]);

        // Legacy wind aliases (`RGB_wind_scheme`, `AGB_wind_scheme`, `*_wind_eta`).
        ctrl.str_into(&mut self.controls.cool_wind_rgb_scheme, "rgb_wind_scheme");
        ctrl.str_into(&mut self.controls.cool_wind_agb_scheme, "agb_wind_scheme");
        ctrl.f64_into(&mut self.controls.reimers_scaling_factor, "reimers_wind_eta");
        ctrl.f64_into(&mut self.controls.blocker_scaling_factor, "blocker_wind_eta");
        ctrl.f64_into(&mut self.controls.vink_scaling_factor, "vink_wind_eta");
        ctrl.f64_into(&mut self.controls.dutch_scaling_factor, "dutch_wind_eta");

        let n_ov = mw.overshoot_sets.len();
        apply_fortran_1based(ctrl, "overshoot_scheme", n_ov, |i, v| mw.overshoot_sets[i].scheme = v, val_str);
        apply_fortran_1based(ctrl, "overshoot_zone_type", n_ov, |i, v| mw.overshoot_sets[i].zone_type = v, val_str);
        apply_fortran_1based(ctrl, "overshoot_zone_loc", n_ov, |i, v| mw.overshoot_sets[i].zone_loc = v, val_str);
        apply_fortran_1based(ctrl, "overshoot_bdy_loc", n_ov, |i, v| mw.overshoot_sets[i].bdy_loc = v, val_str);
        apply_fortran_1based(ctrl, "overshoot_f", n_ov, |i, v| mw.overshoot_sets[i].f = v, val_f64);
        apply_fortran_1based(ctrl, "overshoot_f0", n_ov, |i, v| mw.overshoot_sets[i].f0 = v, val_f64);
        apply_fortran_1based(ctrl, "overshoot_f2", n_ov, |i, v| mw.overshoot_sets[i].f2 = v, val_f64);
        apply_fortran_1based(ctrl, "overshoot_d0", n_ov, |i, v| mw.overshoot_sets[i].d0 = v, val_f64);
        apply_fortran_1based(ctrl, "overshoot_delta0", n_ov, |i, v| mw.overshoot_sets[i].delta0 = v, val_f64);
        apply_fortran_1based(ctrl, "overshoot_mass_full_on", n_ov, |i, v| mw.overshoot_sets[i].mass_full_on = v, val_f64);
        apply_fortran_1based(ctrl, "overshoot_mass_full_off", n_ov, |i, v| mw.overshoot_sets[i].mass_full_off = v, val_f64);

        let n_pr = mw.predictive_sets.len();
        apply_fortran_1based(ctrl, "predictive_mix", n_pr, |i, v| mw.predictive_sets[i].predictive_mix = v, val_bool);
        apply_fortran_1based(ctrl, "predictive_zone_type", n_pr, |i, v| mw.predictive_sets[i].zone_type = v, val_str);
        apply_fortran_1based(ctrl, "predictive_zone_loc", n_pr, |i, v| mw.predictive_sets[i].zone_loc = v, val_str);
        apply_fortran_1based(ctrl, "predictive_bdy_loc", n_pr, |i, v| mw.predictive_sets[i].bdy_loc = v, val_str);
        apply_fortran_1based(ctrl, "predictive_bdy_q_min", n_pr, |i, v| mw.predictive_sets[i].bdy_q_min = v, val_f64);
        apply_fortran_1based(ctrl, "predictive_bdy_q_max", n_pr, |i, v| mw.predictive_sets[i].bdy_q_max = v, val_f64);
        apply_fortran_1based(ctrl, "predictive_superad_thresh", n_pr, |i, v| mw.predictive_sets[i].superad_thresh = v, val_f64);
        apply_fortran_1based(ctrl, "predictive_avoid_reversal", n_pr, |i, v| mw.predictive_sets[i].avoid_reversal = v, val_str);
        apply_fortran_1based(ctrl, "predictive_limit_ingestion", n_pr, |i, v| mw.predictive_sets[i].limit_ingestion = v, val_str);
        apply_fortran_1based(ctrl, "predictive_ingestion_factor", n_pr, |i, v| mw.predictive_sets[i].ingestion_factor = v, val_f64);

        mw.xa_function_species = ctrl.get_indexed_strings("xa_function_species");
        mw.xa_function_weight = ctrl.get_indexed_f64s("xa_function_weight");
        mw.xa_function_param = ctrl.get_indexed_f64s("xa_function_param");

        let ts = &mut self.controls.ts_stop;
        map_f64s!(ctrl, ts, [
            he_core_boundary_h1_fraction, co_core_boundary_he4_fraction,
            one_core_boundary_he4_c12_fraction, fe_core_boundary_si28_fraction,
            neutron_rich_core_boundary_ye_max, min_boundary_fraction,
            fe_core_infall_mass, non_fe_core_infall_mass,
            time_delta_coeff, max_timestep, max_timestep_hi_t_limit,
            hi_t_max_years_for_timestep, min_timestep_factor,
            max_timestep_factor, max_timestep_factor_at_high_t,
            min_logt_for_max_timestep_factor_at_high_t,
            force_timestep, force_timestep_years,
            force_timestep_min, force_timestep_min_years,
            force_timestep_min_factor, timestep_factor_for_retries,
            timestep_dt_factor, min_allowed_varcontrol_target,
            varcontrol_dt_limit_ratio_hard_max,
            dx_mix_dist_limit, dx_nuc_drop_min_x_limit,
            dx_nuc_drop_max_a_limit, dx_nuc_drop_limit_at_high_t,
            dx_nuc_drop_limit, dx_nuc_drop_hard_limit,
            dx_nuc_drop_min_yrs_for_dt,
            dl_div_l_limit, dl_div_l_hard_limit, dl_div_l_limit_min_l,
            delta_lgp_limit, delta_lgp_hard_limit, delta_lgp_limit_min_lgp,
            delta_lgrho_limit, delta_lgrho_hard_limit, delta_lgrho_limit_min_lgrho,
            delta_lgt_limit, delta_lgt_hard_limit, delta_lgt_limit_min_lgt,
            delta_lge_limit, delta_lge_hard_limit, delta_lge_limit_min_lge,
            delta_lgr_limit, delta_lgr_hard_limit, delta_lgr_limit_min_lgr,
            delta_lgl_h_limit, delta_lgl_h_hard_limit, lgl_h_burn_min, lgl_h_drop_factor,
            delta_lgl_he_limit, delta_lgl_he_hard_limit, lgl_he_burn_min, lgl_he_drop_factor,
            delta_lgl_z_limit, delta_lgl_z_hard_limit, lgl_z_burn_min, lgl_z_drop_factor,
            delta_lgl_nuc_limit, delta_lgl_nuc_hard_limit, lgl_nuc_burn_min, lgl_nuc_drop_factor,
            delta_lgl_power_photo_limit, delta_lgl_power_photo_hard_limit,
            min_lgt_for_lgl_power_photo_limit, lgl_power_photo_burn_min,
            lgl_power_photo_drop_factor,
            delta_lgteff_limit, delta_lgteff_hard_limit,
            delta_lgl_limit_l_min, delta_lgl_limit, delta_lgl_hard_limit,
            delta_lgt_cntr_limit, delta_lgt_cntr_hard_limit,
            delta_lgrho_cntr_limit, delta_lgrho_cntr_hard_limit,
            delta_lgp_cntr_limit, delta_lgp_cntr_hard_limit,
            delta_lgt_max_limit, delta_lgt_max_hard_limit,
            delta_hr_ds_l, delta_hr_ds_teff, delta_hr_limit, delta_hr_hard_limit,
            delta_lg_star_mass_limit, delta_lg_star_mass_hard_limit,
            limit_for_rel_error_in_energy_conservation,
            hard_limit_for_rel_error_in_energy_conservation,
            dt_div_min_dr_div_cs_limit, dt_div_min_dr_div_cs_hard_limit,
            dt_div_dt_cell_collapse_limit, dt_div_dt_cell_collapse_hard_limit,
        ]);
        map_bools!(ctrl, ts, [
            check_mass_sum_for_infall, use_dt_low_pass_controller,
            never_skip_hard_limits, relax_hard_limits_after_retry,
        ]);
        map_opt_f64s!(ctrl, ts, [
            non_fe_core_infall_limit, power_h_burn_upper_limit,
            power_h_burn_lower_limit, power_he_burn_lower_limit,
            power_z_burn_upper_limit, power_z_burn_lower_limit,
        ]);
        ts.eta_center_limit = ctrl.get_f64("eta_center_limit").filter(|&v| v < 1.0e90);
        ctrl.usize_into(&mut ts.retry_hold, "retry_hold");
        ctrl.usize_into(&mut ts.neg_mass_fraction_hold, "neg_mass_fraction_hold");
        ctrl.usize_into(&mut ts.retry_limit, "retry_limit");
        ctrl.usize_into(&mut ts.redo_limit, "redo_limit");
        ctrl.isize_into(&mut ts.solver_iters_timestep_limit, "solver_iters_timestep_limit");
        ctrl.isize_into(&mut ts.gold_solver_iters_timestep_limit, "gold_solver_iters_timestep_limit");
        ctrl.isize_into(&mut ts.gold2_solver_iters_timestep_limit, "gold2_solver_iters_timestep_limit");
        ctrl.isize_into(&mut ts.burn_steps_limit, "burn_steps_limit");
        ctrl.isize_into(&mut ts.diffusion_steps_limit, "diffusion_steps_limit");
        ctrl.isize_into(&mut ts.diffusion_steps_hard_limit, "diffusion_steps_hard_limit");
        ctrl.isize_into(&mut ts.diffusion_iters_limit, "diffusion_iters_limit");
        ctrl.isize_into(&mut ts.diffusion_iters_hard_limit, "diffusion_iters_hard_limit");

        let n_dx = ts.dx_limit.len();
        apply_fortran_1based(ctrl, "dx_limit_species", n_dx, |i, v| ts.dx_limit_species[i] = v, val_str);
        apply_fortran_1based(ctrl, "dx_limit_min_x", n_dx, |i, v| ts.dx_limit_min_x[i] = v, val_f64);
        apply_fortran_1based(ctrl, "dx_limit", n_dx, |i, v| ts.dx_limit[i] = v, val_f64);
        apply_fortran_1based(ctrl, "dx_hard_limit", n_dx, |i, v| ts.dx_hard_limit[i] = v, val_f64);
        apply_fortran_1based(ctrl, "dx_decreases_only", n_dx, |i, v| ts.dx_decreases_only[i] = v, val_bool);
        apply_fortran_1based(ctrl, "dx_div_x_limit_min_x", n_dx, |i, v| ts.dx_div_x_limit_min_x[i] = v, val_f64);
        apply_fortran_1based(ctrl, "dx_div_x_limit", n_dx, |i, v| ts.dx_div_x_limit[i] = v, val_f64);
        apply_fortran_1based(ctrl, "dx_div_x_hard_limit", n_dx, |i, v| ts.dx_div_x_hard_limit[i] = v, val_f64);
    }
}
