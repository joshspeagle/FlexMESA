// SPDX-License-Identifier: LGPL-3.0-or-later
//! Namelist population helpers for [`crate::controls_mix_wind::MixWindControls`]
//! and [`crate::controls_timestep_stop::TimestepStopControls`] invoked by
//! [`crate::InlistBundle::populate_typed_structs`].

use flexmesa_schema::ControlValue;

use crate::controls::StarError;
use crate::inlist::{InlistBundle, SectionControls};
use crate::inlist_populate::{map_bools, map_f64s, map_opt_f64s, map_strs};

fn apply_fortran_1based<T, F>(
    ctrl: &SectionControls,
    key: &str,
    len: usize,
    mut set_at: F,
    extract: impl Fn(&ControlValue) -> Result<T, String>,
) -> Result<(), StarError>
where
    T: Clone,
    F: FnMut(usize, T),
{
    let k_low = key.to_ascii_lowercase();
    if let Some(raw_val) = ctrl.scalars.get(&k_low) {
        let v = extract(raw_val).map_err(|msg| {
            StarError::InvalidInput(format!(
                "invalid value for '&controls::{k_low}': {msg}"
            ))
        })?;
        set_at(0, v);
    }
    if let Some(list) = ctrl.indexed.get(&k_low) {
        for (sub, raw_val) in list {
            let v = extract(raw_val).map_err(|msg| {
                StarError::InvalidInput(format!(
                    "invalid value for '&controls::{k_low}({sub})': {msg}"
                ))
            })?;
            let s = sub.trim();
            if s == ":" {
                for idx0 in 0..len {
                    set_at(idx0, v.clone());
                }
            } else if let Some(start_s) = s.strip_suffix(':') {
                let idx1 = start_s.trim().parse::<usize>().map_err(|_| {
                    StarError::InvalidInput(format!(
                        "invalid slice subscript '({sub})' for '&controls::{k_low}'"
                    ))
                })?;
                if idx1 == 0 || idx1 > len {
                    return Err(StarError::InvalidInput(format!(
                        "1-based slice subscript '({sub})' out of range 1..={len} for '&controls::{k_low}'"
                    )));
                }
                for idx0 in (idx1 - 1)..len {
                    set_at(idx0, v.clone());
                }
            } else {
                let idx1 = s.parse::<usize>().map_err(|_| {
                    StarError::InvalidInput(format!(
                        "invalid subscript '({sub})' for '&controls::{k_low}'"
                    ))
                })?;
                if idx1 == 0 || idx1 > len {
                    return Err(StarError::InvalidInput(format!(
                        "1-based subscript '({idx1})' out of range 1..={len} for '&controls::{k_low}'"
                    )));
                }
                set_at(idx1 - 1, v);
            }
        }
    }
    Ok(())
}

fn val_f64(val: &ControlValue) -> Result<f64, String> {
    match val {
        ControlValue::Float(f) if f.is_finite() => Ok(*f),
        ControlValue::Float(f) => Err(format!("non-finite float {f}")),
        ControlValue::Integer(i) => Ok(*i as f64),
        other => Err(format!("expected real number, found {other:?}")),
    }
}

fn val_bool(val: &ControlValue) -> Result<bool, String> {
    match val {
        ControlValue::Logical(b) => Ok(*b),
        other => Err(format!("expected boolean, found {other:?}")),
    }
}

fn val_str(val: &ControlValue) -> Result<String, String> {
    match val {
        ControlValue::String(s) => Ok(s.clone()),
        other => Err(format!("expected string, found {other:?}")),
    }
}

impl InlistBundle {
    #[rustfmt::skip]
    pub(crate) fn populate_controls_phase5b(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        let mw = &mut self.controls.mix_wind;
        map_f64s!("controls", ctrl, mw, [
            overshoot_d_min, overshoot_brunt_b_max, min_overshoot_q, overshoot_alpha,
            burn_z_mix_region_logt, burn_he_mix_region_logt, burn_h_mix_region_logt,
            max_y_for_burn_z_mix_region, max_x_for_burn_he_mix_region,
            conv_premix_time_factor,
            min_logt_for_make_gradr_sticky_in_solver_iters,
            mass_lower_limit_for_min_d_mix, mass_upper_limit_for_min_d_mix,
            min_center_ye_for_min_d_mix, clip_d_limit, min_convective_gap,
            min_thermohaline_gap, min_thermohaline_dropout,
            max_dropout_gradl_sub_grada, min_semiconvection_gap,
            rgb_to_agb_wind_switch, bjorklund_scaling_factor,
            de_jager_scaling_factor, van_loon_scaling_factor,
            nieuwenhuijzen_scaling_factor, min_wind, max_wind,
            mdot_omega_power, max_rotational_mdot_boost,
            max_mdot_jump_for_rotation, rotational_mdot_boost_fac,
            rotational_mdot_kh_fac, min_years_dt_for_redo_mdot,
            surf_omega_div_omega_crit_tol, mdot_revise_factor,
            implicit_mdot_boost, surf_avg_tau_min, surf_avg_tau,
            am_d_mix_factor, super_eddington_wind_ledd_factor,
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
            max_dq, min_dq, min_dq_for_split, min_dq_for_xa,
            min_dq_for_xa_convective, mesh_min_dlnr, mesh_max_allowed_ratio,
            max_delta_x_for_merge, max_surface_cell_dq,
            max_center_cell_dq, p_function_weight, t_function1_weight,
            r_function_weight, m_function_weight, m_function_param,
            gradt_function_weight, omega_function_weight,
            gam_function_weight, remesh_dt_limit,
            delta_mdot_atol, delta_mdot_rtol,
            delta_mdot_limit, delta_mdot_hard_limit, w_div_wcrit_max,
            am_gradmu_factor, d_dsi_factor, d_sh_factor,
            d_ssi_factor, d_es_factor, d_gsf_factor, d_st_factor, d_visc_factor,
            am_nu_factor, am_nu_non_rotation_factor, am_nu_dsi_factor,
            am_nu_sh_factor, am_nu_ssi_factor, am_nu_es_factor,
            am_nu_gsf_factor, am_nu_st_factor, am_nu_visc_factor,
            am_sig_term_limit,
            adjust_j_fraction, min_q_for_adjust_j_lost, min_j_div_delta_j,
            threshold_for_smooth_gradl_composition_term,
            threshold_for_smooth_brunt_b, min_magnitude_brunt_b,
            brunt_n2_coefficient, t_mix_limit, min_d_mix_below_tmax,
            min_d_mix_in_h_he, d_mix_zero_region_bottom_q,
            d_mix_zero_region_top_q, dq_d_mix_zero_at_h_he_crossover,
            dq_d_mix_zero_at_h_c_crossover, prune_bad_cz_min_hp_height,
            prune_bad_cz_min_log_eps_nuc, sig_min_factor_for_high_tcenter,
            tcenter_min_for_sig_min_factor_full_on,
            tcenter_max_for_sig_min_factor_full_off,
            max_delta_m_to_bdy_for_sig_min_factor,
            delta_m_upper_for_sig_min_factor,
            delta_m_lower_for_sig_min_factor, remove_small_d_limit,
            max_logt_for_mlt, max_v_for_convection,
            max_q_for_convection_with_hydro_on, max_v_div_cs_for_convection,
            max_abs_du_div_cs_for_convection, mlt_gradt_fraction,
            superad_reduction_gamma_limit,
            superad_reduction_gamma_limit_scale,
            superad_reduction_gamma_inv_scale,
            superad_reduction_diff_grads_limit, superad_reduction_limit,
            gradt_excess_f1, gradt_excess_f2, gradt_excess_age_fraction,
            gradt_excess_max_change, gradt_excess_lambda1, gradt_excess_beta1,
            gradt_excess_lambda2, gradt_excess_beta2, gradt_excess_dlambda,
            gradt_excess_dbeta, gradt_excess_max_center_h1,
            gradt_excess_min_center_he4, gradt_excess_max_logt,
            gradt_excess_min_log_tau_full_on,
            gradt_excess_max_log_tau_full_off,
        ]);
        map_bools!("controls", ctrl, mw, [
            limit_overshoot_hp_using_size_of_convection_zone,
            conv_premix_avoid_increase, conv_premix_fix_pgas,
            conv_premix_dump_snapshots,
            make_gradr_sticky_in_solver_iters,
            set_min_d_mix, set_min_d_mix_below_tmax, set_min_d_mix_in_h_he,
            recalc_mix_info_after_evolve, okay_to_remove_mixing_singleton,
            remove_embedded_semiconvection,
            use_t_tau_gradr_factor, alt_scale_height_flag,
            mlt_use_rotation_correction, mlt_make_surface_no_mixing,
            no_mlt_below_shock, use_superad_reduction,
            okay_to_reduce_gradt_excess,
            no_wind_if_no_rotation, use_accreted_material_j,
            premix_omega, do_adjust_j_lost, skip_rotation_in_convection_zones,
        ]);
        map_strs!("controls", ctrl, mw, [hot_wind_scheme, dutch_wind_lowt_scheme]);
        if ctrl.try_get_str("controls", "hot_wind_scheme")?.is_some()
            && crate::winds::WindScheme::parse_name(&mw.hot_wind_scheme).is_none()
        {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&controls::hot_wind_scheme' option '{}'",
                mw.hot_wind_scheme
            )));
        }
        if ctrl.try_get_str("controls", "dutch_wind_lowt_scheme")?.is_some()
            && crate::winds::DutchLowTScheme::parse_name(&mw.dutch_wind_lowt_scheme)
                .is_none()
        {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&controls::dutch_wind_lowt_scheme' option '{}'",
                mw.dutch_wind_lowt_scheme
            )));
        }
        ctrl.try_usize_into("controls", &mut mw.max_mdot_redo_cnt, "max_mdot_redo_cnt", 0)?;
        ctrl.try_usize_into(
            "controls",
            &mut mw.num_cells_for_smooth_gradl_composition_term,
            "num_cells_for_smooth_gradl_composition_term",
            0,
        )?;
        ctrl.try_usize_into(
            "controls",
            &mut mw.num_cells_for_smooth_brunt_b,
            "num_cells_for_smooth_brunt_b",
            0,
        )?;
        ctrl.try_usize_into(
            "controls",
            &mut mw.tdc_num_innermost_cells_forced_nonturbulent,
            "tdc_num_innermost_cells_forced_nonturbulent",
            0,
        )?;
        ctrl.try_usize_into(
            "controls",
            &mut mw.tdc_num_outermost_cells_forced_nonturbulent,
            "tdc_num_outermost_cells_forced_nonturbulent",
            0,
        )?;

        let n_ov = mw.overshoot_sets.len();
        apply_fortran_1based(ctrl, "overshoot_scheme", n_ov, |i, v| mw.overshoot_sets[i].scheme = v, val_str)?;
        apply_fortran_1based(ctrl, "overshoot_zone_type", n_ov, |i, v| mw.overshoot_sets[i].zone_type = v, val_str)?;
        apply_fortran_1based(ctrl, "overshoot_zone_loc", n_ov, |i, v| mw.overshoot_sets[i].zone_loc = v, val_str)?;
        apply_fortran_1based(ctrl, "overshoot_bdy_loc", n_ov, |i, v| mw.overshoot_sets[i].bdy_loc = v, val_str)?;
        for (idx, set) in mw.overshoot_sets.iter().enumerate() {
            let s_low = set.scheme.trim().to_ascii_lowercase();
            if !matches!(s_low.as_str(), "" | "none" | "exponential" | "step" | "other") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized/unsupported '&controls::overshoot_scheme({})' = '{}'",
                    idx + 1,
                    set.scheme
                )));
            }
            let zt_low = set.zone_type.trim().to_ascii_lowercase();
            if !matches!(zt_low.as_str(), "" | "any" | "burn_h" | "burn_he" | "burn_z" | "nonburn") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::overshoot_zone_type({})' = '{}'",
                    idx + 1,
                    set.zone_type
                )));
            }
            let zl_low = set.zone_loc.trim().to_ascii_lowercase();
            if !matches!(zl_low.as_str(), "" | "any" | "core" | "shell") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::overshoot_zone_loc({})' = '{}'",
                    idx + 1,
                    set.zone_loc
                )));
            }
            let bl_low = set.bdy_loc.trim().to_ascii_lowercase();
            if !matches!(bl_low.as_str(), "" | "any" | "top" | "bottom") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::overshoot_bdy_loc({})' = '{}'",
                    idx + 1,
                    set.bdy_loc
                )));
            }
        }
        apply_fortran_1based(ctrl, "overshoot_f", n_ov, |i, v| mw.overshoot_sets[i].f = v, val_f64)?;
        apply_fortran_1based(ctrl, "overshoot_f0", n_ov, |i, v| mw.overshoot_sets[i].f0 = v, val_f64)?;
        apply_fortran_1based(ctrl, "overshoot_d0", n_ov, |i, v| mw.overshoot_sets[i].d0 = v, val_f64)?;
        apply_fortran_1based(ctrl, "overshoot_delta0", n_ov, |i, v| mw.overshoot_sets[i].delta0 = v, val_f64)?;
        apply_fortran_1based(ctrl, "overshoot_mass_full_on", n_ov, |i, v| mw.overshoot_sets[i].mass_full_on = v, val_f64)?;
        apply_fortran_1based(ctrl, "overshoot_mass_full_off", n_ov, |i, v| mw.overshoot_sets[i].mass_full_off = v, val_f64)?;

        let n_pr = mw.predictive_sets.len();
        apply_fortran_1based(ctrl, "predictive_mix", n_pr, |i, v| mw.predictive_sets[i].predictive_mix = v, val_bool)?;
        apply_fortran_1based(ctrl, "predictive_zone_type", n_pr, |i, v| mw.predictive_sets[i].zone_type = v, val_str)?;
        apply_fortran_1based(ctrl, "predictive_zone_loc", n_pr, |i, v| mw.predictive_sets[i].zone_loc = v, val_str)?;
        apply_fortran_1based(ctrl, "predictive_bdy_loc", n_pr, |i, v| mw.predictive_sets[i].bdy_loc = v, val_str)?;
        apply_fortran_1based(ctrl, "predictive_bdy_q_min", n_pr, |i, v| mw.predictive_sets[i].bdy_q_min = v, val_f64)?;
        apply_fortran_1based(ctrl, "predictive_bdy_q_max", n_pr, |i, v| mw.predictive_sets[i].bdy_q_max = v, val_f64)?;
        apply_fortran_1based(ctrl, "predictive_superad_thresh", n_pr, |i, v| mw.predictive_sets[i].superad_thresh = v, val_f64)?;
        apply_fortran_1based(ctrl, "predictive_avoid_reversal", n_pr, |i, v| mw.predictive_sets[i].avoid_reversal = v, val_str)?;
        apply_fortran_1based(ctrl, "predictive_limit_ingestion", n_pr, |i, v| mw.predictive_sets[i].limit_ingestion = v, val_str)?;
        apply_fortran_1based(ctrl, "predictive_ingestion_factor", n_pr, |i, v| mw.predictive_sets[i].ingestion_factor = v, val_f64)?;
        for (idx, set) in mw.predictive_sets.iter().enumerate() {
            let zt = set.zone_type.trim();
            if !matches!(zt, "" | "any" | "burn_H" | "burn_He" | "burn_Z" | "nonburn") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::predictive_zone_type({})' = '{}'",
                    idx + 1,
                    set.zone_type
                )));
            }
            let zl = set.zone_loc.trim();
            if !matches!(zl, "" | "any" | "core" | "shell" | "surf") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::predictive_zone_loc({})' = '{}'",
                    idx + 1,
                    set.zone_loc
                )));
            }
            let bl = set.bdy_loc.trim();
            if !matches!(bl, "" | "any" | "top" | "bottom") {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::predictive_bdy_loc({})' = '{}'",
                    idx + 1,
                    set.bdy_loc
                )));
            }
        }

        const NUM_XA_FUNCTION: usize = 9;
        if mw.xa_function_species.len() < NUM_XA_FUNCTION {
            mw.xa_function_species.resize(NUM_XA_FUNCTION, String::new());
        }
        if mw.xa_function_weight.len() < NUM_XA_FUNCTION {
            mw.xa_function_weight.resize(NUM_XA_FUNCTION, 0.0);
        }
        if mw.xa_function_param.len() < NUM_XA_FUNCTION {
            mw.xa_function_param.resize(NUM_XA_FUNCTION, 0.0);
        }
        apply_fortran_1based(ctrl, "xa_function_species", NUM_XA_FUNCTION, |i, v| mw.xa_function_species[i] = v, val_str)?;
        apply_fortran_1based(ctrl, "xa_function_weight", NUM_XA_FUNCTION, |i, v| mw.xa_function_weight[i] = v, val_f64)?;
        apply_fortran_1based(ctrl, "xa_function_param", NUM_XA_FUNCTION, |i, v| mw.xa_function_param[i] = v, val_f64)?;

        let ts = &mut self.controls.ts_stop;
        map_f64s!("controls", ctrl, ts, [
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
            delta_lg_xh_cntr_min, delta_lg_xh_cntr_max,
            delta_lg_xh_cntr_limit, delta_lg_xh_cntr_hard_limit,
            delta_lg_xhe_cntr_min, delta_lg_xhe_cntr_max,
            delta_lg_xhe_cntr_limit, delta_lg_xhe_cntr_hard_limit,
            delta_lg_xc_cntr_min, delta_lg_xc_cntr_max,
            delta_lg_xc_cntr_limit, delta_lg_xc_cntr_hard_limit,
            delta_hr_ds_l, delta_hr_ds_teff, delta_hr_limit, delta_hr_hard_limit,
            delta_lg_star_mass_limit, delta_lg_star_mass_hard_limit,
            limit_for_rel_error_in_energy_conservation,
            hard_limit_for_rel_error_in_energy_conservation,
            dt_div_min_dr_div_cs_limit, dt_div_min_dr_div_cs_hard_limit,
            dt_div_dt_cell_collapse_limit, dt_div_dt_cell_collapse_hard_limit,
        ]);
        if !(ts.timestep_factor_for_retries.is_finite()
            && ts.timestep_factor_for_retries > 0.0
            && ts.timestep_factor_for_retries < 1.0)
        {
            return Err(StarError::InvalidInput(format!(
                "'&controls::timestep_factor_for_retries' must be in (0, 1), got {}",
                ts.timestep_factor_for_retries
            )));
        }
        map_bools!("controls", ctrl, ts, [
            use_dt_low_pass_controller,
            never_skip_hard_limits, relax_hard_limits_after_retry,
            delta_lg_xh_drop_only, delta_lg_xhe_drop_only, delta_lg_xc_drop_only,
        ]);
        map_opt_f64s!("controls", ctrl, ts, [
            non_fe_core_infall_limit, power_h_burn_upper_limit,
            power_h_burn_lower_limit, power_he_burn_lower_limit,
            power_z_burn_upper_limit, power_z_burn_lower_limit,
        ]);
        if let Some(v) = ctrl.try_get_f64("controls", "eta_center_limit")? {
            ts.eta_center_limit = (v < 1.0e90).then_some(v);
        }
        ctrl.try_usize_into("controls", &mut ts.retry_hold, "retry_hold", 0)?;
        ctrl.try_usize_into("controls", &mut ts.neg_mass_fraction_hold, "neg_mass_fraction_hold", 0)?;
        // Upstream applies these limits only when positive and accepts any
        // integer (`star/private/evolve.f90:2161,2214`), so a non-positive
        // value is stored as 0, meaning no limit. Upstream's own
        // Upstream binary/star inlists can set `redo_limit = -1` for unlimited redos.
        if let Some(v) = ctrl.try_get_i64("controls", "retry_limit")? {
            ts.retry_limit = if v <= 0 { 0 } else { v as usize };
        }
        if let Some(v) = ctrl.try_get_i64("controls", "redo_limit")? {
            ts.redo_limit = if v <= 0 { 0 } else { v as usize };
        }
        ctrl.try_isize_into("controls", &mut ts.solver_iters_timestep_limit, "solver_iters_timestep_limit")?;
        ctrl.try_isize_into("controls", &mut ts.gold_solver_iters_timestep_limit, "gold_solver_iters_timestep_limit")?;
        ctrl.try_isize_into("controls", &mut ts.gold2_solver_iters_timestep_limit, "gold2_solver_iters_timestep_limit")?;
        ctrl.try_isize_into("controls", &mut ts.burn_steps_limit, "burn_steps_limit")?;
        ctrl.try_isize_into("controls", &mut ts.diffusion_steps_limit, "diffusion_steps_limit")?;
        ctrl.try_isize_into("controls", &mut ts.diffusion_steps_hard_limit, "diffusion_steps_hard_limit")?;
        ctrl.try_isize_into("controls", &mut ts.diffusion_iters_limit, "diffusion_iters_limit")?;
        ctrl.try_isize_into("controls", &mut ts.diffusion_iters_hard_limit, "diffusion_iters_hard_limit")?;

        let n_dx = ts.dx_limit.len();
        apply_fortran_1based(ctrl, "dx_limit_species", n_dx, |i, v| ts.dx_limit_species[i] = v, val_str)?;
        apply_fortran_1based(ctrl, "dx_limit_min_x", n_dx, |i, v| ts.dx_limit_min_x[i] = v, val_f64)?;
        apply_fortran_1based(ctrl, "dx_limit", n_dx, |i, v| ts.dx_limit[i] = v, val_f64)?;
        apply_fortran_1based(ctrl, "dx_hard_limit", n_dx, |i, v| ts.dx_hard_limit[i] = v, val_f64)?;
        apply_fortran_1based(ctrl, "dx_decreases_only", n_dx, |i, v| ts.dx_decreases_only[i] = v, val_bool)?;
        apply_fortran_1based(ctrl, "dx_div_x_limit_min_x", n_dx, |i, v| ts.dx_div_x_limit_min_x[i] = v, val_f64)?;
        apply_fortran_1based(ctrl, "dx_div_x_limit", n_dx, |i, v| ts.dx_div_x_limit[i] = v, val_f64)?;
        apply_fortran_1based(ctrl, "dx_div_x_hard_limit", n_dx, |i, v| ts.dx_div_x_hard_limit[i] = v, val_f64)?;
        Ok(())
    }
}
