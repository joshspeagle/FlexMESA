// SPDX-License-Identifier: LGPL-3.0-or-later
//! Typed control struct population (`InlistBundle::populate_typed_structs`)
//! mapping flattened `&star_job`, `&eos`, `&kap`, and `&controls` namelist
//! sections into [`StarJobControls`], [`EosConfig`], [`KapConfig`], and
//! [`StarControls`].

use flexmesa_rates::ScreeningMode;
use flexmesa_schema::ControlValue;
use flexmesa_turb::{MltOption, SemiconvectionOption, ThermohalineOption};

use crate::controls::{AtmOption, AtmTTauRelation, EnergyEqnOption};
use crate::inlist::{InlistBundle, SectionControls};

macro_rules! map_bools {
    ($sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $sec.bool_into(&mut $dst.$field, stringify!($field)); )*
    };
}

macro_rules! map_f64s {
    ($sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $sec.f64_into(&mut $dst.$field, stringify!($field)); )*
    };
}

macro_rules! map_strs {
    ($sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $sec.str_into(&mut $dst.$field, stringify!($field)); )*
    };
}

macro_rules! map_opt_f64s {
    ($sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $dst.$field = $sec.get_f64(stringify!($field)); )*
    };
}

impl SectionControls {
    #[inline]
    fn f64_into(&self, dst: &mut f64, key: &str) {
        if let Some(v) = self.get_f64(key) {
            *dst = v;
        }
    }

    #[inline]
    fn bool_into(&self, dst: &mut bool, key: &str) {
        if let Some(v) = self.get_bool(key) {
            *dst = v;
        }
    }

    #[inline]
    fn str_into(&self, dst: &mut String, key: &str) {
        if let Some(s) = self.get_str(key) {
            *dst = s.to_string();
        }
    }

    #[inline]
    fn any_bool(&self, keys: &[&str]) -> Option<bool> {
        keys.iter().find_map(|k| self.get_bool(k))
    }
}

impl InlistBundle {
    /// Populates `self.job`, `self.eos_config`, `self.kap_config`, and
    /// `self.controls` from `self.sections`.
    pub(crate) fn populate_typed_structs(&mut self) {
        self.populate_star_job();
        self.populate_eos_kap();
        self.populate_controls_physics();
        self.populate_controls_limits();
        self.populate_user_x_controls();
    }

    #[rustfmt::skip]
    fn populate_star_job(&mut self) {
        let Some(sj) = self.sections.get("star_job") else {
            return;
        };
        let j = &mut self.job;
        if let Some(b) = sj.any_bool(&["create_pre_main_sequence_model", "create_pre_ms_model"]) {
            j.create_pre_ms_model = b;
        }
        if let Some(v) = sj.get_i64("pre_ms_relax_num_steps").filter(|&v| v >= 0) {
            j.pre_ms_relax_num_steps = v as usize;
        }
        map_bools!(sj, j, [
            create_initial_model, use_other_build_initial_model, create_rsp_model,
            load_saved_model, save_model_when_terminate, save_photo_when_terminate,
            save_pulse_data_when_terminate, save_stella_data_when_terminate,
            auto_extend_net, set_initial_age, set_initial_model_number, set_initial_dt,
            set_uniform_initial_composition, relax_initial_to_xaccrete,
            set_initial_v_center, set_initial_l_center, new_rti_flag, pgstar_flag,
        ]);
        map_f64s!(sj, j, [
            pre_ms_t_c, pre_ms_guess_rho_c, pre_ms_d_log10_p,
            pre_ms_log_p_surf_limit, pre_ms_log_t_surf_limit, pre_ms_log_lum_factor,
            radius_in_cm_for_create_initial_model, mass_in_gm_for_create_initial_model,
            initial_age, years_for_initial_dt, seconds_for_initial_dt,
            initial_h1, initial_h2, initial_he3, initial_he4,
            new_mass, new_z, new_y, core_avg_rho, core_avg_eps, new_r_center, new_l_center,
        ]);
        map_strs!(sj, j, [
            save_model_filename, save_photo_filename, save_pulse_data_filename,
            pulse_data_format, save_stella_data_filename, h_he_net, co_net, adv_net,
            required_termination_code_string,
        ]);
        if let Some(s) = sj.get_str("saved_model_name").or_else(|| sj.get_str("load_model_filename")) {
            j.saved_model_name = s.to_string();
        }
        if let Some(v) = sj.get_i64("save_model_number").filter(|&v| v > 0) {
            j.save_model_number = Some(v as usize);
        }
        if let Some(b) = sj.any_bool(&["change_net", "change_initial_net"]) {
            j.change_net = b;
        }
        if let Some(s) = sj.get_str("new_net_name") {
            j.new_net_name = s.to_string();
            if j.change_net {
                j.net_name = s.to_string();
            }
        }
        if let Some(b) = sj.any_bool(&["change_v_flag", "change_initial_v_flag"]) {
            j.change_v_flag = b;
        }
        if let Some(b) = sj.get_bool("new_v_flag") {
            j.new_v_flag = b;
            if j.change_v_flag {
                self.controls.v_flag = b;
            }
        }
        if let Some(b) = sj.any_bool(&["change_u_flag", "change_initial_u_flag"]) {
            j.change_u_flag = b;
        }
        if let Some(b) = sj.get_bool("new_u_flag") {
            j.new_u_flag = b;
            if j.change_u_flag {
                self.controls.u_flag = b;
            }
        }
        if let Some(b) = sj.any_bool(&["change_rotation_flag", "change_initial_rotation_flag"]) {
            j.change_rotation_flag = b;
        }
        if let Some(b) = sj.get_bool("new_rotation_flag") {
            j.new_rotation_flag = b;
            if j.change_rotation_flag {
                self.controls.rotation_flag = b;
            }
        }
        map_opt_f64s!(sj, j, [new_omega, new_omega_div_omega_crit, new_surface_rotation_v]);
        if let Some(v) = sj.get_i64("initial_model_number").filter(|&v| v >= 0) {
            j.initial_model_number = v as usize;
        }
        if let Some(v) = sj.get_i64("initial_zfracs") {
            j.initial_zfracs = v as i32;
        }
        if let Some(b) = sj.any_bool(&[
            "relax_mass", "relax_initial_mass", "relax_mass_scale", "relax_initial_mass_scale",
        ]) {
            j.relax_mass = b;
        }
        if let Some(b) = sj.any_bool(&["relax_z", "relax_initial_z", "change_z", "change_initial_z"]) {
            j.relax_z = b;
        }
        if let Some(b) = sj.any_bool(&["relax_y", "relax_initial_y", "change_y", "change_initial_y"]) {
            j.relax_y = b;
        }
        if let Some(b) = sj.any_bool(&[
            "set_tau_factor", "set_initial_tau_factor", "relax_tau_factor",
            "relax_initial_tau_factor", "relax_to_this_tau_factor",
            "relax_initial_to_this_tau_factor",
        ]) {
            j.set_tau_factor = b;
        }
        if let Some(v) = sj.get_f64("set_to_this_tau_factor").or_else(|| sj.get_f64("relax_to_this_tau_factor")) {
            j.set_to_this_tau_factor = v;
        }
        if let Some(b) = sj.any_bool(&[
            "relax_core", "relax_initial_core", "relax_m_center", "relax_initial_m_center",
        ]) {
            j.relax_m_center = b;
        }
        if let Some(v) = sj.get_f64("new_core_mass").or_else(|| sj.get_f64("new_mass").filter(|_| j.relax_m_center)) {
            j.new_core_mass = v;
        }
        if let Some(b) = sj.any_bool(&["relax_r_center", "relax_initial_r_center"]) {
            j.relax_r_center = b;
        }
        if let Some(b) = sj.any_bool(&["relax_l_center", "relax_initial_l_center"]) {
            j.relax_l_center = b;
        }
        if let Some(b) = sj.any_bool(&["relax_irradiation", "relax_initial_irradiation"]) {
            j.relax_irradiation = b;
        }
        if let Some(v) = sj.get_f64("new_irradiation_flux").or_else(|| sj.get_f64("relax_to_this_irrad_flux")) {
            j.new_irradiation_flux = v;
        }
        if let Some(v) = sj.get_f64("column_depth_for_irradiation").or_else(|| sj.get_f64("irrad_col_depth")) {
            j.column_depth_for_irradiation = v;
        }
        if let Some(b) = sj.any_bool(&[
            "relax_initial_mass_to_remove_h_env", "remove_surface_by_relax_to_star_cut",
        ]) {
            j.relax_initial_mass_to_remove_h_env = b;
        }
        if let Some(b) = sj.any_bool(&["change_initial_rti_flag", "change_rti_flag"]) {
            j.change_initial_rti_flag = b;
        }
    }

    fn populate_eos_kap(&mut self) {
        if let Some(eos) = self.sections.get("eos") {
            eos.bool_into(&mut self.eos_config.use_free_eos, "use_freeeos");
            map_bools!(
                eos,
                self.eos_config,
                [use_opal_scvh, use_pc, use_skye, use_cms]
            );
        }
        if let Some(kap) = self.sections.get("kap") {
            if let Some(v) = kap.get_f64("zbase") {
                self.kap_config.zbase = v;
                self.controls.zbase = v;
            }
            kap.bool_into(
                &mut self.kap_config.use_type2_opacities,
                "use_type2_opacities",
            );
        }
    }

    #[rustfmt::skip]
    fn populate_controls_physics(&mut self) {
        let has_kap_zbase = self.sections.get("kap").and_then(|k| k.get_f64("zbase")).is_some();
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        let c = &mut self.controls;
        if let Some(v) = ctrl.get_f64("initial_z") {
            c.initial_z = v;
            if !has_kap_zbase {
                c.zbase = v;
                self.kap_config.zbase = v;
            }
        }
        map_f64s!(ctrl, c, [
            initial_mass, initial_y, min_timestep_limit, mixing_length_alpha,
            alpha_semiconvection, thermohaline_coeff, min_d_mix, tau_factor,
            opacity_factor, non_nuc_neu_factor, eps_nuc_factor, dxdt_nuc_factor,
            mesh_delta_coeff, varcontrol_target, mass_change, accretion_h1, accretion_he4,
            super_eddington_scaling_factor, hot_wind_full_on_t, cool_wind_full_on_t,
            reimers_scaling_factor, blocker_scaling_factor, vink_scaling_factor,
            dutch_scaling_factor, irradiation_flux, column_depth_for_irradiation,
            rsp_mass, rsp_teff, rsp_l, rsp_x, rsp_z,
            l_theta_for_velocity_time_centering, p_theta_for_velocity_time_centering,
            max_logt_for_include_p_and_l_in_velocity_time_centering, op_split_burn_min_t,
            tdc_alpha_m, dedt_rti_diffusion_factor, dudt_rti_diffusion_factor,
            drag_coefficient, min_q_for_drag, mlt_pturb_factor, pvsc_cq, pvsc_zshock,
            eps_grav_factor, gamma_ln_s_eps_grav_full_on, gamma_ln_s_eps_grav_full_off,
            velocity_q_upper_bound, velocity_tau_lower_bound, velocity_logt_lower_bound,
            mix_factor, pextra_factor,
        ]);
        map_strs!(ctrl, c, [
            zams_filename, cool_wind_rgb_scheme, cool_wind_agb_scheme, phase_separation_option,
        ]);
        map_bools!(ctrl, c, [
            use_ledoux_criterion, calculate_brunt_n2, use_brunt_gradmux_form,
            op_split_burn, accrete_same_as_surface,
            accrete_given_mass_fractions, do_element_diffusion, do_conv_premix,
            do_phase_separation, do_phase_separation_heating, use_other_cgrav,
            use_other_alpha_mlt, use_other_wind, use_other_adjust_mdot, use_other_torque,
            use_other_energy, use_other_eps_grav, use_other_neu, use_other_kap,
            use_other_surface_pt, use_other_overshooting_scheme, use_other_momentum_implicit,
            using_velocity_time_centering, include_l_in_velocity_time_centering,
            include_p_in_velocity_time_centering,
            use_p_d_1_div_rho_form_of_work_when_time_centering_velocity,
            nonlocal_nico_decay_heat, do_premix_heating,
            tdc_include_eturb_in_energy_equation, rti_flag, use_drag_energy,
            use_pvsc_art_visc, use_time_centered_eps_grav,
            include_composition_in_eps_grav, use_gravity_rotation_correction,
            use_mass_corrections, use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn,
            use_dprad_dm_form_of_t_gradient_eqn, use_rsp_l_eqn_outer_bc,
            do_burn, do_mix,
        ]);
        ctrl.bool_into(&mut self.job.use_other_build_initial_model, "use_other_build_initial_model");
        c.accretion_species_id = ctrl.get_indexed_strings("accretion_species_id");
        c.accretion_species_xa = ctrl.get_indexed_f64s("accretion_species_xa");

        if let Some(s) = ctrl.get_str("mlt_option") {
            c.mlt_option = match s.to_ascii_lowercase().as_str() {
                "henyey" => MltOption::Henyey,
                "ml1" => MltOption::Ml1,
                "ml2" => MltOption::Ml2,
                "mihalas" => MltOption::Mihalas,
                "tdc" => MltOption::Tdc,
                "none" => {
                    c.mixing_length_alpha = 0.0;
                    MltOption::Cox
                }
                _ => MltOption::Cox,
            };
        }
        if let Some(s) = ctrl.get_str("atm_option") {
            c.atm_option = match s.to_ascii_lowercase().as_str() {
                "table" => AtmOption::Table,
                "simple_photosphere" => AtmOption::SimplePhotosphere,
                _ => AtmOption::TTau,
            };
        }
        if let Some(s) = ctrl.get_str("atm_t_tau_relation") {
            c.atm_t_tau_relation = match s.to_ascii_lowercase().as_str() {
                "solar_hopf" => AtmTTauRelation::SolarHopf,
                "krishna_swamy" => AtmTTauRelation::KrishnaSwamy,
                "trampedach_solar" => AtmTTauRelation::TrampedachSolar,
                _ => AtmTTauRelation::Eddington,
            };
        }
        if let Some(s) = ctrl.get_str("screening_mode") {
            c.screening_mode = match s.to_ascii_lowercase().as_str() {
                "no_screening" | "" => ScreeningMode::NoScreening,
                "salpeter" => ScreeningMode::Salpeter,
                "chugunov" => ScreeningMode::Chugunov,
                _ => ScreeningMode::Extended,
            };
        }
        if let Some(s) = ctrl.get_str("energy_eqn_option") {
            c.energy_eqn_option = match s.to_ascii_lowercase().as_str() {
                "eps_grav" => EnergyEqnOption::EpsGrav,
                _ => EnergyEqnOption::Dedt,
            };
        }
        if let Some(s) = ctrl.get_str("semiconvection_option") {
            if let Ok(opt) = s.parse::<SemiconvectionOption>() {
                c.semiconvection_option = opt;
            }
        }
        if let Some(s) = ctrl.get_str("thermohaline_option") {
            if let Ok(opt) = s.parse::<ThermohalineOption>() {
                c.thermohaline_option = opt;
            }
        }
        if let Some(v) = ctrl.get_i64("rsp_max_num_periods") {
            c.rsp_max_num_periods = Some(v as isize);
        }
        if let Some(v) = ctrl.get_i64("rsp_target_steps_per_cycle").filter(|&v| v > 0) {
            c.rsp_target_steps_per_cycle = v as usize;
        }
    }

    #[rustfmt::skip]
    fn populate_controls_limits(&mut self) {
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

    fn populate_user_x_controls(&mut self) {
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
