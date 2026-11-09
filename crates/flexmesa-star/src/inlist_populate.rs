// SPDX-License-Identifier: LGPL-3.0-or-later
//! Typed control struct population (`InlistBundle::populate_typed_structs`)
//! mapping flattened `&star_job`, `&eos`, `&kap`, and `&controls` namelist
//! sections into [`StarJobControls`], [`EosConfig`], [`KapConfig`], and
//! [`StarControls`].

use flexmesa_rates::ScreeningMode;
use flexmesa_schema::ControlValue;
use flexmesa_turb::MltOption;

use crate::controls::{AtmOption, AtmTTauRelation, EnergyEqnOption};
use crate::inlist::{InlistBundle, SectionControls};

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

    fn populate_star_job(&mut self) {
        let Some(sj) = self.sections.get("star_job") else {
            return;
        };
        let j = &mut self.job;
        if let Some(b) = sj.any_bool(&[
            "create_pre_main_sequence_model",
            "create_pre_ms_model",
        ]) {
            j.create_pre_ms_model = b;
        }
        sj.f64_into(&mut j.pre_ms_t_c, "pre_ms_t_c");
        sj.f64_into(&mut j.pre_ms_guess_rho_c, "pre_ms_guess_rho_c");
        sj.f64_into(&mut j.pre_ms_d_log10_p, "pre_ms_d_log10_p");
        sj.f64_into(&mut j.pre_ms_log_p_surf_limit, "pre_ms_log_p_surf_limit");
        sj.f64_into(&mut j.pre_ms_log_t_surf_limit, "pre_ms_log_t_surf_limit");
        sj.f64_into(&mut j.pre_ms_log_lum_factor, "pre_ms_log_lum_factor");
        if let Some(v) =
            sj.get_i64("pre_ms_relax_num_steps").filter(|&v| v >= 0)
        {
            j.pre_ms_relax_num_steps = v as usize;
        }
        sj.bool_into(&mut j.create_initial_model, "create_initial_model");
        sj.f64_into(
            &mut j.radius_in_cm_for_create_initial_model,
            "radius_in_cm_for_create_initial_model",
        );
        sj.f64_into(
            &mut j.mass_in_gm_for_create_initial_model,
            "mass_in_gm_for_create_initial_model",
        );
        sj.bool_into(&mut j.create_rsp_model, "create_rsp_model");
        sj.bool_into(&mut j.load_saved_model, "load_saved_model");
        if let Some(s) = sj
            .get_str("saved_model_name")
            .or_else(|| sj.get_str("load_model_filename"))
        {
            j.saved_model_name = s.to_string();
        }
        sj.bool_into(
            &mut j.save_model_when_terminate,
            "save_model_when_terminate",
        );
        sj.str_into(&mut j.save_model_filename, "save_model_filename");
        if let Some(v) = sj.get_i64("save_model_number").filter(|&v| v > 0) {
            j.save_model_number = Some(v as usize);
        }
        sj.bool_into(
            &mut j.save_photo_when_terminate,
            "save_photo_when_terminate",
        );
        sj.str_into(&mut j.save_photo_filename, "save_photo_filename");
        sj.bool_into(
            &mut j.save_pulse_data_when_terminate,
            "save_pulse_data_when_terminate",
        );
        sj.str_into(
            &mut j.save_pulse_data_filename,
            "save_pulse_data_filename",
        );
        sj.str_into(&mut j.pulse_data_format, "pulse_data_format");
        sj.bool_into(
            &mut j.save_stella_data_when_terminate,
            "save_stella_data_when_terminate",
        );
        sj.str_into(
            &mut j.save_stella_data_filename,
            "save_stella_data_filename",
        );
        if let Some(b) = sj.any_bool(&["change_net", "change_initial_net"]) {
            j.change_net = b;
        }
        if let Some(s) = sj.get_str("new_net_name") {
            j.new_net_name = s.to_string();
            if j.change_net {
                j.net_name = s.to_string();
            }
        }
        sj.bool_into(&mut j.auto_extend_net, "auto_extend_net");
        sj.str_into(&mut j.h_he_net, "h_he_net");
        sj.str_into(&mut j.co_net, "co_net");
        sj.str_into(&mut j.adv_net, "adv_net");

        if let Some(b) =
            sj.any_bool(&["change_v_flag", "change_initial_v_flag"])
        {
            j.change_v_flag = b;
        }
        if let Some(b) = sj.get_bool("new_v_flag") {
            j.new_v_flag = b;
            if j.change_v_flag {
                self.controls.v_flag = b;
            }
        }
        if let Some(b) =
            sj.any_bool(&["change_u_flag", "change_initial_u_flag"])
        {
            j.change_u_flag = b;
        }
        if let Some(b) = sj.get_bool("new_u_flag") {
            j.new_u_flag = b;
            if j.change_u_flag {
                self.controls.u_flag = b;
            }
        }
        if let Some(b) = sj
            .any_bool(&["change_rotation_flag", "change_initial_rotation_flag"])
        {
            j.change_rotation_flag = b;
        }
        if let Some(b) = sj.get_bool("new_rotation_flag") {
            j.new_rotation_flag = b;
            if j.change_rotation_flag {
                self.controls.rotation_flag = b;
            }
        }
        j.new_omega = sj.get_f64("new_omega");
        j.new_omega_div_omega_crit = sj.get_f64("new_omega_div_omega_crit");
        j.new_surface_rotation_v = sj.get_f64("new_surface_rotation_v");

        sj.bool_into(&mut j.set_initial_age, "set_initial_age");
        sj.f64_into(&mut j.initial_age, "initial_age");
        sj.bool_into(
            &mut j.set_initial_model_number,
            "set_initial_model_number",
        );
        if let Some(v) = sj.get_i64("initial_model_number").filter(|&v| v >= 0)
        {
            j.initial_model_number = v as usize;
        }
        sj.bool_into(&mut j.set_initial_dt, "set_initial_dt");
        sj.f64_into(&mut j.years_for_initial_dt, "years_for_initial_dt");
        sj.f64_into(&mut j.seconds_for_initial_dt, "seconds_for_initial_dt");
        sj.bool_into(
            &mut j.set_uniform_initial_composition,
            "set_uniform_initial_composition",
        );
        sj.f64_into(&mut j.initial_h1, "initial_h1");
        sj.f64_into(&mut j.initial_h2, "initial_h2");
        sj.f64_into(&mut j.initial_he3, "initial_he3");
        sj.f64_into(&mut j.initial_he4, "initial_he4");
        if let Some(v) = sj.get_i64("initial_zfracs") {
            j.initial_zfracs = v as i32;
        }
        if let Some(b) = sj.any_bool(&[
            "relax_mass",
            "relax_initial_mass",
            "relax_mass_scale",
            "relax_initial_mass_scale",
        ]) {
            j.relax_mass = b;
        }
        sj.f64_into(&mut j.new_mass, "new_mass");
        if let Some(b) = sj.any_bool(&["relax_z", "relax_initial_z"]) {
            j.relax_z = b;
        }
        sj.f64_into(&mut j.new_z, "new_z");
        if let Some(b) = sj.any_bool(&["relax_y", "relax_initial_y"]) {
            j.relax_y = b;
        }
        sj.f64_into(&mut j.new_y, "new_y");
        if let Some(b) = sj.any_bool(&[
            "set_tau_factor",
            "set_initial_tau_factor",
            "relax_to_this_tau_factor",
            "relax_initial_to_this_tau_factor",
        ]) {
            j.set_tau_factor = b;
        }
        if let Some(v) = sj
            .get_f64("set_to_this_tau_factor")
            .or_else(|| sj.get_f64("relax_to_this_tau_factor"))
        {
            j.set_to_this_tau_factor = v;
        }
        if let Some(b) = sj.any_bool(&[
            "relax_core",
            "relax_initial_core",
            "relax_m_center",
            "relax_initial_m_center",
        ]) {
            j.relax_m_center = b;
        }
        if let Some(v) = sj
            .get_f64("new_core_mass")
            .or_else(|| sj.get_f64("new_mass").filter(|_| j.relax_m_center))
        {
            j.new_core_mass = v;
        }
        if let Some(b) =
            sj.any_bool(&["relax_r_center", "relax_initial_r_center"])
        {
            j.relax_r_center = b;
        }
        sj.f64_into(&mut j.new_r_center, "new_r_center");
        if let Some(b) =
            sj.any_bool(&["relax_l_center", "relax_initial_l_center"])
        {
            j.relax_l_center = b;
        }
        sj.f64_into(&mut j.new_l_center, "new_l_center");
        sj.str_into(
            &mut j.required_termination_code_string,
            "required_termination_code_string",
        );
        sj.bool_into(&mut j.pgstar_flag, "pgstar_flag");
    }

    fn populate_eos_kap(&mut self) {
        if let Some(eos) = self.sections.get("eos") {
            eos.bool_into(&mut self.eos_config.use_free_eos, "use_freeeos");
            eos.bool_into(&mut self.eos_config.use_opal_scvh, "use_opal_scvh");
            eos.bool_into(&mut self.eos_config.use_pc, "use_pc");
            eos.bool_into(&mut self.eos_config.use_skye, "use_skye");
            eos.bool_into(&mut self.eos_config.use_cms, "use_cms");
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

    fn populate_controls_physics(&mut self) {
        let has_kap_zbase = self
            .sections
            .get("kap")
            .and_then(|k| k.get_f64("zbase"))
            .is_some();
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        let c = &mut self.controls;
        ctrl.f64_into(&mut c.initial_mass, "initial_mass");
        if let Some(v) = ctrl.get_f64("initial_z") {
            c.initial_z = v;
            if !has_kap_zbase {
                c.zbase = v;
                self.kap_config.zbase = v;
            }
        }
        ctrl.f64_into(&mut c.initial_y, "initial_y");
        ctrl.f64_into(&mut c.min_timestep_limit, "min_timestep_limit");
        ctrl.f64_into(&mut c.mixing_length_alpha, "mixing_length_alpha");
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
        ctrl.bool_into(&mut c.use_ledoux_criterion, "use_ledoux_criterion");
        ctrl.f64_into(&mut c.alpha_semiconvection, "alpha_semiconvection");
        ctrl.f64_into(&mut c.thermohaline_coeff, "thermohaline_coeff");
        ctrl.f64_into(&mut c.min_d_mix, "min_d_mix");
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
        ctrl.f64_into(&mut c.tau_factor, "tau_factor");
        ctrl.f64_into(&mut c.opacity_factor, "opacity_factor");
        ctrl.f64_into(&mut c.non_nuc_neu_factor, "non_nuc_neu_factor");
        ctrl.bool_into(&mut c.op_split_burn, "op_split_burn");
        ctrl.f64_into(&mut c.eps_nuc_factor, "eps_nuc_factor");
        ctrl.f64_into(&mut c.dxdt_nuc_factor, "dxdt_nuc_factor");
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
        ctrl.f64_into(&mut c.mesh_delta_coeff, "mesh_delta_coeff");
        ctrl.f64_into(&mut c.varcontrol_target, "varcontrol_target");
        ctrl.f64_into(&mut c.mass_change, "mass_change");
        ctrl.f64_into(&mut c.reimers_scaling_factor, "reimers_scaling_factor");
        ctrl.f64_into(&mut c.blocker_scaling_factor, "blocker_scaling_factor");
        ctrl.f64_into(&mut c.vink_scaling_factor, "vink_scaling_factor");
        ctrl.f64_into(&mut c.dutch_scaling_factor, "dutch_scaling_factor");
        ctrl.str_into(&mut c.cool_wind_rgb_scheme, "cool_wind_rgb_scheme");
        ctrl.str_into(&mut c.cool_wind_agb_scheme, "cool_wind_agb_scheme");
        ctrl.bool_into(&mut c.do_element_diffusion, "do_element_diffusion");
        ctrl.bool_into(&mut c.do_conv_premix, "do_conv_premix");
        ctrl.bool_into(&mut c.do_phase_separation, "do_phase_separation");
        if let Some(v) = ctrl.get_i64("rsp_max_num_periods") {
            c.rsp_max_num_periods = Some(v as isize);
        }
        if let Some(v) = ctrl
            .get_i64("rsp_target_steps_per_cycle")
            .filter(|&v| v > 0)
        {
            c.rsp_target_steps_per_cycle = v as usize;
        }
        ctrl.f64_into(&mut c.rsp_mass, "rsp_mass");
        ctrl.f64_into(&mut c.rsp_teff, "rsp_teff");
        ctrl.f64_into(&mut c.rsp_l, "rsp_l");
        ctrl.f64_into(&mut c.rsp_x, "rsp_x");
        ctrl.f64_into(&mut c.rsp_z, "rsp_z");

        ctrl.bool_into(&mut c.use_other_cgrav, "use_other_cgrav");
        ctrl.bool_into(&mut c.use_other_alpha_mlt, "use_other_alpha_mlt");
        ctrl.bool_into(&mut c.use_other_wind, "use_other_wind");
        ctrl.bool_into(&mut c.use_other_adjust_mdot, "use_other_adjust_mdot");
        ctrl.bool_into(&mut c.use_other_torque, "use_other_torque");
        ctrl.bool_into(&mut c.use_other_energy, "use_other_energy");
        ctrl.bool_into(&mut c.use_other_eps_grav, "use_other_eps_grav");
        ctrl.bool_into(&mut c.use_other_neu, "use_other_neu");
        ctrl.bool_into(&mut c.use_other_kap, "use_other_kap");
        ctrl.bool_into(&mut c.use_other_surface_pt, "use_other_surface_pt");
        ctrl.bool_into(
            &mut c.use_other_overshooting_scheme,
            "use_other_overshooting_scheme",
        );
        ctrl.bool_into(
            &mut c.use_other_momentum_implicit,
            "use_other_momentum_implicit",
        );
    }

    fn populate_controls_limits(&mut self) {
        let Some(ctrl) = self.sections.get("controls") else {
            return;
        };
        let c = &mut self.controls;
        if let Some(v) = ctrl.get_i64("max_model_number").filter(|&v| v > 0) {
            c.max_model_number = Some(v as usize);
        }
        c.max_age = ctrl.get_f64("max_age").filter(|&v| v > 0.0);
        c.max_age_in_seconds =
            ctrl.get_f64("max_age_in_seconds").filter(|&v| v > 0.0);
        c.max_years_for_timestep =
            ctrl.get_f64("max_years_for_timestep").filter(|&v| v > 0.0);
        ctrl.bool_into(&mut c.stop_near_zams, "stop_near_zams");
        if let Some(v) = ctrl.get_f64("lnuc_div_l_zams_limit") {
            c.lnuc_div_l_zams_limit = v;
            if ctrl.get_bool("stop_near_zams").is_none() {
                c.stop_near_zams = true;
            }
        }
        c.xa_central_lower_limit_species =
            ctrl.get_indexed_strings("xa_central_lower_limit_species");
        c.xa_central_lower_limit =
            ctrl.get_indexed_f64s("xa_central_lower_limit");
        c.xa_central_upper_limit_species =
            ctrl.get_indexed_strings("xa_central_upper_limit_species");
        c.xa_central_upper_limit =
            ctrl.get_indexed_f64s("xa_central_upper_limit");

        c.log_l_upper_limit = ctrl.get_f64("log_l_upper_limit");
        c.log_l_lower_limit = ctrl.get_f64("log_l_lower_limit");
        c.teff_upper_limit = ctrl.get_f64("teff_upper_limit");
        c.teff_lower_limit = ctrl.get_f64("teff_lower_limit");
        c.log_teff_upper_limit = ctrl.get_f64("log_teff_upper_limit");
        c.log_teff_lower_limit = ctrl.get_f64("log_teff_lower_limit");
        c.log_center_temp_upper_limit =
            ctrl.get_f64("log_center_temp_upper_limit");
        c.log_center_temp_lower_limit =
            ctrl.get_f64("log_center_temp_lower_limit");
        c.log_center_density_upper_limit =
            ctrl.get_f64("log_center_density_upper_limit");
        c.log_center_density_lower_limit =
            ctrl.get_f64("log_center_density_lower_limit");
        c.log_max_temp_upper_limit = ctrl.get_f64("log_max_temp_upper_limit");
        c.log_g_upper_limit = ctrl.get_f64("log_g_upper_limit");
        c.log_g_lower_limit = ctrl.get_f64("log_g_lower_limit");
        c.log_psurf_upper_limit = ctrl.get_f64("log_psurf_upper_limit");
        c.log_rsurf_upper_limit = ctrl.get_f64("log_rsurf_upper_limit");
        c.center_r_lower_limit = ctrl.get_f64("center_r_lower_limit");
        c.he_core_mass_limit = ctrl.get_f64("he_core_mass_limit");
        c.co_core_mass_limit = ctrl.get_f64("co_core_mass_limit");
        c.one_core_mass_limit = ctrl.get_f64("one_core_mass_limit");
        c.envelope_mass_limit = ctrl.get_f64("envelope_mass_limit");
        c.he_layer_mass_lower_limit = ctrl.get_f64("he_layer_mass_lower_limit");
        c.power_nuc_burn_upper_limit =
            ctrl.get_f64("power_nuc_burn_upper_limit");
        c.power_nuc_burn_lower_limit =
            ctrl.get_f64("power_nuc_burn_lower_limit");
        c.power_he_burn_upper_limit = ctrl.get_f64("power_he_burn_upper_limit");
        c.star_mass_max_limit = ctrl.get_f64("star_mass_max_limit");
        c.star_mass_min_limit = ctrl.get_f64("star_mass_min_limit");
        c.remnant_mass_min_limit = ctrl.get_f64("remnant_mass_min_limit");
        c.star_species_mass_min_limit =
            ctrl.get_f64("star_species_mass_min_limit");
        ctrl.str_into(
            &mut c.star_species_mass_min_limit_iso,
            "star_species_mass_min_limit_iso",
        );
        c.star_species_mass_max_limit =
            ctrl.get_f64("star_species_mass_max_limit");
        ctrl.str_into(
            &mut c.star_species_mass_max_limit_iso,
            "star_species_mass_max_limit_iso",
        );
        c.fe_core_infall_limit = ctrl.get_f64("fe_core_infall_limit");
        c.v_surf_div_v_kh_upper_limit =
            ctrl.get_f64("v_surf_div_v_kh_upper_limit");
        ctrl.bool_into(&mut c.stop_at_phase_tp_agb, "stop_at_phase_tp_agb");
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
