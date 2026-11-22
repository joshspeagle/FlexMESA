// SPDX-License-Identifier: LGPL-3.0-or-later
//! Typed control struct population (`InlistBundle::populate_typed_structs`)
//! mapping flattened `&star_job`, `&eos`, `&kap`, and `&controls` namelist
//! sections into [`crate::StarJobControls`], [`flexmesa_eos::EosConfig`],
//! [`flexmesa_kap::KapConfig`], and [`crate::StarControls`].

use flexmesa_rates::ScreeningMode;
use flexmesa_turb::{MltOption, SemiconvectionOption, ThermohalineOption};

use crate::controls::{
    AtmOption, AtmTTauOpacity, AtmTTauRelation, EnergyEqnOption, StarError,
};
use crate::inlist::{InlistBundle, SectionControls};

macro_rules! map_bools {
    ($sec_name:expr, $sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $sec.try_bool_into($sec_name, &mut $dst.$field, stringify!($field))?; )*
    };
}

macro_rules! map_f64s {
    ($sec_name:expr, $sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $sec.try_f64_into($sec_name, &mut $dst.$field, stringify!($field))?; )*
    };
}

macro_rules! map_strs {
    ($sec_name:expr, $sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $( $sec.try_str_into($sec_name, &mut $dst.$field, stringify!($field))?; )*
    };
}

macro_rules! map_opt_f64s {
    ($sec_name:expr, $sec:expr, $dst:expr, [$($field:ident),* $(,)?]) => {
        $(
            if let Some(v) = $sec.try_get_f64($sec_name, stringify!($field))? {
                $dst.$field = Some(v);
            }
        )*
    };
}

pub(crate) use {map_bools, map_f64s, map_opt_f64s, map_strs};

impl SectionControls {
    /// Reads a finite scalar real for `&section::key` into `dst` when present.
    #[inline]
    pub fn try_f64_into(
        &self,
        section: &str,
        dst: &mut f64,
        key: &str,
    ) -> Result<(), StarError> {
        if let Some(v) = self.try_get_f64(section, key)? {
            *dst = v;
        }
        Ok(())
    }

    /// Reads a boolean scalar for `&section::key` into `dst` when present.
    #[inline]
    pub fn try_bool_into(
        &self,
        section: &str,
        dst: &mut bool,
        key: &str,
    ) -> Result<(), StarError> {
        if let Some(v) = self.try_get_bool(section, key)? {
            *dst = v;
        }
        Ok(())
    }

    /// Reads a string scalar for `&section::key` into `dst` when present.
    #[inline]
    pub fn try_str_into(
        &self,
        section: &str,
        dst: &mut String,
        key: &str,
    ) -> Result<(), StarError> {
        if let Some(s) = self.try_get_str(section, key)? {
            *dst = s.to_string();
        }
        Ok(())
    }

    /// Reads a non-negative integer >= `min_val` for `&section::key` into `dst`.
    #[inline]
    pub fn try_usize_into(
        &self,
        section: &str,
        dst: &mut usize,
        key: &str,
        min_val: usize,
    ) -> Result<(), StarError> {
        if let Some(v) = self.try_get_i64(section, key)? {
            if v < (min_val as i64) {
                return Err(StarError::InvalidInput(format!(
                    "'&{section}::{key}' must be >= {min_val}, got {v}"
                )));
            }
            *dst = v as usize;
        }
        Ok(())
    }

    /// Reads a signed integer for `&section::key` into `dst` when present.
    #[inline]
    pub fn try_isize_into(
        &self,
        section: &str,
        dst: &mut isize,
        key: &str,
    ) -> Result<(), StarError> {
        if let Some(v) = self.try_get_i64(section, key)? {
            *dst = v as isize;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn try_any_bool(
        &self,
        section: &str,
        keys: &[&str],
    ) -> Result<Option<bool>, StarError> {
        let mut found: Option<bool> = None;
        for k in keys {
            if let Some(b) = self.try_get_bool(section, k)? {
                if found.is_none() {
                    found = Some(b);
                }
            }
        }
        Ok(found)
    }

    #[inline]
    pub(crate) fn try_first_f64(
        &self,
        section: &str,
        keys: &[&str],
    ) -> Result<Option<f64>, StarError> {
        let mut found: Option<f64> = None;
        for k in keys {
            if let Some(v) = self.try_get_f64(section, k)? {
                if found.is_none() {
                    found = Some(v);
                }
            }
        }
        Ok(found)
    }
}

impl InlistBundle {
    /// Populates `self.job`, `self.eos_config`, `self.kap_config`, and
    /// `self.controls` from `self.sections`, validating types, ranges, and
    /// enumerated option strings (`#147`).
    ///
    /// # Errors
    /// Returns [`StarError::InvalidInput`] on any type mismatch, out-of-range
    /// integer/float, or unrecognized option string.
    pub(crate) fn populate_typed_structs_with_options(
        &mut self,
        allow_unsupported_controls: bool,
    ) -> Result<(), StarError> {
        self.populate_star_job()?;
        self.populate_eos_kap_with_options(allow_unsupported_controls)?;
        self.populate_controls_physics()?;
        self.populate_controls_output()?;
        self.populate_controls_solver()?;
        self.populate_controls_limits()?;
        self.populate_controls_phase5b()?;
        if let Some(ctrl) = self.sections.get("controls") {
            crate::controls_diffusion::populate_diffusion_controls(
                ctrl,
                &mut self.controls.diffusion,
            )?;
        }
        self.populate_user_x_controls()?;
        crate::rsp_config::populate_rsp_inlist(
            self.sections.get("star_job"),
            self.sections.get("controls"),
            &mut self.job,
            &mut self.controls,
        )?;
        Ok(())
    }

    #[rustfmt::skip]
    fn populate_star_job(&mut self) -> Result<(), StarError> {
        let Some(sj) = self.sections.get("star_job") else {
            return Ok(());
        };
        let j = &mut self.job;
        sj.try_bool_into("star_job", &mut j.create_pre_ms_model, "create_pre_main_sequence_model")?;
        sj.try_usize_into("star_job", &mut j.pre_ms_relax_num_steps, "pre_ms_relax_num_steps", 0)?;
        map_bools!("star_job", sj, j, [
            create_initial_model, create_rsp_model,
            load_saved_model, save_model_when_terminate, save_photo_when_terminate,
            save_pulse_data_when_terminate,
            auto_extend_net, set_initial_age, set_initial_model_number,
            set_initial_number_retries, set_initial_dt,
            set_initial_cumulative_energy_error, set_uniform_initial_composition,
            relax_initial_to_xaccrete, set_initial_v_center, set_initial_l_center,
            new_rti_flag, pgstar_flag,
            write_profile_when_terminate, profile_starting_model,
            set_omega, set_initial_omega,
            set_omega_div_omega_crit, set_initial_omega_div_omega_crit,
            set_surface_rotation_v, set_initial_surface_rotation_v,
            astero_just_call_my_extras_check_model,
        ]);
        map_f64s!("star_job", sj, j, [
            pre_ms_t_c, pre_ms_guess_rho_c, pre_ms_d_log10_p,
            radius_in_cm_for_create_initial_model, mass_in_gm_for_create_initial_model,
            initial_age, years_for_initial_dt, seconds_for_initial_dt,
            new_cumulative_energy_error, initial_h1, initial_h2, initial_he3, initial_he4,
            new_mass, new_z, new_y, core_avg_rho, core_avg_eps, new_r_center, new_l_center,
            new_omega, new_omega_div_omega_crit, new_surface_rotation_v,
            remove_initial_center_by_temperature,
        ]);
        map_strs!("star_job", sj, j, [
            save_model_filename, save_photo_filename, save_pulse_data_filename,
            h_he_net, co_net, adv_net,
            history_columns_file, profile_columns_file, filename_for_profile_when_terminate,
        ]);
        let codes = sj.try_get_indexed_strings(
            "star_job",
            "required_termination_code_string",
        )?;
        if !codes.is_empty() {
            j.required_termination_code_string = codes;
        }
        if let Some(s) = sj.try_get_str("star_job", "load_model_filename")? {
            j.saved_model_name = s.to_string();
        }
        sj.try_isize_into("star_job", &mut j.save_model_number, "save_model_number")?;
        if let Some(b) = sj.try_any_bool("star_job", &["change_net", "change_initial_net"])? {
            j.change_net = b;
        }
        if let Some(s) = sj.try_get_str("star_job", "new_net_name")? {
            j.new_net_name = s.to_string();
            if j.change_net {
                j.net_name = s.to_string();
            }
        }
        if let Some(b) = sj.try_any_bool("star_job", &["change_v_flag", "change_initial_v_flag"])? {
            j.change_v_flag = b;
        }
        if let Some(b) = sj.try_get_bool("star_job", "new_v_flag")? {
            j.new_v_flag = b;
            if j.change_v_flag {
                self.controls.v_flag = b;
            }
        }
        if let Some(b) = sj.try_any_bool("star_job", &["change_u_flag", "change_initial_u_flag"])? {
            j.change_u_flag = b;
        }
        if let Some(b) = sj.try_get_bool("star_job", "new_u_flag")? {
            j.new_u_flag = b;
            if j.change_u_flag {
                self.controls.u_flag = b;
            }
        }
        if let Some(b) = sj.try_any_bool("star_job", &["change_rotation_flag", "change_initial_rotation_flag"])? {
            j.change_rotation_flag = b;
        }
        if let Some(b) = sj.try_get_bool("star_job", "new_rotation_flag")? {
            j.new_rotation_flag = b;
            if j.change_rotation_flag {
                self.controls.rotation_flag = b;
            }
        }
        sj.try_usize_into("star_job", &mut j.initial_model_number, "initial_model_number", 0)?;
        sj.try_usize_into("star_job", &mut j.initial_number_retries, "initial_number_retries", 0)?;
        if let Some(v) = sj.try_get_i64("star_job", "initial_zfracs")? {
            if !(1..=10).contains(&v) {
                return Err(StarError::InvalidInput(format!(
                    "unsupported '&star_job::initial_zfracs' = {v} (expected 1..=10)"
                )));
            }
            j.initial_zfracs = v as i32;
        }
        if let Some(b) = sj.try_any_bool("star_job", &[
            "relax_mass", "relax_initial_mass", "relax_mass_scale", "relax_initial_mass_scale",
        ])? {
            j.relax_mass = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["relax_z", "relax_initial_z", "change_z", "change_initial_z"])? {
            j.relax_z = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["relax_y", "relax_initial_y", "change_y", "change_initial_y"])? {
            j.relax_y = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &[
            "set_tau_factor", "set_initial_tau_factor", "relax_tau_factor",
            "relax_initial_tau_factor",
        ])? {
            j.set_tau_factor = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["relax_tau_factor", "relax_initial_tau_factor"])? {
            j.relax_tau_factor = b;
        }
        if let Some(v) = sj.try_first_f64(
            "star_job",
            &["set_to_this_tau_factor", "relax_to_this_tau_factor"],
        )? {
            j.set_to_this_tau_factor = v;
        }
        if let Some(b) = sj.try_any_bool("star_job", &[
            "relax_core", "relax_initial_core", "relax_m_center", "relax_initial_m_center",
        ])? {
            j.relax_m_center = b;
        }
        let core_mass = match sj.try_get_f64("star_job", "new_core_mass")? {
            Some(v) => Some(v),
            None => sj.try_get_f64("star_job", "new_mass")?.filter(|_| j.relax_m_center),
        };
        if let Some(v) = core_mass {
            j.new_core_mass = v;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["relax_r_center", "relax_initial_r_center"])? {
            j.relax_r_center = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["relax_l_center", "relax_initial_l_center"])? {
            j.relax_l_center = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["relax_irradiation", "relax_initial_irradiation"])? {
            j.relax_irradiation = b;
        }
        sj.try_f64_into("star_job", &mut j.new_irradiation_flux, "relax_to_this_irrad_flux")?;
        sj.try_f64_into("star_job", &mut j.column_depth_for_irradiation, "irrad_col_depth")?;
        if let Some(b) = sj.try_any_bool("star_job", &[
            "relax_initial_mass_to_remove_h_env", "remove_surface_by_relax_to_star_cut",
        ])? {
            j.relax_initial_mass_to_remove_h_env = b;
        }
        if let Some(b) = sj.try_any_bool("star_job", &["change_initial_rti_flag", "change_rti_flag"])? {
            j.change_initial_rti_flag = b;
        }
        if j.change_initial_rti_flag {
            self.controls.rti_flag = j.new_rti_flag;
        }
        if let Some(v) = sj.try_get_i64("star_job", "steps_to_take_before_terminate")? {
            if v < -1 {
                return Err(StarError::InvalidInput(format!(
                    "'&star_job::steps_to_take_before_terminate' must be >= -1, got {v}"
                )));
            }
            j.steps_to_take_before_terminate = v as isize;
        }
        Ok(())
    }

    fn populate_eos_kap_with_options(
        &mut self,
        allow_unsupported_controls: bool,
    ) -> Result<(), StarError> {
        self.populate_eos_and_kap_configs(allow_unsupported_controls)
    }

    #[rustfmt::skip]
    fn populate_controls_physics(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        let c = &mut self.controls;
        map_f64s!("controls", ctrl, c, [
            initial_mass, initial_z, initial_y, min_timestep_limit, mixing_length_alpha,
            alpha_semiconvection, thermohaline_coeff, min_d_mix,
            opacity_factor, non_nuc_neu_factor,
            max_logt_for_net, max_abar_for_burning,
            eps_nuc_factor, dxdt_nuc_factor,
            mesh_delta_coeff, varcontrol_target, mass_change, accretion_h1, accretion_he4,
            super_eddington_scaling_factor, hot_wind_full_on_t, cool_wind_full_on_t,
            reimers_scaling_factor, blocker_scaling_factor, vink_scaling_factor,
            dutch_scaling_factor, irradiation_flux, column_depth_for_irradiation,
            rsp_mass, rsp_teff, rsp_l, rsp_x, rsp_z,
            l_theta_for_velocity_time_centering, p_theta_for_velocity_time_centering,
            max_logt_for_include_p_and_l_in_velocity_time_centering,
            op_split_burn_min_t, op_split_burn_eps, op_split_burn_odescal,
            op_split_burn_eps_nuc_infall_limit, op_split_burn_min_t_for_variable_t_solver,
            sig_term_limit, fe56ec_fake_factor, min_t_for_fe56ec_fake_factor,
            max_conv_vel_div_csound, henyey_mlt_y_param, henyey_mlt_nu_param,
            tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt, tdc_alpha_c, tdc_alpha_s, tdc_alpha_m,
            alpha_rti_diffusion_factor, dedt_rti_diffusion_factor,
            dudt_rti_diffusion_factor, dlnddt_rti_diffusion_factor,
            composition_rti_diffusion_factor, alpha_rti_src_min_v_div_cs,
            alpha_rti_src_min_q, alpha_rti_src_max_q,
            rti_a, rti_b, rti_c, rti_d, rti_max_alpha,
            drag_coefficient, min_q_for_drag, mlt_pturb_factor, pvsc_cq,
            eps_grav_factor,
            velocity_q_upper_bound, velocity_tau_lower_bound, velocity_logt_lower_bound,
            mix_factor, pextra_factor, min_kap_for_dprad_dm_eqn,
            fixed_psurf, fixed_vsurf,
        ]);
        ctrl.try_usize_into("controls", &mut c.burn_steps_hard_limit, "burn_steps_hard_limit", 0)?;
        map_strs!("controls", ctrl, c, [
            zams_filename, cool_wind_rgb_scheme, cool_wind_agb_scheme, phase_separation_option,
        ]);
        map_bools!("controls", ctrl, c, [
            use_ledoux_criterion, calculate_brunt_b, calculate_brunt_n2,
            op_split_burn, accrete_same_as_surface,
            accrete_given_mass_fractions, do_element_diffusion, do_conv_premix,
            do_phase_separation, do_phase_separation_heating, use_other_cgrav,
            use_other_alpha_mlt, use_other_wind, use_other_adjust_mdot, use_other_torque,
            use_other_torque_implicit,
            use_other_energy, use_other_eps_grav, use_other_neu, use_other_kap,
            use_other_surface_pt, use_other_momentum_implicit,
            include_l_in_velocity_time_centering,
            include_p_in_velocity_time_centering,
            use_p_d_1_div_rho_form_of_work_when_time_centering_velocity,
            nonlocal_nico_decay_heat, do_premix_heating,
            include_mlt_corr_to_tdc, tdc_include_eturb_in_energy_equation,
            use_drag_energy, use_pvsc_art_visc, use_time_centered_eps_grav,
            include_composition_in_eps_grav, use_gravity_rotation_correction,
            include_rotation_in_total_energy, absolute_cumulative_energy_err,
            use_mass_corrections, use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn,
            use_dprad_dm_form_of_t_gradient_eqn, use_flux_limiting_with_dprad_dm_form,
            use_rsp_l_eqn_outer_bc, use_momentum_outer_bc, use_zero_pgas_outer_bc,
            use_compression_outer_bc, use_fixed_psurf_outer_bc, use_fixed_vsurf_outer_bc,
            okay_to_remesh, remove_mixing_glitches,
            redo_conv_for_dr_lt_mixing_length,
        ]);
        if let Some(v) = ctrl.try_get_f64("controls", "pvsc_zsh")? {
            c.pvsc_zshock = v;
        }
        ctrl.try_f64_into("controls", &mut c.gamma_ln_s_eps_grav_full_on, "gamma_lns_eps_grav_full_on")?;
        ctrl.try_f64_into("controls", &mut c.gamma_ln_s_eps_grav_full_off, "gamma_lns_eps_grav_full_off")?;
        ctrl.try_isize_into("controls", &mut c.steps_before_use_tdc, "steps_before_use_tdc")?;
        ctrl.try_isize_into(
            "controls",
            &mut c.steps_before_use_velocity_time_centering,
            "steps_before_use_velocity_time_centering",
        )?;
        if c.steps_before_use_velocity_time_centering == 0 {
            c.using_velocity_time_centering = true;
        }
        ctrl.try_bool_into("controls", &mut self.job.use_other_build_initial_model, "use_other_build_initial_model")?;
        c.accretion_species_id = ctrl.try_get_indexed_strings("controls", "accretion_species_id")?;
        c.accretion_species_xa = ctrl.try_get_indexed_f64s("controls", "accretion_species_xa")?;

        if ctrl.try_get_str("controls", "cool_wind_rgb_scheme")?.is_some()
            && crate::winds::WindScheme::parse_name(&c.cool_wind_rgb_scheme).is_none()
        {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&controls::cool_wind_rgb_scheme' option '{}'",
                c.cool_wind_rgb_scheme
            )));
        }
        if ctrl.try_get_str("controls", "cool_wind_agb_scheme")?.is_some()
            && crate::winds::WindScheme::parse_name(&c.cool_wind_agb_scheme).is_none()
        {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&controls::cool_wind_agb_scheme' option '{}'",
                c.cool_wind_agb_scheme
            )));
        }
        if ctrl.try_get_str("controls", "phase_separation_option")?.is_some()
            && !matches!(
                c.phase_separation_option.trim().to_ascii_lowercase().as_str(),
                "co" | "one"
            )
        {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&controls::phase_separation_option' option '{}'",
                c.phase_separation_option
            )));
        }

        if let Some(s) = ctrl.try_get_str("controls", "mlt_option")? {
            c.mlt_option = match s.trim().to_ascii_lowercase().as_str() {
                "cox" => MltOption::Cox,
                "henyey" => MltOption::Henyey,
                "ml1" => MltOption::Ml1,
                "ml2" => MltOption::Ml2,
                "mihalas" => MltOption::Mihalas,
                "tdc" => MltOption::Tdc,
                "none" => {
                    c.mixing_length_alpha = 0.0;
                    MltOption::Cox
                }
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::mlt_option' option '{other}'"
                    )));
                }
            };
        }
        if let Some(s) = ctrl.try_get_str("controls", "atm_option")? {
            c.atm_option = match s.trim().to_ascii_lowercase().as_str() {
                "t_tau" => AtmOption::TTau,
                "table" => AtmOption::Table,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized/unsupported '&controls::atm_option' option '{other}' (`star/private/atm_support.f90:900-930`)"
                    )));
                }
            };
        }
        if let Some(s) = ctrl.try_get_str("controls", "atm_table")? {
            c.atm_table = s.trim_matches(|ch: char| ch.is_whitespace() || ch == '\'' || ch == '"')
                .to_ascii_lowercase();
            if (c.atm_option == AtmOption::Table || !c.atm_table.is_empty())
                && atm_table_id(&c.atm_table).is_none()
            {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::atm_table' option '{}'",
                    c.atm_table
                )));
            }
        }
        if let Some(s) = ctrl.try_get_str("controls", "atm_t_tau_relation")? {
            c.atm_t_tau_relation = match s.trim().to_ascii_lowercase().as_str() {
                "eddington" => AtmTTauRelation::Eddington,
                "solar_hopf" => AtmTTauRelation::SolarHopf,
                "krishna_swamy" => AtmTTauRelation::KrishnaSwamy,
                "trampedach_solar" => AtmTTauRelation::TrampedachSolar,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::atm_t_tau_relation' option '{other}'"
                    )));
                }
            };
        }
        if let Some(s) = ctrl.try_get_str("controls", "atm_t_tau_opacity")? {
            c.atm_t_tau_opacity = match s.trim().to_ascii_lowercase().as_str() {
                "fixed" => AtmTTauOpacity::Fixed,
                "iterated" => AtmTTauOpacity::Iterated,
                "varying" => AtmTTauOpacity::Varying,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::atm_t_tau_opacity' option '{other}' (`star/private/atm_support.f90:336-340`)"
                    )));
                }
            };
        }
        ctrl.try_f64_into("controls", &mut c.atm_t_tau_errtol, "atm_t_tau_errtol")?;
        ctrl.try_usize_into("controls", &mut c.atm_t_tau_max_iters, "atm_t_tau_max_iters", 0)?;
        ctrl.try_usize_into("controls", &mut c.atm_t_tau_max_steps, "atm_t_tau_max_steps", 0)?;
        if let Some(s) = ctrl.try_get_str("controls", "screening_mode")? {
            c.screening_mode = match s.trim().to_ascii_lowercase().as_str() {
                "no_screening" | "" => ScreeningMode::NoScreening,
                "salpeter" => ScreeningMode::Salpeter,
                "chugunov" => ScreeningMode::Chugunov,
                "extended" => ScreeningMode::Extended,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::screening_mode' option '{other}'"
                    )));
                }
            };
        }
        if let Some(s) = ctrl.try_get_str("controls", "energy_eqn_option")? {
            c.energy_eqn_option = match s.trim().to_ascii_lowercase().as_str() {
                "dedt" => EnergyEqnOption::Dedt,
                "eps_grav" => EnergyEqnOption::EpsGrav,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::energy_eqn_option' option '{other}'"
                    )));
                }
            };
        }
        if let Some(s) = ctrl.try_get_str("controls", "semiconvection_option")? {
            c.semiconvection_option = match s.trim().to_ascii_lowercase().as_str() {
                "langer_85 mixing; gradt = gradr" => {
                    SemiconvectionOption::Langer85GradTIsGradr
                }
                "langer_85" => SemiconvectionOption::Langer85,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::semiconvection_option' option '{other}'"
                    )));
                }
            };
        }
        if let Some(s) = ctrl.try_get_str("controls", "thermohaline_option")? {
            c.thermohaline_option = match s.trim().to_ascii_lowercase().as_str() {
                "kippenhahn" => ThermohalineOption::Kippenhahn,
                "traxler_garaud_stellmach_11" => {
                    ThermohalineOption::TraxlerGaraudStellmach11
                }
                "brown_garaud_stellmach_13" => {
                    ThermohalineOption::BrownGaraudStellmach13
                }
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::thermohaline_option' option '{other}'"
                    )));
                }
            };
        }
        ctrl.try_isize_into("controls", &mut c.rsp_max_num_periods, "rsp_max_num_periods")?;
        if let Some(v) = ctrl.try_get_i64("controls", "rsp_target_steps_per_cycle")? {
            if v <= 0 {
                return Err(StarError::InvalidInput(format!(
                    "'&controls::rsp_target_steps_per_cycle' must be > 0, got {v}"
                )));
            }
            c.rsp_target_steps_per_cycle = v as usize;
        }
        Ok(())
    }

    fn populate_controls_output(&mut self) -> Result<(), StarError> {
        let Some(ctrl) = self.sections.get("controls") else {
            return Ok(());
        };
        let o = &mut self.controls.output;
        map_strs!(
            "controls",
            ctrl,
            o,
            [
                log_directory,
                star_history_name,
                star_history_header_name,
                profiles_index_name,
                profile_data_prefix,
                profile_data_suffix,
                profile_data_header_suffix,
            ]
        );
        map_bools!(
            "controls",
            ctrl,
            o,
            [
                do_history_file,
                write_profiles_flag,
                profile_header_include_sys_details,
                get_delta_nu_from_scaled_solar,
                report_max_infall_inside_fe_core,
            ]
        );
        map_f64s!(
            "controls",
            ctrl,
            o,
            [
                mixing_d_limit_for_log,
                burn_min1,
                burn_min2,
                center_avg_value_dq,
                surface_avg_abundance_dq,
                surf_avg_tau_min,
                surf_avg_tau,
                nu_max_sun,
                delta_nu_sun,
                astero_teff_sun,
                max_conv_vel_div_csound_maxq,
                gamma1_limit_max_q,
                gamma1_limit_max_v_div_vesc,
                max_x_for_conv_timescale,
                min_x_for_conv_timescale,
                max_q_for_conv_timescale,
                min_q_for_conv_timescale,
                min_q_for_dt_div_min_dr_div_cs_limit,
                max_q_for_dt_div_min_dr_div_cs_limit,
                min_abs_u_div_cs_for_dt_div_min_dr_div_cs_limit,
                min_abs_du_div_cs_for_dt_div_min_dr_div_cs_limit,
            ]
        );
        ctrl.try_usize_into(
            "controls",
            &mut o.min_k_for_dt_div_min_dr_div_cs_limit,
            "min_k_for_dt_div_min_dr_div_cs_limit",
            1,
        )?;
        ctrl.try_usize_into(
            "controls",
            &mut o.history_interval,
            "history_interval",
            0,
        )?;
        ctrl.try_usize_into(
            "controls",
            &mut o.profile_interval,
            "profile_interval",
            0,
        )?;
        ctrl.try_usize_into(
            "controls",
            &mut o.priority_profile_interval,
            "priority_profile_interval",
            0,
        )?;
        ctrl.try_isize_into(
            "controls",
            &mut o.max_num_profile_zones,
            "max_num_profile_zones",
        )?;
        ctrl.try_isize_into(
            "controls",
            &mut o.max_num_profile_models,
            "max_num_profile_models",
        )?;
        ctrl.try_isize_into("controls", &mut o.profile_model, "profile_model")?;
        ctrl.try_str_into(
            "controls",
            &mut self.job.pulse_data_format,
            "pulse_data_format",
        )?;
        if ctrl.try_get_str("controls", "pulse_data_format")?.is_some() {
            match self
                .job
                .pulse_data_format
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "" | "gyre" | "fgong" | "osc" | "gsm" | "saio" | "cafein"
                | "gr1d" => {}
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "unrecognized '&controls::pulse_data_format' option '{other}'"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Maps an upstream `atm_table` string to the corresponding `flexmesa_atm` table ID.
pub(crate) fn atm_table_id(name: &str) -> Option<i32> {
    match name.trim().to_ascii_lowercase().as_str() {
        "photosphere" => Some(flexmesa_atm::ATM_TABLE_PHOTOSPHERE),
        "tau_100" => Some(flexmesa_atm::ATM_TABLE_TAU_100),
        "tau_10" => Some(flexmesa_atm::ATM_TABLE_TAU_10),
        "tau_1" => Some(flexmesa_atm::ATM_TABLE_TAU_1),
        "tau_1m1" => Some(flexmesa_atm::ATM_TABLE_TAU_1M1),
        "wd_tau_25" => Some(flexmesa_atm::ATM_TABLE_WD_TAU_25),
        "db_wd_tau_25" => Some(flexmesa_atm::ATM_TABLE_DB_WD_TAU_25),
        _ => None,
    }
}
