// SPDX-License-Identifier: LGPL-3.0-or-later
//! Job-level initialization, relaxation, and output controls
//! (`StarJobControls`, corresponding to `star_data/private/star_job_controls.inc`).

/// Job-level initialization and model-building controls (`&star_job` namelist
/// in `star_data/private/star_job_controls.inc`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct StarJobControls {
    /// Whether to construct an initial $n = 1.5$ pre-MS polytrope model.
    pub create_pre_ms_model: bool,
    /// Central temperature $T_c$ ($\mathrm{K}$) for `create_pre_ms_model`.
    pub pre_ms_t_c: f64,
    /// Optional initial central density guess $\rho_c$ ($\mathrm{g\,cm^{-3}}$;
    /// $\le 0$ uses the Ushomirsky et al. 1998 analytic $n=1.5$ guess).
    pub pre_ms_guess_rho_c: f64,
    /// Step size $\Delta \log_{10} P_{\mathrm{gas}}$ for outward pre-MS
    /// integration (`0.004` default, `0.02` for coarse grids).
    pub pre_ms_d_log10_p: f64,
    /// Surface $\log_{10} P$ cutoff for pre-MS integration (`3.5` default).
    pub pre_ms_log_p_surf_limit: f64,
    /// Surface $\log_{10} T$ cutoff for pre-MS integration (`3.7` default).
    pub pre_ms_log_t_surf_limit: f64,
    /// Logarithmic luminosity boost factor for pre-MS contraction (`0.0` default).
    pub pre_ms_log_lum_factor: f64,
    /// Number of relaxation steps during pre-MS startup (`100` in MESA).
    pub pre_ms_relax_num_steps: usize,
    /// Whether to construct an isentropic initial planet/WD model (`create_initial_model`).
    pub create_initial_model: bool,
    /// Radius in cm for `create_initial_model`.
    pub radius_in_cm_for_create_initial_model: f64,
    /// Mass in grams for `create_initial_model`.
    pub mass_in_gm_for_create_initial_model: f64,
    /// Whether to construct a static radial stellar pulsation envelope (`create_RSP_model`).
    pub create_rsp_model: bool,
    /// Whether to load a saved `.mod` file on startup.
    pub load_saved_model: bool,
    /// Path to saved `.mod` file when `load_saved_model` is true (`saved_model_name` / `load_model_filename`).
    pub saved_model_name: String,
    /// Whether to save a `.mod` file when evolution terminates (`save_model_when_terminate`).
    pub save_model_when_terminate: bool,
    /// Output `.mod` filename on termination (`save_model_filename`).
    pub save_model_filename: String,
    /// Optional model number at which to save a `.mod` file (`save_model_number`).
    pub save_model_number: Option<usize>,
    /// Whether to save a `.photo` restart file on termination (`save_photo_when_terminate`).
    pub save_photo_when_terminate: bool,
    /// Output `.photo` filename on termination (`save_photo_filename`).
    pub save_photo_filename: String,
    /// Whether to export pulsation data on termination (`save_pulse_data_when_terminate`).
    pub save_pulse_data_when_terminate: bool,
    /// Output pulsation filename on termination (`save_pulse_data_filename`).
    pub save_pulse_data_filename: String,
    /// Pulsation export format (`"GYRE"`, `"FGONG"`, `"OSC"`).
    pub pulse_data_format: String,
    /// Whether to export `STELLA` hydro/abundance data on termination (`save_stella_data_when_terminate`).
    pub save_stella_data_when_terminate: bool,
    /// Output `STELLA` filename prefix (`save_stella_data_filename`).
    pub save_stella_data_filename: String,
    /// Nuclear reaction network name (`"basic.net"` default).
    pub net_name: String,
    /// Whether to switch nuclear reaction networks (`change_net` or `change_initial_net`).
    pub change_net: bool,
    /// Target nuclear reaction network name (`new_net_name`).
    pub new_net_name: String,
    /// Whether to automatically extend the reaction network with evolutionary stage (`auto_extend_net`).
    pub auto_extend_net: bool,
    /// H/He-burning network for `auto_extend_net` (`"basic.net"` default).
    pub h_he_net: String,
    /// C/O-burning network for `auto_extend_net` (`"co_burn.net"` default).
    pub co_net: String,
    /// Advanced burning network for `auto_extend_net` (`"approx21.net"` default).
    pub adv_net: String,
    /// Whether to change `v_flag` on startup (`change_v_flag` / `change_initial_v_flag`).
    pub change_v_flag: bool,
    /// Target `v_flag` when `change_v_flag` is true (`new_v_flag`).
    pub new_v_flag: bool,
    /// Whether to change `u_flag` on startup (`change_u_flag` / `change_initial_u_flag`).
    pub change_u_flag: bool,
    /// Target `u_flag` when `change_u_flag` is true (`new_u_flag`).
    pub new_u_flag: bool,
    /// Whether to change `rotation_flag` on startup (`change_rotation_flag` / `change_initial_rotation_flag`).
    pub change_rotation_flag: bool,
    /// Target `rotation_flag` when `change_rotation_flag` is true (`new_rotation_flag`).
    pub new_rotation_flag: bool,
    /// Optional uniform angular velocity $\Omega$ ($\mathrm{rad\,s^{-1}}$) on startup (`new_omega`).
    pub new_omega: Option<f64>,
    /// Optional uniform $\Omega / \Omega_{\mathrm{crit}}$ on startup (`new_omega_div_omega_crit`).
    pub new_omega_div_omega_crit: Option<f64>,
    /// Optional surface equatorial velocity ($\mathrm{km\,s^{-1}}$) on startup (`new_surface_rotation_v`).
    pub new_surface_rotation_v: Option<f64>,
    /// Whether to set initial stellar age (`set_initial_age`).
    pub set_initial_age: bool,
    /// Initial stellar age in years (`initial_age`).
    pub initial_age: f64,
    /// Whether to set initial model number (`set_initial_model_number`).
    pub set_initial_model_number: bool,
    /// Initial model number (`initial_model_number`).
    pub initial_model_number: usize,
    /// Whether to set initial timestep (`set_initial_dt`).
    pub set_initial_dt: bool,
    /// Initial timestep in years (`years_for_initial_dt`).
    pub years_for_initial_dt: f64,
    /// Initial timestep in seconds (`seconds_for_initial_dt`).
    pub seconds_for_initial_dt: f64,
    /// Whether to impose uniform initial composition (`set_uniform_initial_composition`).
    pub set_uniform_initial_composition: bool,
    /// Uniform initial $^1\mathrm{H}$ mass fraction (`initial_h1`).
    pub initial_h1: f64,
    /// Uniform initial $^2\mathrm{H}$ mass fraction (`initial_h2`).
    pub initial_h2: f64,
    /// Uniform initial $^3\mathrm{He}$ mass fraction (`initial_he3`).
    pub initial_he3: f64,
    /// Uniform initial $^4\mathrm{He}$ mass fraction (`initial_he4`).
    pub initial_he4: f64,
    /// Solar heavy-element mixture selector (`initial_zfracs`, `3` = GS98 default).
    pub initial_zfracs: i32,
    /// Whether to relax stellar mass on startup (`relax_mass` / `relax_initial_mass` / `relax_mass_scale`).
    pub relax_mass: bool,
    /// Target stellar mass in $M_\odot$ for `relax_mass` (`new_mass`).
    pub new_mass: f64,
    /// Whether to relax metal mass fraction $Z$ (`relax_z` / `relax_initial_z`).
    pub relax_z: bool,
    /// Target metal mass fraction $Z$ for `relax_z` (`new_z`).
    pub new_z: f64,
    /// Whether to relax helium mass fraction $Y$ (`relax_y` / `relax_initial_y`).
    pub relax_y: bool,
    /// Target helium mass fraction $Y$ for `relax_y` (`new_y`).
    pub new_y: f64,
    /// Whether to set or relax `tau_factor` (`set_tau_factor` / `relax_to_this_tau_factor`).
    pub set_tau_factor: bool,
    /// Target `tau_factor` (`set_to_this_tau_factor` / `relax_to_this_tau_factor`).
    pub set_to_this_tau_factor: f64,
    /// Whether to relax inner core boundary (`relax_core` / `relax_M_center`).
    pub relax_m_center: bool,
    /// Target inner core mass in $M_\odot$ (`new_mass` / `new_core_mass`).
    pub new_core_mass: f64,
    /// Whether to relax inner boundary radius (`relax_R_center`).
    pub relax_r_center: bool,
    /// Target inner boundary radius in cm (`new_R_center`).
    pub new_r_center: f64,
    /// Whether to relax inner boundary luminosity (`relax_L_center`).
    pub relax_l_center: bool,
    /// Target inner boundary luminosity in $\mathrm{erg\,s^{-1}}$ (`new_L_center`).
    pub new_l_center: f64,
    /// Expected termination code string for `test_suite` verification (`required_termination_code_string`).
    pub required_termination_code_string: String,
    /// Whether `pgstar` live plotting was requested in the inlist (accepted as no-op).
    pub pgstar_flag: bool,
}

impl Default for StarJobControls {
    fn default() -> Self {
        Self {
            create_pre_ms_model: false,
            pre_ms_t_c: 9.0e5,
            pre_ms_guess_rho_c: 0.0,
            pre_ms_d_log10_p: 0.004,
            pre_ms_log_p_surf_limit: 3.5,
            pre_ms_log_t_surf_limit: 3.7,
            pre_ms_log_lum_factor: 0.0,
            pre_ms_relax_num_steps: 20,
            create_initial_model: false,
            radius_in_cm_for_create_initial_model: 0.0,
            mass_in_gm_for_create_initial_model: 0.0,
            create_rsp_model: false,
            load_saved_model: false,
            saved_model_name: String::new(),
            save_model_when_terminate: false,
            save_model_filename: String::new(),
            save_model_number: None,
            save_photo_when_terminate: false,
            save_photo_filename: String::new(),
            save_pulse_data_when_terminate: false,
            save_pulse_data_filename: String::new(),
            pulse_data_format: String::from("GYRE"),
            save_stella_data_when_terminate: false,
            save_stella_data_filename: String::new(),
            net_name: String::from("basic.net"),
            change_net: false,
            new_net_name: String::new(),
            auto_extend_net: false,
            h_he_net: String::from("basic.net"),
            co_net: String::from("co_burn.net"),
            adv_net: String::from("approx21.net"),
            change_v_flag: false,
            new_v_flag: false,
            change_u_flag: false,
            new_u_flag: false,
            change_rotation_flag: false,
            new_rotation_flag: false,
            new_omega: None,
            new_omega_div_omega_crit: None,
            new_surface_rotation_v: None,
            set_initial_age: false,
            initial_age: 0.0,
            set_initial_model_number: false,
            initial_model_number: 0,
            set_initial_dt: false,
            years_for_initial_dt: 0.0,
            seconds_for_initial_dt: 0.0,
            set_uniform_initial_composition: false,
            initial_h1: -1.0,
            initial_h2: -1.0,
            initial_he3: -1.0,
            initial_he4: -1.0,
            initial_zfracs: 3,
            relax_mass: false,
            new_mass: -1.0,
            relax_z: false,
            new_z: -1.0,
            relax_y: false,
            new_y: -1.0,
            set_tau_factor: false,
            set_to_this_tau_factor: 1.0,
            relax_m_center: false,
            new_core_mass: 0.0,
            relax_r_center: false,
            new_r_center: 0.0,
            relax_l_center: false,
            new_l_center: 0.0,
            required_termination_code_string: String::new(),
            pgstar_flag: false,
        }
    }
}
