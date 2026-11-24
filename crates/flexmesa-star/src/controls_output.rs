// SPDX-License-Identifier: GPL-3.0-or-later
//! Photo, history, and radial profile output controls on [`crate::StarControls::output`]
//! corresponding to `star/defaults/controls.defaults:185-857, 1794, 3489, 4883-4900, 7844-7877`.

/// Photo, history, radial profile, and report-averaging controls (`&controls` namelist
/// in `star_data/private/star_controls.inc` and `star/defaults/controls.defaults`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputControls {
    /// Directory for binary `.photo` restart checkpoints (`photo_directory`, `'photos'` default, `star/defaults/controls.defaults:192`).
    pub photo_directory: String,
    /// Step interval for saving periodic restart photos when `mod(model_number, photo_interval) == 0` (`photo_interval`, `50` default, `<= 0` disables, `star/defaults/controls.defaults:203`, `star/private/evolve.f90:2342-2343`).
    pub photo_interval: isize,
    /// Number of trailing `model_number` digits in periodic restart photo names (`photo_digits`, `3` default, `star/defaults/controls.defaults:213`, `star/private/star_utils.f90:329-351`).
    pub photo_digits: usize,
    /// Directory for `history.data`, `profiles.index`, and `profileN.data` (`log_directory`, `'LOGS'` default, `controls.defaults:223`).
    pub log_directory: String,
    /// Enable writing `star_history_name` (`do_history_file`, `true` default, `controls.defaults:233`).
    pub do_history_file: bool,
    /// Step interval for appending to `history.data` (`history_interval`, `5` default, `controls.defaults:243`).
    pub history_interval: usize,
    /// History filename inside `log_directory` (`star_history_name`, `'history.data'` default, `controls.defaults:253`).
    pub star_history_name: String,
    /// Runtime flag signaling the history writer to re-open `star_history_name` with a fresh header (`need_to_set_history_names_etc`, `star_data.inc`, mutated by `rsp.f90:534` and cleared by `history.f90:499`).
    #[serde(default)]
    pub need_to_set_history_names_etc: bool,
    /// Optional separate filename for the history header (`star_history_header_name`, `''` default, `controls.defaults:265`).
    pub star_history_header_name: String,
    /// Enable writing radial structure profiles (`write_profiles_flag`, `true` default, `controls.defaults:305`).
    pub write_profiles_flag: bool,
    /// Step interval for saving a profile (`profile_interval`, `50` default, `controls.defaults:315`).
    pub profile_interval: usize,
    /// Step interval for elevated profile retention priority (`priority_profile_interval`, `1000` default, `controls.defaults:326`).
    pub priority_profile_interval: usize,
    /// Profile index filename inside `log_directory` (`profiles_index_name`, `'profiles.index'` default, `controls.defaults:336`).
    pub profiles_index_name: String,
    /// Prefix for numbered profile files (`profile_data_prefix`, `'profile'` default, `controls.defaults:346`).
    pub profile_data_prefix: String,
    /// Suffix for numbered profile files (`profile_data_suffix`, `'.data'` default, `controls.defaults:356`).
    pub profile_data_suffix: String,
    /// Optional separate suffix for profile headers (`profile_data_header_suffix`, `''` default, `controls.defaults:368`).
    pub profile_data_header_suffix: String,
    /// Optional zone subsample limit for profiles (`max_num_profile_zones`, `-1` default = all zones, `controls.defaults:409`).
    pub max_num_profile_zones: isize,
    /// Maximum retained profile files before priority eviction (`max_num_profile_models`, `100` default, `< 0` = unlimited, `controls.defaults:428`).
    pub max_num_profile_models: isize,
    /// Specific model number at which to force a priority-2 profile (`profile_model`, `-1` default, `controls.defaults:438`).
    pub profile_model: isize,
    /// Include compiler/SDK/math_backend strings in profile headers (`profile_header_include_sys_details`, `true` default, `controls.defaults:448`).
    pub profile_header_include_sys_details: bool,
    /// Minimum non-rotational diffusivity $D_{\mathrm{mix}}$ ($\mathrm{cm^2\,s^{-1}}$) retained in `mixing_regions` history columns (`mixing_D_limit_for_log`, `1e4` default, `controls.defaults:520`).
    pub mixing_d_limit_for_log: f64,
    /// Lower $\epsilon_{\mathrm{nuc}}$ contour ($\mathrm{erg\,g^{-1}\,s^{-1}}$) for `epsnuc_M_1..8` (`burn_min1`, `50.0` default, `controls.defaults:769`).
    pub burn_min1: f64,
    /// Upper $\epsilon_{\mathrm{nuc}}$ contour ($\mathrm{erg\,g^{-1}\,s^{-1}}$) for `epsnuc_M_1..8` (`burn_min2`, `1000.0` default, `controls.defaults:781`).
    pub burn_min2: f64,
    /// Central fractional mass window $\Delta q$ for `center_value` and `center_avg_x` (`center_avg_value_dq`, `1e-8` default, `controls.defaults:837`).
    pub center_avg_value_dq: f64,
    /// Surface fractional mass window $\Delta q$ for `surface_avg_x` (`surface_avg_abundance_dq`, `1e-8` default, `controls.defaults:847`).
    pub surface_avg_abundance_dq: f64,
    /// Inner optical-depth bound for `surf_avg_*` quantities (`surf_avg_tau_min`, `1.0` default, `controls.defaults:4890`).
    pub surf_avg_tau_min: f64,
    /// Outer optical-depth bound for `surf_avg_*` and `eval_Ledd` (`surf_avg_tau`, `100.0` default, `controls.defaults:4900`).
    pub surf_avg_tau: f64,
    /// Solar reference $\nu_{\max,\odot}$ in $\mu\mathrm{Hz}$ (`nu_max_sun`, `3078.0` default, `controls.defaults:7875`).
    pub nu_max_sun: f64,
    /// Solar reference $\Delta\nu_\odot$ in $\mu\mathrm{Hz}$ (`delta_nu_sun`, `134.91` default, `controls.defaults:7876`).
    pub delta_nu_sun: f64,
    /// Solar reference $T_{\mathrm{eff},\odot}$ in $\mathrm{K}$ (`astero_Teff_sun`, `5772.0` default, `controls.defaults:7877`).
    pub astero_teff_sun: f64,
    /// Use scaled-solar relation for `delta_nu` instead of acoustic-radius integral (`get_delta_nu_from_scaled_solar`, `false` default, `controls.defaults:7844`).
    pub get_delta_nu_from_scaled_solar: bool,
    /// Maximum mass coordinate $q$ included when reporting `max_conv_vel_div_csound` (`max_conv_vel_div_csound_maxq`, `-1.0` default = skip all zones, `controls.defaults:3489`).
    pub max_conv_vel_div_csound_maxq: f64,
    /// Report peak `-v` inside `fe_core_mass` rather than global `-velocity(k_min)` (`report_max_infall_inside_fe_core`, `true` default, `controls.defaults:1794`).
    pub report_max_infall_inside_fe_core: bool,
    /// Outer mass-coordinate cap for `set_min_gamma1` (`gamma1_limit_max_q`, `0.95` default, `controls.defaults:2017`).
    pub gamma1_limit_max_q: f64,
    /// Escape-velocity fraction cap for `set_min_gamma1` (`gamma1_limit_max_v_div_vesc`, `0.95` default, `controls.defaults:2027`).
    pub gamma1_limit_max_v_div_vesc: f64,
    /// Upper H mass-fraction bound for `set_conv_time_scales` (`max_X_for_conv_timescale`, `1.0` default, `controls.defaults:8838`).
    pub max_x_for_conv_timescale: f64,
    /// Lower H mass-fraction bound for `set_conv_time_scales` (`min_X_for_conv_timescale`, `0.0` default, `controls.defaults:8839`).
    pub min_x_for_conv_timescale: f64,
    /// Upper $q$ bound for `set_conv_time_scales` (`max_q_for_conv_timescale`, `1.0` default, `controls.defaults:8840`).
    pub max_q_for_conv_timescale: f64,
    /// Lower $q$ bound for `set_conv_time_scales` (`min_q_for_conv_timescale`, `0.0` default, `controls.defaults:8841`).
    pub min_q_for_conv_timescale: f64,
    /// Minimum 1-based zone index for `min_dr_div_cs` (`min_k_for_dt_div_min_dr_div_cs_limit`, `20` default, `controls.defaults:10714`).
    pub min_k_for_dt_div_min_dr_div_cs_limit: usize,
    /// Lower $q$ bound for `min_dr_div_cs` (`min_q_for_dt_div_min_dr_div_cs_limit`, `0.005` default, `controls.defaults:10715`).
    pub min_q_for_dt_div_min_dr_div_cs_limit: f64,
    /// Upper $q$ bound for `min_dr_div_cs` (`max_q_for_dt_div_min_dr_div_cs_limit`, `0.995` default, `controls.defaults:10716`).
    pub max_q_for_dt_div_min_dr_div_cs_limit: f64,
    /// Minimum $|u| / c_s$ Mach filter for `min_dr_div_cs` (`min_abs_u_div_cs_for_dt_div_min_dr_div_cs_limit`, `0.8` default, `controls.defaults:10734`).
    pub min_abs_u_div_cs_for_dt_div_min_dr_div_cs_limit: f64,
    /// Minimum $|\Delta u| / c_s$ jump filter for `min_dr_div_cs` (`min_abs_du_div_cs_for_dt_div_min_dr_div_cs_limit`, `0.01` default, `controls.defaults:10735`).
    pub min_abs_du_div_cs_for_dt_div_min_dr_div_cs_limit: f64,
    /// Relative cumulative energy error threshold emitting a warning (`warn_when_large_rel_run_E_err`, `0.1` default, `controls.defaults:11618`).
    pub warn_when_large_rel_run_e_err: f64,
    /// Relative virial theorem error threshold emitting a warning (`warn_when_large_virial_thm_rel_err`, `1d-2` default, `controls.defaults:11650`).
    #[serde(default = "default_warn_when_large_virial_thm_rel_err")]
    pub warn_when_large_virial_thm_rel_err: f64,
    /// Emit terminal diagnostics and abort `get_tau` on internal errors (`report_ierr`, `.false.` default, `controls.defaults:11473`).
    #[serde(default)]
    pub report_ierr: bool,
    /// Emit a terminal warning when non-split-burn rates reach `max_safe_logT_for_rates` (`warn_rates_for_high_temp`, `.true.` default, `controls.defaults:11604`).
    #[serde(default = "default_true")]
    pub warn_rates_for_high_temp: bool,
}

impl Default for OutputControls {
    fn default() -> Self {
        Self {
            photo_directory: String::from("photos"),
            photo_interval: 50,
            photo_digits: 3,
            log_directory: String::from("LOGS"),
            do_history_file: true,
            history_interval: 5,
            star_history_name: String::from("history.data"),
            need_to_set_history_names_etc: false,
            star_history_header_name: String::new(),
            write_profiles_flag: true,
            profile_interval: 50,
            priority_profile_interval: 1000,
            profiles_index_name: String::from("profiles.index"),
            profile_data_prefix: String::from("profile"),
            profile_data_suffix: String::from(".data"),
            profile_data_header_suffix: String::new(),
            max_num_profile_zones: -1,
            max_num_profile_models: 100,
            profile_model: -1,
            profile_header_include_sys_details: true,
            mixing_d_limit_for_log: 1.0e4,
            burn_min1: 50.0,
            burn_min2: 1000.0,
            center_avg_value_dq: 1.0e-8,
            surface_avg_abundance_dq: 1.0e-8,
            surf_avg_tau_min: 1.0,
            surf_avg_tau: 100.0,
            nu_max_sun: 3078.0,
            delta_nu_sun: 134.91,
            astero_teff_sun: 5772.0,
            get_delta_nu_from_scaled_solar: false,
            max_conv_vel_div_csound_maxq: -1.0,
            report_max_infall_inside_fe_core: true,
            gamma1_limit_max_q: 0.95,
            gamma1_limit_max_v_div_vesc: 0.95,
            max_x_for_conv_timescale: 1.0,
            min_x_for_conv_timescale: 0.0,
            max_q_for_conv_timescale: 1.0,
            min_q_for_conv_timescale: 0.0,
            min_k_for_dt_div_min_dr_div_cs_limit: 20,
            min_q_for_dt_div_min_dr_div_cs_limit: 0.005,
            max_q_for_dt_div_min_dr_div_cs_limit: 0.995,
            min_abs_u_div_cs_for_dt_div_min_dr_div_cs_limit: 0.8,
            min_abs_du_div_cs_for_dt_div_min_dr_div_cs_limit: 0.01,
            warn_when_large_rel_run_e_err: 0.1,
            warn_when_large_virial_thm_rel_err: 1.0e-2,
            report_ierr: false,
            warn_rates_for_high_temp: true,
        }
    }
}

const fn default_warn_when_large_virial_thm_rel_err() -> f64 {
    1.0e-2
}

const fn default_true() -> bool {
    true
}

/// Emits a `report_ierr` diagnostic to stderr when `report_ierr` is active
/// (`star/defaults/controls.defaults:11466-11473`).
pub(crate) fn maybe_report_ierr(
    report_ierr: bool,
    err: &crate::controls::StarError,
) {
    if report_ierr {
        eprintln!("report_ierr: {err}");
    }
}

/// Emits a `report_ierr` diagnostic to stderr before returning `res` when `report_ierr` is active
/// (`star/defaults/controls.defaults:11466-11473`).
pub(crate) fn report_result_ierr<T>(
    report_ierr: bool,
    res: Result<T, crate::controls::StarError>,
) -> Result<T, crate::controls::StarError> {
    if let Err(ref err) = res {
        maybe_report_ierr(report_ierr, err);
    }
    res
}
