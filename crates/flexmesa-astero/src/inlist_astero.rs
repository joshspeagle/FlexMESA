// SPDX-License-Identifier: GPL-3.0-or-later
//! Configuration structs and defaults for `&astero_search_controls` namelists
//! and `from_file` parameter tables (`astero/defaults/astero_search.defaults`,
//! `astero/private/astero_run_support.f90`).

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use flexmesa_star::StarError;

pub use crate::inlist_parser::{
    parse_astero_search_controls, parse_astero_search_controls_with_options,
    supported_astero_control_inventory, validate_astero_sections,
    AsteroControlInventory, ASTERO_SEARCH_CONTROLS_SUPPORTED,
    ASTERO_SEARCH_CONTROLS_UNUSED,
};
use crate::ratios_chi2::AsteroChi2Weights;

/// Searchable stellar model parameter specification (`first_*`, `min_*`, `max_*`, `delta_*`).
#[derive(Debug, Clone, Copy)]
pub struct AsteroSearchParam {
    /// Whether this parameter is varied by the optimizer (`vary_*`).
    pub vary: bool,
    /// Initial / fixed parameter value (`first_*`).
    pub first: f64,
    /// Lower search bound (`min_*`).
    pub min: f64,
    /// Upper search bound (`max_*`).
    pub max: f64,
    /// Grid step size (`delta_*`).
    pub delta: f64,
}

impl AsteroSearchParam {
    /// Creates a fixed parameter with `first = min = max = val`.
    #[must_use]
    pub const fn fixed(val: f64) -> Self {
        Self {
            vary: false,
            first: val,
            min: val,
            max: val,
            delta: 0.0,
        }
    }
}

/// Number of `constraint_name(i)` slots read from `&astero_search_controls`
/// (upstream `max_constraints = 100`, `astero/public/astero_def.f90:107`;
/// the test cases use at most 9).
pub const MAX_CONSTRAINTS: usize = 10;

/// Spectroscopic observable constraint (`*_target`, `*_sigma`, `include_*_in_chi2_spectro`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectroConstraint {
    /// Whether this observable contributes to $\chi^2_{\mathrm{spectro}}$.
    pub include: bool,
    /// Target observed value.
    pub target: f64,
    /// $1\sigma$ observational uncertainty.
    pub sigma: f64,
    /// Multiplier for hard rejection limit (`sigmas_coeff_for_*_limit`).
    pub sigmas_coeff: f64,
}

/// Evaluated parameter vector for one stellar track (`astero_support.f90:54-63`).
#[derive(Debug, Clone, Copy)]
pub struct AsteroParamSet {
    /// Initial stellar mass $M / M_\odot$.
    pub initial_mass: f64,
    /// Initial helium mass fraction $Y_0$.
    pub initial_y: f64,
    /// Initial metallicity $[\mathrm{Fe/H}]_0$.
    pub initial_feh: f64,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$.
    pub alpha: f64,
    /// Exponential convective core overshoot parameter $f_{\mathrm{ov}}$.
    pub f_ov: f64,
    /// User parameter 1 (`my_param1`).
    pub my_param1: f64,
    /// User parameter 2 (`my_param2`).
    pub my_param2: f64,
    /// User parameter 3 (`my_param3`).
    pub my_param3: f64,
}

/// Parsed `&astero_search_controls` configuration (`astero/defaults/astero_search.defaults`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct AsteroSearchControls {
    /// Composite $\chi^2$ weights and normalization flags.
    pub weights: AsteroChi2Weights,
    /// Log $\chi^2_{\Delta\nu}$ trace info (`trace_chi2_seismo_delta_nu_info`).
    pub trace_chi2_seismo_delta_nu_info: bool,
    /// Log $\chi^2_{\nu_{\max}}$ trace info (`trace_chi2_seismo_nu_max_info`).
    pub trace_chi2_seismo_nu_max_info: bool,
    /// Log seismic ratio trace info (`trace_chi2_seismo_ratios_info`).
    pub trace_chi2_seismo_ratios_info: bool,
    /// Log per-frequency $\chi^2$ trace info (`trace_chi2_seismo_frequencies_info`).
    pub trace_chi2_seismo_frequencies_info: bool,
    /// Log spectroscopic $\chi^2$ trace info (`trace_chi2_spectro_info`).
    pub trace_chi2_spectro_info: bool,
    /// Log wall-clock timing in oscillation code calls (`trace_time_in_oscillation_code`).
    pub trace_time_in_oscillation_code: bool,
    /// Whether to log limit-rejection diagnostics (`trace_limits`).
    pub trace_limits: bool,
    /// Solar reference $(Z/X)_\odot$.
    pub z_div_x_solar: f64,
    /// Number of observed target modes per spherical degree $l \in \{0, 1, 2, 3\}$.
    pub nl: [usize; 4],
    /// Target mode frequencies ($\mu\mathrm{Hz}$) per degree $l \in \{0, 1, 2, 3\}$.
    pub freq_target: [Vec<f64>; 4],
    /// Target mode $1\sigma$ uncertainties ($\mu\mathrm{Hz}$) per degree $l \in \{0, 1, 2, 3\}$.
    pub freq_sigma: [Vec<f64>; 4],
    /// Observed radial orders $n$ for $l = 0$ modes (`l0_n_obs`).
    pub l0_n_obs: Vec<f64>,
    /// Observed large frequency separation $\Delta\nu$ ($\mu\mathrm{Hz}$).
    pub delta_nu: f64,
    /// Observed $\Delta\nu$ uncertainty $\sigma_{\Delta\nu}$ ($\mu\mathrm{Hz}$).
    pub delta_nu_sigma: f64,
    /// Multiplier for hard $\Delta\nu$ limit (`sigmas_coeff_for_delta_nu_limit`).
    pub sigmas_coeff_for_delta_nu_limit: f64,
    /// Observed frequency of maximum oscillation power $\nu_{\max}$ ($\mu\mathrm{Hz}$).
    pub nu_max: f64,
    /// Observed $\nu_{\max}$ uncertainty $\sigma_{\nu_{\max}}$ ($\mu\mathrm{Hz}$).
    pub nu_max_sigma: f64,
    /// Oscillation solver backend (`"adipls"` or `"gyre"`).
    pub oscillation_code: String,
    /// Lower frequency scan bracket factor (`nu_lower_factor`).
    pub nu_lower_factor: f64,
    /// Upper frequency scan bracket factor (`nu_upper_factor`).
    pub nu_upper_factor: f64,
    /// Per-degree scan grid density multiplier (`iscan_factor(0..3)`).
    pub iscan_factor: [f64; 4],
    /// Path to the `GYRE` namelist input file (`gyre_input_file`).
    pub gyre_input_file: String,
    /// Whether non-adiabatic `GYRE` mode calculations are enabled (`gyre_non_ad`).
    pub gyre_non_ad: bool,
    /// Whether to attach a $T(\tau)$ atmosphere before pulsation export (`add_atmosphere`).
    pub add_atmosphere: bool,
    /// Whether to retain the outermost $x = 1$ surface zone (`keep_surface_point`).
    pub keep_surface_point: bool,
    /// Whether to add an extrapolated $r = 0$ center point (`add_center_point`).
    pub add_center_point: bool,
    /// Whether to parse `redistrb.c.pruned.in` at startup and redistribute the
    /// mesh before ADIPLS (`do_redistribute_mesh`).
    pub do_redistribute_mesh: bool,
    /// Surface-effect frequency correction scheme (`correction_scheme`).
    pub correction_scheme: String,
    /// Kjeldsen et al. (2008) power-law exponent $b$ (`correction_b`).
    pub correction_b: f64,
    /// Fraction of surface correction applied (`correction_factor`).
    pub correction_factor: f64,
    /// Prescribed Kjeldsen $a/r$ parameter (`a_div_r`).
    pub a_div_r: f64,
    /// Solar reference $\nu_{\max,\odot}$ ($\mu\mathrm{Hz}$).
    pub solar_nu_max: f64,
    /// Solar reference $\Delta\nu_\odot$ ($\mu\mathrm{Hz}$).
    pub solar_delta_nu: f64,
    /// Optimization driver selection (`search_type`).
    pub search_type: String,
    /// Whether stellar age is included in $\chi^2_{\mathrm{spectro}}$.
    pub include_age_in_chi2_spectro: bool,
    /// Whether $\chi^2$ is evaluated only at `age_target` (`eval_chi2_at_target_age_only`).
    pub eval_chi2_at_target_age_only: bool,
    /// Target stellar age (yr).
    pub age_target: f64,
    /// Uncertainty on target stellar age (yr).
    pub age_sigma: f64,
    /// Number of reduced-timestep steps before `age_target`.
    pub num_smaller_steps_before_age_target: usize,
    /// Timestep size (yr) for steps approaching `age_target`.
    pub dt_for_smaller_steps_before_age_target: f64,
    /// Minimum stellar age (yr) for $\chi^2$ evaluation (`min_age_for_chi2`).
    pub min_age_for_chi2: Option<f64>,
    /// Maximum stellar age (yr) for $\chi^2$ evaluation (`max_age_for_chi2`).
    pub max_age_for_chi2: Option<f64>,
    /// Minimum stellar age (yr) before seismic checks start (`min_age_limit`).
    pub min_age_limit: f64,
    /// Minimum $L_{\mathrm{nuc}}/L$ ratio before seismic checks start (`lnuc_div_l_limit`).
    pub lnuc_div_l_limit: f64,
    /// Maximum $\chi^2_{\mathrm{spectro}}$ threshold before mode evaluation (`chi2_spectroscopic_limit`).
    pub chi2_spectroscopic_limit: f64,
    /// Maximum $\chi^2_{\Delta\nu}$ threshold before mode evaluation (`chi2_delta_nu_limit`).
    pub chi2_delta_nu_limit: f64,
    /// Maximum radial-mode $\chi^2_{\mathrm{radial}}$ threshold before nonradial modes (`chi2_radial_limit`).
    pub chi2_radial_limit: f64,
    /// Search limit 1 (`chi2_search_limit1`).
    pub chi2_search_limit1: f64,
    /// Search limit 2 (`chi2_search_limit2`).
    pub chi2_search_limit2: f64,
    /// Relative $\chi^2$ increase threshold (`chi2_relative_increase_limit`).
    pub chi2_relative_increase_limit: f64,
    /// Consecutive steps with increasing $\chi^2$ before track termination (`limit_num_chi2_too_big`).
    pub limit_num_chi2_too_big: usize,
    /// Whether initial helium follows $Y = Y_0 + (dY/dZ) Z$ (`y_depends_on_z`).
    pub y_depends_on_z: bool,
    /// Primordial helium mass fraction $Y_0$ (`y0`).
    pub y0: f64,
    /// Helium-to-metal enrichment slope $dY/dZ$ (`dydz`).
    pub dydz: f64,
    /// Initial mass search parameter (`mass`).
    pub mass: AsteroSearchParam,
    /// Initial helium search parameter (`y`).
    pub y: AsteroSearchParam,
    /// Initial $[\mathrm{Fe/H}]$ search parameter (`feh`).
    pub feh_param: AsteroSearchParam,
    /// Mixing-length $\alpha_{\mathrm{MLT}}$ search parameter (`alpha`).
    pub alpha: AsteroSearchParam,
    /// Core overshoot $f_{\mathrm{ov}}$ search parameter (`f_ov`).
    pub f_ov: AsteroSearchParam,
    /// User-defined search parameters (`my_param1..3`).
    pub my_param: [AsteroSearchParam; 3],
    /// Ratio $f_{0,\mathrm{ov}} / f_{\mathrm{ov}}$ (`f0_ov_div_f_ov`).
    pub f0_ov_div_f_ov: f64,
    /// Primordial ${}^3\mathrm{He}/Y$ fraction (`y_frac_he3`).
    pub y_frac_he3: f64,
    /// Maximum Nelder-Mead iterations (`simplex_itermax`).
    pub simplex_itermax: usize,
    /// Maximum Nelder-Mead function evaluations (`simplex_fcn_calls_max`).
    pub simplex_fcn_calls_max: usize,
    /// Nelder-Mead absolute parameter tolerance (`simplex_x_atol`).
    pub simplex_x_atol: f64,
    /// Nelder-Mead relative parameter tolerance (`simplex_x_rtol`).
    pub simplex_x_rtol: f64,
    /// Nelder-Mead $\chi^2$ termination threshold (`simplex_chi2_tol`).
    pub simplex_chi2_tol: f64,
    /// Nelder-Mead centroid weight power (`simplex_centroid_weight_power`).
    pub simplex_centroid_weight_power: f64,
    /// Whether Nelder-Mead clamps trial points to bounds (`simplex_enforce_bounds`).
    pub simplex_enforce_bounds: bool,
    /// Whether Nelder-Mead tests adaptive random points (`simplex_adaptive_random_search`).
    pub simplex_adaptive_random_search: bool,
    /// RNG seed for Nelder-Mead random search (`simplex_seed`).
    pub simplex_seed: i32,
    /// Nelder-Mead reflection coefficient $\alpha$ (`simplex_alpha`).
    pub simplex_alpha: f64,
    /// Nelder-Mead expansion coefficient $\beta$ (`simplex_beta`).
    pub simplex_beta: f64,
    /// Nelder-Mead contraction coefficient $\gamma$ (`simplex_gamma`).
    pub simplex_gamma: f64,
    /// Nelder-Mead shrink coefficient $\delta$ (`simplex_delta`).
    pub simplex_delta: f64,
    /// Restart Nelder-Mead simplex search from `simplex_output_filename` (`restart_simplex_from_file`).
    pub restart_simplex_from_file: bool,
    /// Output filename for `simplex` search (`simplex_output_filename`).
    pub simplex_output_filename: String,
    /// Restart `scan_grid` search from `scan_grid_output_filename` (`restart_scan_grid_from_file`).
    pub restart_scan_grid_from_file: bool,
    /// Initial normalized trust-region radius for `bobyqa` (`bobyqa_rho_beg`).
    pub bobyqa_rho_beg: f64,
    /// Final normalized trust-region radius for `bobyqa` (`bobyqa_rho_end`).
    pub bobyqa_rho_end: f64,
    /// Maximum function evaluations for `bobyqa` (`bobyqa_max_fcn_calls`).
    pub bobyqa_max_fcn_calls: usize,
    /// Output filename for `bobyqa` search (`bobyqa_output_filename`).
    pub bobyqa_output_filename: String,
    /// Initial normalized trust-region radius for `newuoa` (`newuoa_rho_beg`).
    pub newuoa_rho_beg: f64,
    /// Final normalized trust-region radius for `newuoa` (`newuoa_rho_end`).
    pub newuoa_rho_end: f64,
    /// Maximum function evaluations for `newuoa` (`newuoa_max_fcn_calls`).
    pub newuoa_max_fcn_calls: usize,
    /// Output filename for `newuoa` search (`newuoa_output_filename`).
    pub newuoa_output_filename: String,
    /// Output directory for sample and summary files (`astero_results_directory`).
    pub astero_results_directory: String,
    /// Output filename for `scan_grid` search (`scan_grid_output_filename`).
    pub scan_grid_output_filename: String,
    /// Input parameter table filename for `from_file` (`filename_for_parameters`).
    pub filename_for_parameters: String,
    /// Maximum rows to evaluate from `filename_for_parameters` (`max_num_from_file`).
    pub max_num_from_file: isize,
    /// 1-based column for `initial_mass` in `filename_for_parameters`.
    pub file_column_for_mass: isize,
    /// 1-based column for `initial_Y` in `filename_for_parameters`.
    pub file_column_for_y: isize,
    /// 1-based column for `initial_FeH` in `filename_for_parameters`.
    pub file_column_for_feh: isize,
    /// 1-based column for `alpha` in `filename_for_parameters`.
    pub file_column_for_alpha: isize,
    /// 1-based column for `f_ov` in `filename_for_parameters`.
    pub file_column_for_f_ov: isize,
    /// 1-based columns for `my_param1..3` in `filename_for_parameters`.
    pub file_column_for_my_param: [isize; 3],
    /// Output filename for `from_file` search (`from_file_output_filename`).
    pub from_file_output_filename: String,
    /// Maximum timestep (yr) when far from target box (`max_yrs_dt_when_cold`).
    pub max_yrs_dt_when_cold: f64,
    /// Maximum timestep (yr) when approaching target box (`max_yrs_dt_when_warm`).
    pub max_yrs_dt_when_warm: f64,
    /// Maximum timestep (yr) inside spectroscopic/$\Delta\nu$ box (`max_yrs_dt_when_hot`).
    pub max_yrs_dt_when_hot: f64,
    /// Threshold on $\chi^2$ for reduced timesteps (`chi2_limit_for_small_timesteps`).
    pub chi2_limit_for_small_timesteps: f64,
    /// Maximum timestep (yr) when $\chi^2 \le \chi^2_{\mathrm{limit}}$ (`max_yrs_dt_chi2_small_limit`).
    pub max_yrs_dt_chi2_small_limit: f64,
    /// Threshold on $\chi^2$ for "smaller" timesteps (`chi2_limit_for_smaller_timesteps`).
    pub chi2_limit_for_smaller_timesteps: f64,
    /// Maximum timestep (yr) below `chi2_limit_for_smaller_timesteps` (`max_yrs_dt_chi2_smaller_limit`).
    pub max_yrs_dt_chi2_smaller_limit: f64,
    /// Threshold on $\chi^2$ for "smallest" timesteps (`chi2_limit_for_smallest_timesteps`).
    pub chi2_limit_for_smallest_timesteps: f64,
    /// Maximum timestep (yr) below `chi2_limit_for_smallest_timesteps` (`max_yrs_dt_chi2_smallest_limit`).
    pub max_yrs_dt_chi2_smallest_limit: f64,
    /// Fewest samples before age and model-number averages are formed (`min_num_samples_for_avg`).
    pub min_num_samples_for_avg: usize,
    /// Number of best-$\chi^2$ samples averaged (`max_num_samples_for_avg`).
    pub max_num_samples_for_avg: usize,
    /// Stop a track older than the sample average by this many sigma (`avg_age_sigma_limit`).
    pub avg_age_sigma_limit: f64,
    /// Same for the model number (`avg_model_number_sigma_limit`).
    pub avg_model_number_sigma_limit: f64,
    /// Header name for primary surface-correction coefficient column (`surf_coef1_name`).
    pub surf_coef1_name: String,
    /// Header name for secondary surface-correction coefficient column (`surf_coef2_name`).
    pub surf_coef2_name: String,
    /// Save next-best higher-frequency nonradial matches (`save_next_best_at_higher_frequency`).
    pub save_next_best_at_higher_frequency: bool,
    /// Save next-best lower-frequency nonradial matches (`save_next_best_at_lower_frequency`).
    pub save_next_best_at_lower_frequency: bool,
    /// Fortran edit descriptor for floating-point table columns (`astero_results_dbl_format`).
    pub astero_results_dbl_format: String,
    /// Fortran edit descriptor for integer table columns (`astero_results_int_format`).
    pub astero_results_int_format: String,
    /// Fortran edit descriptor for text table columns (`astero_results_txt_format`).
    pub astero_results_txt_format: String,
    /// Write `sample_NNNN.data` summary file for each accepted sample (`write_best_model_data_for_each_sample`).
    pub write_best_model_data_for_each_sample: bool,
    /// Zero-padded width of sample index in per-sample output filenames (`num_digits`).
    pub num_digits: usize,
    /// Filename prefix for per-sample best-model summary files (`sample_results_prefix`).
    pub sample_results_prefix: String,
    /// Filename suffix for per-sample best-model summary files (`sample_results_postfix`).
    pub sample_results_postfix: String,
    /// Zero-padded width of stellar model number in pulse filenames (`model_num_digits`).
    pub model_num_digits: usize,
    /// Export FGONG data for every model passed to oscillation code (`write_fgong_for_each_model`).
    pub write_fgong_for_each_model: bool,
    /// Filename prefix for per-model FGONG files (`fgong_prefix`).
    pub fgong_prefix: String,
    /// Filename suffix for per-model FGONG files (`fgong_postfix`).
    pub fgong_postfix: String,
    /// Export FGONG file for each track's best-fitting model (`write_fgong_for_best_model`).
    pub write_fgong_for_best_model: bool,
    /// Filename for best-model FGONG export (`best_model_fgong_filename`).
    pub best_model_fgong_filename: String,
    /// Export GYRE data for every model passed to `GYRE` (`write_gyre_for_each_model`).
    pub write_gyre_for_each_model: bool,
    /// Filename prefix for per-model GYRE files (`gyre_prefix`).
    pub gyre_prefix: String,
    /// Filename suffix for per-model GYRE files (`gyre_postfix`).
    pub gyre_postfix: String,
    /// Maximum number of points for GYRE output (`max_num_gyre_points`, active only if `> 1`).
    pub max_num_gyre_points: isize,
    /// Export GYRE file for each track's best-fitting model (`write_gyre_for_best_model`).
    pub write_gyre_for_best_model: bool,
    /// Filename for best-model GYRE export (`best_model_gyre_filename`).
    pub best_model_gyre_filename: String,
    /// Write profile file for each track's best-fitting model (`write_profile_for_best_model`).
    pub write_profile_for_best_model: bool,
    /// Filename for best-model profile export (`best_model_profile_filename`).
    pub best_model_profile_filename: String,
    /// Save MESA `.mod` file for each track's best-fitting model (`save_model_for_best_model`).
    pub save_model_for_best_model: bool,
    /// Filename for best-model `.mod` file (`best_model_save_model_filename`).
    pub best_model_save_model_filename: String,
    /// Save `show_best`-format info for the last evolved model (`save_info_for_last_model`).
    pub save_info_for_last_model: bool,
    /// Filename for last-model summary output (`last_model_save_info_filename`).
    pub last_model_save_info_filename: String,
    /// Dump `&astero_search_controls` namelist at startup (`save_controls`).
    pub save_controls: bool,
    /// Output filename for `save_controls` (`save_controls_filename`).
    pub save_controls_filename: String,
    /// Model number at which to dump eigenfunction data (`save_mode_model_number`).
    pub save_mode_model_number: isize,
    /// Output filename for eigenfunction dump (`save_mode_filename`).
    pub save_mode_filename: String,
    /// Spherical harmonic degree $l$ of mode eigenfunction to save (`el_to_save`).
    pub el_to_save: i32,
    /// Radial order $n_{pg}$ of mode eigenfunction to save (`order_to_save`).
    pub order_to_save: i32,
    /// ADIPLS rotational kernel control (`adipls_irotkr`).
    pub adipls_irotkr: i32,
    /// ADIPLS rotational kernel print point count (`adipls_nprtkr`).
    pub adipls_nprtkr: i32,
    /// ADIPLS $\Gamma_1$ kernel control (`adipls_igm1kr`).
    pub adipls_igm1kr: i32,
    /// ADIPLS $\Gamma_1$ kernel print point count (`adipls_npgmkr`).
    pub adipls_npgmkr: i32,
    /// Extra astero search inlist include flags (`read_extra_astero_search_inlist(1..5)`).
    pub read_extra_astero_search_inlist: [bool; 5],
    /// Extra astero search inlist filenames (`extra_astero_search_inlist_name(1..5)`).
    pub extra_astero_search_inlist_name: [String; 5],
    pub(crate) constraint_names: [String; 10],
    pub(crate) indexed_constraints: [SpectroConstraint; 10],
    pub(crate) param_names: [String; 10],
    pub(crate) indexed_params: [AsteroSearchParam; 10],
    pub(crate) indexed_file_cols: [isize; 10],
    pub(crate) has_custom_param: [bool; 3],
}

impl Default for AsteroSearchControls {
    fn default() -> Self {
        Self {
            weights: AsteroChi2Weights::default(),
            trace_chi2_seismo_delta_nu_info: false,
            trace_chi2_seismo_nu_max_info: false,
            trace_chi2_seismo_ratios_info: false,
            trace_chi2_seismo_frequencies_info: false,
            trace_chi2_spectro_info: false,
            trace_time_in_oscillation_code: false,
            trace_limits: false,
            z_div_x_solar: 0.02293,
            nl: [0; 4],
            freq_target: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            freq_sigma: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            l0_n_obs: Vec::new(),
            delta_nu: -1.0,
            delta_nu_sigma: -1.0,
            sigmas_coeff_for_delta_nu_limit: 0.0,
            nu_max: -1.0,
            nu_max_sigma: -1.0,
            oscillation_code: String::from("adipls"),
            nu_lower_factor: f64::from(0.8_f32),
            nu_upper_factor: f64::from(1.2_f32),
            // `astero/defaults/astero_search.defaults:1372-1375`: iscan_factor(0:3) = 15.
            iscan_factor: [15.0; 4],
            gyre_input_file: String::from("gyre.in"),
            gyre_non_ad: false,
            add_atmosphere: false,
            keep_surface_point: false,
            add_center_point: true,
            // `astero/defaults/astero_search.defaults:1363`: do_redistribute_mesh = .true.
            do_redistribute_mesh: true,
            correction_scheme: String::from("kjeldsen"),
            correction_b: 4.90,
            correction_factor: 0.0,
            a_div_r: 0.0,
            solar_nu_max: 3090.0,
            solar_delta_nu: 135.1,
            search_type: String::from("use_first_values"),
            include_age_in_chi2_spectro: false,
            eval_chi2_at_target_age_only: false,
            age_target: 4.5695e9,
            age_sigma: 0.0065e9,
            num_smaller_steps_before_age_target: 50,
            dt_for_smaller_steps_before_age_target: 6.5e5,
            min_age_for_chi2: None,
            max_age_for_chi2: None,
            min_age_limit: 1.0e6,
            lnuc_div_l_limit: f64::from(0.95_f32),
            chi2_spectroscopic_limit: 1000.0,
            chi2_delta_nu_limit: 1000.0,
            chi2_radial_limit: 100.0,
            chi2_search_limit1: 3.0,
            chi2_search_limit2: 4.0,
            chi2_relative_increase_limit: 2.0,
            limit_num_chi2_too_big: 20,
            y_depends_on_z: false,
            y0: 0.248,
            dydz: 1.4,
            mass: AsteroSearchParam::fixed(1.0),
            y: AsteroSearchParam::fixed(0.274),
            feh_param: AsteroSearchParam::fixed(0.0),
            alpha: AsteroSearchParam::fixed(1.90),
            f_ov: AsteroSearchParam::fixed(0.0),
            my_param: [AsteroSearchParam::fixed(0.0); 3],
            f0_ov_div_f_ov: -1.0,
            y_frac_he3: 1.0e-4,
            simplex_itermax: 1000,
            simplex_fcn_calls_max: 10_000,
            simplex_x_atol: 1.0e-10,
            simplex_x_rtol: 1.0e-10,
            simplex_chi2_tol: 1.0e-10,
            simplex_centroid_weight_power: 0.0,
            simplex_enforce_bounds: false,
            simplex_adaptive_random_search: false,
            simplex_seed: 1_074_698_122,
            simplex_alpha: 1.0,
            simplex_beta: 2.0,
            simplex_gamma: 0.5,
            simplex_delta: 0.5,
            restart_simplex_from_file: false,
            simplex_output_filename: String::from("simplex_results.data"),
            restart_scan_grid_from_file: false,
            bobyqa_rho_beg: 0.45,
            bobyqa_rho_end: 1.0e-6,
            bobyqa_max_fcn_calls: 1000,
            bobyqa_output_filename: String::from("bobyqa_results.data"),
            newuoa_rho_beg: 0.45,
            newuoa_rho_end: 1.0e-6,
            newuoa_max_fcn_calls: 1000,
            newuoa_output_filename: String::from("newuoa_results.data"),
            astero_results_directory: String::from("outputs"),
            scan_grid_output_filename: String::from("scan_grid_results.data"),
            filename_for_parameters: String::from("undefined"),
            max_num_from_file: -1,
            file_column_for_mass: 3,
            file_column_for_y: 4,
            file_column_for_feh: 5,
            file_column_for_alpha: 6,
            file_column_for_f_ov: 7,
            file_column_for_my_param: [8, 9, 10],
            from_file_output_filename: String::from("from_file_results.data"),
            max_yrs_dt_when_cold: 1.0e8,
            max_yrs_dt_when_warm: 1.0e7,
            max_yrs_dt_when_hot: 1.0e6,
            chi2_limit_for_small_timesteps: -1.0,
            max_yrs_dt_chi2_small_limit: 3.0e5,
            chi2_limit_for_smaller_timesteps: -1.0,
            max_yrs_dt_chi2_smaller_limit: 1.0e5,
            chi2_limit_for_smallest_timesteps: -1.0,
            max_yrs_dt_chi2_smallest_limit: 5.0e4,
            min_num_samples_for_avg: 2,
            max_num_samples_for_avg: 10,
            avg_age_sigma_limit: 10.0,
            avg_model_number_sigma_limit: 10.0,
            surf_coef1_name: String::from("a_div_r"),
            surf_coef2_name: String::from("correction_r"),
            save_next_best_at_higher_frequency: false,
            save_next_best_at_lower_frequency: false,
            astero_results_dbl_format: String::from("(1pes26.16)"),
            astero_results_int_format: String::from("(i26)"),
            astero_results_txt_format: String::from("(a26)"),
            write_best_model_data_for_each_sample: true,
            num_digits: 4,
            sample_results_prefix: String::from("sample_"),
            sample_results_postfix: String::from(".data"),
            model_num_digits: 4,
            write_fgong_for_each_model: false,
            fgong_prefix: String::from("fgong_"),
            fgong_postfix: String::from(".data"),
            write_fgong_for_best_model: false,
            best_model_fgong_filename: String::new(),
            write_gyre_for_each_model: false,
            gyre_prefix: String::from("gyre_"),
            gyre_postfix: String::from(".data"),
            max_num_gyre_points: -1,
            write_gyre_for_best_model: false,
            best_model_gyre_filename: String::new(),
            write_profile_for_best_model: false,
            best_model_profile_filename: String::new(),
            save_model_for_best_model: false,
            best_model_save_model_filename: String::new(),
            save_info_for_last_model: false,
            last_model_save_info_filename: String::new(),
            save_controls: false,
            save_controls_filename: String::new(),
            save_mode_model_number: 0,
            save_mode_filename: String::new(),
            el_to_save: 0,
            order_to_save: 0,
            adipls_irotkr: 0,
            adipls_nprtkr: 0,
            adipls_igm1kr: 0,
            adipls_npgmkr: 0,
            read_extra_astero_search_inlist: [false; 5],
            extra_astero_search_inlist_name: std::array::from_fn(|_| String::from("undefined")),
            constraint_names: std::array::from_fn(|_| String::new()),
            indexed_constraints: [SpectroConstraint::default(); 10],
            param_names: std::array::from_fn(|_| String::new()),
            indexed_params: [AsteroSearchParam::fixed(0.0); 10],
            indexed_file_cols: [3, 4, 5, 6, 7, 8, 9, 10, -1, -1],
            has_custom_param: [false; 3],
        }
    }
}

impl AsteroSearchControls {
    /// Returns the initial parameter set (`first_*`).
    #[must_use]
    pub fn first_params(&self) -> AsteroParamSet {
        AsteroParamSet {
            initial_mass: self.mass.first,
            initial_y: self.y.first,
            initial_feh: self.feh_param.first,
            alpha: self.alpha.first,
            f_ov: self.f_ov.first,
            my_param1: self.my_param[0].first,
            my_param2: self.my_param[1].first,
            my_param3: self.my_param[2].first,
        }
    }

    /// Reads candidate parameter rows from `filename_for_parameters` (`from_file`).
    pub fn read_parameters_from_file(
        &self,
        case_dir: &Path,
    ) -> Result<Vec<AsteroParamSet>, StarError> {
        let fname = self.filename_for_parameters.trim();
        if fname.is_empty() || fname.eq_ignore_ascii_case("undefined") {
            return Err(StarError::InvalidInput(String::from(
                "search_type = 'from_file' requires a non-empty filename_for_parameters",
            )));
        }
        let path = case_dir.join(fname);
        let text = fs::read_to_string(&path).map_err(|e| {
            StarError::IoError(format!(
                "failed to open filename_for_parameters '{}': {e}",
                path.display()
            ))
        })?;
        let mut out = Vec::new();
        let mut skipped_header = false;
        for (line_idx, line) in text.lines().enumerate() {
            let trimmed = line.split('!').next().unwrap_or("").trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            let is_col_numbers = tokens.len() >= 3
                && tokens[0].trim() == "1"
                && tokens[1].trim() == "2"
                && tokens[2].trim() == "3";
            if !skipped_header {
                skipped_header = true;
                // Skip non-numeric column header row or `"1 2 3 ..."` index line.
                let first_is_header = tokens
                    .first()
                    .and_then(|t| {
                        t.replace(['d', 'D'], "e").parse::<f64>().ok()
                    })
                    .is_none();
                if first_is_header || is_col_numbers {
                    continue;
                }
            } else if is_col_numbers {
                continue;
            }
            let parse_col = |col_idx: isize,
                             fallback: f64,
                             name: &str,
                             required: bool|
             -> Result<f64, StarError> {
                // `do_get_parameters_from_file`
                // (`astero/private/astero_run_support.f90:277-318`) only reads
                // `filedata(file_column_for_param(i))` when `vary_param(i)` is
                // true; non-varied parameters keep `first_param(i)`
                // (`extras_support.f90:974-988`).
                if !required {
                    return Ok(fallback);
                }
                if col_idx <= 0 {
                    return Err(StarError::InvalidInput(format!(
                        "need to set positive file_column_for_param for varied parameter '{name}'"
                    )));
                }
                let idx = (col_idx as usize) - 1;
                let Some(tok) = tokens.get(idx) else {
                    if required {
                        return Err(StarError::IoError(format!(
                            "missing column {col_idx} ({name}) at {}:{}: row has {} column(s)",
                            path.display(),
                            line_idx + 1,
                            tokens.len()
                        )));
                    }
                    return Ok(fallback);
                };
                let v = tok
                    .replace(['d', 'D'], "e")
                    .parse::<f64>()
                    .map_err(|e| {
                        StarError::IoError(format!(
                            "invalid float '{tok}' in column {col_idx} ({name}) at {}:{}: {e}",
                            path.display(),
                            line_idx + 1
                        ))
                    })?;
                if !v.is_finite() {
                    if required {
                        return Err(StarError::IoError(format!(
                            "non-finite value '{tok}' in column {col_idx} ({name}) at {}:{}",
                            path.display(),
                            line_idx + 1
                        )));
                    }
                    return Ok(fallback);
                }
                Ok(v)
            };
            let mut p = self.first_params();
            p.initial_mass = parse_col(
                self.file_column_for_mass,
                p.initial_mass,
                "initial_mass",
                self.mass.vary,
            )?;
            p.initial_y = parse_col(
                self.file_column_for_y,
                p.initial_y,
                "initial_y",
                self.y.vary,
            )?;
            p.initial_feh = parse_col(
                self.file_column_for_feh,
                p.initial_feh,
                "initial_feh",
                self.feh_param.vary,
            )?;
            p.alpha = parse_col(
                self.file_column_for_alpha,
                p.alpha,
                "alpha",
                self.alpha.vary,
            )?;
            p.f_ov = parse_col(
                self.file_column_for_f_ov,
                p.f_ov,
                "f_ov",
                self.f_ov.vary,
            )?;
            p.my_param1 = parse_col(
                self.file_column_for_my_param[0],
                p.my_param1,
                "my_param1",
                self.my_param[0].vary,
            )?;
            p.my_param2 = parse_col(
                self.file_column_for_my_param[1],
                p.my_param2,
                "my_param2",
                self.my_param[1].vary,
            )?;
            p.my_param3 = parse_col(
                self.file_column_for_my_param[2],
                p.my_param3,
                "my_param3",
                self.my_param[2].vary,
            )?;
            out.push(p);
            if self.max_num_from_file > 0
                && out.len() >= (self.max_num_from_file as usize)
            {
                break;
            }
        }
        if out.is_empty() {
            return Err(StarError::IoError(format!(
                "no candidate parameter rows found in '{}'",
                path.display()
            )));
        }
        Ok(out)
    }

    /// Formats the `&astero_search_controls` namelist (`write_astero_search_controls`,
    /// `astero/public/astero_def.f90:911-940`).
    #[must_use]
    pub fn format_namelist(&self) -> String {
        let fb = |v: bool| if v { ".true." } else { ".false." };
        let fq = |s: &str| format!("'{}'", s.replace('\'', "''"));
        let mut out = String::from("&astero_search_controls\n");
        let _ = writeln!(
            out,
            "   chi2_seismo_fraction = {:.16e}",
            self.weights.chi2_seismo_fraction
        );
        let _ = writeln!(
            out,
            "   chi2_seismo_delta_nu_fraction = {:.16e}",
            self.weights.chi2_seismo_delta_nu_fraction
        );
        let _ = writeln!(
            out,
            "   chi2_seismo_nu_max_fraction = {:.16e}",
            self.weights.chi2_seismo_nu_max_fraction
        );
        let _ = writeln!(
            out,
            "   chi2_seismo_r_010_fraction = {:.16e}",
            self.weights.chi2_seismo_r_010_fraction
        );
        let _ = writeln!(
            out,
            "   chi2_seismo_r_02_fraction = {:.16e}",
            self.weights.chi2_seismo_r_02_fraction
        );
        let _ = writeln!(
            out,
            "   normalize_chi2_seismo_frequencies = {}",
            fb(self.weights.normalize_chi2_seismo_frequencies)
        );
        let _ = writeln!(
            out,
            "   normalize_chi2_seismo_r_010 = {}",
            fb(self.weights.normalize_chi2_seismo_r_010)
        );
        let _ = writeln!(
            out,
            "   normalize_chi2_seismo_r_02 = {}",
            fb(self.weights.normalize_chi2_seismo_r_02)
        );
        let _ = writeln!(
            out,
            "   normalize_chi2_spectro = {}",
            fb(self.weights.normalize_chi2_spectro)
        );
        let _ = writeln!(
            out,
            "   trace_chi2_seismo_delta_nu_info = {}",
            fb(self.trace_chi2_seismo_delta_nu_info)
        );
        let _ = writeln!(
            out,
            "   trace_chi2_seismo_nu_max_info = {}",
            fb(self.trace_chi2_seismo_nu_max_info)
        );
        let _ = writeln!(
            out,
            "   trace_chi2_seismo_ratios_info = {}",
            fb(self.trace_chi2_seismo_ratios_info)
        );
        let _ = writeln!(
            out,
            "   trace_chi2_seismo_frequencies_info = {}",
            fb(self.trace_chi2_seismo_frequencies_info)
        );
        let _ = writeln!(
            out,
            "   trace_chi2_spectro_info = {}",
            fb(self.trace_chi2_spectro_info)
        );
        let _ = writeln!(out, "   trace_limits = {}", fb(self.trace_limits));
        let _ = writeln!(
            out,
            "   trace_time_in_oscillation_code = {}",
            fb(self.trace_time_in_oscillation_code)
        );
        let _ = writeln!(out, "   search_type = {}", fq(&self.search_type));
        let _ = writeln!(
            out,
            "   oscillation_code = {}",
            fq(&self.oscillation_code)
        );
        let _ =
            writeln!(out, "   gyre_input_file = {}", fq(&self.gyre_input_file));
        let _ = writeln!(out, "   gyre_non_ad = {}", fb(self.gyre_non_ad));
        let _ = writeln!(
            out,
            "   correction_scheme = {}",
            fq(&self.correction_scheme)
        );
        let _ = writeln!(out, "   correction_b = {:.16e}", self.correction_b);
        let _ = writeln!(
            out,
            "   correction_factor = {:.16e}",
            self.correction_factor
        );
        let _ =
            writeln!(out, "   surf_coef1_name = {}", fq(&self.surf_coef1_name));
        let _ =
            writeln!(out, "   surf_coef2_name = {}", fq(&self.surf_coef2_name));
        let _ = writeln!(
            out,
            "   save_next_best_at_higher_frequency = {}",
            fb(self.save_next_best_at_higher_frequency)
        );
        let _ = writeln!(
            out,
            "   save_next_best_at_lower_frequency = {}",
            fb(self.save_next_best_at_lower_frequency)
        );
        let _ = writeln!(
            out,
            "   astero_results_directory = {}",
            fq(&self.astero_results_directory)
        );
        let _ = writeln!(
            out,
            "   astero_results_dbl_format = {}",
            fq(&self.astero_results_dbl_format)
        );
        let _ = writeln!(
            out,
            "   astero_results_int_format = {}",
            fq(&self.astero_results_int_format)
        );
        let _ = writeln!(
            out,
            "   astero_results_txt_format = {}",
            fq(&self.astero_results_txt_format)
        );
        let _ = writeln!(
            out,
            "   write_best_model_data_for_each_sample = {}",
            fb(self.write_best_model_data_for_each_sample)
        );
        let _ = writeln!(out, "   num_digits = {}", self.num_digits);
        let _ = writeln!(
            out,
            "   sample_results_prefix = {}",
            fq(&self.sample_results_prefix)
        );
        let _ = writeln!(
            out,
            "   sample_results_postfix = {}",
            fq(&self.sample_results_postfix)
        );
        let _ =
            writeln!(out, "   model_num_digits = {}", self.model_num_digits);
        let _ = writeln!(
            out,
            "   write_fgong_for_each_model = {}",
            fb(self.write_fgong_for_each_model)
        );
        let _ = writeln!(out, "   fgong_prefix = {}", fq(&self.fgong_prefix));
        let _ = writeln!(out, "   fgong_postfix = {}", fq(&self.fgong_postfix));
        let _ = writeln!(
            out,
            "   write_fgong_for_best_model = {}",
            fb(self.write_fgong_for_best_model)
        );
        let _ = writeln!(
            out,
            "   best_model_fgong_filename = {}",
            fq(&self.best_model_fgong_filename)
        );
        let _ = writeln!(
            out,
            "   write_gyre_for_each_model = {}",
            fb(self.write_gyre_for_each_model)
        );
        let _ = writeln!(out, "   gyre_prefix = {}", fq(&self.gyre_prefix));
        let _ = writeln!(out, "   gyre_postfix = {}", fq(&self.gyre_postfix));
        let _ = writeln!(
            out,
            "   max_num_gyre_points = {}",
            self.max_num_gyre_points
        );
        let _ = writeln!(
            out,
            "   write_gyre_for_best_model = {}",
            fb(self.write_gyre_for_best_model)
        );
        let _ = writeln!(
            out,
            "   best_model_gyre_filename = {}",
            fq(&self.best_model_gyre_filename)
        );
        let _ = writeln!(
            out,
            "   write_profile_for_best_model = {}",
            fb(self.write_profile_for_best_model)
        );
        let _ = writeln!(
            out,
            "   best_model_profile_filename = {}",
            fq(&self.best_model_profile_filename)
        );
        let _ = writeln!(
            out,
            "   save_model_for_best_model = {}",
            fb(self.save_model_for_best_model)
        );
        let _ = writeln!(
            out,
            "   best_model_save_model_filename = {}",
            fq(&self.best_model_save_model_filename)
        );
        let _ = writeln!(
            out,
            "   save_info_for_last_model = {}",
            fb(self.save_info_for_last_model)
        );
        let _ = writeln!(
            out,
            "   last_model_save_info_filename = {}",
            fq(&self.last_model_save_info_filename)
        );
        let _ = writeln!(out, "   save_controls = {}", fb(self.save_controls));
        let _ = writeln!(
            out,
            "   save_controls_filename = {}",
            fq(&self.save_controls_filename)
        );
        let _ = writeln!(
            out,
            "   save_mode_model_number = {}",
            self.save_mode_model_number
        );
        let _ = writeln!(
            out,
            "   save_mode_filename = {}",
            fq(&self.save_mode_filename)
        );
        let _ = writeln!(out, "   el_to_save = {}", self.el_to_save);
        let _ = writeln!(out, "   order_to_save = {}", self.order_to_save);
        let _ = writeln!(out, "   adipls_irotkr = {}", self.adipls_irotkr);
        let _ = writeln!(out, "   adipls_nprtkr = {}", self.adipls_nprtkr);
        let _ = writeln!(out, "   adipls_igm1kr = {}", self.adipls_igm1kr);
        let _ = writeln!(out, "   adipls_npgmkr = {}", self.adipls_npgmkr);
        let _ = writeln!(out, "   simplex_alpha = {:.16e}", self.simplex_alpha);
        let _ = writeln!(out, "   simplex_beta = {:.16e}", self.simplex_beta);
        let _ = writeln!(out, "   simplex_gamma = {:.16e}", self.simplex_gamma);
        let _ = writeln!(out, "   simplex_delta = {:.16e}", self.simplex_delta);
        let _ = writeln!(
            out,
            "   restart_simplex_from_file = {}",
            fb(self.restart_simplex_from_file)
        );
        let _ = writeln!(
            out,
            "   restart_scan_grid_from_file = {}",
            fb(self.restart_scan_grid_from_file)
        );
        let _ = writeln!(out, "   y_frac_he3 = {:.16e}", self.y_frac_he3);
        let _ = writeln!(out, "   z_div_x_solar = {:.16e}", self.z_div_x_solar);
        let _ = writeln!(out, "   delta_nu = {:.16e}", self.delta_nu);
        let _ =
            writeln!(out, "   delta_nu_sigma = {:.16e}", self.delta_nu_sigma);
        let _ = writeln!(out, "   nu_max = {:.16e}", self.nu_max);
        let _ = writeln!(out, "   nu_max_sigma = {:.16e}", self.nu_max_sigma);
        for (i, name) in self.param_names.iter().enumerate() {
            if !name.trim().is_empty() {
                let p = self.indexed_params[i];
                let idx = i + 1;
                let _ = writeln!(out, "   param_name({idx}) = {}", fq(name));
                let _ = writeln!(out, "   vary_param({idx}) = {}", fb(p.vary));
                let _ =
                    writeln!(out, "   first_param({idx}) = {:.16e}", p.first);
                let _ = writeln!(out, "   min_param({idx}) = {:.16e}", p.min);
                let _ = writeln!(out, "   max_param({idx}) = {:.16e}", p.max);
                let _ =
                    writeln!(out, "   delta_param({idx}) = {:.16e}", p.delta);
                let _ = writeln!(
                    out,
                    "   file_column_for_param({idx}) = {}",
                    self.indexed_file_cols[i]
                );
            }
        }
        for (i, name) in self.constraint_names.iter().enumerate() {
            let c = self.indexed_constraints[i];
            if !name.trim().is_empty()
                || c.include
                || c.sigma > 0.0
                || c.sigmas_coeff != 0.0
            {
                let idx = i + 1;
                let _ =
                    writeln!(out, "   constraint_name({idx}) = {}", fq(name));
                let _ = writeln!(
                    out,
                    "   constraint_target({idx}) = {:.16e}",
                    c.target
                );
                let _ = writeln!(
                    out,
                    "   constraint_sigma({idx}) = {:.16e}",
                    c.sigma
                );
                let _ = writeln!(
                    out,
                    "   include_constraint_in_chi2_spectro({idx}) = {}",
                    fb(c.include)
                );
                let _ = writeln!(
                    out,
                    "   sigmas_coeff_for_constraint_limit({idx}) = {:.16e}",
                    c.sigmas_coeff
                );
            }
        }
        for l in 0..4 {
            let _ = writeln!(out, "   nl({l}) = {}", self.nl[l]);
            let _ = writeln!(
                out,
                "   iscan_factor({l}) = {:.16e}",
                self.iscan_factor[l]
            );
        }
        out.push_str("/ ! end of astero_search_controls namelist\n");
        out
    }

    /// Writes the `&astero_search_controls` namelist dump when `save_controls`
    /// is enabled (`astero_run_support.f90:133-135`, `astero_def.f90:911-940`).
    pub fn write_controls_if_requested(
        &self,
        out_root: &Path,
    ) -> Result<(), StarError> {
        if !self.save_controls {
            return Ok(());
        }
        let fname = self.save_controls_filename.trim();
        let rel = if fname.is_empty() {
            "astero_search_controls.out"
        } else {
            fname
        };
        let path = out_root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                StarError::IoError(format!(
                    "failed creating directory for save_controls '{}': {e}",
                    parent.display()
                ))
            })?;
        }
        fs::write(&path, self.format_namelist()).map_err(|e| {
            StarError::IoError(format!(
                "failed writing save_controls_filename '{}': {e}",
                path.display()
            ))
        })
    }
}
